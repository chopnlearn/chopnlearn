import argparse
import json
import os
import pathlib
import shutil
import uuid

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from filelock import FileLock
from PIL import Image
from pytorch_fid.fid_score import (calculate_frechet_distance,
                                   compute_statistics_of_path)
from pytorch_fid.inception import InceptionV3
from tqdm import tqdm


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class FIDCalculator:
    def __init__(self, 
        dims=2048, 
        device='cuda', 
        patchwise=False, 
        center_crop_size=384, 
        patch_num=32, 
        patch_size=192,
        cache=False,
        cache_dir='evaluation/fid_cache',
        tmp_dir='/dev/shm/chopnlearn'
    ):
        self.device = device
        self.dims = dims
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device)
        self.cache = cache

        self.patchwise = patchwise
        if patchwise:
            assert center_crop_size is not None
            assert patch_num is not None
            assert patch_size is not None
            self.patch_transform = {}
            self.patch_transform['center_crop'] = transforms.CenterCrop(center_crop_size)
            self.patch_transform['random_crop'] = transforms.RandomCrop(patch_size)

        self.center_crop_size = center_crop_size
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.cache_dir = cache_dir
        if self.cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.tmp_dir = tmp_dir
        os.makedirs(self.tmp_dir, exist_ok=True)

    
    def gen_cache_path(self, path, split):
        if os.path.isdir(path):
            os.makedirs(self.cache_dir, exist_ok=True)
            split = pathlib.Path(split).stem
            cache_path = os.path.join(self.cache_dir, f'fid_stats_{split}_dim{self.dims}_patch{self.patchwise}_ccrop{self.center_crop_size}_pnum{self.patch_num}_psize{self.patch_size}.npz')
        else:
            assert path.endswith('.npz'), f"Path {path} is not a directory or a .npz file"
            cache_path = path # already a cache path
        return cache_path
    
    def make_tmp_imgs(self, path):
        tmp_img_dir = os.path.join(self.tmp_dir, str(uuid.uuid4()))
        os.makedirs(tmp_img_dir, exist_ok=True)
        assert os.path.isdir(tmp_img_dir)
        img_names = [n for n in os.listdir(path) if os.path.splitext(n)[1][1:] in IMAGE_EXTENSIONS]

        print(f'Making tmp imgs at {tmp_img_dir}. Source imgs at {path}')
        for img_name in tqdm(img_names):
            img_stem, img_ext = os.path.splitext(img_name)
            img_path = os.path.join(path, img_name)
            img = Image.open(img_path)
            img = self.patch_transform['center_crop'](img)
            for i in range(self.patch_num):
                patch = self.patch_transform['random_crop'](img)
                patch.save(os.path.join(tmp_img_dir, f'{img_stem}_patch{i}{img_ext}'))

        print('Tmp imgs made')
        return tmp_img_dir



    def compute_fid(self, path_gt, path_gen, split, batch_size=64, device=None, num_workers=0):
        if device is None:
            device = self.device

        cache_path_gt = self.gen_cache_path(path_gt, split)
        if os.path.exists(cache_path_gt) and self.cache:
            print(f"Found cached statistics for {path_gt}")
            print(f"Loading from {cache_path_gt}")

            m1, s1 = compute_statistics_of_path(
                path=cache_path_gt, 
                model=self.model, 
                batch_size=batch_size,
                dims=self.dims,
                device=device, 
                num_workers=num_workers,
            )  
        else:
            print(f"Did not find cached statistics for {path_gt}")
            print(f"Computing statistics for {path_gt}")
            if self.patchwise:
                path_gt = self.make_tmp_imgs(path_gt)
            else:
                path_gt = path_gt
            m1, s1 = compute_statistics_of_path(
                path=path_gt,
                model=self.model,
                batch_size=batch_size,
                dims=self.dims,
                device=device,
                num_workers=num_workers,
            )
            
            if self.cache:
                print(f"Saving statistics to {cache_path_gt}")
                np.savez(cache_path_gt, mu=m1, sigma=s1)

        if self.patchwise:
            path_gen = self.make_tmp_imgs(path_gen)
        else:
            path_gen = path_gen
        m2, s2 = compute_statistics_of_path(
            path=path_gen,
            model=self.model,
            batch_size=batch_size,
            dims=self.dims,
            device=device,
            num_workers=num_workers,
        )

        assert m1.shape == m2.shape, f"m1.shape {m1.shape} != m2.shape {m2.shape}"
        assert s1.shape == s2.shape, f"s1.shape {s1.shape} != s2.shape {s2.shape}"

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value



def get_unique_object_state_pairs(split):
    def load_json(name):
        path = name
        if not os.path.exists(path):
            raise ValueError(f'{path} doesn\'t exists.')
        with open(path) as f:
            return json.load(f)
        
    data_list = []
    for name in split.split('+'):
        data_list.append(load_json(name))

    states = {}
    for data in data_list:
        for state, objects in data.items():
            if state not in states:
                states[state] = []
            states[state] += objects

    return {(obj, state) for state, objs in states.items() for obj in objs}

        



def calculate_fid_scores(
    path_gt, 
    path_gen_list,
    split=None,
    batch_size=64,
    device='cuda',
    num_workers=0,
    dims=2048,
    patchwise=False,
    center_crop_size=384,
    patch_num=32,
    patch_size=192,
    cache_dir='evaluation/fid_cache',
    tmp_dir='/dev/shm/chopnlearn',
    fid_report_path='evaluation/fid_report.csv'
):

    fid_calculator = FIDCalculator(
        dims=dims, 
        device=device, 
        patchwise=patchwise, 
        center_crop_size=center_crop_size, 
        patch_num=patch_num, 
        patch_size=patch_size,
        cache=True,
        cache_dir=cache_dir,
        tmp_dir=tmp_dir
    )

    valid_pairs = None
    if not (os.path.isfile(path_gt) and path_gt.endswith('.npz')) and split is not None:
        valid_pairs = get_unique_object_state_pairs(split)
        selected_names = []
        for name in os.listdir(path_gt):
            obj, state = name.split('_')[:2]
            if (obj, state) in valid_pairs:
                selected_names.append(name)
        new_tmp_dir = os.path.join(fid_calculator.tmp_dir, f'{uuid.uuid4()}')


        if os.path.exists(new_tmp_dir):
            if os.path.isfile(new_tmp_dir):
                os.unlink(new_tmp_dir)
            elif os.path.isdir(new_tmp_dir):
                shutil.rmtree(new_tmp_dir)
            else:
                raise ValueError(f'Unknown path type: {new_tmp_dir}')
        os.makedirs(new_tmp_dir, exist_ok=False)

        # copy selected images to new tmp dir
        for name in selected_names:
            shutil.copy(os.path.join(path_gt, name), new_tmp_dir)

        path_gt = new_tmp_dir


    # make a blank pandas dataframe with columns: split, path_gen, patchwise, center_crop_size, patch_num, patch_size, fid_score
    df = pd.DataFrame(columns=['split', 'path_gen', 'patchwise', 'center_crop_size', 'patch_num', 'patch_size', 'fid_score'])
    for i, path_gen_orig in enumerate(path_gen_list):
        print(f'{i}/{len(path_gen_list)}: Computing FID for {path_gen_orig}')

        if split is not None:
            assert valid_pairs is not None
            selected_names_gen = []
            for name in os.listdir(path_gen_orig):
                assert name.startswith('sample_')
                obj, state = name.split('_')[1:3]
                if (obj, state) in valid_pairs:
                    selected_names_gen.append(name)

            new_tmp_dir_gen = os.path.join(fid_calculator.tmp_dir, f'{uuid.uuid4()}')

            if os.path.exists(new_tmp_dir_gen):
                if os.path.isfile(new_tmp_dir_gen):
                    os.unlink(new_tmp_dir_gen)
                elif os.path.isdir(new_tmp_dir_gen):
                    shutil.rmtree(new_tmp_dir_gen)
                else:
                    raise ValueError(f'Unknown path type: {new_tmp_dir_gen}')

            os.makedirs(new_tmp_dir_gen, exist_ok=False)
        
            for name in selected_names_gen:
                shutil.copy(os.path.join(path_gen_orig, name), new_tmp_dir_gen)

            path_gen = new_tmp_dir_gen

        else:
            path_gen = path_gen_orig

        fid_score = fid_calculator.compute_fid(
            path_gt=path_gt, 
            path_gen=path_gen, 
            split=split,
            batch_size=batch_size, 
            device=device, 
            num_workers=num_workers
        )

        df = pd.concat([df, pd.DataFrame({
            'split': [split],
            'path_gen': [os.path.basename(path_gen_orig)],
            'patchwise': [patchwise],
            'center_crop_size': [center_crop_size],
            'patch_num': [patch_num],
            'patch_size': [patch_size],
            'fid_score': [fid_score]
        })], ignore_index=True)

    if os.path.exists(fid_report_path):
        with FileLock(fid_report_path + '.lock'):
            df_old = pd.read_csv(fid_report_path)
            df = pd.concat([df_old, df], ignore_index=True)    
            df.to_csv(fid_report_path, index=False)
    else:
        df.to_csv(fid_report_path, index=False)
   
    # finally, clean up the tmp dirs
    shutil.rmtree(tmp_dir)
    return df


def get_args(input_args=None):
    """
    Parse command line arguments.

    Args:
        input_args (list, optional): List of input arguments. Defaults to None.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_gen', nargs='+', type=str, required=True,
                        help='Paths to the generated images.')
    parser.add_argument('--path_gt', type=str, default='../image_dataset/images',
                        help='Directory containing the image dataset, GT images.')
    parser.add_argument('--split', type=str, default='test_split.json',
                        help='Name of the JSON file containing the split information.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of images to process in each batch.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker processes for data loading.')
    parser.add_argument('--dims', type=int, default=2048,
                        help='Number of dimensions for feature extraction.')
    parser.add_argument('--nopatchwise', action='store_true',
                        help='Disable patch-wise feature extraction.')
    parser.add_argument('--center_crop_size', type=int, default=384,
                        help='Size of the center crop for each image.')
    parser.add_argument('--patch_num', type=int, default=32,
                        help='Number of patches to extract from each image.')
    parser.add_argument('--patch_size', type=int, default=224,
                        help='Size of each patch.')
    parser.add_argument('--cache_dir', type=str, default='evaluation/fid_cache',
                        help='Directory to store the FID cache files.')
    parser.add_argument('--fid_report_path', type=str, default='evaluation/fid_report.csv',
                        help='Path to the FID report CSV file.')
    parser.add_argument('--tmp_dir', type=str, default='/dev/shm/chopnlearn',
                        help='Temporary directory for storing image patches.')

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    if os.path.exists(args.fid_report_path):
        df_old = pd.read_csv(args.fid_report_path)
        args.path_gen = [path_gen for path_gen in args.path_gen if os.path.basename(path_gen) not in df_old['path_gen'].values]

    if args.nopatchwise:
        args.center_crop_size = None
        args.patch_num = None
        args.patch_size = None

    if args.path_gen:
        df = calculate_fid_scores(
            path_gt=args.path_gt,
            path_gen_list=args.path_gen,
            split=args.split,
            batch_size=args.batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            num_workers=args.num_workers,
            dims=args.dims,
            patchwise=not args.nopatchwise,
            center_crop_size=args.center_crop_size,
            patch_num=args.patch_num,
            patch_size=args.patch_size,
            cache_dir=args.cache_dir,
            fid_report_path=args.fid_report_path,
            tmp_dir=args.tmp_dir
        )
        print(df)
    else:
        print('No new path_gen to evaluate, skipping')

    