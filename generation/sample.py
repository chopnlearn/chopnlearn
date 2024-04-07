import argparse
import math
import os
import time

import torch
from dataset import PROMPT, STATE_TO_TEXT, ObjectStateDataset
from diffusers import DiffusionPipeline


def sample_images(
        ckpt_path, 
        data_dir,
        data_folder,
        split, 
        out_dir='eval/samples/debug', 
        num_images_per_prompt=10,
        batch_size=2,
        starting_idx=0,
        textual_inversion=False,
        train_split='train_split.json'
    ):
    print(f'Sampling {num_images_per_prompt} images per prompt from {ckpt_path}')
    print(f'Using split {split}')
    print(f'Saving to {out_dir}')
    pipeline = DiffusionPipeline.from_pretrained(ckpt_path, safety_checker=None, requires_safety_checker=False).to('cuda')

    os.makedirs(out_dir, exist_ok=True)

    train_dataset = ObjectStateDataset(
        data_root=data_dir,
        tokenizer=pipeline.tokenizer,
        data_folder=data_folder,
        size=512,
        split=train_split,
        exclude_states=['p'],
    )   

    if textual_inversion:
        new_state_to_text = {state: f'*_state_{state}' for state in train_dataset.unique_states}
        new_object_to_text = {obj: f'*_obj_{obj}' for obj in train_dataset.unique_objects}
    else:
        new_state_to_text = {state: STATE_TO_TEXT[state] for state in train_dataset.unique_states}
        new_object_to_text = {obj: obj for obj in train_dataset.unique_objects}

    del train_dataset

    sample_dataset = ObjectStateDataset(
        data_root=data_dir,
        tokenizer=pipeline.tokenizer,
        data_folder=data_folder,
        size=512,
        split=split,
        exclude_states=['p'],
    )   

    num_batches = math.ceil(num_images_per_prompt / batch_size)

    print(f'Will sample {num_images_per_prompt} images per prompt, {num_images_per_prompt * len(sample_dataset.unique_object_state_pairs)} total')
    print('Sampling images...')

    start_time = time.time()

    with torch.no_grad():
        for i_pair, (obj, st) in enumerate(sorted(list(sample_dataset.unique_object_state_pairs))):
            pair_start_time = time.time()
            prompt = PROMPT.format(new_object_to_text[obj], new_state_to_text[st])
            images = []
            for i in range(num_batches):
                print(f'Sampling {obj}_{st} {i_pair + 1}/{len(sample_dataset.unique_object_state_pairs)} batch {i+1}/{num_batches}')
                image_batch = pipeline(
                    prompt=prompt, height=512,
                    width=512, num_inference_steps=50,
                    num_images_per_prompt=batch_size
                ).images
                images += image_batch

            for i, img in enumerate(images):
                save_path = os.path.join(out_dir, f'sample_{obj}_{st}_{starting_idx+i}.png')
                img.save(save_path)

            print(f'Finished sampling {obj}_{st} {i_pair + 1}/{len(sample_dataset.unique_object_state_pairs)} in {time.time() - pair_start_time:.2f} seconds')
            print(f'Total time elapsed: {time.time() - start_time:.2f} seconds')


def get_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--data', type=str, default='../image_dataset/images')
    parser.add_argument('--split_json', '--split', type=str, default='../image_dataset/test_split.json')
    parser.add_argument(
        '--train_split_json',
        type=str,
        default=None,
        help='Only used for creating the dataset. If not provided, will use train_split.json in the same directory as split_json',
    )
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--num_images_per_prompt', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--starting_idx', type=int, default=0)
    parser.add_argument(
        '--textual_inversion',
        '--ti',
        action='store_true',
        help='Use textual inversion instead of object state inversion',
    )
    args = parser.parse_args(input_args) if input_args else parser.parse_args()

    args.data_dir = os.path.dirname(args.data)
    args.data_folder = os.path.basename(args.data)

    if args.out_dir is None:
        ckpt_name = args.ckpt_path.split('/')[-2]
        args.out_dir = os.path.join('./samples', f'{ckpt_name}_{args.split}')

    if args.train_split_json is None:
        args.train_split_json = os.path.join(
            os.path.dirname(args.split_json), 'train_split.json'
        )

    return args


if __name__ == '__main__':
    args = get_args()

    sample_images(
        ckpt_path=args.ckpt_path,
        data_dir=args.data_dir,
        data_folder=args.data_folder,
        split=args.split_json,
        out_dir=args.out_dir,
        num_images_per_prompt=args.num_images_per_prompt,
        batch_size=args.batch_size,
        starting_idx=args.starting_idx,
        textual_inversion=args.textual_inversion,
        train_split=args.train_split_json
    )
