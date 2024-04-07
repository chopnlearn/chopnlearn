import inspect
import json
import os
from glob import glob
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


STATE_TO_TEXT = {
    "b": "baton style",
    "lc": "large pieces",
    "rs": "round slices",
    "w": "raw style",
    "j": "julienne style",
    "p": "peeled style",
    "hrs": "half round slices",
    "sc": "small pieces"
}


PROMPT = "An image of {} in {}"
PROMPT_object = "An image of {}"
PROMPT_state = "An image of an object in {}"


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def collate_fn_clip(item):
    keys = item[0].keys()
    batch = {}
    for key in keys:
        batch[key] = torch.stack([example[key] for example in item]).to(memory_format=torch.contiguous_format)
    return batch


def build_object_state_label_dict(image_folder, exclude_states=[]):
    """
    Build a dictionary of object labels state labels.
    """
    object_names = set()
    state_names = set()

    for img_path in glob(f'{image_folder}/*.png'):
        object_name, state_name = Path(img_path).stem.split('_')[:2]
        object_names.add(object_name)
        if state_name not in exclude_states:
            state_names.add(state_name)

    object_id2label = {i: label for i, label in enumerate(sorted(list(object_names)))}
    state_id2label = {i: label for i, label in enumerate(sorted(list(state_names)))}

    object_label2id = {label: i for i, label in object_id2label.items()}
    state_label2id = {label: i for i, label in state_id2label.items()}

    return object_label2id, state_label2id, object_id2label, state_id2label



class CameraSpecificTransform(torch.nn.Module):
    def __init__(self, 
        size=512,
        mean = [0.5, 0.5, 0.5], # [0.48145466, 0.4578275, 0.40821073] for CLIP visual backbones
        std = [0.5, 0.5, 0.5], # [0.26862954, 0.26130258, 0.27577711] for CLIP visual backbones
        augmentation=False
    ):
        super().__init__()
        self.size = size

        self.crop_func = {
            'cam1': lambda img: F.crop(img, top=0, left=950, height=1800, width=1800),
            'cam2': lambda img: F.crop(img, top=0, left=930, height=2000, width=2000),
            'cam3': lambda img: F.crop(img, top=250, left=990, height=1800, width=1800),
            'cam4': lambda img: F.crop(img, top=280, left=1170, height=1800, width=1800)
        }


        self.resize = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.aug = transforms.AutoAugment() if augmentation else None

        self.addition_transform = transforms.Compose([
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),   
        ])


    def forward(self, image, camera):
        if image.size != (self.size, self.size):
            image = self.crop_func[camera](image)
            image = self.resize(image)
        if self.aug:
            image = self.aug(image)
        image = self.addition_transform(image)
        return image



class ObjectStateDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        data_root,
        tokenizer,
        data_folder='images',
        split='train',
        center_crop=True,
        size=512,
        camera_specific_transform=True, # center_crop will be ignored if image_transforms is True
        exclude_states=[],
        state_to_text=STATE_TO_TEXT,
        object_to_text=None,
        return_image=True,
        cameras=['cam1', 'cam2', 'cam3', 'cam4'],
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.camera_specific_transform = camera_specific_transform
        self.cameras = cameras

        self.data_root = Path(data_root)
        img_path = self.data_root / data_folder

        self.return_image = return_image

        def load_json(path):
            if not os.path.exists(path):
                raise ValueError(f'{path} doesn\'t exists.')
            with open(path) as f:
                return json.load(f)
            
        data_list = []
        for path in split.split('+'):
            data_list.append(load_json(path))

        states = {}
        for data in data_list:
            for state, objects in data.items():
                if state not in states:
                    states[state] = []
                states[state] += objects

        self.states = states

        if exclude_states:
            assert (isinstance(exclude_states, list) or isinstance(exclude_states, tuple)) and isinstance(exclude_states[0], str)
            for state in exclude_states:
                if state in self.states:
                    self.states.pop(state)

        self.state_to_text = state_to_text
        self.object_to_text = object_to_text

        self.image_paths = []
        self.object_labels = []
        self.state_labels = []

        def is_from_valid_camera(path):
            return any([f'_{camera}_' in str(path) for camera in self.cameras])


        for state, objects in self.states.items():
            for object in objects:
                images = [x for x in img_path.glob(f'{object}_{state}*.png') if is_from_valid_camera(x.name)]
                self.image_paths += images
                self.object_labels += [object] * len(images)
                self.state_labels += [state] * len(images)

        assert len(self.image_paths) > 0, f'No images found in {os.path.abspath(img_path)}'

        if self.object_to_text is None:
            self.object_to_text = {object: object for object in self.unique_objects}

        if camera_specific_transform:
            self.image_transforms = CameraSpecificTransform(size)
        else:
            self.image_transforms = transforms.Compose(
                [
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        example = {}
        index = index % len(self)

        if self.return_image:
            instance_image = Image.open(self.image_paths[index])
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            if self.camera_specific_transform:
                camera = self.image_paths[index].stem.split('_')[-2]
                example["instance_images"] = self.image_transforms(instance_image, camera)
            else:
                example["instance_images"] = self.image_transforms(instance_image)
        
        example["instance_prompt_ids"] = self.tokenizer(
            PROMPT.format(self.object_to_text[self.object_labels[index]], self.state_to_text[self.state_labels[index]]),
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze()

        return example

    @property
    def unique_object_state_pairs(self) -> set[tuple[str, str]]:
        return {(obj, state) for state, objs in self.states.items() for obj in objs}

    @property
    def unique_states(self) -> set[str]:
        return set(self.state_labels)

    @property
    def unique_objects(self) -> set[str]:
        # return set(sum(self.states.values(), []))
        return set(self.object_labels)


class BaseCLIPObjectStateClassifierDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        data_folder='images',
        split='train_split_classifier_from_diffusion.json',
        transform=None,
        return_text_features=False, # if True, return text features instead of token ids
        clip_model=None,
        exclude_states=[], # exclude states from the dataset
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.data_folder = data_folder
        self.transform = None
        self.camera_specific_transform = False

        image_folder = self.data_root / data_folder

        self.object_label2id, self.state_label2id, self.object_id2label, self.state_id2label = \
            build_object_state_label_dict(image_folder, exclude_states=exclude_states)
        
        self.tokenizer = tokenizer

        self.split = split
        self.transform = transform
        self.exclude_states = exclude_states
        self.load_data()
        self.return_text_features = return_text_features
        self.clip_model = clip_model
        self.build_text_dict(build_feature_dict=self.return_text_features)


    @property
    def num_object_classes(self):
        raise NotImplementedError
    
    @property
    def num_state_classes(self):
        raise NotImplementedError
    
    @property
    def unique_objects(self):
        raise NotImplementedError

    @property
    def unique_states(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
    def load_data(self):
        raise NotImplementedError
       
    def get_data(self, idx):
        raise NotImplementedError  

    def __getitem__(self, idx):
        item = {}
        img, object_label, state_label, camera = self.get_data(idx)

        if self.transform:
            if self.camera_specific_transform:
                img = self.transform(img, camera)
            else:
                img = self.transform(img)

        item['pixel_values'] = img

        if self.return_text_features:
            item['object_text_features'] = self.text_feature_dict[object_label]
            item['state_text_features'] = self.text_feature_dict[state_label]
        else:
            item['input_ids_object'] = self.tokenizer(
                PROMPT_object.format(object_label),
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze()
            item['input_ids_state'] = self.tokenizer(
                PROMPT_state.format(STATE_TO_TEXT[state_label]),
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze()

        item['object_ids'] = torch.tensor(self.object_label2id[object_label]).long()
        item['state_ids'] = torch.tensor(self.state_label2id[state_label]).long()
        return item
    

    @torch.no_grad()
    def build_text_dict(self, build_feature_dict=False):
        self.text_ids_dict = {}
        if build_feature_dict:
            self.text_feature_dict = {}
        for object_label in self.unique_objects:
            input_ids = self.tokenizer(
                PROMPT_object.format(object_label),
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze()
            self.text_ids_dict[object_label] = input_ids

            if build_feature_dict:
                text_features = self.clip_model.get_text_features(input_ids.unsqueeze(0).to(self.clip_model.device))
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                self.text_feature_dict[object_label] = text_features.detach().squeeze()

        for state_label in self.unique_states:
            input_ids = self.tokenizer(
                PROMPT_state.format(STATE_TO_TEXT[state_label]),
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze()
            self.text_ids_dict[state_label] = input_ids

            if build_feature_dict:
                text_features = self.clip_model.get_text_features(input_ids.unsqueeze(0).to(self.clip_model.device))
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                self.text_feature_dict[state_label] = text_features.detach().squeeze()


class CLIPObjectStateImageFolderDataset(BaseCLIPObjectStateClassifierDataset):
    def __init__(
        self,
        image_folder,
        data_root,
        tokenizer,
        data_folder='images',
        split='test_split.json',
        transform=None,
        return_text_features=False, # if True, return text features instead of token ids
        clip_model=None,
        exclude_states=[]
    ) -> None:
        self.image_folder = image_folder
        super().__init__(
            data_root=data_root,
            data_folder=data_folder,
            tokenizer=tokenizer,
            split=split,
            transform=transform,
            return_text_features=return_text_features,
            clip_model=clip_model,
            exclude_states=exclude_states
        )


    def load_data(self):
        if self.transform is not None and hasattr(self.transform, 'forward'):
            signature = inspect.signature(self.transform.forward)
            assert 'camera' not in signature.parameters


        def load_json(path):
            if not os.path.exists(path):
                raise ValueError(f'{path} doesn\'t exists.')

            print(f'Using split {path}')

            with open(path) as f:
                return json.load(f)
            

        paths = self.split.split('+')

        data_list = []
        for path in paths:
            data_list.append(load_json(path))

        if isinstance(data_list[0], list):
            self.img_names = [n for n in sum(data_list, []) if all([f'_{state}_' not in n for state in self.exclude_states])]
        else:
            assert isinstance(data_list[0], dict)
            states = {}
            for data in data_list:
                for state, objects in data.items():
                    if state not in states:
                        states[state] = []
                    states[state] += objects
            for state in self.exclude_states:
                if state in states:
                    states.pop(state)

            self.states = states
            
            # load images
            img_path = self.image_folder
            self.img_names = []

            unique_pairs = self.unique_object_state_pairs
            for img_name in os.listdir(img_path):
                assert img_name.endswith('.png') and img_name.startswith('sample_'), f'Invalid image name {img_name} in {img_path}'
                object_state = img_name.split('_')[1:3]
                if tuple(object_state) in unique_pairs:
                    self.img_names.append(img_name)

        self.img_paths = sorted(
            [Path(self.image_folder) / img_name for img_name in self.img_names])

    @property
    def num_object_classes(self):
        return len(self.object_id2label)
    
    @property
    def num_state_classes(self):
        return len(self.state_id2label)
    
    @property
    def unique_objects(self):
        return set(self.object_label2id.keys())
    
    @property
    def unique_states(self):
        return set(self.state_label2id.keys())
    
    @property
    def unique_object_state_pairs(self) -> set[tuple[str, str]]:
        assert hasattr(self, 'states') and self.states is not None
        return {(obj, state) for state, objs in self.states.items() for obj in objs}

    def __len__(self):
        return len(self.img_paths)

    def get_data(self, idx):
        img_path = self.img_paths[idx]
        object_label, state_label = img_path.stem.split('_')[1:3]
        img = Image.open(img_path).convert('RGB')

        if self.camera_specific_transform:
            camera = img_path.stem.split('_')[-2]
        else:
            camera = None

        return img, object_label, state_label, camera

