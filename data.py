import torch
import torchvision
import kornia.augmentation.functional as F
from torch.utils.data import Dataset
import numpy as np
from copy import deepcopy


class Transforms(object):
    @staticmethod
    def color_transform(sample, config={}):
        image = sample['image']
        color_low, color_high = config.get('color', {}).get(sample['digit'], (0, 128))
        color = np.random.uniform(color_low, color_high)
        image[image < color] = color

        sample['image'] = image
        sample['color'] = color / 255.
        
    @staticmethod
    def invert_transform(sample, config={}):
        image = sample['image']
        invert = np.random.uniform() > 0.5
        if invert:
            image = 255 - image

        sample['image'] = image
        sample['invert'] = invert

    @staticmethod
    def rotate_transform(sample, config={}):
        image = sample['image'].unsqueeze(0)
        assert image.dim() == 4
        rotation = np.random.uniform(-180, 180)
        params = {
            'degrees': torch.tensor(rotation),
            'interpolation': torch.tensor(1),
            'align_corners': torch.tensor(True)
        }
        image = F.apply_rotation(image, params)[0]

        assert image.dim() == 3

        sample['image'] = image
        sample['rotation'] = rotation
        
    @staticmethod
    def from_string(t: str):
        if t == 'color':
            return Transforms.color_transform
        elif t == 'rotate':
            return Transforms.rotate_transform
        elif t == 'invert':
            return Transforms.invert_transform


class AugMNISTDataset(Dataset):
    def __init__(self, transforms=['rotate', 'color', 'invert'], config={}):
        self._dataset = torchvision.datasets.mnist.MNIST('/iris/u/em7/code/disentanglement/.data')
        self._transforms = [Transforms.from_string(t) for t in transforms]
        self._config = config
        
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        image, label = self._dataset[idx]

        sample = {
            'image': torch.tensor(np.array(image)).unsqueeze(0).float(),
            'digit': label,
            'color': 0,
            'invert': False,
            'rotation': 0.0
        }
        
        for t in self._transforms:
            t(sample, config=self._config)

        sample['image'] /= 255.0
        sample['label'] = label

        assert sample['image'].dim() == 3, f"Bad image shape {sample['image'].shape}"

        return sample
