from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from load_data import load_pfm
from skimage import io
from utils import gunzip_shutil
import math

class DepthMapDataset(Dataset):
    """Depth Map dataset."""

    def __init__(self, dataset, transform=None):
        """
        Args:
            dataset (file): dataset file
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        """
        self.dataset = dataset
        self.transform = transform
        self.folders = os.listdir(dataset)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        folder = self.folders[idx]
        root = f'{self.dataset}/{folder}/cyl'
        image_path = os.path.join(root, 'photo_crop.jpg')
        depth_map_path = f'{root}/distance_crop.pfm.gz'
        
        if not os.path.exists('temp'):
            os.makedirs('temp')
            
        gunzip_shutil(depth_map_path, 'temp/file.pfm')
        depth_map, scale = load_pfm(open('temp/file.pfm', 'rb'))
        depth_map = depth_map[::-1]
        image = io.imread(image_path)

        sample = {'image': image, 'depth_map': depth_map, 'scale': scale}

        if self.transform:
            sample = self.transform(sample)

        return sample

class DepthMapDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=4, shuffle=True):
        DataLoader.__init__(self, dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def add_padding(image, new_w, new_h):
    if len(image.shape) == 2:
        new_image = np.zeros((new_w, new_h), dtype=image.dtype)
        new_image[:image.shape[0], :image.shape[1]] = image
        return new_image
        
    if len(image.shape) == 3:
        new_image = np.zeros((new_w, new_h, 3), dtype=image.dtype)
        new_image[:image.shape[0], :image.shape[1], :] = image
        return new_image
    
size = 256

def collate_fn(batch):
    """Collate Function

    Args:
        batch

    Returns:
        Batch (dictionary) containing padded images with depth maps and shapes of the images
    """
    images = [item['image'] for item in batch]
    shapes = [image.shape[:2] for image in images]
    max_w = math.ceil(max(image.shape[0] for image in images) / size) * size
    max_h = math.ceil(max(image.shape[1] for image in images) / size) * size
    
    new_images = [torch.from_numpy(np.rollaxis(add_padding(image, max_w, max_h), 2)) for image in images]
    depth_maps = [torch.from_numpy(add_padding(item['depth_map'], max_w, max_h)) for item in batch]
    
    batch = {'images': torch.stack(new_images).float(), 'depth_maps': torch.stack(depth_maps), 'shapes': shapes}
    del images[:]
    del new_images[:]
    del depth_maps[:]

    return batch
            