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

def collate_fn(batch):
    images = [item['image'] for item in batch]
    depth_maps = [item['depth_map'] for item in batch]
    scales = [item['scale'] for item in batch]
    return [images, scales, depth_maps]
            