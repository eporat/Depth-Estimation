from load_data import load_pfm
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import DepthMapDataset, DepthMapDataLoader

dataset = DepthMapDataset("C:\\Users\\ethan\\Downloads\\geoPose3K_final_publish\\geoPose3K_final_publish")
dataloader = DepthMapDataLoader(dataset=dataset)

for i_batch, sample_batched in enumerate(dataloader):
    images, depth_maps = sample_batched
    
    for image, depth_map in zip(images, depth_maps):
        plt.figure(1)
        plt.imshow(image)
        plt.figure(2)
        plt.imshow(depth_map)
        plt.show()
        