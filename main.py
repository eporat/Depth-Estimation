from load_data import load_pfm
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import DepthMapDataset, DepthMapDataLoader

dataset = DepthMapDataset("C:\\depth estimation\\mini-dataset")
dataloader = DepthMapDataLoader(dataset=dataset)

for i_batch, sample_batched in enumerate(dataloader):
    images, scales, depth_maps = sample_batched
    print(scales)

    for image, depth_map in zip(images, depth_maps):
        fig, (ax1, ax2) = plt.subplots(ncols=2)

        # plot just the positive data and save the
        # color "mappable" object returned by ax1.imshow
        image_plot = ax1.imshow(image)
        depth_map_plot = ax2.imshow(depth_map)
        fig.colorbar(depth_map_plot, ax=ax2)
        plt.show()

        
