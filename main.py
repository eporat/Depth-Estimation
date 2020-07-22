from load_data import load_pfm
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from imageio import imread
import tarfile
from load_data import load_pfm
import gzip
import shutil

dataset = "C:\\Users\\ethan\\Downloads\\geoPose3K_final_publish\\geoPose3K_final_publish"

def gunzip_shutil(source_filepath, dest_filepath, block_size=65536):
    with gzip.open(source_filepath, 'rb') as s_file, \
            open(dest_filepath, 'wb') as d_file:
        shutil.copyfileobj(s_file, d_file, block_size)


for folder in os.listdir(dataset):
    image_path = f'{dataset}/{folder}/cyl/photo_crop.jpg'
    depth_map_path = f'{dataset}/{folder}/cyl/distance_crop.pfm.gz'
    gunzip_shutil(depth_map_path, 'temp/file.pfm')
    depth_map, scale = load_pfm(open('temp/file.pfm', 'rb'))
    depth_map = depth_map[::-1]
    image = imread(image_path)
    plt.figure(1)
    plt.imshow(depth_map)
    plt.figure(2)
    plt.imshow(image)
    plt.show()