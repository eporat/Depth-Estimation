import torch
import torch.nn as nn
from unet import ResNetUNet
from dataset import DepthMapDataset, DepthMapDataLoader
import matplotlib.pyplot as plt

num_epochs = 1

dataset = DepthMapDataset("C:\\depth estimation\\mini-dataset")
dataloader = DepthMapDataLoader(dataset=dataset, batch_size=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNetUNet()
model = model.to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
optimizer.zero_grad()

criterion = torch.nn.MSELoss()

model.train()

for epoch in range(num_epochs):  
  for i_batch, sample_batched in enumerate(dataloader):
    images, depth_maps, shapes = sample_batched['images'], sample_batched['depth_maps'], sample_batched['shapes']
    
    output = model(images)
        
    for pred, real, shape in zip(output.float(), depth_maps, shapes): 
        loss = criterion(pred[0, :shape[0], :shape[1]], real[:shape[0], :shape[1]])
        loss.backward()
        optimizer.step()