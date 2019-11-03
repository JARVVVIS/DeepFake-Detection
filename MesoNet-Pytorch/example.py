from classifiers import *
from pipeline import *

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import matplotlib.pyplot as plt


## set up the device
device = torch.device('cuda')

# 1 - Load the model and its pretrained weights
model = MesoInception4()
model.load_state_dict(torch.load('./weights/MesoInceptionNaive.pt'))
model.to(device)

video_dir = ['/media/ruchit/Data/deepfakes/data/manipulated_sequences/DeepFakeDetection/c23/videos/','/media/ruchit/Data/deepfakes/data/original_sequences/actors/c23/videos']


predictions = compute_accuracy(model, 'test',box=True)
print('Final Predictions ',predictions.values())
