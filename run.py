from __future__ import print_function, division

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle
from PIL import Image

from datasets.basic_dataset import Dataset
from train import train_model

plt.ion()   # interactive mode


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters
generator_params = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 6
}

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
}

with open('./data/train_val_split.pkl', 'rb') as f:
    partition = pickle.load(f)

with open('./data/malpositions.pkl', 'rb') as f:
    malpositions = pickle.load(f)

train_dataset_params = {
    'root': '/home/stanislaw/datasets/ranzcr-clip/train',
    'img_uids': partition['train'],
    'malpositions': malpositions,
    'transforms': data_transforms['train']
}

val_dataset_params = {
    'root': '/home/stanislaw/datasets/ranzcr-clip/train',
    'img_uids': partition['val'],
    'malpositions': malpositions,
    'transforms': data_transforms['val']
}
   
num_epochs = 100

# Datasets & Generators
training_set = Dataset(**train_dataset_params)
training_generator = torch.utils.data.DataLoader(training_set, **generator_params)
training_set_size = len(training_set)

validation_set = Dataset(**val_dataset_params)
validation_generator = torch.utils.data.DataLoader(validation_set, **generator_params)
validation_set_size = len(validation_set)

# Finetuning the convnet
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

# Here the size of each output sample is set to 11.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 11)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

train_params = {
    'model': model_ft,
    'criterion': criterion,
    'optimizer': optimizer_ft,
    'scheduler': exp_lr_scheduler,
    'num_epochs': num_epochs,
    't_gen': training_generator,
    't_size': training_set_size,
    'v_gen': validation_generator,
    'v_size': validation_set_size,
    'device': device
}

model_ft = train_model(**train_params)

# visualize_model(model_ft)

# ConvNet as fixed feature extractor

# for param in model_conv.parameters():
#     param.requires_grad = False

# # Parameters of newly constructed modules have requires_grad=True by default
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, 2)

# model_conv = model_conv.to(device)

# criterion = nn.CrossEntropyLoss()

# # Observe that only parameters of final layer are being optimized as
# # opposed to before.
# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)



    
    
    
    
    
        
