from __future__ import print_function, division

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models
from torchvision import transforms as T
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle
from PIL import Image

from datasets.train_dataset import TrainDataset
from train import train_model
from models.resnet50 import CustomResnext
from utils import make_directories


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters
generator_params = {
    'batch_size': 32,
    'shuffle': True,
    'num_workers': 6
}

# data augmentation and normalization for training & just normalization for validation
resize = 256
crop_size = 224
# , interpolation=Image.NEAREST
data_transforms = {
    'train': T.Compose([
        T.Resize((resize, resize)),
        T.CenterCrop(crop_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': T.Compose([
        T.Resize((resize, resize)),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

with open('./data/train_val_split.pkl', 'rb') as f:
    partition = pickle.load(f)

with open('./data/malpositions.pkl', 'rb') as f:
    malpositions = pickle.load(f)

train_dataset_params = {
    'root': '/home/mszmelcz/Datasets/ranzcr-clip/train',
    'img_uids': partition['train'],
    'malpositions': malpositions,
    'transforms': data_transforms['train']
}

val_dataset_params = {
    'root': '/home/mszmelcz/Datasets/ranzcr-clip/train',
    'img_uids': partition['val'],
    'malpositions': malpositions,
    'transforms': data_transforms['val']
}
   
num_epochs = 300

# Datasets & Generators
training_set = TrainDataset(**train_dataset_params)
training_generator = torch.utils.data.DataLoader(training_set, **generator_params)

validation_set = TrainDataset(**val_dataset_params)
validation_generator = torch.utils.data.DataLoader(validation_set, **generator_params)

# Finetuning the pretrained resnext50_32x4d
model = CustomResnext(num_classes=11, pretrained=True)

# Move model to GPU if possible
model = model.to(device)

# Set up loss function
pos_weight = torch.ones([11], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Decay LR by a factor of 0.01 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)

model_dir = 'resnext50'
run_path, params_path, tensorboard_path = make_directories(model_dir)

train_params = {
    'model': model,
    'criterion': criterion,
    'optimizer': optimizer,
    'scheduler': exp_lr_scheduler,
    'num_epochs': num_epochs,
    't_gen': training_generator,
    'v_gen': validation_generator,
    'log_dir': run_path,
    'save_dir': params_path,
    'model_dir': model_dir,
    'device': device
}

model = train_model(**train_params)



    
    
    
    
    
        
