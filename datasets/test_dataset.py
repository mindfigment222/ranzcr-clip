import os
import numpy as np
import torch
from PIL import Image


class TestDataset(torch.utils.data.Dataset):
    
    def __init__(self, root, img_uids, transforms=None):
        'Initialization'
        self.root = root # '/kaggle/input/ranzcr-clip-catheter-line-classification/'
        self.transforms = transforms
        self.img_uids = sorted(img_uids)
        
    
    def __getitem__(self, index):
        'Generates one sample of data'
        uid = self.img_uids[index]
        img_path = os.path.join(self.root, uid + '.jpg')
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img

    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_uids)