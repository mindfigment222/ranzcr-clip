from torch import nn
from torchvision import models



class CustomResnext(nn.Module):
    
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        
        # Finetuning the convnet
        self.model = models.resnext50_32x4d(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    
    def forward(self, x):
        x = self.model(x)
        return x

