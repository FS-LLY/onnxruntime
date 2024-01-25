import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    
BATCH_SIZE = 128
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
NUM_WORKERS = 2

custom_transforms = transforms.Compose([
    transforms.CenterCrop((160, 160)),
    transforms.Resize([128, 128]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_dataloaders_celeba(batch_size, num_workers=0,
                           train_transforms=None,
                           test_transforms=None,
                           download=True):

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()
        
    get_smile = lambda attr: attr[31]

    train_dataset = datasets.CelebA(root='/dataset',
                                    split='train',
                                    transform=train_transforms,
                                    target_type='attr',
                                    target_transform=get_smile,
                                    download=download)

    valid_dataset = datasets.CelebA(root='/dataset',
                                    split='valid',
                                    target_type='attr',
                                    target_transform=get_smile,
                                    transform=test_transforms)

    test_dataset = datasets.CelebA(root='/dataset',
                                   split='test',
                                   target_type='attr',
                                   target_transform=get_smile,
                                   transform=test_transforms)


    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)

    valid_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    return train_loader, valid_loader, test_loader

class vgg16_conv_block(nn.Module):
    def __init__(self, input_channels, out_channels, rate=0.4, drop=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, out_channels, 3 ,1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(rate)
        self.drop =drop
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.drop:
            x = self.dropout(x)
        return(x)

def vgg16_layer(input_channels, out_channels, num, dropout=[0.4, 0.4]):
    result = []
    result.append(vgg16_conv_block(input_channels, out_channels, dropout[0]))
    for i in range(1, num-1):
        result.append(vgg16_conv_block(out_channels, out_channels, dropout[1]))
    if num>1:
        result.append(vgg16_conv_block(out_channels, out_channels, drop=False))
    result.append(nn.MaxPool2d(2,2))
    return(result)






class VGG16(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(VGG16, self).__init__()
        
        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2
        
        b1 = nn.Sequential(*vgg16_layer(3,64,2,[0.3,0.4]), *vgg16_layer(64,128,2), *vgg16_layer(128,256,3), 
                   *vgg16_layer(256,512,3),*vgg16_layer(512,512,3))
        b2 = nn.Sequential(nn.Dropout(0.5), nn.Flatten(), nn.Linear(512, 512, bias=True), nn.BatchNorm1d(512), nn.ReLU(inplace=True), 
                  nn.Linear(512,num_classes, bias=True))
        self.net = nn.Sequential(b1, b2)
            
        
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, np.sqrt(2. / n))
                m.weight.detach().normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.detach().normal_(0, 0.05)
                m.bias.detach().detach().zero_()
        
        
    def forward(self, x):

        logits = self.net(x)
        probas = F.softmax(logits, dim=1)
        
        
        return logits, probas