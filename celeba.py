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

get_smile = lambda attr: attr[31]


test_dataset = datasets.CelebA(root='/dataset',
                                   split='test',
                                   target_type='attr',
                                   target_transform=get_smile,
                                   )

data_size = len(test_dataset)

labelfile = "labelfile.txt"
with open(labelfile, 'w') as f:
    for idx in range(data_size ):    
        name = test_dataset.filename[idx]
        features, targets = test_dataset[idx]
        lines = [name, " ", str(targets.item()), "\n"]
        f.writelines(lines)
   