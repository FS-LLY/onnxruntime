from torch.utils.data import Dataset
import re
import numpy as np
import os
from PIL import Image

class CaltechDataset(Dataset):
    def __init__(self, train=True,large = True,transform = None):
        self.train = train
        self.transform = transform
        images = []
        labels = []
        pattern = re.compile(r'(\d+)\_(\d+)\.png')
        path = "/dataset/caltech-101"
        if large:
            path = os.path.join(path,"224_224")
        else:
            path = os.path.join(path,"32_32")
        if train:
            path = os.path.join(path,"train")
        else:
            path = os.path.join(path,"test")
        #images
        for root, _, files in os.walk(path):
            for name in files:
                #print(name)
                match = pattern.match(name)
                if match:
                    number = int(match.group(1))  # 提取种类
                    label = int(match.group(2))  # 提取编号并转换为整数
                    labels.append(label)
                    images.append(os.path.join(root, name))
                    

        temp = np.array([images, labels])
        temp = temp.transpose()
        np.random.shuffle(temp)

        self.image_pos = list(temp[:, 0])
        label_list = list(temp[:, 1])

        self.label_list = [int(float(i)) for i in label_list]

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        image_path = self.image_pos[idx]
        image = Image.open(image_path)
    
        if image is None:
            print(f"Warning: Unable to read image at {image_path}. Skipping...")
            return None, None  # Or handle this case appropriately

        image = self.transform(image)
        label = self.label_list[idx]
        
        return image, label