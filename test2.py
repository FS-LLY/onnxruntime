from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from PIL import Image
import cv2

data_transform = {                      # 数据预处理
    "train": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize(224, antialias=True)
    ]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #transforms.Resize(224, antialias=True)
    ])
}
test_data = datasets.CIFAR10(root='/dataset', train=False, transform=data_transform["val"], download=True)

specific_image = Image.open("/dataset/cifar-10-batches-py/test/5_1000.jpg")
image1=data_transform["val"](specific_image)

image2 = test_data[1000]

image1= (image1*255).to(dtype= torch.int).numpy()
image1 = image1.transpose(1, 2, 0)
image2= (image2[0]*255).to(dtype= torch.int).numpy()
image2 = image2.transpose(1, 2, 0)
cv2.imwrite("/data/ONNX/image1.jpg", image1)
cv2.imwrite("/data/ONNX/image2.jpg", image2)