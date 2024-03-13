from torchvision import datasets
from torchvision import transforms
from PIL import Image
import os
import numpy as np


transform_test = transforms.Compose(
    [
     #transforms.Normalize((0.1307), (0.3081)),
     transforms.Resize([224,224],transforms.InterpolationMode.BILINEAR)
     ])

testset = datasets.Caltech101("/dataset",transform=transform_test,target_transform=None)
dir = "/dataset/caltech-101"

for i in range(len(testset)):
    image = testset[i][0]
    label = testset[i][1]
    image = image.convert('RGB')
    s_image = image.resize([32,32], Image.BILINEAR)
    img_path = str(i) +"_"+ str(label) + ".png"
    if i%5 != 0:
        image.save(os.path.join(dir,"224_224","train",img_path),mode ="RGB")
        s_image.save(os.path.join(dir,"32_32","train",img_path),mode ="RGB")
    else:
        image.save(os.path.join(dir,"224_224","test",img_path),mode ="RGB")
        s_image.save(os.path.join(dir,"32_32","test",img_path),mode ="RGB")
    

print("Done.")