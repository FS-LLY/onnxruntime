from torchvision import datasets
from torchvision import transforms
from PIL import Image
import os
import numpy as np


transform_test = transforms.Compose(
    [
    transforms.ToTensor(),
     #transforms.Normalize((0.1307), (0.3081)),
     transforms.Resize([32,32],transforms.InterpolationMode.NEAREST)
     ])

testset = datasets.FashionMNIST("/dataset",train=False,transform=transform_test,target_transform=None)
dir = "/dataset/FashionMNIST/test"

for i in range(len(testset)):
    image =testset[i]
    label = image[1]
    image = image[0]
    image = image.reshape([32,32]).numpy()
    s_image = (image * 255).astype(np.uint8)
    s_image = Image.fromarray(s_image)
    img_path = str(i)+"_"+str(label)+".png"
    s_image = s_image.convert("RGB")
    s_image.save(os.path.join(dir,"32_32",img_path),mode ="RGB")
    

print("Done.")