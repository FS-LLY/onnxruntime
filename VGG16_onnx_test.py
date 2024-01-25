import torchvision
import torchvision.transforms as transforms
import cv2
from PIL import Image
import onnxruntime as ort
import torch
import os
import re

folder_path = "/dataset/cifar-100-python/testdir/"

# Define the regular expression pattern
label_regex = re.compile(r"(\d+)_(\d+).jpg")
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
ort_sess = ort.InferenceSession('VGG16_cifar100_single.onnx')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
total_number = 0
correct_number = 0
# Iterate through files in the directory
for entry in os.scandir(folder_path):
    if entry.is_file() and entry.name.endswith(".jpg"):
        # Get the path of the current file
        img_path = entry.path
        
        # Match the regular expression pattern
        match = label_regex.search(img_path)
        
        if match:
            # Extract matched groups
            label_str, image_number_str = match.groups()
            total_number+=1
            # Convert strings to integers
            label = int(label_str)
            image_number = int(image_number_str)
            specific_image = Image.open(img_path)
            
            specific_image = transform(specific_image).unsqueeze(0) # Apply the same transformation
            specific_image = specific_image[:,[2,1,0],:,:]
            out = ort_sess.run(None,{"input":specific_image.cpu().numpy()})       # 计算测试数据的输出logits
            out = torch.from_numpy(out[0]).to(device)
            prediction = out.argmax(dim=1)
            if prediction == label:
                correct_number+=1

print("total: ",total_number)
print("correct: ",correct_number)
print("acc: ",correct_number/total_number)

