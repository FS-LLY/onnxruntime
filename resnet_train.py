import torch
import visdom
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from resnet import *
from torch.nn import CrossEntropyLoss
from torch import optim
import time


BATCH_SIZE = 32                        # 超参数batch大小
save_path = "./CIFAR10_ResNet18.pth"    # 模型权重参数保存位置
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')    # CIFAR10数据集类别
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                         # 创建GPU运算环境
print(device)

data_transform = {                      # 数据预处理
    "train": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "vaild": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
}

train_data = datasets.CIFAR10(root='/dataset', train=True, transform=data_transform["train"], download=True)
vaild_data = datasets.CIFAR10(root='/dataset', train=False,transform=data_transform["vaild"], download=True)
test_data = datasets.CIFAR10(root='/dataset', train=False, transform=data_transform["val"], download=True)

train_loader = torch.utils.data.DataLoader(train_data, BATCH_SIZE, True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(vaild_data, BATCH_SIZE, False, num_workers = 0)
test_loader = torch.utils.data.DataLoader(test_data, BATCH_SIZE, False, num_workers=0)

# # 展示图片
# x = 0
# for images, labels in train_data:
#     plt.subplot(3,3,x+1)
#     plt.tight_layout()
#     images = images.numpy().transpose(1, 2, 0)  # 把channel那一维放到最后
#     plt.title(str(classes[labels]))
#     plt.imshow(images)
#     plt.xticks([])
#     plt.yticks([])
#     x += 1
#     if x == 9:
#         break
# plt.show()

# 创建一个visdom，将训练测试情况可视化
#viz = visdom.Visdom()

# 测试函数，传入模型和数据读取接口
def evalute(model, loader):
    # correct为总正确数量，total为总测试数量
    correct = 0
    total = len(loader.dataset)
    # 取测试数据
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # validation和test过程不需要反向传播
        model.eval()
        with torch.no_grad():
            out = model(x)       # 计算测试数据的输出logits
            # 计算出out在第一维度上最大值对应编号，得模型的预测值
            prediction = out.argmax(dim=1)
        # 预测正确的数量correct
        correct += torch.eq(prediction, y).float().sum().item()
    # 最终返回正确率
    return correct / total


# Hyperparameters
random_seed = 1
learning_rate = 0.001
num_epochs = 100
weight_decay = 5e-4
momentum = 0.9
layers = [3,4,6,3]

# Architecture
num_classes = 10


torch.manual_seed(random_seed)
model = ResNet(block=Bottleneck,
               layers=layers,
              num_classes=num_classes,
              zero_init_residual= False,
              groups = 1,
              width_per_group = 64,
              replace_stride_with_dilation = None,
              norm_layer = None,)

model = model.to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, nesterov=True)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, threshold=0.001, mode='max', verbose=True)

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)
        predicted_labels = logits.argmax(dim = 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100
    
def validate(model):
    running_vloss = 0.0
    with torch.no_grad():
        for i, (vinputs, vlabels) in enumerate(valid_loader):
            
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            
            voutputs = model(vinputs)
            cost_fun = nn.CrossEntropyLoss()
            vloss = cost_fun(voutputs, vlabels)
            running_vloss += vloss
            
    avg_vloss = running_vloss / (i + 1)
    return avg_vloss

start_time = time.time()
overall_vloss = 0.0
for epoch in range(num_epochs):
    
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(device)
        targets = targets.to(device)
        
        ### FORWARD AND BACK PROP
        logits = model(features)
        cost_fun = nn.CrossEntropyLoss()
        loss = cost_fun(logits, targets)
        optimizer.zero_grad()
        
        loss.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), loss))
    
    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
              epoch+1, num_epochs, 
              compute_accuracy(model, train_loader),
              compute_accuracy(model, valid_loader)))
        
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    if epoch % 5 == 0:
        overall_vloss += validate(model)
        vloss = overall_vloss / 5
        overall_vloss = 0.0
        
        scheduler.step(vloss)
    else:
         overall_vloss += validate(model)
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))

'''
torch.onnx.export(net,                                # model being run
                  torch.randn(512,3, 32, 32).to(device),    # model input (or a tuple for multiple inputs)
                  "resnet18_cifar10.onnx",           # where to save the model (can be a file or file-like object)
                  input_names = ['input'],              # the model's input names
                  output_names = ['output'])            # the model's output names

'''
print("Finish !")