import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from model import VGG16
import time

# 图像增强
num_features = 32*32
num_classes = 10
batch_size = 128
random_seed = 1
learning_rate = 0.005
num_epochs = 30

torch.manual_seed(random_seed)
model = VGG16(num_features=num_features,
              num_classes=num_classes)
torch.cuda.empty_cache()
torch.cuda.max_split_size_mb = 140
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10,gamma=0.5,last_epoch=-1)

transform_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomAffine(degrees=15, translate=(0.1,0.1)),
    transforms.Normalize((0.1307), (0.3081)),
     transforms.Resize(32, antialias=True)
    ])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307), (0.3081)),
     transforms.Resize(32, antialias=True)
     ])


#trainset = torchvision.datasets.CIFAR10(root='/dataset', train=True,
#                                    download=True, transform=transform_train)
trainset = torchvision.datasets.FashionMNIST("/dataset","train",transform_train,None,True)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=8)

#validset = torchvision.datasets.CIFAR10(root='/dataset', train=False,
#                                        download=False, transform=transform_test)
vaildset = torchvision.datasets.FashionMNIST("/dataset","test",transform_test,None,True)
vaildloader = torch.utils.data.DataLoader(vaildset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)

#testset = torchvision.datasets.CIFAR10(root='/dataset', train=False,
#                                       download=True, transform=transform_test)
testset = torchvision.datasets.FashionMNIST("/dataset","test",transform_test,None,True)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#torch.backends.cudnn.enabled = False


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    
    return correct_pred.float()/num_examples * 100
    

start_time = time.time()
for epoch in range(num_epochs):
    
    model.train()
    for batch_idx, (features, targets) in enumerate(trainloader):
        
        features = features.to(device)
        targets = targets.to(device)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(trainloader), cost))
            
    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
            epoch+1, num_epochs, 
            compute_accuracy(model, trainloader),
            compute_accuracy(model, vaildloader)))
    
    lr_scheduler.step()
    print('Learning rate:',lr_scheduler.get_last_lr())
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, testloader)))

# 定义单张图片的输入大小
single_input_size = (1, 1, 32, 32)

# 导出为ONNX文件
torch.onnx.export(model,
                  torch.randn(*single_input_size).to(device),
                  "VGG16_FashionMNIST_single.onnx",
                  input_names=['input'],
                  output_names=['output'])

