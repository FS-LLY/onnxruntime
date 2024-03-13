import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                  nn.BatchNorm2d(64), nn.ReLU(),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                   nn.AdaptiveAvgPool2d((1, 1)),
                   nn.Flatten(), nn.Linear(512, 10))

num_features = 224*224
num_classes = 2

random_seed = 1
learning_rate = 0.001
num_epochs = 1

torch.manual_seed(random_seed)

 
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# def train(net, optimizer, loss, train_iter, test_iter, num_epochs, lr, device):
#     """用GPU训练模型"""
#     def init_weights(m):
#         if type(m) == nn.Linear or type(m) == nn.Conv2d:
#             nn.init.xavier_uniform_(m.weight)
#     net.apply(init_weights)
#     print('training on', device)
#     net.to(device)
#     for epoch in range(num_epochs):
#         # 训练损失之和，训练准确率之和，样本数
#         metric = Accumulator(3)
#         net.train()
#         for i, (X, y) in tqdm(enumerate(train_iter)):
#             optimizer.zero_grad()
#             X, y = X.to(device), y.to(device)
#             y_hat = net(X)
#             l = loss(y_hat, y)
#             l.backward()
#             optimizer.step()
#             with torch.no_grad():
#                 metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
#         train_l = metric[0] / metric[2]
#         train_acc = metric[1] / metric[2]
#         test_acc = evaluate_accuracy_gpu(net, test_iter)
#         print("train loss:", train_l)
#         print("train acc:", train_acc)
#         print("test acc:", test_acc)
#     print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
#           f'test acc {test_acc:.3f}')
#     print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
#           f'on {str(device)}')

def train(net, optimizer, loss, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5,gamma=0.5,last_epoch=-1)
    print('training on', device)
    net.to(device)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        net.train()
        train_iter_with_progress = tqdm(train_iter)  # 创建带有进度条的迭代器
        for i, (X, y) in enumerate(train_iter_with_progress):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            #print(X.shape)
            #print(y.shape)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_iter_with_progress.set_description(f"Epoch {epoch+1}/{num_epochs}")  # 更新进度条的描述
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        lr_scheduler.step()
        print("train loss:", train_l)
        print("train acc:", train_acc)
        print("test acc:", test_acc)



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(224, antialias=True)])

transform_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(224, transforms.InterpolationMode.NEAREST),
     transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomAffine(degrees=15, translate=(0.1,0.1)),
    ])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(224, transforms.InterpolationMode.NEAREST)
     ])

batch_size = 128

#trainset = torchvision.datasets.CIFAR10(root='/dataset', train=True,
#                                        download=True, transform=transform)
#trainset = torchvision.datasets.CIFAR10(root='/dataset', train=True,
#                                    download=True, transform=transform)

#trainset = torchvision.datasets.SVHN("/dataset","train",transform_train,None,True)

#trainset = torchvision.datasets.STL10("/dataset",'train', transform=transform_test,target_transform= None, download=True)

trainset = torchvision.datasets.FashionMNIST("/dataset",train=True,transform=transform_train,target_transform=None,download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=8)



#testset = torchvision.datasets.CIFAR10(root='/dataset', train=False,
#                                       download=True, transform=transform)
#testset = torchvision.datasets.CIFAR10(root='/dataset', train=False,
#                                       download=True, transform=transform)
#testset = torchvision.datasets.SVHN("/dataset","test",transform_test,None,True)

#testset = torchvision.datasets.STL10("/dataset","test", transform=transform_test,target_transform= None, download=True)

testset = torchvision.datasets.FashionMNIST("/dataset",train=False,transform=transform_test,target_transform=None,download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)


#torch.backends.cudnn.enabled = False
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

lr, num_epochs = 0.01, 30
train(net, optimizer, criterion, trainloader, testloader, num_epochs, lr, device)

# 定义单张图片的输入大小
single_input_size = (1, 1, 224, 224)

# 导出为ONNX文件
torch.onnx.export(net,
                  torch.randn(*single_input_size).to(device),
                  "../resnet_FashionMNIST_single.onnx",
                  input_names=['input'],
                  output_names=['output'])

