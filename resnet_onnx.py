import torch
import onnx
import onnxruntime as ort
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
      

def evalute(model, loader):
    # correct为总正确数量，total为总测试数量
    correct = 0
    total = len(loader.dataset)
    # 取测试数据
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # validation和test过程不需要反向传播
        #model.eval()
        with torch.no_grad():
            if x.shape[0] == 1:
                out = model.run(None,{"input":x.cpu().numpy()})       # 计算测试数据的输出logits
                # 计算出out在第一维度上最大值对应编号，得模型的预测值
                out = torch.from_numpy(out[0]).to(device)
                prediction = out.argmax(dim=1)
                # 预测正确的数量correct
                correct += torch.eq(prediction, y).float().sum().item()
    # 最终返回正确率
    return correct / total
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
BATCH_SIZE = 1  
data_transform = {                      # 数据预处理
    "train": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize(224, antialias=True)
    ]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize(224, antialias=True)
    ])
}
onnx_model = onnx.load("resnet_cifar10_single.onnx")
onnx.checker.check_model(onnx_model)
ort_sess = ort.InferenceSession('resnet_cifar10_single.onnx')
test_data = datasets.CIFAR10(root='/dataset', train=False, transform=data_transform["val"], download=True)
test_dataloader = torch.utils.data.DataLoader(test_data, BATCH_SIZE, False, num_workers=0)
T1 = time.time()
test_acc = evalute(ort_sess, test_dataloader)
T2 = time.time()
print("test acc:{}".format(test_acc))
print("Running time:{}ms".format((T2-T1)*1000))