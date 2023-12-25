import onnx
import onnxruntime as ort
import torch
from model import *
import numpy as np

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)
        if targets.shape[0] == 128:
            logits, probas = model.run(None,{"input":features.cpu().numpy()})
            probas = torch.from_numpy(probas).to(device)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

onnx_model = onnx.load("VGG16_model.onnx")
onnx.checker.check_model(onnx_model)
ort_sess = ort.InferenceSession('VGG16_model.onnx')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
train_loader, valid_loader, test_loader = get_dataloaders_celeba(
    batch_size=BATCH_SIZE,
    train_transforms=custom_transforms,
    test_transforms=custom_transforms,
    download=False,
    num_workers=4)


with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(ort_sess, test_loader)))


