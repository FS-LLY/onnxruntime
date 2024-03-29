# Training and export ONNX model

Only need pytorch environment

## VGG16 

Need: VGG16_model.py  and corresponding .py file for different dataset

### celeba
Train:
```
python VGG16_train.py
```
Model input size: (128,3, 128, 128)

Output size: (128,2)

### cifar-10
Train:
```
python VGG16_cifar10.py
```
Model input size: (1,3, 32, 32)

Output size: (1,10)

### cifar-10
Train:
```
python VGG16_cifar100.py
```
Model input size: (1,3, 32, 32)

Output size: (1,100)

## Resnet 
Need: resnet_model.py  and corresponding .py file for different dataset

### cifar-10
Train:
```
python resnet_train_cifar10_single.py
```
Model input size: (1,3, 224, 224)

Output size: (1,10)
### cifar-100
Train:
```
python resnet_train_cifar100_single.py
```
Model input size: (1,3, 224, 224)

Output size: (1,10)

### catanddog
Downloading data in feishu group chat.

Train:
```
python resnet_train_catanddog.py
```
Model input size: (128,3, 224, 224)

Output size: (128,2)

### 102flower

Train:
```
python resnet_train_102flower.py
```
Model input size: (128,3, 224, 224)

Output size: (128,103) (Additional type for non-flower)

# Running ONNX model in python

## Install

```
pip install onnxruntime (cpu version)
pip install onnxruntime-gpu (cpu+gpu version)
```

## Running
```
python VGG16_onnx_test.py (VGG16 + celeba dataset)
python resnet_onnx.py (Resnet18 + cifar-10)
```

# Quantification (dynamic)

Preprocess of your model first. (Use resnet & cifar10 as example)

```
python -m onnxruntime.quantization.preprocess --input resnet_cifar10_single.onnx --output resnet_cifar10_single_infer.onnx
```
Then make a quantification on the model after preprocessing

```
python resnet_cifar10_quantize.py
```
The model could be directly used as original model, but be careful to the opset used.

# Rasberry

Buy a mini_HDMI - HDMI line: https://m.tb.cn/h.5ndT8Ir?tk=QkygWgtvV2O

Prepare a display, keybroad, mouse, a SD Card and a SD Card reader.

Downloading software and tutorial: https://www.yahboom.com/study/raspberry4B Password: nvz8

OS install: https://www.raspberrypi.com/software/ 

Please select your device, 64-bit OS and the drive letter of the SD card, and wait for write-in.

If your device is rasberry 4B, you could try to install the OS in package (.img files)

Insering SD card, connecting Rasberry to power, display, keyboard and mouse (see the picture in tutorial) 

Connect your resberry to wifi. And enter
```
ifconfig
```

to get IP address. Then you could connect it with ssh and VNC viewer.

For SSH connection, VNC viewer, file translation and so on, read official tutorial for more details. 


# onnxruntime in c++

## Install library
Opencv 
```
apt install libopencv-dev  
```

Config where your header file and lib file installed. Normally it should in /usr/local/lib

onnxruntime: https://github.com/microsoft/onnxruntime/releases/tag/v1.16.3

Download x64 version for Ubuntu 20.04, aarch64 version for Rasberry

Just extract the package 

## Data process

### Cifar-10
```
python cifar10.py
```
### Cifar-100
```
python cifar100.py
```
### celeba
```
python celeba.py
```
Here it will export a file with a list of the test picture and the label (whether it's smiling)

## Compile
Use resnet18 & cifar10 as example
```
g++ resnet_cifar10.cpp -o test  -I /data/ONNX/onnxruntime-linux-x64-gpu-1.16.3/include -L /data/ONNX/onnxruntime-linux-x64-gpu-1.16.3/lib  -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lonnxruntime -std=c++17
```

## Excute
```
export LD_LIBRARY_PATH=/data/ONNX/onnxruntime-linux-x64-gpu-1.16.3/lib:$LD_LIBRARY_PATH
./test
```
# Result
## VGG16 (celeba)
### Acc
Official Dataset : 89.62%

onnxruntime_python: 89.63%

onnxruntime_c++ :91.40% 

onnxruntime_c++  (Quantification to Uint8, on Intel(R) Xeon(R) W-2265 CPU) : 91.3%

onnxruntime_c++ （Quantification to Uint8, In rasberry）: 91.1%

### Time consuming (1000 pictures)
#### before quantification (float32 model)

On Intel(R) Xeon(R) W-2265 CPU @ 3.50GHz (linux_x64_gpu_v1.16.3) : 42.5193 s

On Respberry Pi 5 (linux_Aarch64_v1.16.3) : 182.307s


#### Uint8 quantification

On Intel(R) Xeon(R) W-2265 CPU @ 3.50GHz (linux_x64_gpu_v1.16.3) :  62.8612 s 

On Respberry Pi 5 (linux_Aarch64_v1.16.3) :  157.009s

## VGG16 (cifar-10)
### Acc
Official Dataset : 87.43%

onnxruntime_python: 87.44%

onnxruntime_c++ : 87.44%

onnxruntime_c++  (Quantification to Uint8, on Intel(R) Xeon(R) W-2265 CPU) : 87.41%

onnxruntime_c++ （Quantification to Uint8, In rasberry）: 87.41%

### Time consuming (10000 pictures)
#### before quantification (float32 model)

On Intel(R) Xeon(R) W-2265 CPU @ 3.50GHz (linux_x64_gpu_v1.16.3) :  25.375 s

On Respberry Pi 5 (linux_Aarch64_v1.16.3) :  226.165 s

#### Uint8 quantification

On Intel(R) Xeon(R) W-2265 CPU @ 3.50GHz (linux_x64_gpu_v1.16.3) :   34.5375 s 

On Respberry Pi 5 (linux_Aarch64_v1.16.3) :  80.4149 s

## VGG16 (cifar-100)
### Acc
Official Dataset : 63.79%

onnxruntime_python: 63.79%

onnxruntime_c++ : 63.79%

onnxruntime_c++  (Quantification to Uint8, on Intel(R) Xeon(R) W-2265 CPU) :63.79%

onnxruntime_c++ （Quantification to Uint8, In rasberry）: 63.70%

### Time consuming (10000 pictures)
#### before quantification (float32 model)

On Intel(R) Xeon(R) W-2265 CPU @ 3.50GHz (linux_x64_gpu_v1.16.3) :  25.7986 s 

On Respberry Pi 5 (linux_Aarch64_v1.16.3) :  218.697 s

#### Uint8 quantification

On Intel(R) Xeon(R) W-2265 CPU @ 3.50GHz (linux_x64_gpu_v1.16.3) :   32.9697 s 

On Respberry Pi 5 (linux_Aarch64_v1.16.3) :  82.2498 s 

## Resnet (cifar-10)
### Acc
Official Dataset :81.96%

onnxruntime_python: 81.96%

onnxruntime_c++ :81.96%

onnxruntime_c++  (Quantification to Uint8, on Intel(R) Xeon(R) W-2265 CPU) :81.94%

onnxruntime_c++ （Quantification to Uint8, In rasberry）: 81.99%

### Time consuming (10000 pictures)
#### before quantification (float32 model)

On Intel(R) Xeon(R) W-2265 CPU @ 3.50GHz (linux_x64_gpu_v1.16.3) :  54.4086 s

On Respberry Pi 5 (linux_Aarch64_v1.16.3) : 557.804s

#### Uint8 quantification

On Intel(R) Xeon(R) W-2265 CPU @ 3.50GHz (linux_x64_gpu_v1.16.3) : 96.0546 s

On Respberry Pi 5 (linux_Aarch64_v1.16.3) : 387.571 s

## Resnet (cifar-100)
### Acc
Official Dataset :55.54%

onnxruntime_python: 56.78%

onnxruntime_c++ : 56.78%

onnxruntime_c++  (Quantification to Uint8, on Intel(R) Xeon(R) W-2265 CPU) : 54.70%

onnxruntime_c++ （Quantification to Uint8, In rasberry）:56.71%

### Time consuming (10000 pictures)
#### before quantification (float32 model)

On Intel(R) Xeon(R) W-2265 CPU @ 3.50GHz (linux_x64_gpu_v1.16.3) : 54.9864 s 

On Respberry Pi 5 (linux_Aarch64_v1.16.3) : 538.99 s

#### Uint8 quantification

On Intel(R) Xeon(R) W-2265 CPU @ 3.50GHz (linux_x64_gpu_v1.16.3) : 97.4339 s

On Respberry Pi 5 (linux_Aarch64_v1.16.3) : 406.11 s





