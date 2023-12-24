# Rasberry

Buy a mini_HDMI - HDMI line: https://m.tb.cn/h.5ndT8Ir?tk=QkygWgtvV2O

OS install: https://www.raspberrypi.com/software/

See the tutorial for more detail 


# onnxruntime

## Install
Opencv: apt install libopencv-dev

onnxruntime: https://github.com/microsoft/onnxruntime/releases/tag/v1.16.3

## Compile 
g++ resnet_cifar10.cpp -o test  -I /data/ONNX/onnxruntime-linux-x64-gpu-1.12.0/include -L /data/ONNX/onnxruntime-linux-x64-gpu-1.12.0/lib  -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lonnxruntime -std=c++17

## Excute
export LD_LIBRARY_PATH=/data/ONNX/onnxruntime-linux-x64-gpu-1.12.0/lib:$LD_LIBRARY_PATH

# Result

## Resnet (cifar-10)

Official Dataset :81.96%

onnxruntime_python: 81.96%

onnxruntime_c++ :82.00% 

In Intel(R) Xeon(R) W-2265 CPU @ 3.50GHz (linux_x64_gpu_v1.12.0) : 70.9374 s

In Respberry Pi 5 (linux_Aarch64_v1.16.3) : 567.451s

