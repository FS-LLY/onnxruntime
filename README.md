# onnxruntime

# compile 
g++ resnet_cifar10.cpp -o test  -I /data/ONNX/onnxruntime-linux-x64-gpu-1.12.0/include -L /data/ONNX/onnxruntime-linux-x64-gpu-1.12.0/lib  -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lonnxruntime -std=c++17

 export LD_LIBRARY_PATH=/data/ONNX/onnxruntime-linux-x64-gpu-1.12.0/lib:$LD_LIBRARY_PATH
