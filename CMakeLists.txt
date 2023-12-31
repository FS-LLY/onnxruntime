cmake_minimum_required(VERSION 3.10.2)
project(YourProjectName)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add the path to ONNX Runtime headers
include_directories(/data/ONNX/onnxruntime-linux-x64-gpu-1.12.0/include)

# Add the path to ONNX Runtime and OpenCV libraries
link_directories(/data/ONNX/onnxruntime-linux-x64-gpu-1.12.0/lib /usr/local/lib)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add your source file
add_executable(resnet_cifar10 resnet_cifar10.cpp)

# Link against necessary libraries
target_link_libraries(resnet_cifar10 PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
)

# Optional: If you need GPU support, you might also want to link against CUDA libraries
# find_package(CUDA REQUIRED)
# target_link_libraries(test PRIVATE ${CUDA_LIBRARIES})
