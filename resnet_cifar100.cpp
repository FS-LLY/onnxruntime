#include "onnxruntime_cxx_api.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <filesystem>
#include <iostream>
#include <regex>
#include <chrono>   

namespace fs = std::filesystem;
using namespace std::chrono;
int main()
{
    Ort::Env env;
    std::string weightFile = "./resnet_cifar100_single.onnx";

    Ort::SessionOptions session_options;
    OrtCUDAProviderOptions options;
    options.device_id = 0;
    options.arena_extend_strategy = 0;
    // options.cuda_mem_limit = (size_t)1 * 1024 * 1024 * 1024;//onnxruntime1.7.0
    options.gpu_mem_limit = (size_t)1 * 1024 * 1024 * 1024; // onnxruntime1.8.1, onnxruntime1.9.0
    options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
    options.do_copy_in_default_stream = 1;
    session_options.AppendExecutionProvider_CUDA(options);
    Ort::Session session_{env, weightFile.c_str(), Ort::SessionOptions{nullptr}}; // CPU
    // Ort::Session session_{env, weightFile.c_str(), session_options}; //GPU

    static constexpr const int width_ = 224;  // 模型input width
    static constexpr const int height_ = 224; // 模型input height
    Ort::Value input_tensor_{nullptr};
    std::array<int64_t, 4> input_shape_{1, 3, height_, width_}; // NCHW, 1x3xHxW

    Ort::Value output_tensor_{nullptr};
    std::array<int64_t, 2> output_shape_{1, 100}; // 模型output shape，此处假设是二维的(1,10)

    std::array<float, width_ * height_ * 3> input_image_{}; // 输入图片，HWC
    std::array<float, 100> results_{};                       // 模型输出，注意和output_shape_对应
    std::string folderPath = "/dataset/cifar-100-python/testdir/";
    int num = 0;
    int correct_num = 0;
    auto start = system_clock::now();
    for (const auto &entry : fs::directory_iterator(folderPath))
    {
        std::string imgPath = entry.path().string();
        std::regex labelRegex("(\\d+)_(\\d+).png");
        std::smatch match;
        if (std::regex_search(imgPath, match, labelRegex))
        {
            // match[1] 匹配 x，match[2] 匹配 yyyy
            std::string labelStr = match[1];
            std::string imageNumberStr = match[2];
            // std::cout<<match[1]<<std::endl;
            int label = std::stoi(labelStr);
            int imageNumber = std::stoi(imageNumberStr);
            // int label = 0;
            // std::cin>>label;
            /*
            if(imageNumber!=8870){
                continue;
            }
            */
            num++;
            // 读取图像
            cv::Mat img = cv::imread(imgPath,-1);
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
            output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
            const char *input_names[] = {"input"};   // 输入节点名
            const char *output_names[] = {"output"}; // 输出节点名
            // 预处理
            cv::Mat img_f32;
            img.convertTo(img_f32, CV_32FC3, 1.0 / 255.0);
            //img_f64.convertTo(img_f64, CV_64FC3, 1.0 / 255.0);
            cv::Size dsize = cv::Size(224, 224);
            cv::Mat img_resized;
            cv::resize(img_f32, img_resized, dsize, 0, 0, cv::INTER_LINEAR);
            cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);
            // Calculate mean and standard deviation
            // Flatten and copy data to input_image_
            int channels = img_resized.channels();
            int height = img_resized.rows;
            int width = img_resized.cols;

            // Calculate mean and standard deviation
            //cv::Scalar mean, stddev;
            //cv::meanStdDev(img_resized, mean, stddev);

            //cv::Mat img_normalized;
            //cv::subtract(img_resized, mean, img_normalized);
            //cv::divide(img_normalized, stddev, img_normalized);
            cv::Mat img_data;
            img_data = img_resized;
            float mean[3] = {0.5, 0.5, 0.5};
            float stddev[3] = {0.5, 0.5, 0.5};

            for (int c = 0; c < channels; ++c)
            {
                for (int h = 0; h < height; ++h)
                {
                    
                    for (int w = 0; w < width; ++w)
                    {
                        int index = c * height * width + h * width + w;
                        //std::cout<<img_data.at<cv::Vec3f>(h, w)[c]<<" ";
                        input_image_[index] = (img_data.at<cv::Vec3f>(h, w)[c] - mean[c]) / stddev[c];
                        //std::cout<<input_image_[index]<<std::endl;
                        //std::cin>>a;
                    }
                    //std::cout<<std::endl;
                }
            }
            //std::copy(img_data.begin(), img_data.end(), input_image_.data());
            session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);

            // 获取output的shape
            Ort::TensorTypeAndShapeInfo shape_info = output_tensor_.GetTensorTypeAndShapeInfo();

            // 获取output的dim
            size_t dim_count = shape_info.GetDimensionsCount();
            // std::cout<< dim_count << std::endl;

            // 获取output的shape
            std::vector<int64_t> dims = shape_info.GetShape();
            // std::cout<< dims[0] << "," << dims[1] << std::endl;
            // 取output数据
            float *f = output_tensor_.GetTensorMutableData<float>();
            int max = 0;
            float max_prob = f[0];
            for (int i = 0; i < dims[1]; i++)
            {
                //std::cout << "i: " << i << " prob: " << f[i] << std::endl;
                if (f[i] > max_prob)
                {
                    max = i;
                    max_prob = f[i];
                }
            }
            // std::cout << "result:" << max << std::endl;
            if (max == label)
            {
                correct_num++;
            }
            else{
                //std::cout<<label<<", "<<imageNumber<<std::endl;
            }
        }
        else
        {
            std::cerr << "Invalid file name format: " << imgPath << std::endl;
        }
    }
    if (num > 0)
    {   
        auto end   = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        std::cout << "Total pictures: " << num << std::endl;
        std::cout << "Correct pictures: " << correct_num << std::endl;
        std::cout << "acc: " << (float)correct_num / (float)num << std::endl;
        std::cout << "time cost: "<<double(duration.count()) * microseconds::period::num / microseconds::period::den <<" s "<<std::endl;
    }
    // std::string imgPath = "/dataset/cifar-10-batches-py/test/0_4387.jpg";
    // cv::Mat img = cv::imread(imgPath);
}
