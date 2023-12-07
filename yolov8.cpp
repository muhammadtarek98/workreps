#include<iostream>

#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include<opencv2/opencv.hpp>

//#include<opencv2/dnn.hpp>

#include<torch/torch.h>
#include<ATen/TensorIndexing.h>
#include<Windows.h>
#include"yolov8.hpp"


//torch::Tensor MatToTensor(cv::Mat const& img, int batchSize) {
//        return torch::from_blob(img.data, { batchSize, img.rows, img.cols, static_cast<int64_t>(img.channels()) }).permute({ 0, 3, 1, 2 });
//    }
int main()
{
    // load an image
    cv::Mat img = cv::imread("E:/Datasets/flanges/test/images/din-standard.png", cv::IMREAD_COLOR);
    std::cout << "image read: " << img.size << '\n';

    torch::Device device(torch::kCUDA,0);

    // load the model
    auto model = torch::jit::load("yolov8n_cpu.torchscript", device);
    //model.to(at::kCUDA);
    std::cout << "model loaded\n";
    
    
    //mat to tensor
    auto img_tensor = torch::from_blob(img.data, { 1, 3,640, 640 });
    //auto img_tensor = cv::dnn::blobFromImage();
    std::cout << "image tensor: " << img_tensor.sizes() << '\n';
        
    auto output = model.forward({img_tensor.to(device)});
    std::cout << "Done!\n";
    auto e = output.toTensor().detach().index({0});
    for (auto box : e) {

    }
    //outputs.toTuple()->elements()[0].toTensor().detach().squeeze().permute({ 1, 2, 0 }).contiguous().mul(255).clamp(0, 255).to(torch::kU8);
    //std::cout << output.toTuple()->elements()[0].toTensor().detach().sizes();

    return 0;
}