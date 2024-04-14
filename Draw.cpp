#include <bits/stdc++.h>
#include "open3d/Open3D.h"
#include <torch/torch.h>


int main(int argc, char *argv[]) {

    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;

    auto sphere = open3d::geometry::TriangleMesh::CreateSphere(1.0);
    sphere->ComputeVertexNormals();
    sphere->PaintUniformColor({0.0, 1.0, 0.0});
    open3d::visualization::DrawGeometries({sphere});
    return 0;
}