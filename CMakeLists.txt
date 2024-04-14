cmake_minimum_required(VERSION 3.18)

project(open3d-cmake-find-package LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

# Find Open3D
find_package(Open3D REQUIRED)

# Find Torch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Set the CUDA architecture (optional but recommended)
set(CMAKE_CUDA_ARCHITECTURES 75)  # Adjust this value based on your GPU architecture

# Add executable and link libraries
add_executable(Draw Draw.cpp)
target_link_libraries(Draw PRIVATE Open3D::Open3D ${TORCH_LIBRARIES})
