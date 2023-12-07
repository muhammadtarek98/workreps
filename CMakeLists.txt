cmake_minimum_required(VERSION 3.1)
cmake_policy(SET CMP0111 NEW)
project(example)
set(CMAKE_PREFIX_PATH "C:/libtorch-win-shared-with-deps-debug-2.1.1+cu118/libtorch")
set(CMAKE_MODULE_PATH "C:/libtorch-win-shared-with-deps-debug-2.1.1+cu118/libtorch/share/cmake/Torch" ${CMAKE_MODULE_PATH})
find_package(Torch REQUIRED)
#find_package(OpenCV CONFIG REQUIRED highgui imgproc cudacodec cudaimgproc cudawarping ximgproc)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
add_executable(example example.cpp)
target_link_libraries(example ${TORCH_LIBRARIES} )
#                      opencv_highgui
#                      opencv_imgproc
#                      opencv_cudacodec
#                      opencv_cudaimgproc
#                      opencv_cudawarping
#                      opencv_ximgproc)
set_property(TARGET example PROPERTY CXX_STANDARD 17)
