cmake_minimum_required(VERSION 3.1)
cmake_policy(SET CMP0111 NEW)
project(yolov8)
set(CMAKE_PREFIX_PATH "E:/Downloads/libtorch-win-shared-with-deps-1.8.1 cu101/libtorch")
set(CMAKE_MODULE_PATH "E:/Downloads/libtorch-win-shared-with-deps-1.8.1 cu101/libtorch/share/cmake/Torch" ${CMAKE_MODULE_PATH})
find_package(Torch REQUIRED)
find_package(OpenCV CONFIG REQUIRED highgui imgproc cudacodec cudaimgproc cudawarping ximgproc)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
add_executable(yolov8 yolov8.cpp)
target_link_libraries(yolov8 ${TORCH_LIBRARIES} 
                      opencv_highgui
                      opencv_imgproc
                      opencv_cudacodec
                      opencv_cudaimgproc
                      opencv_cudawarping
                      opencv_ximgproc)
set_property(TARGET yolov8 PROPERTY CXX_STANDARD 14)
