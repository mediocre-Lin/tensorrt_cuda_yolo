cmake_minimum_required(VERSION 3.2)
message(${CMAKE_VERSION})
project(test LANGUAGES CXX)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_CXX_STANDARD 17)
include_directories(${PROJECT_SOURCE_DIR})


set(LIB_ALL "")
set(SRC_ALL "")
message(${PROJECT_SOURCE_DIR})

file(GLOB  SRC_COMMON  ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB  CUDA_SRC_COMMON  ${PROJECT_SOURCE_DIR}/*.cu)

list(APPEND SRC_ALL ${SRC_COMMON})
list(APPEND SRC_ALL ${CUDA_SRC_COMMON})
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin/nvcc.exe")
find_package(CUDA REQUIRED)
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
set(CUDA_GEN_CODE "-arch=sm_52 \

-gencode=arch=compute_52,code=sm_52 \

-gencode=arch=compute_60,code=sm_60 \

-gencode=arch=compute_61,code=sm_61 \

-gencode=arch=compute_70,code=sm_70 \

-gencode=arch=compute_75,code=sm_75 \

-gencode=arch=compute_80,code=sm_80 \

-gencode=arch=compute_86,code=sm_86 \

-gencode=arch=compute_86,code=compute_86\
-gencode=arch=compute_90,code=compute_90")
# set(CUDA_GEN_CODE "-gencode=arch=compute_86,code=sm_86")# compute_75：需要根据自己设备的算力来设置                  
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/include")
set(CUDA_LIB_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/lib/x64")

# include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/include")
# set(CUDA_LIB_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/lib/x64")
file(GLOB CUDA_LIBS "${CUDA_LIB_PATH}/*.lib")
list(APPEND LIB_ALL ${CUDA_LIBS})

find_package(OpenMP)
include_directories("D:/dev/env/opencv440-cuda/opencv4.4/include")
include_directories("D:/dev/env/opencv440-cuda/opencv4.4/include/opencv2")
file(GLOB OpenCV_LIBS "D:/dev/env/opencv440-cuda/opencv4.4/x64/vc16/lib/*.lib")
list(APPEND LIB_ALL ${OpenCV_LIBS})


include_directories("D:/dev/env/tensorRT/cu11/TensorRT-8.4.3.1/include")
set(TENSORRT_LIB_PATH "D:/dev/env/tensorRT/cu11/TensorRT-8.4.3.1/lib")

# include_directories("D:/dev/env/tensorRT/TensorRT-10.0.0.6.Windows10.win10.cuda-11.8/TensorRT-10.0.0.6/include")
# set(TENSORRT_LIB_PATH "D:/dev/env/tensorRT/TensorRT-10.0.0.6.Windows10.win10.cuda-11.8/TensorRT-10.0.0.6/lib")

# include_directories("D:/dev/env/tensorRT/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/include")
# set(TENSORRT_LIB_PATH "D:/dev/env/tensorRT/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/lib")
file(GLOB TENOSRT_LIBS "${TENSORRT_LIB_PATH}/*.lib")
list(APPEND LIB_ALL ${TENOSRT_LIBS})

message("LIBS: "${LIB_ALL})
message("SRC: "${SRC_ALL})

cuda_add_executable(test test.cpp ${SRC_ALL})

target_link_libraries(${PROJECT_NAME}  ${LIB_ALL} OpenMP::OpenMP_CXX)

