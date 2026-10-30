cmake_minimum_required(VERSION 3.24)
project(tiny_llm_infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# RTX 5060 = Blackwell consumer = compute capability 12.0. Change this for other GPUs.
set(CMAKE_CUDA_ARCHITECTURES 120)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(tiny_infer src/main.cu src/loader.cu src/kernels.cu)
target_include_directories(tiny_infer PRIVATE include src)
target_link_libraries(tiny_infer PRIVATE CUDA::cudart CUDA::cublas)
