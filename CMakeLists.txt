cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

## Specify C and C++ Compiler path in CMAKE_C_COMPILER and CMAKE_CXX_COMPILER variable
#set(CMAKE_C_COMPILER   "/opt/intel/compilers_and_libraries/linux/bin/intel64/icc")
#set(CMAKE_CXX_COMPILER "/opt/intel/compilers_and_libraries/linux/bin/intel64/icpc")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

## Set the LANGUAGES for CMAKE Project
project(AMG LANGUAGES CXX CUDA)

## Set CMAKE CXX_BASIC_FLAGS
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11")
#set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

#list(APPEND CXX_BASIC_FLAGS "-g" "-O3" "-fopenmp" "-Wno-cpp")

#include_directories("${CMAKE_SOURCE_DIR}/include")
#include_directories("/opt/intel/mkl/include")
find_package(CUDA REQUIRED)
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})
cuda_add_executable(megasolve main.cpp solver/bicgstab.cu solver/bicgstab.cuh
        solver/ilu.cu
        solver/ilu.cuh
        solve.hpp)
target_link_libraries(megasolve ${CUDA_cusparse_LIBRARY} )
target_link_libraries(megasolve ${CUDA_cublas_LIBRARY} )
target_link_libraries(megasolve ${CUDA_LIBRARIES})
set(CUDA_NVCC_FLAGS -Xcompiler;-fPIC;-lcublas;-lcusparse;-shared)
