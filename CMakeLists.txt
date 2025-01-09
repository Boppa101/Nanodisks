cmake_minimum_required(VERSION 3.22.1)
project(CCode C)

set(CMAKE_C_STANDARD 99) #11

#include_directories("/opt/intel/oneapi/mkl/latest/include")
#link_directories("/opt/intel/oneapi/mkl/latest/lib")
set(CMAKE_PREFIX_PATH "/opt/intel/oneapi/mkl/latest/lib/cmake")
find_package(MKL REQUIRED)

add_executable(CCode main.c
        Constants.c
        Constants.h
        Functions.c
        Functions.h)

add_executable(Tests Tests.c
        Constants.c
        Constants.h
        Functions.c
        Functions.h)

target_link_libraries(CCode m gsl gslcblas MKL::MKL)
target_link_libraries(Tests m gsl gslcblas MKL::MKL)

#target_include_directories(CCode PRIVATE "/opt/intel/oneapi/mkl/latest/include")
#target_include_directories(Tests PRIVATE "/opt/intel/oneapi/mkl/latest/include")