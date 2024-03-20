cmake_minimum_required(VERSION 3.16)

project(HPC-Convolution DESCRIPTION "COMS4040A HPC Assignment 1: Convolution" LANGUAGES CXX CUDA)

add_executable(convolution src/main.cpp)
target_include_directories(convolution PUBLIC include)
set_target_properties(convolution PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)