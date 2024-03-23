cmake_minimum_required(VERSION 3.16)

project(HPC-Convolution DESCRIPTION "COMS4040A HPC Assignment 1: Convolution" LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(lib/fmt)

add_executable(convolution 
    src/main.cu src/auxillary.cpp 
    src/matrix.cpp
    src/serial.cpp src/global_cuda.cu
)

target_include_directories(convolution PUBLIC include PUBLIC lib/fmt/include)
target_link_directories(convolution PRIVATE src PRIVATE lib/fmt/src)
target_link_libraries(convolution fmt)
target_compile_options(convolution PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
set_target_properties(convolution PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)