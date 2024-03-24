# COMS4040A HPC - Assignment 1: Convolution

## Dependencies

Tools:
* Git
* Cmake
* Cuda
* C++ compiler with C++20 support

Libraries:
* fmtlib

## Setting up the project

Run `setup.sh`
```bash
./setup.sh
```

## Compiling the project

Run `compile.sh`
```bash
./compile.sh
```

## Running the Project

Kernels:
* 1: Average 5x5
* 2: Sharpen 3x3
* 3: Emboss 5x5

```bash
./convolution "<filepath of input pgm>" "<filepath of ouput directory and file name>" <selected_kernel_num>
```

Example:
```bash
./convolution "./resources/image21.pgm" "./resources/out/image21" 1
```