
#include <cuda_runtime.h>
#include "pnm.hpp"
#include "fmt/core.h"

#include "auxillary.cuh"

void handle_cuda_error(cudaError_t error) {
    if (error == cudaSuccess)
        return;
    
    fmt::println(stderr, "CUDA Error:");
    fmt::println(stderr, "\t{}", cudaGetErrorString(error));
    cudaDeviceReset();
    exit(EXIT_FAILURE);
}