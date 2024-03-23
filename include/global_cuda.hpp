#pragma once

#include "auxillary.hpp"

namespace global_cuda {
    __global__ void convolve(Matrix *image, Matrix *kernel, Matrix *result);
}