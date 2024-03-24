#pragma once

#include "matrix.hpp"

namespace sharedmem {
    __global__ void convolve(Matrix *image, Matrix *kernel, Matrix *result);
}
