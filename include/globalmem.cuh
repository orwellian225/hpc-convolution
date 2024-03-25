#pragma once

#include "matrix.hpp"

namespace globalmem {
    __global__ void convolve(Matrix *image, Matrix *kernel, Matrix *result);
}