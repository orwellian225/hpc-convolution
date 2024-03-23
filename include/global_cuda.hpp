#pragma once

#include "auxillary.hpp"

namespace global_cuda {
    __global__ void convolve(PGMRaw *image, ConvolveMask *kernel, PGMRaw *result);
}