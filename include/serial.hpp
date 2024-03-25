#pragma once

#include "matrix.hpp"

namespace serial {
    void convolve(Matrix& image, Matrix& kernel, Matrix& result);
}