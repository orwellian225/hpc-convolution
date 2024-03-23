#include <fmt/core.h>

#include "matrix.hpp"
#include "serial.hpp"

void serial::convolve(Matrix& image, Matrix& kernel, Matrix& result) {
    for (size_t r = 0; r < image.height; ++r) {
        for (size_t c = 0; c < image.width; ++c) {

            size_t image_idx = r * image.height + c;

            // for width = 3 => kernel_i in [ - (3 / 2), 3 / 2 ] = [ -1, 1 ]
            // for width = 5 => kernel_i in [ - (5 / 2), 5 / 2 ] = [ -2, 2 ]
            // for width = 7 => kernel_i in [ - (7 / 2), 7 / 2 ] = [ -3, 3 ]
            // so this should in theory work
            for (int32_t kernel_r = -(int32_t)(kernel.height / 2); kernel_r <= (int32_t)(kernel.height / 2); ++kernel_r) {
                for (int32_t kernel_c = -(int32_t)(kernel.width / 2); kernel_c <= (int32_t)(kernel.width / 2); ++kernel_c) {

                    int32_t kernel_image_r = r + kernel_r;
                    int32_t kernel_image_c = c + kernel_c;

                    int32_t kernel_idx = (kernel_r + kernel.height / 2) * kernel.height + (kernel_c + kernel.width / 2);
                    int32_t kernel_image_idx = kernel_image_r * image.height + kernel_image_c;

                    if ( -1 < kernel_image_r && kernel_image_r < image.height && -1 < kernel_image_c && kernel_image_c < image.width )
                        result.data[image_idx] += image.data[kernel_image_idx] * kernel.data[kernel_idx];

                }
            }
        }
    }
}