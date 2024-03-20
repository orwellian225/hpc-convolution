#include <fmt/core.h>

#include "serial.hpp"

PGMRaw serial::convolve(PGMRaw& image, ConvolveMask& kernel) {
    PGMRaw result;
    result.width = image.width;
    result.height = image.height;
    result.data = new float[result.width * result.height];

    for (size_t i = 0; i < result.width; ++i) {
        for (size_t j = 0; j < result.height; ++j) {

            size_t image_idx = j * result.height + i;
            float sum = 0.;

            // for width = 3 => kernel_i in [ - (3 / 2), 3 / 2 ] = [ -1, 1 ]
            // for width = 5 => kernel_i in [ - (5 / 2), 5 / 2 ] = [ -2, 2 ]
            // for width = 7 => kernel_i in [ - (7 / 2), 7 / 2 ] = [ -3, 3 ]
            // so this should in theory work
            for (int64_t kernel_i = - (int64_t)(kernel.width / 2); kernel_i <= (int64_t)(kernel.width / 2); ++kernel_i) {
                for (int64_t kernel_j = - (int64_t)(kernel.height / 2); kernel_j <=  (int64_t)(kernel.height / 2); ++kernel_j) {

                    int32_t kernel_image_i = i + kernel_i, kernel_image_j = j + kernel_j;
                    size_t kernel_image_idx = kernel_image_j * result.height + kernel_image_i;
                    size_t kernel_idx = (kernel_j + kernel.height / 2) * kernel.height + (kernel_i + kernel.width / 2);
                    float image_val;


                    if ( -1 < kernel_image_i && kernel_image_i < image.width && -1 < kernel_image_j && kernel_image_j < image.height )
                        image_val = image.data[kernel_image_idx];
                    else
                        image_val = 0;

                    sum += (image_val * kernel.data[kernel_idx]);
                }
            }

            result.data[image_idx] = sum;
        }
    }

    return result;
}