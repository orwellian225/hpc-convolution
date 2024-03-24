#include <cuda_runtime.h>
#include <fmt/core.h>

#include "matrix.hpp"
#include "globalmem.hpp"

__global__ void globalmem::convolve(Matrix *image, Matrix *kernel, Matrix *result) {
    size_t image_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (image_idx >= image->size)
        return;

    result->data[image_idx] = 0.;
    size_t r = image_idx / image->width;
    size_t c = image_idx % image->width;

    for (int32_t kernel_r = -(int32_t)(kernel->height / 2); kernel_r <= (int32_t)(kernel->height / 2); ++kernel_r) {
        for (int32_t kernel_c = -(int32_t)(kernel->width / 2); kernel_c <= (int32_t)(kernel->width / 2); ++kernel_c) {

            int32_t kernel_image_r = r + kernel_r;
            int32_t kernel_image_c = c + kernel_c;

            int32_t kernel_idx = (kernel_r + kernel->height / 2) * kernel->height + (kernel_c + kernel->width / 2);
            int32_t kernel_image_idx = kernel_image_r * image->height + kernel_image_c;

            if ( -1 < kernel_image_r && kernel_image_r < image->height && -1 < kernel_image_c && kernel_image_c < image->width )
                result->data[image_idx] += image->data[kernel_image_idx] * kernel->data[kernel_idx];
        }
    }

}