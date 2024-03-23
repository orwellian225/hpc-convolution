#include "global_cuda.hpp"

#include <cuda_runtime.h>

#include <fmt/core.h>

__global__ void global_cuda::convolve(PGMRaw *image, ConvolveMask *kernel, PGMRaw *result) {
    size_t image_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (image_idx >= image->width * image->height)
        return;


    size_t i = 516 / image->width;
    size_t j = 516 % image->width;
    float sum = 0.;

    printf("image idx: %d\n", image_idx);
    printf("GC: Height = %d | Width = %d | size = %d \n", image->height, image->width, sizeof(image->data));
    printf("GC: height add = %p | width add = %p \n", &image->height, &image->width);

    // for (int64_t kernel_i = - (int64_t)(kernel->width / 2); kernel_i <= (int64_t)(kernel->width / 2); ++kernel_i) {
    //     for (int64_t kernel_j = - (int64_t)(kernel->height / 2); kernel_j <=  (int64_t)(kernel->height / 2); ++kernel_j) {

    //         int32_t kernel_image_i = i + kernel_i, kernel_image_j = j + kernel_j;
    //         size_t kernel_image_idx = kernel_image_j * result->height + kernel_image_i;
    //         size_t kernel_idx = (kernel_j + kernel->height / 2) * kernel->height + (kernel_i + kernel->width / 2);
    //         float image_val;


    //         if ( -1 < kernel_image_i && kernel_image_i < image->width && -1 < kernel_image_j && kernel_image_j < image->height )
    //             image_val = image->data[kernel_image_idx];
    //         else
    //             image_val = 0;

    //         sum += (image_val * kernel->data[kernel_idx]);
    //     }
    // }
}