#include "sharedmem.cuh"
#include "auxillary.cuh"

__global__ void sharedmem::convolve(Matrix *image, Matrix *kernel, Matrix *result) {
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float arena[];
    float *kernel_buffer = arena;
    float *row_buffer = arena + sizeof(float) * kernel->size;

    // Loading kernel into shared memory
    if (threadIdx.x < kernel->size) {
        kernel_buffer[threadIdx.x] = kernel->data[threadIdx.x];
    }

    int32_t image_r = thread_idx / image->width;
    int32_t image_c = threadIdx.x;
    int32_t image_height = image->height;
    int32_t image_width = image->width;

    int32_t kernel_width = kernel->width;
    int32_t rb_width = blockDim.x;

    int32_t kernel_center = (int32_t)(kernel->height / 2);
    for (int32_t copy_r = -kernel_center; copy_r <= kernel_center; ++copy_r) {
        if (-1 < copy_r + image_r && copy_r + image_r < image->height) {
            row_buffer[(copy_r + kernel_center) * rb_width + image_c] = image->data[(image_r + copy_r) * image_width + image_c];
        }
    }

    __syncthreads();

    if (thread_idx >= image->size)
        return;

    float sum = 0.;
    for (int32_t kernel_r = -kernel_center; kernel_r <= kernel_center; ++kernel_r) {
        for (int32_t kernel_c = -kernel_center; kernel_c <= kernel_center; ++kernel_c) {

            if (-1 < image_c + kernel_c && image_c + kernel_c < rb_width && -1 < image_r + kernel_r && image_r + kernel_r < image_height) {
                sum += kernel_buffer[(kernel_r + kernel_center) * kernel_width + kernel_c + kernel_center] * row_buffer[(kernel_r + kernel_center) * rb_width + image_c + kernel_c];
            } 
        }
    }

    result->data[thread_idx] = sum;
}