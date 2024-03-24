#include "sharedmem.hpp"
#include "auxillary.hpp"

__global__ void sharedmem::convolve(Matrix *image, Matrix *kernel, Matrix *result) {
    int32_t image_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (image_idx >= image->size)
        return;

    int32_t image_r = image_idx / image->width;
    int32_t image_c = image_idx % image->width;

    extern __shared__ uint8_t arena[];
    float *shared = (float*)(arena);

    int32_t shared_width = blockDim.x + kernel->width - 1;
    int32_t shared_height = kernel->height;

    int32_t copy_r_start = -(int32_t)(kernel->height / 2);
    int32_t copy_r_end = (int32_t)(kernel->height / 2);

    for (int32_t copy_r = copy_r_start; copy_r <= copy_r_end; ++copy_r) {
        int32_t fetch_image_idx = (image_r + copy_r) * image->width + image_c;
        int32_t shared_idx = (copy_r + copy_r_end) * shared_width + (threadIdx.x + kernel->width / 2); // leave columns before current thread column

        if ( -1 < copy_r + image_r && copy_r + image_r < image->height && -1 < image_c && image_c < image->width ) {
            shared[shared_idx] = image->data[fetch_image_idx];
        } else {
            shared[shared_idx] = 0.;
        }
    }

    if (threadIdx.x == 0) {
        for (int32_t copy_c = 1; copy_c <= kernel->width / 2; ++copy_c) {
            for (int32_t copy_r = copy_r_start; copy_r <= copy_r_end; ++copy_r) {
                int32_t fetch_image_idx = (image_r + copy_r) * image->width + image_c - copy_c;
                int32_t shared_idx = (copy_r + copy_r_end) * shared_width + copy_c - 1; // always first columns in array

                if ( -1 < copy_r + image_r && copy_r + image_r < image->height && -1 < image_c - copy_c && image_c - copy_c < image->width ) {
                    shared[shared_idx] = image->data[fetch_image_idx];
                } else {
                    shared[shared_idx] = 0.;
                }
            }
        }
    }

    if (threadIdx.x == image->width - 1) {
        for (int32_t copy_c = 1; copy_c <= kernel->width / 2; ++copy_c) {
            for (int32_t copy_r = copy_r_start; copy_r <= copy_r_end; ++copy_r) {
                int32_t fetch_image_idx = (image_r + copy_r) * image->width + image_c + copy_c;
                int32_t shared_idx = (copy_r + copy_r_end) * shared_width + image_c + copy_c; // always last columns in array

                if ( -1 < copy_r + image_r && copy_r + image_r < image->height && -1 < image_c + copy_c && image_c + copy_c < image->width ) {
                    shared[shared_idx] = image->data[fetch_image_idx];
                } else {
                    shared[shared_idx] = 0.;
                }
            }
        }
    }

    __syncthreads();

    // Get the index in shared memory of our current thread
    int32_t shared_idx = shared_width * (shared_height / 2) + threadIdx.x + kernel->width / 2;
    int32_t shared_r = shared_idx / shared_width;
    int32_t shared_c = shared_idx % shared_width;
    int32_t kernel_r_start = -(int32_t)( kernel->height / 2 ), kernel_r_end = (int32_t)( kernel->height / 2 );
    int32_t kernel_c_start = -(int32_t)( kernel->width / 2 ), kernel_c_end = (int32_t)( kernel->width / 2 );

    float sum = 0.;
    for (int32_t kernel_r = kernel_r_start; kernel_r <= kernel_r_end; ++kernel_r) {
        for (int32_t kernel_c = kernel_c_start; kernel_c <= kernel_c_end; ++kernel_c) {
            int32_t kernel_shared_r = shared_r + kernel_r;
            int32_t kernel_shared_c = shared_c + kernel_c;

            int32_t kernel_idx = (kernel_r + kernel->height / 2) * kernel->height + (kernel_c + kernel->width / 2);
            int32_t kernel_shared_idx = kernel_shared_r * shared_width + kernel_shared_c;

            sum += shared[kernel_shared_idx] * kernel->data[kernel_idx];
        }
    }

    result->data[image_idx] = sum;
}