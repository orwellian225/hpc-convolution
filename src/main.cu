#include <fmt/core.h>

#include <pnm.hpp>
#include <cuda_runtime.h>

#include "auxillary.hpp"
#include "serial.hpp"
#include "global_cuda.hpp"

void test_serial_convolution(ConvolveMask &kernel);

__global__ void observe_data(PGMRaw* data) {
    size_t image_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (image_idx > data->height * data->width)
        return;

    data->data[image_idx] = 0.0;
}

int main(int argc, char **argv) {
    fmt::println("COMS4040A High Performance Computing Assignment 1");
    fmt::println("Brendan Griffiths - 2426285");
    fmt::println("Convolution on Portable Gray Map images");
    fmt::println("---------------------------------------");

    if (argc != 3) {
        fmt::println(stderr, "Incorrect arguments - please specify a pgm image to load, and a resultant pgm file");
        return -1;
    }

    // test_serial_convolution();

    std::string pgm_infilepath(argv[1]);
    std::string pgm_outfilepath(argv[2]);


    pnm::pgm_image pgm = pnm::read_pgm_binary(pgm_infilepath);
    PGMRaw raw_pgm(pgm);

    ConvolveMask sharpen_kernel(3,3);
    sharpen_kernel.data = new float[9];
        sharpen_kernel.data[0] = -1; sharpen_kernel.data[1] = -1; sharpen_kernel.data[2] = -1;
        sharpen_kernel.data[3] = -1; sharpen_kernel.data[4] = 9; sharpen_kernel.data[5] = -1;
        sharpen_kernel.data[6] = -1; sharpen_kernel.data[7] = -1; sharpen_kernel.data[8] = -1;

    ConvolveMask average_kernel(5, 5);
    average_kernel.data = new float[25];
    for (size_t i = 0; i < 25; ++i)
        average_kernel.data[i] = 0.04;

    ConvolveMask emboss_kernel(5, 5);
    emboss_kernel.data = new float[25];
    for (size_t i = 0; i < 25; ++i)
        emboss_kernel.data[i] = 0.0;
    emboss_kernel.data[0] = 1.;
    emboss_kernel.data[6] = 1.;
    emboss_kernel.data[18] = -1.;
    emboss_kernel.data[24] = -1.;

    ConvolveMask sum_kernel(5, 5);
    sum_kernel.data = new float[25];
    for (size_t i = 0; i < 25; ++i)
        sum_kernel.data[i] = 1.0;

    PGMRaw serial_convolved_pgm_raw = serial::convolve(raw_pgm, average_kernel);
    pnm::pgm_image serial_convolved_pgm = serial_convolved_pgm_raw.to_pnm();

    float *h_data;

    const size_t num_elements = raw_pgm.height * raw_pgm.width;
    const size_t num_bytes = num_elements * sizeof(float);
    PGMRaw h_global_pgm_raw(raw_pgm.height, raw_pgm.width);
    h_global_pgm_raw.data = new float[num_elements];

    PGMRaw *d_raw_pgm;
    handle_cuda_error(cudaMalloc(&d_raw_pgm, sizeof(PGMRaw)));
    handle_cuda_error(cudaMalloc(&h_data, num_bytes));
    handle_cuda_error(cudaMemcpy(h_data, raw_pgm.data, num_bytes, cudaMemcpyHostToDevice));
    handle_cuda_error(cudaMemcpy(&d_raw_pgm->height, &raw_pgm.height, sizeof(size_t), cudaMemcpyHostToDevice));
    handle_cuda_error(cudaMemcpy(&d_raw_pgm->width, &raw_pgm.width, sizeof(size_t), cudaMemcpyHostToDevice));
    handle_cuda_error(cudaMemcpy(&d_raw_pgm->data, &h_data, sizeof(float*), cudaMemcpyHostToDevice));

    ConvolveMask *d_kernel;
    handle_cuda_error(cudaMalloc(&d_kernel, sizeof(ConvolveMask)));
    handle_cuda_error(cudaMalloc(&h_data, sizeof(float) * average_kernel.width * average_kernel.height));
    handle_cuda_error(cudaMemcpy(h_data, average_kernel.data, sizeof(float) * average_kernel.width * average_kernel.height, cudaMemcpyHostToDevice));
    handle_cuda_error(cudaMemcpy(&d_kernel->height, &average_kernel.height, sizeof(size_t), cudaMemcpyHostToDevice));
    handle_cuda_error(cudaMemcpy(&d_kernel->width, &average_kernel.width, sizeof(size_t), cudaMemcpyHostToDevice));
    handle_cuda_error(cudaMemcpy(&d_kernel->data, &h_data, sizeof(float*), cudaMemcpyHostToDevice));

    PGMRaw *d_gc_convolved_pgm;
    handle_cuda_error(cudaMalloc(&d_gc_convolved_pgm, sizeof(PGMRaw)));
    handle_cuda_error(cudaMalloc(&h_data, num_bytes));
    handle_cuda_error(cudaMemcpy(&d_gc_convolved_pgm->height, &raw_pgm.height, sizeof(size_t), cudaMemcpyHostToDevice));
    handle_cuda_error(cudaMemcpy(&d_gc_convolved_pgm->width, &raw_pgm.width, sizeof(size_t), cudaMemcpyHostToDevice));
    handle_cuda_error(cudaMemcpy(&d_gc_convolved_pgm->data, &h_data, sizeof(float*), cudaMemcpyHostToDevice));

    const size_t block_size = raw_pgm.height * raw_pgm.width < 1024 ? raw_pgm.height * raw_pgm.width : 1024; // number of threads
    const size_t grid_size = raw_pgm.height * raw_pgm.width / block_size + 1;
    global_cuda::convolve<<<grid_size, block_size>>>(d_raw_pgm, d_kernel, d_gc_convolved_pgm);
    cudaDeviceSynchronize();

    float *h_data_ptr;
    handle_cuda_error(cudaMemcpy(&h_data_ptr, &d_gc_convolved_pgm->data, sizeof(float*), cudaMemcpyDeviceToHost));
    handle_cuda_error(cudaMemcpy(h_global_pgm_raw.data, h_data_ptr, num_bytes, cudaMemcpyDeviceToHost));

    pnm::pgm_image global_convolved_pgm = h_global_pgm_raw.to_pnm();

    fmt::println("Input image");
    fmt::println("\tfilepath:\t\t{}", pgm_infilepath);
    fmt::println("\tdimensions:\t\t{} x {}", pgm.width(), pgm.height());
    fmt::println("\tData size (bytes):\t{} ", pgm.width() * pgm.height());
    fmt::println("\tData size (kilobytes):\t{} ", pgm.width() * pgm.height() / 1000.);
    fmt::println("\tData size (kibibytes):\t{} ", pgm.width() * pgm.height() / 1024.);
    fmt::println("---------------------------------------");

    fmt::println("Output image");
    fmt::println("\tfilepath:\t\t{}", pgm_outfilepath);
    fmt::println("\tdimensions:\t\t{} x {}", serial_convolved_pgm.width(), serial_convolved_pgm.height());
    fmt::println("\tData size (bytes):\t{} ", serial_convolved_pgm.width() * pgm.height());
    fmt::println("\tData size (kilobytes):\t{} ", serial_convolved_pgm.width() * serial_convolved_pgm.height() / 1000.);
    fmt::println("\tData size (kibibytes):\t{} ", serial_convolved_pgm.width() * serial_convolved_pgm.height() / 1024.);
    fmt::println("---------------------------------------");

    bool correct_global_mem_impl = true;
    for (size_t i = 0; i < serial_convolved_pgm_raw.height * serial_convolved_pgm_raw.width; ++i) {
        if (abs(serial_convolved_pgm_raw.data[i] - h_global_pgm_raw.data[i]) > 0.001) {
            fmt::println(stderr, "Incorrect GC: {} != {} by 1e-3 @ {}", serial_convolved_pgm_raw.data[i], h_global_pgm_raw.data[i], i);
            correct_global_mem_impl = false;
        }
    }

    fmt::println("Serial implementation");
    fmt::println("---------------------------------------");

    fmt::println("Global Memory CUDA implementation");
    fmt::println("\tCorrect: {}", correct_global_mem_impl ? "Yes" : "No");
    fmt::println("---------------------------------------");

    fmt::println("Shared Memory CUDA implementation");
    fmt::println("---------------------------------------");


    pnm::write_pgm_binary(fmt::format("{}_serial.pgm", pgm_outfilepath), serial_convolved_pgm);
    pnm::write_pgm_binary(fmt::format("{}_global.pgm", pgm_outfilepath), global_convolved_pgm);

    return 0;
}

void test_serial_convolution(ConvolveMask &kernel) {
    PGMRaw test_image(4, 4);
    test_image.data = new float[16];
        test_image.data[0] = 1.; test_image.data[1] = 1.; test_image.data[2] = 1.; test_image.data[3] = 1.;
        test_image.data[4] = 1.; test_image.data[5] = 1.; test_image.data[6] = 1.; test_image.data[7] = 1.;
        test_image.data[8] = 1.; test_image.data[9] = 1.; test_image.data[10] = 1.; test_image.data[11] = 1.;
        test_image.data[12] = 1.; test_image.data[13] = 1.; test_image.data[14] = 1.; test_image.data[15] = 1.;

    // ConvolveMask test_kernel(3,3);
    // test_kernel.data = new int32_t[9];
    //     test_kernel.data[0] = 1; test_kernel.data[1] = 1; test_kernel.data[2] = 1;
    //     test_kernel.data[3] = 1; test_kernel.data[4] = 1; test_kernel.data[5] = 1;
    //     test_kernel.data[6] = 1; test_kernel.data[7] = 1; test_kernel.data[8] = 1;

    PGMRaw test_convolved_image = serial::convolve(test_image, kernel);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            fmt::print("{:.2f} ({}) ", test_convolved_image.data[j * 4 + i], (uint8_t)(test_convolved_image.data[j * 4 + i] * 255));
        }
        fmt::print("\n");
    }
}