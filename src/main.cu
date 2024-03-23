#include <fmt/core.h>

#include <pnm.hpp>
#include <cuda_runtime.h>

#include "auxillary.hpp"
#include "serial.hpp"
#include "global_cuda.hpp"

void test_serial_convolution(ConvolveMask &kernel);

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

    // test_serial_convolution(average_kernel);

    fmt::println("First value: {}", raw_pgm.data[516]);
    PGMRaw serial_convolved_pgm_raw = serial::convolve(raw_pgm, average_kernel);
    pnm::pgm_image serial_convolved_pgm = serial_convolved_pgm_raw.to_pnm();

    PGMRaw *device_raw_pgm;
    float *temp_data;
    cudaMalloc(&device_raw_pgm, sizeof(PGMRaw));
    cudaMalloc(&temp_data, sizeof(float) * raw_pgm.height * raw_pgm.width);
    cudaMemcpy(&device_raw_pgm->data, &temp_data, sizeof(float) * raw_pgm.height * raw_pgm.width, cudaMemcpyHostToDevice);
    cudaMemcpy(&device_raw_pgm->height, &raw_pgm.height, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_raw_pgm->width, &raw_pgm.width, sizeof(size_t), cudaMemcpyHostToDevice);

    PGMRaw host_gc_convolved_pgm_raw, *device_gc_convolved_pgm_raw;
    host_gc_convolved_pgm_raw.height = raw_pgm.height;
    host_gc_convolved_pgm_raw.width = raw_pgm.width;
    host_gc_convolved_pgm_raw.data = new float[raw_pgm.height * raw_pgm.width];

    ConvolveMask *device_kernel;
    cudaMalloc(&device_kernel, sizeof(ConvolveMask));
    cudaMemcpy(&device_kernel->height, &average_kernel.height, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_kernel->width, &average_kernel.width, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_kernel->data, &average_kernel.data, sizeof(float) * average_kernel.height * average_kernel.width, cudaMemcpyHostToDevice);

    cudaMalloc(&device_gc_convolved_pgm_raw, sizeof(PGMRaw));
    cudaMemcpy(&device_gc_convolved_pgm_raw->height, &raw_pgm.height, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_gc_convolved_pgm_raw->height, &raw_pgm.height, sizeof(size_t), cudaMemcpyHostToDevice);

    // const size_t block_size = 1024; // number of threads
    // const size_t grid_size = raw_pgm.height * raw_pgm.width / block_size + 1;
    const size_t block_size = 1;
    const size_t grid_size = 1;
    global_cuda::convolve<<<grid_size, block_size>>>(device_raw_pgm, device_kernel, device_gc_convolved_pgm_raw);
    cudaDeviceSynchronize();
    cudaMemcpy(&host_gc_convolved_pgm_raw.data, &device_gc_convolved_pgm_raw->data, sizeof(float) * raw_pgm.height * raw_pgm.width, cudaMemcpyDeviceToHost);

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

    bool incorrect_global_cuda_flag = false;
    for (size_t i = 0; i < serial_convolved_pgm_raw.height * serial_convolved_pgm_raw.width; ++i) {
        if (serial_convolved_pgm_raw.data[i] != host_gc_convolved_pgm_raw.data[i]) {
            fmt::println(stderr, "Incorrect global impl: {} != {} @ {}", serial_convolved_pgm_raw.data[i], host_gc_convolved_pgm_raw.data[i], i);
            incorrect_global_cuda_flag = true;
            break;
        }
    }

    fmt::println("Serial implementation");
    fmt::println("---------------------------------------");

    fmt::println("Global Memory CUDA implementation");
    fmt::println("\tCorrect: {}", incorrect_global_cuda_flag ? "No" : "Yes");
    fmt::println("---------------------------------------");

    fmt::println("Shared Memory CUDA implementation");
    fmt::println("---------------------------------------");

    pnm::write_pgm_binary(pgm_outfilepath, serial_convolved_pgm);

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