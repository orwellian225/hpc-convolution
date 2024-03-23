#include <chrono>

#include <fmt/core.h>
#include <pnm.hpp>
#include <cuda_runtime.h>

#include "auxillary.hpp"
#include "matrix.hpp"
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
    fmt::println("{:-<80}", "-");

    if (argc != 3) {
        fmt::println(stderr, "Incorrect arguments - please specify a pgm image to load, and a resultant pgm file");
        return -1;
    }

    std::string pgm_infilepath(argv[1]);
    std::string pgm_outfilepath(argv[2]);

    pnm::pgm_image pgm = pnm::read_pgm_binary(pgm_infilepath);

    std::string serial_pgm_out_filepath = fmt::format("{}_serial.pgm", pgm_outfilepath);
    std::string global_pgm_out_filepath = fmt::format("{}_global.pgm", pgm_outfilepath);
    std::string shared_pgm_out_filepath = fmt::format("{}_shared.pgm", pgm_outfilepath);

    const size_t num_elements = pgm.height() * pgm.width();
    const uint32_t img_width = pgm.width(), img_height = pgm.height(); 
    const size_t num_image_bytes = num_elements * sizeof(uint8_t); // every pixel is u8 grayscale so only 1 byte
    const size_t num_float_bytes = num_elements * sizeof(float);

    fmt::println("Image Properties");
    fmt::println("\t{:<32} {}", "Input Filepath:", pgm_infilepath);
    fmt::println("\t{:<32} {}", "Serial output filepath:", serial_pgm_out_filepath);
    fmt::println("\t{:<32} {}", "Global output filepath:", global_pgm_out_filepath);
    fmt::println("\t{:<32} {}", "Shared output filepath:", shared_pgm_out_filepath);
    fmt::println("\t{:<32} {} x {}", "Dimensions:", img_width, img_height);
    fmt::println("\t{:<32} {:<10}", "Image data size (bytes):", num_image_bytes);
    fmt::println("\t\t{:<24} {:<10.3f}", "(kilobytes / kB):", num_image_bytes / 1000.0);
    fmt::println("\t\t{:<24} {:<10.3f}", "(kibibytes / kiB):", num_image_bytes / 1024.0);
    fmt::println("\t{:<32} {:<10}", "Matrix data size (bytes):", num_float_bytes);
    fmt::println("\t\t{:<24} {:<10.3f}", "(kilobytes / kB):", num_float_bytes / 1000.0);
    fmt::println("\t\t{:<24} {:<10.3f}", "(kibibytes / kiB):", num_float_bytes / 1024.0);
    fmt::println("");

    Matrix image_matrix(pgm);
    Matrix serial_convolved_matrix(img_width, img_height, 0.);
    Matrix globalmem_convolved_matrix(img_width, img_height, 0.);
    Matrix sharedmem_convolved_matrix(img_width, img_height, 0.);

    Matrix kernel(5, 5, 0.04);

    const size_t block_size = 1024;
    const size_t grid_size = num_elements / block_size + 1;
    float serial_duration_ms = 0., globalmem_duration_ms = 0., sharedmem_duration_ms = 0.;

    auto serial_start = std::chrono::high_resolution_clock::now();
        serial::convolve(image_matrix, kernel, serial_convolved_matrix);
    auto serial_end = std::chrono::high_resolution_clock::now();
    serial_duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(serial_end - serial_start).count() / 1000.;

    Matrix *d_image_matrix = Matrix::to_device(image_matrix);
    Matrix *d_kernel = Matrix::to_device(kernel);
    Matrix *d_globalmem_matrix = Matrix::to_device(globalmem_convolved_matrix);
    Matrix *d_sharedmem_matrix = Matrix::to_device(sharedmem_convolved_matrix);

    cudaEvent_t globalmem_start, globalmem_end;    
    cudaEventCreate(&globalmem_start);
    cudaEventCreate(&globalmem_end);
    cudaEventRecord(globalmem_start, 0);
        global_cuda::convolve<<<grid_size, block_size>>>(d_image_matrix, d_kernel, d_globalmem_matrix);
    cudaEventRecord(globalmem_end, 0);
    cudaEventSynchronize(globalmem_end);
    globalmem_convolved_matrix = Matrix::to_host(d_globalmem_matrix);
    cudaEventElapsedTime(&globalmem_duration_ms, globalmem_start, globalmem_end);

    cudaEvent_t sharedmem_start, sharedmem_end;    
    cudaEventCreate(&sharedmem_start);
    cudaEventCreate(&sharedmem_end);

    bool globalmem_correct = serial_convolved_matrix.equals(globalmem_convolved_matrix, 0.001);
    bool sharedmem_correct = serial_convolved_matrix.equals(sharedmem_convolved_matrix, 0.001);

    fmt::println("{:-<178}", "-");
    fmt::println("| {:<24} | {:<19} | {:<24} | {:<23} | {:<25} | {:<24} | {:<7} | {:<7} |", 
        "Algorithm", 
        "Execution Time (ms)", 
        "Image Throughput (kiB/s)", 
        "Image Throughput (kB/s)", 
        "Matrix Throughput (kiB/s)",
        "Matrix Throughput (kB/s)",
        "Speedup",
        "Correct"
    );
    fmt::println("{:-<178}", "-");

    fmt::println("| {:<24} | {:^19.5f} | {:^24.2f} | {:^23.2f} | {:^25.2f} | {:^24.2f} | {:^7} | {:^7} |", 
        "Serial", 
        serial_duration_ms,
        (num_image_bytes / 1024.) / (serial_duration_ms / 1000.),
        (num_image_bytes / 1000.) / (serial_duration_ms / 1000.),
        (num_float_bytes / 1024.) / (serial_duration_ms / 1000.),
        (num_float_bytes / 1000.) / (serial_duration_ms / 1000.),
        "/",
        "/"
    );

    fmt::println("| {:<24} | {:^19.5f} | {:^24.2f} | {:^23.2f} | {:^25.2f} | {:^24.2f} | {:^7.2f} | {:^7} |", 
        "Global Memory CUDA", 
        globalmem_duration_ms,
        (num_image_bytes / 1024.) / (globalmem_duration_ms / 1000.),
        (num_image_bytes / 1000.) / (globalmem_duration_ms / 1000.),
        (num_float_bytes / 1024.) / (globalmem_duration_ms / 1000.),
        (num_float_bytes / 1000.) / (globalmem_duration_ms / 1000.),
        serial_duration_ms / globalmem_duration_ms,
        globalmem_correct ? "Yes" : "No"
    );
    fmt::println("| {:<24} | {:^19.5f} | {:^24.2f} | {:^23.2f} | {:^25.2f} | {:^24.2f} | {:^7.2f} | {:^7} |", 
        "Shared Memory CUDA", 
        sharedmem_duration_ms,
        (num_image_bytes / 1024.) / (sharedmem_duration_ms / 1000.),
        (num_image_bytes / 1000.) / (sharedmem_duration_ms / 1000.),
        (num_float_bytes / 1024.) / (sharedmem_duration_ms / 1000.),
        (num_float_bytes / 1000.) / (sharedmem_duration_ms / 1000.),
        serial_duration_ms / sharedmem_duration_ms,
        sharedmem_correct ? "Yes" : "No"
    );
    fmt::println("{:-<178}", "-");

    pnm::pgm_image serial_pgm = serial_convolved_matrix.to_pnm();
    pnm::pgm_image globalmem_pgm = globalmem_convolved_matrix.to_pnm();

    pnm::write_pgm_binary(serial_pgm_out_filepath, serial_pgm);
    pnm::write_pgm_binary(global_pgm_out_filepath, globalmem_pgm);

    return 0;
}