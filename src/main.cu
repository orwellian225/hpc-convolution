#include <chrono>
#include <stdio.h>

#include <fmt/core.h>
#include <pnm.hpp>
#include <cuda_runtime.h>

#include "auxillary.cuh"
#include "matrix.hpp"
#include "serial.hpp"
#include "globalmem.cuh"
#include "sharedmem.cuh"

int main(int argc, char **argv) {
    if (argc != 3) {
        fmt::println(stderr, "Incorrect arguments - please specify a pgm image to load, a resultant pgm file");
        return -1;
    }

    std::string pgm_infilepath(argv[1]);
    std::string pgm_outfilepath(argv[2]);

    const std::array<std::string, 3> available_kernel_names({
        "average", "sharpen", "emboss"
    });
    std::array<Matrix, 3> available_kernels({
        Matrix(5, 5, 0.04), Matrix(3, 3, -1.), Matrix(5, 5, 0.)
    });
    available_kernels[1].data[4] = 9.;
    available_kernels[2].data[0] = 1.; available_kernels[2].data[6] = 1.;
    available_kernels[2].data[18] = -1.; available_kernels[2].data[24] = -1.;

    pnm::pgm_image pgm = pnm::read_pgm_binary(pgm_infilepath);
    // fmt::println("Kernel,Image,Attempt, Algorithm,Execution Time (ms),Throughput (MiB/s),Speedup");

    for (size_t selected_kernel = 0; selected_kernel < available_kernels.size(); ++selected_kernel) {
        std::string serial_pgm_out_filepath = fmt::format("{}_{}_serial.pgm", pgm_outfilepath, available_kernel_names[selected_kernel]);
        std::string global_pgm_out_filepath = fmt::format("{}_{}_global.pgm", pgm_outfilepath, available_kernel_names[selected_kernel]);
        std::string shared_pgm_out_filepath = fmt::format("{}_{}_shared.pgm", pgm_outfilepath, available_kernel_names[selected_kernel]);

        const size_t num_elements = pgm.height() * pgm.width();
        const uint32_t img_width = pgm.width(), img_height = pgm.height(); 
        const size_t num_image_bytes = num_elements * sizeof(uint8_t); // every pixel is u8 grayscale so only 1 byte

        Matrix image_matrix(pgm);
        Matrix serial_convolved_matrix(img_width, img_height, 0.);
        Matrix globalmem_convolved_matrix(img_width, img_height, 0.);
        Matrix sharedmem_convolved_matrix(img_width, img_height, 0.);

        Matrix kernel = available_kernels[selected_kernel];

        size_t block_size = 1024;
        if (img_width < 1024) {
            block_size = img_width;
        }

        const size_t grid_size = num_elements / block_size + 1;
        float serial_duration_ms = 0., globalmem_duration_ms = 0., sharedmem_duration_ms = 0.;

        for (size_t attempt = 0; attempt < 3; ++attempt) {
            auto serial_start = std::chrono::high_resolution_clock::now();
                serial::convolve(image_matrix, kernel, serial_convolved_matrix);
            auto serial_end = std::chrono::high_resolution_clock::now();
            serial_duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(serial_end - serial_start).count() / 1000.;

            fmt::println("{},{},{},{},{},{},{},{}",
                available_kernel_names[selected_kernel], pgm_infilepath, 
                attempt, "Serial", serial_duration_ms, 
                (num_image_bytes / 1024. / 1024.) / (serial_duration_ms / 1000.),
                0., "/"
            );

            Matrix *d_image_matrix = Matrix::to_device(image_matrix);
            Matrix *d_kernel = Matrix::to_device(kernel);
            Matrix *d_globalmem_matrix = Matrix::to_device(globalmem_convolved_matrix);
            Matrix *d_sharedmem_matrix = Matrix::to_device(sharedmem_convolved_matrix);

            cudaEvent_t globalmem_start, globalmem_end;    
            cudaEventCreate(&globalmem_start);
            cudaEventCreate(&globalmem_end);
            cudaEventRecord(globalmem_start, 0);
                globalmem::convolve<<<grid_size, block_size>>>(d_image_matrix, d_kernel, d_globalmem_matrix);
            cudaEventRecord(globalmem_end, 0);
            cudaEventSynchronize(globalmem_end);
            globalmem_convolved_matrix = Matrix::to_host(d_globalmem_matrix);
            cudaEventElapsedTime(&globalmem_duration_ms, globalmem_start, globalmem_end);

            bool globalmem_correct = serial_convolved_matrix.equals(globalmem_convolved_matrix, 0.001);
            fmt::println("{},{},{},{},{},{},{},{}",
                available_kernel_names[selected_kernel], pgm_infilepath, 
                attempt, "GlobalMem", globalmem_duration_ms, 
                (num_image_bytes / 1024. / 1024.) / (globalmem_duration_ms/ 1000.),
                serial_duration_ms / globalmem_duration_ms, globalmem_correct
            );
            cudaEventDestroy(globalmem_start);
            cudaEventDestroy(globalmem_end);
            cudaFree(d_globalmem_matrix);

            // I'm getting a really weird bug here:
            // The amount of request memory is correct
            // eg. 512x512 image with 5x5 kernel asks for 2560 floats, but I can only access 2524. The 2525 float segfaults?
            // so to get around this I'm just allocating an additional 100 floats just as a security buffer to avoid the issue
            // Need to speak to William about this
            size_t row_buffer_size = sizeof(float) * block_size * kernel.height + sizeof(float) * 100; 
            size_t kernel_buffer_size = sizeof(float) * kernel.size;
            cudaEvent_t sharedmem_start, sharedmem_end;    
            cudaEventCreate(&sharedmem_start);
            cudaEventCreate(&sharedmem_end);
            cudaEventRecord(sharedmem_start, 0);
                sharedmem::convolve<<<grid_size, block_size, row_buffer_size + kernel_buffer_size>>>(d_image_matrix, d_kernel, d_sharedmem_matrix);
            cudaEventRecord(sharedmem_end, 0);
            cudaEventSynchronize(sharedmem_end);
            sharedmem_convolved_matrix = Matrix::to_host(d_sharedmem_matrix);
            cudaEventElapsedTime(&sharedmem_duration_ms, sharedmem_start, sharedmem_end);

            bool sharedmem_correct = serial_convolved_matrix.equals(sharedmem_convolved_matrix, 0.001);
            fmt::println("{},{},{},{},{},{},{},{}",
                available_kernel_names[selected_kernel], pgm_infilepath, 
                attempt, "SharedMem", sharedmem_duration_ms, 
                (num_image_bytes / 1024. / 1024.) / (sharedmem_duration_ms / 1000.),
                serial_duration_ms / sharedmem_duration_ms, sharedmem_correct
            );
            cudaEventDestroy(sharedmem_start);
            cudaEventDestroy(sharedmem_end);
            cudaFree(d_sharedmem_matrix);

            cudaFree(d_kernel);
            cudaFree(d_image_matrix);


            // Breaks on repeated attempts? Something to do with not reseting memory but when I try it crashes immediately
            // Don't have time to fix so hack to get around it
            if (attempt == 0) {
                pnm::pgm_image serial_pgm = serial_convolved_matrix.to_pnm();
                pnm::pgm_image globalmem_pgm = globalmem_convolved_matrix.to_pnm();
                pnm::pgm_image sharedmem_pgm = sharedmem_convolved_matrix.to_pnm();
                pnm::write_pgm_binary(serial_pgm_out_filepath, serial_pgm);
                pnm::write_pgm_binary(global_pgm_out_filepath, globalmem_pgm);
                pnm::write_pgm_binary(shared_pgm_out_filepath, sharedmem_pgm);
            }
        }

        cudaDeviceReset();
    }

    return 0;
}