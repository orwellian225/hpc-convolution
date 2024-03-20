#include <fmt/core.h>

#include <pnm.hpp>

#include "auxillary.hpp"
#include "serial.hpp"

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

    PGMRaw serial_convolved_pgm_raw = serial::convolve(raw_pgm, emboss_kernel);
    pnm::pgm_image serial_convolved_pgm = serial_convolved_pgm_raw.to_pnm();

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