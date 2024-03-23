#pragma once

#include <stdint.h>

#include "pnm.hpp"

struct Matrix {
    uint32_t width;
    uint32_t height;
    uint32_t size;
    float *data;

    Matrix();
    Matrix(uint32_t width, uint32_t height, float initial_value);
    Matrix(pnm::pgm_image &pgm);

    pnm::pgm_image to_pnm();

    bool equals(Matrix& other, float margin);

    void print();

    static Matrix to_host(Matrix *device_matrix);
    static Matrix* to_device(Matrix &host_matrix);

    void operator=(const Matrix& other);

    ~Matrix();
};