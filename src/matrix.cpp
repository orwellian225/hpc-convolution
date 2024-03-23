#include <stdint.h>
#include <limits>

#include <cuda_runtime.h>

#include "matrix.hpp"
#include "pnm.hpp"
#include "auxillary.hpp"


Matrix::Matrix() {
    width = 0;
    height = 0;
    size = 0;
    data = nullptr;
}

Matrix::Matrix(uint32_t width, uint32_t height, float initial_value) {
    this->width = width;
    this->height = height;
    size = width * height;

    data = new float[size];
    for (size_t i = 0; i < size; ++i)
        data[i] = initial_value;
}

Matrix::Matrix(pnm::pgm_image &pgm) {
    height = pgm.height();
    width = pgm.width();
    size = width * height;
    data = new float[size];

    for (size_t i = 0; i < pgm.width(); ++i) {
        for (size_t j = 0; j < pgm.height(); ++j) {
            data[j * height + i] = pgm[i][j].value / 255.;
        }
    }
}

pnm::pgm_image Matrix::to_pnm() {
    pnm::pgm_image result(width, height);

    float max = std::numeric_limits<float>::lowest(), min = std::numeric_limits<float>::max();
    for (size_t i = 0; i < size; ++i) {
        if (data[i] > max) {
            max = data[i];
        } 

        if (data[i] < min) {
            min = data[i];
        }
    }

    // if the range is in [0, 1], then don't normalize
    if (max <= 1) {
        max = 1.;
    }
    if (min >= 0.) {
        min = 0.;
    }

    for (size_t i = 0; i < result.width(); ++i) {
        for (size_t j = 0; j < result.height(); ++j) {
            result[i][j] = (uint8_t)(
                // (data[j * height + i] * 255)
                (data[j * height + i] - min) / (max - min) * 255
            );
        }
    }

    return result;
}

void Matrix::print() {
    for (size_t i = 0; i < width; ++i) {
        for (size_t j = 0; j < height; ++j) {
            fmt::print("{:.2f} ", data[j * height + i]);
        }
        fmt::println("");
    }
}

bool Matrix::equals(Matrix& other, float margin) {
    float other_value, this_value;
    for (size_t i = 0; i < other.size; ++i) {
        other_value = other.data[i];
        this_value = this->data[i];

        if ( abs(other_value - this_value) > margin )
            return false;
    }

    return true;
}

Matrix Matrix::to_host(Matrix *device_matrix) {
    Matrix host_matrix;
    float *h_data;

    handle_cuda_error(cudaMemcpy(
        &host_matrix.width, &device_matrix->width,
        sizeof(uint32_t), cudaMemcpyDeviceToHost
    ));

    handle_cuda_error(cudaMemcpy(
        &host_matrix.height, &device_matrix->height,
        sizeof(uint32_t), cudaMemcpyDeviceToHost
    ));

    handle_cuda_error(cudaMemcpy(
        &host_matrix.size, &device_matrix->size,
        sizeof(uint32_t), cudaMemcpyDeviceToHost
    ));

    handle_cuda_error(cudaMemcpy(
        &h_data, &device_matrix->data,
        sizeof(float*), cudaMemcpyDeviceToHost
    ));

    host_matrix.data = new float[host_matrix.size];
    handle_cuda_error(cudaMemcpy(
        host_matrix.data, h_data,
        host_matrix.size * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    return host_matrix;
}

Matrix* Matrix::to_device(Matrix &host_matrix) {
    Matrix *device_matrix;
    float *h_data;

    handle_cuda_error(cudaMalloc(&device_matrix, sizeof(Matrix)));
    handle_cuda_error(cudaMalloc(&h_data, host_matrix.size * sizeof(float)));

    handle_cuda_error(cudaMemcpy(
        &device_matrix->width, &host_matrix.width,
        sizeof(uint32_t), cudaMemcpyHostToDevice
    ));

    handle_cuda_error(cudaMemcpy(
        &device_matrix->height, &host_matrix.height,
        sizeof(uint32_t), cudaMemcpyHostToDevice
    ));

    handle_cuda_error(cudaMemcpy(
        &device_matrix->size, &host_matrix.size,
        sizeof(uint32_t), cudaMemcpyHostToDevice
    ));

    handle_cuda_error(cudaMemcpy(
        h_data, host_matrix.data,
        host_matrix.size * sizeof(float), cudaMemcpyHostToDevice
    ));

    handle_cuda_error(cudaMemcpy(
        &device_matrix->data, &h_data,
        sizeof(float*), cudaMemcpyHostToDevice
    ));

    return device_matrix;
}

void Matrix::operator=(const Matrix& other) {
    width = other.width;
    height = other.height;
    size = other.size;
    data = other.data;
}

Matrix::~Matrix() {}