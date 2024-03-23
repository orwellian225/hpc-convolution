#include <stdint.h>

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
        data[i] = 0.;
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

    for (size_t i = 0; i < result.width(); ++i) {
        for (size_t j = 0; j < result.height(); ++j) {
            result[i][j] = (uint8_t)(data[j * height + i] * 255);
        }
    }

    return result;
}

bool Matrix::equals(Matrix& other, float margin) {
    for (size_t i = 0; i < other.size; ++i) {
        if ( abs(other.data[i] - data[i]) > margin )
            return false;
    }

    return true;
}

static Matrix to_host(Matrix *device_matrix) {
    Matrix host_matrix;
    float *h_data;

    handle_cuda_error(cudaMemcpy(
        &h_data, &device_matrix->data,
        sizeof(float*), cudaMemcpyDeviceToHost
    ));

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
        host_matrix.data, h_data,
        device_matrix->size * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    return host_matrix;
}

static Matrix *to_device(Matrix &host_matrix) {
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
        &device_matrix->data, &h_data,
        sizeof(float*), cudaMemcpyHostToDevice
    ));

    return device_matrix;
}

Matrix::~Matrix() {
    if (data != nullptr)
        delete[] data;
}