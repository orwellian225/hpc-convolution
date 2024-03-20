#include <pnm.hpp>

#include "auxillary.hpp"

PGMRaw::PGMRaw() {
    height = 0;
    width = 0;
    data = nullptr;
}

PGMRaw::PGMRaw(pnm::pgm_image &pgm) {
    height = pgm.height();
    width = pgm.width();
    data = new float[height * width];

    for (size_t i = 0; i < pgm.width(); ++i) {
        for (size_t j = 0; j < pgm.height(); ++j) {
            data[j * height + i] = pgm[i][j].value / 255.;
        }
    }
}

PGMRaw::PGMRaw(uint32_t width, uint32_t height) {
    this->width = width;
    this->height = height;
    data = nullptr;
}

PGMRaw::~PGMRaw() {
    if (data != nullptr)
        delete[] data;
};

pnm::pgm_image PGMRaw::to_pnm() {
    pnm::pgm_image result(width, height);

    for (size_t i = 0; i < result.width(); ++i) {
        for (size_t j = 0; j < result.height(); ++j) {
            result[i][j] = (uint8_t)(data[j * height + i] * 255);
        }
    }

    return result;
}

ConvolveMask::ConvolveMask() {
    width = 0;
    height = 0;
    data = nullptr;
}

ConvolveMask::ConvolveMask(uint32_t width, uint32_t height) {
    this->width = width;
    this->height = height;
    data = nullptr;
}

ConvolveMask::~ConvolveMask() {
    if (data != nullptr)
        delete[] data;
}