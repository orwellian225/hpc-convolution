#pragma once

#include <stdint.h>

#include <pnm.hpp>

struct PGMRaw {
    uint32_t width;
    uint32_t height;
    float *data;

    PGMRaw();
    PGMRaw(pnm::pgm_image &pgm);
    PGMRaw(uint32_t width, uint32_t height);
    ~PGMRaw();

    pnm::pgm_image to_pnm();
};


struct ConvolveMask {
    uint32_t width;
    uint32_t height;
    float *data;

    ConvolveMask();
    ConvolveMask(uint32_t width, uint32_t height);
    ~ConvolveMask();
};