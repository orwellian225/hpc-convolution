#pragma once

#include <stdint.h>

#include <pnm.hpp>

struct PGMRaw {
    uint32_t width;
    uint32_t height;
    uint8_t *data;

    ~PGMRaw();
};

PGMRaw pnm_to_raw(pnm::pgm_image &pgm);