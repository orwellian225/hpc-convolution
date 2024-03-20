#include <pnm.hpp>

#include "auxillary.hpp"

PGMRaw pnm_to_raw(pnm::pgm_image &pgm) {
    PGMRaw raw = { 0 };
    raw.height = pgm.height();
    raw.width = pgm.width();
    raw.data = new uint8_t[raw.height * raw.width];

    for (size_t i = 0; i < pgm.width(); ++i) {
        for (size_t j = 0; j < pgm.height(); ++j) {
            raw.data[j * raw.height + i] = pgm[i][j].value;
        }
    }

    return raw;
}

PGMRaw::~PGMRaw() {
    delete[] data;
}