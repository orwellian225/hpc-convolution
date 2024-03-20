#include <fmt/core.h>

#include <pnm.hpp>

#include "auxillary.hpp"

int main(int argc, char **argv) {
    if (argc != 2) {
        fmt::println(stderr, "Incorrect arguments - please specify a loaded image");
        return -1;
    }

    std::string pgm_filepath(argv[1]);
    pnm::pgm_image pgm = pnm::read_pgm_binary(pgm_filepath);
    PGMRaw raw_pgm = pnm_to_raw(pgm);

    for (size_t i = 0; i < pgm.width(); ++i) {
        for (size_t j = 0; j < pgm.height(); ++j) {
            if (raw_pgm.data[j * raw_pgm.height + i] != pgm[i][j].value) {
                fmt::println(
                    stderr, 
                    "Incorrectly converted pnm_pgm to PGMRaw ({}) == ({}) @ [{}] == [{}, {}]", 
                    raw_pgm.data[j * raw_pgm.height + i],
                    pgm[i][j].value,
                    j * raw_pgm.height + i, 
                    i, j
                );
                return -1;
            }
        }
    }

    return 0;
}