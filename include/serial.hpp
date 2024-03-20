#pragma once

#include "auxillary.hpp"

namespace serial {

    /** 
     * convolve:
     *      input:
     *          PGMRaw image
     *          ConvolveMask kernal
     *      output:
     *          PGMRaw convolved_image -- the pgm will have smaller 
     */

    PGMRaw convolve(PGMRaw& image, ConvolveMask& kernel);

}