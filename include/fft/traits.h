#ifndef _FFTTRAITS_CLASS_
#define _FFTTRAITS_CLASS_

#include <fftw3-mpi.h>

template<typename T>
struct FFTWTraits;

template<>
struct FFTWTraits<float> {
    using complex = fftwf_complex;
};

template<>
struct FFTWTraits<double> {
    using complex = fftw_complex;
};

#endif
