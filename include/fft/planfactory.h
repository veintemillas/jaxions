#ifndef _FFTPLAN_FACTORY_
#define _FFTPLAN_FACTORY_

#include <fftw3-mpi.h>
#include <type_traits>
#include "traits.h"

enum class FFTKind {
    DFT,
    R2C,
    C2R
};

template<typename Float>
inline void* makeFFTPlan(
    FFTKind kind,
    size_t Nz, size_t Ny, size_t Nx,
    void* in, void* out,
    int direction,
    unsigned baseFlags)
{
    using Traits = FFTWTraits<Float>;
    const bool is2D = (Ny == 1);

    switch (kind) {

    case FFTKind::DFT:
    {
        auto* cin  = static_cast<typename Traits::complex*>(in);
        auto* cout = static_cast<typename Traits::complex*>(out);

        if constexpr (std::is_same_v<Float,float>) {
            if (is2D) {
                return (void*) fftwf_mpi_plan_dft_2d(
                    Nz, Nx, cin, cout, MPI_COMM_WORLD, direction, baseFlags
                );
            } else {
                unsigned flags = baseFlags |
                    ((direction == FFTW_FORWARD) ? FFTW_MPI_TRANSPOSED_OUT
                                                 : FFTW_MPI_TRANSPOSED_IN);
                return (void*) fftwf_mpi_plan_dft_3d(
                    Nz, Ny, Nx, cin, cout, MPI_COMM_WORLD, direction, flags
                );
            }
        } else {
            if (is2D) {
                return (void*) fftw_mpi_plan_dft_2d(
                    Nz, Nx, cin, cout, MPI_COMM_WORLD, direction, baseFlags
                );
            } else {
                unsigned flags = baseFlags |
                    ((direction == FFTW_FORWARD) ? FFTW_MPI_TRANSPOSED_OUT
                                                 : FFTW_MPI_TRANSPOSED_IN);
                return (void*) fftw_mpi_plan_dft_3d(
                    Nz, Ny, Nx, cin, cout, MPI_COMM_WORLD, direction, flags
                );
            }
        }
    }

    case FFTKind::R2C:
    {
        auto* rin  = static_cast<Float*>(in);
        auto* cout = static_cast<typename Traits::complex*>(out);

        if constexpr (std::is_same_v<Float,float>) {
            if (is2D) {
                return (void*) fftwf_mpi_plan_dft_r2c_2d(
                    Nz, Nx, rin, cout, MPI_COMM_WORLD, baseFlags
                );
            } else {
                return (void*) fftwf_mpi_plan_dft_r2c_3d(
                    Nz, Ny, Nx, rin, cout, MPI_COMM_WORLD, baseFlags
                );
            }
        } else {
            if (is2D) {
                return (void*) fftw_mpi_plan_dft_r2c_2d(
                    Nz, Nx, rin, cout, MPI_COMM_WORLD, baseFlags
                );
            } else {
                return (void*) fftw_mpi_plan_dft_r2c_3d(
                    Nz, Ny, Nx, rin, cout, MPI_COMM_WORLD, baseFlags
                );
            }
        }
    }

    case FFTKind::C2R:
    {
        auto* cin  = static_cast<typename Traits::complex*>(in);
        auto* rout = static_cast<Float*>(out);

        if constexpr (std::is_same_v<Float,float>) {
            if (is2D) {
                return (void*) fftwf_mpi_plan_dft_c2r_2d(
                    Nz, Nx, cin, rout, MPI_COMM_WORLD, baseFlags
                );
            } else {
                return (void*) fftwf_mpi_plan_dft_c2r_3d(
                    Nz, Ny, Nx, cin, rout, MPI_COMM_WORLD, baseFlags
                );
            }
        } else {
            if (is2D) {
                return (void*) fftw_mpi_plan_dft_c2r_2d(
                    Nz, Nx, cin, rout, MPI_COMM_WORLD, baseFlags
                );
            } else {
                return (void*) fftw_mpi_plan_dft_c2r_3d(
                    Nz, Ny, Nx, cin, rout, MPI_COMM_WORLD, baseFlags
                );
            }
        }
    }

    }

    return nullptr;
}

#endif
