#include <complex>
#include <random>
#include <omp.h>

#include "scalar/scalarField.h"
#include "enum-field.h"
#include "utils/memAlloc.h"
#include "utils/parse.h"

template<typename Float>
void	randXeon (std::complex<Float> * __restrict__ m, const size_t Vo, const size_t Vf)
{
	int	maxThreads = omp_get_max_threads();
	int	*sd;

	trackAlloc((void **) &sd, sizeof(int)*maxThreads);

	std::random_device seed;		// Totally random seed coming from memory garbage

	for (int i=0; i<maxThreads; i++)
		sd[i] = seed();

	#pragma omp parallel default(shared)
	{
		int nThread = omp_get_thread_num();

		std::mt19937_64 mt64(sd[nThread]);		// Mersenne-Twister 64 bits, independent per thread
		std::uniform_real_distribution<Float> uni(-1.0, 1.0);

		#pragma omp for schedule(static)	// This is NON-REPRODUCIBLE, unless one thread is used. Alternatively one can fix the seeds
		for (size_t idx=Vo; idx<Vf; idx++)
		{
			//RANDOM INITIAL CONDITIONS
			//m[idx]   = std::complex<Float>(uni(mt64), uni(mt64));
			//RANDOM AXIONS AROUND CP CONSERVING MINIMUM
			m[idx]   = std::complex<Float>(0.2, uni(mt64)/1.);
			//RANDOM AXIONS AROUND CP CONSERVING MINIMUM WITH A LITTLE 0 MODE
			//m[idx]   = std::complex<Float>(1.0, 0.1+uni(mt64)/1.);
			//MORE AXIONS
			//m[idx]   = std::complex<Float>(0.0+0.7*uni(mt64), 1.0);
			//LARGE AMPLITUDE AXIONS ZERO MODE
			//m[idx]   = std::complex<Float>(0.0, 1.000001);
			//to produce only SAXIONS for testing
			//m[idx]   = std::complex<Float>(1.2+uni(mt64)/20., 0.0);

			//	MINICLUSTER
			// size_t pidx = idx-Vo;
			// size_t iz = pidx/Vo ;
			// size_t iy = (pidx%Vo)/sizeN ;
			// size_t ix = (pidx%Vo)%sizeN ;
			// if (iz>sizeN/2) {iz = iz-sizeN; }
			// if (iy>sizeN/2) {iy = iy-sizeN; }
			// if (ix>sizeN/2) {ix = ix-sizeN; }
			//
			// Float theta = ((Float) (ix*ix+iy*iy+iz*iz))/(Vo);
			// theta = exp(-20*theta);
			// m[idx] = std::complex<Float>(cos(theta), sin(theta));

			//// if(ix<2)
			//// {
			//// 	printf("MINICLUSTER data! %d %d (%d,%d,%d) %f %f \n",idx, (pidx%Vo)%sizeN, ix,iy,iz,m[idx].real(),m[idx].imag());
			//// }
		}
	}

	trackFree((void **) &sd, ALLOC_TRACK);
}

void	randConf (Scalar *field)
{
	switch (field->Precision())
	{
		case FIELD_DOUBLE:
		randXeon(static_cast<std::complex<double>*> (field->mCpu()), field->Surf(), field->Size()+field->Surf());
		break;

		case FIELD_SINGLE:
		randXeon(static_cast<std::complex<float> *> (field->mCpu()), field->Surf(), field->Size()+field->Surf());
		break;

		default:
		break;
	}
}
