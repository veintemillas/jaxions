#include <complex>
#include <random>
#include <omp.h>

#include "scalar/scalarField.h"
#include "enum-field.h"
#include "utils/memAlloc.h"
#include "utils/parse.h"

#include "comms/comms.h"

template<typename Float>
void	randXeon (std::complex<Float> * __restrict__ m, const size_t Vo, const size_t Vf)
{
	int	maxThreads = omp_get_max_threads();
	int	*sd;

	trackAlloc((void **) &sd, sizeof(int)*maxThreads);

	std::random_device seed;		// Totally random seed coming from memory garbage

	for (int i=0; i<maxThreads; i++)
		sd[i] = seed()*(1 + commRank());

	const int ene = sqrt(Vo);

	#pragma omp parallel default(shared)
	{
		int nThread = omp_get_thread_num();
		int rank = commRank();
		size_t Lz = sizeN/commSize();
		size_t local_z_start = rank*Lz;
		//printf("rank %d (t %d)-> N=%d Lz %d lzs = %d \n", rank, nThread, sizeN, Lz, local_z_start);

		std::mt19937_64 mt64(sd[nThread]);		// Mersenne-Twister 64 bits, independent per thread
		std::uniform_real_distribution<Float> uni(-1.0, 1.0);

		#pragma omp for schedule(static)	// This is NON-REPRODUCIBLE, unless one thread is used. Alternatively one can fix the seeds
		for (size_t idx=Vo; idx<Vf; idx++)
		{
			//RANDOM INITIAL CONDITIONS
			//m[idx]   = std::complex<Float>(uni(mt64), uni(mt64));
			//RANDOM AXIONS AROUND CP CONSERVING MINIMUM
			//m[idx]   = std::complex<Float>(0.2, uni(mt64)/10.);
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
			// size_t iz = pidx/Vo + local_z_start;
			// size_t iy = (pidx%Vo)/sizeN ;
			// size_t ix = (pidx%Vo)%sizeN ;
			// int z = iz;
			// int y = iy;
			// int x = ix;
			// //CENTERED AT GRID
			// Float theta = ((Float) ((x-sizeN/2)*(x-sizeN/2)+(y-sizeN/2)*(y-sizeN/2)+(z-sizeN/2)*(z-sizeN/2)))/(Vo);
			//CENTERED AT ZERO
			// if (z>sizeN/2) {z = z-sizeN; }
			// if (y>sizeN/2) {y = y-sizeN; }
			// if (x>sizeN/2) {x = x-sizeN; }
			//
			// Float theta = ((Float) (x*x + y*y + z*z))/(Vo);

			// theta = exp(-theta*30.)*12.;
			// m[idx] = std::complex<Float>(cos(theta), sin(theta));

			//	ONE MODE

			// size_t pidx = idx-Vo;
			// size_t iz = pidx/Vo + local_z_start;
			// size_t iy = (pidx%Vo)/sizeN ;
			// size_t ix = (pidx%Vo)%sizeN ;
			//
			// Float theta = ((Float) 0.0001*cos(3.14159*2.*iz*3/ene)+0.0*cos(3.14159*2.*ix*5/ene));
			// m[idx] = std::complex<Float>(cos(theta), sin(theta));


			//// if(ix<2)
			//// {
			//// 	printf("MINICLUSTER data! %d %d (%d,%d,%d) %f %f \n",idx, (pidx%Vo)%sizeN, ix,iy,iz,m[idx].real(),m[idx].imag());
			//// }

			//	MINICLUSTER

			size_t pidx = idx-Vo;
			size_t iz = pidx/Vo + local_z_start;
			size_t iy = (pidx%Vo)/sizeN ;
			size_t ix = (pidx%Vo)%sizeN ;
			int z = iz;
			int y = iy;
			int x = ix;
			//CENTERED AT GRID, z=0
			if (z>sizeN/2) {z = z-sizeN; }
			Float aL = (Float) 1./4.;	//RADIUS
			Float rho2 = ((Float) ((x-sizeN/2)*(x-sizeN/2)+(y-sizeN/2)*(y-sizeN/2))/Vo);
			Float d12 = (sqrt(rho2) + aL)*(sqrt(rho2) + aL) + (z*z/Vo) ;
			Float d22 = (sqrt(rho2) - aL)*(sqrt(rho2) - aL) + (z*z/Vo) ;
			Float theta = 1.5707963*(0.5 + (4. - d12 - d22)/(4.*sqrt(d12*d22)))*z/abs(z+ 0.0000001)	;
			m[idx] = std::complex<Float>(cos(theta), sin(theta));
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
