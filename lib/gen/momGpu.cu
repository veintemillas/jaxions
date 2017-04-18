#include <random>
#include <curand.h>
#include <curand_kernel.h>

#include "complexGpu.cuh"
#include "utils/index.cuh"
#include "scalar/scalarField.h"
#include "enum-field.h"

#include "comms/comms.h"

#define	BLSIZE	512

using namespace gpuCu;
using namespace indexHelper;

template<typename Float>
__global__ void	momKernelGpu (curandState_t * state, complex<Float> * __restrict__ m, const uint Lx, const uint Lz, const uint Vo, const uint Vf,
			      const uint seed, const uint rank, const uint kMax, const Float kCrit)
{
	uint size = Vf - Vo;
	uint idx  = threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y);

	if	(idx < size) {

		uint	X[3];
		idx2Vec(idx, X, Lx);

		X[2] += Lz*rank;

		const uint p2 = X[0]*X[0] + X[1]*X[1] + X[2]*X[2]; 

		if	(p2 <= kMax) {

			curand_init (seed, idx, rank*size, &state[idx]);

			Float	ph  = (Float) (2.*M_PI*curand_uniform (&state[idx]));
			Float	mod = sqrt((Float) p2)/kCrit;
			Float	sc  = (mod == 0.) ? 1.0 : sin(mod)/mod;

			m[Vo + idx]	= complex<Float>(cos(ph), sin(ph))*sc;
		}
	}
}

void	momGpu (Scalar *field, const uint kMax, const double kCrit)
{
	const uint	Lx = field->Length();
	const uint	S  = field->Surf();
	const uint	V  = field->Size();
	const uint	Lz = field->Depth();

	dim3		gridSize((S+BLSIZE-1)/BLSIZE,Lz,1);
	dim3		blockSize(BLSIZE,1,1);

	cudaStream_t	&stream = static_cast<cudaStream_t *>(field->Streams())[0];

	curandState_t	*state;
	if ((cudaMalloc(&state, V*sizeof(curandState))) == NULL) {
		printf("Error: Couldn't allocate memory in device for random number generator\n");
		exit (1);
	}

	std::random_device seed;

	switch (field->Precision())
	{
		case FIELD_DOUBLE:
		momKernelGpu<<<gridSize,blockSize,0,stream>>>(state, static_cast<complex<double>*> (field->mGpu()), Lx, Lz, S, V+S, seed(), commRank(), kMax, kCrit);
		break;

		case FIELD_SINGLE:
		momKernelGpu<<<gridSize,blockSize,0,stream>>>(state, static_cast<complex<float> *> (field->mGpu()), Lx, Lz, S, V+S, seed(), commRank(), kMax, (float) kCrit);
		break;

		default:
		break;
	}

	cudaFree (state);
}

#undef	BLSIZE
