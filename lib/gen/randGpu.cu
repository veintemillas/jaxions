#include <random>
#include <curand.h>
#include <curand_kernel.h>

#include "complexGpu.cuh"
#include "scalar/scalarField.h"
#include "enum-field.h"

#include "comms/comms.h"

#define	BLSIZE	512
#define	CTSIZE	262144

using namespace gpuCu;

__global__ void	randInitGpu (curandState_t * state, const uint seed, const uint rank, const uint size)
{
	uint bIdx = blockIdx.x + gridDim.x*blockIdx.y;
	uint idx  = threadIdx.x + blockDim.x*bIdx;

	curand_init (seed*gridDim.x*gridDim.y + rank*size*gridDim.x*gridDim.y + bIdx, threadIdx.x, 0, &state[idx]);
}

template<typename Float>
__global__ void	randKernelGpu (curandState_t * __restrict__ state, complex<Float> * __restrict__ m, const uint Vo, const uint Vf, const uint workPerThread)
{
	uint size = Vf - Vo;
	uint idx  = threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y);
	uint wIdx = idx*workPerThread;

	if	(wIdx < size) {

		curandState_t localState = state[idx];

		#pragma unroll
		for	(int k=0; k<workPerThread; k++) {

			Float	rP = (Float) (2.*curand_uniform (&localState) - 1.);
			Float	iP = (Float) (2.*curand_uniform (&localState) - 1.);

			m[Vo + wIdx + k]	= complex<Float>(rP, iP);
		}

		state[idx]	= localState;
	}
}

void	randGpu (Scalar *field)
{
	const uint	S  = field->Surf();
	const uint	V  = field->Size();
	const uint	Lz = field->Depth();

	size_t		memGpu = gpuMemAvail();

	cudaStream_t	&stream = static_cast<cudaStream_t *>(field->Streams())[0];

	if	(field->LowMem())
		memGpu -= 2*(V+S)*field->DataSize();
	else
		memGpu -= (3*V+4*S)*field->DataSize();

	memGpu	*= 3;		// El factor 0.75 es para dejar hueco por si las moscas
	memGpu	/= 4*sizeof(curandState_t);

	if	(memGpu > V)
		memGpu = V;

	printf	("Allocating %lu bytes for the pRNG\n", memGpu*sizeof(curandState_t));
	fflush	(stdout);

	curandState_t	*state;
	if ((cudaMalloc(&state, memGpu*sizeof(curandState_t))) != cudaSuccess) {
		printf("Error: Couldn't allocate %zu bytes in device for random number generator\n", memGpu*sizeof(curandState_t));
		exit (1);
	}

	printf("\nParallel RNG using %zu bytes in device\n", memGpu*sizeof(curandState_t));
	fflush(stdout);

	std::random_device seed;

	dim3		gridSize((memGpu/Lz+BLSIZE-1)/BLSIZE,Lz,1);
	dim3		blockSize(BLSIZE,1,1);

	randInitGpu  <<<gridSize,blockSize,0,stream>>>(state, seed(), commRank(), memGpu);

	uint	workPerThread = S/(memGpu/Lz);

	printf("GridSize\t(%d %d %d)\nBlockSize\t(%d %d %d)\nWork per Thread %d\n", gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z, workPerThread);
	fflush(stdout);

	switch (field->Precision())
	{
		case FIELD_DOUBLE:
		//randKernelGpu<<<gridSize,blockSize,0,stream>>>(state, static_cast<complex<double>*> (field->mGpu()), S, V+S, workPerThread);
		break;

		case FIELD_SINGLE:
		//randKernelGpu<<<gridSize,blockSize,0,stream>>>(state, static_cast<complex<float> *> (field->mGpu()), S, V+S, workPerThread);
		break;

		default:
		break;
	}

	cudaFree (state);
}

#undef	BLSIZE
