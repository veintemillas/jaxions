#include "complexGpu.cuh"
#include "utils/index.cuh"

#include "enum-field.h"
#include "cub/cub.cuh"

#define	BLSIZE 512

using namespace gpuCu;
using namespace indexHelper;


template<typename Float>
static __device__ __forceinline__ int	stringHand(const complex<Float> s1, const complex<Float> s2)
{
	int hand = 0;

	if (s1.imag()*s2.imag() < 0)
		hand = (((s1*conj(s2)).imag() > 0)<<1) - 1;

	return hand;
}


template<typename Float>
static __device__ __forceinline__ void	stringCoreGpu(const uint idx, const complex<Float> * __restrict__ m, const uint Lx, const uint Sf, void * __restrict__ str)
{
	uint X[3], idxPx, idxPy, idxXY, idxYZ, idxZX;

	complex<Float> mel, mPx, mXY, mPy, mYZ, mPz, mZX;
	uint sIdx = idx-Sf;
	int hand = 0;
	char strDf = 0;

	idx2Vec(idx, X, Lx);

	if (X[0] == Lx-1)
	{
		idxPx = idx - Lx + 1;
		idxZX = idxPx + Sf;

		if (X[1] == Lx-1)
		{
			idxPy = sIdx + Lx;
			idxXY = sIdx + 1;
			idxYZ = idx + Lx;
		} else {
			idxPy = idx + Lx;
			idxXY = idx + 1;
			idxYZ = idx + Sf + Lx;
		}
	} else {
		idxPx = idx + 1;
		idxZX = idxPx + Sf;

		if (X[1] == Lx-1)
		{
			idxPy = sIdx + Lx;
			idxYZ = idx + Lx;
		} else {
			idxPy = idx + Lx;
			idxYZ = idx + Sf + Lx;
		}

		idxXY = idxPy + 1;
	}

	mel = m[idx];
	mPx = m[idxPx];
	mPy = m[idxPy];
	mXY = m[idxXY];
	mPz = m[idx+Sf];
	mZX = m[idxZX];
	mYZ = m[idxYZ];

	// Primera plaqueta XY

	hand += stringHand (mel, mPx);
	hand += stringHand (mPx, mXY);
	hand += stringHand (mXY, mPy);
	hand += stringHand (mPy, mel);
/*	ARREGLAR PARA QUE SOLO HAYA UN STORE	*/
/*	LA QUIRALIDAD SE GUARDA MAL	*/
	if (hand == 2)
		strDf |= STRING_XY_POSITIVE;
	else if (hand == -2)
		strDf |= STRING_XY_NEGATIVE;

	hand = 0;

	// Segunda plaqueta YZ

	hand += stringHand (mel, mPy);
	hand += stringHand (mPy, mYZ);
	hand += stringHand (mYZ, mPz);
	hand += stringHand (mPz, mel);

	if (hand == 2)
		strDf |= STRING_YZ_POSITIVE;
	else if (hand == -2)
		strDf |= STRING_YZ_NEGATIVE;

	hand = 0;

	// Tercera plaqueta ZX

	hand += stringHand (mel, mPz);
	hand += stringHand (mPz, mZX);
	hand += stringHand (mZX, mPx);
	hand += stringHand (mPx, mel);

	if (hand == 2)
		strDf |= STRING_ZX_POSITIVE;
	else if (hand == -2)
		strDf |= STRING_ZX_NEGATIVE;

	static_cast<char *>(str)[sIdx] = strDf;
}

template<typename Float>
__global__ void	stringKernel(void * __restrict__ strg, const complex<Float> * __restrict__ m, const uint Lx, const uint Sf, const uint V)
{
	uint idx = Sf + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	if	(idx < V)
		stringCoreGpu<Float>(idx, m, Lx, Sf, strg);
}

size_t	stringGpu	(const void * __restrict__ m, const uint Lx, const uint V, const uint S, FieldPrecision precision, void * __restrict__ str, cudaStream_t &stream)
{
	const uint Vm = V+S;
	const uint Lz2 = V/(Lx*Lx);
	dim3  gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3  blockSize(BLSIZE,1,1);

	void   *strg;

	if (cudaMalloc(&strg, sizeof(char)*V) != cudaSuccess)
		return -1;

	if (precision == FIELD_DOUBLE)
		stringKernel<<<gridSize,blockSize,0,stream>>> (strg, static_cast<const complex<double>*>(m), Lx, S, Vm);
	else if (precision == FIELD_SINGLE)
		stringKernel<<<gridSize,blockSize,0,stream>>> (strg, static_cast<const complex<float>*>(m), Lx, S, Vm);

	cudaDeviceSynchronize();

	cudaMemcpy(str, strg, sizeof(char)*V, cudaMemcpyDeviceToHost);
	cudaFree(strg);

	return	0;
}
