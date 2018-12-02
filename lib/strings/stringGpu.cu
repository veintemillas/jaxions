#include "complexGpu.cuh"
#include "utils/index.cuh"

#include "enum-field.h"

#include "utils/reduceGpu.cuh"

#define	BLSIZE 256

using namespace gpuCu;
using namespace indexHelper;

template<typename Float>
static __device__ __forceinline__ int	stringHand(const complex<Float> s1, const complex<Float> s2)
{
	int hand = 0;

	if (s1.imag()*s2.imag() < 0)
		hand = (((s1.imag()*s2.real() - s1.real()*s2.imag()) > 0) << 1) - 1;

	return hand;
}

template<typename Float>
static __device__ __forceinline__ int2	stringWall(const complex<Float> s1, const complex<Float> s2)
{
	int2 hand = make_int2(0,0);

	if (s1.imag()*s2.imag() < 0) {
		Float cross = s1.imag()*s2.real() - s1.real()*s2.imag();
		hand.x = ((cross > 0) << 1) - 1;
		hand.y = ((cross*(s1.imag() - s2.imag())) < 0);
	}

	return hand;
}


template<typename Float>
static __device__ __forceinline__ void	stringCoreGpu(const uint idx, const complex<Float> * __restrict__ m, const uint Lx, const uint Sf, const uint rLx, const uint rSf,
						      const Float ratio, const Float datio, void * __restrict__ str, uint * __restrict__ tStr)
{
	uint X[3], idxPx, idxPy, idxXY, idxYZ, idxZX;
	int2 hand0X, hand0Y, hand0Z;
	int  handXY, handXZ, handYX, handYZ, handZX, handZY;

	complex<Float> mel, mPx, mXY, mPy, mYZ, mPz, mZX;
	uint sIdx = idx-Sf;
	int  hand = 0;
	int  strDf = 0;

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

	hand0X = stringWall(mel, mPx);
	hand0Y = stringWall(mel, mPy);
	hand0Z = stringWall(mel, mPz);

	handXY = stringHand(mPx, mXY);
	handXZ = stringHand(mPx, mZX);
	handYX = stringHand(mPy, mXY);
	handYZ = stringHand(mPy, mYZ);
	handZX = stringHand(mPz, mZX);
	handZY = stringHand(mPz, mYZ);

	// Primera plaqueta XY

	hand = hand0X.x + handXY - handYX - hand0Y.x;

	if (hand == 2) {
		tStr[0]++;
		tStr[1]++;
		strDf |= STRING_XY_POSITIVE;
	} else if (hand == -2) {
		tStr[0]++;
		tStr[1]--;
		strDf |= STRING_XY_NEGATIVE;
	}

	// Segunda plaqueta YZ

	hand = hand0Y.x + handYZ - handZY - hand0Z.x;

	if (hand == 2) {
		tStr[0]++;
		tStr[1]++;
		strDf |= STRING_YZ_POSITIVE;
	} else if (hand == -2) {
		tStr[0]++;
		tStr[1]--;
		strDf |= STRING_YZ_NEGATIVE;
	}

	// Tercera plaqueta ZX

	hand = hand0Z.x + handZX - handXZ - hand0X.x;

	if (hand == 2) {
		tStr[0]++;
		tStr[1]++;
		strDf |= STRING_ZX_POSITIVE;
	} else if (hand == -2) {
		tStr[0]++;
		tStr[1]--;
		strDf |= STRING_ZX_NEGATIVE;
	}

	auto nWall = hand0X.y | hand0Y.y | hand0Z.y;

	tStr[2] += nWall;
	strDf   |= nWall << 6;	// STRING_WALL = 64 = 2^6, and we don't distinguish the wall direction (lack of bits) 

	uint oIdx = ((uint)(((Float) X[0])*ratio)) + ((uint)(((Float) X[1])*ratio))*rLx + ((uint)((Float) (X[2]-1)*datio))*rSf;
	uint bIdx = oIdx>>2;
	uint rIdx = oIdx - (bIdx<<2);
	strDf <<= rIdx*8;
// Several threads can write in the same place
// There is a problem here
//	static_cast<char *>(str)[oIdx] |= strDf;
	atomicOr(static_cast<int *>(str) + bIdx, strDf);

	return;
}

template<typename Float>
__global__ void	stringKernel(void * __restrict__ strg, const complex<Float> * __restrict__ m, const uint Lx, const uint Sf, const uint rLx, const uint rSf, const uint V,
			     const Float ratio, const Float datio, uint *nStr, uint *partial)
{
	uint idx = Sf + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));
	uint tStr[3] = { 0, 0, 0 };

	if	(idx < V)
		stringCoreGpu<Float>(idx, m, Lx, Sf, rLx, rSf, ratio, datio, strg, tStr);

	reduction<BLSIZE,uint,3>   (nStr, tStr, partial);
}

uint3	stringGpu	(const void * __restrict__ m, const uint Lx, const uint Lz, const uint rLx, const uint rLz, const uint S, const uint V,
			 FieldPrecision precision, void * __restrict__ str, cudaStream_t &stream)
{
	const uint Vm = V+S;
	const uint Lz2 = V/(Lx*Lx);
	dim3  gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3  blockSize(BLSIZE,1,1);

	const int nBlocks = gridSize.x*gridSize.y;

	void  *strg;
	uint  *d_str, *partial, nStr[3];
	uint  rSf = rLx*rLx;
	uint  rV  = rSf*rLz;

	if (cudaMalloc(&strg, sizeof(char)*rV+4) != cudaSuccess)
		return	make_uint3(UINT_MAX, UINT_MAX, UINT_MAX);

	if ((cudaMalloc(&d_str, sizeof(uint)*3) != cudaSuccess) || (cudaMalloc(&partial, sizeof(uint)*nBlocks*8) != cudaSuccess))
		return	make_uint3(UINT_MAX, UINT_MAX, UINT_MAX);

	double	ratio = ((double) rLx)/((double) Lx);
	double	datio = ((double) rLz)/((double) Lz);

	if (precision == FIELD_DOUBLE)
		stringKernel<<<gridSize,blockSize,0,stream>>> (strg, static_cast<const complex<double>*>(m), Lx, S, rLx, rSf, Vm,         ratio,         datio, d_str, partial);
	else if (precision == FIELD_SINGLE)
		stringKernel<<<gridSize,blockSize,0,stream>>> (strg, static_cast<const complex<float> *>(m), Lx, S, rLx, rSf, Vm, (float) ratio, (float) datio, d_str, partial);

	cudaDeviceSynchronize();
	cudaMemcpy(str,   strg, sizeof(char)*rV, cudaMemcpyDeviceToHost);
	cudaMemcpy(nStr, d_str, sizeof(uint)*3,  cudaMemcpyDeviceToHost);

	cudaFree(strg);
	cudaFree(d_str);
	cudaFree(partial);

	return	make_uint3(nStr[0], nStr[1], nStr[2]);
}
