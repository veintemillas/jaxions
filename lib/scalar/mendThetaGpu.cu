#include "utils/index.cuh"

#include "enum-field.h"

#include "utils/reduceGpu.cuh"

#define	BLSIZE 256

using namespace indexHelper;

template<typename Float>
static __device__ __forceinline__ void	mendThetaCoreGpu(const uint idx, Float * __restrict__ m, Float * __restrict__ v, const Float zPi, const uint Lx, const uint Sf, uint &count)
{
	uint X[3], idxPx, idxPy, idxPz = idx + Sf;

	Float mel, mPx, mPy, mPz, vPx, vPy, vPz, tmp;

	constexpr Float vPi = 2.*M_PI;

	idx2Vec(idx, X, Lx);

	if (X[0] == Lx-1)
		idxPx = idx - Lx + 1;
	else
		idxPx = idx + 1;

	if (X[1] == Lx-1)
		idxPy = idx + Lx;
	else
		idxPy = idx + Lx;

	uint	idxVx = idxPx - Sf, idxVy = idxPy - Sf, idxVz = idx;

	mel = m[idx];
	mPx = m[idxPx];
	mPy = m[idxPy];
	mPz = m[idxPz];
	vPx = v[idxVx];
	vPy = v[idxVy];
	vPz = v[idxVz];

	//	X-Direction

	tmp = mPx - mel;

	if (tmp > zPi) {
		mPx -= zPi;
		vPx -= vPi;
		count++;
	} else if (tmp < -zPi) {
		mPx += zPi;
		vPx += vPi;
		count++;
	}

	//	Y-Direction

	tmp = mPy - mel;

	if (tmp > zPi) {
		mPy -= zPi;
		vPy -= vPi;
		count++;
	} else if (tmp < -zPi) {
		mPy -= zPi;
		vPy -= vPi;
		count++;
	}

	//	Z-Direction

	tmp = mPz - mel;

	if (tmp > zPi) {
		mPz -= zPi;
		vPz -= vPi;
		count++;
	} else if (tmp < -zPi) {
		mPz -= zPi;
		vPz -= vPi;
		count++;
	}

	m[idxPx] = mPx;
	m[idxPy] = mPy;
	m[idxPz] = mPz;
	v[idxVx] = vPx;
	v[idxVy] = vPy;
	v[idxVz] = vPz;

	return;
}

template<typename Float>
__global__ void	mendThetaKernel(Float * __restrict__ m, Float * __restrict__ v, const Float zPi, const uint Lx, const uint Sf, const uint V, uint * __restrict__ nJmp, uint * __restrict__ partial)
{
	uint idx = Sf + (threadIdx.x + blockDim.x*blockIdx.x);
	uint count = 0;

	if	(idx < V)
		mendThetaCoreGpu<Float>(idx, m, v, zPi, Lx, Sf, count);

	reduction<BLSIZE,uint,1>   (nJmp, &count, partial);
}

uint	mendThetaGpu	(void * __restrict__ m, void * __restrict__ v, const double z, const uint Lx, const uint Vo, const uint Vf, FieldPrecision precision, cudaStream_t &stream)
{
	dim3  gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,1,1);
	dim3  blockSize(BLSIZE,1,1);

	const int nBlocks = gridSize.x;

	uint  *d_cnt, *partial, nJmp;

	if ((cudaMalloc(&d_cnt, sizeof(uint)) != cudaSuccess) || (cudaMalloc(&partial, sizeof(uint)*nBlocks*8) != cudaSuccess))
		return	0;

	const double zPi = 2.*M_PI*z;

	if (precision == FIELD_DOUBLE)
		mendThetaKernel<<<gridSize,blockSize,0,stream>>> (static_cast<double*>(m), static_cast<double*>(v),         zPi, Lx, Vo, Vf, d_cnt, partial);
	else if (precision == FIELD_SINGLE)                                                
		mendThetaKernel<<<gridSize,blockSize,0,stream>>> (static_cast<float *>(m), static_cast<float *>(v), (float) zPi, Lx, Vo, Vf, d_cnt, partial);

	cudaDeviceSynchronize();
	cudaMemcpy(&nJmp, d_cnt, sizeof(uint), cudaMemcpyDeviceToHost);

	cudaFree(d_cnt);
	cudaFree(partial);

	return	nJmp;
}
