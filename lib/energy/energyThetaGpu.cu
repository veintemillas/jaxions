#include "complexGpu.cuh"
#include "utils/index.cuh"

#include "enum-field.h"
#include "cub/cub.cuh"

#include "scalar/varNQCD.h"
#include "utils/parse.h"

#define	BLSIZE 512

using namespace gpuCu;
using namespace indexHelper;

__device__ uint bCount = 0;

template <int bSize, typename Float>
__device__ inline void reductionTheta(Float * __restrict__ eRes, const Float * __restrict__ tmp, Float *partial)
{
	typedef cub::BlockReduce<Float, bSize, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduce;
	const int blockSurf = gridDim.x*gridDim.y;

	__shared__ bool isLastBlockDone;
	__shared__ typename BlockReduce::TempStorage cub_tmp[5];


	Float tmpGthx = BlockReduce(cub_tmp[0]).Sum(tmp[0]);
	Float tmpGthy = BlockReduce(cub_tmp[1]).Sum(tmp[1]);
	Float tmpGthz = BlockReduce(cub_tmp[2]).Sum(tmp[2]);
	Float tmpKth  = BlockReduce(cub_tmp[3]).Sum(tmp[3]);
	Float tmpVth  = BlockReduce(cub_tmp[4]).Sum(tmp[4]);

	if (threadIdx.x == 0)
	{
		const int bIdx = blockIdx.x + gridDim.x*blockIdx.y;

		partial[bIdx + 0*blockSurf] = tmpGthx;
		partial[bIdx + 1*blockSurf] = tmpGthy;
		partial[bIdx + 2*blockSurf] = tmpGthz;
		partial[bIdx + 3*blockSurf] = tmpKth;
		partial[bIdx + 4*blockSurf] = tmpVth;

		__threadfence();

		unsigned int cBlock = atomicInc(&bCount, blockSurf);
		isLastBlockDone = (cBlock == (blockSurf-1));
	}

	__syncthreads();

	// finish the reduction if last block
	if (isLastBlockDone)
	{
		uint i = threadIdx.x;

		tmpGthx = 0.;
		tmpGthy = 0.;
		tmpGthz = 0.;
		tmpKth  = 0.;
		tmpVth  = 0.;

//		tmp = 0.;

		while (i < blockSurf)
		{

			tmpGthx += partial[i + 0*blockSurf];
			tmpGthy += partial[i + 1*blockSurf];
			tmpGthz += partial[i + 2*blockSurf];
			tmpKth  += partial[i + 3*blockSurf];
			tmpVth  += partial[i + 4*blockSurf];

//			tmp  += partial[i];

			i += bSize;
		}

		tmpGthx = BlockReduce(cub_tmp[0]).Sum(tmpGthx);
		tmpGthy = BlockReduce(cub_tmp[1]).Sum(tmpGthy);
		tmpGthz = BlockReduce(cub_tmp[2]).Sum(tmpGthz);
		tmpKth  = BlockReduce(cub_tmp[3]).Sum(tmpKth);
		tmpVth  = BlockReduce(cub_tmp[4]).Sum(tmpVth);

		if (threadIdx.x == 0)
		{

			eRes[0] = tmpGthx;
			eRes[1] = tmpGthy;
			eRes[2] = tmpGthz;
			eRes[3] = tmpKth;
			eRes[4] = tmpVth;

			bCount = 0;
		}
	}
}

template<typename Float>
static __device__ __forceinline__ Float modPi (const Float x, const Float OneOvPi, const Float TwoPiZ)
{
	const Float tmp = x*OneOvPi;

	if (tmp >=  1.)
		return (x-TwoPiZ);

	if (tmp <  -1.)
		return (x+TwoPiZ);

	return x;
}

template<typename Float>
static __device__ __forceinline__ void	energyThetaCoreGpu(const uint idx, const Float * __restrict__ m, const Float * __restrict__ v, const uint Lx, const uint Sf, const Float iZ, const Float zP, const Float tPz, double *tR)
{
	uint X[3], idxPx, idxPy, idxMx, idxMy;

	Float mel, vel, tmpPx, tmpPy, tmpPz, tmpMx, tmpMy, tmpMz, aX, aY, aZ, Kt, Vt;

	idx2Vec(idx, X, Lx);

	if (X[0] == Lx-1)
		idxPx = idx - Lx+1;
	else
		idxPx = idx+1;

	if (X[0] == 0)
		idxMx = idx + Lx-1;
	else
		idxMx = idx-1;

	if (X[1] == Lx-1)
		idxPy = idx - Sf + Lx;
	else
		idxPy = idx + Lx;

	if (X[1] == 0)
		idxMy = idx + Sf - Lx;
	else
		idxMy = idx - Lx;

	mel = m[idx];
	vel = v[idx-Sf];

	tmpPx = modPi(m[idxPx]    - mel, zP, tPz); 
	tmpPy = modPi(m[idxPy]    - mel, zP, tPz); 
	tmpPz = modPi(m[idx + Sf] - mel, zP, tPz); 
	tmpMx = modPi(m[idxMx]    - mel, zP, tPz); 
	tmpMy = modPi(m[idxMy]    - mel, zP, tPz); 
	tmpMz = modPi(m[idx - Sf] - mel, zP, tPz); 

	aX = tmpPx*tmpPx + tmpMx*tmpMx;
	aY = tmpPy*tmpPy + tmpMy*tmpMy;
	aZ = tmpPz*tmpPz + tmpMz*tmpMz;

	Kt = vel*vel;
	Vt = 1.0f - cos(mel*iZ);

	tR[0] = (double) aX;
	tR[1] = (double) aY;
	tR[2] = (double) aZ;
	tR[3] = (double) Kt;
	tR[4] = (double) Vt;
}

template<typename Float>
__global__ void	energyThetaKernel(const Float * __restrict__ m, const Float * __restrict__ v, const uint Lx, const uint Sf, const uint V, const Float iZ, const Float zP, const Float tPz, double *eR, double *partial)
{
	uint idx = Sf + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	double tmp[5] = { 0., 0., 0., 0., 0. };

	if	(idx < V)
		energyThetaCoreGpu<Float>(idx, m, v, Lx, Sf, iZ, zP, tPz, tmp);

	reductionTheta<BLSIZE,double>   (eR, tmp, partial);
}

int	energyThetaGpu	(const void * __restrict__ m, const void * __restrict__ v, double *z, const double delta2, const double nQcd,
			 const uint Lx, const uint Lz, const uint V, const uint S, FieldPrecision precision, double *eR, cudaStream_t &stream)
{
	const uint Vm = V+S;
	const uint Lz2 = V/(Lx*Lx);
	dim3  gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3  blockSize(BLSIZE,1,1);
	const int nBlocks = gridSize.x*gridSize.y;

	const double zR   = *z;
	double *tR, *partial;

	if ((cudaMalloc(&tR, sizeof(double)*5) != cudaSuccess) || (cudaMalloc(&partial, sizeof(double)*5*nBlocks*4) != cudaSuccess))
	{
		return -1;
	}

	if (precision == FIELD_DOUBLE)
	{
		const double iZ  = 1./zR;
		const double zP  = M_1_PI*iZ;
		const double tPz = 2.0*M_PI*zR;
		energyThetaKernel<<<gridSize,blockSize,0,stream>>> (static_cast<const double*>(m), static_cast<const double*>(v), Lx, S, Vm, iZ, zP, tPz, tR, partial);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float iZ  = 1.0/zR;
		const float zP  = M_1_PI*iZ;
		const float tPz = 2.f*M_PI*zR;
		energyThetaKernel<<<gridSize,blockSize,0,stream>>> (static_cast<const float*>(m), static_cast<const float*>(v), Lx, S, Vm, iZ, zP, tPz, tR, partial);
	}

	cudaDeviceSynchronize();

	cudaMemcpy(eR, tR, sizeof(double)*5, cudaMemcpyDeviceToHost);
	cudaFree(tR); cudaFree(partial);

	const double o2  = 0.25/delta2;
	const double zQ  = axionmass2((float) zR, nQcd, zthres, zrestore)*zR*zR;
	const double iz2 = 1./(zR*zR);

	eR[0] *= o2*iz2;
	eR[1] *= o2*iz2;
	eR[2] *= o2*iz2;
	eR[3] *= .5*iz2;
	eR[4] *= zQ;

	return	0;
}

