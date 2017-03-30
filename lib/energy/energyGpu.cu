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
__device__ inline void reduction(Float * __restrict__ eRes, const Float * __restrict__ tmp, Float *partial)
{
	typedef cub::BlockReduce<Float, bSize, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduce;
	const int blockSurf = gridDim.x*gridDim.y;

	__shared__ bool isLastBlockDone;
	__shared__ typename BlockReduce::TempStorage cub_tmp[10];
//	__shared__ typename BlockReduce::TempStorage cub_tmp;


	Float tmpGrhx = BlockReduce(cub_tmp[0]).Sum(tmp[0]);
	Float tmpGthx = BlockReduce(cub_tmp[1]).Sum(tmp[1]);
	Float tmpGrhy = BlockReduce(cub_tmp[2]).Sum(tmp[2]);
	Float tmpGthy = BlockReduce(cub_tmp[3]).Sum(tmp[3]);
	Float tmpGrhz = BlockReduce(cub_tmp[4]).Sum(tmp[4]);
	Float tmpGthz = BlockReduce(cub_tmp[5]).Sum(tmp[5]);
	Float tmpVrho = BlockReduce(cub_tmp[6]).Sum(tmp[6]);
	Float tmpVth  = BlockReduce(cub_tmp[7]).Sum(tmp[7]);
	Float tmpKrho = BlockReduce(cub_tmp[8]).Sum(tmp[8]);
	Float tmpKth  = BlockReduce(cub_tmp[9]).Sum(tmp[9]);

//	Float tmp = BlockReduce(cub_tmp).Sum(*tmpC);

	if (threadIdx.x == 0)
	{
		const int bIdx = blockIdx.x + gridDim.x*blockIdx.y;

		partial[bIdx + 0*blockSurf] = tmpGrhx;
		partial[bIdx + 1*blockSurf] = tmpGthx;
		partial[bIdx + 2*blockSurf] = tmpGrhy;
		partial[bIdx + 3*blockSurf] = tmpGthy;
		partial[bIdx + 4*blockSurf] = tmpGrhz;
		partial[bIdx + 5*blockSurf] = tmpGthz;
		partial[bIdx + 6*blockSurf] = tmpVrho;
		partial[bIdx + 7*blockSurf] = tmpVth;
		partial[bIdx + 8*blockSurf] = tmpKrho;
		partial[bIdx + 9*blockSurf] = tmpKth;

//		partial[bIdx] = tmp;

		__threadfence();

		unsigned int cBlock = atomicInc(&bCount, blockSurf);
		isLastBlockDone = (cBlock == (blockSurf-1));
	}

	__syncthreads();

	// finish the reduction if last block
	if (isLastBlockDone)
	{
		uint i = threadIdx.x;

		tmpGrhx = 0., tmpGthx = 0.;
		tmpGrhy = 0., tmpGthy = 0.;
		tmpGrhz = 0., tmpGthz = 0.;
		tmpVrho = 0., tmpVth  = 0.;
		tmpKrho = 0., tmpKth  = 0.;

//		tmp = 0.;

		while (i < blockSurf)
		{

			tmpGrhx += partial[i + 0*blockSurf];
			tmpGthx += partial[i + 1*blockSurf];
			tmpGrhy += partial[i + 2*blockSurf];
			tmpGthy += partial[i + 3*blockSurf];
			tmpGrhz += partial[i + 4*blockSurf];
			tmpGthz += partial[i + 5*blockSurf];
			tmpVrho += partial[i + 6*blockSurf];
			tmpVth  += partial[i + 7*blockSurf];
			tmpKrho += partial[i + 8*blockSurf];
			tmpKth  += partial[i + 9*blockSurf];

//			tmp  += partial[i];

			i += bSize;
		}

		tmpGrhx = BlockReduce(cub_tmp[0]).Sum(tmpGrhx);
		tmpGthx = BlockReduce(cub_tmp[1]).Sum(tmpGthx);
		tmpGrhy = BlockReduce(cub_tmp[2]).Sum(tmpGrhy);
		tmpGthy = BlockReduce(cub_tmp[3]).Sum(tmpGthy);
		tmpGrhz = BlockReduce(cub_tmp[4]).Sum(tmpGrhz);
		tmpGthz = BlockReduce(cub_tmp[5]).Sum(tmpGthz);
		tmpVrho = BlockReduce(cub_tmp[6]).Sum(tmpVrho);
		tmpVth  = BlockReduce(cub_tmp[7]).Sum(tmpVth);
		tmpKrho = BlockReduce(cub_tmp[8]).Sum(tmpKrho);
		tmpKth  = BlockReduce(cub_tmp[9]).Sum(tmpKth);

//		tmp = BlockReduce(cub_tmp).Sum(tmp);

		if (threadIdx.x == 0)
		{

			eRes[0] = tmpGrhx;
			eRes[1] = tmpGthx;
			eRes[2] = tmpGrhy;
			eRes[3] = tmpGthy;
			eRes[4] = tmpGrhz;
			eRes[5] = tmpGthz;
			eRes[6] = tmpVrho;
			eRes[7] = tmpVth;
			eRes[8] = tmpKrho;
			eRes[9] = tmpKth;

//			eRes[0] = tmp;

			bCount = 0;
		}
	}
}


template<typename Float>
static __device__ __forceinline__ void	energyCoreGpu(const uint idx, const complex<Float> * __restrict__ m, const complex<Float> * __restrict__ v, const uint Lx, const uint Sf, const double iZ, const double iZ2, double *tR, const Float shift)
{
	uint X[3], idxPx, idxPy, idxMx, idxMy;

	complex<Float> mPX, mPY, mPZ, mMX, mMY, mMZ, tmp, tp2, vOm;

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

	tp2 = m[idx];
	tmp = tp2 - shift;

	Float mod = tmp.real()*tmp.real() + tmp.imag()*tmp.imag();
	Float mFac = iZ2*mod;
	Float iMod = 1./mod;

	mPX = (m[idxPx]  - tp2)*conj(m[idx])*iMod;
	mPY = (m[idxPy]  - tp2)*conj(m[idx])*iMod;
	mPZ = (m[idx+Sf] - tp2)*conj(m[idx])*iMod;
	mMX = (m[idxMx]  - tp2)*conj(m[idx])*iMod;
	mMY = (m[idxMy]  - tp2)*conj(m[idx])*iMod;
	mMZ = (m[idx-Sf] - tp2)*conj(m[idx])*iMod;
	vOm = v[idx-Sf]*conj(tmp)*iMod - gpuCu::complex<Float>(iZ, 0.);

	tR[0] = (double) ((Float) (mFac*(mPX.real()*mPX.real() + mMX.real()*mMX.real())));
	tR[1] = (double) ((Float) (mFac*(mPX.imag()*mPX.imag() + mMX.imag()*mMX.imag())));
	tR[2] = (double) ((Float) (mFac*(mPY.real()*mPY.real() + mMY.real()*mMY.real())));
	tR[3] = (double) ((Float) (mFac*(mPY.imag()*mPY.imag() + mMY.imag()*mMY.imag())));
	tR[4] = (double) ((Float) (mFac*(mPZ.real()*mPZ.real() + mMZ.real()*mMZ.real())));
	tR[5] = (double) ((Float) (mFac*(mPZ.imag()*mPZ.imag() + mMZ.imag()*mMZ.imag())));
	tR[6] = (double) ((Float) (mFac - 1.)*(mFac - 1.));
	//tR[7] = (double) (((Float) 1.) - tmp.real()*iZ);	// Old potential
	tR[7] = (double) (((Float) 1.) - tmp.real()/sqrt(mod));
	tR[8] = (double) ((Float) (mFac*vOm.real()*vOm.real()));
	tR[9] = (double) ((Float) (mFac*vOm.imag()*vOm.imag()));
}

template<typename Float>
__global__ void	energyKernel(const complex<Float> * __restrict__ m, const complex<Float> * __restrict__ v, const uint Lx, const uint Sf, const uint V, const double iZ, const double iZ2, double *eR, double *partial, const Float shift)
{
	uint idx = Sf + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	double tmp[10] = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };

	if	(idx < V)
		energyCoreGpu<Float>(idx, m, v, Lx, Sf, iZ, iZ2, tmp, shift);

	reduction<BLSIZE,double>   (eR, tmp, partial);
}

int	energyGpu	(const void * __restrict__ m, const void * __restrict__ v, double *z, const double delta2, const double LL, const double nQcd, const double shift,
			 const uint Lx, const uint Lz, const uint V, const uint S, FieldPrecision precision, double *eR, cudaStream_t &stream)
{
	const uint Vm = V+S;
	const uint Lz2 = V/(Lx*Lx);
	dim3  gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3  blockSize(BLSIZE,1,1);
	const int nBlocks = gridSize.x*gridSize.y;

	const double zR   = *z;
	double *tR, *partial;

	if ((cudaMalloc(&tR, sizeof(double)*10) != cudaSuccess) || (cudaMalloc(&partial, sizeof(double)*10*nBlocks*4) != cudaSuccess))
	{
		return -1;
	}

	if (precision == FIELD_DOUBLE)
	{
		const double iZ  = 1./zR;
		const double iZ2 = iZ*iZ;
		energyKernel<<<gridSize,blockSize,0,stream>>> (static_cast<const complex<double>*>(m), static_cast<const complex<double>*>(v), Lx, S, Vm, iZ, iZ2, tR, partial, shift);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float iZ = 1./zR;
		const float iZ2 = iZ*iZ;
		energyKernel<<<gridSize,blockSize,0,stream>>> (static_cast<const complex<float>*>(m), static_cast<const complex<float>*>(v), Lx, S, Vm, iZ, iZ2, tR, partial, (float) shift);
	}

	cudaDeviceSynchronize();

	cudaMemcpy(eR, tR, sizeof(double)*10, cudaMemcpyDeviceToHost);
	cudaFree(tR); cudaFree(partial);

	const double o2 = 0.25/delta2;
	const double zQ = axionmass2(zR, nQcd, zthres, zrestore)*zR*zR;//9.*pow(zR, nQcd+2.);
	const double lZ = 0.25*LL*zR*zR;

	eR[0] *= o2;
	eR[1] *= o2;
	eR[2] *= o2;
	eR[3] *= o2;
	eR[4] *= o2;
	eR[5] *= o2;
	eR[6] *= lZ;
	eR[7] *= zQ;
	eR[8] *= .5;
	eR[9] *= .5;

	return	0;
}
/*
int	energyThetaGpu	(const void * __restrict__ m, const void * __restrict__ v, double *z, const double delta2, const double LL, const double nQcd,
			 const uint Lx, const uint Lz, const uint V, const uint Vt, const uint S, FieldPrecision precision, double *eR, cudaStream_t &stream)
{
	return	1;
}*/
