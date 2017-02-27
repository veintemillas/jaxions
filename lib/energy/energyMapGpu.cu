#include "complexGpu.cuh"
#include "index.cuh"

#include "enum-field.h"
#include "cub/cub.cuh"

#define	BLSIZE 512

using namespace gpuCu;
using namespace indexHelper;

template<typename Float>
static __device__ __forceinline__ void	energyMapCoreGpu(const uint idx, const complex<Float> * __restrict__ m, const complex<Float> * __restrict__ v, complex<Float> * __restrict__ m2, const uint Lx, const uint Sf, const double iZ, const double iZ2, double *tR) //zQ, o2
{
	uint X[3], idxPx, idxPy, idxMx, idxMy;

	complex<Float> mDX, mDY, mDZ, tmp, vOm;

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

	tmp = m[idx];

	Float mod = tmp.real()*tmp.real() + tmp.imag()*tmp.imag();
	Float mFac = iZ2*mod;
	Float iMod = 1./mod;


	vOm = v[idx-Sf]*conj(m[idx])*iMod - gpuCu::complex<Float>(iZ, 0.);
	mDX = (m[idxPx]  - m[idxMx]) *conj(m[idx])*iMod;
	mDY = (m[idxPy]  - m[idxMy]) *conj(m[idx])*iMod;
	mDZ = (m[idx+Sf] - m[idx-Sf])*conj(m[idx])*iMod;


	m2[idx] = 0.5*((mFac*mDX.imag()*mDX.imag()) + (mFac*mDY.imag()*mDY.imag()) + (mFac*mDZ.imag()*mDZ.imag()))
		+ o2*(((Float) 1.) - tmp.real()*iZ) + (mFac*vOm.imag()*vOm.imag())*zQ;
}

template<typename Float>
__global__ void	energyKernel(const complex<Float> * __restrict__ m, const complex<Float> * __restrict__ v, complex<Float> * __restrict__ m2, const uint Lx, const uint Sf, const uint V, const double iZ, const double iZ2, double *eR, double *partial)
{
	uint idx = Sf + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	double tmp[10] = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };

	if	(idx < V)
		energyMapCoreGpu<Float>(idx, m, v, Lx, Sf, iZ, iZ2, tmp);
}

void	energyMapGpu	(const void * __restrict__ m, const void * __restrict__ v, void * __restrict__ m2, double *z, const double delta2, const double nQcd,
			 const uint Lx, const uint Lz, const uint V, const uint S, FieldPrecision precision, cudaStream_t &stream)
{
	const uint Vm = V+S;
	const uint Lz2 = V/(Lx*Lx);
	dim3  gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3  blockSize(BLSIZE,1,1);
	const int nBlocks = gridSize.x*gridSize.y;

	const double zR   = *z;

	if (precision == FIELD_DOUBLE)
	{
		const double iZ  = 1./zR;
		const double iZ2 = iZ*iZ;
		energyMapKernel<<<gridSize,blockSize,0,stream>>> (static_cast<const complex<double>*>(m), static_cast<const complex<double>*>(v), static_cast<complex<double>*>(m2), Lx, S, Vm, iZ, iZ2);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float iZ = 1./zR;
		const float iZ2 = iZ*iZ;
		energyMapKernel<<<gridSize,blockSize,0,stream>>> (static_cast<const complex<float>*>(m), static_cast<const complex<float>*>(v), static_cast<complex<float>*>(m2), Lx, S, Vm, iZ, iZ2);
	}

	cudaDeviceSynchronize();

	return;
}
