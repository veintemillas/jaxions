#include "complexGpu.cuh"
#include "utils/index.cuh"

#include "enum-field.h"

#include "scalar/varNQCD.h"
#include "utils/parse.h"

#include "utils/reduceGpu.cuh"

#define	BLSIZE 512

using namespace gpuCu;
using namespace indexHelper;

template<const VqcdType VQcd, typename Float>
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

	tmp = m[idx];
	tp2 = tmp - shift;

	Float mod = tmp.real()*tmp.real() + tmp.imag()*tmp.imag();
	Float md2 = tp2.real()*tp2.real() + tp2.imag()*tp2.imag();
	Float mFac = iZ2*mod;
	Float mFc2 = iZ2*md2;
	Float iMod = 1./mod;

	mPX = (m[idxPx]  - tmp)*conj(tmp)*iMod;
	mPY = (m[idxPy]  - tmp)*conj(tmp)*iMod;
	mPZ = (m[idx+Sf] - tmp)*conj(tmp)*iMod;
	mMX = (m[idxMx]  - tmp)*conj(tmp)*iMod;
	mMY = (m[idxMy]  - tmp)*conj(tmp)*iMod;
	mMZ = (m[idx-Sf] - tmp)*conj(tmp)*iMod;
	vOm = v[idx-Sf]*conj(tmp)*iMod - gpuCu::complex<Float>(iZ, 0.);

	tR[RH_GRX] = (double) ((Float) (mFac*(mPX.real()*mPX.real() + mMX.real()*mMX.real())));
	tR[TH_GRX] = (double) ((Float) (mFac*(mPX.imag()*mPX.imag() + mMX.imag()*mMX.imag())));
	tR[RH_GRY] = (double) ((Float) (mFac*(mPY.real()*mPY.real() + mMY.real()*mMY.real())));
	tR[TH_GRY] = (double) ((Float) (mFac*(mPY.imag()*mPY.imag() + mMY.imag()*mMY.imag())));
	tR[RH_GRZ] = (double) ((Float) (mFac*(mPZ.real()*mPZ.real() + mMZ.real()*mMZ.real())));
	tR[TH_GRZ] = (double) ((Float) (mFac*(mPZ.imag()*mPZ.imag() + mMZ.imag()*mMZ.imag())));
	tR[RH_POT] = (double) ((Float) (mFc2 - 1.)*(mFc2 - 1.));
	tR[RH_KIN] = (double) ((Float) (mFac*vOm.real()*vOm.real()));
	tR[TH_KIN] = (double) ((Float) (mFac*vOm.imag()*vOm.imag()));

	switch (VQcd) {
		case	VQCD_1:
			tR[TH_POT] = (double) (((Float) 1.) - tp2.real()/sqrt(md2));
			break;
		case	VQCD_2:
			double smp = (double) (((Float) 1.) - tp2.real()*iZ);
			tR[TH_POT] = smp*smp;
	}
}

template<const VqcdType VQcd, typename Float>
__global__ void	energyKernel(const complex<Float> * __restrict__ m, const complex<Float> * __restrict__ v, const uint Lx, const uint Sf, const uint V, const double iZ, const double iZ2, double *eR, double *partial, const Float shift)
{
	uint idx = Sf + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	double tmp[10] = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };

	if	(idx < V)
		energyCoreGpu<VQcd,Float>(idx, m, v, Lx, Sf, iZ, iZ2, tmp, shift);

	reduction<BLSIZE,double,10>   (eR, tmp, partial);
}

int	energyGpu	(const void * __restrict__ m, const void * __restrict__ v, double *z, const double delta2, const double LL, const double nQcd, const double shift,
			 const VqcdType VQcd, const uint Lx, const uint Lz, const uint V, const uint S, FieldPrecision precision, double *eR, cudaStream_t &stream, const bool map)
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

		switch (VQcd) {
			case	VQCD_1:
				energyKernel<VQCD_1><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<double>*>(m), static_cast<const complex<double>*>(v), Lx, S, Vm, iZ, iZ2, tR, partial, shift);
				break;

			case	VQCD_2:
				energyKernel<VQCD_2><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<double>*>(m), static_cast<const complex<double>*>(v), Lx, S, Vm, iZ, iZ2, tR, partial, shift);
				break;
		}
	}
	else if (precision == FIELD_SINGLE)
	{
		const float iZ = 1./zR;
		const float iZ2 = iZ*iZ;

		switch (VQcd) {
			case	VQCD_1:
				energyKernel<VQCD_1><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<float>*>(m), static_cast<const complex<float>*>(v), Lx, S, Vm, iZ, iZ2, tR, partial, (float) shift);
				break;

			case	VQCD_2:
				energyKernel<VQCD_2><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<float>*>(m), static_cast<const complex<float>*>(v), Lx, S, Vm, iZ, iZ2, tR, partial, (float) shift);
				break;
		}
	}

	cudaDeviceSynchronize();

	cudaMemcpy(eR, tR, sizeof(double)*10, cudaMemcpyDeviceToHost);
	cudaFree(tR); cudaFree(partial);

	const double o2 = 0.25/delta2;
	const double zQ = axionmass2(zR, nQcd, zthres, zrestore)*zR*zR;//9.*pow(zR, nQcd+2.);
	const double lZ = 0.25*LL*zR*zR;

	eR[TH_GRX] *= o2;
	eR[TH_GRY] *= o2;
	eR[TH_GRZ] *= o2;
	eR[TH_KIN] *= .5;
	eR[TH_POT] *= zQ;
	eR[RH_GRX] *= o2;
	eR[RH_GRY] *= o2;
	eR[RH_GRZ] *= o2;
	eR[RH_KIN] *= .5;
	eR[RH_POT] *= lZ;

	return	0;
}
