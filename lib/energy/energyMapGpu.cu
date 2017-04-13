#include "complexGpu.cuh"
#include "utils/index.cuh"

#include "enum-field.h"

#include "scalar/varNQCD.h"
#include "utils/parse.h"

#define	BLSIZE 512

using namespace gpuCu;
using namespace indexHelper;

template<const VqcdType VQcd, typename Float>
static __device__ __forceinline__ void	energyMapCoreGpu(const uint idx, const complex<Float> * __restrict__ m, const complex<Float> * __restrict__ v, complex<Float> * __restrict__ m2, const uint Lx, const uint Sf, const Float iZ, const Float iZ2, const Float zQ, const Float lZ, const Float o2, const Float shift)
{
	uint X[3], idxPx, idxPy, idxMx, idxMy, idxPz = idx + Sf, idxMz = idx - Sf;

	complex<Float> mPx, mPy, mPz, mMx, mMy, mMz, tmp, tp2, vOm;

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

	mPx = (m[idxPx] - tmp)*conj(tmp)*iMod;
	mPy = (m[idxPy] - tmp)*conj(tmp)*iMod;
	mPz = (m[idxPz] - tmp)*conj(tmp)*iMod;
	mMx = (m[idxMx] - tmp)*conj(tmp)*iMod;
	mMy = (m[idxMy] - tmp)*conj(tmp)*iMod;
	mMz = (m[idxMz] - tmp)*conj(tmp)*iMod;
	vOm = v[idxMz]*conj(tmp)*iMod - gpuCu::complex<Float>(iZ, 0.);

	Float	rP = o2*mFac*(mPx.real()*mPx.real() + mMx.real()*mMx.real() + mPy.real()*mPy.real() + mMy.real()*mMy.real() + mPz.real()*mPz.real() + mMz.real()*mMz.real())
		   + 0.5*mFac*vOm.real()*vOm.real() + lZ*((Float) (mFc2 - 1.)*(mFc2 - 1.));
	Float	iP = o2*mFac*(mPx.imag()*mPx.imag() + mMx.imag()*mMx.imag() + mPy.imag()*mPy.imag() + mMy.imag()*mMy.imag() + mPz.imag()*mPz.imag() + mMz.imag()*mMz.imag())
		   + 0.5*mFac*vOm.imag()*vOm.imag();

	switch	(VQcd) {
		case	VQCD_1:
			iP += zQ*(((Float) 1.) - tp2.real()/(sqrt(md2)));
			break;

		case	VQCD_2:
			md2 = ((Float) 1.) - tp2.real()*iZ;
			iP += zQ*md2*md2;
			break;
	}

	m2[idxMz] = complex<Float>(rP,iP);
}

template<const VqcdType VQcd, typename Float>
__global__ void	energyMapKernel(const complex<Float> * __restrict__ m, const complex<Float> * __restrict__ v, complex<Float> * __restrict__ m2, const uint Lx, const uint Sf, const uint V, const Float iZ, const Float iZ2, const Float zQ, const Float lZ, const Float o2, const Float shift)
{
	uint idx = Sf + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	if	(idx < V)
		energyMapCoreGpu<VQcd, Float>(idx, m, v, m2, Lx, Sf, iZ, iZ2, zQ, lZ, o2, shift);
}

void	energyMapGpu	(const void * __restrict__ m, const void * __restrict__ v, void * __restrict__ m2, double *z, const double delta2, const double nQcd, const double lambda,
			 const double shift, const VqcdType VQcd, const uint Lx, const uint Lz, const uint V, const uint S, FieldPrecision precision, cudaStream_t &stream)
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
		const double o2  = 0.25/delta2;
		const double zQ  = axionmass2(zR, nQcd, zthres, zrestore)*zR*zR;
		const double lZ  = 0.25*lambda*zR*zR;

		switch (VQcd) {
			case    VQCD_1:
				energyMapKernel<VQCD_1><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<double>*>(m), static_cast<const complex<double>*>(v),
											  static_cast<complex<double>*>(m2), Lx, S, Vm, iZ, iZ2, zQ, lZ, o2, shift);
				break;

			case    VQCD_2:
				energyMapKernel<VQCD_2><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<double>*>(m), static_cast<const complex<double>*>(v),
											  static_cast<complex<double>*>(m2), Lx, S, Vm, iZ, iZ2, zQ, lZ, o2, shift);
				break;
		}
	}
	else if (precision == FIELD_SINGLE)
	{
		const float iZ  = 1./zR;
		const float iZ2 = iZ*iZ;
		const float o2  = 0.25/delta2;
		const float zQ  = axionmass2(zR, nQcd, zthres, zrestore)*zR*zR;
		const float lZ  = 0.25*lambda*zR*zR;

		switch (VQcd) {
			case    VQCD_1:
				energyMapKernel<VQCD_1><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<float>*>(m), static_cast<const complex<float>*>(v),
											  static_cast<complex<float>*>(m2), Lx, S, Vm, iZ, iZ2, zQ, lZ, o2, (float) shift);
				break;

			case    VQCD_2:
				energyMapKernel<VQCD_2><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<float>*>(m), static_cast<const complex<float>*>(v),
											  static_cast<complex<float>*>(m2), Lx, S, Vm, iZ, iZ2, zQ, lZ, o2, (float) shift);
				break;
		}
	}

	cudaDeviceSynchronize();

	return;
}
