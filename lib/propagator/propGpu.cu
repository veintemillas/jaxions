#include "kernelParms.cuh"
#include "complexGpu.cuh"
#include "index.cuh"

#include "enum-field.h"

using namespace gpuCu;
using namespace indexHelper;

template<typename Float>
static __device__ __forceinline__ void	propagateCoreGpu(const uint idx, const complex<Float> * __restrict__ m, complex<Float> * __restrict__ v, complex<Float> * __restrict__ m2, const Float z2,
							 const Float zQ, const Float dzc, const Float dzd, const Float ood2, const Float LL, const uint Lx, const uint Sf)
{
	uint X[3], idxPx, idxPy, idxMx, idxMy;

	complex<Float> mel, a, tmp;

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
	mel = m[idxMx] + m[idxPx] + m[idxPy] + m[idxMy] + m[idx+Sf] + m[idx-Sf];

	a = (mel-((Float) 6.)*tmp)*ood2 + zQ - tmp*(((Float) LL)*(tmp.real()*tmp.real() + tmp.imag()*tmp.imag() - z2));

	mel = v[idx-Sf];
	mel += a*dzc;
	v[idx-Sf] = mel;
	mel *= dzd;
	tmp += mel;
	m2[idx] = tmp;
}

template<typename Float>
__global__ void	propagateKernel(const complex<Float> * __restrict__ m, complex<Float> * __restrict__ v, complex<Float> * __restrict__ m2, const Float z2, const Float zQ,
				const Float dzc, const Float dzd, const Float ood2, const Float LL, const uint Lx, const uint Sf, const uint Vo, const uint Vf)
{
	uint idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	if	(idx >= Vf)
		return;

	propagateCoreGpu<Float>(idx, m, v, m2, z2, zQ, dzc, dzd, ood2, LL, Lx, Sf);
}

void	propagateGpu(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, double *z, const double dz, const double c, const double d,
		     const double delta2, const double LL, const double nQcd, const uint Lx, const uint Lz, const uint Vo, const uint Vf, FieldPrecision precision, cudaStream_t &stream)
{
	#define	BLSIZE 512
	const uint Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3	  gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3	  blockSize(BLSIZE,1,1);

	if (precision == FIELD_DOUBLE)
	{
		const double dzc  = dz*c;
		const double dzd  = dz*d;
		const double zR   = *z;
		const double z2   = zR*zR;
		const double zQ   = 9.*pow(zR, nQcd+3.);
		const double ood2 = 1./delta2;
		propagateKernel<<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2, z2, zQ, dzc, dzd, ood2, (double) LL, Lx, Lx*Lx, Vo, Vf);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float dzc = dz*c;
		const float dzd = dz*d;
		const float zR = *z;
		const float z2 = zR*zR;
		const float zQ = 9.*powf(zR, nQcd+3.);
		const float ood2 = 1./delta2;
		propagateKernel<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2, z2, zQ, dzc, dzd, ood2, (float) LL, Lx, Lx*Lx, Vo, Vf);
	}
}

template<typename Float>
static __device__ void	__forceinline__ updateMCoreGpu(const uint idx, complex<Float> * __restrict__ m, const complex<Float> * __restrict__ v, const Float dzd, const uint Sf)
{
	complex<Float> mm = m[idx], vv = v[idx-Sf];

	mm += vv*dzd;
	m[idx] = mm;
}

template<typename Float>
static __device__ void __forceinline__	updateVCoreGpu(const uint idx, const complex<Float> * __restrict__ m, complex<Float> * __restrict__ v, const Float z2, const Float zQ,
						       const Float dzc, const Float ood2, const Float LL, const uint Lx, const uint Sf)
{
	uint X[3], idxMx, idxPx, idxMy, idxPy;

	complex<Float> mel, a, tmp;

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
	mel = m[idxMx] + m[idxPx] + m[idxPy] + m[idxMy] + m[idx+Sf] + m[idx-Sf];

	a = (mel-((Float) 6.)*tmp)*ood2 + zQ - tmp*(((Float) LL)*((tmp.real()*tmp.real() + tmp.imag()*tmp.imag())- z2));

	mel = v[idx-Sf];
	mel += a*dzc;
	v[idx-Sf] = mel;
}

template<typename Float>
__global__ void	updateMKernel(complex<Float> * __restrict__ m, const complex<Float> * __restrict__ v, const Float dzd, const uint Vo, const uint Vf, const uint Sf)
{
	uint idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	if	(idx >= Vf)
		return;

	updateMCoreGpu<Float>(idx, m, v, dzd, Sf);
}

template<typename Float>
__global__ void	updateVKernel(const complex<Float> * __restrict__ m, complex<Float> * __restrict__ v, const Float z2, const Float zQ, const Float dzc,
				const Float ood2, const Float LL, const uint Lx, const uint Lz, const uint Vo, const uint Vf)
{
	uint idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	if	(idx >= Vf)
		return;

	updateVCoreGpu<Float>(idx, m, v, z2, zQ, dzc, ood2, LL, Lx, Lz);
}

void	updateMGpu(void * __restrict__ m, const void * __restrict__ v, const double dz, const double d, const uint Lx, const uint Vo, const uint Vf, FieldPrecision precision, cudaStream_t &stream)
{
	#define	BSSIZE 512
	const uint Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3	gridSize((Lx*Lx+BSSIZE-1)/BSSIZE,Lz2,1);
	dim3	blockSize(BSSIZE,1,1);

	if (precision == FIELD_DOUBLE)
	{
		const double dzd  = dz*d;
		updateMKernel<<<gridSize,blockSize,0,stream>>> ((complex<double> *) m, (const complex<double> *) v, dzd, Vo, Vf, Lx*Lx);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float dzd  = dz*d;
		updateMKernel<<<gridSize,blockSize,0,stream>>> ((complex<float> *) m, (const complex<float> *) v, dzd, Vo, Vf, Lx*Lx);
	}
}

void	updateVGpu(const void * __restrict__ m, void * __restrict__ v, double *z, const double dz, const double c,
		     const double delta2, const double LL, const double nQcd, const uint Lx, const uint Lz, const uint Vo, const uint Vf, FieldPrecision precision, cudaStream_t &stream)
{
	#define	BLSIZE 512
	const uint Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3	gridSize((Lx*Lx+BSSIZE-1)/BSSIZE,Lz2,1);
	dim3	blockSize(BSSIZE,1,1);

	if (precision == FIELD_DOUBLE)
	{
		const double zR   = *z;
		const double z2   = zR*zR;
		const double zQ   = 9.*pow(zR, nQcd+3.);
		const double dzc  = dz*c;
		const double ood2 = 1./delta2;
		updateVKernel<<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, z2, zQ, dzc, ood2, (double) LL, Lx, Lx*Lx, Vo, Vf);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float zR   = *z;
		const float z2   = zR*zR;
		const float zQ   = 9.*powf(zR, nQcd+3.);
		const float dzc  = dz*c;
		const float ood2 = 1./delta2;
		updateVKernel<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, z2, zQ, dzc, ood2, (float) LL, Lx, Lx*Lx, Vo, Vf);
	}
}
