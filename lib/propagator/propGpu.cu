#include "kernelParms.cuh"
#include "complexGpu.cuh"
#include "utils/index.cuh"

#include "enum-field.h"

//#include "utils/parse.h"
//#include "scalar/varNQCD.h"

#define	BLSIZE 256
#define	BSSIZE 256

using namespace gpuCu;
using namespace indexHelper;

template<typename Float, const VqcdType VQcd>
static __device__ __forceinline__ void	propagateCoreGpu(const uint idx, const complex<Float> * __restrict__ m, complex<Float> * __restrict__ v, complex<Float> * __restrict__ m2,
							 const Float z, const Float z2, const Float z4, const Float zQ, const Float gFac, const Float eps, const Float dp1, const Float dp2,
							 const Float dzc, const Float dzd, const Float ood2, const Float LL, const uint Lx, const uint Sf)
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

	Float pot = tmp.real()*tmp.real() + tmp.imag()*tmp.imag();

	switch (VQcd & VQCD_TYPE) {
		case	VQCD_PQ_ONLY:
		a = (mel-((Float) 6.)*tmp)*ood2 - tmp*(((Float) LL)*(pot - z2));
		break;

		case	VQCD_1:
		a = (mel-((Float) 6.)*tmp)*ood2 + zQ - tmp*(((Float) LL)*(pot - z2));
		break;

		case	VQCD_1_PQ_2:
		a = (mel-((Float) 6.)*tmp)*ood2 + zQ - tmp*pot*(((Float) LL)*(pot*pot - z4))*((Float) 2.)/z4;
		break;

		case	VQCD_2:
		a = (mel-((Float) 6.)*tmp)*ood2 - (tmp - z)*zQ - tmp*(((Float) LL)*(pot - z2));
		break;

		case	VQCD_NONE:
		a = 0.;
	}

	mel = v[idx-Sf];

	switch (VQcd & VQCD_DAMP) {
		case	VQCD_NONE:
		mel += a*dzc;
		break;

		case	VQCD_DAMP_RHO:
		{
			Float vec  = tmp.real()*mel.real() + tmp.imag()*mel.imag();
			Float vea  = tmp.real()*a.real()   + tmp.imag()*a.imag();
			a   += tmp*gFac;
			mel += a*dzc - (tmp/pot)*eps*(((Float) 2.)*vec + vea*dzc);
		}
		break;

		case	VQCD_DAMP_ALL:
		mel = mel*dp2 + a*dp1*dzc;
		break;
	}


	if (VQcd & VQCD_EVOL_RHO) {
		Float kReal = tmp.real()*mel.real() + tmp.imag()*mel.imag();
		mel *= kReal/pot;
	}

	v[idx-Sf] = mel;
	mel *= dzd;
	tmp += mel;
	m2[idx] = tmp;
}

template<typename Float, const VqcdType VQcd>
__global__ void	propagateKernel(const complex<Float> * __restrict__ m, complex<Float> * __restrict__ v, complex<Float> * __restrict__ m2, const Float z, const Float z2, const Float z4,
				const Float zQ, const Float gFac, const Float eps, const Float dp1, const Float dp2, const Float dzc, const Float dzd, const Float ood2, const Float LL,
				const uint Lx, const uint Sf, const uint Vo, const uint Vf)
{
	//uint idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));
	uint idx = Vo + (threadIdx.x + blockDim.x*blockIdx.x) + Sf*(threadIdx.y + blockDim.y*blockIdx.y);

	if	(idx >= Vf)
		return;

	propagateCoreGpu<Float, VQcd>(idx, m, v, m2, z, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, LL, Lx, Sf);
}

void	propagateGpu(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, double *z, const double dz, const double c, const double d, const double ood2,
		     const double LL, const double aMass2, const double gamma, const uint Lx, const uint Lz, const uint Vo, const uint Vf, const VqcdType VQcd, FieldPrecision precision,
		     const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream)
{
/*
	const uint Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3	  gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3	  blockSize(BLSIZE,1,1);
*/
	const uint Sf  = Lx*Lx;
	const uint Lz2 = (Vf-Vo)/Sf;
	dim3 gridSize((Sf+xBlock-1)/xBlock, (Lz2+yBlock-1)/yBlock, 1);
	dim3 blockSize(xBlock, yBlock, 1);

	if (precision == FIELD_DOUBLE)
	{
		const double dzc  = dz*c;
		const double dzd  = dz*d;
		const double zR   = *z;
		const double z2   = zR*zR;
		const double z4   = z2*z2;
		const double zQ   = aMass2*z2*zR;
		const double gFp1 = sqrt(ood2)*gamma;
		const double gFac = gFp1/zR;
		const double gFp2 = gFp1*dzc/2.;
		const double eps  = gFp2/(1. + gFp2);
		const double dp1  =   1./(1. + gFp2);
		const double dp2  = (1. - gFp2)*dp1;

		switch (VQcd) {
			case	VQCD_PQ_ONLY:
			propagateKernel<double, VQCD_PQ_ONLY>		<<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1:
			propagateKernel<double, VQCD_1>		<<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2:
			propagateKernel<double, VQCD_1_PQ_2>	<<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2:
			propagateKernel<double, VQCD_2>		<<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_RHO:
			propagateKernel<double, VQCD_1_RHO>	<<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_RHO:
			propagateKernel<double, VQCD_1_PQ_2_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_RHO:
			propagateKernel<double, VQCD_2_RHO>	<<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_DRHO:
			propagateKernel<double, VQCD_1_DRHO>	<<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_DRHO:
			propagateKernel<double, VQCD_1_PQ_2_DRHO><<<gridSize,blockSize,0,stream>>>((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_DRHO:
			propagateKernel<double, VQCD_2_DRHO>	<<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_DALL:
			propagateKernel<double, VQCD_1_DALL>	<<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_DALL:
			propagateKernel<double, VQCD_1_PQ_2_DALL><<<gridSize,blockSize,0,stream>>>((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_DALL:
			propagateKernel<double, VQCD_2_DALL>	<<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_DRHO_RHO:
			propagateKernel<double, VQCD_1_DRHO_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_DRHO_RHO:
			propagateKernel<double, VQCD_1_PQ_2_DRHO_RHO><<<gridSize,blockSize,0,stream>>>((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_DRHO_RHO:
			propagateKernel<double, VQCD_2_DRHO_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_DALL_RHO:
			propagateKernel<double, VQCD_1_DALL_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_DALL_RHO:
			propagateKernel<double, VQCD_1_PQ_2_DALL_RHO><<<gridSize,blockSize,0,stream>>>((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_DALL_RHO:
			propagateKernel<double, VQCD_2_DALL_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			default:
			return;
		}
	} else if (precision == FIELD_SINGLE) {
		const float dzc  = dz*c;
		const float dzd  = dz*d;
		const float zR   = *z;
		const float z2   = zR*zR;
		const float z4   = z2*z2;
		const float zQ   = aMass2*z2*zR;//(float) axionmass2(*z, nQcd, zthres, zrestore)*z2*zR;
		const float gFp1 = sqrt(ood2)*gamma;
		const float gFac = gFp1/zR;
		const float gFp2 = gFp1*dzc/2.;
		const float eps  = gFp2/(1. + gFp2);
		const float dp1  =   1./(1. + gFp2);
		const float dp2  = (1. - gFp2)*dp1;

		switch (VQcd) {
			case	VQCD_PQ_ONLY:
			propagateKernel<float, VQCD_PQ_ONLY>		<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1:
			propagateKernel<float, VQCD_1>		<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2:
			propagateKernel<float, VQCD_1_PQ_2>	<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2:
			propagateKernel<float, VQCD_2>		<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_RHO:
			propagateKernel<float, VQCD_1_RHO>	<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_RHO:
			propagateKernel<float, VQCD_1_PQ_2_RHO>	<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_RHO:
			propagateKernel<float, VQCD_2_RHO>	<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_DRHO:
			propagateKernel<float, VQCD_1_DRHO>	<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_DRHO:
			propagateKernel<float, VQCD_1_PQ_2_DRHO><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_DRHO:
			propagateKernel<float, VQCD_2_DRHO>	<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_DALL:
			propagateKernel<float, VQCD_1_DALL>	<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_DALL:
			propagateKernel<float, VQCD_1_PQ_2_DALL><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_DALL:
			propagateKernel<float, VQCD_2_DALL>	<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_DRHO_RHO:
			propagateKernel<float, VQCD_1_DRHO_RHO>	<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_DRHO_RHO:
			propagateKernel<float, VQCD_1_PQ_2_DRHO_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_DRHO_RHO:
			propagateKernel<float, VQCD_2_DRHO_RHO>	<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_DALL_RHO:
			propagateKernel<float, VQCD_1_DALL_RHO>	<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_DALL_RHO:
			propagateKernel<float, VQCD_1_PQ_2_DALL_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_DALL_RHO:
			propagateKernel<float, VQCD_2_DALL_RHO>	<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2,
												  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			default:
			return;
		}
	}
}

template<typename Float>
static __device__ void	__forceinline__ updateMCoreGpu(const uint idx, complex<Float> * __restrict__ m, const complex<Float> * __restrict__ v, const Float dzd, const uint Sf)
{
	complex<Float> mm = m[idx], vv = v[idx-Sf];

	mm += vv*dzd;
	m[idx] = mm;
}

template<typename Float, const VqcdType VQcd>
static __device__ void __forceinline__	updateVCoreGpu(const uint idx, const complex<Float> * __restrict__ m, complex<Float> * __restrict__ v, const Float z, const Float z2,
						       const Float z4, const Float zQ, const Float gFac, const Float eps, const Float dp1, const Float dp2, const Float dzc,
						       const Float ood2, const Float LL, const uint Lx, const uint Sf)
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

	Float pot = tmp.real()*tmp.real() + tmp.imag()*tmp.imag();

	switch (VQcd & VQCD_TYPE) {
		case	VQCD_PQ_ONLY:
		a = (mel-((Float) 6.)*tmp)*ood2 - tmp*(((Float) LL)*(pot - z2));
		break;

		case	VQCD_1:
		a = (mel-((Float) 6.)*tmp)*ood2 + zQ - tmp*(((Float) LL)*(pot - z2));
		break;

		case	VQCD_1_PQ_2:
		a = (mel-((Float) 6.)*tmp)*ood2 + zQ - tmp*pot*(((Float) LL)*(pot*pot - z4))*((Float) 2.)/z4;
		break;

		case	VQCD_2:
		a = (mel-((Float) 6.)*tmp)*ood2 - zQ*(tmp - z) - tmp*(((Float) LL)*(pot - z2));
		break;
	}

	mel = v[idx-Sf];

	switch (VQcd & VQCD_DAMP) {
		case	VQCD_NONE:
		mel += a*dzc;
		break;

		case	VQCD_DAMP_RHO:
		{
			Float vec  = tmp.real()*mel.real() + tmp.imag()*mel.imag();
			Float vea  = tmp.real()*a.real()   + tmp.imag()*a.imag();
			a   += tmp*gFac;
			mel += a*dzc - (tmp/pot)*eps*(((Float) 2.)*vec + vea*dzc);
		}
		break;

		case	VQCD_DAMP_ALL:
		mel = mel*dp2 + a*dp1*dzc;
		break;
	}


	if (VQcd & VQCD_EVOL_RHO) {
		Float kReal = tmp.real()*mel.real() + tmp.imag()*mel.imag();
		mel *= kReal/pot;
	}

	v[idx-Sf] = mel;
}

template<typename Float>
__global__ void	updateMKernel(complex<Float> * __restrict__ m, const complex<Float> * __restrict__ v, const Float dzd, const uint Lx, const uint Sf, const uint Vo, const uint Vf)
{
	//uint idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));
	uint idx = Vo + (threadIdx.x + blockDim.x*blockIdx.x) + Sf*(threadIdx.y + blockDim.y*blockIdx.y);

	if	(idx >= Vf)
		return;

	updateMCoreGpu<Float>(idx, m, v, dzd, Sf);
}

template<typename Float, const VqcdType VQcd>
__global__ void	updateVKernel(const complex<Float> * __restrict__ m, complex<Float> * __restrict__ v, const Float z, const Float z2, const Float z4, const Float zQ, const Float gFac,
			      const Float eps, const Float dp1, const Float dp2, const Float dzc, const Float ood2, const Float LL, const uint Lx, const uint Sf, const uint Vo,
			      const uint Vf)
{
	//uint idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));
	uint idx = Vo + (threadIdx.x + blockDim.x*blockIdx.x) + Sf*(threadIdx.y + blockDim.y*blockIdx.y);

	if	(idx >= Vf)
		return;

	updateVCoreGpu<Float, VQcd>(idx, m, v, z, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, LL, Lx, Sf);
}

void	updateMGpu(void * __restrict__ m, const void * __restrict__ v, const double dz, const double d, const uint Lx, const uint Vo, const uint Vf, FieldPrecision precision,
		   const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream)
{
/*
	const uint Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3	gridSize((Lx*Lx+BSSIZE-1)/BSSIZE,Lz2,1);
	dim3	blockSize(BSSIZE,1,1);
*/
	const uint Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3 gridSize((Lx*Lx+xBlock-1)/xBlock, (Lz2+yBlock-1)/yBlock, 1);
	dim3 blockSize(xBlock, yBlock, 1);

	if (precision == FIELD_DOUBLE)
	{
		const double dzd  = dz*d;
		updateMKernel<<<gridSize,blockSize,0,stream>>> ((complex<double>*) m, (const complex<double>*) v, dzd, Lx, Lx*Lx, Vo, Vf);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float dzd  = dz*d;
		updateMKernel<<<gridSize,blockSize,0,stream>>> ((complex<float> *) m, (const complex<float> *) v, dzd, Lx, Lx*Lx, Vo, Vf);
	}
}

void	updateVGpu(const void * __restrict__ m, void * __restrict__ v, double *z, const double dz, const double c, const double ood2, const double LL, const double aMass2,
		   const double gamma, const uint Lx, const uint Lz, const uint Vo, const uint Vf, const VqcdType VQcd, FieldPrecision precision,
		   const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream)
{
/*
	const uint Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3	gridSize((Lx*Lx+BSSIZE-1)/BSSIZE,Lz2,1);
	dim3	blockSize(BSSIZE,1,1);
*/
	const uint Sf  = Lx*Lx;
	const uint Lz2 = (Vf-Vo)/Sf;
	dim3 gridSize((Sf+xBlock-1)/xBlock, (Lz2+yBlock-1)/yBlock, 1);
	dim3 blockSize(xBlock, yBlock, 1);

	if (precision == FIELD_DOUBLE)
	{
		const double zR   = *z;
		const double z2   = zR*zR;
		const double z4   = z2*z2;
		const double zQ   = aMass2*z2*zR;
		const double dzc  = dz*c;
		const double gFp1 = sqrt(ood2)*gamma;
		const double gFac = gFp1/zR;
		const double gFp2 = gFp1*dzc/2.;
		const double eps  = gFp2/(1. + gFp2);
		const double dp1  =   1./(1. + gFp2);
		const double dp2  = (1. - gFp2)*dp1;

		switch (VQcd) {
			case	VQCD_1:
			updateVKernel<double, VQCD_1><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_PQ_ONLY:
			updateVKernel<double, VQCD_PQ_ONLY><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2:
			updateVKernel<double, VQCD_1_PQ_2><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											     zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc,  ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2:
			updateVKernel<double, VQCD_2><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											  	zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_RHO:
			updateVKernel<double, VQCD_1_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											  	zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_RHO:
			updateVKernel<double, VQCD_1_PQ_2_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											  	zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_RHO:
			updateVKernel<double, VQCD_2_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											  	zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_DRHO:
			updateVKernel<double, VQCD_1_DRHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											  	zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_DRHO:
			updateVKernel<double, VQCD_1_PQ_2_DRHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											  	zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_DRHO:
			updateVKernel<double, VQCD_2_DRHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											  	zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_DALL:
			updateVKernel<double, VQCD_1_DALL><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											  	zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_DALL:
			updateVKernel<double, VQCD_1_PQ_2_DALL><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											  	zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_DALL:
			updateVKernel<double, VQCD_2_DALL><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											  	zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_DRHO_RHO:
			updateVKernel<double, VQCD_1_DRHO_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											  	zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_DRHO_RHO:
			updateVKernel<double, VQCD_1_PQ_2_DRHO_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											  	zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_DRHO_RHO:
			updateVKernel<double, VQCD_2_DRHO_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											  	zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_DALL_RHO:
			updateVKernel<double, VQCD_1_DALL_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											  	zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_DALL_RHO:
			updateVKernel<double, VQCD_1_PQ_2_DALL_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											  	zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_DALL_RHO:
			updateVKernel<double, VQCD_2_DALL_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v,
											  	zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (double) LL, Lx, Sf, Vo, Vf);
			break;

			default:
			return;
		}
	} else if (precision == FIELD_SINGLE) {
		const float zR   = *z;
		const float z2   = zR*zR;
		const float z4   = z2*z2;
		const float zQ   = aMass2*z2*zR;//xionmass2((double) zR, nQcd, 1.5 , 3.)*zR*zR*zR;
		const float dzc  = dz*c;
		const float gFp1 = sqrt(ood2)*gamma;
		const float gFac = gFp1/zR;
		const float gFp2 = gFp1*dzc/2.;
		const float eps  = gFp2/(1. + gFp2);
		const float dp1  =   1./(1. + gFp2);
		const float dp2  = (1. - gFp2)*dp1;

		switch (VQcd) {

			case	VQCD_1:
			updateVKernel<float, VQCD_1><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc,
											  ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_PQ_ONLY:
			updateVKernel<float, VQCD_PQ_ONLY><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc,
											  ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2:
			updateVKernel<float, VQCD_1_PQ_2><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc,
											       ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2:
			updateVKernel<float, VQCD_2><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v,
											  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_RHO:
			updateVKernel<float, VQCD_1_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v,
											  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_RHO:
			updateVKernel<float, VQCD_1_PQ_2_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v,
											  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_RHO:
			updateVKernel<float, VQCD_2_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v,
											  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_DRHO:
			updateVKernel<float, VQCD_1_DRHO><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v,
											  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_DRHO:
			updateVKernel<float, VQCD_1_PQ_2_DRHO><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v,
											  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_DRHO:
			updateVKernel<float, VQCD_2_DRHO><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v,
											  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_DALL:
			updateVKernel<float, VQCD_1_DALL><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v,
											  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_DALL:
			updateVKernel<float, VQCD_1_PQ_2_DALL><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v,
											  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_DALL:
			updateVKernel<float, VQCD_2_DALL><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v,
											  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_DRHO_RHO:
			updateVKernel<float, VQCD_1_DRHO_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v,
											  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_DRHO_RHO:
			updateVKernel<float, VQCD_1_PQ_2_DRHO_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v,
											  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_DRHO_RHO:
			updateVKernel<float, VQCD_2_DRHO_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v,
											  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_DALL_RHO:
			updateVKernel<float, VQCD_1_DALL_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v,
											  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_1_PQ_2_DALL_RHO:
			updateVKernel<float, VQCD_1_PQ_2_DALL_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v,
											  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			case	VQCD_2_DALL_RHO:
			updateVKernel<float, VQCD_2_DALL_RHO><<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v,
											  zR, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, (float) LL, Lx, Sf, Vo, Vf);
			break;

			default:
			return;
		}
	}
}
