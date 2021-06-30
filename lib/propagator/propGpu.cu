#include "kernelParms.cuh"
#include "complexGpu.cuh"
#include "utils/index.cuh"

#include "enum-field.h"
#include "propagator/prop-def-mac.h"

//#include "utils/parse.h"
//#include "scalar/varNQCD.h"

#define	BLSIZE 256
#define	BSSIZE 256

using namespace gpuCu;
using namespace indexHelper;

template<typename Float, const VqcdType VQcd>
static __device__ __forceinline__ void	propagateCoreGpu(const uint idx, const complex<Float> * __restrict__ m, complex<Float> * __restrict__ v, complex<Float> * __restrict__ m2,
							 const Float z, const Float z2, const Float z4, const Float zQ, const Float gFac, const Float eps, const Float dp1, const Float dp2,
							 const Float dzc, const Float dzd, const Float *ood2, const Float LL, const uint Lx, const uint Sf, const uint NN)
{
	uint X[3], idxPx, idxPy, idxMx, idxMy;

	complex<Float> mel, a, tmp, zN;

	switch	(VQcd & V_QCD) {
			case	V_QCD2:
			zN = (Float) (zQ/z)/2 * complex<Float>(1,-1);
			break;

			default:
			case	V_QCDC:
			case	V_QCDL:
			zN = (Float) (zQ*z) * complex<Float>(1,-1);
			break;
	}

	idx2Vec(idx, X, Lx);

	mel = 0;
	tmp = m[idx];
	for (size_t nv=1; nv <= NN; nv++)
	{
		if (X[0] + nv >= Lx)
			idxPx = idx + nv - Lx;
		else
			idxPx = idx + nv;

		if (X[0] < nv)
			idxMx = idx + Lx - nv;
		else
			idxMx = idx - nv;

		if (X[1] + nv >= Lx)
			idxPy = idx + nv*Lx - Sf;
		else
			idxPy = idx + nv*Lx;

		if (X[1] < nv)
			idxMy = idx + Sf - nv*Lx;
		else
			idxMy = idx - nv*Lx;

		mel += (m[idxMx] + m[idxPx] + m[idxPy] + m[idxMy] + m[idx+nv*Sf] + m[idx-nv*Sf] - ((Float) 6.)*tmp)*ood2[nv-1];
	}


	Float pot = tmp.real()*tmp.real() + tmp.imag()*tmp.imag();

	switch (VQcd & V_PQ) {
		default:
		case	V_PQ1:
			a  = mel - tmp*(((Float) LL)*(pot - z2));
			break;
		case	V_PQ2:
			a  = mel - tmp*pot*(((Float) LL)*(pot*pot - z4))*((Float) 2.)/z4;
			break;
		case	V_NONE:
			a  = mel ;
			break;
	}

	switch (VQcd & V_QCD) {
		default:
		case	V_QCD1:
			a  += complex<Float>(zQ,0);
			break;
		case	V_QCDV:
			a  += (z - tmp)*zQ;
			break;
		case	V_QCD2:
			a  -= tmp*zN;
			break;
		case	V_QCDC:
			{
			Float pota = 1/pot*sqrt(pot);
			mel = complex<Float>(tmp.imag()*tmp.imag(),tmp.imag()*tmp.real());
			a  -= pota*mel*zN;
			}
			break;
		case	V_QCDL:
			mel = complex<Float>(tmp.imag(),tmp.real());
			a += mel*atan2(tmp.real(),tmp.imag())*zN/pot;
			break;
		case	V_QCD0:
			break;
	}

	mel = v[idx-Sf];

	switch (VQcd & V_DAMP) {
		case	V_NONE:
		mel += a*dzc;
		break;

		case	V_DAMP_RHO:
		{
			Float vec  = tmp.real()*mel.real() + tmp.imag()*mel.imag();
			Float vea  = tmp.real()*a.real()   + tmp.imag()*a.imag();
			a   += tmp*gFac;
			mel += a*dzc - (tmp/pot)*eps*(((Float) 2.)*vec + vea*dzc);
		}
		break;

		case	V_DAMP_ALL:
		mel = mel*dp2 + a*dp1*dzc;
		break;
	}


	if (VQcd & V_EVOL_RHO) {
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
				const Float zQ, const Float gFac, const Float eps, const Float dp1, const Float dp2, const Float dzc, const Float dzd, const Float *ood2, const Float LL,
				const uint Lx, const uint Sf, const uint Vo, const uint Vf, const uint NN)
{
	//uint idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));
	uint idx = Vo + (threadIdx.x + blockDim.x*blockIdx.x) + Sf*(threadIdx.y + blockDim.y*blockIdx.y);

	if	(idx >= Vf)
		return;

	propagateCoreGpu<Float, VQcd>(idx, m, v, m2, z, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, LL, Lx, Sf, NN);
}

void	propagateGpu(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, PropParms ppar, const double dz, const double c, const double d,
				const uint Vo, const uint Vf, const VqcdType VQcd, FieldPrecision precision, const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream)
{
	if (Vo>Vf)
		return ;
/*
	const uint Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3	  gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3	  blockSize(BLSIZE,1,1);
*/
	const uint Lx    = ppar.Lx;
	const uint Sf  = Lx*Lx;
	const uint Lz2 = (Vf-Vo)/Sf;
	dim3 gridSize((Sf+xBlock-1)/xBlock, (Lz2+yBlock-1)/yBlock, 1);
	dim3 blockSize(xBlock, yBlock, 1);

	const uint NN    = ppar.Ng;

	if (precision == FIELD_DOUBLE)
	{
		const double dzc  = dz*c;
		const double dzd  = dz*d;
		const double zR   = ppar.R;
		const double z2   = zR*zR;
		const double z4   = z2*z2;
		const double zQ   = ppar.massA2*z2*zR;
		const double LL   = ppar.lambda;
		const double gFp1 = ppar.gamma/zR;
		const double gFac = gFp1/zR;
		const double gFp2 = gFp1*dzc/2.;
		const double eps  = gFp2/(1. + gFp2);
		const double dp1  =   1./(1. + gFp2);
		const double dp2  = (1. - gFp2)*dp1;

		double pood2[NN] ;
		for (int i =0; i<NN; i++)
			pood2[i] = (ppar.PC)[i]*ppar.ood2a;
		const double *ood2 = &(pood2[0]);

		switch (VQcd) {

			DEFALLPROPTEM_K_GPU(double)

			default:
			return;
		}
	} else if (precision == FIELD_SINGLE) {
		const float dzc  = dz*c;
		const float dzd  = dz*d;
		const float zR   = ppar.R;
		const float z2   = zR*zR;
		const float z4   = z2*z2;
		const float zQ   = ppar.massA2*z2*zR;//(float) axionmass2(*z, nQcd, zthres, zrestore)*z2*zR;
		const float LL   = ppar.lambda;
		const float gFp1 = ppar.gamma/zR;
		const float gFac = gFp1/zR;
		const float gFp2 = gFp1*dzc/2.;
		const float eps  = gFp2/(1. + gFp2);
		const float dp1  =   1./(1. + gFp2);
		const float dp2  = (1. - gFp2)*dp1;

		float  food2[NN] ;
		for (int i =0; i<NN; i++)
			food2[i] = (float) (ppar.PC)[i]*ppar.ood2a;
		const float *ood2 = &(food2[0]);

		switch (VQcd) {

			DEFALLPROPTEM_K_GPU(float)

			default:
			return;
		}
	}
}

template<typename cFloat, typename Float>
static __device__ void	__forceinline__ updateMCoreGpu(const uint idx, cFloat * __restrict__ m, const cFloat * __restrict__ v, const Float dzd, const uint Sf)
{
	cFloat mm = m[idx], vv = v[idx-Sf];

	mm += vv*dzd;
	m[idx] = mm;
}

template<typename Float, const VqcdType VQcd>
static __device__ void __forceinline__	updateVCoreGpu(const uint idx, const complex<Float> * __restrict__ m, complex<Float> * __restrict__ v, const Float z, const Float z2,
						       const Float z4, const Float zQ, const Float gFac, const Float eps, const Float dp1, const Float dp2, const Float dzc,
						       const Float *ood2, const Float LL, const uint Lx, const uint Sf, const uint NN)
{
	uint X[3], idxMx, idxPx, idxMy, idxPy;

	complex<Float> mel, a, tmp, zN;

	switch	(VQcd & V_QCD) {
			case	V_QCD2:
			zN = (Float) (zQ/z)/2 * complex<Float>(1,-1);
			break;

			default:
			case	V_QCDC:
			case	V_QCDL:
			zN = (Float) (zQ*z) * complex<Float>(1,-1);
			break;
	}

	idx2Vec(idx, X, Lx);

	mel = 0;
	tmp = m[idx];

	for (size_t nv=1; nv <= NN; nv++)
	{
		if (X[0] + nv >= Lx)
			idxPx = idx + nv - Lx;
		else
			idxPx = idx + nv;

		if (X[0] < nv)
			idxMx = idx + Lx - nv;
		else
			idxMx = idx - nv;

		if (X[1] + nv >= Lx)
			idxPy = idx + nv*Lx - Sf;
		else
			idxPy = idx + nv*Lx;

		if (X[1] < nv)
			idxMy = idx + Sf - nv*Lx;
		else
			idxMy = idx - nv*Lx;

		mel += (m[idxMx] + m[idxPx] + m[idxPy] + m[idxMy] + m[idx+nv*Sf] + m[idx-nv*Sf] - ((Float) 6.)*tmp)*ood2[nv-1];
	}


	Float pot = tmp.real()*tmp.real() + tmp.imag()*tmp.imag();

	switch (VQcd & V_PQ) {
		default:
		case	V_PQ1:
			a  = mel - tmp*(((Float) LL)*(pot - z2));
			break;
		case	V_PQ2:
			a  = mel - tmp*pot*(((Float) LL)*(pot*pot - z4))*((Float) 2.)/z4;
			break;
		case	V_NONE:
			a  = mel;
			break;
	}

	switch (VQcd & V_QCD) {
		default:
		case	V_QCD1:
			a  += complex<Float>(zQ,0);
			break;
		case	V_QCDV:
			a  += (z - tmp)*zQ;
			break;
		case	V_QCD2:
			a  -= tmp*zN;
			break;
		case	V_QCDC:
			{
			Float pota = 1/pot*sqrt(pot);
			mel = complex<Float>(tmp.imag()*tmp.imag(),tmp.imag()*tmp.real());
			a  -= pota*mel*zN;
			}
			break;
		case	V_QCDL:
			mel = complex<Float>(tmp.imag(),tmp.real());
			a += mel*atan2(tmp.real(),tmp.imag())*zN/pot;
			break;
		case	V_QCD0:
			break;
	}

	mel = v[idx-Sf];

	switch (VQcd & V_DAMP) {
		case	V_NONE:
		mel += a*dzc;
		break;

		case	V_DAMP_RHO:
		{
			Float vec  = tmp.real()*mel.real() + tmp.imag()*mel.imag();
			Float vea  = tmp.real()*a.real()   + tmp.imag()*a.imag();
			a   += tmp*gFac;
			mel += a*dzc - (tmp/pot)*eps*(((Float) 2.)*vec + vea*dzc);
		}
		break;

		case	V_DAMP_ALL:
		mel = mel*dp2 + a*dp1*dzc;
		break;
	}


	if (VQcd & V_EVOL_RHO) {
		Float kReal = tmp.real()*mel.real() + tmp.imag()*mel.imag();
		mel *= kReal/pot;
	}

	v[idx-Sf] = mel;
}

template<typename cFloat, typename Float>
__global__ void	updateMKernel(cFloat * __restrict__ m, const cFloat * __restrict__ v, const Float dzd, const uint Lx, const uint Sf, const uint Vo, const uint Vf)
{
	//uint idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));
	uint idx = Vo + (threadIdx.x + blockDim.x*blockIdx.x) + Sf*(threadIdx.y + blockDim.y*blockIdx.y);

	if	(idx >= Vf)
		return;

	updateMCoreGpu<cFloat,Float>(idx, m, v, dzd, Sf);
}

template<typename Float, const VqcdType VQcd>
__global__ void	updateVKernel(const complex<Float> * __restrict__ m, complex<Float> * __restrict__ v, const Float z, const Float z2, const Float z4, const Float zQ, const Float gFac,
			      const Float eps, const Float dp1, const Float dp2, const Float dzc, const Float *ood2, const Float LL, const uint Lx, const uint Sf, const uint Vo,
			      const uint Vf)
{
	//uint idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));
	uint idx = Vo + (threadIdx.x + blockDim.x*blockIdx.x) + Sf*(threadIdx.y + blockDim.y*blockIdx.y);

	if	(idx >= Vf)
		return;

	updateVCoreGpu<Float, VQcd>(idx, m, v, z, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, ood2, LL, Lx, Sf);
}

void	updateMGpu(void * __restrict__ m, const void * __restrict__ v, const double dz, const double d, const uint Lx, const uint Vo, const uint Vf, FieldPrecision precision,
		   const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream, FieldType fType=FIELD_SAXION)
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
		if (fType & FIELD_AXION)
			updateMKernel<<<gridSize,blockSize,0,stream>>> ((        double *) m, (const         double *) v, dzd, Lx, Lx*Lx, Vo, Vf);
		else
			updateMKernel<<<gridSize,blockSize,0,stream>>> ((complex<double>*) m, (const complex<double>*) v, dzd, Lx, Lx*Lx, Vo, Vf);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float dzd  = dz*d;
		if (fType & FIELD_AXION)
			updateMKernel<<<gridSize,blockSize,0,stream>>> ((        float  *) m, (const         float  *) v, dzd, Lx, Lx*Lx, Vo, Vf);
		else
			updateMKernel<<<gridSize,blockSize,0,stream>>> ((complex<float> *) m, (const complex<float> *) v, dzd, Lx, Lx*Lx, Vo, Vf);
	}
}

void	updateVGpu(const void * __restrict__ m, void * __restrict__ v, PropParms ppar, const double dz, const double c,
				const uint Vo, const uint Vf, const VqcdType VQcd, FieldPrecision precision, const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream)
{
	if (Vo>Vf)
		return ;
/*
	const uint Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3	gridSize((Lx*Lx+BSSIZE-1)/BSSIZE,Lz2,1);
	dim3	blockSize(BSSIZE,1,1);
*/
	const uint Lx    = ppar.Lx;
	const uint Sf  = Lx*Lx;
	const uint Lz2 = (Vf-Vo)/Sf;
	dim3 gridSize((Sf+xBlock-1)/xBlock, (Lz2+yBlock-1)/yBlock, 1);
	dim3 blockSize(xBlock, yBlock, 1);

	const uint NN    = ppar.Ng;

	if (precision == FIELD_DOUBLE)
	{
		const double zR   = ppar.R;
		const double z2   = zR*zR;
		const double z4   = z2*z2;
		const double zQ   = ppar.massA2*z2*zR;
		const double LL   = ppar.lambda;
		const double dzc  = dz*c;
		const double gFp1 = ppar.gamma/zR;
		const double gFac = gFp1/zR;
		const double gFp2 = gFp1*dzc/2.;
		const double eps  = gFp2/(1. + gFp2);
		const double dp1  =   1./(1. + gFp2);
		const double dp2  = (1. - gFp2)*dp1;

		double pood2[NN] ;
		for (int i =0; i<NN; i++)
			pood2[i] = (ppar.PC)[i]*ppar.ood2a;
		const double *ood2 = &(pood2[0]);

		switch (VQcd) {

			DEFALLPROPTEM_U_GPU(double)

			default:
			return;
		}
	} else if (precision == FIELD_SINGLE) {
		const float zR   = ppar.R;
		const float z2   = zR*zR;
		const float z4   = z2*z2;
		const float zQ   = ppar.massA2*z2*zR;//xionmass2((double) zR, nQcd, 1.5 , 3.)*zR*zR*zR;
		const float LL   = ppar.lambda;
		const float dzc  = dz*c;
		const float gFp1 = ppar.gamma/zR;
		const float gFac = gFp1/zR;
		const float gFp2 = gFp1*dzc/2.;
		const float eps  = gFp2/(1. + gFp2);
		const float dp1  =   1./(1. + gFp2);
		const float dp2  = (1. - gFp2)*dp1;

		float  food2[NN] ;
		for (int i =0; i<NN; i++)
			food2[i] = (float) (ppar.PC)[i]*ppar.ood2a;
		const float *ood2 = &(food2[0]);

		switch (VQcd) {

			DEFALLPROPTEM_U_GPU(float)

			default:
			return;
		}
	}
}
