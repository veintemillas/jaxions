#include "kernelParms.cuh"
#include "complexGpu.cuh"
#include "utils/index.cuh"

#include "enum-field.h"

#include "scalar/varNQCD.h"
#include "utils/parse.h"

using namespace gpuCu;
using namespace indexHelper;

#define	TwoPi	(2.*M_PI)

template<typename Float>
static __device__ __forceinline__ Float modPi (const Float x, const Float OneOvPi, const Float TwoPiZ)
{
	const Float tmp = x*OneOvPi;

	if (tmp >=  1.)
		return (x-TwoPiZ);

	if (tmp <  -1.)
		return (x+TwoPiZ);

	return x;
/*
	const Float tP  = x+TwoPiZ;
	const Float tM  = x-TwoPiZ;
	const Float tP2 = tP*tP;
	const Float tM2 = tM*tM;
	const Float x2  = x*x;

	if (tP2 < tM2) {
		if (tP2 < x2)
			return	tP;
		else
			return	x;
	} else {
		if (tM2 < x2)
			return	tM;
		else
			return	x;
	}
*/
}

/*	Useless because the gpu has a proper sine, but this way we can compare implementations	*/

template<typename Float>
static __device__ __forceinline__ Float mySin (const Float x)
{
        constexpr Float a = -0.0415758*4., b = 0.00134813*6., c = -(1+M_PI*M_PI*a+M_PI*M_PI*M_PI*M_PI*b)/(M_PI*M_PI*M_PI*M_PI*M_PI*M_PI);
	return (x + x*x*x*(a + x*x*(b + x*x*c)));
}

template<typename Float>
static __device__ __forceinline__ void	propThetaModCoreGpu(const uint idx, const Float * __restrict__ m, Float * __restrict__ v, Float * __restrict__ m2, const Float zP, const Float tPz,
							    const Float iz, const Float zQ, const Float dzc, const Float dzd, const Float ood2, const uint Lx, const uint Sf)
{
	uint X[3], idxPx, idxPy, idxMx, idxMy;

	Float mel, a, tmp;

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
	mel = modPi(m[idxPx]  - tmp, zP, tPz) + modPi(m[idxMx]  - tmp, zP, tPz) +
	      modPi(m[idxPy]  - tmp, zP, tPz) + modPi(m[idxMy]  - tmp, zP, tPz) +
	      modPi(m[idx+Sf] - tmp, zP, tPz) + modPi(m[idx-Sf] - tmp, zP, tPz);

	a = mel*ood2 - zQ*sin(tmp*iz);

	mel = v[idx-Sf];
	mel += a*dzc;
	v[idx-Sf] = mel;
	mel *= dzd;
	tmp += mel;
	m2[idx] = modPi(tmp, zP, tPz);
}

template<typename Float>
__global__ void	propThetaModKernel(const Float * __restrict__ m, Float * __restrict__ v, Float * __restrict__ m2, const Float zP, const Float tPz, const Float iz, const Float zQ,
				   const Float dzc, const Float dzd, const Float ood2, const uint Lx, const uint Sf, const uint Vo, const uint Vf)
{
	uint idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	if	(idx >= Vf)
		return;

	propagateThetaCoreGpu<Float>(idx, m, v, m2, zP, tPz, iz, zQ, dzc, dzd, ood2, Lx, Sf);
}

void	propThetaModGpu(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, double *z, const double dz, const double c, const double d, const double delta2,
			const double nQcd, const uint Lx, const uint Lz, const uint Vo, const uint Vf, FieldPrecision precision, cudaStream_t &stream)
{
	#define	BLSIZE 256
	const uint Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3 gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3 blockSize(BLSIZE,1,1);

	if (precision == FIELD_DOUBLE)
	{
		const double dzc  = dz*c;
		const double dzd  = dz*d;
		const double zR   = *z;
		const double iz   = 1./zR;
		const double zQ   = axionmass2((double) zR, nQcd, zthres, zrestore)*zR*zR*zR;
		const double ood2 = 1./delta2;
		const double tPz  = 2.*M_PI*zR;
		propThetaModKernel<<<gridSize,blockSize,0,stream>>>((const double *) m, (double *) v, (double *) m2, M_1_PI*iz, tPz, iz, zQ, dzc, dzd, ood2, Lx, Lx*Lx, Vo, Vf);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float dzc = dz*c;
		const float dzd = dz*d;
		const float zR = *z;
		const float iz = 1./zR;
		const float zQ = (float) axionmass2((double) zR, nQcd, zthres, zrestore)*zR*zR*zR;
		const float ood2 = 1./delta2;
		const float tPz  = 2.*M_PI*zR;
		propThetaModKernel<<<gridSize,blockSize,0,stream>>>((const float *) m, (float *) v, (float *) m2, (float) (M_1_PI*iz), tPz, iz, zQ, dzc, dzd, ood2, Lx, Lx*Lx, Vo, Vf);
	}
}

