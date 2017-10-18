#include "complexGpu.cuh"
#include "utils/index.cuh"

#include "enum-field.h"

#include "scalar/varNQCD.h"
#include "utils/parse.h"

#include "utils/reduceGpu.cuh"

#define	BLSIZE 512

using namespace gpuCu;
using namespace indexHelper;

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

template<typename Float, const bool map, const bool wMod>
static __device__ __forceinline__ void	energyThetaCoreGpu(const uint idx, const Float * __restrict__ m, const Float * __restrict__ v, Float * __restrict__ m2, const uint Lx, const uint Sf,
							   const Float iZ, const Float zP, const Float tPz, double *tR, const Float o2=0., const Float zQ=0., const Float izh=0.)
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

	if (wMod) {
		tmpPx = modPi(m[idxPx]    - mel, zP, tPz); 
		tmpPy = modPi(m[idxPy]    - mel, zP, tPz); 
		tmpPz = modPi(m[idx + Sf] - mel, zP, tPz); 
		tmpMx = modPi(m[idxMx]    - mel, zP, tPz); 
		tmpMy = modPi(m[idxMy]    - mel, zP, tPz); 
		tmpMz = modPi(m[idx - Sf] - mel, zP, tPz); 
	} else {
		tmpPx = m[idxPx]    - mel; 
		tmpPy = m[idxPy]    - mel; 
		tmpPz = m[idx + Sf] - mel; 
		tmpMx = m[idxMx]    - mel; 
		tmpMy = m[idxMy]    - mel; 
		tmpMz = m[idx - Sf] - mel; 
	}

	aX = tmpPx*tmpPx + tmpMx*tmpMx;
	aY = tmpPy*tmpPy + tmpMy*tmpMy;
	aZ = tmpPz*tmpPz + tmpMz*tmpMz;

	Kt = vel*vel;
	Vt = 1.0f - cos(mel*iZ);

	tR[TH_GRX] = (double) aX;
	tR[TH_GRY] = (double) aY;
	tR[TH_GRZ] = (double) aZ;
	tR[TH_KIN] = (double) Kt;
	tR[TH_POT] = (double) Vt;

	if (map == true)
		m2[idx] = (aX+aY+aZ)*o2 + Vt*zQ + Kt*izh;
}

template<typename Float, const bool map, const bool wMod>
__global__ void	energyThetaKernel(const Float * __restrict__ m, const Float * __restrict__ v, Float * __restrict__ m2, const uint Lx, const uint Sf, const uint V, const Float iZ,
				  const Float zP, const Float tPz, double *eR, double *partial, const Float o2 = 0., const Float zQ = 0., const Float izh = 0.)
{
	uint idx = Sf + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	double tmp[5] = { 0., 0., 0., 0., 0. };

	if	(idx < V)
		energyThetaCoreGpu<Float, map, wMod> (idx, m, v, m2, Lx, Sf, iZ, zP, tPz, tmp, o2, zQ, izh);

	reduction<BLSIZE,double,5>   (eR, tmp, partial);
}

template<const bool wMod>
int	energyThetaGpu	(const void * __restrict__ m, const void * __restrict__ v, void * __restrict__ m2, double *z, const double delta2, const double nQcd,
			 const uint Lx, const uint Lz, const uint V, const uint S, FieldPrecision precision, double *eR, cudaStream_t &stream, const bool map)
{
	const uint Vm = V+S;
	const uint Lz2 = V/(Lx*Lx);
	dim3  gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3  blockSize(BLSIZE,1,1);
	const int nBlocks = gridSize.x*gridSize.y;

	const double zR   = *z;
	double *tR, *partial;

	const double zQ  = axionmass2((float) zR, nQcd, zthres, zrestore)*zR*zR;
	const double iZ  = 1./zR;
	const double iz2 = iZ*iZ;
	const double o2  = 0.25/delta2*iz2;

	if ((cudaMalloc(&tR, sizeof(double)*5) != cudaSuccess) || (cudaMalloc(&partial, sizeof(double)*5*nBlocks*4) != cudaSuccess))
	{
		return -1;
	}

	if (precision == FIELD_DOUBLE)
	{
		const double zP  = M_1_PI*iZ;
		const double tPz = 2.0*M_PI*zR;
		if (map == true)
			energyThetaKernel<double,true, wMod><<<gridSize,blockSize,0,stream>>> (static_cast<const double*>(m), static_cast<const double*>(v), static_cast<double*>(m2), Lx, S, Vm, iZ, zP, tPz, tR, partial, o2, zQ, iz2*.5);
		else
			energyThetaKernel<double,false,wMod><<<gridSize,blockSize,0,stream>>> (static_cast<const double*>(m), static_cast<const double*>(v), static_cast<double*>(m2), Lx, S, Vm, iZ, zP, tPz, tR, partial);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float zP  = M_1_PI*iZ;
		const float tPz = 2.f*M_PI*zR;
		if (map == true)
			energyThetaKernel<float, true, wMod><<<gridSize,blockSize,0,stream>>> (static_cast<const float* >(m), static_cast<const float* >(v), static_cast<float* >(m2), Lx, S, Vm, iZ, zP, tPz, tR, partial, o2, zQ, iz2*.5);
		else
			energyThetaKernel<float, false,wMod><<<gridSize,blockSize,0,stream>>> (static_cast<const float* >(m), static_cast<const float* >(v), static_cast<float* >(m2), Lx, S, Vm, iZ, zP, tPz, tR, partial);
	}

	cudaDeviceSynchronize();

	cudaMemcpy(eR, tR, sizeof(double)*5, cudaMemcpyDeviceToHost);
	cudaFree(tR); cudaFree(partial);

	eR[TH_GRX] *= o2;
	eR[TH_GRY] *= o2;
	eR[TH_GRZ] *= o2;
	eR[TH_KIN] *= .5*iz2;
	eR[TH_POT] *= zQ;

	return	0;
}

int	energyThetaGpu	(const void * __restrict__ m, const void * __restrict__ v, void * __restrict__ m2, double *z, const double delta2, const double nQcd,
			 const uint Lx, const uint Lz, const uint V, const uint S, FieldPrecision precision, double *eR, cudaStream_t &stream, const bool map, const bool wMod)
{
	switch (wMod) {
		case true:
			return	energyThetaGpu<true>(m, v, m2, z, delta2, nQcd, Lx, Lz, V, S, precision, eR, stream, map);
			break;
		case false:
			return	energyThetaGpu<true>(m, v, m2, z, delta2, nQcd, Lx, Lz, V, S, precision, eR, stream, map);
			break;
	}

	return	-1;
}
			
