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

	tR[TH_GRX] = (double) aX;
	tR[TH_GRY] = (double) aY;
	tR[TH_GRZ] = (double) aZ;
	tR[TH_KIN] = (double) Kt;
	tR[TH_POT] = (double) Vt;
}

template<typename Float>
__global__ void	energyThetaKernel(const Float * __restrict__ m, const Float * __restrict__ v, const uint Lx, const uint Sf, const uint V, const Float iZ, const Float zP, const Float tPz, double *eR, double *partial)
{
	uint idx = Sf + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	double tmp[5] = { 0., 0., 0., 0., 0. };

	if	(idx < V)
		energyThetaCoreGpu<Float>(idx, m, v, Lx, Sf, iZ, zP, tPz, tmp);

	reduction<BLSIZE,double,5>   (eR, tmp, partial);
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

	eR[TH_GRX] *= o2*iz2;
	eR[TH_GRY] *= o2*iz2;
	eR[TH_GRZ] *= o2*iz2;
	eR[TH_KIN] *= .5*iz2;
	eR[TH_POT] *= zQ;

	return	0;
}

