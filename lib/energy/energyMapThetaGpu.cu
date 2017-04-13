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
static __device__ __forceinline__ void	energyMapThetaCoreGpu(const uint idx, const Float * __restrict__ m, const Float * __restrict__ v, Float * __restrict__ m2, const uint Lx, const uint Sf, const Float iZ, const Float iZ2, const Float zP, const Float tPz, const Float o2, const Float zQ)
{
	uint X[3], idxPx, idxPy, idxMx, idxMy, idxPz = idx + Sf, idxMz = idx - Sf;

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
	vel = v[idxMz];

	tmpPx = modPi(m[idxPx] - mel, zP, tPz); 
	tmpPy = modPi(m[idxPy] - mel, zP, tPz); 
	tmpPz = modPi(m[idxPz] - mel, zP, tPz); 
	tmpMx = modPi(m[idxMx] - mel, zP, tPz); 
	tmpMy = modPi(m[idxMy] - mel, zP, tPz); 
	tmpMz = modPi(m[idxMz] - mel, zP, tPz); 

	aX = tmpPx*tmpPx + tmpMx*tmpMx;
	aY = tmpPy*tmpPy + tmpMy*tmpMy;
	aZ = tmpPz*tmpPz + tmpMz*tmpMz;

	Kt = 0.5*vel*vel;
	Vt = zQ*(1.0f - cos(mel*iZ));

	m2[idxMz] = iZ2*(o2*(aX + aY + aZ) + Kt) + Vt;
}

template<typename Float>
__global__ void	energyMapThetaKernel(const Float * __restrict__ m, const Float * __restrict__ v, Float * __restrict__ m2, const uint Lx, const uint Sf, const uint V, const Float iZ, const Float iZ2, const Float zP, const Float tPz, const Float o2, const Float zQ)
{
	uint idx = Sf + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	if	(idx < V)
		energyMapThetaCoreGpu<Float>(idx, m, v, m2, Lx, Sf, iZ, iZ2, zP, tPz, o2, zQ);
}

void	energyMapThetaGpu	(const void * __restrict__ m, const void * __restrict__ v, void * __restrict__ m2, double *z, const double delta2, const double nQcd,
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
		const double iz  = 1./zR;
		const double zP  = M_1_PI*iz;
		const double tPz = 2.0*M_PI*zR;
		const double o2  = 0.25/delta2;
		const double zQ  = axionmass2(zR, nQcd, zthres, zrestore)*zR*zR;
		const double iz2 = 1./(zR*zR);

		energyMapThetaKernel<<<gridSize,blockSize,0,stream>>> (static_cast<const double*>(m), static_cast<const double*>(v), static_cast<double*>(m2), Lx, S, Vm, iz, iz2, zP, tPz, o2, zQ);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float iz  = 1.0/zR;
		const float zP  = M_1_PI*iz;
		const float tPz = 2.f*M_PI*zR;
		const float o2  = 0.25/delta2;
		const float zQ  = axionmass2((float) zR, nQcd, zthres, zrestore)*zR*zR;
		const float iz2 = 1./(zR*zR);

		energyMapThetaKernel<<<gridSize,blockSize,0,stream>>> (static_cast<const float*>(m), static_cast<const float*>(v), static_cast<float*>(m2), Lx, S, Vm, iz, iz2, zP, tPz, o2, zQ);
	}

	cudaDeviceSynchronize();

	return;
}

