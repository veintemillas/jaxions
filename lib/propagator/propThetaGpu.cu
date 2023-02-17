#include "kernelParms.cuh"
#include "complexGpu.cuh"
#include "utils/index.cuh"
#include "cudaErrors.h"

#include "enum-field.h"

//#include "scalar/varNQCD.h"
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
}

template<typename Float, const bool wMod>
static __device__ __forceinline__ void	updateMThetaCoreGpu(const uint idx, Float * __restrict__ m, Float * __restrict__ v, Float * __restrict__ m2, const Float zQ, const Float iz,
							      const Float dzc, const Float dzd, void  *ood2, const uint Lx, const uint Sf, const uint NN, const Float zP, const Float tPz)
{
	m[idx] = m[idx] + v[idx - NN*Sf]*dzd;
}



template<typename Float, const bool wMod>
static __device__ __forceinline__ void	updateVThetaCoreGpu(const uint idx, const Float * __restrict__ m, Float * __restrict__ v, Float * __restrict__ m2, const Float zQ, const Float iz,
							      const Float dzc, const Float dzd, void  *ood2, const uint Lx, const uint Sf, const uint NN, const Float zP, const Float tPz)
{
	uint X[3], idxPx, idxPy, idxMx, idxMy, idxPz, idxMz;

	Float mel, a, tmp;

	idx2Vec(idx, X, Lx);

	mel = (Float) 0.;
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

		idxPz = idx + nv*Sf;
		idxMz = idx - nv*Sf;

		if (wMod) {
			mel += (modPi(m[idxPx] - tmp, zP, tPz) + modPi(m[idxMx] - tmp, zP, tPz) +
			        modPi(m[idxPy] - tmp, zP, tPz) + modPi(m[idxMy] - tmp, zP, tPz) +
			        modPi(m[idxPz] - tmp, zP, tPz) + modPi(m[idxMz] - tmp, zP, tPz))*static_cast<Float*>(ood2)[nv-1];
		} else
			mel += (m[idxPx] + m[idxMx] + m[idxPy] + m[idxMy] + m[idxPz] + m[idxMz] - ((Float) 6.)*tmp)*static_cast<Float*>(ood2)[nv-1];
	}
	a = mel - zQ*sin(tmp*iz);

	mel = v[idx - NN*Sf];
	mel += a*dzc;
	v[idx - NN*Sf] = mel;
	//UP TO THIS POINT THE FUNCTION IS THE SAME AS PROPAGATOR, BUT WE DO NOT COPY ON M2
	//mel *= dzd;
	//tmp += mel;
	//if (wMod)
	//	m2[idx] = modPi(tmp, zP, tPz);
	//else
	//	m2[idx] = tmp;
}


template<typename Float, const bool wMod>
static __device__ __forceinline__ void	propagateThetaCoreGpu(const uint idx, const Float * __restrict__ m, Float * __restrict__ v, Float * __restrict__ m2, const Float zQ, const Float iz,
							      const Float dzc, const Float dzd, void  *ood2, const uint Lx, const uint Sf, const uint NN, const Float zP, const Float tPz)
{
	uint X[3], idxPx, idxPy, idxMx, idxMy, idxPz, idxMz;

	Float mel, a, tmp;

	idx2Vec(idx, X, Lx);

	mel = (Float) 0.;
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

		idxPz = idx + nv*Sf;
		idxMz = idx - nv*Sf;

		if (wMod) {
			mel += (modPi(m[idxPx] - tmp, zP, tPz) + modPi(m[idxMx] - tmp, zP, tPz) +
			        modPi(m[idxPy] - tmp, zP, tPz) + modPi(m[idxMy] - tmp, zP, tPz) +
			        modPi(m[idxPz] - tmp, zP, tPz) + modPi(m[idxMz] - tmp, zP, tPz))*static_cast<Float*>(ood2)[nv-1];
		} else
			mel += (m[idxPx] + m[idxMx] + m[idxPy] + m[idxMy] + m[idxPz] + m[idxMz] - ((Float) 6.)*tmp)*static_cast<Float*>(ood2)[nv-1];
	}
	a = mel - zQ*sin(tmp*iz);

	mel = v[idx - NN*Sf];
	mel += a*dzc;
	v[idx - NN*Sf] = mel;
	mel *= dzd;
	tmp += mel;
	if (wMod)
		m2[idx] = modPi(tmp, zP, tPz);
	else
		m2[idx] = tmp;
}

template<typename Float, const bool wMod>
__global__ void	updateVThetaKernel(const Float * __restrict__ m, Float * __restrict__ v, Float * __restrict__ m2, const Float zQ, const Float dzc, const Float dzd,
				     void *ood2, const Float iz, const uint Lx, const uint Sf, const uint Vo, const uint Vf, const uint NN, const Float zP=0, const Float tPz=0)
{
	//uint idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));
	uint idx = Vo + (threadIdx.x + blockDim.x*blockIdx.x) + Sf*(threadIdx.y + blockDim.y*blockIdx.y);

	if	(idx >= Vf)
		return;

	updateVThetaCoreGpu<Float,wMod>(idx, m, v, m2, zQ, iz, dzc, dzd, ood2, Lx, Sf, NN, zP, tPz);
}

template<typename Float, const bool wMod>
__global__ void	updateMThetaKernel(Float * __restrict__ m, Float * __restrict__ v, Float * __restrict__ m2, const Float zQ, const Float dzc, const Float dzd,
				     void *ood2, const Float iz, const uint Lx, const uint Sf, const uint Vo, const uint Vf, const uint NN, const Float zP=0, const Float tPz=0)
{
	//uint idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));
	uint idx = Vo + (threadIdx.x + blockDim.x*blockIdx.x) + Sf*(threadIdx.y + blockDim.y*blockIdx.y);

	if	(idx >= Vf)
		return;

	updateMThetaCoreGpu<Float,wMod>(idx, m, v, m2, zQ, iz, dzc, dzd, ood2, Lx, Sf, NN, zP, tPz);
}


template<typename Float, const bool wMod>
__global__ void	propagateThetaKernel(const Float * __restrict__ m, Float * __restrict__ v, Float * __restrict__ m2, const Float zQ, const Float dzc, const Float dzd,
				     void *ood2, const Float iz, const uint Lx, const uint Sf, const uint Vo, const uint Vf, const uint NN, const Float zP=0, const Float tPz=0)
{
	//uint idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));
	uint idx = Vo + (threadIdx.x + blockDim.x*blockIdx.x) + Sf*(threadIdx.y + blockDim.y*blockIdx.y);

	if	(idx >= Vf)
		return;

	propagateThetaCoreGpu<Float,wMod>(idx, m, v, m2, zQ, iz, dzc, dzd, ood2, Lx, Sf, NN, zP, tPz);
}

void	updateVThNmdGpu(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, PropParms ppar, const double dz, const double c, const double d,
	const uint Vo, const uint Vf, FieldPrecision precision, const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream)
{
	const uint Lx    = ppar.Lx;

	#define	BLSIZE 256
	const uint Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3 gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3 blockSize(BLSIZE,1,1);

	const uint NN    = ppar.Ng;
	void *ood2;
	cudaMalloc(&ood2, NN*sizeof(double));

	if (precision == FIELD_DOUBLE)
	{
		const double dzc  = dz*c;
		const double dzd  = dz*d;
		const double zR   = ppar.R;
		const double zQ   = ppar.massA2*zR*zR*zR;
		const double iZ   = 1./zR;
		double aux[NN];
		for (int i =0; i<NN; i++)
						aux[i] = (double) ((ppar.PC)[i]*ppar.ood2a);
		cudaMemcpy(ood2,aux,NN*sizeof(double),cudaMemcpyHostToDevice);
		updateVThetaKernel<double,false><<<gridSize,blockSize,0,stream>>>((const double *) m, (double *) v, (double *) m2, zQ, dzc, dzd, ood2, iZ, Lx, Lx*Lx, Vo, Vf, NN);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float dzc = dz*c;
		const float dzd = dz*d;
		const float zR = ppar.R;
		const float zQ = (float) (ppar.massA2*zR*zR*zR);
		const float iZ   = 1./zR;
		float aux[NN];
		for (int i =0; i<NN; i++)
			aux[i] = (float) ((ppar.PC)[i]*ppar.ood2a);
		cudaMemcpy(ood2,aux,NN*sizeof(float),cudaMemcpyHostToDevice);

		updateVThetaKernel<float, false><<<gridSize,blockSize,0,stream>>>((const float *) m, (float *) v, (float *) m2, zQ, dzc, dzd, ood2, iZ, Lx, Lx*Lx, Vo, Vf, NN);
	}
	cudaFree(ood2);
	CudaCheckError();
}

void	updateMThGpu(void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, PropParms ppar, const double dz, const double c, const double d,
	const uint Vo, const uint Vf, FieldPrecision precision, const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream)
{
	const uint Lx    = ppar.Lx;

	#define	BLSIZE 256
	const uint Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3 gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3 blockSize(BLSIZE,1,1);

	const uint NN    = ppar.Ng;
	void *ood2;
	cudaMalloc(&ood2, NN*sizeof(double));

	if (precision == FIELD_DOUBLE)
	{
		const double dzc  = dz*c;
		const double dzd  = dz*d;
		const double zR   = ppar.R;
		const double zQ   = ppar.massA2*zR*zR*zR;
		const double iZ   = 1./zR;
		double aux[NN];
		for (int i =0; i<NN; i++)
						aux[i] = (double) ((ppar.PC)[i]*ppar.ood2a);
		cudaMemcpy(ood2,aux,NN*sizeof(double),cudaMemcpyHostToDevice);
		updateMThetaKernel<double,false><<<gridSize,blockSize,0,stream>>>((double *) m, (double *) v, (double *) m2, zQ, dzc, dzd, ood2, iZ, Lx, Lx*Lx, Vo, Vf, NN);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float dzc = dz*c;
		const float dzd = dz*d;
		const float zR = ppar.R;
		const float zQ = (float) (ppar.massA2*zR*zR*zR);
		const float iZ   = 1./zR;
		float aux[NN];
		for (int i =0; i<NN; i++)
			aux[i] = (float) ((ppar.PC)[i]*ppar.ood2a);
		cudaMemcpy(ood2,aux,NN*sizeof(float),cudaMemcpyHostToDevice);

		updateMThetaKernel<float, false><<<gridSize,blockSize,0,stream>>>((float *) m, (float *) v, (float *) m2, zQ, dzc, dzd, ood2, iZ, Lx, Lx*Lx, Vo, Vf, NN);
	}
	cudaFree(ood2);
	CudaCheckError();
}



void	propThNmdGpu(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, PropParms ppar, const double dz, const double c, const double d,
	const uint Vo, const uint Vf, FieldPrecision precision, const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream)
{
	const uint Lx    = ppar.Lx;

	#define	BLSIZE 256
	const uint Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3 gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3 blockSize(BLSIZE,1,1);

	const uint NN    = ppar.Ng;
	void *ood2;
	cudaMalloc(&ood2, NN*sizeof(double));

	if (precision == FIELD_DOUBLE)
	{
		const double dzc  = dz*c;
		const double dzd  = dz*d;
		const double zR   = ppar.R;
		const double zQ   = ppar.massA2*zR*zR*zR;
		const double iZ   = 1./zR;
		double aux[NN];
		for (int i =0; i<NN; i++)
						aux[i] = (double) ((ppar.PC)[i]*ppar.ood2a);
		cudaMemcpy(ood2,aux,NN*sizeof(double),cudaMemcpyHostToDevice);
		propagateThetaKernel<double,false><<<gridSize,blockSize,0,stream>>>((const double *) m, (double *) v, (double *) m2, zQ, dzc, dzd, ood2, iZ, Lx, Lx*Lx, Vo, Vf, NN);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float dzc = dz*c;
		const float dzd = dz*d;
		const float zR = ppar.R;
		const float zQ = (float) (ppar.massA2*zR*zR*zR);
		const float iZ   = 1./zR;
		float aux[NN];
		for (int i =0; i<NN; i++)
			aux[i] = (float) ((ppar.PC)[i]*ppar.ood2a);
		cudaMemcpy(ood2,aux,NN*sizeof(float),cudaMemcpyHostToDevice);

		propagateThetaKernel<float, false><<<gridSize,blockSize,0,stream>>>((const float *) m, (float *) v, (float *) m2, zQ, dzc, dzd, ood2, iZ, Lx, Lx*Lx, Vo, Vf, NN);
	}
	cudaFree(ood2);
	CudaCheckError();
}



void	updateVThModGpu(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, PropParms ppar, const double dz, const double c, const double d,
	const uint Vo, const uint Vf, FieldPrecision precision, const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream)
{
	const uint Lx    = ppar.Lx;
	const uint Sf  = Lx*Lx;
	const uint Lz2 = (Vf-Vo)/Sf;
//	dim3 gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
//	dim3 blockSize(BLSIZE,1,1);
	dim3 gridSize((Sf+xBlock-1)/xBlock,(Lz2+yBlock-1)/yBlock,1);
	dim3 blockSize(xBlock,yBlock,1);

	const uint NN    = ppar.Ng;
	void *ood2;
	cudaMalloc(&ood2, NN*sizeof(double));

	if (precision == FIELD_DOUBLE)
	{
		const double dzc  = dz*c;
		const double dzd  = dz*d;
		const double zR   = ppar.R;
		const double zQ   = ppar.massA2*zR*zR*zR;
		const double iZ   = 1./zR;
		const double tPz  = 2.*M_PI*zR;
		double aux[NN];
		for (int i =0; i<NN; i++)
						aux[i] = (double) ((ppar.PC)[i]*ppar.ood2a);
		cudaMemcpy(ood2,aux,NN*sizeof(double),cudaMemcpyHostToDevice);
		updateVThetaKernel<double,true><<<gridSize,blockSize,0,stream>>>((const double*) m, (double*) v, (double*) m2, zQ, dzc, dzd, ood2, iZ, Lx, Lx*Lx, Vo, Vf, NN, M_1_PI*iZ, tPz);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float dzc = dz*c;
		const float dzd = dz*d;
		const float zR = ppar.R;
		const float zQ = (float) (ppar.massA2*zR*zR*zR);
		const float iZ   = 1./zR;
		const float tPz  = 2.*M_PI*zR;
		float aux[NN];
		for (int i =0; i<NN; i++)
			aux[i] = (float) ((ppar.PC)[i]*ppar.ood2a);
		cudaMemcpy(ood2,aux,NN*sizeof(float),cudaMemcpyHostToDevice);
		updateVThetaKernel<float, true><<<gridSize,blockSize,0,stream>>>((const float *) m, (float *) v, (float *) m2, zQ, dzc, dzd, ood2, iZ, Lx, Lx*Lx, Vo, Vf, NN, M_1_PI*iZ, tPz);
	}
}



void	propThModGpu(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, PropParms ppar, const double dz, const double c, const double d,
	const uint Vo, const uint Vf, FieldPrecision precision, const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream)
{
	const uint Lx    = ppar.Lx;
	const uint Sf  = Lx*Lx;
	const uint Lz2 = (Vf-Vo)/Sf;
//	dim3 gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
//	dim3 blockSize(BLSIZE,1,1);
	dim3 gridSize((Sf+xBlock-1)/xBlock,(Lz2+yBlock-1)/yBlock,1);
	dim3 blockSize(xBlock,yBlock,1);

	const uint NN    = ppar.Ng;
	void *ood2;
	cudaMalloc(&ood2, NN*sizeof(double));

	if (precision == FIELD_DOUBLE)
	{
		const double dzc  = dz*c;
		const double dzd  = dz*d;
		const double zR   = ppar.R;
		const double zQ   = ppar.massA2*zR*zR*zR;
		const double iZ   = 1./zR;
		const double tPz  = 2.*M_PI*zR;
		double aux[NN];
		for (int i =0; i<NN; i++)
						aux[i] = (double) ((ppar.PC)[i]*ppar.ood2a);
		cudaMemcpy(ood2,aux,NN*sizeof(double),cudaMemcpyHostToDevice);
		propagateThetaKernel<double,true><<<gridSize,blockSize,0,stream>>>((const double*) m, (double*) v, (double*) m2, zQ, dzc, dzd, ood2, iZ, Lx, Lx*Lx, Vo, Vf, NN, M_1_PI*iZ, tPz);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float dzc = dz*c;
		const float dzd = dz*d;
		const float zR = ppar.R;
		const float zQ = (float) (ppar.massA2*zR*zR*zR);
		const float iZ   = 1./zR;
		const float tPz  = 2.*M_PI*zR;
		float aux[NN];
		for (int i =0; i<NN; i++)
			aux[i] = (float) ((ppar.PC)[i]*ppar.ood2a);
		cudaMemcpy(ood2,aux,NN*sizeof(float),cudaMemcpyHostToDevice);
		propagateThetaKernel<float, true><<<gridSize,blockSize,0,stream>>>((const float *) m, (float *) v, (float *) m2, zQ, dzc, dzd, ood2, iZ, Lx, Lx*Lx, Vo, Vf, NN, M_1_PI*iZ, tPz);
	}
}

void	updateMThetaGpu(void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, PropParms ppar, const double dz, const double c, const double d,
	const uint Vo, const uint Vf, FieldPrecision precision, const int xBlock, const int yBlock, const int zBlock,
		     cudaStream_t &stream, const bool wMod)
{
	if (Vo>Vf)
		return ;

	updateMThGpu(m, v, m2, ppar, dz, c, d, Vo, Vf, precision, xBlock, yBlock, zBlock, stream);
	return;
}


void	updateVThetaGpu(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, PropParms ppar, const double dz, const double c, const double d,
	const uint Vo, const uint Vf, FieldPrecision precision, const int xBlock, const int yBlock, const int zBlock,
		     cudaStream_t &stream, const bool wMod)
{
	if (Vo>Vf)
		return ;

	switch (wMod) {

		case	true:
			updateVThModGpu(m, v, m2, ppar, dz, c, d, Vo, Vf, precision, xBlock, yBlock, zBlock, stream);
			break;

		case	false:
			updateVThNmdGpu(m, v, m2, ppar, dz, c, d, Vo, Vf, precision, xBlock, yBlock, zBlock, stream);
			break;
	}

	return;
}

void	propThetaGpu(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, PropParms ppar, const double dz, const double c, const double d,
	const uint Vo, const uint Vf, FieldPrecision precision, const int xBlock, const int yBlock, const int zBlock,
		     cudaStream_t &stream, const bool wMod)
{
	if (Vo>Vf)
		return ;

	switch (wMod) {

		case	true:
			propThModGpu(m, v, m2, ppar, dz, c, d, Vo, Vf, precision, xBlock, yBlock, zBlock, stream);
			break;

		case	false:
			propThNmdGpu(m, v, m2, ppar, dz, c, d, Vo, Vf, precision, xBlock, yBlock, zBlock, stream);
			break;
	}

	return;
}
