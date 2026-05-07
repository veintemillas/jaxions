#include "kernelParms.cuh"
#include "complexGpu.cuh"
#include "utils/index.cuh"
#include "cudaErrors.h"

#include "enum-field.h"

#include "utils/parse.h"

using namespace gpuCu;
using namespace indexHelper;

/* updates im with laplacian from re 
can be used reversely! */

template<typename Float>
static __device__ __forceinline__ void	propagatePaxCoreGpu_LAP(
                                        const uint idx, 
                                        const Float * __restrict__ re, 
                                        Float * __restrict__ im, 
                                        const Float * __restrict__ ood2,
                                        const uint Lx, 
                                        const uint Sf, 
                                        const uint NN)
{
	uint X[3], idxPx, idxPy, idxMx, idxMy, idxPz, idxMz;

	Float lap, re0;

	idx2Vec(idx, X, Lx);

	lap = (Float) 0.;
	re0 = re[idx];

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

        // ood2 contains already the necesary constants
        // including deta
        lap += (re[idxPx] + re[idxMx] + re[idxPy] + re[idxMy] + re[idxPz] + re[idxMz] 
                            - ((Float) 6.)*re0)*ood2[nv-1];
	}

	im[idx] += lap;
}

template<typename Float>
__global__ void	propagatePAXKernel_LAP(
                        const Float * __restrict__ m, 
                        Float * __restrict__ v, 
                        const Float * __restrict__ ood2, 
                        const uint Lx, 
                        const uint Sf, 
                        const uint Vo, const uint Vf, 
                        const uint NN)
{
	const uint local = threadIdx.x + blockDim.x*blockIdx.x;
	const uint zloc  = threadIdx.y + blockDim.y*blockIdx.y;

    if (local >= Sf)
		return;
	const uint idx = Vo + local + Sf*zloc;

	if	(idx >= Vf)
		return;

	propagatePaxCoreGpu_LAP<Float>(idx, m, v, ood2, Lx, Sf, NN);
}

/* rotates re,im with position dependent rotation skipping []
iUPS' = [-Lap(UPS)/2mc] + mc/2 () UPS
() = Bessel1(2x)/x-1
 x = |UPS|/sqrt(mc R^2)) */

template<typename Float>
__device__ __forceinline__ Float pax_sqrt(Float x);

template<>
__device__ __forceinline__ float pax_sqrt<float>(float x) {
	return sqrtf(x);
}

template<>
__device__ __forceinline__ double pax_sqrt<double>(double x) {
	return sqrt(x);
}

template<typename Float>
__device__ __forceinline__ Float pax_sin(Float x);

template<>
__device__ __forceinline__ float pax_sin<float>(float x) {
	return sinf(x);
}

template<>
__device__ __forceinline__ double pax_sin<double>(double x) {
	return sin(x);
}

template<typename Float>
__device__ __forceinline__ Float pax_cos(Float x);

template<>
__device__ __forceinline__ float pax_cos<float>(float x) {
	return cosf(x);
}

template<>
__device__ __forceinline__ double pax_cos<double>(double x) {
	return cos(x);
}

template<typename Float>
__device__ __forceinline__ Float pax_j1(Float x);

template<>
__device__ __forceinline__ float pax_j1<float>(float x) {
	return j1f(x);
}

template<>
__device__ __forceinline__ double pax_j1<double>(double x) {
	return j1(x);
}

template<typename Float>
__device__ __forceinline__ void pax_sincos(Float x, Float *s, Float *c);

template<>
__device__ __forceinline__ void pax_sincos<float>(float x, float *s, float *c) {
    __sincosf(x, s, c);
}

template<>
__device__ __forceinline__ void pax_sincos<double>(double x, double *s, double *c) {
    sincos(x, s, c);
}

template<typename Float>
static __device__ __forceinline__ void propagatePaxCoreGpu_POT(
	const uint idx,
	Float * __restrict__ re,
	Float * __restrict__ im,
	const Float mcdth,
	const Float isqrtmcR2
) {
	const Float rho2 = re[idx]*re[idx] + im[idx]*im[idx];
	const Float x = pax_sqrt<Float>(rho2) * isqrtmcR2;

	Float phase;

	if (x > (Float) 0.0)
		phase = mcdth*(pax_j1<Float>((Float) 2.0*x)/x - (Float) 1.0);
	else
		phase = (Float) 0.0;

	// const Float s = pax_sin<Float>(phase);
	// const Float c = pax_cos<Float>(phase);
    Float s, c;
    pax_sincos<Float>(phase, &s, &c);

	const Float re_t = re[idx]*c - im[idx]*s;
	im[idx] = re[idx]*s + im[idx]*c;
	re[idx] = re_t;
}

template<typename Float>
__global__ void propagatePAXKernel_POT(
	Float * __restrict__ m,
	Float * __restrict__ v,
	const uint Sf,
	const uint Vo,
	const uint Vf,
	const Float mcdth,
	const Float isqrtmcR2
) {
	const uint local = threadIdx.x + blockDim.x*blockIdx.x;
	const uint zloc  = threadIdx.y + blockDim.y*blockIdx.y;

	if (local >= Sf)
		return;

	const uint idx = Vo + local + Sf*zloc;

	if (idx >= Vf)
		return;

	propagatePaxCoreGpu_POT<Float>(idx, m, v, mcdth, isqrtmcR2);
}

template<KickDriftType kidi>
void propagatePaxGPU(
    void *m,
    void *v,
    PropParms ppar,
    const double dt,
    const size_t Vo,
    const size_t Vf,
    const FieldPrecision precision,
    const size_t xBlock,
    const size_t yBlock,
    const size_t zBlock,
    cudaStream_t stream
) {
    const uint Lx = static_cast<uint>(ppar.Lx);
    const uint Sf = Lx*Lx;
    const size_t vol = Vf - Vo;

    if (vol == 0)
        return;

	#define	BLSIZE 256
	const uint Lz2 = (Vf-Vo)/Sf;
	dim3 gridSize((Sf+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3 blockSize(BLSIZE,1,1);

	const uint NN    = ppar.Ng;

    if constexpr (kidi == KIDI_LAP)
    {
        const double itwomc = ppar.sign*dt/(2*ppar.massA*ppar.R);

        switch (precision) 
            {
                case FIELD_SINGLE:
                {
                    float aux[NN];
                    for (int i =0; i<NN; i++)
                        aux[i] = (float) ((ppar.PC)[i]*ppar.ood2a*itwomc);
                    float *ood2 = nullptr;
                    cudaMalloc(&ood2, NN*sizeof(float));
                    cudaMemcpy(ood2,aux,NN*sizeof(float),cudaMemcpyHostToDevice);
                    propagatePAXKernel_LAP<float>
                        <<<gridSize, blockSize, 0, stream>>>(
                            static_cast<float *>(m),
                            static_cast<float *>(v),
                            ood2, Lx, Sf, Vo, Vf, NN
                        );
                    cudaStreamSynchronize(stream);
                    cudaFree(ood2);
                }
                break;

                case FIELD_DOUBLE:
                {
                    double aux[NN];
                    for (int i =0; i<NN; i++)
                        aux[i] = (double) ((ppar.PC)[i]*ppar.ood2a*itwomc);
                    double *ood2 = nullptr;
                    cudaMalloc(&ood2, NN*sizeof(double));
                    cudaMemcpy(ood2,aux,NN*sizeof(double),cudaMemcpyHostToDevice);
                    propagatePAXKernel_LAP<double>
                        <<<gridSize, blockSize, 0, stream>>>(
                            static_cast<double *>(m),
                            static_cast<double *>(v),
                            ood2, Lx, Sf, Vo, Vf, NN
                        );
                    cudaStreamSynchronize(stream);
                    cudaFree(ood2);
                } 
                break;
                
            }
        

    } 
    
    else if constexpr (kidi == KIDI_POT)
    {
        double mcdth = dt*ppar.massA*ppar.R/2.0 ;
        double isqrtmcR2 = 1.0/std::sqrt(ppar.massA*ppar.R*ppar.R*ppar.R);
        switch (precision) 
            {
                case FIELD_SINGLE:
                {
                    float mcdth_f     = (float) mcdth;
                    float isqrtmcR2_f = (float) isqrtmcR2;
                    propagatePAXKernel_POT<float>
                        <<<gridSize, blockSize, 0, stream>>>(
                            static_cast<float *>(m),
                            static_cast<float *>(v),
                            Sf, Vo, Vf, mcdth_f, isqrtmcR2_f
                        );
                }
                break;

                case FIELD_DOUBLE:
                {
                    propagatePAXKernel_POT<double>
                        <<<gridSize, blockSize, 0, stream>>>(
                            static_cast<double *>(m),
                            static_cast<double *>(v),
                            Sf, Vo, Vf, mcdth, isqrtmcR2
                        );
                }   
                break;
            } 
    }



CudaCheckError();
}

template void propagatePaxGPU<AxionEnum::KIDI_LAP>(
    void*, void*, AxionEnum::PropParms_v,
    double, size_t, size_t, AxionEnum::FieldPrecision_s,
    size_t, size_t, size_t, cudaStream_t
);

template void propagatePaxGPU<AxionEnum::KIDI_POT>(
    void*, void*, AxionEnum::PropParms_v,
    double, size_t, size_t, AxionEnum::FieldPrecision_s,
    size_t, size_t, size_t, cudaStream_t
);