#include "kernelParms.cuh"
#include "complexGpu.cuh"
#include "utils/index.cuh"

#include "enum-field.h"
#include "scalar/scalarField.h"
//#include "utils/utils.h"

using namespace gpuCu;
using namespace indexHelper;

template<class Float>
static __device__ __forceinline__ void	toThetaCoreGpu (const uint idx, const uint cIdx, const uint bIdx, complex<Float> *mC, Float *m, complex<Float> *vC, Float *v, Float z, const uint S, const Float shift)
{
  complex<Float> mTmp = mC[cIdx] - complex<Float>(shift,0.);

	Float iMod = z/(mTmp.real()*mTmp.real() + mTmp.imag()*mTmp.imag());
	m[idx]	   = arg(mTmp);
	m[bIdx]	   = (vC[cIdx-S]*conj(mTmp)).imag()*iMod + m[idx];
	m[idx]	  *= z;
}

template<typename Float>
__global__ void toThetaKernelGpu (complex<Float> *mC, Float *m, complex<Float> *vC, Float *v, Float z, const uint S, const uint ofC, const uint ofB, const Float shift)
{
	const uint idx = (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	if      (idx >= S)
		return;

	const uint cIdx = idx + ofC;
	const uint bIdx = idx + ofB;

	toThetaCoreGpu (idx, cIdx, bIdx, mC, m, vC, v, z, S, shift);
}

template<typename Float>
void	toThetaTemplateGpu (Scalar *sField, const Float shift)
{
	const uint V  = sField->Size();
	const uint S  = sField->Surf();
	const uint Lz = sField->Depth();
	const uint Lx = sField->Length();
	const uint Go = 2*(V+S);

	#define BSSIZE 512
	dim3 gridSize((Lx*Lx+BSSIZE-1)/BSSIZE,1,1);
	dim3 blockSize(BSSIZE,1,1);

	Float *m  = static_cast<Float*>(sField->mGpu());
	Float *v  = static_cast<Float*>(sField->mGpu()) + 2*S + V;
	Float *vT = static_cast<Float*>(sField->vGpu());

	complex<Float> *mC = static_cast<complex<Float>*>(sField->mGpu());
	complex<Float> *vC = static_cast<complex<Float>*>(sField->vGpu());

	const Float z = (Float) (*sField->zV());

	for (uint cZ = 1; cZ < Lz+1; cZ++)
	{
		const uint Vo = cZ*S;

		toThetaKernelGpu<Float><<<gridSize,blockSize,0,((cudaStream_t *)sField->Streams())[0]>>>(mC, m, vC, v, z, S, Vo, Go, shift);

		cudaMemcpy (m + Vo,      m,      sizeof(Float)*S, cudaMemcpyDeviceToDevice);
		cudaMemcpy (vT + Vo - S, m + Go, sizeof(Float)*S, cudaMemcpyDeviceToDevice);
	}

	cudaMemcpy (v, vT, sizeof(Float)*V, cudaMemcpyDeviceToDevice);
}

void	toThetaGpu (Scalar *sField, const double shift)
{
	switch (sField->Precision())
	{
		case FIELD_DOUBLE:

			toThetaTemplateGpu<double> (sField, shift);
			break;

		case FIELD_SINGLE:

			toThetaTemplateGpu<float>  (sField, (float) shift);
			break;

		default:

			LogError ("Wrong precision");
			exit  (1);
			break;
	}
}
