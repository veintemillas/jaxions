#include "kernelParms.cuh"
#include "complexGpu.cuh"

#include "scalar/scalarField.h"
#include "utils/utils.h"

using namespace gpuCu;

template<typename Float>
static __device__ __forceinline__ void th2cxCoreGpu (uint idx, complex<Float> * __restrict__ mFd, complex<Float> * __restrict__ vFd, const Float r, const Float ir)
{
	auto II = complex<Float>(0,1);

	mFd[idx] = r*exp(II*(mFd[idx]).real());
	vFd[idx] = mFd[idx]*(II*vFd[idx] + ir);
}


template<typename Float>
__global__ void th2cxKernelGpu(complex<Float> * __restrict__ mFd, complex<Float> * __restrict__ vFd, const Float r, const Float ir, uint V)
{
	uint idx = (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	if      (idx >= V)
		return;
	
	th2cxCoreGpu<Float>(idx, mFd, vFd, r, ir);
}

void	th2cxGpu (Scalar *sField)
{
	const uint Lx = sField->Length();
	const uint Lz = sField->Depth();
	const uint S  = sField->Surf();
	const uint V  = sField->Size();

	#define	BLSIZE 512
	dim3	   gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz,1);
	dim3	   blockSize(BLSIZE,1,1);

	const size_t bytes = V*sField->DataSize();

	switch (sField->Precision())
	{
		case FIELD_DOUBLE:
		{
			double r  = *sField->RV();
			double ir = 1.0/r;

			complex<double>	*mFd = static_cast<complex<double>*>(sField->mGpu()) + S; 
			complex<double>	*vFd = static_cast<complex<double>*>(sField->vGpu()); 

			th2cxKernelGpu<double><<<gridSize, blockSize, 0, ((cudaStream_t *)sField->Streams())[0]>>> (mFd, vFd, r, ir, V);

			break;
		}

		case FIELD_SINGLE:
		{
			float r  = *sField->RV();
			float ir = 1.0f/r;

			complex<float>	*mFd = static_cast<complex<float>*>(sField->mGpu()) + S; 
			complex<float>	*vFd = static_cast<complex<float>*>(sField->vGpu()); 

			th2cxKernelGpu<float> <<<gridSize, blockSize, 0, ((cudaStream_t *)sField->Streams())[0]>>> (mFd, vFd, r, ir, V);

			break;
		}

		default:

		LogError ("Unrecognized precision");
		exit(1);
		break;
	}
}
