#include "kernelParms.cuh"
#include "complexGpu.cuh"

#include "scalar/scalarField.h"
//#include "utils/utils.h"

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

template<typename Float>
__global__ void th2cxM2KernelGpu (const Float * __restrict__ in,
	complex<Float> * __restrict__ out, const Float ir, const uint V)
{
	const uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx < V) {
		const Float theta = in[idx]*ir;
		out[idx] = complex<Float>(::cos(theta), ::sin(theta));
	}
}

void th2cxM2Gpu (Scalar *sField)
{
	const uint V = sField->eSize();
	const uint blocks = (V + BLSIZE - 1)/BLSIZE;
	auto stream = ((cudaStream_t *)sField->Streams())[0];

	if (sField->Precision() == FIELD_DOUBLE) {
		th2cxM2KernelGpu<double><<<blocks, BLSIZE, 0, stream>>>(
			static_cast<const double *>(sField->mGpu()),
			static_cast<complex<double> *>(sField->m2Gpu()),
			1.0/(*sField->RV()), V);
	} else if (sField->Precision() == FIELD_SINGLE) {
		th2cxM2KernelGpu<float><<<blocks, BLSIZE, 0, stream>>>(
			static_cast<const float *>(sField->mGpu()),
			static_cast<complex<float> *>(sField->m2Gpu()),
			1.0f/float(*sField->RV()), V);
	}
}
