#include "kernelParms.cuh"
#include "complexGpu.cuh"
#include "utils/index.cuh"

#include "enum-field.h"
#include "scalar/scalarField.h"
//#include "utils/utils.h"

using namespace gpuCu;
using namespace indexHelper;

template<typename Float>
static __device__ __forceinline__ void normCoreCoreGpu (uint idx, const complex<Float> * __restrict__ mCp, complex<Float> * __restrict__ vCp, Float lambda, Float sqLzd, uint Lx, uint S, uint V)
{
	uint X[3], idxPx, idxPy, idxMx, idxMy;

	complex<Float> iM;

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
		idxPy = idx - S + Lx;
	else
		idxPy = idx + Lx;

	if (X[1] == 0)
		idxMy = idx + S - Lx;
	else
		idxMy = idx - Lx;

	Float gradx, grady, gradz, sss, sss2, sss4, rhof;

	iM = complex<Float>(1.,0.)/mCp[idx];

	gradx = ((mCp[idxPx] - mCp[idxMx])*iM).imag();
	grady = ((mCp[idxPy] - mCp[idxMy])*iM).imag();
	gradz = ((mCp[idx+S] - mCp[idx-S])*iM).imag();

	gradx = gradx*gradx + grady*grady + gradz*gradz;

	if (gradx > 0.001) {
		sss  = sqLzd/sqrt(gradx);
		sss2 = sss*sss;
		sss4 = sss2*sss2;
		rhof  = (0.6081*sss+0.328*sss2+0.144*sss4)/(1.0+0.5515*sss+0.4*sss2+0.144*sss4);
	} else {
		rhof = 1.0 ;
	}

	vCp[idx-S] = mCp[idx]*rhof/abs(mCp[idx]);
}

template<typename Float>
__global__ void normCoreKernelGpu(const complex<Float> * __restrict__ mCp, complex<Float> * __restrict__ vCp, Float lambda, Float sqLzd, uint Lx, uint S, uint V)
{
	uint idx = (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	if      (idx >= V)
		return;

	normCoreCoreGpu<Float>(idx+S, mCp, vCp, lambda, sqLzd, Lx, S, V);
}


void	normCoreGpu (Scalar *sField)
{
	const uint Lx = sField->Length();
	const uint Lz = sField->Depth();
	const uint S  = sField->Surf();
	const uint V  = sField->Size();

	#define	BLSIZE 512
	dim3	   gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz,1);
	dim3	   blockSize(BLSIZE,1,1);

	const size_t bytes = V*sField->DataSize();

	sField->exchangeGhosts(FIELD_M);

	switch (sField->Precision())
	{
		case FIELD_DOUBLE:
		{
			const double deltaA = sField->Delta();
			const double zia = (*sField->zV());
			const double LLa = (sField->LambdaT() == LAMBDA_FIXED) ? sField->BckGnd()->Lambda() : 1.125/(pow(deltaA*zia,2.));

			const double sqLzd = sqrt(LLa)*zia*deltaA;

			normCoreKernelGpu<double><<<gridSize, blockSize, 0, ((cudaStream_t *)sField->Streams())[0]>>> (static_cast<const complex<double>*>(sField->mGpu()), static_cast<complex<double>*>(sField->vGpu()), LLa, sqLzd, Lx, S, V);

			break;
		}

		case FIELD_SINGLE:
		{
			const float deltaA = sField->Delta();
			const float zia = (*sField->zV());
			const float LLa = (sField->LambdaT() == LAMBDA_FIXED) ? sField->BckGnd()->Lambda() : 1.125f/(powf(deltaA*zia,2.f));

			const float sqLzd = sqrt(LLa)*zia*deltaA;

			normCoreKernelGpu<float><<<gridSize, blockSize, 0, ((cudaStream_t *)sField->Streams())[0]>>> (static_cast<const complex<float>*>(sField->mGpu()), static_cast<complex<float>*>(sField->vGpu()), LLa, sqLzd, Lx, S, V);

			break;
		}

		default:

		LogError ("Unrecognized precision");
		exit(1);
		break;
	}

	cudaMemcpy(static_cast<char *>(sField->mGpu()) + S*sField->DataSize(), static_cast<char *>(sField->vGpu()),  bytes, cudaMemcpyDeviceToDevice);
	sField->exchangeGhosts(FIELD_M);
}
