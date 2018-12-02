#include "kernelParms.cuh"
#include "complexGpu.cuh"
#include "utils/index.cuh"

#include "enum-field.h"
#include "scalar/scalarField.h"

using namespace gpuCu;
using namespace indexHelper;

template<typename Float>
static __device__ __forceinline__ void iteraCore (const uint idx, const complex<Float> * __restrict__ mCp, complex<Float> * __restrict__ vCp, const uint Lx, const uint Sf, const Float alpha)
{
	const Float One = 1.;
	const Float OneSixth = (0.16666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666667);

	uint X[3], idxPx, idxPy, idxMx, idxMy;

	complex<Float> mel, a, tmp;

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

	mel = mCp[idx]*alpha + (One-alpha)*OneSixth*(mCp[idxMx] + mCp[idxPx] + mCp[idxPy] + mCp[idxMy] + mCp[idx+Sf] + mCp[idx-Sf]);
	vCp[idx-Sf] = mel;
}

template<typename Float>
__global__ void	iteraKernel (const complex<Float> * __restrict__ mCp, complex<Float> * __restrict__ vCp, const uint Lx, const uint S, const uint V, const Float alpha)
{
	uint idx = S + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	if	(idx >= V+S)
		return;

	iteraCore<Float>(idx, mCp, vCp, Lx, S, alpha);
}

void	smoothGpu (Scalar *field, const size_t iter, const double alpha)
{
	#define BLSIZE 256
	const uint bytes = field->DataSize()*field->Size();
	const uint Lx    = field->Length();
	const uint Lz    = field->Depth();
	const uint Sf    = field->Surf();
	const uint Vf    = field->Size();

	dim3	   gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz,1);
	dim3	   blockSize(BLSIZE,1,1);

	field->exchangeGhosts(FIELD_M);

	switch	(field->Precision())
	{
		case	FIELD_DOUBLE:
		for (size_t it=0; it<iter; it++)
		{
			iteraKernel<<<gridSize,blockSize>>> (static_cast<const complex<double>*>(field->mGpu()), static_cast<complex<double>*>(field->vGpu()), Lx, Sf, Vf, alpha);
			cudaMemcpy(static_cast<char *>(field->mGpu()) + Sf*field->DataSize(), static_cast<char *>(field->vGpu()),  bytes, cudaMemcpyDeviceToDevice);
			field->exchangeGhosts(FIELD_M);
		}
		break;

		case	FIELD_SINGLE:
		for (size_t it=0; it<iter; it++)
		{
			iteraKernel<<<gridSize,blockSize>>> (static_cast<const complex<float>*>(field->mGpu()), static_cast<complex<float>*>(field->vGpu()), Lx, Sf, Vf, static_cast<float>(alpha));
			cudaMemcpy(static_cast<char *>(field->mGpu()) + Sf*field->DataSize(), static_cast<char *>(field->vGpu()),  bytes, cudaMemcpyDeviceToDevice);
			field->exchangeGhosts(FIELD_M);
		}
		break;

		default:
		printf("Unrecognized precision\n");
		exit(1);
		break;
	}
}
