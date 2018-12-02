#include "kernelParms.cuh"
#include "complexGpu.cuh"
#include "utils/index.cuh"

#include "enum-field.h"
#include "scalar/scalarField.h"
#include "utils/parse.h"

using namespace gpuCu;
using namespace indexHelper;

template<class Float>
__global__ void	scaleKernel (complex<Float> * __restrict__ fD, uint V, Float factor)
{
	uint idx = (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	if      (idx >= V)
		return;

	fD[idx] *= factor;
}

void    scaleGpu (Scalar *sField, FieldIndex fIdx, double factor)
{
	const uint Lx = sField->Length();
	const uint Lz = sField->Depth();

	#define	BLSIZE 512
	dim3	   gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz+2,1);
	dim3	   blockSize(BLSIZE,1,1);

	switch (sField->Precision())
	{
		case FIELD_DOUBLE:
		{
			complex<double> *field;
			uint V = sField->Size();

			switch (fIdx)
			{
				case FIELD_M:
				field = static_cast<complex<double>*> (sField->mGpu());
				V = sField->eSize();
				break;

				case FIELD_V:
				field = static_cast<complex<double>*> (sField->vGpu());
				break;

				case FIELD_M2:
				if (sField->LowMem()) {
					printf ("Wrong field. Lowmem forbids the use of m2");
					return;
				}

				field = static_cast<complex<double>*> (sField->m2Gpu());
				V = sField->eSize();
				break;

				case FIELD_MV:
				printf ("Not implemented yet. Please call scale with FIELD_M and then with FIELD_V\n");
				break;

				default:
				printf ("Wrong field. Valid possibilities: FIELD_M, FIELD_M2 and FIELD_V");
				return;
			}

			scaleKernel<double><<<gridSize, blockSize, 0, ((cudaStream_t *)sField->Streams())[0]>>> (field, V, factor);

			break;
		}

		case FIELD_SINGLE:
		{
			complex<float> *field;
			uint V = sField->Size();

			switch (fIdx)
			{
				case FIELD_M:
				field = static_cast<complex<float> *> (sField->mGpu());
				V = sField->eSize();
				break;

				case FIELD_V:
				field = static_cast<complex<float> *> (sField->vGpu());
				break;

				case FIELD_M2:
				if (sField->LowMem()) {
					printf ("Wrong field. Lowmem forbids the use of m2");
					return;
				}

				field = static_cast<complex<float> *> (sField->m2Gpu());
				V = sField->eSize();
				break;

				case FIELD_MV:
				printf ("Not implemented yet. Please call scale with FIELD_M and then with FIELD_V\n");
				break;

				default:
				printf ("Wrong field. Valid possibilities: FIELD_M, FIELD_M2 and FIELD_V");
				break;
			}

			scaleKernel<float><<<gridSize, blockSize, 0, ((cudaStream_t *)sField->Streams())[0]>>> (field, V, (float) factor);

			break;
		}

		default:
		printf("Unrecognized precision\n");
		exit(1);
		break;
	}
}
