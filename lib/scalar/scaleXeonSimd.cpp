#include <complex>

#include "scalar/scalarField.h"
#include "utils/utils.h"

using namespace std;

void	scaleXeon (Scalar *sField, FieldIndex fIdx, double factor)
{
	switch (sField->Precision())
	{
		case FIELD_DOUBLE:
		{
			double *field;
			size_t vSize = sField->DataAlign()/sizeof(double);
			size_t vol   = (sField->Size()*2)/vSize;

			switch (fIdx)
			{
				case FIELD_M:
				field = static_cast<double*> (sField->mStart());
				break;

				case FIELD_V:
				field = static_cast<double*> (sField->vCpu());
				break;

				case FIELD_M2:
				if (sField->LowMem()) {
					LogError ("Error: can't scale m2 with lowmem");
					return;
				}

				field = static_cast<double*> (sField->m2Cpu());
				break;

				default:
				LogError ("Error: unrecognized field type");
				return;
				break;
			}

			#pragma omp parallel for default(shared) schedule(static)
			for (size_t lpc = 0; lpc < vol; lpc++) {
				#pragma omp simd
				for (size_t i = 0; i < vSize; i++)
					field[lpc*vSize+i] *= factor;
			}

			break;
		}

		case FIELD_SINGLE:
		{
			float *field;
			float  fac = factor;
			size_t vSize = sField->DataAlign()/sizeof(float);
			size_t vol   = sField->Size()/vSize;

			switch (fIdx)
			{
				case FIELD_M:
				field = static_cast<float *> (sField->mStart());
				break;

				case FIELD_V:
				field = static_cast<float *> (sField->vCpu());
				break;

				case FIELD_M2:
				if (sField->LowMem()) {
					LogError ("Error: can't scale m2 with lowmem");
					return;
				}

				field = static_cast<float *> (sField->m2Cpu());
				break;

				default:
				LogError ("Error: unrecognized field type");
				break;
			}

			#pragma omp parallel for default(shared) schedule(static)
			for (size_t lpc = 0; lpc < vol; lpc++) {
				#pragma omp simd
				for (size_t i = 0; i < vSize; i++)
					field[lpc*vSize+i] *= fac;
			}

			break;
		}

		default:
		LogError ("Invalid precision");
		exit(1);
		break;
	}
}
