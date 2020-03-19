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
			complex<double> *field;
			size_t vol = sField->eSize();

			switch (fIdx)
			{
				case FIELD_M:
				field = static_cast<complex<double>*> (sField->mCpu());
				break;

				case FIELD_V:
				field = static_cast<complex<double>*> (sField->vCpu());
				break;

				case FIELD_M2:
				if (sField->LowMem()) {
					LogError ("Error: can't scale m2 with lowmem");
					return;
				}

				field = static_cast<complex<double>*> (sField->m2Cpu());
				break;

				default:
				LogError ("Error: unrecognized field type");
				return;
				break;
			}

			#pragma omp parallel for default(shared) schedule(static)
			for (size_t lpc = 0; lpc < vol; lpc++)
				field[lpc] *= factor;

			break;
		}

		case FIELD_SINGLE:
		{
			complex<float> *field;
			float  fac = factor;
			size_t vol = sField->eSize();

			switch (fIdx)
			{
				case FIELD_M:
				field = static_cast<complex<float> *> (sField->mCpu());
				break;

				case FIELD_V:
				field = static_cast<complex<float> *> (sField->vCpu());
				break;

				case FIELD_M2:
				if (sField->LowMem()) {
					LogError ("Error: can't scale m2 with lowmem");
					return;
				}

				field = static_cast<complex<float> *> (sField->m2Cpu());
				break;

				default:
				LogError ("Error: unrecognized field type");
				break;
			}

			#pragma omp parallel for default(shared) schedule(static)
			for (size_t lpc = 0; lpc < vol; lpc++)
				field[lpc] *= fac;

			break;
		}

		default:
		LogError ("Invalid precision");
		exit(1);
		break;
	}
}
