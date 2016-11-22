#include <complex>

#include "scalar/scalarField.h"

using namespace std;

void	normXeon (Scalar *sField, FieldIndex fIdx)
{
	switch (sField->Precision())
	{
		case FIELD_DOUBLE:
		{
			complex<double> *field;
			size_t vol = sField->Size();

			switch (fIdx)
			{
				case FIELD_M:
				field = static_cast<complex<double>*> (sField->mCpu());
				vol = sField->eSize();
				break;

				case FIELD_V:
				field = static_cast<complex<double>*> (sField->vCpu());
				break;

				case FIELD_M2:
				if (sField->LowMem()) {
					printf ("Wrong field. Lowmem forbids the use of m2");
					return;
				}

				field = static_cast<complex<double>*> (sField->m2Cpu());
				vol = sField->eSize();
				break;

				default:
				printf ("Wrong field. Valid possibilities: FIELD_M, FIELD_M2 and FIELD_V");
				return;
				break;
			}

			#pragma omp parallel for default(shared) schedule(static)
			for (size_t lpc = 0; lpc < vol; lpc++)
			{
				double mod = abs(field[lpc]);
				field[lpc] /= mod;
			}

			break;
		}

		case FIELD_SINGLE:
		{
			complex<float> *field;
			size_t vol = sField->Size();

			switch (fIdx)
			{
				case FIELD_M:
				field = static_cast<complex<float> *> (sField->mCpu());
				vol = sField->eSize();
				break;

				case FIELD_V:
				field = static_cast<complex<float> *> (sField->vCpu());
				break;

				case FIELD_M2:
				if (sField->LowMem()) {
					printf ("Wrong field. Lowmem forbids the use of m2");
					return;
				}

				field = static_cast<complex<float> *> (sField->m2Cpu());
				vol = sField->eSize();
				break;

				default:
				printf ("Wrong field. Valid possibilities: FIELD_M, FIELD_M2 and FIELD_V");
				break;
			}

			#pragma omp parallel for default(shared) schedule(static)
			for (size_t lpc = 0; lpc < vol; lpc++)
			{
				float mod = abs(field[lpc]);
				field[lpc] /= mod;
			}

			break;
		}

		default:
		printf("Unrecognized precision\n");
		exit(1);
		break;
	}
}
