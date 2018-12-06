#include <complex>

#include "scalar/scalarField.h"

using namespace std;

void	th2cxXeon (Scalar *sField)
{
	switch (sField->Precision())
	{
		case FIELD_DOUBLE:
		{
			complex<double> *field, *field2;
			complex<double> II = complex<double>{0,1};
			size_t vol = sField->Size();
			double r = *sField->RV();
			double ir = 1/r;

			// The real part of m field has theta already normalised
			field = static_cast<complex<double>*> (sField->mStart());
			// The real part of v field has theta' already normalised
			field2 = static_cast<complex<double>*> (sField->vCpu());

			#pragma omp parallel for default(shared) schedule(static)
			for (size_t lpc = 0; lpc < vol; lpc++)
			{
				field[lpc] = r*exp(II*real(field[lpc]));
				// asumes rho' = 0
				field2[lpc] = field[lpc]*(ir+II*field2[lpc]);
			}
			break;
		}

		case FIELD_SINGLE:
		{
			complex<float> *field, *field2;
			complex<float> II = complex<float>{0,1};
			size_t vol = sField->Size();
			float r = *sField->RV();
			float ir = 1/r;

			// The real part of m field has theta already normalised
			field = static_cast<complex<float>*> (sField->mStart());
			// The real part of v field has theta' already normalised
			field2 = static_cast<complex<float>*> (sField->vCpu());

			#pragma omp parallel for default(shared) schedule(static)
			for (size_t lpc = 0; lpc < vol; lpc++)
			{
				field[lpc] = r*exp(II*real(field[lpc]));
				// asumes rho' = 0
				field2[lpc] = field[lpc]*(ir + II*field2[lpc]);
			}

			break;
		}

		default:
		printf("Unrecognized precision\n");
		exit(1);
		break;
	}
}
