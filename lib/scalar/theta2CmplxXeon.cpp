#include <complex>

#include "scalar/scalarField.h"

using namespace std;


/* builds cphi, cphi_ct from theta, theta_ct
	 with VEV = 1
	 as
	 cphi    = R exp(I theta)
	 cphi_ct = R' exp(I theta) + R I theta_ct exp(I theta) = (R'/R + I theta_ct) cphi */

void	th2cxXeon (Scalar *sField)
{
	switch (sField->Precision())
	{
		case FIELD_DOUBLE:
		{
			// takes into account extra ghosts Ng
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
				field2[lpc] = field[lpc]*(ir+II*real(field2[lpc]));
			}
			break;
		}

		case FIELD_SINGLE:
		{
			// takes into account extra ghosts Ng
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
				field2[lpc] = field[lpc]*(ir + II*real(field2[lpc]));
			}

			break;
		}

		default:
		printf("Unrecognized precision\n");
		exit(1);
		break;
	}
}

template<typename Float>
static void th2cxM2XeonKernel (Scalar *sField)
{
	const Float * __restrict__ in = static_cast<const Float *>(sField->mCpu());
	complex<Float> * __restrict__ out = static_cast<complex<Float> *>(sField->m2Cpu());
	const Float ir = Float(1)/Float(*sField->RV());

	#pragma omp parallel for default(shared) schedule(static)
	for (size_t idx = 0; idx < sField->eSize(); ++idx) {
		const Float theta = in[idx]*ir;
		out[idx] = complex<Float>(cos(theta), sin(theta));
	}
}

void th2cxM2Xeon (Scalar *sField)
{
	if (sField->Precision() == FIELD_DOUBLE)
		th2cxM2XeonKernel<double>(sField);
	else if (sField->Precision() == FIELD_SINGLE)
		th2cxM2XeonKernel<float>(sField);
}
