#include <complex>
#include <cstring>

#include "scalar/scalarField.h"

using namespace std;

template<typename Float>
void	toThetaKernelXeon (Scalar *sField)
{
	const size_t V  = sField->Size();
	const size_t S  = sField->Surf();
	const size_t Lz = sField->Depth();
	const size_t Go = 2*(V+S);
	Float *mField   = static_cast<Float*>(sField->mCpu());
	Float *cmField  = static_cast<complex<Float>*>(sField->mCpu());
	Float *vField   = static_cast<Float*>(sField->vCpu());
	Float *cvField  = static_cast<complex<Float>*>(sField->vCpu());

	for (size_t cZ = 1; cZ < Lz+1; cZ++)
	{
		const size_t Vo = cZ*S;
		#pragma omp parallel for default(shared) schedule(static)
		for (size_t lpc = 0; lpc < S; lpc++)
		{
			Float yDx      = cmField[Vo+lpc].imag()/cmField[Vo+lpc].real();
			Float dyDx     = cvField[Vo+lpc].imag()/cmField[Vo+lpc].real();
			Float dxDx     = cvField[Vo+lpc].real()/cmField[Vo+lpc].real();
			mField[Go+lpc] = (dyDx-yDx*dxDx)/(1.+yDx*yDx);
			mField[lpc]    = arg(cmField[Vo+lpc]);
		}

		memcpy (mField + Vo,   mField,      sizeof(Float)*S);
		memcpy (vField + Vo-S, mField + Go, sizeof(Float)*S);
	}
}

void	toThetaXeon (Scalar *sField)
{
	switch (sField->Precision())
	{
		case FIELD_DOUBLE:

			toThetaKernelXeon<double> (sField);
			break;

		case FIELD_SINGLE:

			toThetaKernelXeon<float> (sField);
			break;

		default:

			printf("Wrong precision\n);
			exit  (1);
			break;
	}
}
