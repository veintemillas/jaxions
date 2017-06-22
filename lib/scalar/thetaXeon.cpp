#include <complex>
#include <cstring>

#include "scalar/scalarField.h"

using namespace std;

template<typename Float>
void	toThetaKernelXeon (Scalar *sField, const Float shift)
{
	const size_t V  = sField->Size();
	const size_t S  = sField->Surf();
	const size_t Lz = sField->Depth();
	const size_t Go = 2*(V+S);
	//POINTERS FOR THETA
	//(conformal)THETA STARTS AT m
	Float *mField   = static_cast<Float*>(sField->mCpu());
	//(conformal)THETA' will be stored in the (almost) second half of m
	Float *vField   = static_cast<Float*>(sField->mCpu()) + 2*S + V;
	//but we need an intermediate storage array
	Float *vFieldaux   = static_cast<Float*>(sField->vCpu()) ;
	//POINTERS FOR COMPLEX PQ FIELD
	complex<Float> *cmField  = static_cast<complex<Float>*>(sField->mCpu());
	complex<Float> *cvField  = static_cast<complex<Float>*>(sField->vCpu());

	const Float z = static_cast<Float>(sField->zV()[0]);

		for (size_t cZ = 1; cZ < Lz+1; cZ++)
	{
		const size_t Vo = cZ*S;
		#pragma omp parallel for default(shared) schedule(static)
		for (size_t lpc = 0; lpc < S; lpc++)
		{
			complex<Float> mTmp = cmField[Vo+lpc] - complex<Float>(shift,0.);
			//aux quantity z/|m|^2
			Float iMod      = z/(mTmp.real()*mTmp.real() + mTmp.imag()*mTmp.imag());
			//theta, starts reading after buffer, copies into buffer
			mField[lpc]     = arg(mTmp);
			//c_theta' = Im [v conj(m)]z/|m|^2+theta
			// the v array has no buffer hence the need of -S in their index
			// gets temporarily stored outside the Physical range of mField V+2S and vField V -> i.e. 2(V+S)=Go
			mField[Go+lpc]  = (cvField[Vo-S+lpc]*conj(mTmp)).imag()*iMod + mField[lpc];
			//mField[Go+lpc]  = z*(cvField[Vo-S+lpc]/mTmp).imag() + mField[lpc];
			//c_theta
			mField[lpc]    *= z;
		}
		//displaces the mField from buffer to position
		memcpy (mField + Vo,   mField,      sizeof(Float)*S);
		//displaces the vField from buffer' to aux position in varray (but in zone already read)
		memcpy (vFieldaux + Vo-S, mField + Go, sizeof(Float)*S);
	}

	// copies v from auxiliary position to final position in second half of complex m
	memcpy (vField, vFieldaux, sizeof(Float)*V);


}

void	toThetaXeon (Scalar *sField, const double shift)
{
	switch (sField->Precision())
	{
		case FIELD_DOUBLE:

			toThetaKernelXeon<double> (sField, shift);
			break;

		case FIELD_SINGLE:

			toThetaKernelXeon<float> (sField, (float) shift);
			break;

		default:

			printf("Wrong precision\n");
			exit  (1);
			break;
	}
}
