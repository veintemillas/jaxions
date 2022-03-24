#include <complex>

#include "scalar/scalarField.h"

using namespace std;

template<typename Float>
void	th2PaxionXeon (Scalar *sField)
{
	/* Prepares axion into Paxion mode by
			renormalising it */

		const size_t NG = sField->getNg();
		const size_t V  = sField->Size();
		const size_t S  = sField->Surf();

		double mA    = sField->AxionMass();
		double R2    = (*sField->RV())*(*sField->RV());
		double ct    = (*sField->zV());
		double frw   = (sField->BckGnd()->Frw());
		double mA2   = (sField->AxionMassSq());
		double DmA2ct= sField->BckGnd()->DAxionMass2Dct(*sField->zV());

		Float sqcms1 = (Float) sqrt(0.5*mA*(*sField->RV()));
		Float sqcms2 = (Float) sqrt(2.0*mA*(*sField->RV()));
		Float adiab  = (Float) (DmA2ct + 2*frw*mA2*R2/ct)/(4*mA2*mA) ;
		Float *cfield = static_cast<Float*> (sField->mStart());
		Float *cveloc = static_cast<Float*> (sField->vCpu());
		Float *faxion = static_cast<Float*> (sField->m2Cpu());

		#pragma omp parallel for default(shared) schedule(static)
		for (size_t lpc = 0; lpc < V; lpc++)
		{
			cfield[lpc] *= sqcms1 ;
			cveloc[lpc] /= sqcms2 ;
			cveloc[lpc] += cfield[lpc]*adiab ;
		}

	/* Paxion velocity will be Ghosted! */
	memmove(static_cast<char*>(sField->vCpu())+NG*S*sField->Precision(),sField->vCpu(),V*sField->Precision());
}

void th2PaxionXeon(Scalar *axionField)
{
	if (axionField->Precision()==FIELD_SINGLE)
		th2PaxionXeon<float>(axionField);
	else
		th2PaxionXeon<double>(axionField);
}
