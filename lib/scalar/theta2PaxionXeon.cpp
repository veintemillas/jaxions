#include <complex>

#include "scalar/scalarField.h"

using namespace std;

void	th2PaxionXeon (Scalar *sField)
{
	/* Prepares axion into Paxion mode by
			renormalising it
		*/

		const size_t NG = sField->getNg();
		const size_t V  = sField->Size();
		const size_t S  = sField->Surf();

		double mA    = sField->AxionMass();
		double R2    = (*sField->RV())*(*sField->RV());
		double ct    = (*sField->zV());
		double frw   = (sField->BckGnd()->Frw());
		double mA2   = (sField->AxionMassSq());
		double DmA2ct= sField->BckGnd()->DAxionMass2Dct(*sField->zV());

	switch (sField->Precision())
	{
		case FIELD_DOUBLE:
		{

			double sqcms = sqrt(mA*(*sField->RV()));
			double adiab = (DmA2ct + 2*frw*mA2*R2/ct)/(4*mA2*mA) ;
			double *cfield = static_cast<double*> (sField->mStart());
			double *cveloc = static_cast<double*> (sField->vCpu());

			#pragma omp parallel for default(shared) schedule(static)
			for (size_t lpc = 0; lpc < V; lpc++)
			{
				cfield[lpc] *= sqcms  ;
				cveloc[lpc] /= sqcms ;
				cveloc[lpc] += cfield[lpc]*adiab ;
			}
			break;
		}

		case FIELD_SINGLE:
		{

			float sqcms = (float) sqrt(mA*(*sField->RV()));
			float adiab = (float) (DmA2ct + 2*frw*mA2*R2/ct)/(4*mA2*sField->AxionMass()) ;
			float *cfield = static_cast<float*> (sField->mStart());
			float *cveloc = static_cast<float*> (sField->vCpu());

			#pragma omp parallel for default(shared) schedule(static)
			for (size_t lpc = 0; lpc < V; lpc++)
			{
				cfield[lpc] *= sqcms  ;
				cveloc[lpc] /= sqcms ;
				cveloc[lpc] += cfield[lpc]* ((float) adiab) ;
			}
			break;
		}

		default:
		printf("Unrecognized precision\n");
		exit(1);
		break;
	}
	/* Paxion velocity is Ghosted! */
	memmove(static_cast<char*>(sField->vStart())+NG*S*sField->Precision(),sField->vCpu(),V*sField->Precision());
}
