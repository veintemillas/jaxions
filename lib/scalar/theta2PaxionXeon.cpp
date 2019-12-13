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

	switch (sField->Precision())
	{
		case FIELD_DOUBLE:
		{
			double sqcms = sqrt(sField->AxionMass()*(*sField->RV()));
			double *cfield = static_cast<double*> (sField->mStart());
			double *cveloc = static_cast<double*> (sField->vCpu());

			#pragma omp parallel for default(shared) schedule(static)
			for (size_t lpc = 0; lpc < V; lpc++)
			{
				cfield[lpc] *= sqcms  ;
				cveloc[lpc] /= -sqcms ;
			}
			break;
		}

		case FIELD_SINGLE:
		{
			float sqcms = (float) sqrt(sField->AxionMass()*(*sField->RV()));
			float *cfield = static_cast<float*> (sField->mStart());
			float *cveloc = static_cast<float*> (sField->vCpu());

			#pragma omp parallel for default(shared) schedule(static)
			for (size_t lpc = 0; lpc < V; lpc++)
			{
				cfield[lpc] *= sqcms  ;
				cveloc[lpc] /= -sqcms ;
			}
			break;
		}

		default:
		printf("Unrecognized precision\n");
		exit(1);
		break;
	}

}
