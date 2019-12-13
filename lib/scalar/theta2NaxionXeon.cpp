#include <complex>

#include "scalar/scalarField.h"

using namespace std;

void	th2NaxionXeon (Scalar *sField)
{
	/* Reduces the axion field to its non-relativistic part
		Uses the approximation
		P = (sqrt(m_ctheta) * ( ctheta - i ctheta'/sqrt(m_ctheta) ) exp(I phi )
		but the phase
			phi = int_0^ct dct m_ctheta
		is irrelevant later so we neglect it.

		The field is rebuilt in M,
		V becomes a second auxiliary field together with M2
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
			double *cP     = static_cast<double*> (sField->mCpu() + 2*sField->Precision()*S*NG );
			double *m2     = static_cast<double*> (sField->m2Cpu());

			#pragma omp parallel for default(shared) schedule(static)
			for (size_t lpc = 0; lpc < V; lpc++)
			{
				m2[2*lpc  ] =  cfield[lpc]*sqcms  ;
				m2[2*lpc+1] = -cveloc[lpc]/sqcms ;
			}
			memcpy(static_cast<char*>(static_cast<void*>(cP)),static_cast<char*>(static_cast<void*>(m2)),2*sField->Precision()*V);
			break;
		}

		case FIELD_SINGLE:
		{
			float sqcms = (float) sqrt(sField->AxionMass()*(*sField->RV()));
			float *cfield = static_cast<float*> (sField->mStart());
			float *cveloc = static_cast<float*> (sField->vCpu());
			float *cP     = static_cast<float*> (sField->mCpu() + 2*sField->Precision()*S*NG );
			float *m2     = static_cast<float*> (sField->m2Cpu());

			#pragma omp parallel for default(shared) schedule(static)
			for (size_t lpc = 0; lpc < V; lpc++)
			{
				m2[2*lpc  ] =  cfield[lpc]*sqcms  ;
				m2[2*lpc+1] = -cveloc[lpc]/sqcms ;
			}
			memcpy(static_cast<char*>(static_cast<void*>(cP)),static_cast<char*>(static_cast<void*>(m2)),2*sField->Precision()*V);
			break;
		}

		default:
		printf("Unrecognized precision\n");
		exit(1);
		break;
	}

}
