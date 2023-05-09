#include <complex>
#include <cstring>

#include "scalar/scalarField.h"
#include "utils/index.h"
#include "utils/utils.h"

using namespace std;

template<typename Float>
void	iteraXeon (const complex<Float> * __restrict__ mCp, complex<Float> * __restrict__ vCp, const size_t Lx, const size_t S, const size_t V, const Float alpha)
{
	const Float One = 1.;
	const Float OneSixth = (1./6.);

	#pragma omp parallel for default(shared) schedule(static)
	for (size_t idx=0; idx<V; idx++)
	{
		long long int iMz;
		size_t iPx, iMx, iPy, iMy, iPz,  X[3];
		indexXeon::idx2Vec (idx, X, Lx);

		if (X[0] == 0)
		{
			iPx = idx + 1;
			iMx = idx + Lx - 1;
		} else {
			if (X[0] == Lx - 1)
			{
				iPx = idx - Lx + 1;
				iMx = idx - 1;
			} else {
				iPx = idx + 1;
				iMx = idx - 1;
			}
		}

		if (X[1] == 0)
		{
			iPy = idx + Lx;
			iMy = idx + S - Lx;
		} else {
			if (X[1] == Lx - 1)
			{
				iPy = idx - S + Lx;
				iMy = idx - Lx;
			} else {
				iPy = idx + Lx;
				iMy = idx - Lx;
			}
		}

		iPz = idx + S;
		iMz = (long long int) idx - (long long int) S;
		//Uses v to copy the smoothed configuration
		vCp[idx]   = alpha*mCp[idx] + OneSixth*(One-alpha)*(mCp[iPx] + mCp[iMx] + mCp[iPy] + mCp[iMy] + mCp[iPz] + mCp[iMz]);
	}

}

void	smoothXeon (Scalar *field, const size_t iter, const double alpha)
{
	LogMsg(VERB_SILENT,"[smo] Called smoothXeon");
	field->exchangeGhosts(FIELD_M);

	switch	(field->Precision())
	{
		case	FIELD_DOUBLE:
		for (size_t it=0; it<iter; it++)
		{
			LogMsg(VERB_HIGH,"[smo] Smoothing step %d",it);
			iteraXeon(static_cast<const complex<double>*>(field->mStart()), static_cast<complex<double>*>(field->m2Cpu()), field->Length(), field->Surf(), field->Size(), alpha);
			memcpy (static_cast<char *>(field->mStart()), static_cast<char*>(field->m2Cpu()), field->DataSize()*field->Size());
			field->exchangeGhosts(FIELD_M);
		}
		break;

		case	FIELD_SINGLE:
		for (size_t it=0; it<iter; it++)
		{
			LogMsg(VERB_HIGH,"[smo] Smoothing step %d",it);
			iteraXeon(static_cast<const complex<float>*>(field->mStart()), static_cast<complex<float>*>(field->m2Cpu()), field->Length(), field->Surf(), field->Size(), static_cast<float>(alpha));
			memcpy (static_cast<char *>(field->mStart()), static_cast<char*>(field->m2Cpu()), field->DataSize()*field->Size());
			field->exchangeGhosts(FIELD_M);
		}
		break;

		default:
		LogError ("Unrecognized precision");
		exit(1);
		break;
	}
	LogMsg(VERB_SILENT,"[smo] END smoothXeon ");
}
