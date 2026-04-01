#include <complex>
#include <cstring>

#include "scalar/scalarField.h"
#include "utils/utils.h"

using namespace std;

template<typename Float>
void	iteraXeon (const complex<Float> * __restrict__ mCp, complex<Float> * __restrict__ vCp, const size_t Lx, const size_t S, const size_t V, const Float alpha)
{
	const Float One = 1.;
	const Float OneSixth = (1./6.);
	const Float beta = OneSixth*(One-alpha);

	// Since S = Lx * Ly, recover Ly from S and Lx
	const size_t Ly = S / Lx;

	#pragma omp parallel for default(shared) schedule(static)
	for (size_t idx=0; idx<V; idx++)
	{
		size_t X[3];
		size_t O[4];

		indexXeon::idx2VecNeigh(idx, X, O, Lx, Ly);

		const size_t iPx = O[0];
		const size_t iMx = O[1];
		const size_t iPy = O[2];
		const size_t iMy = O[3];
		const size_t iPz = idx + S;
		const size_t iMz = idx - S;

		//Uses v to copy the smoothed configuration
		vCp[idx]   = alpha*mCp[idx] + beta*(mCp[iPx] + mCp[iMx] + mCp[iPy] + mCp[iMy] + mCp[iPz] + mCp[iMz]);
	}

}

void	smoothXeon (Scalar *field, const size_t iter, const double alpha)
{
	LogMsg(VERB_SILENT,"[smo] Called smoothXeon");
	field->exchangeGhosts(FIELD_M);

	// m into m2
	if (!field->LowMem())
	{
		size_t peter = iter/2;
		switch	(field->Precision())
		{
			case	FIELD_DOUBLE:
			for (size_t it=0; it<peter; it++)
			{
				LogMsg(VERB_PARANOID,"[smo] Smoothing step %d",it);
				iteraXeon(static_cast<const complex<double>*>(field->mStart()), static_cast<complex<double>*>(field->m2Start()), field->Length(), field->Surf(), field->Size(), alpha);
				field->exchangeGhosts(FIELD_M2);
				iteraXeon(static_cast<const complex<double>*>(field->m2Start()), static_cast<complex<double>*>(field->mStart()), field->Length(), field->Surf(), field->Size(), alpha);
				field->exchangeGhosts(FIELD_M);
			}
			break;

			case	FIELD_SINGLE:
			for (size_t it=0; it<peter; it++)
			{
				LogMsg(VERB_PARANOID,"[smo] Smoothing step %d",it);
				iteraXeon(static_cast<const complex<float>*>(field->mStart()), static_cast<complex<float>*>(field->m2Start()), field->Length(), field->Surf(), field->Size(), static_cast<float>(alpha));
				field->exchangeGhosts(FIELD_M2);
				iteraXeon(static_cast<const complex<float>*>(field->m2Start()), static_cast<complex<float>*>(field->mStart()), field->Length(), field->Surf(), field->Size(), static_cast<float>(alpha));
				field->exchangeGhosts(FIELD_M);
			}
			break;

			default:
			LogError ("Unrecognized precision");
			exit(1);
			break;
		}
	}
	else
	{
		switch	(field->Precision())
		{
			case	FIELD_DOUBLE:
			for (size_t it=0; it<iter; it++)
			{
				LogMsg(VERB_PARANOID,"[smo] Smoothing step %d",it);
				iteraXeon(static_cast<const complex<double>*>(field->mStart()), static_cast<complex<double>*>(field->vCpu()), field->Length(), field->Surf(), field->Size(), alpha);
				memcpy (static_cast<char *>(field->mStart()), static_cast<char*>(field->vCpu()), field->DataSize()*field->Size());
				field->exchangeGhosts(FIELD_M);
			}
			break;

			case	VERB_PARANOID:
			for (size_t it=0; it<iter; it++)
			{
				LogMsg(VERB_HIGH,"[smo] Smoothing step %d",it);
				iteraXeon(static_cast<const complex<float>*>(field->mStart()), static_cast<complex<float>*>(field->vCpu()), field->Length(), field->Surf(), field->Size(), static_cast<float>(alpha));
				memcpy (static_cast<char *>(field->mStart()), static_cast<char*>(field->vCpu()), field->DataSize()*field->Size());
				field->exchangeGhosts(FIELD_M);
			}
			break;

			default:
			LogError ("Unrecognized precision");
			exit(1);
			break;
		}
	}
	LogMsg(VERB_SILENT,"[smo] END smoothXeon ");
}
