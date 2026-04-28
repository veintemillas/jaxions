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
		double R     = (*sField->RV());
		double R2    = R*R;
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

		constexpr double twopi = 2.0*M_PI;
		// 1. Local sum of theta = psi/R over physical sites
		long double local_sum_theta = 0.0L;

		#pragma omp parallel for reduction(+:local_sum_theta) schedule(static)
		for (size_t lpc = 0; lpc < V; lpc++)
			local_sum_theta += (long double)cfield[lpc] / (long double) R;

		unsigned long long local_count = (unsigned long long) V;
		long double global_sum_theta = 0.0L;
		unsigned long long global_count = 0;

		MPI_Allreduce(&local_sum_theta, &global_sum_theta,
                  1, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		MPI_Allreduce(&local_count, &global_count,
					1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

		long double mean_theta = global_sum_theta / (long double) global_count;

		long long N = llround(mean_theta / (long double)twopi);

		LogMsg(VERB_NORMAL,"[thpX] N calculated %ld",N);

		// 3. Apply the same branch shift everywhere
	    Float shift_psi  = (Float)(twopi * (long double)N * (long double)R);

		// Assuming R'/R = frw/ct
		Float Hconf      = (Float)(frw / ct);
		Float shift_psip = shift_psi * Hconf;

		#pragma omp parallel for default(shared) schedule(static)
		for (size_t lpc = 0; lpc < V; lpc++)
		{
			// cfield[lpc] *= sqcms1 ;
			// cveloc[lpc] /= sqcms2 ;
			// cveloc[lpc] += cfield[lpc]*adiab ;

			Float psi  = cfield[lpc] - shift_psi;
			Float psip = cveloc[lpc] - shift_psip;

			cfield[lpc] = psi * sqcms1;
			cveloc[lpc] = psip / sqcms2 + cfield[lpc] * adiab;
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
