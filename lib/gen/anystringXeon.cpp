#include <complex>
#include <random>
#include <omp.h>

#include "scalar/scalarField.h"
#include "enum-field.h"
#include "utils/memAlloc.h"
#include "utils/parse.h"

#include "enum-field.h"
#include "comms/comms.h"

#include "gen/anystringXeon.h"

// ========== NEW entry point using your variables (computes θ directly) ==========
template<typename Float>
void anystringXeon (Scalar *field, IcData ic,
                         double *xs, double *ys, double *zs, size_t len,
                         int *eps, size_t eps_len)
{
    LogMsg(VERB_NORMAL,"[θX] Any-string θ via solid angle (triangle fan)");

    const size_t Lx = field->Length();
    const size_t Sf = field->Surf();
    const size_t V  = field->Size();
    const size_t Lz = field->Depth();

    Float* th = static_cast<Float*> (field->vCpu());

    // MPI split
    const int nSplit  = commSize();
    const int rank    = commRank();

    size_t local_z_start = size_t(rank) * Lz;

    LogMsg(VERB_NORMAL,"[θX] Calculation with %d copies in dipole approx with Bext field %d %.5e",ic.kMax, ic.Bext, ic.kcr);
    cthetaSolidAngleXeon(field->mStart(), Lx, Lz, local_z_start, Lz*nSplit, field->Precision(), len, xs, ys, zs, eps,ic.kMax, ic.Bext, ic.kcr);

}


void	anystringConf (Scalar *field, IcData ic, double *x, double *y, double *z, size_t len, int  *ep, size_t len_ep)
{
	switch (field->Precision())
	{
		case FIELD_DOUBLE:
		{
		std::complex<double>* ma;
		if (ic.fieldindex == FIELD_M){
		 	ma = static_cast<std::complex<double>*> (field->mStart());
			LogMsg(VERB_NORMAL,"[RC] Generating double conf in mS! ");
		}
		else if (ic.fieldindex == FIELD_V){
			ma = static_cast<std::complex<double>*> (field->vCpu());
			LogMsg(VERB_NORMAL,"[RC] Generating double conf in v! ");
		}
		else if (ic.fieldindex == FIELD_M2){
			ma = static_cast<std::complex<double>*> (field->m2Cpu());
			LogMsg(VERB_NORMAL,"[RC] Generating double conf in m2! ");
		}

		anystringXeon<double>(field, ic, x,y,z,len, ep, len_ep);
		}
		break;

		case FIELD_SINGLE:
		{
		std::complex<float>* ma;
		if (ic.fieldindex == FIELD_M){
			ma = static_cast<std::complex<float>*> (field->mStart());
			LogMsg(VERB_NORMAL,"[RC] Generating single conf in mS! ");
		}
		else if (ic.fieldindex == FIELD_V){
			ma = static_cast<std::complex<float>*> (field->vCpu());
			LogMsg(VERB_NORMAL,"[RC] Generating single conf in v! type %d",ic.smvarType);
		}
		else if (ic.fieldindex == FIELD_M2){
			ma = static_cast<std::complex<float>*> (field->m2Cpu());
			LogMsg(VERB_NORMAL,"[RC] Generating single conf in m2! ");
		}

		anystringXeon<float> (field, ic, x,y,z,len, ep, len_ep);
		}
		break;

		default:
		break;
	}
}
