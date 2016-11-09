#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include "scalarField.h"
#include "enum-field.h"
#include "RKParms.h"
#include "index.h"

using namespace std;

template<typename Float>
void	propSimpleCore (const complex<Float> *m, complex<Float> *v, complex<Float> *m2, const Float z2, const Float zQ,
			const Float dzc, const Float dzd, const Float ood2, const Float LL, const size_t Lx, const size_t Sf, const size_t Vf)
{
	#pragma omp parallel for default(shared) schedule(static)
	for (size_t idx=Sf; idx<Vf+Sf; idx++)
	{
		size_t X[3], idxPx, idxPy, idxMx, idxMy;

		complex<Float> mel, a, tmp;

		indexXeon::idx2Vec(idx, X, Lx);

		if (X[0] == Lx-1)
			idxPx = idx - Lx+1;
		else
			idxPx = idx+1;

		if (X[0] == 0)
			idxMx = idx + Lx-1;
		else
			idxMx = idx-1;

		if (X[1] == Lx-1)
			idxPy = idx - Sf + Lx;
		else
			idxPy = idx + Lx;

		if (X[1] == 0)
			idxMy = idx + Sf - Lx;
		else
			idxMy = idx - Lx;

		mel = m[idxMx] + m[idxPx] + m[idxPy] + m[idxMy] + m[idx+Sf] + m[idx-Sf];
		tmp = m[idx];

		a = (mel-((Float) 6.)*tmp)*ood2 + zQ - tmp*(((Float) LL)*(tmp.real()*tmp.real() + tmp.imag()*tmp.imag() - z2));

		mel = v[idx-Sf];
		mel += a*dzc;
		v[idx-Sf] = mel;
		mel *= dzd;
		tmp += mel;
		m2[idx] = tmp;

	}
}

void	propSimpleKernel(Scalar *field, const double LL, const double nQcd, const double delta, const double dz, const double c, const double d, bool st)
{
	const size_t Lx = field->Length();
	const size_t S = field->Surf();
	const size_t V = field->Size();

	switch(field->Precision())
	{
		case FIELD_DOUBLE:
		{
			double *z = field->zV();
			const double zR = *z;
			const double z2 =zR*zR;
			const double zQ = 9.*pow(zR, nQcd+3.);
			const double dzc = dz*c;
			const double dzd = dz*d;
			const double ood2 = 1/(delta*delta);

			if (st == true)
			{
				field->exchangeGhosts(FIELD_M2);
				propSimpleCore<double>(static_cast<const complex<double>*>(field->m2Cpu()), static_cast<complex<double>*>(field->vCpu()),
						       static_cast<complex<double>*>(field->mCpu()), z2, zQ, dzc, dzd, ood2, LL, Lx, S, V+S);
			} else {
				field->exchangeGhosts(FIELD_M);
				propSimpleCore<double>(static_cast<const complex<double>*>(field->mCpu()), static_cast<complex<double>*>(field->vCpu()),
						       static_cast<complex<double>*>(field->m2Cpu()), z2, zQ, dzc, dzd, ood2, LL, Lx, S, V+S);
			}

			*z += dzd;

			break;
		}

		case FIELD_SINGLE:
		{
			double *z = field->zV();
			const double zR = *z;
			const float  z2 = zR*zR;
			const float  zQ = 9.f*powf(((float) zR), 3.f+((float) nQcd));
			const float  dzc = dz*c;
			const double dzd = dz*d;
			const float  ood2 = 1.f/(delta*delta);

			if (st == true)
			{
				field->exchangeGhosts(FIELD_M2);
				propSimpleCore<float>(static_cast<const complex<float>*>(field->m2Cpu()), static_cast<complex<float>*>(field->vCpu()),
						      static_cast<complex<float>*>(field->mCpu()), z2, zQ, dzc, dzd, ood2, LL, Lx, S, V+S);
			} else {
				field->exchangeGhosts(FIELD_M);
				propSimpleCore<float>(static_cast<const complex<float>*>(field->mCpu()), static_cast<complex<float>*>(field->vCpu()),
						      static_cast<complex<float>*>(field->m2Cpu()), z2, zQ, dzc, dzd, ood2, LL, Lx, S, V+S);
			}

			*z += dzd;

			break;
		}

		default:
		printf ("Not a valid precision.\n");

		break;
	}
}

void	propagateSimple(Scalar *field, const double dz, const double LL, const double nQcd, const double delta)
{
	propSimpleKernel(field, LL, nQcd, delta, dz, C1, D1, 0);
	propSimpleKernel(field, LL, nQcd, delta, dz, C2, D2, 1);
	propSimpleKernel(field, LL, nQcd, delta, dz, C3, D3, 0);
	propSimpleKernel(field, LL, nQcd, delta, dz, C4, D4, 1);
}
