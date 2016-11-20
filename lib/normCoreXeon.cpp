#include <complex>
#include <cstring>

#include "scalarField.h"
#include "parse.h"
#include "index.h"

using namespace std;

template<typename Float>
void normCoreKernelXeon (Scalar *field, Float alph)
{
	const Float deltaa = sizeL/sizeN;
	const Float LLa = LL;
	const Float zia = static_cast<Float>(*field->zV());

	const size_t n1 = field->Length();
	const size_t n2 = field->Surf();
	const size_t n3 = field->Size();

	field->exchangeGhosts(FIELD_M);

	complex<Float> *mCp = static_cast<complex<Float>*> (field->mCpu());
	complex<Float> *vCp = static_cast<complex<Float>*> (field->vCpu());

	printf("Entering CORE smoothing\n");
	fflush (stdout);

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)
		for (size_t idx=0; idx<n3; idx++)
		{
			size_t iPx, iMx, iPy, iMy, iPz, iMz, X[3];
			indexXeon::idx2Vec (idx, X, n1);

			Float gradx, grady, gradz, sss, sss2, sss4, rhof;

			if (X[0] == 0)
			{
				iPx = idx + 1;
				iMx = idx + n1 - 1;
			} else {
				if (X[0] == n1 - 1)
				{
					iPx = idx - n1 + 1;
					iMx = idx - 1;
				} else {
					iPx = idx + 1;
					iMx = idx - 1;
				}
			}

			if (X[1] == 0)
			{
				iPy = idx + n1;
				iMy = idx + n2 - n1;
			} else {
				if (X[1] == n1 - 1)
				{
					iPy = idx - n2 + n1;
					iMy = idx - n1;
				} else {
					iPy = idx + n1;
					iMy = idx - n1;
				}
			}

			iPz = idx + n2;
			iMz = idx - n2;
			//Uses v to copy the smoothed configuration
			//vCp[idx]   = alph*mCp[idx+n2] + OneSixth*(One-alph)*(mCp[iPx+n2] + mCp[iMx+n2] + mCp[iPy+n2] + mCp[iMy+n2] + mCp[iPz+n2] + mCp[iMz+n2]);
			//vCp[idx]   = vCp[idx]/abs(vCp[idx]);
			gradx = imag((mCp[iPx+n2] - mCp[iMx+n2])/mCp[idx+n2]);
			grady = imag((mCp[iPy+n2] - mCp[iMy+n2])/mCp[idx+n2]);
			gradz = imag((mCp[iPz+n2] - mCp[iMz+n2])/mCp[idx+n2]);
			//JAVIER added an artificial factor of 1.0, can be changed
			sss  = 1.0*sqrt(LLa)*zia*2.0*deltaa/sqrt(gradx*gradx + grady*grady + gradz*gradz);
			//rhof  = 0.5832*sss*(sss+1.0)*(sss+1.0)/(1.0+0.5832*sss*(1.5 + 2.0*sss + sss*sss));
			sss2 = sss*sss;
			sss4 = sss2*sss2;
			rhof  = (0.6081*sss+0.328*sss2+0.144*sss4)/(1.0+0.5515*sss+0.4*sss2+0.144*sss4);

			vCp[idx] = mCp[idx+n2]*rhof/abs(mCp[idx+n2]);

			//if(idx % sizeN*sizeN*10 == 0)
			//{
			//printf("CORE sets, (%f,%f,%f,%f,%f,%f,%f)= \n", gradx,sss,rhof,abs(vCp[idx]),sqrt(LLa),zia,deltaa);
			//}

		}
	}

	//Copies v to m
	memcpy (static_cast<char *>(field->mCpu()) + field->DataSize()*n2, field->vCpu(), field->DataSize()*n3);
	field->exchangeGhosts(FIELD_M);

	printf("CORE smoothing Done\n");
	fflush (stdout);
}

void	normCoreXeon (Scalar *sField, const double alph)
{
	switch (sField->Precision())
	{
		case FIELD_DOUBLE:

			normCoreKernelXeon<double> (sField, alph);
			break;

		case FIELD_SINGLE:

			normCoreKernelXeon<float> (sField, static_cast<float>(alph));
			break;

		default:
			printf("Unrecognized precision\n");
			exit(1);
			break;
	}
}
