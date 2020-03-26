#include <complex>
#include <cstring>
#include "comms/comms.h"

#include "scalar/scalarField.h"
#include "utils/parse.h"
#include "utils/index.h"

using namespace std;

template<typename Float>
void normCoreKernelXeon (Scalar *field)
{

	//printf("Entering CORE smoothing ");
	//fflush (stdout);

	const Float deltaa = field->Delta();
	const Float zia = static_cast<Float>(*field->RV());

	Float LLa = field->LambdaP();

	const size_t n1 = field->Length();
	const size_t n2 = field->Surf();
	const size_t n3 = field->Size();

	field->exchangeGhosts(FIELD_M);

	complex<Float> *mCp = static_cast<complex<Float>*> (field->mStart());
	complex<Float> *vCp = static_cast<complex<Float>*> (field->vCpu());

	#pragma omp parallel for default(shared) schedule(static)
	for (size_t idx=0; idx<n3; idx++)
	{
		size_t iPx, iMx, iPy, iMy, iPz, iMz, X[3];
		indexXeon::idx2Vec (idx, X, n1);

		Float gradtot, gradx, grady, gradz, sss, sss2, sss4, rhof;

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

		gradx = imag((mCp[iPx] - mCp[idx])/mCp[idx]);
		gradtot = gradx*gradx ;
		gradx = imag((mCp[idx] - mCp[iMx])/mCp[idx]);
		gradtot += gradx*gradx ;
		grady = imag((mCp[iPy] - mCp[idx])/mCp[idx]);
		gradtot += grady*grady ;
		grady = imag((mCp[idx] - mCp[iMy])/mCp[idx]);
		gradtot += grady*grady ;
		gradz = imag((mCp[iPz] - mCp[idx])/mCp[idx]);
		gradtot += gradz*gradz ;
		gradz = imag((mCp[idx] - mCp[iMz])/mCp[idx]);
		gradtot += gradz*gradz ;

		if (gradtot > 0.0000001)
		{
					sss  = 2.0*sqrt(LLa)*zia*deltaa/sqrt(gradtot);
					//rhof  = 0.5832*sss*(sss+1.0)*(sss+1.0)/(1.0+0.5832*sss*(1.5 + 2.0*sss + sss*sss));
					sss2 = sss*sss;
					sss4 = sss2*sss2;
					// rhof  = (0.6081*sss+0.328*sss2+0.144*sss4)/(1.0+0.5515*sss+0.4*sss2+0.144*sss4);
					rhof  = (0.43*sss + 0.164*sss2 + 0.036*sss4)/(1.0+0.39*sss+0.2*sss2+0.036*sss4);
					// sss  = sqrt(LLa)*zia*deltaa/sqrt(gradtot);
					// if (sss < 1.64447) {
					// rhof  = 0.6081*sss ;
					// }
					// else
					// {
					// 	rhof = 1.0	;
					// }


		}
		else
		{
			//printf("shock!");
			rhof = 1.0 ;
		}
		vCp[idx] = mCp[idx]*rhof/abs(mCp[idx]);

		//if(idx % sizeN*sizeN*10 == 0)
		//{
		//printf("CORE sets, (%f,%f,%f,%f,%f,%f,%f)= \n", gradx,sss,rhof,abs(vCp[idx]),sqrt(LLa),zia,deltaa);
		//}

	}

	//Copies v to m
	memcpy (static_cast<char *>(field->mStart()), field->vCpu(), field->DataSize()*n3);
	field->exchangeGhosts(FIELD_M);

	commSync();



}

void	normCoreXeon (Scalar *sField)
{
	switch (sField->Precision())
	{
		case FIELD_DOUBLE:

			normCoreKernelXeon<double> (sField);
			break;

		case FIELD_SINGLE:

			normCoreKernelXeon<float> (sField);
			break;

		default:
			printf("Unrecognized precision\n");
			exit(1);
			break;
	}
}
