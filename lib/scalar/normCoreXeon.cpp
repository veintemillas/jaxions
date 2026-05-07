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

	const Float delta = field->Delta();
	const Float R = static_cast<Float>(*field->RV());
	const Float msa = sqrt(2*field->LambdaP())*R*delta;
	const size_t Nx = field->NX();
	const size_t Ny = field->NY();
	const size_t S = field->NXY();
	const size_t V = field->NXYZ();

	field->exchangeGhosts(FIELD_M);

	complex<Float> *mCp = static_cast<complex<Float>*> (field->mStart());
	complex<Float> *maux;

	if (field->LowMem()){
		maux = static_cast<complex<Float>*> (field->vCpu());
		LogError("NormcoreXeon deletes v!!!");
	}
	else
		maux = static_cast<complex<Float>*> (field->m2Cpu());

	#pragma omp parallel for default(shared) schedule(static)
	for (size_t idx=0; idx<V; idx++)
	{

		size_t X[3];
    size_t O[4];
		indexXeon::idx2VecNeigh(idx, X, O, Nx, Ny);

		complex<Float> fp_x = mCp[O[0]];
		complex<Float> fm_x = mCp[O[1]];
		complex<Float> fp_y = mCp[O[2]];
		complex<Float> fm_y = mCp[O[3]];
		complex<Float> fp_z = mCp[idx+S];
		complex<Float> fm_z = mCp[idx-S];
		complex<Float> f0   = mCp[idx];
		complex<Float> if0  = complex<Float>(1,0)/f0;

		if (imag(f0]) == ((Float) 0.0) && real(f0) == ((Float) 0.0)){
			rhof = 0.0;
		}
		else {
			
			Float gradx   = imag((fp_x - f0)*if0);
			Float gradtot = gradx*gradx;

			gradx   = imag((f0 - fm_x)*if0);
			gradtot += gradx*gradx;

			Float grady   = imag((fp_y - f0)*if0);
			gradtot += grady*grady;

			grady   = imag((f0 - fm_y)*if0);
			gradtot += grady*grady;

			Float gradz   = imag((fp_z - f0)*if0);
			gradtot += gradz*gradz;

			gradz   = imag((f0 - fm_z)*if0);
			gradtot += gradz*gradz;

			Float rhof = (float) 1.0;

			if (gradtot > 0.0000001)
			{
						Float sss  = msa/sqrt(gradtot/2.);
						Float sss2 = sss*sss;
						Float sss4 = sss2*sss2;
						rhof  = (0.43*sss + 0.164*sss2 + 0.036*sss4)/(1.0+0.39*sss+0.2*sss2+0.036*sss4);
			}
		}
		maux[idx] = mCp[idx]*rhof/abs(mCp[idx]);

	}

	//Copies maux to m
	memcpy (static_cast<char *>(field->mStart()), static_cast<char *>(static_cast<void *>(maux)), field->DataSize()*V);

	commSync();



}

void	normCoreXeon (Scalar *sField)
{

	sField->exchangeGhosts(FIELD_M);

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
