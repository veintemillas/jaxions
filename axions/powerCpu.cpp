#include <complex>

#include "index.h"
#include "scalarField.h"

using namespace std;


void	analyzePow (Scalar *axion, double *powera, int *nmodes, const int K3, bool gpu, const int index)
{
	//--------------------------------------------------
	// POWER SPECTRUM
	//--------------------------------------------------

	const int n1 = axion->Length();
	const int NH = n1/2;

	axion->prepareCpu(window);
//	axion->transferGpu(FIELD_M2);
	axion->fftCpu();
	axion->squareCpu();
//	axion->transferCpu(FIELD_M2);

	#pragma omp parallel for default(shared) schedule(static)             
	for (int i=0; i < K3; i++)
	{
		powera[i] = 0.0;
		nmodes[i] = 0;
	}

	#pragma omp parallel for default(shared) private(NH)
	for (int idx=0; idx<axion->Size(); idx++)
	{
		int X[3], nX[3], A[3], B[3];

		idx2Vec(idx, X, n1);

		A[0] = X[0]/NH; B[0] = 1 - 2*A[0];
		A[1] = X[1]/NH; B[1] = 1 - 2*A[1];
		A[2] = X[2]/NH; B[2] = 1 - 2*A[2];

		nX[0] = NH*A[0] + B[0]*(X[0]%NH);
		nX[1] = NH*A[1] + B[1]*(X[1]%NH);
		nX[2] = NH*A[2] + B[2]*(X[2]%NH);

		double nn = nX[0]*nX[0] + nX[1]*nX[1] + nX[2]*nX[2];

		int kk = floor(sqrt(nn));

		#pragma omp critical
		{
			powera[kk] += ((complex<double> *) axion->mCpu())[idx].real();
			nmodes[kk]++;
		}
	}

	char stoPow[256];

	sprintf(stoPow, "out/pow/pow-%05d.dat", index);

	FILE *powWrite = NULL;

	if ((powWrite  = fopen(stoPow, "w+")) == NULL)
	{
		printf("Error: Couldn't open file %s for writing\n", stoPow);
		return 1;
	}

	for(kk=0; kk<K3; kk++)
		fprintf(powWrite,"%d %d %f \n", kk, nmodes[kk], powera[kk]);
   
	fclose(powWrite);           
  
	printf("\nPower spectrum printed\n");
}


