#include <cstdio>
#include <complex>

#include "comms/comms.h"
#include "scalar/scalarField.h"
#include "utils/parse.h"

using namespace std;
// WUITAR n1*n1 !!!
template<typename Float>
void	writeData	(complex<Float> *m, complex<Float> *v, const Float z, const size_t n1, FILE *atWrite, FILE *rhoWrite, FILE *densWrite)
{
	fprintf(atWrite,  "# %d %f %f %f %d \n", sizeN , sizeL , sizeL/sizeN , z , 0);
	fprintf(rhoWrite, "# %d %f %f %f %d \n", sizeN , sizeL , sizeL/sizeN , z , 0 );
	fprintf(densWrite, "# %d %f %f %f %d \n", sizeN , sizeL , sizeL/sizeN , z , 0);
	//int n2 = n1*n1;
	size_t nlast = n1*n1*sizeZ+n1*n1;
	for (size_t ly = 0; ly < n1; ly++)
	{
		for (size_t lx = 0; lx < n1; lx++)
		{

			fprintf(atWrite,  "%f ", arg(m[lx + n1*ly]) );
			fprintf(rhoWrite, "%f ", abs(m[lx + n1*ly])/z);
			fprintf(densWrite, "%f ", pow(abs(m[lx + n1*ly])/z,2)*( pow(arg(m[lx + n1*ly]),2)+
								pow(imag(m[lx + n1*ly + nlast]/m[lx + n1*ly]),2)/(9.0*pow(z,nQcd+2))) );


		}

		fprintf(atWrite , "\n");
		fprintf(rhoWrite, "\n");
		fprintf(densWrite, "\n");
	}
}

template<typename Float>
void	writeDatafromTheta	(Float *m, Float *v, const Float z, const size_t n1, FILE *atWrite, FILE *rhoWrite, FILE *densWrite)
{
	fprintf(atWrite,  "# %d %f %f %f %d \n", sizeN , sizeL , sizeL/sizeN , z , 1);
	fprintf(rhoWrite, "# %d %f %f %f %d \n", sizeN , sizeL , sizeL/sizeN , z , 1);
	fprintf(densWrite, "# %d %f %f %f %d \n", sizeN , sizeL , sizeL/sizeN , z , 1);
	//int n2 = n1*n1;
	size_t nlast = n1*n1*sizeZ + n1*n1;

	for (size_t ly = 0; ly < n1; ly++)
	{
		for (size_t lx = 0; lx < n1; lx++)
		{

			// m contains c_theta = z*theta
			fprintf(atWrite,  "%f ", m[lx + n1*ly]/z) ;
			fprintf(rhoWrite, "%f ", 1);
			fprintf(densWrite, "%f ", ( pow(m[lx + n1*ly]/z,2) + pow(m[lx + n1*ly + nlast],2)/(9.0*pow(z,nQcd+2))) );

		}

		fprintf(atWrite , "\n");
		fprintf(rhoWrite, "\n");
		fprintf(densWrite, "\n");
	}
}


void	writeMap	(Scalar *axion, const int index)
{
	if (commRank() != 0)
		return;

	//--------------------------------------------------
	// DIRECT SHORT OUTPUT PICTURES 2D
	//--------------------------------------------------

	char stoRoh[256];
	char stoAt[256];
	char stoDens[256];

	const int n1 = axion->Length();

	sprintf(stoRoh,  "out/rho/rho-%05d.txt",   index);
	sprintf(stoAt,   "out/at/at-%05d.txt",     index);
	sprintf(stoDens, "out/dens/dens-%05d.txt", index);

	FILE *atWrite = NULL;
	FILE *rhoWrite = NULL;
	FILE *densWrite = NULL;

	if ((atWrite  = fopen(stoAt, "w+")) == NULL)
	{
		printf ("Couldn't open file %s for writing\n", stoAt);
		return ;
	}

	if ((rhoWrite = fopen(stoRoh, "w+")) == NULL)
	{
		printf ("Couldn't open file %s for writing\n", stoRoh);
		return ;
	}

	if ((densWrite = fopen(stoDens, "w+")) == NULL)
	{
		printf ("Couldn't open file %s for writing\n", stoDens);
		return ;
	}

	if ( axion->Fieldo() == FIELD_SAXION)
	{
		switch (axion->Precision())
		{
			case FIELD_DOUBLE:
				writeData(static_cast<complex<double> *> (axion->mCpu()), static_cast<complex<double> *> (axion->vCpu()) ,         (*axion->zV()), n1, atWrite, rhoWrite, densWrite);
				break;

			case FIELD_SINGLE:
				writeData(static_cast<complex<float> *>  (axion->mCpu()), static_cast<complex<float> *>  (axion->vCpu()),  (float) (*axion->zV()), n1, atWrite, rhoWrite, densWrite);
				break;
		}
	}
	else // Fieldo() = FIELD_AXION
	{
		switch (axion->Precision())
		{
			case FIELD_DOUBLE:
				writeDatafromTheta(static_cast<double *> (axion->mCpu()), static_cast<double *> (axion->vCpu()),         (*axion->zV()), n1, atWrite, rhoWrite, densWrite);
				break;

			case FIELD_SINGLE:
				writeDatafromTheta(static_cast<float *>  (axion->mCpu()), static_cast<float *>  (axion->vCpu()), (float) (*axion->zV()), n1, atWrite, rhoWrite, densWrite);
				break;
		}
	}

	fclose(atWrite);
	fclose(rhoWrite);
	fclose(densWrite);
}
