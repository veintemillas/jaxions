#include <cstdio>
#include <complex>

#include "comms/comms.h"
#include "scalar/scalarField.h"
#include "utils/parse.h"
#include "scalar/varNQCD.h"

using namespace std;
// WUITAR n1*n1 !!!
template<typename Float>
void	writeData	(complex<Float> *m, complex<Float> *v, const Float z, const size_t n1, FILE *atWrite, FILE *rhoWrite, FILE *densWrite)
{
	fprintf(atWrite,   "# %d %f %f %f %d \n", sizeN , sizeL , sizeL/sizeN , z , 0);
	fprintf(rhoWrite,  "# %d %f %f %f %d \n", sizeN , sizeL , sizeL/sizeN , z , 0);
	fprintf(densWrite, "# %d %f %f %f %d \n", sizeN , sizeL , sizeL/sizeN , z , 0);
	//int n2 = n1*n1;
	size_t nlast = n1*n1*sizeZ+n1*n1;

	const Float iZ = 1./z;
	//const Float zQ = 1./(9.*pow(z,nQcd+2));
	//const Float zQ = 3.*pow(z,nQcd/2.+1);
	const Float zQ = (Float) axionmass((double) z, nQcd, zthres, zrestore)*((double) z);
	for (size_t ly = 0; ly < n1; ly++)
	{
		for (size_t lx = 0; lx < n1; lx++)
		{
			Float thethis = arg(m[lx + n1*ly]);
			fprintf(atWrite,   "%f ", thethis);
			fprintf(rhoWrite,  "%f ", abs(m[lx + n1*ly])*iZ);
			fprintf(densWrite, "%f ", pow(abs(m[lx + n1*ly])*iZ,2)*
						(	(pow(thethis*z,2)*zQ + pow(imag(m[lx + n1*ly + nlast]/m[lx + n1*ly])*z+thethis,2)/zQ) ) );


		}

		fprintf(atWrite ,  "\n");
		fprintf(rhoWrite,  "\n");
		fprintf(densWrite, "\n");
	}
}

template<typename Float>
void	writeDatafromTheta	(Float *m, Float *v, const Float z, const size_t n1, FILE *atWrite, FILE *rhoWrite, FILE *densWrite)
{
	fprintf(atWrite,   "# %d %f %f %f %d \n", sizeN , sizeL , sizeL/sizeN , z , 1);
	fprintf(rhoWrite,  "# %d %f %f %f %d \n", sizeN , sizeL , sizeL/sizeN , z , 1);
	fprintf(densWrite, "# %d %f %f %f %d \n", sizeN , sizeL , sizeL/sizeN , z , 1);
	//int n2 = n1*n1;
	size_t nlast = n1*n1*sizeZ + n1*n1;

	const Float iZ = 1./z;
	//const Float zQ = 1./(9.*pow(z,nQcd+2));
	//const Float zQ = 3.*pow(z,nQcd/2.+1);
	const Float zQ = (Float) axionmass((double) z, nQcd, zthres, zrestore)*((double) z);

	for (size_t ly = 0; ly < n1; ly++)
	{
		for (size_t lx = 0; lx < n1; lx++)
		{

			// m contains c_theta = z*theta
			fprintf(atWrite,   "%f ", m[lx + n1*ly]*iZ) ;
			fprintf(rhoWrite,  "%f ", 1.);
			fprintf(densWrite, "%f ", ( pow(m[lx + n1*ly],2)*zQ + pow(m[lx + n1*ly + nlast],2)/zQ) );
		}

		fprintf(atWrite ,  "\n");
		fprintf(rhoWrite,  "\n");
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

	if ( axion->Field() == FIELD_SAXION)
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
	else // Field() = FIELD_AXION
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

//--------------------------------------------------
// OVERDENSITY MAP 3D
//--------------------------------------------------

//
// template<typename Float>
// void	writeDatafromTheta3D	(Float *m2, const Float z, const size_t n1, const size_t size, FILE *DDDWrite)
// {
// 	fprintf(DDDWrite,   "# %d %f %f %f %d \n", sizeN , sizeL , sizeL/sizeN , z , 1);
//
// 	// ASSUMES M2 CONTAINS DENSITY CONTRAST
// 	// ASSUMES M2 UNFOLDED
// 	// PRINTS ONLY POINTS WITH D>10
// 	size_t n2 = n1*n1 ;
//
// 	#pragma omp parallel for default(shared) schedule(static)
// 	for (size_t idx = 0; idx < size; idx++)
// 	{
// 		int ix, iy, iz;
// 			if (m2[n2+idx].real() > 10.)
// 			{
// 				iz = idx/n2 ;
// 				iy = (idx%n2)/n1 ;
// 				ix = (idx%n2)%n1 ;
// 				#pragma omp critical
// 				{
// 					fprintf(DDDWrite,   "%d %d %d %f", ix, iy, iz, m2[n2+idx].real()) ;
// 				}
// 			}
// 	}
//
// }
//
//
// void	writeDensityMap3D	(Scalar *axion, const int index)
// {
// 	if (commRank() != 0)
// 		return;
//
// 	//--------------------------------------------------
// 	// OVERDENSITY MAP 3D
// 	//--------------------------------------------------
//
// 	char sto3D[256];
//
// 	const size_t n1 = axion->Length();
// 	const size_t size = axion->Size();
//
// 	sprintf(sto3D,  "out/con/con-%05d.txt",   index);
//
// 	FILE *DDDWrite = NULL;
//
// 	if ((DDDWrite  = fopen(sto3D, "w+")) == NULL)
// 	{
// 		printf ("Couldn't open file %s for writing\n", sto3D);
// 		return ;
// 	}
//
// 	if ( axion->Field() == FIELD_AXION)
// 	{
// 		switch (axion->Precision())
// 		{
// 			case FIELD_DOUBLE:
// 				writeDatafromTheta3D(static_cast<complex<double> *>(axion->m2Cpu()), (*axion->zV()),         n1, size, DDDWrite);
// 				break;
//
// 			case FIELD_SINGLE:
// 				writeDatafromTheta3D(static_cast<complex<float> *> (axion->m2Cpu()), (float) (*axion->zV()), n1, size, DDDWrite);
// 				break;
// 		}
// 	}
// 	fclose(3DWrite);
// }
