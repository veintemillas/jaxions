#include <cstdio>
#include <complex>

#include "scalar/scalarField.h"
#include "utils/parse.h"

using namespace std;

template<typename Float>
void	writeData	(complex<Float> *m, complex<Float> *v, const Float z, const size_t n1, FILE *atWrite, FILE *rhoWrite, FILE *densWrite)
{
	fprintf(atWrite,  "# %d %f %f %f \n", sizeN , sizeL , sizeL/sizeN , z );
	fprintf(rhoWrite, "# %d %f %f %f \n", sizeN , sizeL , sizeL/sizeN , z );
	fprintf(densWrite, "# %d %f %f %f \n", sizeN , sizeL , sizeL/sizeN , z );
	int n2 = n1*n1;
	int n3 = n2*n1;
	for (size_t ly = 0; ly < n1; ly++)
	{
		for (size_t lx = 0; lx < n1; lx++)
		{

			fprintf(atWrite,  "%f ", arg(m[lx + n1*ly]) );
			fprintf(rhoWrite, "%f ", abs(m[lx + n1*ly])/z);
			fprintf(densWrite, "%f ", pow(abs(m[lx + n1*ly])/z,2)*( pow(arg(m[lx + n1*ly]),2)+pow(imag(m[lx + n1*ly + n3 + n2]/m[lx + n1*ly]),2)/(9.0*pow(z,nQcd+2))) );
		}

		fprintf(atWrite , "\n");
		fprintf(rhoWrite, "\n");
		fprintf(densWrite, "\n");
	}
}

void	writeMap	(Scalar *axion, const int index)
{
	//--------------------------------------------------
	// DIRECT SHORT OUTPUT PICTURES 2D
	//--------------------------------------------------

	char stoRoh[256];
	char stoAt[256];
	char stoDens[256];

	const int n1 = axion->Length();

	sprintf(stoRoh, "out/rho/rho-%05d.txt", index);
	sprintf( stoAt, "out/at/at-%05d.txt", index);
	sprintf( stoDens, "out/dens/dens-%05d.txt", index);

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

	switch (axion->Precision())
	{
		case FIELD_DOUBLE:
			writeData(static_cast<complex<double> *> (axion->mCpu()), static_cast<complex<double> *> (axion->vCpu()) ,         (*axion->zV()), n1, atWrite, rhoWrite, densWrite);
			break;

		case FIELD_SINGLE:
			writeData(static_cast<complex<float> *>  (axion->mCpu()), static_cast<complex<float> *>  (axion->vCpu()),  (float) (*axion->zV()), n1, atWrite, rhoWrite, densWrite);
			break;
	}

	fclose(atWrite);
	fclose(rhoWrite);
	fclose(densWrite);

	//printf("Map printed...\n");

	/*	ESTO HABRA QUE BORRARLO EVENTUALMENTE	*/
	//printf("z = %lf - ", *(axion->zV()));
	//JAVIER included switch to test outputsin double and single
	//switch (axion->Precision())
	//{
	//	case FIELD_DOUBLE:
	//		printf("(MAP) m[0]= %lf + %lf*I = |%lf|exp(I*%lf)\n",  ((complex<double> *) axion->mCpu())[0].real(), ((complex<double> *) axion->mCpu())[0].imag(), abs( ((complex<double> *) axion->mCpu())[0]), arg( ((complex<double> *) axion->mCpu())[0])/(*(axion->zV() )) );
	//		break;

	//	case FIELD_SINGLE:
	//		printf("(MAP) m[0]= %f + %f*I = |%f|exp(I*%f)\n",  ((complex<float> *) axion->mCpu())[0].real(), ((complex<float> *) axion->mCpu())[0].imag(), abs( ((complex<float> *) axion->mCpu())[0]), arg( ((complex<float> *) axion->mCpu())[0])/(*(axion->zV() )) );
	//		break;
	//}
}
