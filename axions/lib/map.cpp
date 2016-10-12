#include <cstdio>
#include <complex>

#include "scalarField.h"

using namespace std;

template<typename Float>
void	writeData	(complex<Float> *m, const Float z, const int n1, FILE *atWrite, FILE *rhoWrite)
{
	for (int ly = 0; ly < n1; ly++)
	{
		for (int lz = 0; lz < n1; lz++)
		{
			fprintf(atWrite,  "%f ", arg(m[lz + n1*ly + n1*n1]) );
			fprintf(rhoWrite, "%f ", abs(m[lz + n1*ly + n1*n1])/z);
		}

		fprintf(atWrite , "\n");
		fprintf(rhoWrite, "\n");
	}
}

void	writeMap	(Scalar *axion, const int index)
{
	//--------------------------------------------------
	// DIRECT SHORT OUTPUT PICTURES 2D
	//--------------------------------------------------

	char stoRoh[256];
	char stoAt[256];

	const int n1 = axion->Length();

	sprintf(stoRoh, "out/rho/rho-%05d.txt", index);
	sprintf( stoAt, "out/at/at-%05d.txt", index);

	FILE *atWrite = NULL;
	FILE *rhoWrite = NULL;

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

	switch (axion->Precision())
	{
		case FIELD_DOUBLE:
			writeData((complex<double> *) axion->mCpu(),         (*axion->zV()), n1, atWrite, rhoWrite);
			break;

		case FIELD_SINGLE:
			writeData((complex<float> *)  axion->mCpu(), (float) (*axion->zV()), n1, atWrite, rhoWrite);
			break;
	}

	fclose(atWrite);
	fclose(rhoWrite);

	printf("Map printed...\n");

/*	ESTO HABRA QUE BORRARLO EVENTUALMENTE	*/
    printf("z = %lf -", *(axion->zV()));   
    printf("-Examples m: m[0]= %f + %f*I = |%f|exp(I*%f)\n",  ((complex<float> *) axion->mCpu())[0].real(), ((complex<float> *) axion->mCpu())[0].imag(), arg( ((complex<float> *) axion->mCpu())[0]), abs( ((complex<float> *) axion->mCpu())[0])/(*(axion->zV() )) );

}
