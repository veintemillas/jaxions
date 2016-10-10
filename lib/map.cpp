#include <cstdio>
#include <complex>

#include "scalarField.h"

using namespace std;

void	writeMap	(Scalar *axion, const int index)
{
	//--------------------------------------------------
	// DIRECT SHORT OUTPUT PICTURES 2D
	//--------------------------------------------------

	char stoRoh[256];
	char stoAt[256];

	const int n1 = axion->Length();

	sprintf(stoRoh, "out/rho/rho-%05d.dat", index);
	sprintf( stoAt, "out/at/at-%05d.dat", index);

	FILE *atWrite = NULL;
	FILE *rhoWrite = NULL;

	if ((atWrite  = fopen(stoAt, "w+")) == NULL)
	{
		printf ("Couldn't open file %s for writing\n", stoAt);
		return 1;
	}

	if ((rhoWrite = fopen(stoRoh, "w+")) == NULL)
	{
		printf ("Couldn't open file %s for writing\n", stoRoh);
		return 1;
	}


	for (int ly = 0; ly < n1; ly++)
	{
		for (int lz = 0; lz < n1; lz++)
		{
			fprintf(atWrite,  "%f ", arg( ((complex<double> *) axion->mCpu())[lz + n1*ly + n1*n1]) );
			fprintf(rhoWrite, "%f ", abs( ((complex<double> *) axion->mCpu())[lz + n1*ly + n1*n1])/(*(axion->zV())));
		}

		fprintf(atWrite , "\n");
		fprintf(rhoWrite, "\n");
	}

	fclose(atWrite);
	fclose(rhoWrite);

	printf("\nMap printed...\n");
}
