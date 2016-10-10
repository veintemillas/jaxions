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

//JAVIER changed .dat to .txt    
	sprintf(stoRoh, "out/rho/rho-%05d.txt", index);
	sprintf( stoAt, "out/at/at-%05d.txt", index);

	FILE *atWrite = NULL;
	FILE *rhoWrite = NULL;

	if ((atWrite  = fopen(stoAt, "w+")) == NULL)
	{
		printf ("Couldn't open file %s for writing\n", stoAt);
//JAVIER changed return 1; -> return ;        
		return ;
	}

	if ((rhoWrite = fopen(stoRoh, "w+")) == NULL)
	{
		printf ("Couldn't open file %s for writing\n", stoRoh);
//JAVIER changed return 1; -> return ;        
        return ;
	}


	for (int ly = 0; ly < n1; ly++)
	{
		for (int lz = 0; lz < n1; lz++)
		{
			fprintf(atWrite,  "%f ", arg( ((complex<float> *) axion->mCpu())[lz + n1*ly + n1*n1]) );
			fprintf(rhoWrite, "%f ", abs( ((complex<float> *) axion->mCpu())[lz + n1*ly + n1*n1])/(*(axion->zV())));
		}

		fprintf(atWrite , "\n");
		fprintf(rhoWrite, "\n");
	}

	fclose(atWrite);
	fclose(rhoWrite);

//JAVIER removed a \n before Map
	printf("Map printed...\n");
//JAVIER    
    printf("z = %lf -", *(axion->zV()));   
    printf("-Examples m: m[0]= %f + %f*I = |%f|exp(I*%f)\n",  ((complex<float> *) axion->mCpu())[0].real(), ((complex<float> *) axion->mCpu())[0].imag(), arg( ((complex<float> *) axion->mCpu())[0]), abs( ((complex<float> *) axion->mCpu())[0])/(*(axion->zV() )) );

}
