#include <cstdio>
#include <complex>

#include "scalarField.h"
#include "parse.h"

using namespace std;

template<typename Float>
void	writeData	(complex<Float> *m, const Float z, const size_t n1, FILE *atWrite, FILE *rhoWrite)
{
	fprintf(atWrite,  "# %d %f %f %f \n", sizeN , sizeL , sizeL/sizeN , z );
	fprintf(rhoWrite, "# %d %f %f %f \n", sizeN , sizeL , sizeL/sizeN , z );
	for (size_t ly = 0; ly < n1; ly++)
	{
		for (size_t lx = 0; lx < n1; lx++)
		{
			fprintf(atWrite,  "%f ", arg(m[lx + n1*ly]) );
			fprintf(rhoWrite, "%f ", abs(m[lx + n1*ly])/z);
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
			writeData(static_cast<complex<double> *> (axion->mCpu()),         (*axion->zV()), n1, atWrite, rhoWrite);
			break;

		case FIELD_SINGLE:
			writeData(static_cast<complex<float> *>  (axion->mCpu()), (float) (*axion->zV()), n1, atWrite, rhoWrite);
			break;
	}

	fclose(atWrite);
	fclose(rhoWrite);

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
