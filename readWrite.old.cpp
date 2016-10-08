#include <cstdio>
#include <cstdlib>

#include "scalarField.h"
#include "parse.h"

void	writeConf (Scalar *axion, int index)
{
	FILE *outM, *outV;
	char base[256];

	sprintf(base, "out/dump/mField.%05d", index);

	if ((outM = fopen(base, "w+")) == NULL)
	{
		printf("Error: Can't write to file %s\n", base);
		exit(1);
	}

	sprintf(base, "out/dump/vField.%05d", index);

	if ((outV = fopen(base, "w+")) == NULL)
	{
		printf("Error: Can't write to file %s\n", base);
		exit(1);
	}

	switch (axion->Precision())
	{
		case FIELD_SINGLE:
		case FIELD_MIXED:
		{
			int prec = 4;
			int dummy = 0;
			float LLf = LL;
			float nQcdf = nQcd;
			float sizeLf = sizeL;

			float zR = (float) (*(axion->zV()));

			// Header data

			fwrite(&prec,   sizeof(int),   1, outM);	// Precision of the data
			fwrite(&sizeN,  sizeof(int),   1, outM);	// Length X
			fwrite(&sizeN,  sizeof(int),   1, outM);	// Length Y
			fwrite(&sizeN,  sizeof(int),   1, outM);	// Length Z
			fwrite(&dummy,  sizeof(int),   1, outM);	// X Position of the block, for MPI
			fwrite(&dummy,  sizeof(int),   1, outM);	// Y Position of the block, for MPI
			fwrite(&dummy,  sizeof(int),   1, outM);	// Z Position of the block, for MPI
			fwrite(&LLf,    sizeof(float), 1, outM);	// Value of LL
			fwrite(&nQcdf,  sizeof(float), 1, outM);	// nQcd
			fwrite(&sizeLf, sizeof(float), 1, outM);	// Physical size
			fwrite(&zR,     sizeof(float), 1, outM);	// Value of z

			// Raw data

			fwrite(axion->mCpu(), sizeof(float)*2, axion->Size(), outM);

			// Header data

			fwrite(&prec,   sizeof(int),   1, outV);	// Precision of the data
			fwrite(&sizeN,  sizeof(int),   1, outV);	// Length X
			fwrite(&sizeN,  sizeof(int),   1, outV);	// Length Y
			fwrite(&sizeN,  sizeof(int),   1, outV);	// Length Z
			fwrite(&dummy,  sizeof(int),   1, outV);	// X Position of the block, for MPI
			fwrite(&dummy,  sizeof(int),   1, outV);	// Y Position of the block, for MPI
			fwrite(&dummy,  sizeof(int),   1, outV);	// Z Position of the block, for MPI
			fwrite(&LLf,    sizeof(float), 1, outV);	// Value of LL
			fwrite(&nQcdf,  sizeof(float), 1, outV);	// nQcd
			fwrite(&sizeLf, sizeof(float), 1, outV);	// Physical size
			fwrite(&zR,     sizeof(float), 1, outV);	// Value of z

			// Raw data

			fwrite(axion->vCpu(), sizeof(float)*2, axion->Size(), outV);
		}
	
		break;

		case FIELD_DOUBLE:
		{
			int prec = 8;
			int dummy = 0;

			double zR = (double) (*(axion->zV()));

			// Header data

			fwrite(&prec,  sizeof(int),    1, outM);	// Precision of the data
			fwrite(&sizeN, sizeof(int),    1, outM);	// Length X
			fwrite(&sizeN, sizeof(int),    1, outM);	// Length Y
			fwrite(&sizeN, sizeof(int),    1, outM);	// Length Z
			fwrite(&dummy, sizeof(int),    1, outM);	// X Position of the block, for MPI
			fwrite(&dummy, sizeof(int),    1, outM);	// Y Position of the block, for MPI
			fwrite(&dummy, sizeof(int),    1, outM);	// Z Position of the block, for MPI
			fwrite(&LL,    sizeof(double), 1, outM);	// Value of LL
			fwrite(&nQcd,  sizeof(double), 1, outM);	// nQcd
			fwrite(&sizeL, sizeof(double), 1, outM);	// Physical size
			fwrite(&zR,    sizeof(double), 1, outM);	// Value of z

			// Raw data

			fwrite(axion->mCpu(), sizeof(double)*2, axion->Size(), outM);

			// Header data

			fwrite(&prec,  sizeof(int),    1, outV);	// Precision of the data
			fwrite(&sizeN, sizeof(int),    1, outV);	// Length X
			fwrite(&sizeN, sizeof(int),    1, outV);	// Length Y
			fwrite(&sizeN, sizeof(int),    1, outV);	// Length Z
			fwrite(&dummy, sizeof(int),    1, outV);	// X Position of the block, for MPI
			fwrite(&dummy, sizeof(int),    1, outV);	// Y Position of the block, for MPI
			fwrite(&dummy, sizeof(int),    1, outV);	// Z Position of the block, for MPI
			fwrite(&LL,    sizeof(double), 1, outV);	// Value of LL
			fwrite(&nQcd,  sizeof(double), 1, outV);	// nQcd
			fwrite(&sizeL, sizeof(double), 1, outV);	// Physical size
			fwrite(&zR,    sizeof(double), 1, outV);	// Value of z

			// Raw data

			fwrite(axion->vCpu(), sizeof(double)*2, axion->Size(), outV);
		}

		break;

		default:

		printf("Error: Invalid precision. How did you get this far?\n");
		exit(1);

		break;
	}

	fclose(outM);
	fclose(outV);
}


