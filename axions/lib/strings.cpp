#include <cstdio>
#include <cstdlib>
#include "scalarField.h"
#include "enum-field.h"

#include "stringXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "stringGpu.h"
#endif

#include "flopCounter.h"
#include "memAlloc.h"

#include <mpi.h>

class	Strings
{
	private:

	const size_t Lx, V, S;

	FieldPrecision precision;

	void    *string;
	Scalar	*axionField;

	public:

		 Strings(Scalar *field, void *str);
		~Strings() {};

	double	runCpu	();
	double	runGpu	();
	double	runXeon	();
};

	Strings::Strings(Scalar *field, void *str) : axionField(field), Lx(field->Length()), V(field->Size()), S(field->Surf()), precision(field->Precision()), string(str)
{
}

double	Strings::runGpu	()
{
#ifdef	USE_GPU
/*
	const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
	const uint ext = uV + uS;
	double *z = axionField->zV();

        energyGpu(axionField->mGpu(), axionField->vGpu(), z, delta2, LL, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[0]);
	axionField->exchangeGhosts(FIELD_M);
        energyGpu(axionField->mGpu(), axionField->vGpu(), z, delta2, LL, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
        energyGpu(axionField->mGpu(), axionField->vGpu(), z, delta2, LL, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[0]);

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
*/
#else
	printf("Gpu support not built");
	exit(1);
#endif
}

double	Strings::runCpu	()
{
	return	stringCpu(axionField, Lx, V, S, precision, string);
}

double	Strings::runXeon	()
{
#ifdef	USE_XEON
	return	stringXeon(axionField, Lx, V, S, precision, string);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

double	strings	(Scalar *field, DeviceType dev, void *string, FlopCounter *fCount)
{
	Strings *eStr = new Strings(field, string);

	double	strDen = 0., strTmp = 0.;

	switch (dev)
	{
		case DEV_CPU:
			strTmp = eStr->runCpu ();
			break;

		case DEV_GPU:
			strTmp = eStr->runGpu ();
			break;

		case	DEV_XEON:
			strTmp = eStr->runXeon();
			break;

		default:
			printf ("Not a valid device\n");
			break;
	}

	delete	eStr;

	MPI_Allreduce(&strTmp, &strDen, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//	fCount->addFlops((75.*field->Size() - 10.)*1.e-9, 8.*field->dataSize()*field->Size()*1.e-9);

	return	(strDen*field->Size())/((double) field->TotalSize());
}
