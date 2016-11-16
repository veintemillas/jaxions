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

	void    *strData;
	Scalar	*axionField;

	public:

		 Strings(Scalar *field, void *str);
		~Strings() {};

	double	runCpu	();
	double	runGpu	();
	double	runXeon	();
};

	Strings::Strings(Scalar *field, void *str) : axionField(field), Lx(field->Length()), V(field->Size()), S(field->Surf()), precision(field->Precision()), strData(str)
{
}

double	Strings::runGpu	()
{
#ifdef	USE_GPU
	const uint uLx = Lx, uS = S, uV = V;

	axionField->exchangeGhosts(FIELD_M);
	stringGpu(axionField->mGpu(), uLx, uV, uS, precision, strData, ((cudaStream_t *)axionField->Streams())[0]);

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot

#else
	printf("Gpu support not built");
	exit(1);
#endif
}

double	Strings::runCpu	()
{
	return	stringCpu(axionField, Lx, V, S, precision, strData);
}

double	Strings::runXeon	()
{
#ifdef	USE_XEON
	return	stringXeon(axionField, Lx, V, S, precision, strData);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

double	strings	(Scalar *field, DeviceType dev, void *strData, FlopCounter *fCount)
{
	Strings *eStr = new Strings(field, strData);

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
