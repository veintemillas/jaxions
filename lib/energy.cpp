#include <cstdio>
#include <cstdlib>
#include "scalarField.h"
#include "enum-field.h"

#include "energyXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "energyGpu.h"
#endif

#include "flopCounter.h"

class	Energy
{
	private:

	const double delta2, dz;
	const double nQcd, LL;
	const size_t Lx, Lz, V, S;

	FieldPrecision precision;

	Scalar	*axionField;

	public:

		 Energy(Scalar *field, const double LL, const double nQcd, const double delta, const double dz);
		~Energy() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	Energy::Energy(Scalar *field, const double LL, const double nQcd, const double delta, const double dz) : axionField(field), dz(dz), Lx(field->Length()), Lz(field->eDepth()), V(field->Size()),
				S(field->Surf()), delta2(delta*delta), precision(field->Precision()), LL(LL), nQcd(nQcd)
{
}

void	Energy::runGpu	()
{
#ifdef	USE_GPU
	const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
	const uint ext = uV + uS;
	double *z = axionField->zV();

        energyGpu(axionField->mGpu(), axionField->vGpu(), z, dz, delta2, LL, nQcd, uLx, uLz, 2*uS, uV, precision, ((cudaStream_t *)axionField->Streams())[0]);
	axionField->exchangeGhosts(FIELD_M);
        energyGpu(axionField->mGpu(), axionField->vGpu(), z, dz, delta2, LL, nQcd, uLx, uLz, uS, 2*uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
        energyGpu(axionField->mGpu(), axionField->vGpu(), z, dz, delta2, LL, nQcd, uLx, uLz, uV, ext, precision, ((cudaStream_t *)axionField->Streams())[0]);

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
#else
	printf("Gpu support not built");
	exit(1);
#endif
}

void	Energy::runCpu	()
{
	energyCpu(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision);
}

void	Propagator::runXeon	()
{
#ifdef	USE_XEON
	energyXeon(axionField, dz, delta2, LL, nQcd, Lx, V, S, precision);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	energy	(Scalar *field, const double dz, const double LL, const double nQcd, const double delta, DeviceType dev, FlopCounter *fCount)
{
	Energy *eDark = new Energy(field, LL, nQcd, delta, dz);

	switch (dev)
	{
		case DEV_CPU:
			eDark->runCpu ();
			break;

		case DEV_GPU:
			eDark->runGpu ();
			break;

		case	DEV_XEON:
			eDark->runXeon();
			break;

		default:
			printf ("Not a valid device\n");
			break;
	}

	delete	eDark;

	fCount->addFlops(32.*4.*field->Size()*1.e-9, 10.*4.*field->dataSize()*field->Size()*1.e-9);

	return;
}
