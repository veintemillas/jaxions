#include <cstdio>
#include <cstdlib>
#include "scalar/scalarField.h"
#include "enum-field.h"

#include "energy/energyMapXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "energy/energyMapGpu.h"
#endif

#include "utils/flopCounter.h"
#include "utils/memAlloc.h"

#include <mpi.h>

class	EnergyMap
{
	private:

	const double delta2, nQcd;
	const size_t Lx, Lz, V, S;

	FieldPrecision precision;

	Scalar	*axionField;

	public:

		 EnergyMap(Scalar *field, const double nQcd, const double delta);
		~EnergyMap() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	EnergyMap::EnergyMap(Scalar *field, const double nQcd, const double delta) : axionField(field), Lx(field->Length()), Lz(field->eDepth()), V(field->Size()),
				S(field->Surf()), delta2(delta*delta), precision(field->Precision()), nQcd(nQcd)
{
}

void	EnergyMap::runGpu	()
{
#ifdef	USE_GPU

	const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
	double *z = axionField->zV();

	axionField->exchangeGhosts(FIELD_M);
//	int st = energyMapGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, delta2, nQcd, uLx, uLz, uV, uS, precision, ((cudaStream_t *)axionField->Streams())[0]);

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot

//	if (st != 0)
	{
		printf("Gpu error computing energy.");
		exit(1);
	}

#else
	printf("Gpu support not built");
	exit(1);
#endif
}

void	EnergyMap::runCpu	()
{
	energyMapCpu(axionField, delta2, nQcd, Lx, V, S);
}

void	EnergyMap::runXeon	()
{
#ifdef	USE_XEON
	energyMapXeon(axionField, delta2, nQcd, Lx, V, S);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	energyMap	(Scalar *field, const double nQcd, const double delta, DeviceType dev, FlopCounter *fCount)
{
	EnergyMap *eDark = new EnergyMap(field, nQcd, delta);

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

	fCount->addFlops((75.*field->Size() - 10.)*1.e-9, 8.*field->DataSize()*field->Size()*1.e-9);

	return;
}
