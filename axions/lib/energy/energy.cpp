#include <cstdio>
#include <cstdlib>
#include "scalar/scalarField.h"
#include "enum-field.h"

#include "energy/energyXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "energy/energyGpu.h"
#endif

#include "utils/flopCounter.h"
#include "utils/memAlloc.h"

#include <mpi.h>

class	Energy
{
	private:

	const double delta2;
	const double nQcd, LL;
	const size_t Lx, Lz, V, S, Vt;

	FieldPrecision precision;

	void    *eRes;
	Scalar	*axionField;

	public:

		 Energy(Scalar *field, const double LL, const double nQcd, const double delta, void *eRes);
		~Energy() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	Energy::Energy(Scalar *field, const double LL, const double nQcd, const double delta, void *eRes) : axionField(field), Lx(field->Length()), Lz(field->eDepth()), V(field->Size()),
				S(field->Surf()), Vt(field->TotalSize()), delta2(delta*delta), precision(field->Precision()), LL(LL), nQcd(nQcd), eRes(eRes)
{
}

void	Energy::runGpu	()
{
#ifdef	USE_GPU

	const uint uLx = Lx, uLz = Lz, uS = S, uV = V, uVt = Vt;
	double *z = axionField->zV();

	axionField->exchangeGhosts(FIELD_M);
	int st = energyGpu(axionField->mGpu(), axionField->vGpu(), z, delta2, LL, nQcd, uLx, uLz, uV, uVt, uS, precision, static_cast<double*>(eRes), ((cudaStream_t *)axionField->Streams())[0]);

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot

	if (st != 0)
	{
		printf("Gpu error computing energy.");
		exit(1);
	}

#else
	printf("Gpu support not built");
	exit(1);
#endif
}

void	Energy::runCpu	()
{
	energyCpu(axionField, delta2, LL, nQcd, Lx, V, S, Vt, precision, eRes);
}

void	Energy::runXeon	()
{
#ifdef	USE_XEON
	energyXeon(axionField, delta2, LL, nQcd, Lx, V, S, Vt, precision, eRes);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	energy	(Scalar *field, const double LL, const double nQcd, const double delta, DeviceType dev, void *eRes, FlopCounter *fCount)
{
	void *eTmp;
	trackAlloc(&eTmp, 128);

	Energy *eDark = new Energy(field, LL, nQcd, delta, eTmp);

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

	MPI_Allreduce(eTmp, eRes, 10, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	trackFree(&eTmp, ALLOC_TRACK);

	fCount->addFlops((75.*field->Size() - 10.)*1.e-9, 8.*field->DataSize()*field->Size()*1.e-9);

	return;
}
