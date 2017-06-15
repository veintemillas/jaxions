#include <cstdio>
#include <cstdlib>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
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
	const double nQcd, LL, shift;
	const size_t Lx, Lz, V, S, Vt;

	FieldPrecision precision;
	FieldType fType;
	VqcdType pot;

	void    *eRes;
	Scalar	*axionField;

	public:

		 Energy(Scalar *field, const double LL, const double nQcd, const double delta, void *eRes, VqcdType pot, const double sh);
		~Energy() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	Energy::Energy(Scalar *field, const double LL, const double nQcd, const double delta, void *eRes, VqcdType pot, const double sh) : axionField(field), Lx(field->Length()), Lz(field->eDepth()),
	V(field->Size()), S(field->Surf()), Vt(field->TotalSize()), delta2(delta*delta), precision(field->Precision()), nQcd(nQcd), eRes(eRes), pot(pot), fType(field->Field()), shift(sh),
	LL(field->Lambda() == LAMBDA_Z2 ? LL/((*field->zV())*(*field->zV())) : LL)
{
}

void	Energy::runGpu	()
{
#ifdef	USE_GPU

	const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
	double *z = axionField->zV();
	int st;

	axionField->exchangeGhosts(FIELD_M);

	if (fType == FIELD_SAXION)
		energyGpu(axionField->mGpu(), axionField->vGpu(), z, delta2, LL, nQcd, shift, pot, uLx, uLz, uV, uS, precision, static_cast<double*>(eRes), ((cudaStream_t *)axionField->Streams())[0]);
	else
		energyThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, delta2, nQcd, uLx, uLz, uV, uS, precision, static_cast<double*>(eRes), ((cudaStream_t *)axionField->Streams())[0]);

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
	if (fType == FIELD_SAXION) {
		energyCpu	(axionField, delta2, LL, nQcd, Lx, V, S, precision, eRes, shift, pot);
	} else {
		energyThetaCpu	(axionField, delta2, nQcd, Lx, V, S, eRes);
	}
}

void	Energy::runXeon	()
{
#ifdef	USE_XEON
	if (fType == FIELD_SAXION) {
		energyXeon	(axionField, delta2, LL, nQcd, Lx, V, S, precision, eRes, shift, pot);
	} else {
		energyThetaXeon	(axionField, delta2, nQcd, Lx, V, S, eRes);
	}
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	energy	(Scalar *field, const double LL, const double nQcd, const double delta, DeviceType dev, void *eRes, FlopCounter *fCount, VqcdType pot, const double shift)
{
	void *eTmp;
	trackAlloc(&eTmp, 128);

	Energy *eDark = new Energy(field, LL, nQcd, delta, eTmp, pot, shift);

	if	(!field->Folded())
	{
		Folder	munge(field);
		munge(FOLD_ALL);
	}

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

	const int size = field->Field() == FIELD_SAXION ? 10 : 5;

	MPI_Allreduce(eTmp, eRes, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	trackFree(&eTmp, ALLOC_TRACK);

	const double Vt = 1./(field->TotalSize());

	#pragma unroll
	for (int i=0; i<size; i++)
		static_cast<double*>(eRes)[i] *= Vt;

	double flops = (field->Field() == FIELD_SAXION ? (pot == VQCD_1 ? 111 : 112) : 25)*field->Size()*1e-9;
	double bytes = 8.*field->DataSize()*field->Size()*1e-9;

	fCount->addFlops(flops, bytes);

	return;
}
