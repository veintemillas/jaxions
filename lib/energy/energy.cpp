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
	const bool   map;

	FieldType	fType;
	VqcdType	pot;

	void    *eRes;
	Scalar	*field;

	public:

		 Energy(Scalar *field, const double LL, const double nQcd, const double delta, void *eRes, VqcdType pot, const double sh, const bool map);
		~Energy() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	Energy::Energy(Scalar *field, const double LL, const double nQcd, const double delta, void *eRes, VqcdType pot, const double sh, const bool map) : field(field), Lx(field->Length()), Lz(field->eDepth()),
	V(field->Size()), S(field->Surf()), Vt(field->TotalSize()), delta2(delta*delta), nQcd(nQcd), eRes(eRes), pot(pot), fType(field->Field()),
	shift(sh), LL(field->Lambda() == LAMBDA_Z2 ? LL/((*field->zV())*(*field->zV())) : LL), map(map)
{
}

void	Energy::runGpu	()
{
#ifdef	USE_GPU

	const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
	double *z = field->zV();
	int st;

	field->exchangeGhosts(FIELD_M);

	if (fType == FIELD_SAXION)
		energyGpu(field->mGpu(), field->vGpu(), field->m2Gpu(), z, delta2, LL, nQcd, shift, pot, uLx, uLz, uV, uS, field->Precision(), static_cast<double*>(eRes), ((cudaStream_t *)field->Streams())[0], map);
	else
		energyThetaGpu(field->mGpu(), field->vGpu(), field->m2Gpu(), z, delta2, nQcd, uLx, uLz, uV, uS, field->Precision(), static_cast<double*>(eRes), ((cudaStream_t *)field->Streams())[0], map);

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
		energyCpu	(field, delta2, LL, nQcd, Lx, V, S, eRes, shift, pot, map);
	} else {
		energyThetaCpu	(field, delta2, nQcd, Lx, V, S, eRes, map);
	}
}

void	Energy::runXeon	()
{
#ifdef	USE_XEON
	if (fType == FIELD_SAXION) {
		energyXeon	(field, delta2, LL, nQcd, Lx, V, S, precision, eRes, shift, pot, map);
	} else {
		energyThetaXeon	(field, delta2, nQcd, Lx, V, S, eRes, map);
	}
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	energy	(Scalar *field, FlopCounter *fCount, void *eRes, const bool map, const double delta, const double nQcd, const double LL, VqcdType pot, const double shift)
{
	if (map && (field->Field() == FIELD_SAXION) && field->LowMem())
	{
		printf	("Can't compute energy map for saxion wit lowmem kernels\n");
		return;
	}

	void *eTmp;
	trackAlloc(&eTmp, 128);

	Energy *eDark = new Energy(field, LL, nQcd, delta, eTmp, pot, shift, map);

	if	(!field->Folded())
	{
		Folder	munge(field);
		munge(FOLD_ALL);
	}

	switch (field->Device())
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
