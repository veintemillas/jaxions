#include <cstdio>
#include <cstdlib>
#include "scalar/scalarField.h"
#include "enum-field.h"

#include "energy/energyMapXeon.h"
#include "energy/energyXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "energy/energyMapGpu.h"
	#include "energy/energyGpu.h"
#endif

#include "utils/flopCounter.h"
#include "utils/memAlloc.h"

#include <mpi.h>

class	EnergyMap
{
	private:

	const double delta2;
	const double nQcd, LL, shift;
	const size_t Lx, Lz, V, S;

	FieldPrecision precision;
	FieldType fType;
	VqcdType pot;

	void	*eRes;
	Scalar	*axionField;

	public:

		 EnergyMap(Scalar *field, const double LL, const double nQcd, const double delta, void *eRes, VqcdType pot, const double sh);
		~EnergyMap() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	EnergyMap::EnergyMap(Scalar *field, const double LL, const double nQcd, const double delta, void *eRes, VqcdType pot, const double sh) : axionField(field), Lx(field->Length()), Lz(field->eDepth()),
				V(field->Size()), S(field->Surf()), delta2(delta*delta), precision(field->Precision()), nQcd(nQcd), pot(pot), fType(field->Field()), shift(sh),
				LL(field->Lambda() == LAMBDA_Z2 ? LL/((*field->zV())*(*field->zV())) : LL), eRes(eRes)
{
}

void	EnergyMap::runGpu	()
{
#ifdef	USE_GPU

	const uint uLx = Lx, uLz = Lz, uS = S, uV = V;
	double *z = axionField->zV();

	axionField->exchangeGhosts(FIELD_M);

	if (fType == FIELD_SAXION) {
		energyMapGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, delta2, nQcd, LL, shift, pot, uLx, uLz, uV, uS, precision, ((cudaStream_t *)axionField->Streams())[0]);
	} else {
		energyThetaGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, delta2, nQcd, uLx, uLz, uV, uS, precision, static_cast<double*>(eRes), ((cudaStream_t *)axionField->Streams())[0], true);
	}

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot

#else
	printf("Gpu support not built");
	exit(1);
#endif
}

void	EnergyMap::runCpu	()
{
	if (fType == FIELD_SAXION)
		energyMapCpu(axionField, delta2, LL, nQcd, Lx, V, S, precision, shift, pot);
	else
		energyThetaCpu(axionField, delta2, nQcd, Lx, V, S, static_cast<double*>(eRes), true);
}

void	EnergyMap::runXeon	()
{
#ifdef	USE_XEON
	if (fType == FIELD_SAXION)
		energyMapXeon(axionField, delta2, lambda, nQcd, Lx, V, S, precision, shift, pot);
	else
		energyMapThetaXeon(axionField, delta2, nQcd, Lx, V, S, static_cast<double*>(eRes), true);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	energyMap	(Scalar *field, const double LL, const double nQcd, const double delta, void *eRes, FlopCounter *fCount, const VqcdType pot, const double sh)
{
	void *eTmp;
	trackAlloc(&eTmp, 128);

	EnergyMap *eDark = new EnergyMap(field, LL, nQcd, delta, eRes, pot, sh);

	switch (field->Device())
	{
		case DEV_CPU:
			eDark->runCpu ();
			break;

		case DEV_GPU:
			eDark->runGpu ();
			break;

		case DEV_XEON:
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
	double bytes = 9.*field->DataSize()*field->Size()*1e-9;

	fCount->addFlops(flops, bytes);

	return;
}
