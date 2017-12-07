#include <memory>
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

#include "utils/utils.h"

#include <mpi.h>

class	Energy : public Tunable
{
	private:

	const double delta2;
	const double nQcd, LL, shift;
	const size_t Vt;
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

	Energy::Energy(Scalar *field, const double LL, const double nQcd, const double delta, void *eRes, VqcdType pot, const double sh, const bool map) : field(field),
	Vt(field->TotalSize()), delta2(delta*delta), nQcd(nQcd), eRes(eRes), pot(pot), fType(field->Field()), shift(sh),
	LL(field->Lambda() == LAMBDA_Z2 ? LL/((*field->zV())*(*field->zV())) : LL), map(map)
{
}

void	Energy::runGpu	()
{
#ifdef	USE_GPU

	const uint uLx = field->Length();
	const uint uLz = field->Depth();
	const uint uS  = field->Surf();
	const uint uV  = field->Size();
	double *z = field->zV();

	field->exchangeGhosts(FIELD_M);

	switch (fType) {
		case	FIELD_SAXION:
			setName		("Energy Saxion");
			energyGpu(field->mGpu(), field->vGpu(), field->m2Gpu(), z, delta2, LL, nQcd, shift, pot, uLx, uLz, uV, uS, field->Precision(), static_cast<double*>(eRes), ((cudaStream_t *)field->Streams())[0], map);
			break;

		case	FIELD_AXION:
			setName		("Energy Axion");
			energyThetaGpu(field->mGpu(), field->vGpu(), field->m2Gpu(), z, delta2, nQcd, uLx, uLz, uV, uS, field->Precision(), static_cast<double*>(eRes), ((cudaStream_t *)field->Streams())[0], map, false);
			break;

		case	FIELD_AXION_MOD:
			setName		("Energy Axion (mod)");
			energyThetaGpu(field->mGpu(), field->vGpu(), field->m2Gpu(), z, delta2, nQcd, uLx, uLz, uV, uS, field->Precision(), static_cast<double*>(eRes), ((cudaStream_t *)field->Streams())[0], map, true);
			break;
	}

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
#else
	LogError ("Gpu support not built");
	exit(1);
#endif
}

void	Energy::runCpu	()
{
	switch (fType) {
		case	FIELD_SAXION:
			setName		("Energy Saxion");
			energyCpu	(field, delta2, LL, nQcd, eRes, shift, pot, map);
			break;

		case	FIELD_AXION:
			setName		("Energy Axion");
			energyThetaCpu	(field, delta2, nQcd, eRes, map, false);
			break;

		case	FIELD_AXION_MOD:
			setName		("Energy Axion (mod)");
			energyThetaCpu	(field, delta2, nQcd, eRes, map, true);
			break;
	}
}

using namespace profiler;

void	energy	(Scalar *field, void *eRes, const bool map, const double delta, const double nQcd, const double LL, VqcdType pot, const double shift)
{
	if (map && (field->Field() == FIELD_SAXION) && field->LowMem())
	{
		LogError ("Can't compute energy map for saxion wit lowmem kernels\n");
		return;
	}

	LogMsg  (VERB_HIGH, "Called energy");
	Profiler &prof = getProfiler(PROF_ENERGY);

	void *eTmp;
	trackAlloc(&eTmp, 128);

	auto	eDark = std::make_unique<Energy>(field, LL, nQcd, delta, eTmp, pot, shift, map);

	if	(!field->Folded())
	{
		Folder	munge(field);
		munge(FOLD_ALL);
	}

	prof.start();

	switch (field->Device())
	{
		case DEV_CPU:
			eDark->runCpu ();
			break;

		case DEV_GPU:
			eDark->runGpu ();
			break;

		default:
			LogError ("Error: invalid device");
			prof.stop();
			trackFree(eTmp);
			return;
	}

	const int size = field->Field() == FIELD_SAXION ? 10 : 5;

	MPI_Allreduce(eTmp, eRes, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	trackFree(eTmp);

	const double Vt = 1./(field->TotalSize());

	#pragma unroll
	for (int i=0; i<size; i++)
		static_cast<double*>(eRes)[i] *= Vt;

	prof.stop();

	field->setReduced(false);

	double flops = (field->Field() == FIELD_SAXION ? (pot == VQCD_1 ? 111 : 112) : 25)*field->Size()*1e-9;
	double bytes = 8.*field->DataSize()*field->Size()*1e-9;

	if (map) {
		eDark->appendName(" Map");
		bytes *= 9./8;
	}

	eDark->add(flops, bytes);		// Flops are not exact
	prof.add(eDark->Name(), flops, bytes);

	LogMsg	(VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", eDark->Name().c_str(), prof.Prof()[eDark->Name()].GFlops(), prof.Prof()[eDark->Name()].GBytes());

	return;
}
