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

	Scalar	*field;
	const size_t Vt;

	const double delta2, aMass2;
	void    *eRes;

	VqcdType	pot;
	FieldType	fType;

	const double shift, LL;
	const EnType mapmask;

	public:

		 Energy(Scalar *field, const double LL, const double delta, void *eRes, VqcdType pot, const double sh, const EnType emap);
		~Energy() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	Energy::Energy(Scalar *field, const double LL, const double delta, void *eRes, VqcdType pot, const double sh, const EnType emap) : field(field),
	Vt(field->TotalSize()), delta2(delta*delta), aMass2(field->AxionMassSq()), eRes(eRes), pot(pot), fType(field->Field()), shift(sh),
	LL(field->LambdaP()), mapmask(emap)
{
}

void	Energy::runGpu	()
{
#ifdef	USE_GPU

	const uint uLx = field->Length();
	const uint uLz = field->Depth();
	const uint uS  = field->Surf();
	const uint uV  = field->Size();
	//double *z = field->zV();
	double R       = *field->RV();
	double Rpp     = field->BckGnd()->Rp(*field->zV())*R;

	const bool map = (mapmask & EN_MAP);

	switch (fType) {
		case	FIELD_SAXION:
			setName		("Energy Saxion");
			energyGpu     (field->mGpu(), field->vGpu(), field->m2Gpu(), R, Rpp, delta2, LL, aMass2, shift, pot, uLx, uLz, uV, uS, field->Precision(), static_cast<double*>(eRes), ((cudaStream_t *)field->Streams())[0], map);
			break;

		case	FIELD_AXION:
			setName		("Energy Axion");
			energyThetaGpu(field->mGpu(), field->vGpu(), field->m2Gpu(), R, delta2, aMass2, uLx, uLz, uV, uS, field->Precision(), static_cast<double*>(eRes), ((cudaStream_t *)field->Streams())[0], map, false);
			break;

		case	FIELD_AXION_MOD:
			setName		("Energy Axion (mod)");
			energyThetaGpu(field->mGpu(), field->vGpu(), field->m2Gpu(), R, delta2, aMass2, uLx, uLz, uV, uS, field->Precision(), static_cast<double*>(eRes), ((cudaStream_t *)field->Streams())[0], map, true);
			break;

		// case	FIELD_NAXION:
		// 	setName		("Energy Naxion ");
		// 	energyNaxionGpu(field->mGpu(), field->vGpu(), field->m2Gpu(), R, delta2, aMass2, uLx, uLz, uV, uS, field->Precision(), static_cast<double*>(eRes), ((cudaStream_t *)field->Streams())[0], map, true);
		// 	break;

		default:
			LogError ("Energy not supported for this kind of field");
			return;
	}

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
	field->transferCpu(FIELD_M2);
#else
	LogError ("Gpu support not built");
	exit(1);
#endif
}

void	Energy::runCpu	()
{
	const bool map = (mapmask & EN_MAP); //Update if mask in theta is required FIX ME

	switch (fType) {
		case	FIELD_SAXION:
			setName		("Energy Saxion");
			energyCpu	(field, delta2, LL, aMass2, eRes, shift, pot, mapmask);
			break;

		case	FIELD_AXION:
			setName		("Energy Axion");
			energyThetaCpu	(field, delta2, aMass2, eRes, map, false);
			break;

		case	FIELD_AXION_MOD:
			setName		("Energy Axion (mod)");
			energyThetaCpu	(field, delta2, aMass2, eRes, map, true);
			break;

		case	FIELD_NAXION:
			setName		("Energy Naxion");
			energyNaxionCpu	(field, delta2, aMass2, eRes, map);
			break;

		case	FIELD_PAXION:
			setName		("Energy Paxion");
			energyPaxionCpu	(field, eRes, map);
			break;

		default:
			LogError ("Energy not supported for this kind of field");
			return;
	}
}

using namespace profiler;

void	energy	(Scalar *field, void *eRes, const EnType emap, const double shift)
{
	if ( (emap & EN_MAP) && (field->Field() == FIELD_SAXION) && field->LowMem())
	{
		LogError ("Error: Can't compute energy map for saxion with lowmem kernels\n");
		return;
	}

	if ( (emap & EN_MASK) && !(field->sDStatus() & SD_MASK))
	{
		LogError ("Error: Can't compute masked energy because there is no mask!\n");
		return;
	}


	LogMsg  (VERB_HIGH, "Called energy");
	Profiler &prof = getProfiler(PROF_ENERGY);

	void *eTmp;
	trackAlloc(&eTmp, 256);

	auto LL   = field->LambdaP(); // obsolete
	auto pot  = field->BckGnd()->QcdPot();
	auto dlta = field->Delta();

	auto	eDark = std::make_unique<Energy>(field, LL, dlta, eTmp, pot, shift, emap);

	if	(!field->Folded())
	{
		Folder	munge(field);
		munge(FOLD_ALL);
	}

	field->exchangeGhosts(FIELD_M);

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

	// Changed SAXION CASE from 10 to 22 for now there are masked energies + rho field
	const int size = field->Field() == FIELD_SAXION ? 23 : 5;

	MPI_Allreduce(eTmp, eRes, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	trackFree(eTmp);

	const double Vt  = 1./(field->TotalSize());
	const double Vt2 = 1./( static_cast<double*>(eRes)[22] );
	const double mi  = 1.*( static_cast<double*>(eRes)[22] );

	#pragma unroll
	for (int i=0; i<11; i++)
		static_cast<double*>(eRes)[i] *= Vt;

	#pragma unroll
	for (int i=11; i<22; i++)
		static_cast<double*>(eRes)[i] *= Vt2;

	prof.stop();

	field->setReduced(false);
	if (emap & EN_MAP) {
		field->setM2     (M2_ENERGY);
	}

	// masked energy is non-deterministic but if mask is small is negligible
	double flops = (field->Field() == FIELD_SAXION ? (pot == VQCD_1 ? 111 : 112) : 25)*field->Size()*1e-9;
	double bytes = 8.*field->DataSize()*field->Size()*1e-9;

	if (emap & EN_MAP) {
		eDark->appendName(" Map");
		bytes *= 9./8;
	}

	eDark->add(flops, bytes);		// Flops are not exact
	prof.add(eDark->Name(), flops, bytes);

	LogMsg	(VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", eDark->Name().c_str(), prof.Prof()[eDark->Name()].GFlops(), prof.Prof()[eDark->Name()].GBytes());

	return;
}
