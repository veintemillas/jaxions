#include <memory>
#include <complex>
#include <functional>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"

#include "fft/fftCode.h"
#include "utils/utils.h"

using namespace std;
using namespace profiler;

template<typename Float>
class	Projector : public Tunable
{
	private:

	Scalar	     *axion;

	std::function<double(double)> filter;

	public:

		 Projector(Scalar *field, std::function<double(double)> myFilter) :
			   axion(field), filter(myFilter) {};
		~Projector() {};

	void	runCpu	();
	void	runGpu	();
};

template<typename Float>
void	Projector<Float>::runGpu	()
{
#ifdef	USE_GPU
	LogMsg	 (VERB_NORMAL, "FIXME!! Projector runs on cpu");

	axion->transferCpu(FIELD_M2);
	runCpu();
	return;
#else
	LogError ("Error: gpu support not built");
	exit(1);
#endif
}

template<typename Float>
void	Projector<Float>::runCpu	() {

	const auto Sf = axion->Surf();
	const auto Tz = axion->TotalDepth();

	Float *inData  = static_cast<Float*>(axion->m2Cpu());
	Float *medData = static_cast<Float*>(axion->mBackGhost ());
	Float *outData = static_cast<Float*>(axion->mFrontGhost());

	#pragma	omp parallel for schedule(static)
	for (size_t pt = 0; pt < Sf; pt++) {
		outData[pt] = (Float) 0.;
		medData[pt] = (Float) 0.;

		for (size_t zc = 0; zc < axion->Depth(); zc++) {
			Float  x = inData[pt + Sf*zc];
			double y = filter((double) x);
			medData[pt] += (Float) y;
		}
	}

	if (axion->Precision() == FIELD_DOUBLE)
		MPI_Reduce (medData, outData, Sf, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	else
		MPI_Reduce (medData, outData, Sf, MPI_FLOAT,  MPI_SUM, 0, MPI_COMM_WORLD);

	Float	oZ = 1./((Float) Tz);

	#pragma	omp parallel for schedule(static)
	for (size_t pt = 0; pt < Sf; pt++)
		outData[pt] *= oZ;
}

template<typename Float>
void	projectField	(Scalar *field, std::function<double(double)> myFilter)
{
	if (field->LowMem()) {
		LogError("Error: lowmem not supported");
		return;
	}

	if (field->m2Status() != M2_ENERGY) {
		if (field->m2Status() != M2_ANTIMASK) {
			if (field->m2Status() != M2_MASK) {
				LogError("Error: projector only works with energies or anti-masks and the energy has not been computed");
				return;
		}}
	}

	auto	projector = std::make_unique<Projector<Float>> (field, myFilter);
	Profiler &prof = getProfiler(PROF_PROJECTOR);

	std::stringstream ss;
	ss << "Project " << field->Length() << "x" << field->Length() << "x" << field->TotalDepth() << " with " << commSize() << "ranks";

	projector->setName(ss.str().c_str());
	prof.start();

	switch (field->Device())
	{
		case DEV_CPU:
			projector->runCpu ();
			break;

		case DEV_GPU:
			projector->runGpu ();
			break;

		default:
			LogError ("Error: invalid device, projection ignored");
			prof.stop();
			return;
			break;
	}

	projector->add(field->Size()*1e-9, field->DataSize()*(field->Size() + field->Surf())*1e-9);

	prof.stop();
	prof.add(projector->Name(), projector->GFlops(), projector->GBytes());

	LogMsg (VERB_NORMAL, "%s finished", projector->Name().c_str());
	LogMsg (VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", projector->Name().c_str(), prof.Prof()[projector->Name()].GFlops(), prof.Prof()[projector->Name()].GBytes());

	return;
}

void	projectField	(Scalar *field, std::function<double(double)> myFilter) {
	if (field->Precision() == FIELD_DOUBLE)
		projectField<double>(field, myFilter);
	else
		projectField<float> (field, myFilter);

	return;
}
