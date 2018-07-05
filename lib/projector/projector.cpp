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

	std::function<Float(Float)> filter;

	public:

		 Reducer(Scalar *field, std::function<Float(Float)> myFilter) :
			axion(field), filter(myFilter) {};
		~Reducer() {};

	void	runCpu	();
	void	runGpu	();
};

template<typename Float>
void	Projector<Float>::runGpu	()
{
#ifdef	USE_GPU
	LogMsg	 (VERB_NORMAL, "FIXME!! Projector runs on cpu");

	axionField->transferCpu(FIELD_M2);
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
	const auto Sz = axion->Size();
	const auto Tz = axion->TotalDepth();

	Float *inData  = static_cast<Float*>(axion->m2Cpu());
	Float *medData = static_cast<Float*>(axion->mCpu())[2*(Sz+Sf)];	// Backghosts in m
	Float *outData = static_cast<Float*>(axion->mCpu());		// Frontghosts in m

	#pragma	omp parallel for schedule(static)
	for (int pt = 0; pt < Sf; pt++) {
		outData[pt] = 0.;

		for (int zc = 0; zc < axion->Depth(); zc++) {
			medData[pt] += filter(inData[pt + Sf*zc]);
		}
	}

	MPI_Reduce (medData, outData, Sf, Float, MPI_SUM, 0, MPI_COMM_WORLD);

	Float	oZ = 1./((Float) Tz);

	#pragma	omp parallel for schedule(static)
	for (int pt = 0; pt < Sf; pt++)
		outData[pt] *= oz;
}	

template<typename Float>
void	projectField	(Scalar *field, std::function<Float(Float)> myFilter)
{
	if (field->LowMem()) {
		LogError("Error: lowmem not supported");
		return;
	}

	if (field->m2Status() != M2_ENERGY) {
		LogError("Error: projector only works with energies and the energy has not been computed");
		return;
	}

	auto	projector = std::make_unique<Projector<Float>> (field, myFilter);
	Profiler &prof = getProfiler(PROF_PROJECTOR);

	std::stringstream ss;
	ss << "Project " << field->Length() << "x" << field->Length() << "x" << field->TotalDepth() << " with " << commSize() << "ranks"; 

	reducer->setName(ss.str().c_str());
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

	projector->add(field->Size(), field->DataSize()*(field->Size() + field->Surf()));

	prof.stop();
	prof.add(projector->Name(), projector->GFlops(), projector->GBytes());

	LogMsg (VERB_NORMAL, "%s finished", projector->Name().c_str());
	LogMsg (VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", projector->Name().c_str(), prof.Prof()[projector->Name()].GFlops(), prof.Prof()[projector->Name()].GBytes());

	return;
}

void	projectField	(Scalar *field, std::function<float(float)> myFilter) {
	if (field->Precision() != FIELD_SINGLE) {
		LogError("Error: double precision filter for single precision configuration");
		return	nullptr;
	}
	
	projectField<float>(field, myFilter);
	return;
}

void	projectField	(Scalar *field, std::function<double(double)> myFilter) {
	if (field->Precision() != FIELD_DOUBLE) {
		LogError("Error: single precision filter for double precision configuration");
		return;
	}
	
	projectField(field, myFilter);
	return;
}
