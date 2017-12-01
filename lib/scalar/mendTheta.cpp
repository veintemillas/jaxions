#include <memory>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"

#include "scalar/mendThetaXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "scalar/mendThetaGpu.h"
#endif

#include "utils/utils.h"

using namespace profiler;

class	MendTheta : public Tunable
{
	private:

	Scalar	*axionField;

	public:

		 MendTheta(Scalar *field) : axionField(field) {};
		~MendTheta() {};

	size_t	runCpu	();
	size_t	runGpu	();
};

size_t	MendTheta::runGpu	()
{
#ifdef	USE_GPU
	LogMsg (VERB_NORMAL, "Using cpu for mendTheta");
	axionField->transferCpu(FIELD_MV);
	Folder munge(axionField);
	munge(FOLD_ALL);
	auto nJmps = runCpu();
	munge(UNFOLD_ALL);
	axionField->transferDev(FIELD_MV);
	return	nJmps;
//	return	mendThetaGpu(axionField);
#else
	LogError ("Error: gpu support not built");
	exit(1);
#endif
}

size_t	MendTheta::runCpu	()
{
	return	mendThetaXeon(axionField);
}

size_t	mendTheta	(Scalar *field)
{
	size_t	nJmps = 0;

	if (!(field->Field() & FIELD_AXION)) {
		LogError ("Error: mendTheta can only be applied to axion fields");
		return	0;
	}

	auto	theta = std::make_unique<MendTheta>    (field);
	Profiler &prof = getProfiler(PROF_SCALAR);

	theta->setName("Mend Theta");
	prof.start();

	switch (field->Device())
	{
		case DEV_CPU:
			nJmps = theta->runCpu ();
			break;

		case DEV_GPU:
			nJmps = theta->runGpu ();
			break;

		default:
			LogError ("Error: invalid device");
			exit(1);
			break;
	}

	theta->add(field->Size()*6.e-9, field->DataSize()*field->Size()*7.e-9);

	prof.stop();
	prof.add(theta->Name(), theta->GFlops(), theta->GBytes());

	LogMsg(VERB_NORMAL, "%s finished with %lu jumps", theta->Name().c_str(), nJmps);
	LogMsg  (VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", theta->Name().c_str(), prof.Prof()[theta->Name()].GFlops(), prof.Prof()[theta->Name()].GBytes());

	return	nJmps;
}
