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

	int	runCpu	();
	int	runGpu	();
	int	runXeon	();
};

int	MendTheta::runGpu	()
{
#ifdef	USE_GPU
	const uint uLx = axionField->Length(), uS = axionField->Surf(), uV = axionField->Size(), uLz = axionField->Depth();

	uint	count = 0;
	int	mIter = 0;

	const double zVar = *(axionField->zV());

	auto wJmp = mendSliceXeon (axionField, 0);	// After exchanging ghosts, operate on slice 0 (ghost), so we go up to Lz-1

	for (uint i=0; i<(uLz-1); i++) {
		do {
			axionField->exchangeGhosts(FIELD_M);
			count = mendThetaGpu(axionField->mGpu(), axionField->vGpu(), zVar, uLx, (i+1)*uS, i*uS, axionField->Precision(), ((cudaStream_t *)axionField->Streams())[0]);
			mIter++;
		}	while (count != 0);
	}

	return	mIter/uLz;
#else
	LogError ("Error: gpu support not built");
	exit(1);
#endif
}

int	MendTheta::runCpu	()
{
	bool	wJmp  = false;
	int	mIter = 0;

	do {
		axionField->exchangeGhosts(FIELD_M);
		wJmp = mendThetaXeon(axionField);
		mIter++;
	}	while (wJmp);

	return	mIter;
}

void	mendTheta	(Scalar *field)
{
	int mIter = 0;

	if (!(field->Field() & FIELD_AXION)) {
		LogError ("Error: mendTheta can only be applied to axion fields");
		return;
	}

	auto	theta = std::make_unique<MendTheta>    (field);
	Profiler &prof = getProfiler(PROF_SCALAR);

	theta->setName("Mend Theta");
	prof.start();

	switch (field->Device())
	{
		case DEV_CPU:
			mIter = theta->runCpu ();
			break;

		case DEV_GPU:
			mIter = theta->runGpu ();
			break;

		default:
			LogError ("Error: invalid device");
			exit(1);
			break;
	}

	theta->add(field->Size()*6.e-9*mIter, field->DataSize()*field->Size()*7.e-9*mIter);

	prof.stop();
	prof.add(theta->Name(), theta->GFlops(), theta->GBytes());

	LogMsg(VERB_NORMAL, "%s finished in %d iterations", theta->Name().c_str(), mIter);
	LogMsg  (VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", theta->Name().c_str(), prof.Prof()[theta->Name()].GFlops(), prof.Prof()[theta->Name()].GBytes());

	return;
}
