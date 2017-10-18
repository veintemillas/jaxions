#include <memory>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"

#include "scalar/thetaXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "scalar/thetaGpu.h"
#endif

#include "utils/utils.h"

using namespace profiler;

class	CmplxToTheta : public Tunable
{
	private:

	Scalar	*axionField;
	const double shift;

	public:

		 CmplxToTheta(Scalar *field, const double sh);
		~CmplxToTheta() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	CmplxToTheta::CmplxToTheta(Scalar *field, const double sh) : axionField(field), shift(sh)
{
}

void	CmplxToTheta::runGpu	()
{
#ifdef	USE_GPU
	toThetaGpu(axionField,shift);
#else
	LogError ("Error: gpu support not built");
	exit(1);
#endif
}

void	CmplxToTheta::runCpu	()
{
	toThetaXeon(axionField,shift);
}

void	cmplxToTheta	(Scalar *field, const double shift, const bool wMod)
{
	auto	theta = std::make_unique<CmplxToTheta>    (field, shift);
	Profiler &prof = getProfiler(PROF_SCALAR);

	Folder munge(field);

	munge(UNFOLD_ALL);

	theta->setName("Complex to Theta");
	prof.start();

	switch (field->Device())
	{
		case DEV_CPU:
			theta->runCpu ();
			break;

		case DEV_GPU:
			theta->runGpu ();
			break;

		default:
			LogError ("Error: invalid device");
			exit(1);
			break;
	}

	if (wMod == true)
		field->setField(FIELD_AXION_MOD);
	else {
		field->setField(FIELD_AXION);
		// Call MendTheta here!!
	}

	field->setLowMem(false);

	theta->add(field->Size()*12.e-9, field->DataSize()*field->Size()*6.e-9);

	prof.stop();
	prof.add(theta->Name(), theta->GFlops(), theta->GBytes());

	LogMsg  (VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", theta->Name().c_str(), prof.Prof()[theta->Name()].GFlops(), prof.Prof()[theta->Name()].GBytes());

	munge(FOLD_ALL);

	return;
}
