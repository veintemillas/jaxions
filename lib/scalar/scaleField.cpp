#include <memory>
#include "scalar/scalarField.h"
#include "enum-field.h"

#include "scalar/scaleXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "scalar/scaleGpu.h"
#endif

#include "utils/utils.h"

class	ScaleField : public Tunable
{
	private:

	const double factor;

	const FieldIndex fIdx;

	Scalar	*axionField;

	public:

		 ScaleField(Scalar *field, const FieldIndex fIdx, const double factor);
		~ScaleField() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	ScaleField::ScaleField(Scalar *field, const FieldIndex fIdx, const double factor) : axionField(field), fIdx(fIdx), factor(factor)
{
}

void	ScaleField::runGpu	()
{
#ifdef	USE_GPU
	scaleGpu(axionField, fIdx, factor);
#else
	LogError ("Gpu support not built");
	exit(1);
#endif
}

void	ScaleField::runCpu	()
{
	scaleXeon(axionField, fIdx, factor);
}

using namespace profiler;

void	scaleField	(Scalar *field, const FieldIndex fIdx, const double factor)
{
	LogMsg  (VERB_HIGH, "Called scale field");
	Profiler &prof = getProfiler(PROF_SCALAR);

	auto	scale = std::make_unique<ScaleField>    (field, fIdx, factor);

	prof.start();
	scale->setName("Scale");

	switch (field->Device())
	{
		case DEV_CPU:
			scale->runCpu ();
			break;

		case DEV_GPU:
			scale->runGpu ();
			break;

		default:
			LogError ("Error: invalid device");
			prof.stop();
			return;
	}

	scale->add(field->Size()*2.e-9, field->DataSize()*field->Size()*1.e-9);

	prof.stop();
	prof.add(scale->Name(), scale->GFlops(), scale->GBytes());

	LogMsg  (VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", scale->Name().c_str(), prof.Prof()[scale->Name()].GFlops(), prof.Prof()[scale->Name()].GBytes());

	return;
}
