#include <cstdio>
#include <cstdlib>
#include "scalar/scalarField.h"
#include "enum-field.h"
#include "utils/utils.h"

#include "scalar/normCoreXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "scalar/normCoreGpu.h"
#endif

#include "utils/utils.h"

class	NormCoreField : public Tunable
{
	private:

	Scalar *axionField;

	public:

		 NormCoreField(Scalar *field);
		~NormCoreField() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	NormCoreField::NormCoreField(Scalar *field) : axionField(field)
{
}

void	NormCoreField::runGpu	()
{
#ifdef	USE_GPU
	normCoreGpu(axionField);
#else
	LogError ("Gpu support not built");
	exit(1);
#endif
}

void	NormCoreField::runCpu	()
{
	normCoreXeon(axionField);
}

using namespace profiler;

void	normCoreField	(Scalar *field)
{
	LogMsg  (VERB_HIGH, "Called normalise for string cores");
	Profiler &prof = getProfiler(PROF_SCALAR);

	NormCoreField *nField = new NormCoreField(field);

	prof.start();
	nField->setName("Normalise Core");

	switch (field->Device())
	{
		case DEV_CPU:
			nField->runCpu ();
			break;

		case DEV_GPU:
			nField->runGpu ();
			break;

		default:
			LogError ("Not a valid device");
			break;
	}

	nField->add(field->Size()*106.e-9, field->DataSize()*field->Size()*8.e-9);	// Assumes gradient^2 > 0.001 always
	prof.stop();
	prof.add(nField->Name(), nField->GFlops(), nField->GBytes());

	LogMsg  (VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", nField->Name().c_str(), prof.Prof()[nField->Name()].GFlops(), prof.Prof()[nField->Name()].GBytes());

	delete	nField;

	return;
}
