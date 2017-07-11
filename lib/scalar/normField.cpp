#include <cstdio>
#include <cstdlib>
#include "scalar/scalarField.h"
#include "enum-field.h"

#include "scalar/normXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "scalar/normGpu.h"
#endif

#include "utils/utils.h"

class	NormaliseField : public Tunable
{
	private:

	const FieldIndex fIdx;

	Scalar	*axionField;

	public:

		 NormaliseField(Scalar *field, const FieldIndex fIdx);
		~NormaliseField() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	NormaliseField::NormaliseField(Scalar *field, const FieldIndex fIdx) : axionField(field), fIdx(fIdx)
{
}

void	NormaliseField::runGpu	()
{
#ifdef	USE_GPU
	normGpu(axionField, fIdx);
#else
	LogError ("Gpu support not built");
	exit(1);
#endif
}

void	NormaliseField::runCpu	()
{
	normXeon(axionField, fIdx);
}

void	NormaliseField::runXeon	()
{
#ifdef	USE_XEON
	normXeon(axionField, fIdx);
#else
	LogError ("Xeon Phi support not built");
	exit(1);
#endif
}

using namespace profiler;

void	normaliseField	(Scalar *field, const FieldIndex fIdx)
{
	LogMsg  (VERB_HIGH, "Called normalise field");
	Profiler &prof = getProfiler(PROF_SCALAR);

	NormaliseField *nField = new NormaliseField(field, fIdx);

	prof.start();
	nField->setName("Normalise");

	switch (field->Device())
	{
		case DEV_CPU:
			nField->runCpu ();
			break;

		case DEV_GPU:
			nField->runGpu ();
			break;

		case	DEV_XEON:
			nField->runXeon();
			break;

		default:
			LogError ("Not a valid device");
			prof.stop();
			delete nField;
			return;
	}

	nField->add(field->Size()*5.e-9, field->DataSize()*field->Size()*1.e-9);
	prof.stop();
	prof.add(nField->Name(), nField->GFlops(), nField->GBytes());

	LogMsg  (VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", nField->Name().c_str(), prof.Prof()[nField->Name()].GFlops(), prof.Prof()[nField->Name()].GBytes());

	delete	nField;

	return;
}
