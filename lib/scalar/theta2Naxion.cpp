#include <cstdio>
#include <cstdlib>
#include "scalar/scalarField.h"
#include "enum-field.h"
#include "scalar/folder.h"

#include "propagator/propagator.h"
#include "scalar/theta2NaxionXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	// #include "scalar/theta2CmplxGpu.h"
#endif

#include "utils/utils.h"

class	Theta2Naxion : public Tunable
{
	private:

	Scalar		*axionField;

	public:

		 Theta2Naxion(Scalar *field);
		~Theta2Naxion() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	Theta2Naxion::Theta2Naxion(Scalar *field) : axionField(field)
{
}

void	Theta2Naxion::runGpu	()
{
#ifdef	USE_GPU
	// th2NaxionGpu(axionField);
	LogError ("Gpu version not coded!");
#else
	LogError ("Called Gpu support not built");
	exit(1);
#endif
}

void	Theta2Naxion::runCpu	()
{
	th2NaxionXeon(axionField);
}

using namespace profiler;

void	thetaToNaxion	(Scalar *field)
{
	LogMsg  (VERB_NORMAL, "Called thetaToNaxion ");
	Profiler &prof = getProfiler(PROF_SCALAR);

	Folder munge(field);

	munge(UNFOLD_ALL);

	Theta2Naxion *t2n = new Theta2Naxion(field);

	prof.start();
	t2n->setName("theta2naxion");

	switch (field->Device())
	{
		case DEV_CPU:
			t2n->runCpu ();
			break;

		case DEV_GPU:
			t2n->runGpu ();
			break;

		default:
			LogError ("Not a valid device");
			prof.stop();
			delete t2n;
			return;
	}

	field->setField(FIELD_NAXION);

	munge(FOLD_ALL);

		/* profiling */

	t2n->add(field->Size()*5.e-9, field->DataSize()*field->Size()*1.e-9);
	prof.stop();
	prof.add(t2n->Name(), t2n->GFlops(), t2n->GBytes());

	LogMsg  (VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", t2n->Name().c_str(), prof.Prof()[t2n->Name()].GFlops(), prof.Prof()[t2n->Name()].GBytes());

	delete	t2n;

	resetPropagator(field);

	return;
}
