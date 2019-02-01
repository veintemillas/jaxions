#include <cstdio>
#include <cstdlib>
#include "scalar/scalarField.h"
#include "enum-field.h"

#include "scalar/theta2CmplxXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "scalar/theta2CmplxGpu.h"
#endif

#include "utils/utils.h"

class	Theta2Cmplx : public Tunable
{
	private:

	Scalar		*axionField;

	public:

		 Theta2Cmplx(Scalar *field);
		~Theta2Cmplx() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	Theta2Cmplx::Theta2Cmplx(Scalar *field) : axionField(field)
{
}

void	Theta2Cmplx::runGpu	()
{
#ifdef	USE_GPU
	th2cxGpu(axionField);
#else
	LogError ("Gpu support not built");
	exit(1);
#endif
}

void	Theta2Cmplx::runCpu	()
{
	th2cxXeon(axionField);
}

using namespace profiler;

void	theta2Cmplx	(Scalar *field)
{
	LogMsg  (VERB_HIGH, "Called theta2cmplx ");
	Profiler &prof = getProfiler(PROF_SCALAR);

	Theta2Cmplx *t2c = new Theta2Cmplx(field);

	prof.start();
	t2c->setName("theta2complex");

	switch (field->Device())
	{
		case DEV_CPU:
			t2c->runCpu ();
			break;

		case DEV_GPU:
			t2c->runGpu ();
			break;

		default:
			LogError ("Not a valid device");
			prof.stop();
			delete t2c;
			return;
	}

	t2c->add(field->Size()*5.e-9, field->DataSize()*field->Size()*1.e-9);
	prof.stop();
	prof.add(t2c->Name(), t2c->GFlops(), t2c->GBytes());

	LogMsg  (VERB_HIGH, "%s reporting %lf GFlops %lf GBytes", t2c->Name().c_str(), prof.Prof()[t2c->Name()].GFlops(), prof.Prof()[t2c->Name()].GBytes());

	delete	t2c;

	return;
}
