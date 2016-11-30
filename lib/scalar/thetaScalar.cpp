#include <cstdio>
#include <cstdlib>
#include "scalar/scalarField.h"
#include "enum-field.h"

#include "scalar/thetaXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "scalar/thetaGpu.h"
#endif

#include "utils/flopCounter.h"
#include "utils/memAlloc.h"

class	CmplxToTheta
{
	private:

	Scalar	*axionField;

	public:

		 CmplxToTheta(Scalar *field);
		~() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	ScaleField::ScaleField(Scalar *field) : axionField(field)
{
}

void	ScaleField::runGpu	()
{
#ifdef	USE_GPU
#else
	printf("Gpu support not built");
	exit(1);
#endif
}

void	ScaleField::runCpu	()
{
	toThetaXeon(axionField);
}

void	ScaleField::runXeon	()
{
#ifdef	USE_XEON
	toThetaXeon(axionField);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	cmplxToTheta	(Scalar *field, FlopCounter *fCount)
{
	CmplxToTheta *theta = new CmplxToTheta(field);

	switch (field->Device())
	{
		case DEV_CPU:
			theta->runCpu ();
			break;

		case DEV_GPU:
			theta->runGpu ();
			break;

		case	DEV_XEON:
			theta->runXeon();
			break;

		default:
			printf ("Not a valid device\n");
			break;
	}

	delete	theta;

	field->setField(AXION_FIELD);

	fCount->addFlops(field->Size()*2.e-9, field->DataSize()*field->Size()*1.e-9);

	return;
}
