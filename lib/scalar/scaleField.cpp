#include <cstdio>
#include <cstdlib>
#include "scalar/scalarField.h"
#include "enum-field.h"

#include "scalar/scaleXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "scalar/scaleGpu.h"
#endif

#include "utils/flopCounter.h"
#include "utils/memAlloc.h"

class	ScaleField
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
#else
	printf("Gpu support not built");
	exit(1);
#endif
}

void	ScaleField::runCpu	()
{
	scaleXeon(axionField, fIdx, factor);
}

void	ScaleField::runXeon	()
{
#ifdef	USE_XEON
	scaleXeon(axionField, fIdx, factor);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	scaleField	(Scalar *field, const FieldIndex fIdx, const double factor, FlopCounter *fCount)
{
	ScaleField *scale = new ScaleField(field, fIdx, factor);

	switch (field->Device())
	{
		case DEV_CPU:
			scale->runCpu ();
			break;

		case DEV_GPU:
			scale->runGpu ();
			break;

		case	DEV_XEON:
			scale->runXeon();
			break;

		default:
			printf ("Not a valid device\n");
			break;
	}

	delete	scale;

	fCount->addFlops(field->Size()*2.e-9, field->DataSize()*field->Size()*1.e-9);

	return;
}
