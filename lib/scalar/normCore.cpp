#include <cstdio>
#include <cstdlib>
#include "scalar/scalarField.h"
#include "enum-field.h"

#include "scalar/normCoreXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
//	#include "scalar/normCoreGpu.h"
#endif

#include "utils/flopCounter.h"
#include "utils/memAlloc.h"

class	NormCoreField
{
	private:

	const double alpha;
	Scalar *axionField;

	public:

		 NormCoreField(Scalar *field, const double alpha);
		~NormCoreField() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	NormCoreField::NormCoreField(Scalar *field, const double alpha) : axionField(field), alpha(alpha)
{
}

void	NormCoreField::runGpu	()
{
#ifdef	USE_GPU
	printf("Field will be core-smoothed in the CPU");
	normCoreXeon(axionField, alpha);
#else
	printf("Gpu support not built");
	exit(1);
#endif
}

void	NormCoreField::runCpu	()
{
	normCoreXeon(axionField, alpha);
}

void	NormCoreField::runXeon	()
{
#ifdef	USE_XEON
	normCoreXeon(axionField, alpha);
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	normCoreField	(Scalar *field, const double alpha, FlopCounter *fCount)
{
	NormCoreField *nField = new NormCoreField(field, alpha);

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
			printf ("Not a valid device\n");
			break;
	}

	delete	nField;

//	fCount->addFlops(field->Size()*5.e-9, field->dataSize()*field->Size()*1.e-9);

	return;
}
