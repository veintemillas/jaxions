#include <cstdio>
#include <cstdlib>
#include "scalar/scalarField.h"
#include "enum-field.h"

#include "scalar/normXeon.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
//	#include "scalar/normGpu.h"
#endif

#include "utils/flopCounter.h"
#include "utils/memAlloc.h"

class	NormaliseField
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
	printf("Field will be normalized in the CPU");
	normXeon(axionField, fIdx);
#else
	printf("Gpu support not built");
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
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	normaliseField	(Scalar *field, const FieldIndex fIdx, FlopCounter *fCount)
{
	NormaliseField *nField = new NormaliseField(field, fIdx);

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

	fCount->addFlops(field->Size()*5.e-9, field->DataSize()*field->Size()*1.e-9);

	return;
}
