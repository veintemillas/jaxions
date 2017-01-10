#include <cstdio>
#include <cstdlib>
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

#include "utils/flopCounter.h"
#include "utils/memAlloc.h"

class	CmplxToTheta
{
	private:

	Scalar	*axionField;

	public:

		 CmplxToTheta(Scalar *field);
		~CmplxToTheta() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	CmplxToTheta::CmplxToTheta(Scalar *field) : axionField(field)
{
}

void	CmplxToTheta::runGpu	()
{
#ifdef	USE_GPU
#else
	printf("Gpu support not built");
	exit(1);
#endif
}

void	CmplxToTheta::runCpu	()
{
	toThetaXeon(axionField);
}

void	CmplxToTheta::runXeon	()
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
	Folder	     *munge = new Folder(field);
	//NORMALLY CALLED WHEN FOLDED
	(*munge)(UNFOLD_ALL);
	delete	munge;

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


	field->setField(FIELD_AXION);

	//printf("folding... \n");fflush(stdout);

	Folder	     *munge2 = new Folder(field);
	(*munge2)(FOLD_ALL);

	delete	munge2;

	fCount->addFlops(field->Size()*12.e-9, field->DataSize()*field->Size()*6.e-9);

	return;
}
