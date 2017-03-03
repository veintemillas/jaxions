#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "enum-field.h"
#include "scalar/scalarField.h"
#include "scalar/scaleField.h"
#include "scalar/normField.h"
#include "scalar/normCore.h"
#include "gen/momConf.h"
#include "gen/randXeon.h"
#include "gen/smoothXeon.h"
#include "io/readWrite.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
//	#include "gen/randGpu.h"
	#include "gen/smoothGpu.h"
#endif

#include "utils/flopCounter.h"
#include "utils/memAlloc.h"

//#include <mpi.h>

class	ConfGenerator
{
	private:

	ConfType cType;

	size_t	kMax;
	size_t	sIter;

	double	kCrt;
	double	alpha;

	int	index;

	Scalar	*axionField;
	FlopCounter *fCount;

	public:

		 ConfGenerator(Scalar *field, ConfType type, FlopCounter *fCount);
		 ConfGenerator(Scalar *field, ConfType type, size_t parm1, double parm2, FlopCounter *fCount);
		~ConfGenerator() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	ConfGenerator::ConfGenerator(Scalar *field, ConfType type, size_t parm1, double parm2, FlopCounter *fCount) : axionField(field), cType(type), fCount(fCount)
{
	switch (type)
	{
		case CONF_KMAX:
		case CONF_TKACHEV:

		kMax = parm1;
		kCrt = parm2;
		alpha = 0.143;
		break;

		case CONF_SMOOTH:

		sIter = parm1;
		alpha = parm2;
		break;

		case CONF_READ:

		index = static_cast<int>(parm1);
		break;

		case CONF_NONE:
		default:
		break;
	}
}

	ConfGenerator::ConfGenerator(Scalar *field, ConfType type, FlopCounter *fCount) : axionField(field), cType(type), fCount(fCount)
{
	switch (type)
	{
		case CONF_KMAX:
		case CONF_TKACHEV:

		kMax = 2;
		kCrt = 1.0;
		alpha = 0.143;
		break;

		case CONF_SMOOTH:

		sIter = 40;
		alpha = 0.143;
		break;

		case CONF_READ:

		index = 0;
		break;

		case CONF_NONE:
		default:
		break;
	}
}

void	ConfGenerator::runGpu	()
{
#ifdef	USE_GPU
	printf("The configuration will be generated on host");

	switch (cType)
	{
		case CONF_NONE:
		break;

		case CONF_READ:
		readConf (&axionField, index);
		break;

		case CONF_TKACHEV:
		momConf(axionField, kMax, kCrt);
		axionField->fftCpu(1);
		axionField->sendGhosts(FIELD_M, COMM_SDRV);
		axionField->sendGhosts(FIELD_M, COMM_WAIT);

		cudaMemcpy (axionField->vGpu(), static_cast<char *> (axionField->mGpu()) + axionField->DataSize()*axionField->Surf(), axionField->DataSize()*axionField->Size(), cudaMemcpyDeviceToDevice);
		scaleField (axionField, FIELD_M, *axionField->zV(), fCount);

		axionField->transferDev(FIELD_MV);
		break;

		case CONF_KMAX:
		momConf(axionField, kMax, kCrt);
		axionField->fftCpu(1);

		axionField->sendGhosts(FIELD_M, COMM_SDRV);
		axionField->sendGhosts(FIELD_M, COMM_WAIT);
		axionField->transferDev(FIELD_M);

		normaliseField(axionField, FIELD_M, fCount);
		normCoreField (axionField, fCount);
		scaleField (axionField, FIELD_M, *axionField->zV(), fCount);

		axionField->transferCpu(FIELD_MV);
		break;

		case CONF_SMOOTH:
		randConf (axionField);

		axionField->transferDev(FIELD_M);

		smoothGpu (axionField, sIter, alpha);
		normCoreField (axionField, fCount);
		scaleField (axionField, FIELD_M, *axionField->zV(), fCount);

		axionField->transferCpu(FIELD_MV);
		break;
	}

#else
	printf("Gpu support not built");
	exit(1);
#endif
}

void	ConfGenerator::runCpu	()
{
	switch (cType)
	{
		case CONF_NONE:
		break;

		case CONF_READ:
		readConf (&axionField, index);
		break;

		case CONF_TKACHEV:
		momConf(axionField, kMax, kCrt);
		axionField->fftCpu(1);
		axionField->exchangeGhosts(FIELD_M);
		break;

		case CONF_KMAX:
		momConf(axionField, kMax, kCrt);
		axionField->fftCpu(1);
		axionField->exchangeGhosts(FIELD_M);
		normaliseField(axionField, FIELD_M, fCount);
		normCoreField (axionField, fCount);
		break;

		case CONF_SMOOTH:
		randConf (axionField);
		smoothXeon (axionField, sIter, alpha);
		normCoreField (axionField, fCount);
		break;
	}

	if ((cType == CONF_KMAX) || (cType == CONF_SMOOTH) || (cType == CONF_TKACHEV))
	{
		memcpy (axionField->vCpu(), static_cast<char *> (axionField->mCpu()) + axionField->DataSize()*axionField->Surf(), axionField->DataSize()*axionField->Size());
		scaleField (axionField, FIELD_M, *axionField->zV(), fCount);
	}

}

void	ConfGenerator::runXeon	()
{
#ifdef	USE_XEON
	printf("The configuration will be generated on host");
	runCpu();
#else
	printf("Xeon Phi support not built");
	exit(1);
#endif
}

void	genConf	(Scalar *field, ConfType cType, FlopCounter *fCount)
{
	ConfGenerator *cGen = new ConfGenerator(field, cType, fCount);

	switch (field->Device())
	{
		case DEV_CPU:
			cGen->runCpu ();
			field->exchangeGhosts(FIELD_M);
			break;

		case DEV_GPU:
			cGen->runGpu ();
			field->exchangeGhosts(FIELD_M);
			break;

		case DEV_XEON:
			cGen->runXeon();
			field->exchangeGhosts(FIELD_M);
			break;

		default:
			printf ("Not a valid device\n");
			break;
	}

	delete	cGen;

	return;
}

void	genConf	(Scalar *field, ConfType cType, size_t parm1, double parm2, FlopCounter *fCount)
{
	ConfGenerator *cGen = new ConfGenerator(field, cType, parm1, parm2, fCount);

	switch (field->Device())
	{
		case DEV_CPU:
			cGen->runCpu ();
			field->exchangeGhosts(FIELD_M);
			break;

		case DEV_GPU:
			cGen->runGpu ();
			field->exchangeGhosts(FIELD_M);
			break;

		case DEV_XEON:
			cGen->runXeon();
			field->exchangeGhosts(FIELD_M);
			break;

		default:
			printf ("Not a valid device\n");
			break;
	}

	delete	cGen;

	return;
}
