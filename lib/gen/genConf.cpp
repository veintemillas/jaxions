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
	#include "gen/momGpu.h"
	#include "gen/randGpu.h"
	#include "gen/smoothGpu.h"
#endif

#include "utils/utils.h"
#include "fft/fftCode.h"

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

	public:

		 ConfGenerator(Scalar *field, ConfType type);
		 ConfGenerator(Scalar *field, ConfType type, size_t parm1, double parm2);
		~ConfGenerator() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	ConfGenerator::ConfGenerator(Scalar *field, ConfType type, size_t parm1, double parm2) : axionField(field), cType(type)
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

	ConfGenerator::ConfGenerator(Scalar *field, ConfType type) : axionField(field), cType(type)
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

using namespace std;
using namespace profiler;

void	ConfGenerator::runGpu	()
{
#ifdef	USE_GPU
	LogMsg (VERB_HIGH, "Random numbers will be generated on host");

	Profiler &prof = getProfiler(PROF_GENCONF);

	string	momName ("MomConf");
	string	randName("Random");
	string	smthName("Smoother");

	switch (cType)
	{
		case CONF_NONE:
		break;

		case CONF_READ:
		readConf (&axionField, index);
		break;

		case CONF_TKACHEV: {
			auto &myPlan = AxionFFT::fetchPlan("Init");
			prof.start();
			momConf(axionField, kMax, kCrt);
			prof.stop();
			prof.add(momName, 14e-9*axionField->Size(), axionField->Size()*axionField->DataSize()*1e-9);
			myPlan.run(FFT_BCK);
			axionField->transferDev(FIELD_M);
			cudaMemcpy (axionField->vGpu(), static_cast<char *> (axionField->mGpu()) + axionField->DataSize()*axionField->Surf(), axionField->DataSize()*axionField->Size(), cudaMemcpyDeviceToDevice);
			scaleField (axionField, FIELD_M, *axionField->zV());
			axionField->transferCpu(FIELD_MV);
		}
		break;

		case CONF_KMAX: {
			auto &myPlan = AxionFFT::fetchPlan("Init");
			prof.start();
			momConf(axionField, kMax, kCrt);
			prof.stop();
			prof.add(momName, 14e-9*axionField->Size(), axionField->Size()*axionField->DataSize()*1e-9);
			myPlan.run(FFT_BCK);

			axionField->transferDev(FIELD_M);
			normaliseField(axionField, FIELD_M);
			cudaMemcpy (axionField->vGpu(), static_cast<char *> (axionField->mGpu()) + axionField->DataSize()*axionField->Surf(), axionField->DataSize()*axionField->Size(), cudaMemcpyDeviceToDevice);
			scaleField (axionField, FIELD_M, *axionField->zV());
			axionField->transferCpu(FIELD_MV);
		}
		break;

		case CONF_SMOOTH:
		prof.start();
		randConf (axionField);
		prof.stop();
		prof.add(randName, 0., axionField->Size()*axionField->DataSize()*1e-9);
		axionField->transferDev(FIELD_M);

		prof.start();
		smoothGpu (axionField, sIter, alpha);
		prof.stop();
		prof.add(smthName, 18.e-9*axionField->Size()*sIter, 8.e-9*axionField->Size()*axionField->DataSize()*sIter);

		if (smvarType != CONF_SAXNOISE)
			normaliseField(axionField, FIELD_M);

		cudaMemcpy (axionField->vGpu(), static_cast<char *> (axionField->mGpu()) + axionField->DataSize()*axionField->Surf(), axionField->DataSize()*axionField->Size(), cudaMemcpyDeviceToDevice);
		scaleField (axionField, FIELD_M, *axionField->zV());
		axionField->transferCpu(FIELD_MV);
		break;
	}

#else
	LogError ("Gpu support not built");
	exit(1);
#endif
}

using namespace std;
using namespace profiler;

void	ConfGenerator::runCpu	()
{
	Profiler &prof = getProfiler(PROF_GENCONF);

	string	momName ("MomConf");
	string	randName("Random");
	string	smthName("Smoother");

	switch (cType)
	{
		case CONF_NONE:
		break;

		case CONF_READ:
		readConf (&axionField, index);
		break;

		case CONF_TKACHEV: {
			auto &myPlan = AxionFFT::fetchPlan("Init");
			prof.start();
			momConf(axionField, kMax, kCrt);
			prof.stop();
			prof.add(momName, 14e-9*axionField->Size(), axionField->Size()*axionField->DataSize()*1e-9);
			myPlan.run(FFT_BCK);
		}
		break;

		case CONF_KMAX: {
			auto &myPlan = AxionFFT::fetchPlan("Init");
			prof.start();
			momConf(axionField, kMax, kCrt);
			prof.stop();
			prof.add(momName, 14e-9*axionField->Size(), axionField->Size()*axionField->DataSize()*1e-9);
			myPlan.run(FFT_BCK);
			normaliseField(axionField, FIELD_M);
			normCoreField	(axionField);
		}
		break;

		case CONF_SMOOTH:
		prof.start();
		randConf (axionField);
		prof.stop();
		prof.add(randName, 0., axionField->Size()*axionField->DataSize()*1e-9);
		prof.start();
		smoothXeon (axionField, sIter, alpha);
		prof.stop();
		prof.add(smthName, 18.e-9*axionField->Size()*sIter, 8.e-9*axionField->Size()*axionField->DataSize()*sIter);
		if (smvarType != CONF_SAXNOISE)
			normaliseField(axionField, FIELD_M);
		break;
	}

	if ((cType == CONF_KMAX) || (cType == CONF_SMOOTH) || (cType == CONF_TKACHEV))
	{
		memcpy (axionField->vCpu(), static_cast<char *> (axionField->mCpu()) + axionField->DataSize()*axionField->Surf(), axionField->DataSize()*axionField->Size());
		scaleField (axionField, FIELD_M, *axionField->zV());
	}

}

void	genConf	(Scalar *field, ConfType cType)
{
	LogMsg  (VERB_HIGH, "Called configurator generator");

	auto	cGen = std::make_unique<ConfGenerator> (field, cType);

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

		default:
			LogError ("Not a valid device");
			break;
	}

	return;
}

void	genConf	(Scalar *field, ConfType cType, size_t parm1, double parm2)
{
	LogMsg  (VERB_HIGH, "Called configurator generator");

	auto	cGen = std::make_unique<ConfGenerator> (field, cType, parm1, parm2);

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

		default:
			LogError ("Not a valid device");
			break;
	}

	return;
}
