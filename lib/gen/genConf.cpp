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

	Cosmos	*myCosmos;
	Scalar	*axionField;

	ConfType cType;

	size_t	kMax;
	size_t	sIter;

	double	kCrt;
	double	alpha;

	int	index;

	public:

		 ConfGenerator(Cosmos *myCosmos, Scalar *field, ConfType type);
		 ConfGenerator(Cosmos *myCosmos, Scalar *field, ConfType type, size_t parm1, double parm2);
		~ConfGenerator() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
};

	ConfGenerator::ConfGenerator(Cosmos *myCosmos, Scalar *field, ConfType type, size_t parm1, double parm2) : myCosmos(myCosmos), axionField(field), cType(type)
{
	switch (type)
	{
		case CONF_KMAX:
		case CONF_TKACHEV:
		case CONF_VILGORK:
		case CONF_VILGORS:

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

	ConfGenerator::ConfGenerator(Cosmos *myCosmos, Scalar *field, ConfType type) : myCosmos(myCosmos), axionField(field), cType(type)
{
	switch (type)
	{
		case CONF_KMAX:
		case CONF_TKACHEV:
		case CONF_VILGOR:

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
		default:
		case CONF_NONE:
		break;

		case CONF_READ:
		readConf (myCosmos, &axionField, index);
		break;

		case CONF_TKACHEV: {
			auto &myPlan = AxionFFT::fetchPlan("Init");
			prof.start();
			momConf(axionField, kMax, kCrt);
			prof.stop();
			prof.add(momName, 14e-9*axionField->Size(), axionField->Size()*axionField->DataSize()*1e-9);
			myPlan.run(FFT_BCK);
			axionField->transferDev(FIELD_M);
			if (!myCosmos->Mink()){
				cudaMemcpy (axionField->vGpu(), static_cast<char *> (axionField->mGpu()) + axionField->DataSize()*axionField->Surf(), axionField->DataSize()*axionField->Size(), cudaMemcpyDeviceToDevice);
				scaleField (axionField, FIELD_M, *axionField->zV());
			}
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

			if (!myCosmos->Mink()){
				// possible fix needed
				cudaMemcpy (axionField->vGpu(), static_cast<char *> (axionField->mGpu()) + axionField->DataSize()*axionField->Surf(), axionField->DataSize()*axionField->Size(), cudaMemcpyDeviceToDevice);
				scaleField (axionField, FIELD_M, *axionField->RV());
			}
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

		if (!myCosmos->Mink()){
		cudaMemcpy (axionField->vGpu(), static_cast<char *> (axionField->mGpu()) + axionField->DataSize()*axionField->Surf(), axionField->DataSize()*axionField->Size(), cudaMemcpyDeviceToDevice);
		scaleField (axionField, FIELD_M, *axionField->RV());
		}
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
		readConf (myCosmos, &axionField, index);
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
			LogMsg(VERB_NORMAL,"[GEN] CONF_KMAX started!\n ");
			auto &myPlan = AxionFFT::fetchPlan("Init");
			prof.start();
			// LogOut("[GEN] momConf with kMax %zu kCrit %f!\n ", kMax, kCrt);
			momConf(axionField, kMax, kCrt);
			prof.stop();
			prof.add(momName, 14e-9*axionField->Size(), axionField->Size()*axionField->DataSize()*1e-9);
			myPlan.run(FFT_BCK);
			normaliseField(axionField, FIELD_M);
			normCoreField	(axionField);
		}
		break;

		case CONF_VILGOR:
		case CONF_VILGORK:{
			LogMsg(VERB_NORMAL,"[GEN] CONF_VILGORk started!\n ");
			auto &myPlan = AxionFFT::fetchPlan("Init");

			// logi = log ms/H is taken to be zInit (which was input in command line)
			// depending on k, we use momentum or smooth initial conditions
			// temperarily we use momentum space by defaul, see below for smooth
			double logi = *axionField->zV();
			// such a logi and msa give a different initial time! redefine
			*axionField->zV() = (axionField->Delta())*exp(logi)/axionField->Msa();
			axionField->updateR();
			// LogOut("[GEN] time reset to z=%f to start with kappa(=logi)=%f",*axionField->zV(), logi);
			LogMsg(VERB_NORMAL,"[GEN] time reset to z=%f to start with kappa(=logi)=%f",*axionField->zV(), logi);

			double xit = (249.48 + 38.8431*logi + 1086.06* logi*logi)/(21775.3 + 3665.11*logi)  ;
			double nN3 = (6.0*xit*axionField->Msa()*axionField->Msa()*exp(-2.0*logi));
			double nc = sizeN*std::sqrt((nN3/4.7)*pow(1.-pow(nN3,1.5),-1./1.5));
			// LogOut("[GEN] estimated nN3 = %f -> n_critical = %f!",nN3,nc);
			LogMsg(VERB_NORMAL,"[GEN] estimated nN3 = %f -> n_critical = %f!",nN3,nc);

			prof.start();
			// LogOut("[GEN] momConf with kMax %d kCrit %f!\n ",sizeN,nc);
			momConf(axionField, sizeN, nc);
			prof.stop();
			prof.add(momName, 14e-9*axionField->Size(), axionField->Size()*axionField->DataSize()*1e-9);

			myPlan.run(FFT_BCK);

			normaliseField(axionField, FIELD_M);
			normCoreField	(axionField);

			if (!myCosmos->Mink()){
				//LogOut("rescalo!! con R %f",*axionField->RV());
			memcpy (axionField->vCpu(), static_cast<char *> (axionField->mStart()), axionField->DataSize()*axionField->Size());
			scaleField (axionField, FIELD_M, *axionField->RV());
			}
			// initPropagator (pType, axionField, (axionField->BckGnd().QcdPot() & VQCD_TYPE) | VQCD_EVOL_RHO);
			// tunePropagator (axiona);
			// if (int i ==0; i<10; i++ ){
			// 	dzaux = axion->dzSize(zInit);
			// 	propagate (axiona, dzaux);
			// }
		}
		break;

		case CONF_VILGORS:{
			LogMsg(VERB_NORMAL,"[GEN] CONF_VILGORs started!\n ");
			prof.start();
			randConf (axionField);
			prof.stop();
			prof.add(randName, 0., axionField->Size()*axionField->DataSize()*1e-9);

			// logi = log ms/H is taken to be zInit (which was input in command line)
			// depending on k, we use momentum or smooth initial conditions
			// in vilgors we use smooth
			// number of iterations needed = 0.8/(#/N^3)
			// desired 0.8/(#/N^3) is 6*xi(logi)*msa^2*exp(-2logi)
			double logi = *axionField->zV();
			// such a logi and msa give a different initial time! redefine
			*axionField->zV() = (axionField->Delta())*exp(logi)/axionField->Msa();
			axionField->updateR();
			LogMsg(VERB_NORMAL,"[GEN] time reset to z=%f to start with kappa(=logi)=%f",*axionField->zV(), logi);

			double xit = (249.48 + 38.8431*logi + 1086.06* logi*logi)/(21775.3 + 3665.11*logi)  ;
			double nN3 = (6.0*xit*axionField->Msa()*axionField->Msa()*exp(-2.0*logi));
			int niter = (int) (0.8/nN3);
			LogMsg(VERB_NORMAL,"[GEN] estimated nN3 = %f -> n_iterations = %d!",nN3,niter);

			LogMsg(VERB_NORMAL,"[GEN] smoothXeon called with %d iterations and alpha = %f!",niter,alpha);
			if (niter>100){
					LogMsg(VERB_NORMAL,"WARNING!! More than 100 iterations is not particularly efficient! update VILGOR algorithm to use FFTs!!\n");
			}
			prof.start();
			smoothXeon (axionField, niter, alpha);
			prof.stop();
			prof.add(smthName, 18.e-9*axionField->Size()*sIter, 8.e-9*axionField->Size()*axionField->DataSize()*sIter);

			normaliseField(axionField, FIELD_M);
			normCoreField	(axionField);

			if (!myCosmos->Mink()){
			memcpy (axionField->vCpu(), static_cast<char *> (axionField->mStart()), axionField->DataSize()*axionField->Size());
			scaleField (axionField, FIELD_M, *axionField->RV());
			}
			// initPropagator (pType, axionField, (axionField->BckGnd().QcdPot() & VQCD_TYPE) | VQCD_EVOL_RHO);
			// tunePropagator (axiona);
			// if (int i ==0; i<10; i++ ){
			// 	dzaux = axion->dzSize(zInit);
			// 	propagate (axiona, dzaux);
			// }
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
		normCoreField	(axionField);
		break;
	}

	axionField->setFolded(false);

	if ((cType == CONF_KMAX) || (cType == CONF_SMOOTH) || (cType == CONF_TKACHEV))
	{
		if (!myCosmos->Mink()){
		memcpy (axionField->vCpu(), static_cast<char *> (axionField->mCpu()) + axionField->DataSize()*axionField->Surf(), axionField->DataSize()*axionField->Size());
		scaleField (axionField, FIELD_M, *axionField->RV());
		}
	}

}

void	genConf	(Cosmos *myCosmos, Scalar *field, ConfType cType)
{
	LogMsg  (VERB_NORMAL, "Called configurator generator");

	auto	cGen = std::make_unique<ConfGenerator> (myCosmos, field, cType);

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

void	genConf	(Cosmos *myCosmos, Scalar *field, ConfType cType, size_t parm1, double parm2)
{
	LogMsg  (VERB_NORMAL, "Called configurator generator");

	auto	cGen = std::make_unique<ConfGenerator> (myCosmos, field, cType, parm1, parm2);

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
