#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <complex>

#include "enum-field.h"
#include "scalar/scalarField.h"
#include "scalar/scaleField.h"
#include "scalar/normField.h"
#include "scalar/normCore.h"
#include "scalar/theta2Cmplx.h"
#include "gen/momConf.h"
#include "gen/randXeon.h"
#include "gen/anystringXeon.h"
#include "gen/smoothXeon.h"
#include "gen/prepropa.h"
#include "io/readWrite.h"
#include "propagator/propXeon.h"
#include "scalar/folder.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	//#include "kernelParms.cuh"
	#include "gen/momGpu.h"
	#include "gen/randGpu.h"
	#include "gen/smoothGpu.h"
	#include "propagator/propGpu.h"
#endif

#include "utils/utils.h"
#include "fft/fftCode.h"
// #include "utils/pad.h"

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

		 ConfGenerator(Cosmos *myCosmos, Scalar *field);
		~ConfGenerator() {};

	void	 runCpu	();
	void	 runGpu	();
	void	 runXeon	();

	void   confsmooth(Cosmos *myCosmos, Scalar *field);
	void   conflola(Cosmos *myCosmos, Scalar *field);
	void   confcole(Cosmos *myCosmos, Scalar *field);
	void   confspax(Cosmos *myCosmos, Scalar *field);
	void   conftkac(Cosmos *myCosmos, Scalar *field);
	void   confthermal(Cosmos *myCosmos, Scalar *field);
	void	 confstring(Cosmos *myCosmos, Scalar *axionField);
	void	 confstring2(Cosmos *myCosmos, Scalar *axionField);
	// void   confapr(Cosmos *myCosmos, Scalar *field);

	void   putxi(double xit, bool kspace);
	double anymean(FieldIndex ftipo);

	void   susum(FieldIndex from, FieldIndex to);
	void   mulmul(FieldIndex from, FieldIndex to);
	void   axby(FieldIndex from, FieldIndex to, double a, double b);
};

ConfGenerator::ConfGenerator(Cosmos *myCosmos, Scalar *field) : myCosmos(myCosmos), axionField(field)
{
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

	IcData ic = myCosmos->ICData();
	cType = ic.cType;

	switch (cType)
	{
		default:
		case CONF_NONE:
		break;

		case CONF_READ:
		readConf (myCosmos, &axionField, index);
		break;

		case CONF_TKACHEV: {
			// TODO THIS WILL NOT WORK
			LogError("Configuration Tkachev in GPU not ready!");
			auto &myPlan = AxionFFT::fetchPlan("Init"); // now transposed
			prof.start();
			MomParms mopa;
			mopa.kMax = kMax;
			mopa.kCrt = kCrt;
			mopa.mocoty = MOM_MFLAT;
			mopa.cmplx = true;
			momConf(axionField, mopa);
			prof.stop();
			prof.add(momName, 14e-9*axionField->Size(), axionField->Size()*axionField->DataSize()*1e-9);
			myPlan.run(FFT_BCK);
			axionField->transferDev(FIELD_M);
/*			FIXME See below
			if (!myCosmos->Mink()){
				cudaMemcpy (axionField->vGpu(), static_cast<char *> (axionField->mGpu()) + axionField->DataSize()*axionField->Surf(), axionField->DataSize()*axionField->Size(), cudaMemcpyDeviceToDevice);
				scaleField (axionField, FIELD_M, *axionField->zV());
			}
			axionField->transferCpu(FIELD_MV);
*/
		}
		break;

		case CONF_KMAX: {
			LogError("Configuration KMAX in GPU not ready! check FFT plans");
			auto &myPlan = AxionFFT::fetchPlan("Init"); // now transposed
			prof.start();
			MomParms mopa;
				mopa.kMax = kMax;
				mopa.kCrt = kCrt;
				mopa.mocoty = MOM_MEXP2;
				mopa.cmplx = true;
			momConf(axionField, mopa);
			prof.stop();
			prof.add(momName, 14e-9*axionField->Size(), axionField->Size()*axionField->DataSize()*1e-9);
			myPlan.run(FFT_BCK);

			axionField->transferDev(FIELD_M);
			normaliseField(axionField, FIELD_M);

/*			FIXME See below
			if (!myCosmos->Mink()){
				// possible fix needed
				cudaMemcpy (axionField->vGpu(), static_cast<char *> (axionField->mGpu()) + axionField->DataSize()*axionField->Surf(), axionField->DataSize()*axionField->Size(), cudaMemcpyDeviceToDevice);
				scaleField (axionField, FIELD_M, *axionField->RV());
			}
			axionField->transferCpu(FIELD_MV);
*/
		}
		break;

		case CONF_SMOOTH:
		prof.start();
		ic.fieldindex = FIELD_M;
		randConf (axionField,ic);
		prof.stop();
		prof.add(randName, 0., axionField->Size()*axionField->DataSize()*1e-9);
		axionField->transferDev(FIELD_M);

		prof.start();
		smoothGpu (axionField, sIter, alpha);
		prof.stop();
		prof.add(smthName, 18.e-9*axionField->Size()*sIter, 8.e-9*axionField->Size()*axionField->DataSize()*sIter);

		if (smvarType != CONF_SAXNOISE)
			normaliseField(axionField, FIELD_M);
		break;
	}


	if ((cType != CONF_READ) && (cType != CONF_NONE)) {

		if (!myCosmos->Mink()) {
			PropParms ppar;
			ppar.R      = *axionField->RV();
			ppar.ood2a  = 1./(axionField->Delta()*axionField->Delta());;
			ppar.lambda = axionField->LambdaP();
			uint	  S    = axionField->Surf();
			uint	  V    = axionField->Size();
			uint	  Vo   = S;
			uint	  Vf   = V+S;
			cudaMemcpy(axionField->vGpu(), static_cast<char *> (axionField->mGpu()) + axionField->DataSize()*axionField->Surf(), axionField->DataSize()*axionField->Size(), cudaMemcpyDeviceToDevice);
			scaleField(axionField, FIELD_M, *axionField->RV());
			axionField->exchangeGhosts(FIELD_M);
			updateVGpu(axionField->mGpu(), axionField->vGpu(), ppar, ppar.R, 1.0, Vo, Vf,
				   axionField->BckGnd()->QcdPot() & V_TYPE, axionField->Precision(), 512, 1, 1, ((cudaStream_t *)axionField->Streams())[2]);
					// FIXME --> xDefaultBlockGpu, yDefaultBlockGpu, zDefaultBlockGpu, ((cudaStream_t *)axionField->Streams())[2]);
		}

		axionField->transferCpu(FIELD_MV);
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

	/* Ic conditions were fed into myCosmos and the axion field */
	IcData ic = myCosmos->ICData();
	cType = ic.cType;

	LogMsg(VERB_NORMAL,"\n");
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.icdrule  %d",ic.icdrule  );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.preprop  %d",ic.preprop  );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.icstudy  %d",ic.icstudy  );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.prepstL  %f",ic.prepstL  );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.prepcoe  %f",ic.prepcoe  );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.pregammo %f",ic.pregammo );
	LogMsg(VERB_NORMAL,"[sca] myCosmos.ic.prelZ2e  %f",ic.prelZ2e  );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.prevtype %d",ic.prevtype );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.normcore %d",ic.normcore );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.alpha    %f",ic.alpha    );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.siter    %d",ic.siter    );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.kcr      %f",ic.kcr      );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.kMax     %d",ic.kMax     );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.mode0    %f",ic.mode0    );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.kickalpha%f",ic.kickalpha);
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.zi       %f",ic.zi       );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.logi     %f",ic.logi     );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.cType    %d",ic.cType    );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.smvarTy  %d",ic.smvarType);
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.mocoty   %d",ic.mocoty   );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.randommom   %d",ic.randommom   );
	LogMsg(VERB_NORMAL, "[gen] cType is %d",cType);

	switch (cType)
	{
		case CONF_NONE:
		break;

		case CONF_READ:
		readConf (myCosmos, &axionField, index);
		break;

		case CONF_LOLA:
			conflola(myCosmos,axionField);
		break;

		case CONF_COLE:
			confcole(myCosmos,axionField);
		break;

		case CONF_SPAX:
			confspax(myCosmos,axionField);
		break;

		case CONF_TKACHEV:
			conftkac(myCosmos,axionField);
		break;

		case CONF_THERMAL:
			confthermal(myCosmos,axionField);
		break;

		case CONF_STRING:
			confstring2(myCosmos,axionField);
		break;

		case CONF_SMOOTH:
			confsmooth(myCosmos,axionField);
		break;

		// case CONF_APR:
		// 	confapr(myCosmos,axionField);
		// break;

		case CONF_KMAX: {
			LogMsg(VERB_NORMAL,"[GEN] CONF_KMAX started!\n ");
			LogFlush();
			AxionFFT::initPlan (axionField, FFT_CtoC_MtoM,  FFT_FWDBCK, "InitKMAX");
			auto &myPlan = AxionFFT::fetchPlan("InitKMAX");
			prof.start();

			// LogOut("[GEN] momConf with kMax %zu kCrit %f!\n ", kMax, kCrt);

			MomParms mopa;
				mopa.kMax = ic.kMax;
				mopa.kCrt = ic.kcr;
				mopa.mocoty = ic.mocoty;
				mopa.cmplx = true;
				mopa.mp = axionField->mStart();
			momConf(axionField, mopa);

			LogFlush();
			prof.stop();
			prof.add(momName, 14e-9*axionField->Size(), axionField->Size()*axionField->DataSize()*1e-9);
			myPlan.run(FFT_BCK);
			normaliseField(axionField, FIELD_M);
			if (myCosmos->ICData().normcore)
				normCoreField	(axionField);
		}
		axionField->setFolded(false);
		break;

		case CONF_VILGOR:{
			LogMsg(VERB_NORMAL,"\n ");
			LogMsg(VERB_NORMAL,"[GEN] CONF_VILGOR started! ");
			AxionFFT::initPlan (axionField, FFT_CtoC_MtoM, FFT_FWDBCK, "InitVilgor");
			auto &myPlan = AxionFFT::fetchPlan("InitVilgor");  // now transposed

			double LALA = axionField->BckGnd()->Lambda();

			LogMsg(VERB_NORMAL,"[GEN] Current Msa() = %f",axionField->Msa());

			// logi = log ms/H is taken to be zInit (which was input in command line)
			LogMsg(VERB_NORMAL,"[GEN] zV %f zi %f logi %f",*axionField->zV(), ic.zi, ic.logi);

			double xit;
			if (axionField->LambdaT() == LAMBDA_Z2) // Strictly valid only for LamZ2e = 2.0
				xit = (249.48 + 38.8431*ic.logi + 1086.06* ic.logi*ic.logi)/(21775.3 + 3665.11*ic.logi)  ;
			else // We use this as a nice approximation (for low logi)
				xit = (249.48 + 38.8431*ic.logi + 1086.06* ic.logi*ic.logi)/(21775.3 + 3665.11*ic.logi)  ;
				// (9.31021 + 1.38292e-6*logi + 0.713821*logi*logi)/(42.8748 + 0.788167*logi);

			/*This is the expression that follows from the definition of xi*/
			double nN3 = (6.0*xit*axionField->Delta()*axionField->Delta()/ic.zi/ic.zi);
			double nc = sizeN*std::sqrt((nN3/4.7)*pow(1.-pow(nN3,1.5),-1./1.5));
			// LogOut("[GEN] estimated nN3 = %f -> n_critical = %f!",nN3,nc);
			LogMsg(VERB_NORMAL,"[GEN] xit(logi)= %f estimated nN3 = %f -> n_critical = %f!",xit, nN3, nc);

			LogMsg(VERB_NORMAL,"[GEN] sIter %d!",ic.siter);

			if (ic.siter == 1) {
				nN3 = min(ic.kcr*nN3,1.0);
				nc = sizeN*std::sqrt((nN3/4.7)*pow(1.-pow(nN3,1.5),-1./1.5));
				// LogOut("[GEN] estimated nN3 = %f -> n_critical = %f!",nN3,nc);
				LogMsg(VERB_NORMAL,"[GEN] Input kcr %f > modifies to nN3 = %f -> n_critical = %f!",ic.kcr, nN3,nc);
			} else if (ic.siter == 2) {
			// add random noise in the initial time ~ random in xi (not really)
				double r = 0 ;
				int myRank = commRank();

				if (myRank == 0) {
					std::random_device rd;
					std::mt19937 mt(rd());
					std::uniform_real_distribution<double> dist(-1.0, 1.0);
					// srand (static_cast <unsigned> (time(0)));
					r = dist(mt);
				}

				MPI_Bcast (&r, sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);
				commSync();
				// printf("hello from rank %d, r = %f\n",myRank,r);

				nN3 = min(pow(ic.kcr,r)*nN3,1.0); ;
				nc = sizeN*std::sqrt((nN3/4.7)*pow(1.-pow(nN3,1.5),-1./1.5));
				LogMsg(VERB_NORMAL,"[GEN] random,kCrit %f,%f,%f > modifies to nN3 = %f -> n_critical = %f!",r, ic.kcr,pow(ic.kcr,r), nN3,nc);
			}

			LogMsg(VERB_NORMAL,"[GEN] momConf with kMax %d kcr %f! momConf %d\n ",sizeN,nc,ic.mocoty);
			prof.start();
			MomParms mopa;
				mopa.kMax = sizeN;
				mopa.kCrt = nc;
				mopa.mocoty = ic.mocoty;
				mopa.cmplx = true;
				mopa.mp = axionField->mStart();
			momConf(axionField, mopa);

			prof.stop();
			prof.add(momName, 14e-9*axionField->Size(), axionField->Size()*axionField->DataSize()*1e-9);
			axionField->setFolded(false);

			myPlan.run(FFT_BCK);

			normaliseField(axionField, FIELD_M);

			if (!myCosmos->Mink() && !ic.preprop) {
				if (myCosmos->ICData().normcore)
					normCoreField	(axionField);
				memcpy	   (axionField->vCpu(), static_cast<char *> (axionField->mStart()), axionField->DataSize()*axionField->Size());
				if ( !(ic.kickalpha == 0.0) )
					scaleField (axionField, FIELD_V, 1.0+ic.kickalpha);
				scaleField (axionField, FIELD_M, *axionField->RV());
			}

			if (ic.preprop) {
				/* Go back in time the preprop factor*/
				*axionField->zV() /= ic.prepcoe; // now z <zi
				axionField->updateR();
				LogMsg(VERB_NORMAL,"[GEN] prepropagator, jumped back in time to %f",*axionField->zV());

				/* Although this is weird it is included in the prepropagator */
				// if (myCosmos->ICData().normcore)
				// 	normCoreField	(axionField);
				// memcpy	   (axionField->vCpu(), static_cast<char *> (axionField->mStart()), axionField->DataSize()*axionField->Size());
				// scaleField (axionField, FIELD_M, *axionField->RV());

				prepropa2  (axionField);
			}

		}
		break;

		case CONF_VILGORK: {
			LogMsg(VERB_NORMAL,"[GEN] CONF_VILGORk started! ");
			AxionFFT::initPlan (axionField, FFT_CtoC_MtoM, FFT_FWDBCK, "InitVilgorK");
			auto &myPlan = AxionFFT::fetchPlan("InitVilgorK");

			double LALA = axionField->BckGnd()->Lambda();
			if (preprop) {
					axionField->BckGnd()->SetLambda(LALA*prepcoe*prepcoe);
					LogOut("[GEN] Mira qe cambio LL %f -> %f\n",LALA,axionField->BckGnd()->Lambda());
			}

			double msafromLL = sqrt(2*axionField->BckGnd()->Lambda())*axionField->Delta();
			LogMsg(VERB_NORMAL,"[GEN] msa %f and msa = %f",msafromLL,axionField->Msa());

			// logi = log ms/H is taken to be zInit (which was input in command line)
			LogMsg(VERB_NORMAL,"[GEN] zV %f zInit %f ",*axionField->zV(), zInit);
			double logi = *axionField->zV();

			// such a logi and msa give a different initial time! redefine
			*axionField->zV() = (axionField->Delta())*exp(logi)/msafromLL;
			axionField->updateR();
			LogMsg(VERB_NORMAL,"[GEN] time reset to z=%f to start with kappa(=logi)=%f",*axionField->zV(), logi);

			double xit = (249.48 + 38.8431*logi + 1086.06* logi*logi)/(21775.3 + 3665.11*logi)  ;

			/* if prepropagator this number increases */
			double nN3 = (6.0*xit*msafromLL*msafromLL*exp(-2.0*logi));
			double nc = sizeN*std::sqrt((nN3/4.7)*pow(1.-pow(nN3,1.5),-1./1.5));
			// LogOut("[GEN] estimated nN3 = %f -> n_critical = %f!",nN3,nc);
			LogMsg(VERB_NORMAL,"[GEN] xit(logi)= %f estimated nN3 = %f -> n_critical = %f!",xit, nN3, nc);

			LogMsg(VERB_NORMAL,"[GEN] sIter %d!",ic.siter);

			if (ic.siter == 1) {
				nN3 = min(ic.kcr*nN3,1.0);
				nc = sizeN*std::sqrt((nN3/4.7)*pow(1.-pow(nN3,1.5),-1./1.5));
				// LogOut("[GEN] estimated nN3 = %f -> n_critical = %f!",nN3,nc);
				LogMsg(VERB_NORMAL,"[GEN] kCrit %f > modifies to nN3 = %f -> n_critical = %f!",ic.kcr, nN3,nc);
			} else if (ic.siter > 1) {
			// add random noise in the initial time ~ random in xi (not really)
				double r = 0 ;
				int myRank = commRank();

				if (myRank == 0) {
					std::random_device rd;
					std::mt19937 mt(rd());
					std::uniform_real_distribution<double> dist(-1.0, 1.0);
					// srand (static_cast <unsigned> (time(0)));
					r = dist(mt);
				}

				MPI_Bcast (&r, sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);
				commSync();
				// printf("hello from rank %d, r = %f\n",myRank,r);

				nN3 = min(pow(kCrit,r)*nN3,1.0); ;
				nc = sizeN*std::sqrt((nN3/4.7)*pow(1.-pow(nN3,1.5),-1./1.5));
				LogMsg(VERB_NORMAL,"[GEN] random,kCrit %f,%f,%f > modifies to nN3 = %f -> n_critical = %f!",r, kCrit,pow(kCrit,r), nN3,nc);
			}

			LogMsg(VERB_NORMAL,"[GEN] momConf with kMax %d kCrit %f!\n ",sizeN,nc);
			prof.start();
			MomParms mopa;
				mopa.kMax = sizeN;
				mopa.kCrt = nc;
				mopa.mocoty = MOM_MEXP2;
				mopa.cmplx = true;
				mopa.mp = axionField->mStart();
			momConf(axionField, mopa);

			prof.stop();
			prof.add(momName, 14e-9*axionField->Size(), axionField->Size()*axionField->DataSize()*1e-9);
			axionField->setFolded(false);

			myPlan.run(FFT_BCK);

			normaliseField(axionField, FIELD_M);
			if (myCosmos->ICData().normcore)
				normCoreField	(axionField);

			if (!myCosmos->Mink()) {

				double	   lTmp = axionField->BckGnd()->Lambda()/((*axionField->RV()) * (*axionField->RV()));
				double	   ood2 = 1./(axionField->Delta()*axionField->Delta());
				auto	   S  = axionField->Surf();
				auto	   V  = axionField->Size();
				auto	   Vo = S;
				auto	   Vf = V+S;
				double   hzi = *axionField->zV()/2.0;

				memcpy	   (axionField->vCpu(), static_cast<char *> (axionField->mStart()), axionField->DataSize()*axionField->Size());
				scaleField (axionField, FIELD_M, *axionField->RV());
			}

			if (preprop) {
				if (pregammo == 0.0) {
					prepropa  (axionField);
					axionField->BckGnd()->SetLambda(LALA);
					double zsave = *axionField->zV();
					double rsave = 1/(*axionField->RV());

					/* travel back in time to have the originally hoped for value of logi */
					*axionField->zV() = (axionField->Delta())*exp(logi)/axionField->Msa();
					axionField->updateR();

					double ska = *axionField->RV()*rsave;
					size_t vol = axionField->Size();

					if (axionField->Precision() == FIELD_DOUBLE) {
						rsave *= (1-ska);
						std::complex<double> *mi = static_cast<std::complex<double> *>(axionField->mStart());
						std::complex<double> *vi = static_cast<std::complex<double> *>(axionField->vCpu());

						#pragma omp parallel for schedule(static)
						for (size_t idx=0; idx < vol; idx++)
							vi[idx] = vi[idx]*ska + mi[idx]*rsave;

						scaleField (axionField, FIELD_M, ska);
					} else {
						float skaf = (float) ska;
						float rsavef = rsave*(1-skaf);
						std::complex<float> *mi = static_cast<std::complex<float> *>(axionField->mStart());
						std::complex<float> *vi = static_cast<std::complex<float> *>(axionField->vCpu());

						#pragma omp parallel for schedule(static)
						for (size_t idx=0; idx < vol; idx++)
							vi[idx] = vi[idx]*skaf + mi[idx]*rsavef;

						scaleField (axionField, FIELD_M, skaf);
					}
				} // end damping-less cases
				else if (pregammo > 0.0)
				{
					axionField->BckGnd()->SetLambda(LALA);
					relaxrho(axionField);
				}
			}

		}

		break;

		case CONF_VILGORS: {
			LogMsg(VERB_NORMAL,"[GEN] CONF_VILGORs started!\n ");
			prof.start();
			ic.fieldindex = FIELD_M;
			randConf (axionField,ic);
			prof.stop();
			prof.add(randName, 0., axionField->Size()*axionField->DataSize()*1e-9);

			// logi = log ms/H is taken to be zInit (which was input in command line)
			// number of iterations needed = 0.8/(#/N^3)
			// desired 0.8/(#/N^3) is 6*xi(logi)*msa^2*exp(-2logi)
			double logi = std::log(std::sqrt(axionField->SaxionMassSq())*(*axionField->RV())*(*axionField->RV()));
			//*axionField->zV() = (axionField->Delta())*exp(logi)/axionField->Msa();
			//axionField->updateR();
			LogMsg(VERB_NORMAL,"[GEN] time reset to z=%f to start with kappa(=logi)=%f",*axionField->zV(), logi);

			double xit = (249.48 + 38.8431*logi + 1086.06* logi*logi)/(21775.3 + 3665.11*logi)  ;
			double nN3 = (6.0*xit*axionField->Msa()*axionField->Msa()*exp(-2.0*logi));
			int niter = (int) (0.8/nN3);
			LogMsg(VERB_NORMAL,"[GEN] estimated nN3 = %f -> n_iterations = %d!",nN3,niter);

			if (ic.siter == 1) {
				nN3 = min(kCrit*nN3,1.0);
				int niter = (int) (0.8/nN3);
				// LogOut("[GEN] estimated nN3 = %f -> n_critical = %f!",nN3,nc);
			LogMsg(VERB_NORMAL,"[GEN] kCrit %f > modifies to nN3 = %f -> n_iterations = %d!",kCrit, nN3,niter);
		} else if (ic.siter > 1) {
				// add random noise in the initial time ~ random in xi (not really)
				double r = 0 ;
				int  myRank   = commRank();
				if (myRank == 0) {
					std::random_device rd;
					std::mt19937 mt(rd());
					std::uniform_real_distribution<double> dist(-1.0, 1.0);
					// srand (static_cast <unsigned> (time(0)));
					r = dist(mt);
				}
				MPI_Bcast (&r, sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);
				commSync();
				// printf("hello from rank %d, r = %f\n",myRank,r);

				nN3 = min(pow(kCrit,r)*nN3,1.0); ;
				int niter = (int) (0.8/nN3);
				LogMsg(VERB_NORMAL,"[GEN] random,kCrit %f,%f,%f > modifies to nN3 = %f -> n_iterations = %d!",r, kCrit,pow(kCrit,r), nN3,niter);
			}

			LogMsg(VERB_NORMAL,"[GEN] smoothXeon called with %d iterations and alpha = %f!",niter,alpha);
			if (niter>100) {
					LogMsg(VERB_NORMAL,"WARNING!! More than 100 iterations is not particularly efficient! update VILGOR algorithm to use FFTs!!\n");
			}


			prof.start();
			smoothXeon (axionField, niter, alpha);
			prof.stop();
			prof.add(smthName, 18.e-9*axionField->Size()*ic.siter, 8.e-9*axionField->Size()*axionField->DataSize()*ic.siter);

			normaliseField(axionField, FIELD_M);
			if (myCosmos->ICData().normcore)
				normCoreField	(axionField);
/*			FIXME	See below

			if (!myCosmos->Mink()){
			memcpy (axionField->vCpu(), static_cast<char *> (axionField->mStart()), axionField->DataSize()*axionField->Size());
			scaleField (axionField, FIELD_M, *axionField->RV());
			}
*/
			// initPropagator (pType, axionField, (axionField->BckGnd().QcdPot() & V_TYPE) | V_EVOL_RHO);
			// tunePropagator (axiona);
			// if (int i ==0; i<10; i++ ){
			// 	dzaux = axion->dzSize(zInit);
			// 	propagate (axiona, dzaux);
			// }
		}
		axionField->setFolded(false);
		break;




	}


	if ( (cType == CONF_KMAX) || (cType == CONF_VILGORS) ) {
		LogMsg(VERB_NORMAL,"[GEN] final rescaling for KMAX/VILGORS!");
		if (!myCosmos->Mink()) {
			double	   lTmp = axionField->BckGnd()->Lambda()/((*axionField->RV()) * (*axionField->RV()));
			double	   ood2 = 1./(axionField->Delta()*axionField->Delta());
			memcpy     (axionField->vCpu(), static_cast<char *> (axionField->mStart()), axionField->DataSize()*axionField->Size());
			scaleField (axionField, FIELD_M, *axionField->RV());

			auto	   S  = axionField->Surf();
			auto	   V  = axionField->Size();
			auto	   Vo = S;
			auto	   Vf = V+S;
			double   hzi = *axionField->zV()/2.;

		}
	}
	LogMsg(VERB_NORMAL,"[GEN] done!");
}










/* Collection functions */

void	ConfGenerator::confsmooth(Cosmos *myCosmos, Scalar *axionField)
{
	Profiler &prof = getProfiler(PROF_GENCONF);
	string	randName("Random");
	string	smthName("Smoother");

	IcData ic = myCosmos->ICData();

	LogMsg(VERB_NORMAL,"\n ");
	LogMsg(VERB_NORMAL,"[GEN] CONF_SMOOTH started! ");

	double R  = *axionField->RV();
	double Rp = myCosmos->Rp(*axionField->zV());
	LogMsg(VERB_NORMAL,"[GEN] time %f scale factor R %f Rp %f ",*axionField->zV(),*axionField->RV(),Rp);

	/* Field m */
	prof.start();

	ic.fieldindex = FIELD_M;
	randConf (axionField,ic);

	/*exp new piece FIXME */
	// if (ic.smvarType == CONF_RAND)
	// if (ic.mode0 != 0.0){
	// 	ic.fieldindex = FIELD_V;
	// 	ic.smvarType = CONF_VELRAND;
	// 	randConf (axionField,ic);
	// 	mulmul(FIELD_M,FIELD_V);
	// 	ic.fieldindex = FIELD_M;
	// 	ic.smvarType = CONF_RAND;
	// }

	/* Field velocity */
	if (ic.smvarType == CONF_STRWAVE){
		ic.fieldindex = FIELD_V;
		ic.smvarType = CONF_THETAVEL;
		ic.kcr = 1.;
		randConf (axionField,ic);
		mulmul(FIELD_M,FIELD_V);
		ic.smvarType = CONF_STRWAVE;
	}

	prof.stop();
	prof.add(randName, 0., axionField->Size()*axionField->DataSize()*1e-9);

	if (ic.siter>0)
	{
		prof.start();
		smoothXeon (axionField, ic.siter, ic.alpha);
		prof.stop();
		prof.add(smthName, 18.e-9*axionField->Size()*ic.siter, 8.e-9*axionField->Size()*axionField->DataSize()*ic.siter);
	}

	if (!ic.preprop)
	{
		if ((ic.smvarType != CONF_SAXNOISE) && (ic.smvarType != CONF_PARRES))
			normaliseField(axionField, FIELD_M);

		if (myCosmos->ICData().normcore)
			normCoreField	(axionField);

		if (!myCosmos->Mink()) /* In Minkowski this is trivial */
			axby(FIELD_M,FIELD_V,Rp*R,R);

		if ( !(ic.kickalpha == 0.0) )
			scaleField (axionField, FIELD_V, 1.0+ic.kickalpha);

		if (!myCosmos->Mink()) /* In Minkowski this is trivial */
			scaleField (axionField, FIELD_M, *axionField->RV());
	}

	/*Note that prepropagation folds the field ;-)*/
	axionField->setFolded(false);


	if (ic.preprop) {
		/* Go back in time the preprop factor*/
		*axionField->zV() /= ic.prepcoe; // now z <zi
		axionField->updateR();
		LogMsg(VERB_NORMAL,"[GEN] prepropagator, jumped back in time to %f",*axionField->zV());
		prepropa2  (axionField);
	}
}





void	ConfGenerator::conflola(Cosmos *myCosmos, Scalar *axionField)
{

	IcData ic = myCosmos->ICData();

	LogMsg(VERB_NORMAL,"\n ");
	LogMsg(VERB_NORMAL,"[GEN] CONF_LOLA started! ");

	double LALA = axionField->BckGnd()->Lambda();

	LogMsg(VERB_NORMAL,"[GEN] Current Msa() = %f",axionField->Msa());
	LogMsg(VERB_NORMAL,"[GEN] zV %f zi %f logi %f",*axionField->zV(), ic.zi, ic.logi);

	double xit;
	double logit;
	if (ic.preprop) {
		double Rnow = axionField->Rfromct(ic.zi);
		double Rpre = axionField->Rfromct(ic.zi/ic.prepcoe);
		double prelz2e = ic.prelZ2e;
		double llp;
		if (axionField->LambdaT() == LAMBDA_FIXED) {
			llp = LALA;
		} else {
			llp = LALA/pow(Rnow,axionField->BckGnd()->LamZ2Exp());
		}
		double preLambdaP = llp/pow(Rpre/Rnow,prelz2e);
		logit = log(sqrt(2*preLambdaP)*Rpre*Rpre);
		LogMsg(VERB_NORMAL,"[GEN] logi jumped from %f to %f for prepropagator",ic.logi,logit);
		LogMsg(VERB_NORMAL,"[GEN] Ri %f Rpi %f llp %f LambdaP %f prelZ2e %f preLambdaP %f",Rnow,Rpre,llp,axionField->LambdaP(),prelz2e,preLambdaP);
	} else {
		logit = ic.logi;
	}
	if (axionField->LambdaT() == LAMBDA_Z2) // Strictly valid only for LamZ2e = 2.0
	  xit = (249.48 + 38.8431*logit + 1086.06*logit*logit)/(21775.3 + 3665.11*logit)  ;
	else // We use this as a nice approximation (for low logi)
	  xit = (249.48 + 38.8431*logit + 1086.06*logit*logit)/(21775.3 + 3665.11*logit)  ;
	  // (9.31021 + 1.38292e-6*logi + 0.713821*logi*logi)/(42.8748 + 0.788167*logi);
	if (ic.preprop) xit *= ic.prepcoe*ic.prepcoe;

	if (ic.siter == 1) {
		xit *= ic.kcr;
	} else if (ic.siter == 2) {
	// add random noise in the initial time ~ random in xi (not really)
	  double r = 0 ; int myRank = commRank();

	  if (myRank == 0) {
			std::random_device rd;
			std::mt19937 mt(rd());
			std::uniform_real_distribution<double> dist(-1.0, 1.0);
	    r = dist(mt);}

	  MPI_Bcast (&r, sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);
	  commSync();
	  LogMsg(VERB_NORMAL,"[GEN] random %f,%f -> %f",r, ic.kcr, pow(ic.kcr,r));
	  xit *= pow(ic.kcr,r);
	}

	/* Creates the configuration with correct value of xit*/
	LogMsg(VERB_NORMAL,"[GEN] call putxi mode %d",ic.kMax);

	putxi(xit,ic.kMax);

	normaliseField(axionField, FIELD_M);

	if (!myCosmos->Mink() && !ic.preprop) {
	  if (myCosmos->ICData().normcore)
	    normCoreField	(axionField);
	  memcpy	   (axionField->vCpu(), static_cast<char *> (axionField->mStart()), axionField->DataSize()*axionField->Size());
	  if ( !(ic.kickalpha == 0.0) )
	    scaleField (axionField, FIELD_V, 1.0+ic.kickalpha);
	  scaleField (axionField, FIELD_M, *axionField->RV());
	}

	if (ic.preprop) {
	  /* Go back in time the preprop factor*/
	  *axionField->zV() /= ic.prepcoe; // now z <zi
	  axionField->updateR();
	  LogMsg(VERB_NORMAL,"[GEN] prepropagator, jumped back in time to %f",*axionField->zV());

	  /* Although this is weird it is included in the prepropagator */
	  // if (myCosmos->ICData().normcore)
	  // 	normCoreField	(axionField);
	  // memcpy	   (axionField->vCpu(), static_cast<char *> (axionField->mStart()), axionField->DataSize()*axionField->Size());
	  // scaleField (axionField, FIELD_M, *axionField->RV());

	  prepropa2  (axionField);
	}
} // endconflola





void	ConfGenerator::confcole(Cosmos *myCosmos, Scalar *axionField)
{
	LogMsg(VERB_NORMAL,"\n ");
	LogMsg(VERB_NORMAL,"[GEN] CONF_COLE started! ");

	size_t VD = (axionField->DataSize())*(axionField->Size());

	/* Configuration is calculated with 1/(1+k2) and exp cut times correction
	kl = (2pi n /L)* l
	l in natural units is ct
	l modulated with kcr
		kl = n/ncrit -> ncrit = L/ [2pi(ct kcr)]
	exponential with ???
	*/

	if (axionField->LowMem()){
		LogError("[GEN] Problems in Lowmem! ");
		return;
	}

	AxionFFT::initPlan(axionField, FFT_CtoC_M2toM2, FFT_FWDBCK, "C2CM22M2");
	auto &myPlan = AxionFFT::fetchPlan("C2CM22M2");
	IcData ic = myCosmos->ICData();

	MomParms mopa;
		mopa.kMax = axionField->Length();
		mopa.kCrt = axionField->BckGnd()->PhysSize()/(6.283185307179586*(*axionField->zV())*ic.kcr);
		mopa.mocoty = MOM_COLE;
		mopa.cmplx = true;
	momConf(axionField, mopa);
	myPlan.run(FFT_BCK);

	char *mm = static_cast<char *>(axionField->mStart());
	char *vv = static_cast<char *>(axionField->vCpu());
	char *m2 = static_cast<char *>(axionField->m2Cpu());
	memmove	(mm, m2, VD);

	/* Instead of
	normaliseField(axionField, FIELD_M);
	*/
	double rhomean = anymean(FIELD_M);
	LogMsg(VERB_NORMAL,"[GEN] rhomean %.2f",rhomean);
	scaleField (axionField, FIELD_M, 1./rhomean);

	if (ic.normcore)
		normCoreField	(axionField);
	memcpy (vv, mm, VD);
	if ( !(ic.kickalpha == 0.0) )
		scaleField (axionField, FIELD_V, 1.0+ic.kickalpha);
	scaleField (axionField, FIELD_M, *axionField->RV());

	if (ic.extrav != 0.0){
		// extra noise in v!
			mopa.kMax = axionField->Length();
			mopa.kCrt = axionField->BckGnd()->PhysSize()/(6.283185307179586*(*axionField->zV())*ic.kcr);
			mopa.mocoty = MOM_KCOLE;
		momConf(axionField, mopa);
		myPlan.run(FFT_BCK);

		rhomean = anymean(FIELD_M2);
		LogMsg(VERB_NORMAL,"[GEN] M2 mean %.2f",rhomean);
		scaleField (axionField, FIELD_M2, ic.extrav/rhomean);
		susum	(FIELD_M2, FIELD_V);
	}

	LogMsg(VERB_NORMAL,"[GEN] CONF_COLE end! \n");
} // endconf cole




void	ConfGenerator::confspax(Cosmos *myCosmos, Scalar *axionField)
{
	Profiler &prof = getProfiler(PROF_GENCONF);
	LogMsg(VERB_NORMAL,"\n ");
	LogMsg(VERB_NORMAL,"[GEN] CONF_SPAX started! ");
	IcData ic = myCosmos->ICData();
	LogFlush();

	size_t VD;
	if (myCosmos->ICData().fType == FIELD_AXION)
	{
		LogMsg(VERB_NORMAL,"[GEN] set field to axion! ");
		axionField->setField(FIELD_AXION);
		VD = (axionField->DataSize())*(axionField->Size());
	} else if (myCosmos->ICData().fType == FIELD_SAXION)
	{
		LogMsg(VERB_NORMAL,"[GEN] SPAX with saxion ... works? ");
	}

	LogMsg(VERB_NORMAL,"[GEN] Read data file! ");
	LogFlush();

	std::vector<double> mm,vv;
	{
			FILE *cacheFile = nullptr;
			if (((cacheFile  = fopen("./initialspectrum.dat", "r")) == nullptr)){
				printf("No initialspectrum.dat ! Exit!");
				exit(1);
			}
			else
			{					int ii = 0;
								double ma, va;
								while(!feof(cacheFile)){
										fscanf (cacheFile ,"%lf %lf", &ma, &va);
										LogMsg(VERB_PARANOID," m %.3e v %.3e !",ma,va);
										// printf(" m %.3e v %.3e ! ",ma,va);
										mm.push_back(ma);
										vv.push_back(va);
										ii++;
								}
			}
	}
	/* Generate axion in momentum space */
	prof.start();
	LogMsg(VERB_NORMAL,"[GEN] Create axion field! ");
	MomParms mopa;
		mopa.kMax = axionField->Length();
		// mopa.kMax = ic.kMax;
		mopa.kCrt = axionField->BckGnd()->PhysSize()/(6.283185307179586*(*axionField->zV())*ic.kcr);
		mopa.mocoty = MOM_SPAX;
		mopa.mfttab = mm;
		mopa.cmplx = false;
		mopa.randommom = ic.randommom;
		mopa.mp = axionField->m2Cpu();

	momConf(axionField, mopa);

	prof.stop();
	prof.add("momConf", 0., axionField->Size()*axionField->DataSize()*1e-9);

	/* iFFT */
	prof.start();
	LogMsg(VERB_NORMAL,"[GEN] fft! ");
	auto &myPlan = AxionFFT::fetchPlan("pSpecAx");
	myPlan.run(FFT_BCK);

	prof.stop();
	prof.add("fft", 0., axionField->Size()*axionField->DataSize()*1e-9);

	/* unpad */
	prof.start();

	LogMsg(VERB_NORMAL,"[GEN] unpad! ");
	size_t dl = axionField->Length()*axionField->Precision();
	size_t pl = (axionField->Length()+2)*axionField->Precision();
	size_t ss	= axionField->Length()*axionField->Depth();

	char *ms = static_cast<char *>(axionField->mStart());
	char *vs = static_cast<char *>(axionField->vCpu());
	char *m2 = static_cast<char *>(axionField->m2Cpu());

	for (size_t sl=0; sl<ss; sl++) {
		size_t	oOff = sl*dl;
		size_t	fOff = sl*pl;
		memmove	(ms+oOff, m2+fOff, dl);
	}

	prof.stop();
	prof.add("unpad", 0., axionField->Size()*axionField->DataSize()*1e-9);


	LogMsg(VERB_NORMAL,"[GEN] Create axion velocity! ");
	prof.start();
	mopa.mfttab = vv;
	momConf(axionField, mopa);
	prof.stop();
	prof.add("momConf v", 0., axionField->Size()*axionField->DataSize()*1e-9);


	LogMsg(VERB_NORMAL,"[GEN] fft! ");
	prof.start();

	myPlan.run(FFT_BCK);
	prof.stop();
	prof.add("fft", 0., axionField->Size()*axionField->DataSize()*1e-9);

/* unpad */
	LogMsg(VERB_NORMAL,"[GEN] unpad! ");
	prof.start();
	for (size_t sl=0; sl<ss; sl++) {
		size_t	oOff = sl*dl;
		size_t	fOff = sl*pl;
		memmove	(vs+oOff, m2+fOff, dl);
	}
	prof.stop();
	prof.add("unpad", 0., axionField->Size()*axionField->DataSize()*1e-9);



	/* If saxion was specified, convert axion only to saxion
	and add ... possibly ... string network
	*/

	if (myCosmos->ICData().fType == FIELD_SAXION)
	{
		LogMsg(VERB_NORMAL,"[GEN] CONF_SPAX SAXION MODE! \n");
		/* Now add strings a la "lola" (they go to m) */

		double logit = ic.logi;
		double xit = (249.48 + 38.8431*logit + 1086.06*logit*logit)/(21775.3 + 3665.11*logit)  ;
		if (ic.siter == 1)
			xit *= ic.kcr;
		if (xit > 0.0)
		{
			LogMsg(VERB_NORMAL,"[GEN] Creating string on top of SPAX! \n");
			/* moves ctheta to m2*/
			memcpy	   (m2, ms, axionField->DataSize()*axionField->Size());
			float *lerdo = static_cast<float*>(axionField->m2Cpu());
			putxi(xit,ic.kMax);
			normaliseField(axionField, FIELD_M);
		}

		/* Build the scalar field merging the extra waves

		we have :
			strings in mStart, phi_s
			extra waves psi = theta*R in m2Cpu
			extra waves psi' = theta'*R+theta R' in vCpu
		we need :
			conformal scalar Phi = phi_s * exp(i theta) * R, theta = psi/R IN MSTART
			conformal velocity Phi' = phi_s * exp(i theta) = phi_s * exp(i theta) * (R' + i theta')
				theta' = psi'/R - theta R'/R
		psi' = (thetaR)' = theta'R + theta R'
		psi  = theta R
		Phi  = phi exp(I psi/R) R
		phi' = (rho'  + rho i theta')exp(I theta)
				 > (rho'  + rho i theta'+extra)exp(I (theta+extra))
*/
		size_t vol = axionField->Size();


		if (axionField->Precision() == FIELD_SINGLE)
		{
				float R    = *axionField->RV();
				float Rp   = myCosmos->Rp(*axionField->zV());
				complex<float> *co_m = static_cast<complex<float>*>(axionField->mStart());
				complex<float> *co_v = static_cast<complex<float>*>(axionField->vCpu());
				float *f_m  = static_cast<float*>(axionField->m2Cpu());
				float *f_v  = static_cast<float*>(axionField->vCpu());
				#pragma omp parallel for default(shared) schedule(static)
				for (size_t pidx = 0; pidx < vol; pidx++){
					size_t idx = vol-pidx-1;
					float theta  = f_m[idx]/R;
					float thetap = f_v[idx]/R-theta*Rp;
					co_m[idx] = co_m[idx]*exp(complex<float>(0.0,theta))*R;
					co_v[idx] = complex<float>(Rp,thetap)*co_m[idx];
				}
			}
			else
			{
				double R    = *axionField->RV();
				double Rp   = myCosmos->Rp(*axionField->zV());
				complex<double> *co_m = static_cast<complex<double>*>(axionField->mStart());
				complex<double> *co_v = static_cast<complex<double>*>(axionField->vCpu());
				double *f_m  = static_cast<double*>(axionField->m2Cpu());
				double *f_v  = static_cast<double*>(axionField->vCpu());
				#pragma omp parallel for default(shared) schedule(static)
				for (size_t pidx = 0; pidx < vol; pidx++){
					size_t idx = vol-pidx-1;
					float theta  = f_m[idx]/R;
					float thetap = f_v[idx]/R-theta*Rp;
					co_m[idx] = co_m[idx]*exp(complex<double>(0.0,theta))*R;
					co_v[idx] = complex<double>(Rp,thetap)*co_m[idx];
				}
			}
		/* Normalise cores? */
		if (myCosmos->ICData().normcore)
			normCoreField	(axionField);
	}
	LogMsg(VERB_NORMAL,"[GEN] CONF_SPAX end! \n");
} // endconf spectrum axions



void	ConfGenerator::conftkac(Cosmos *myCosmos, Scalar *axionField)
{
		std::complex<float> *ma = static_cast<std::complex<float>*>(axionField->mStart());
		std::complex<float> *va = static_cast<std::complex<float>*> (axionField->vCpu());
		std::complex<float> *m2 = static_cast<std::complex<float>*> (axionField->m2Cpu());

	IcData ic = myCosmos->ICData();

	LogMsg(VERB_NORMAL,"\n ");
	LogMsg(VERB_NORMAL,"[GEN] CONF_TKACHEV started!\n ");
	//these initial conditions make sense only for RD
	if (axionField->BckGnd()->Frw() != 1.0)
		{
			LogMsg(VERB_NORMAL,"[GEN] Tkachev Initial conditions only prepared for RD !\n ");
		}

	AxionFFT::initPlan (axionField, FFT_CtoC_MtoM, FFT_FWDBCK, "InitM");
	AxionFFT::initPlan (axionField, FFT_CtoC_VtoV, FFT_FWDBCK, "InitV");
	auto &myPlanM = AxionFFT::fetchPlan("InitM"); // now transposed
	auto &myPlanV = AxionFFT::fetchPlan("InitV"); // now transposed
	//probably the profiling has to be modified

	double kCritz = axionField->BckGnd()->PhysSize()/(6.283185307179586*(*axionField->zV()));
	LogMsg(VERB_NORMAL,"[GEN] kCrit changed according to initial time %f to kCrit %f !\n ", (*axionField->zV()),kCritz);

	// ft_theta' in M2, ft_theta in V
	MomParms mopa;
		mopa.kMax = ic.kMax;
		mopa.kCrt = kCritz;
		mopa.mocoty = MOM_MVSINCOS;
		mopa.cmplx = true;
		mopa.mp = axionField->mStart();
		mopa.vp = axionField->vStart();
	momConf(axionField, mopa);

	// LogOut("m %e %e \n", real(ma[0]), imag(ma[0]));
	// LogOut("v %e %e \n", real(va[0]), imag(va[0]));
	// LogOut("2 %e %e \n", real(m2[1]), imag(m2[1]));


	// The normalisation factor is
	// <theta^2> = pi^2*/3 - we use KCrit as a multiplicicative Factor
	// <theta^2> = 1/2 sum |~theta|^2 =1/2 4pi/3 nmax^3 <|~theta|^2>
	// <|~theta|^2> = pi^2*kCrit/3 * (3/2pi nmax^3)
	// <|~theta|> = sqrt(pi/2 kCrit/nmax^3)
	// and therefore we need to multiply theta and theta' by this factor (was 1)

	/* A better estimate ignoring modes in the corners of phase spaced
	and using that |~theta|^2 = constant^2 * (sin(k z)/kz)^2
	<theta^2> = 1/2 constant^2 4pi
	  	(1/2)(1/k0z)^3(kmax z - 0.5 sin(2 kmax z))
	note that the last term does indeed behave as kmax
	for kmax z < 1
	 */
	LogMsg(VERB_NORMAL,"kCrit %e kMax %d \n", ic.kcr, ic.kMax);
	// old naive version
	// double norma = std::sqrt(1.5707963*ic.kcr/(ic.kMax*ic.kMax*ic.kMax));
	// better version including mode decay inside horizon

	// better version adjust by hand

	/* Analytical estimate */
	// double norma = ic.kcr/std::sqrt(3.14159*kCritz*kCritz*kCritz*(ic.kMax/kCritz - 0.5*std::sin(2*ic.kMax/kCritz)));
  // LogMsg(VERB_NORMAL,"norma1 %e \n",norma);
	// scaleField (axionField, FIELD_V, norma);
	// norma /= (*axionField->zV());
	// scaleField (axionField, FIELD_M2, norma);
	// LogMsg(VERB_NORMAL,"norma2 %e \n",norma);

	// LogOut("m %e %e \n", real(ma[0]), imag(ma[0]));
	// LogOut("v %e %e \n", real(va[0]), imag(va[0]));
	// LogOut("2 %e %e \n", real(m2[1]), imag(m2[1]));

	myPlanM.run(FFT_BCK);
	myPlanV.run(FFT_BCK);
	// takes only the real parts and builds exp(itheta), ...
	// it builds
	double theta2_loco = 0;
	double theta2 = 0;
	if (axionField->Precision() == FIELD_SINGLE) {
		float* thets = static_cast<float*> (axionField->mStart());
		#pragma parallel for reduce(+:theta2)
		for (size_t idx = 0; idx < axionField->Size(); idx++)
		{
			theta2_loco += (double) pow(thets[2*idx],2);
		}
	} else {
		double* thets = static_cast<double*> (axionField->mStart());
		#pragma parallel for reduce(+:theta2)
		for (size_t idx = 0; idx < axionField->Size(); idx++)
		{
			theta2_loco += pow(thets[2*idx],2);
		}
	}
	MPI_Allreduce (&theta2_loco, &theta2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	// printf("theta2_loco %.2e\n",theta2_loco);

	theta2 /= (double) axionField->TotalSize();
	LogMsg(VERB_NORMAL,"theta2 is %.2e we will normalise to %.2e \n", theta2,M_PI*M_PI/3.0 * ic.kcr);

	double norma = M_PI/(std::sqrt(3*theta2/ic.kcr));

	LogMsg(VERB_NORMAL,"norma1 %e \n",norma);
	scaleField (axionField, FIELD_M, norma);
	norma /= (*axionField->zV());
	scaleField (axionField, FIELD_V, norma);
	LogMsg(VERB_NORMAL,"norma2 %e \n",norma);

	theta2Cmplx	(axionField);
	axionField->setFolded(false);
}



void	ConfGenerator::confthermal(Cosmos *myCosmos, Scalar *axionField)
{
		std::complex<float> *ma = static_cast<std::complex<float>*>(axionField->mStart());
		std::complex<float> *va = static_cast<std::complex<float>*> (axionField->vCpu());
		std::complex<float> *m2 = static_cast<std::complex<float>*> (axionField->m2Cpu());

	IcData ic = myCosmos->ICData();

	LogMsg(VERB_NORMAL,"\n ");
	LogMsg(VERB_NORMAL,"[GEN] CONF_THERMAL started!\n ");
	//these initial conditions make sense only for RD

	AxionFFT::initPlan (axionField, FFT_CtoC_MtoM, FFT_FWDBCK, "InitM");
	AxionFFT::initPlan (axionField, FFT_CtoC_VtoV, FFT_FWDBCK, "InitV");
	auto &myPlanM = AxionFFT::fetchPlan("InitM"); // now transposed
	auto &myPlanV = AxionFFT::fetchPlan("InitV");


	// ft_theta' in M2, ft_theta in V

	/* mass2 in ADM units it follows from temperature (in ic.kcr in ADM units)
	and the conformal potential
	 lambda/4 ( Phi^2-v^2)^2 +lambda/6 T^2 Phi^2
	        >>>>>>>> lambda/4 ( cPhi^2-R^2)^2 +lambda/6 (T^2 R^2/v^2) cPhi^2
	The minimum of the potential happens at
	x (x^2-R^2) + cT^2 x/3 = 0
  and the mass2 is
	lambda [3x^2-R^2 + cT^2/3]_xmin
	which are
	x^2 +cT2/3 - R^2 = 0 >>>> xmin = sqrt(R^2-cT2/3) or 0
	m2 = lambda [3(R^2-T2/3)-R^2 + T^2/3] = lambda * 2(R^2-cT2/3) or lambda * (cT2/3-R^2)

	then
		m2 = max (lambda * 2 * R^2 (1-T2/v^2/3), 0)

		*/
		double T2   = ic.kcr*ic.kcr; // in units of [v^2]
		double R2   = (*axionField->RV())*(*axionField->RV());
		double mS2  = axionField->LambdaP()*( (T2 > 3) ? R2*(T2/3- 1) : 2*R2*(1 - T2/3));
		LogMsg(VERB_NORMAL,"[GEN] lambda %e", axionField->LambdaP());
		LogMsg(VERB_NORMAL,"[GEN] T (kcr) %e", ic.kcr);
		LogMsg(VERB_NORMAL,"[GEN] k0 (2pi/L) %e ", 6.283185307179586/axionField->BckGnd()->PhysSize());
		LogMsg(VERB_NORMAL,"[GEN] mass %e", sqrt(mS2));

	MomParms mopa;
		mopa.kMax   = axionField->Length();
		mopa.mass2  = mS2;
		mopa.k0     = 6.283185307179586/axionField->BckGnd()->PhysSize();
		mopa.kCrt   = ic.kcr;
		mopa.mocoty = MOM_MVTHERMAL;
		mopa.cmplx  = true;
		mopa.mp = axionField->mStart();
		mopa.vp = axionField->vStart();
	momConf(axionField, mopa);

	myPlanM.run(FFT_BCK);
	myPlanV.run(FFT_BCK);
	// cphi' in m array

	double norma = 1./pow(axionField->BckGnd()->PhysSize(),1.5);

	LogMsg(VERB_NORMAL,"L %e, norma %e \n",axionField->BckGnd()->PhysSize(), norma);
	scaleField (axionField, FIELD_M, norma);
	scaleField (axionField, FIELD_V, norma);


	axionField->setFolded(false);
}



/* Configuration string:
	reads a txt files with string coordinates, and calculates theta
	from the static solution to the KR field generated by the string

	we decompose
	theta = theta0(x,y,z)+theta1(y,z)+theta2(z)

	where
	partial_x theta_1 = 0
	partial_y theta_2 = 0

	1 - read string coordinates in N, not L
	2 - produce theta(k) as as explicit sum according to coordinates
	3 - iFT to calculate theta ()*/

void	ConfGenerator::confstring(Cosmos *myCosmos, Scalar *axionField)
{
	Profiler &prof = getProfiler(PROF_GENCONF);
	IcData ic = myCosmos->ICData();

	LogMsg(VERB_NORMAL,"\n ");
	LogMsg(VERB_NORMAL,"[GEN] CONF_STRING started!\n ");

	LogMsg(VERB_NORMAL,"[STR] Searching for string data file string.dat !");

	FILE *stringFile = nullptr;
	std::vector<double> xx,yy,zz;
	if (((stringFile  = fopen("./string.dat", "r")) == nullptr)){
		LogMsg(VERB_NORMAL,"[STR] none found ! ");
	}
	else
	{
		LogMsg(VERB_NORMAL,"[STR] Found ! ");
			int ii = 0;
			double xa, ya, za;
			while(!feof(stringFile)){
				fscanf (stringFile ,"%lf %lf %lf", &xa, &ya, &za);
				LogMsg(VERB_PARANOID," x,y,z %.2f %.2f %.2f !",xa, ya, za);
				xx.push_back(xa);
				yy.push_back(ya);
				zz.push_back(za);
				ii++;
			}
	}

	LogMsg(VERB_NORMAL,"[STR] Searching for center data file x0.dat !\n ");
	std::vector<double> x0;
	FILE *stringFile2 = nullptr;

	if (((stringFile2  = fopen("./x0.dat", "r")) == nullptr)){
		LogMsg(VERB_NORMAL,"[STR] none found! chosing default parameters !\n ");
		x0.push_back(axionField->Length()/2);
		x0.push_back(axionField->Length()/2);
		x0.push_back(0);
	}
	else
	{
		LogMsg(VERB_NORMAL,"[STR] Found !\n ");
			int ii = 0;
			double xa;
			while(!feof(stringFile2)){
				fscanf (stringFile2 ,"%lf", &xa);
				x0.push_back(xa);
				LogMsg(VERB_NORMAL," x0[%d] %.f",ii,x0[ii]);
				ii++;
			}
			LogMsg(VERB_NORMAL," x0,y0,z0 %.2f %.2f %.2f !",x0[0],x0[1],x0[2]);
	}

	/* Generate axion in momentum space */
	prof.start();
	LogMsg(VERB_NORMAL,"[GEN] Create FT of axion field ! ");
	MomParms mopa;
		mopa.kMax = axionField->Length();
		mopa.kCrt = ic.kcr;
		mopa.mocoty = MOM_STRING;
		mopa.cmplx = false;
		mopa.randommom = false;
		mopa.k0 = 2*M_PI/axionField->BckGnd()->PhysSize();
		mopa.mp = axionField->m2Cpu();
		mopa.xx = xx;
		mopa.yy = yy;
		mopa.zz = zz;
		mopa.x0 = x0;

	momConf(axionField, mopa);
	prof.stop();
	/* TODO wrong flops */
	prof.add("momConf", 0., axionField->Size()*axionField->DataSize()*1e-9);

	/* iFFT */
	prof.start();
	LogMsg(VERB_NORMAL,"[GEN] fft! ");
	auto &myPlan = AxionFFT::fetchPlan("pSpecAx");
	myPlan.run(FFT_BCK);

	prof.stop();
	prof.add("fft", 0., axionField->Size()*axionField->DataSize()*1e-9);

	/* unpad and complexify */
	prof.start();
	{
		LogMsg(VERB_NORMAL,"[GEN] unpad! ");
		size_t dl = axionField->Length()*axionField->Precision();
		size_t pl = (axionField->Length()+2)*axionField->Precision();
		size_t ss	= axionField->Length()*axionField->Depth();

		char *m2 = static_cast<char *>(axionField->m2Cpu());

		for (size_t sl=0; sl<ss; sl++) {
			size_t	oOff = sl*dl;
			size_t	fOff = sl*pl;
			memmove	(m2+oOff, m2+fOff, dl);
		}
	}
	prof.stop();
	prof.add("unpad", 0., axionField->Size()*axionField->DataSize()*1e-9);


	double norma = 1.0/pow(axionField->BckGnd()->PhysSize(),3);
	LogMsg(VERB_NORMAL,"L %e, norma %e ",axionField->BckGnd()->PhysSize(), norma);
	scaleField (axionField, FIELD_M2, norma);

	LogMsg(VERB_NORMAL,"theta to complex ");
	if (axionField->Precision() == FIELD_SINGLE) {
		float* m2 = static_cast<float*> (axionField->m2Cpu());
		complex<float>* mS = static_cast<complex<float>*> (axionField->mStart());
		#pragma parallel for
		for (size_t idx = 0; idx < axionField->Size(); idx++)
		{
			mS[idx] = exp(complex<float>(0,m2[idx]));
		}
	} else {
		double* m2 = static_cast<double*> (axionField->m2Cpu());
		complex<double>* mS = static_cast<complex<double>*> (axionField->mStart());
		#pragma parallel for
		for (size_t idx = 0; idx < axionField->Size(); idx++)
		{
			mS[idx] = exp(complex<double>(0,m2[idx]));
		}
	}

	axionField->setFolded(false);
	LogMsg(VERB_NORMAL,"[GEN] CONF_STRING ended! ");
}


void	ConfGenerator::confstring2(Cosmos *myCosmos, Scalar *axionField)
{
	Profiler &prof = getProfiler(PROF_GENCONF);
	IcData ic = myCosmos->ICData();
	string	smthName("Smoother");

	LogMsg(VERB_NORMAL,"\n ");
	LogMsg(VERB_NORMAL,"[GEN] CONF_STRING' started!\n ");

	LogMsg(VERB_NORMAL,"[STR] Searching for string data file string.dat !");

	FILE *stringFile = nullptr;
	std::vector<double> xx,yy,zz;
	std::vector<int> endpoints;

	if (((stringFile = fopen("./string.dat", "r")) == nullptr)) {
    	LogMsg(VERB_NORMAL, "[STR] none found !");
	}
	else {
    	LogMsg(VERB_NORMAL, "[STR] Found ! ");
    	int ii = 0;
    	double xa, ya, za;
    	bool ep;
    	char line[256]; // Assuming a maximum line length of 256 characters

    	while (fgets(line, sizeof(line), stringFile) != nullptr) {
        	// Check if the line starts with '#' (header line)
        	if (line[0] == '#') {
            	continue; // Skip header lines
        	}

        	// Process data if it's not a header line
        	sscanf(line, "%lf %lf %lf %d", &xa, &ya, &za, &ep);
        	LogMsg(VERB_PARANOID, " x,y,z %.2f %.2f %.2f %d !", xa, ya, za, ep);
        	xx.push_back(xa);
        	yy.push_back(ya);
        	zz.push_back(za);
        	endpoints.push_back(ep);
        	ii++;
    	}

    	fclose(stringFile); // Don't forget to close the file when done


		// Add periodic copies of the string outside the simulation volume
		const int numCopies = 1; // Number of copies to add
		//const double boxSize = myCosmos->TotalDepth();

		for (int i = -numCopies; i < numCopies + 1; i++) {
			if (i == 0)
				continue;

				for (int ind = 0; ind < ii; ind++) {
					// Add a copy shifted in the x-direction
					double newX = xx.at(ind);
					xx.push_back(newX);
		   		// Add a copy shifted in the y-direction
				double newY = yy.at(ind);
				yy.push_back(newY);
		   		// Add a copy shifted in the z-direction
				double newZ = zz.at(ind) + i * axionField->TotalDepth();
				zz.push_back(newZ);
			}
		}

	}

	prof.start();
	anystringConf (axionField, ic, xx.data(), yy.data(), zz.data(), xx.size(), endpoints.data(), endpoints.size());
	prof.stop();

	/* TODO wrong flops */
	prof.add("anystringConf", 0., xx.size()*axionField->Size()*axionField->DataSize()*1e-9);

	if (ic.siter>0)
	{
		prof.start();
		smoothXeon (axionField, ic.siter, ic.alpha);
		prof.stop();
		prof.add(smthName, 18.e-9*axionField->Size()*ic.siter, 8.e-9*axionField->Size()*axionField->DataSize()*ic.siter);
	}

	normaliseField(axionField, FIELD_M);

	if (myCosmos->ICData().normcore)
		normCoreField	(axionField);

	if (!myCosmos->Mink()){ /* In Minkowski this is trivial */
		double R  = *axionField->RV();
		double Rp = myCosmos->Rp(*axionField->zV());
		axby(FIELD_M,FIELD_V,Rp*R,R);
		}
	if ( !(ic.kickalpha == 0.0) )
		scaleField (axionField, FIELD_V, 1.0+ic.kickalpha);

	if (!myCosmos->Mink()) /* In Minkowski this is trivial */
		scaleField (axionField, FIELD_M, *axionField->RV());

	axionField->setFolded(false);
	LogMsg(VERB_NORMAL,"[GEN] CONF_STRING2 ended'' ");
}


// void	ConfGenerator::confapr(Cosmos *myCosmos, Scalar *axionField)
// {
// 		std::complex<float> *ma = static_cast<std::complex<float>*>(axionField->mStart());
// 		std::complex<float> *va = static_cast<std::complex<float>*> (axionField->vCpu());
// 		std::complex<float> *m2 = static_cast<std::complex<float>*> (axionField->m2Cpu());
//
// 	IcData ic = myCosmos->ICData();
//
// 	LogMsg(VERB_NORMAL,"\n ");
// 	LogMsg(VERB_NORMAL,"[GEN] CONF_APR started! adds two initial condition functions smooth + smooth or smooth + spax\n ");
//
// }




void	ConfGenerator::putxi(double xit, bool kspace)
{
	/* Builds a configuration with a given value of xit */
	LogMsg(VERB_NORMAL,"[GENputxi] putxi xi = %f, mode %d ",xit,kspace);
	/*This is the expression that follows from the definition of xi*/
	double nN3 = (6.0*xit*axionField->Delta()*axionField->Delta()/pow(*axionField->zV(),2));
	nN3 = min(nN3,1.0);

	if (kspace)
	{	AxionFFT::initPlan (axionField, FFT_CtoC_MtoM,  FFT_FWDBCK, "Initputxi");
		auto &myPlan = AxionFFT::fetchPlan("Initputxi");  // now transposed

		/* This is the phenomenological calibrated fit with MOM_MEXP2*/
		double nc = axionField->Length()*std::sqrt((nN3/4.7)*pow(1.-pow(nN3,1.5),-1./1.5));

		LogMsg(VERB_NORMAL,"[GENputxi] xit(logi)= %f estimated nN3 = %f -> n_critical = %f!",xit, nN3, nc);

		LogMsg(VERB_NORMAL,"[GENputxi] momConf with axionField->Length() %d kMax %d kcr %f! momConf %d\n ",axionField->Length(),axionField->Length(),nc,MOM_MEXP2);

		MomParms mopa;
			mopa.kMax = axionField->Length();
			mopa.kCrt = nc;
			mopa.mocoty = MOM_MEXP2;
			mopa.cmplx = true;
			mopa.mp = axionField->mStart();
		momConf(axionField, mopa);

		axionField->setFolded(false);

		myPlan.run(FFT_BCK);
	} else {
	/* This is the phenomenological calibrated fit with alpha =0.143*/
		int niter = (int) (0.8/nN3);

		LogMsg(VERB_NORMAL,"[GENputxi-smooth] xit(logi)= %f estimated nN3 = %f -> niter = %f!",xit, nN3, niter);

		IcData ic = axionField->BckGnd()->ICData();
		ic.fieldindex = FIELD_M;
		randConf (axionField,ic);

		if (niter>100) {
				LogMsg(VERB_NORMAL,"[GENputxi] WARNING!! More than 100 iterations is not particularly efficient! update VILGOR algorithm to use FFTs!!\n");
		}
		smoothXeon (axionField, niter, 0.143);
		axionField->setFolded(false);
	}
} //end putxi




double	ConfGenerator::anymean(FieldIndex ftipo)
{
	/* Computes the average value of rho for normalisation purposes */
	LogMsg(VERB_NORMAL,"[GEN] any mean %d ",ftipo);
	void* meandro ;
	double mean, Mean;

	switch(ftipo){
		case FIELD_M:
			meandro = static_cast<void *>(axionField->mStart());
		break;
		case FIELD_V:
			meandro = static_cast<void *>(axionField->vCpu());
		break;
		case FIELD_M2:
			meandro = static_cast<void *>(axionField->m2Cpu());
		break;
	}

	switch(axionField->Precision()){
		case FIELD_DOUBLE:{
			complex<double> *m = static_cast<complex<double>*>(meandro);
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static) reduction(+:mean)
				for (size_t idx = 0; idx < axionField->Size(); idx++)
				mean += abs(m[idx]);
			}
		} break;
		case FIELD_SINGLE:{
			complex<float> *m = static_cast<complex<float>*>(meandro);
			#pragma omp parallel
			{
				#pragma omp for schedule(static) reduction(+:mean)
				for (size_t idx = 0; idx < axionField->Size(); idx++)
				mean += (double) abs(m[idx]);
			}
		} break;
		}

	MPI_Allreduce(&mean, &Mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	LogMsg(VERB_NORMAL,"[GEN] mean %.10e ",Mean/((double) axionField->eSize()));
	return Mean/((double) axionField->eSize());
} //end rhomean




void	ConfGenerator::susum(FieldIndex ftipo1,FieldIndex ftipo2)
{
	/* Sum something 1 into something 2*/
	LogMsg(VERB_NORMAL,"[GEN] sum field%d into field$d ",ftipo1,ftipo2);
	void* meandro ;
	void* meandro2 ;

	switch(ftipo1){
		case FIELD_M:
			meandro = static_cast<void *>(axionField->mStart());
		break;
		case FIELD_V:
			meandro = static_cast<void *>(axionField->vCpu());
		break;
		case FIELD_M2:
			meandro = static_cast<void *>(axionField->m2Cpu());
		break;
	}
	switch(ftipo2){
		case FIELD_M:
			meandro2 = static_cast<void *>(axionField->mStart());
		break;
		case FIELD_V:
			meandro2 = static_cast<void *>(axionField->vCpu());
		break;
		case FIELD_M2:
			meandro2 = static_cast<void *>(axionField->m2Cpu());
		break;
	}

	switch(axionField->Precision()){
		case FIELD_DOUBLE:{
			complex<double> *m = static_cast<complex<double>*>(meandro);
			complex<double> *d = static_cast<complex<double>*>(meandro2);
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static)
				for (size_t idx = 0; idx < axionField->Size(); idx++)
				d[idx] += m[idx];
			}
		} break;
		case FIELD_SINGLE:{
			complex<float> *m = static_cast<complex<float>*>(meandro);
			complex<float> *d = static_cast<complex<float>*>(meandro2);
			#pragma omp parallel
			{
				#pragma omp for schedule(static)
				for (size_t idx = 0; idx < axionField->Size(); idx++)
				d[idx] += m[idx];
			}
		} break;
		}

	return;
} //end susum




void	ConfGenerator::mulmul(FieldIndex ftipo1,FieldIndex ftipo2)
{
	/* Multiplies something 1 into something 2*/
	LogMsg(VERB_NORMAL,"[GEN] multiply field%d times field%d ",ftipo2,ftipo1);
	void* meandro ;
	void* meandro2 ;

	switch(ftipo1){
		case FIELD_M:
			meandro = static_cast<void *>(axionField->mStart());
		break;
		case FIELD_V:
			meandro = static_cast<void *>(axionField->vCpu());
		break;
		case FIELD_M2:
			meandro = static_cast<void *>(axionField->m2Cpu());
		break;
	}
	switch(ftipo2){
		case FIELD_M:
			meandro2 = static_cast<void *>(axionField->mStart());
		break;
		case FIELD_V:
			meandro2 = static_cast<void *>(axionField->vCpu());
		break;
		case FIELD_M2:
			meandro2 = static_cast<void *>(axionField->m2Cpu());
		break;
	}

	switch(axionField->Precision()){
		case FIELD_DOUBLE:{
			complex<double> *m = static_cast<complex<double>*>(meandro);
			complex<double> *d = static_cast<complex<double>*>(meandro2);
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static)
				for (size_t idx = 0; idx < axionField->Size(); idx++)
				d[idx] *= m[idx];
			}
		} break;
		case FIELD_SINGLE:{
			complex<float> *m = static_cast<complex<float>*>(meandro);
			complex<float> *d = static_cast<complex<float>*>(meandro2);
			#pragma omp parallel
			{
				#pragma omp for schedule(static)
				for (size_t idx = 0; idx < axionField->Size(); idx++)
				d[idx] *= m[idx];
			}
		} break;
		}

	return;
} //end mulmul

void	ConfGenerator::axby(FieldIndex ftipo1, FieldIndex ftipo2, double a, double b)
{
	/* Multiplies something 1 into something 2*/
	LogMsg(VERB_NORMAL,"[GEN] axby (f%d) = (f%d)x%.e + (f%d)x%.e",ftipo2,ftipo2,b,ftipo1,a);
	void* meandro ;
	void* meandro2 ;

	switch(ftipo1){
		case FIELD_M:
			meandro = static_cast<void *>(axionField->mStart());
		break;
		case FIELD_V:
			meandro = static_cast<void *>(axionField->vCpu());
		break;
		case FIELD_M2:
			meandro = static_cast<void *>(axionField->m2Cpu());
		break;
	}
	switch(ftipo2){
		case FIELD_M:
			meandro2 = static_cast<void *>(axionField->mStart());
		break;
		case FIELD_V:
			meandro2 = static_cast<void *>(axionField->vCpu());
		break;
		case FIELD_M2:
			meandro2 = static_cast<void *>(axionField->m2Cpu());
		break;
	}

	switch(axionField->Precision()){
		case FIELD_DOUBLE:{
			complex<double> *m = static_cast<complex<double>*>(meandro);
			complex<double> *d = static_cast<complex<double>*>(meandro2);
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static)
				for (size_t idx = 0; idx < axionField->Size(); idx++)
				d[idx] = b*d[idx] + a*m[idx];
			}
		} break;
		case FIELD_SINGLE:{
			complex<float> *m = static_cast<complex<float>*>(meandro);
			complex<float> *d = static_cast<complex<float>*>(meandro2);
			#pragma omp parallel
			{
				#pragma omp for schedule(static)
				for (size_t idx = 0; idx < axionField->Size(); idx++)
				d[idx] = ((float) b)*d[idx] + ((float) a)*m[idx];
			}
		} break;
		}

	return;
} //end mulmul

void	genConf	(Cosmos *myCosmos, Scalar *field)
{
	LogMsg  (VERB_NORMAL, "[GEN] Called configurator generator II");
	LogFlush();

	auto	cGen = std::make_unique<ConfGenerator> (myCosmos, field);

	switch (field->Device())
	{
		case DEV_CPU:
			cGen->runCpu ();
			field->exchangeGhosts(FIELD_M);
			break;

		case DEV_GPU:
			// cGen->runGpu ();
			// field->exchangeGhosts(FIELD_M);
			field->setDev(DEV_CPU);
			cGen->runCpu ();
			field->setDev(DEV_GPU);
			field->transferDev(FIELD_MV);
			break;

		default:
			LogError ("Not a valid device");
			break;
	}

	return;
}
