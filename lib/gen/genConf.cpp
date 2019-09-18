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
#include "gen/smoothXeon.h"
#include "gen/prepropa.h"
#include "io/readWrite.h"
#include "propagator/propXeon.h"
#include "scalar/folder.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
	#include "kernelParms.cuh"
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

		 ConfGenerator(Cosmos *myCosmos, Scalar *field);
		~ConfGenerator() {};

	void	runCpu	();
	void	runGpu	();
	void	runXeon	();
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

	switch (cType)
	{
		default:
		case CONF_NONE:
		break;

		case CONF_READ:
		readConf (myCosmos, &axionField, index);
		break;

		case CONF_TKACHEV: {
			auto &myPlan = AxionFFT::fetchPlan("Init"); // now transposed
			prof.start();
			momConf(axionField, kMax, kCrt, MOM_MFLAT);
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
			auto &myPlan = AxionFFT::fetchPlan("Init"); // now transposed
			prof.start();
			momConf(axionField, kMax, kCrt, MOM_MEXP2);
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
		break;
	}


	if ((cType != CONF_READ) && (cType != CONF_NONE)) {

		if (!myCosmos->Mink()) {
			double	  lTmp = axionField->BckGnd()->Lambda()/((*axionField->RV()) * (*axionField->RV()));
			double	  ood2 = 1./(axionField->Delta()*axionField->Delta());
			uint	  S    = axionField->Surf();
			uint	  V    = axionField->Size();
			uint	  Vo   = S;
			uint	  Vf   = V+S;
			cudaMemcpy(axionField->vGpu(), static_cast<char *> (axionField->mGpu()) + axionField->DataSize()*axionField->Surf(), axionField->DataSize()*axionField->Size(), cudaMemcpyDeviceToDevice);
			scaleField(axionField, FIELD_M, *axionField->RV());
			axionField->exchangeGhosts(FIELD_M);
			updateVGpu(axionField->mGpu(), axionField->vGpu(), *axionField->RV(), *axionField->RV(), 1.0, ood2, lTmp, axionField->AxionMassSq(), 0.0, axionField->Length(), axionField->Depth(), Vo, Vf, S,
				   axionField->BckGnd()->QcdPot() & VQCD_TYPE, axionField->Precision(), xBlockDefaultGpu, yBlockDefaultGpu, zBlockDefaultGpu, ((cudaStream_t *)axionField->Streams())[2]);
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
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.zi       %f",ic.zi       );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.logi     %f",ic.logi     );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.cType    %d",ic.cType    );
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.smvarTy  %d",ic.smvarType);
	LogMsg(VERB_NORMAL,"[gen] myCosmos.ic.momConf  %d",ic.momConf);

	LogMsg(VERB_NORMAL, "[gen] cType is %d",cType);

	switch (cType)
	{
		case CONF_NONE:
		break;

		case CONF_READ:
		readConf (myCosmos, &axionField, index);
		break;

		case CONF_TKACHEV: {

			std::complex<float> *ma = static_cast<std::complex<float>*>(axionField->mStart());
			std::complex<float> *va = static_cast<std::complex<float>*> (axionField->vCpu());
			std::complex<float> *m2 = static_cast<std::complex<float>*> (axionField->m2Cpu());


			LogMsg(VERB_NORMAL,"\n ");
			LogMsg(VERB_NORMAL,"[GEN] CONF_TKACHEV started!\n ");
			//these initial conditions make sense only for RD
			if (axionField->BckGnd()->Frw() != 1.0)
				{
					LogMsg(VERB_NORMAL,"[GEN] Tkachev Initial conditions only prepared for RD !\n ");
				}
			LogMsg(VERB_NORMAL,"[GEN] CONF_TKACHEV started!\n ");
			auto &myPlan = AxionFFT::fetchPlan("Init"); // now transposed
			//probably the profiling has to be modified
			prof.start();
			double kCritz = axionField->BckGnd()->PhysSize()/(6.283185307179586*(*axionField->zV()));
			LogMsg(VERB_NORMAL,"[GEN] kCrit changed according to initial time %f to kCrit %f !\n ", (*axionField->zV()),kCritz);

			// ft_theta' in M2, ft_theta in V
			momConf(axionField, ic.kMax, kCritz, MOM_MVSINCOS);
			// LogOut("m %e %e \n", real(ma[0]), imag(ma[0]));
			// LogOut("v %e %e \n", real(va[0]), imag(va[0]));
			// LogOut("2 %e %e \n", real(m2[1]), imag(m2[1]));


			// The normalisation factor is
			// <theta^2> = pi^2*/3 - we use KCrit as a multiplicicative Factor
			// <theta^2> = 1/2 sum |~theta|^2 =1/2 4pi/3 nmax^3 <|~theta|^2>
			// <|~theta|^2> = pi^2*kCrit/3 * (3/2pi nmax^3)
			// <|~theta|> = sqrt(pi/2 kCrit/nmax^3)
			// and therefore we need to multiply theta and theta' by this factor (was 1)
			LogMsg(VERB_NORMAL,"kCrit %e kMax %d \n", ic.kcr, ic.kMax);
			double norma = std::sqrt(1.5707963*ic.kcr/(ic.kMax*ic.kMax*ic.kMax));
			LogMsg(VERB_NORMAL,"norma1 %e \n",norma);
			scaleField (axionField, FIELD_V, norma);
			norma /= (*axionField->zV());
			scaleField (axionField, FIELD_M2, norma);
			LogMsg(VERB_NORMAL,"norma2 %e \n",norma);

			// LogOut("m %e %e \n", real(ma[0]), imag(ma[0]));
			// LogOut("v %e %e \n", real(va[0]), imag(va[0]));
			// LogOut("2 %e %e \n", real(m2[1]), imag(m2[1]));

			prof.stop();
			prof.add(momName, 14e-9*axionField->Size(), axionField->Size()*axionField->DataSize()*1e-9);
			myPlan.run(FFT_BCK);
			// theta' into M

			// LogOut("m %e %e \n", real(ma[0]), imag(ma[0]));
			// LogOut("v %e %e \n", real(va[0]), imag(va[0]));
			// LogOut("2 %e %e \n", real(m2[1]), imag(m2[1]));

			// FIX FOR LOWMEM!
			// move theta from V to M2
			size_t volData = axionField->Size()*axionField->DataSize();
			memcpy(static_cast<char *>(axionField->m2Cpu()), static_cast<char *>(axionField->vCpu()), volData);
			// move theta' from M to V
			memcpy(static_cast<char *>(axionField->vCpu()), static_cast<char *>(axionField->mStart()), volData);
			myPlan.run(FFT_BCK);
			// takes only the real parts and builds exp(itheta), ...
			// it builds
			theta2Cmplx	(axionField);
		}
		axionField->setFolded(false);
		break;

		case CONF_KMAX: {
			LogMsg(VERB_NORMAL,"[GEN] CONF_KMAX started!\n ");
			LogFlush();
			auto &myPlan = AxionFFT::fetchPlan("Init"); // now transposed
			LogFlush();
			prof.start();
			// LogOut("[GEN] momConf with kMax %zu kCrit %f!\n ", kMax, kCrt);
			momConf(axionField, ic.kMax, ic.kcr, ic.momConf);
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
			auto &myPlan = AxionFFT::fetchPlan("Init");  // now transposed

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

			LogMsg(VERB_NORMAL,"[GEN] momConf with kMax %d kcr %f! momConf %d\n ",sizeN,nc,ic.momConf);
			prof.start();
			momConf(axionField, sizeN, nc, ic.momConf);
			prof.stop();
			prof.add(momName, 14e-9*axionField->Size(), axionField->Size()*axionField->DataSize()*1e-9);
			axionField->setFolded(false);

			myPlan.run(FFT_BCK);

			normaliseField(axionField, FIELD_M);

			if (!myCosmos->Mink() && !ic.preprop) {
				if (myCosmos->ICData().normcore)
					normCoreField	(axionField);
				memcpy	   (axionField->vCpu(), static_cast<char *> (axionField->mStart()), axionField->DataSize()*axionField->Size());
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
			auto &myPlan = AxionFFT::fetchPlan("Init");  // now transposed

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
			momConf(axionField, sizeN, nc, MOM_MEXP2);
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

				// LogMsg (VERB_HIGH, "[GEN] Rescale from phi to conformal field (estimates A1=%f A2=%f)", (3.14*hzi)*(3.14*hzi)*ood2, lTmp*hzi*hzi*hzi*hzi);
				//
				// Folder	munge(axionField);
				// munge(FOLD_ALL);
				// axionField->exchangeGhosts(FIELD_M);
				//
				// switch (axionField->BckGnd()->QcdPot() & VQCD_TYPE) {
				// 	case	VQCD_1:
				// 	updateVXeon<VQCD_1>	(axionField->mCpu(), axionField->vCpu(), axionField->RV(), hzi, 1.0, ood2, lTmp, axionField->AxionMassSq(), 0.0,
				// 				 axionField->Length(), Vo, Vf, S, axionField->Precision());
				// 	break;
				//
				// 	case	VQCD_2:
				// 	updateVXeon<VQCD_2>	(axionField->mCpu(), axionField->vCpu(), axionField->RV(), hzi, 1.0, ood2, lTmp, axionField->AxionMassSq(), 0.0,
				// 				 axionField->Length(), Vo, Vf, S, axionField->Precision());
				// 	break;
				//
				// 	case	VQCD_1_PQ_2:
				// 	updateVXeon<VQCD_1_PQ_2>(axionField->mCpu(), axionField->vCpu(), axionField->RV(), hzi, 1.0, ood2, lTmp, axionField->AxionMassSq(), 0.0,
				// 				 axionField->Length(), Vo, Vf, S, axionField->Precision());
				// 	break;
				//
				// 	case	VQCD_1N2:
				// 	updateVXeon<VQCD_1N2>	(axionField->mCpu(), axionField->vCpu(), axionField->RV(), hzi, 1.0, ood2, lTmp, axionField->AxionMassSq(), 0.0,
				// 				 axionField->Length(), Vo, Vf, S, axionField->Precision());
				// 	break;
				//
				// 	case	VQCD_QUAD:
				// 	updateVXeon<VQCD_QUAD>	(axionField->mCpu(), axionField->vCpu(), axionField->RV(), hzi, 1.0, ood2, lTmp, axionField->AxionMassSq(), 0.0,
				// 				 axionField->Length(), Vo, Vf, S, axionField->Precision());
				// 	break;
				// }
				// munge(UNFOLD_ALL);
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
			randConf (axionField);
			prof.stop();
			prof.add(randName, 0., axionField->Size()*axionField->DataSize()*1e-9);

			// logi = log ms/H is taken to be zInit (which was input in command line)
			// number of iterations needed = 0.8/(#/N^3)
			// desired 0.8/(#/N^3) is 6*xi(logi)*msa^2*exp(-2logi)
			double logi = *axionField->zV();
			*axionField->zV() = (axionField->Delta())*exp(logi)/axionField->Msa();
			axionField->updateR();
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
			// initPropagator (pType, axionField, (axionField->BckGnd().QcdPot() & VQCD_TYPE) | VQCD_EVOL_RHO);
			// tunePropagator (axiona);
			// if (int i ==0; i<10; i++ ){
			// 	dzaux = axion->dzSize(zInit);
			// 	propagate (axiona, dzaux);
			// }
		}
		axionField->setFolded(false);
		break;

		case CONF_SMOOTH:
		prof.start();
		randConf (axionField);
		prof.stop();
		prof.add(randName, 0., axionField->Size()*axionField->DataSize()*1e-9);
		prof.start();
		smoothXeon (axionField, ic.siter, ic.alpha);
		prof.stop();
		prof.add(smthName, 18.e-9*axionField->Size()*ic.siter, 8.e-9*axionField->Size()*axionField->DataSize()*ic.siter);
		if (ic.smvarType != CONF_SAXNOISE)
			normaliseField(axionField, FIELD_M);
		if (myCosmos->ICData().normcore)
			normCoreField	(axionField);

		axionField->setFolded(false);
		break;
	}



	if (((cType == CONF_KMAX) || (cType == CONF_SMOOTH)) || (cType == CONF_VILGORS)) {

		if (!myCosmos->Mink()) {
			double	   lTmp = axionField->BckGnd()->Lambda()/((*axionField->RV()) * (*axionField->RV()));
			double	   ood2 = 1./(axionField->Delta()*axionField->Delta());
			memcpy     (axionField->vCpu(), static_cast<char *> (axionField->mCpu()) + axionField->DataSize()*axionField->Surf(), axionField->DataSize()*axionField->Size());
			scaleField (axionField, FIELD_M, *axionField->RV());

			auto	   S  = axionField->Surf();
			auto	   V  = axionField->Size();
			auto	   Vo = S;
			auto	   Vf = V+S;
			double   hzi = *axionField->zV()/2.;

			// LogMsg (VERB_HIGH, "[GEN] Rescale from phi to conformal field (estimates A1=%f A2=%f)", (3.14*hzi)*(3.14*hzi)*ood2, lTmp*hzi*hzi*hzi*hzi);
			//
			// Folder	munge(axionField);
			// munge(FOLD_ALL);
			// axionField->exchangeGhosts(FIELD_M);
			//
			// switch (axionField->BckGnd()->QcdPot() & VQCD_TYPE) {
			// 	case	VQCD_1:
			// 	updateVXeon<VQCD_1>	(axionField->mCpu(), axionField->vCpu(), axionField->RV(), hzi, 1.0, ood2, lTmp, axionField->AxionMassSq(), 0.0,
			// 				 axionField->Length(), Vo, Vf, S, axionField->Precision());
			// 	break;
			//
			// 	case	VQCD_2:
			// 	updateVXeon<VQCD_2>	(axionField->mCpu(), axionField->vCpu(), axionField->RV(), hzi, 1.0, ood2, lTmp, axionField->AxionMassSq(), 0.0,
			// 				 axionField->Length(), Vo, Vf, S, axionField->Precision());
			// 	break;
			//
			// 	case	VQCD_1_PQ_2:
			// 	updateVXeon<VQCD_1_PQ_2>(axionField->mCpu(), axionField->vCpu(), axionField->RV(), hzi, 1.0, ood2, lTmp, axionField->AxionMassSq(), 0.0,
			// 				 axionField->Length(), Vo, Vf, S, axionField->Precision());
			// 	break;
			//
			// 	case	VQCD_1N2:
			// 	updateVXeon<VQCD_1N2>	(axionField->mCpu(), axionField->vCpu(), axionField->RV(), hzi, 1.0, ood2, lTmp, axionField->AxionMassSq(), 0.0,
			// 				 axionField->Length(), Vo, Vf, S, axionField->Precision());
			// 	break;
			//
			// 	case	VQCD_QUAD:
			// 	updateVXeon<VQCD_QUAD>	(axionField->mCpu(), axionField->vCpu(), axionField->RV(), hzi, 1.0, ood2, lTmp, axionField->AxionMassSq(), 0.0,
			// 				 axionField->Length(), Vo, Vf, S, axionField->Precision());
			// 	break;
			// }
			// munge(UNFOLD_ALL);

		}
	}

}

void	genConf	(Cosmos *myCosmos, Scalar *field)
{
	LogMsg  (VERB_NORMAL, "[GEN] Called configurator generator II");

	auto	cGen = std::make_unique<ConfGenerator> (myCosmos, field);

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
