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
	//#include "kernelParms.cuh"
	#include "gen/momGpu.h"
	#include "gen/randGpu.h"
	#include "gen/smoothGpu.h"
	#include "propagator/propGpu.h"
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

	void	 runCpu	();
	void	 runGpu	();
	void	 runXeon	();

	void   conflola(Cosmos *myCosmos, Scalar *field);
	void   confcole(Cosmos *myCosmos, Scalar *field);

	void   putxi(double xit, bool kspace);
	double anymean(FieldIndex ftipo);

	void   susum(FieldIndex from, FieldIndex to);
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
			auto &myPlan = AxionFFT::fetchPlan("Init"); // now transposed
			prof.start();
			MomParms mopa;
			mopa.kMax = kMax;
			mopa.kCrt = kCrt;
			mopa.mocoty = MOM_MFLAT;
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
			auto &myPlan = AxionFFT::fetchPlan("Init"); // now transposed
			prof.start();
			MomParms mopa;
				mopa.kMax = kMax;
				mopa.kCrt = kCrt;
				mopa.mocoty = MOM_MEXP2;
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
			double	  lTmp = axionField->BckGnd()->Lambda()/((*axionField->RV()) * (*axionField->RV()));
			double	  ood2 = 1./(axionField->Delta()*axionField->Delta());
			uint	  S    = axionField->Surf();
			uint	  V    = axionField->Size();
			uint	  Vo   = S;
			uint	  Vf   = V+S;
			cudaMemcpy(axionField->vGpu(), static_cast<char *> (axionField->mGpu()) + axionField->DataSize()*axionField->Surf(), axionField->DataSize()*axionField->Size(), cudaMemcpyDeviceToDevice);
			scaleField(axionField, FIELD_M, *axionField->RV());
			axionField->exchangeGhosts(FIELD_M);
			updateVGpu(axionField->mGpu(), axionField->vGpu(), axionField->RV(), *axionField->RV(), 1.0, ood2, lTmp, axionField->AxionMassSq(), 0.0, axionField->Length(), axionField->Depth(), Vo, Vf,
				   axionField->BckGnd()->QcdPot() & VQCD_TYPE, axionField->Precision(), 512, 1, 1, ((cudaStream_t *)axionField->Streams())[2]);
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
			auto &myPlan = AxionFFT::fetchPlan("Init"); // now transposed
			//probably the profiling has to be modified
			prof.start();
			double kCritz = axionField->BckGnd()->PhysSize()/(6.283185307179586*(*axionField->zV()));
			LogMsg(VERB_NORMAL,"[GEN] kCrit changed according to initial time %f to kCrit %f !\n ", (*axionField->zV()),kCritz);

			// ft_theta' in M2, ft_theta in V
			MomParms mopa;
				mopa.kMax = ic.kMax;
				mopa.kCrt = kCritz;
				mopa.mocoty = MOM_MVSINCOS;
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

			MomParms mopa;
				mopa.kMax = ic.kMax;
				mopa.kCrt = ic.kcr;
				mopa.mocoty = ic.mocoty;
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

			LogMsg(VERB_NORMAL,"[GEN] momConf with kMax %d kcr %f! momConf %d\n ",sizeN,nc,ic.mocoty);
			prof.start();
			MomParms mopa;
				mopa.kMax = sizeN;
				mopa.kCrt = nc;
				mopa.mocoty = ic.mocoty;
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

		case CONF_LOLA:
			conflola(myCosmos,axionField);
		break;

		case CONF_COLE:
			confcole(myCosmos,axionField);
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
			MomParms mopa;
				mopa.kMax = sizeN;
				mopa.kCrt = nc;
				mopa.mocoty = MOM_MEXP2;
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
			randConf (axionField,ic);
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
		randConf (axionField,ic);
		prof.stop();
		prof.add(randName, 0., axionField->Size()*axionField->DataSize()*1e-9);
		prof.start();
		smoothXeon (axionField, ic.siter, ic.alpha);
		prof.stop();
		prof.add(smthName, 18.e-9*axionField->Size()*ic.siter, 8.e-9*axionField->Size()*axionField->DataSize()*ic.siter);

		if (!myCosmos->Mink() && !ic.preprop) {
		if ((ic.smvarType != CONF_SAXNOISE) && (ic.smvarType != CONF_PARRES))
			normaliseField(axionField, FIELD_M);
		if (myCosmos->ICData().normcore)
			normCoreField	(axionField);
		memcpy	   (axionField->vCpu(), static_cast<char *> (axionField->mStart()), axionField->DataSize()*axionField->Size());
		if ( !(ic.kickalpha == 0.0) )
			scaleField (axionField, FIELD_V, 1.0+ic.kickalpha);
		scaleField (axionField, FIELD_M, *axionField->RV());

		/*Note that prepropagation folds the field ;-)*/
		axionField->setFolded(false);
	}


		if (ic.preprop) {
		  /* Go back in time the preprop factor*/
		  *axionField->zV() /= ic.prepcoe; // now z <zi
		  axionField->updateR();
		  LogMsg(VERB_NORMAL,"[GEN] prepropagator, jumped back in time to %f",*axionField->zV());
		  prepropa2  (axionField);
		}


		break;




	}


	if ( (cType == CONF_KMAX) || (cType == CONF_VILGORS) ) {

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
	LogMsg(VERB_NORMAL,"[GEN] done!",*axionField->zV());LogFlush();
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
	if (axionField->LambdaT() == LAMBDA_Z2) // Strictly valid only for LamZ2e = 2.0
	  xit = (249.48 + 38.8431*ic.logi + 1086.06* ic.logi*ic.logi)/(21775.3 + 3665.11*ic.logi)  ;
	else // We use this as a nice approximation (for low logi)
	  xit = (249.48 + 38.8431*ic.logi + 1086.06* ic.logi*ic.logi)/(21775.3 + 3665.11*ic.logi)  ;
	  // (9.31021 + 1.38292e-6*logi + 0.713821*logi*logi)/(42.8748 + 0.788167*logi);

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
	putxi(xit,true);

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








void	ConfGenerator::putxi(double xit, bool kspace)
{
	/* Builds a configuration with a given value of xit */
	LogMsg(VERB_NORMAL,"[GENputxi] putxi -> %f! ",xit);
	/*This is the expression that follows from the definition of xi*/
	double nN3 = (6.0*xit*axionField->Delta()*axionField->Delta()/pow(*axionField->zV(),2));
	nN3 = min(nN3,1.0);

	if (kspace)
	{
		auto &myPlan = AxionFFT::fetchPlan("Init");  // now transposed

		/* This is the phenomenological calibrated fit with MOM_MEXP2*/
		double nc = axionField->Length()*std::sqrt((nN3/4.7)*pow(1.-pow(nN3,1.5),-1./1.5));

		LogMsg(VERB_NORMAL,"[GENputxi] xit(logi)= %f estimated nN3 = %f -> n_critical = %f!",xit, nN3, nc);

		LogMsg(VERB_NORMAL,"[GENputxi] momConf with axionField->Length() %d kMax %d kcr %f! momConf %d\n ",axionField->Length(),axionField->Length(),nc,MOM_MEXP2);

		MomParms mopa;
			mopa.kMax = axionField->Length();
			mopa.kCrt = nc;
			mopa.mocoty = MOM_MEXP2;
		momConf(axionField, mopa);

		axionField->setFolded(false);

		myPlan.run(FFT_BCK);
	} else {
	/* This is the phenomenological calibrated fit with alpha =0.143*/
		int niter = (int) (0.8/nN3);
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
	return Mean/((double) axionField->eSize());
} //end rhomean




void	ConfGenerator::susum(FieldIndex ftipo1,FieldIndex ftipo2)
{
	/* Sum something 1 into something 2*/

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
