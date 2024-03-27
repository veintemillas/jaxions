#include <cmath>
#include <chrono>
#include <complex>

#include "energy/energy.h"
#include "propagator/allProp.h"
#include "utils/utils.h"
#include "io/readWrite.h"
#include "comms/comms.h"
#include "map/map.h"
#include "strings/strings.h"
#include "scalar/scalar.h"

#include "meas/measa.h"


using namespace std;

double findct2(double pre_msa, double msa, double ct0, double delta);

void	prepropa  (Scalar *axiona)
{
	MeasData lm;
	//- number of plaquetes pierced by strings
	lm.str.strDen = 0 ;
	//- Info to measurement
	MeasInfo ninfa;
	//- information needs to be passed onto measurement files
	ninfa.sliceprint = 0;
	ninfa.idxprint = 0 ;
	ninfa.index = 0;

	// default measurement type is parsed
	ninfa.measdata = defaultmeasType;
	ninfa.mask = spmask;
	ninfa.rmask = rmask;
	ninfa.nrt = nrt;
  ninfa.nbinsspec = -1;
	ninfa.maty = maty;
	ninfa.i_rmask = rmask_tab.size();
	ninfa.rmask_tab = rmask_tab;
	ninfa.maskenergyonly = false;

	//-maximum value of the theta angle in the simulation
	double maximumtheta = M_PI;
	lm.maxTheta = M_PI;

	// 	void *eRes;
	// 	trackAlloc(&eRes, 128);
	// 	memset(eRes, 0, 128);
	// 	double *eR = static_cast<double *> (eRes);

	Folder munge(axiona);

	if (cDev != DEV_GPU){
		LogOut ("Folding configuration\n");
		munge(FOLD_ALL);
	}

	double dzaux;
	initPropagator (pType, axiona, V_QCD1_PQ1);
	tunePropagator (axiona);

	double ct0 = *axiona->zV();
	// zInit is the logi we wanted
	double masa = axiona->Msa();
	double delto = axiona->Delta();
	double ct2 = findct2(prepcoe*masa, masa, ct0, delto);

	LogOut("[prep] Started prepropaga %f with msa=%f\n",ct0,masa);
	LogOut("[prep] We propagate until nN3(logi+log(prepcoe))[ct2] = nN3(logi)[ct0]\n");
	LogOut("[prep] Since nN3 = 6 xit(logi)*(delta/ct)^2 we find ... ct2 ~ %f\n",ct2);

	double logi = zInit;
	double xit =  (249.48 + 38.8431*logi + 1086.06* logi*logi)/(21775.3 + 3665.11*logi)  ;
	double cta = delto*exp(logi)/masa;; // corresponds to tthis time
	double goalnN3 = 6*xit*pow(delto/cta,2);

	fIndex2 = 0 ;
	while ( *axiona->zV() < 1.2*ct2 )
	{
		if (icstudy){
			ninfa.index=fIndex2;
			lm = Measureme (axiona, ninfa);
			fIndex2++;
		} else
			lm.str = strings(axiona);

		dzaux = axiona->dzSize();
		// energy(axiona, eRes, EN_ENE, 0.);
		// float eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);
		// float eMeanS = (eR[5] + eR[6] + eR[7] + eR[8] + eR[9]);
		// LogOut("%f %f %f\n",*axiona->zV(),eMean, eMeanS);
		// MeasDataOut.str = strings(axiona);
		propagate (axiona, dzaux);

		double trala = ((double) lm.str.strDen)/((double) axiona->TotalSize());
		if (trala < goalnN3)
			break;
	}
}

void	prepropa2  (Scalar *axiona)
{
	LogOut("\nPrepropagator\n");

	LambdaType ltype_aux = axiona->LambdaT();
	double lamba = axiona->BckGnd()->Lambda();
	double cti  = axiona->BckGnd()->ICData().zi;
	double Rnow = axiona->Rfromct(cti);
	double lz2e_aux = axiona->BckGnd()->LamZ2Exp();
	double prelz2e  = axiona->BckGnd()->ICData().prelZ2e;
	LogMsg(VERB_NORMAL,"[prep] Prepropagator 2 (from ct = %f to %f ) with Lambda = L/R^%.1f fixed",*axiona->zV(),cti,prelz2e);
	LogMsg(VERB_NORMAL,"[prep] Currently lamba %f cti %f Ri %f LamZ2Exp = %.2f ",lamba, cti, Rnow, lz2e_aux);
	LogMsg(VERB_NORMAL,"[prep] LambdaType is %d (FIXED/Z2/CONFORMAL is 0/1/2)", ltype_aux);
	/* Physical value of Lambda at Ri*/
	double llp;
	if      (ltype_aux == LAMBDA_FIXED)
		llp = lamba;
	else
		llp = lamba/pow(Rnow,lz2e_aux); /*Usually is 2 (PRS), but it could be different*/

	/* Calculate the new Lambda required to match at zi
	   and set Z2 with the IC exponent */
	axiona->BckGnd()->SetLambda(llp*pow(Rnow,prelz2e));
	LogMsg(VERB_NORMAL,"[prep] lambda = %e reset to %e",lamba,axiona->BckGnd()->Lambda());
	axiona->setLambdaT(LAMBDA_Z2);
	axiona->BckGnd()->SetLamZ2Exp(prelz2e);
	LogMsg(VERB_NORMAL,"[prep] LamZ2Exp seto to %.2f",lz2e_aux);
	LogMsg(VERB_NORMAL,"[prep] Set LAMBDA_Z2 (it was %d) with LamZ2Exp = %.2f ",ltype_aux,axiona->BckGnd()->ICData().prelZ2e);

	/* Now normalising the core makes sense */
	if (axiona->BckGnd()->ICData().normcore)
		normCoreField	(axiona);
	memcpy	   (axiona->vCpu(), static_cast<char *> (axiona->mStart()), axiona->DataSize()*axiona->Size());
	scaleField (axiona, FIELD_M, *axiona->RV());


	MeasData lm;
	//- number of plaquetes pierced by strings
	lm.str.strDen = 0 ;
	//- Info to measurement
	MeasInfo ninfa;
	//- information needs to be passed onto measurement files
	ninfa.sliceprint = 0;
	ninfa.idxprint = 0 ;
	ninfa.index = 0;

	// default measurement type is parsed
	ninfa.measdata = defaultmeasType;
	ninfa.mask = spmask;
	ninfa.rmask = rmask;
	ninfa.nrt = nrt;
  ninfa.nbinsspec = -1;
	ninfa.maty = maty;
	ninfa.i_rmask = rmask_tab.size();
	ninfa.rmask_tab = rmask_tab;
	ninfa.maskenergyonly = false;

	//-maximum value of the theta angle in the simulation
	double maximumtheta = M_PI;
	lm.maxTheta = M_PI;

	// 	void *eRes;
	// 	trackAlloc(&eRes, 128);
	// 	memset(eRes, 0, 128);
	// 	double *eR = static_cast<double *> (eRes);

	double dzaux;
	//by default V_QCD1_PQ1_RHO, only rho evolution
	double gamma_aux = axiona->BckGnd()->Gamma();
	if (axiona->BckGnd()->ICData().pregammo > 0.0){
		axiona->BckGnd()->SetGamma(axiona->BckGnd()->ICData().pregammo);
		LogMsg(VERB_NORMAL,"[prep] Set damping with pregammo = %f (gam = %f)",axiona->BckGnd()->ICData().pregammo,gamma_aux);
	}

	initPropagator (pType, axiona, axiona->BckGnd()->ICData().prevtype);
	tunePropagator (axiona);

	int dump = axiona->BckGnd()->ICData().dumpicmeas;
	int iz = 0;
	fIndex2 = 0 ;
	while ( *axiona->zV() <  cti )
	{
		if (icstudy && !(iz%dump)){
			ninfa.index=fIndex2;
			lm = Measureme (axiona, ninfa);
			fIndex2++;
		} else
			lm.str = strings(axiona);

		dzaux = axiona->dzSize();
		if (*axiona->zV() + dzaux > cti) {
			dzaux = cti - *axiona->zV();
			LogMsg(VERB_NORMAL,"[prep] zi almost reached! adjusting dz! ");
			// break;
		}
		propagate (axiona, dzaux);
		iz++;
	}

	LogMsg(VERB_NORMAL,"[prep] done ");
	LogMsg(VERB_NORMAL,"[prep] LambdType was %d, (CONFORMAL/Z2/FIXED=%d,%d) ",ltype_aux,LAMBDA_CONF,LAMBDA_Z2,LAMBDA_FIXED);

	LogMsg(VERB_NORMAL,"[prep] Restore values");
	if (ltype_aux == LAMBDA_FIXED)
		{
			axiona->setLambdaT(LAMBDA_FIXED);
			axiona->BckGnd()->SetLambda(lamba);
			axiona->BckGnd()->SetLamZ2Exp(lz2e_aux);
			LogMsg(VERB_NORMAL,"[prep] Set LAMBDA_FIXED ");
		} else if (ltype_aux == LAMBDA_Z2)
		{
			axiona->setLambdaT(LAMBDA_Z2);
			axiona->BckGnd()->SetLambda(lamba);
			axiona->BckGnd()->SetLamZ2Exp(lz2e_aux);
			LogMsg(VERB_NORMAL,"[prep] Set LAMBDA_Z2 with LamZ2Exp = .2f",lz2e_aux);
		} else if (ltype_aux == LAMBDA_CONF)
		{
			axiona->setLambdaT(LAMBDA_CONF);
			axiona->BckGnd()->SetLambda(lamba);
			axiona->BckGnd()->SetLamZ2Exp(lz2e_aux);
			LogMsg(VERB_NORMAL,"[prep] Set LAMBDA_CONF with LamZ2Exp = .2f",lz2e_aux);
		}

		if (axiona->BckGnd()->ICData().pregammo > 0.0){
			axiona->BckGnd()->SetGamma( axiona->BckGnd()->Gamma());
			LogMsg(VERB_NORMAL,"[prep] ReSet damping to gam = %f (was %f)",axiona->BckGnd()->Gamma(),gamma_aux);
		}

}



double findct2(double pre_msa, double msa, double ct0, double delta)
{
	double ct2 =ct0;
	double logi = zInit; // logi as input
	double cta = delta*exp(logi)/msa;; // corresponds to tthis time
	double xit =  (249.48 + 38.8431*logi + 1086.06* logi*logi)/(21775.3 + 3665.11*logi)  ;
	double goalnN3 = 6*xit*pow(delta/cta,2);
	logi = log(pre_msa/delta*ct0);
	xit =  (249.48 + 38.8431*logi + 1086.06* logi*logi)/(21775.3 + 3665.11*logi)  ;
	double nN3 = 6*xit*pow(delta/ct0,2);
	printf("Goal nN3 =%f",goalnN3);
	while (goalnN3 < nN3)
	{
		ct2 += ct2/20.;
		logi = log(pre_msa/delta*ct2);
		xit =  (249.48 + 38.8431*logi + 1086.06* logi*logi)/(21775.3 + 3665.11*logi)  ;
		nN3 = 6*xit*pow(delta/ct2,2);
	}
	LogMsg(VERB_NORMAL,"ct2 %f ", ct2 );
	return ct2 ;

}

void	relaxrho  (Scalar *axiona)
{
	MeasData lm;
	lm.str.strDen = 0 ;
	MeasInfo ninfa;
	ninfa.sliceprint = 0;
	ninfa.idxprint = 0 ;
	ninfa.index = 0;
	ninfa.measdata = defaultmeasType;

	Folder munge(axiona);
	if (cDev != DEV_GPU){
		LogOut ("Folding configuration\n");
		munge(FOLD_ALL);
	}

	double dzaux;
	initPropagator (pType, axiona, V_QCD1_PQ1_DRHO);
	tunePropagator (axiona);

	double ct0 = *axiona->zV();
	// zInit is the logi we wanted
	LogOut("[prep] Started prepropaga %f with damping %f\n", ct0, pregammo);

	double masa = axiona->Msa();
	double logi = zInit;
	double xit =  (249.48 + 38.8431*logi + 1086.06* logi*logi)/(21775.3 + 3665.11*logi)  ;
	double delto = axiona->Delta() ;
	double cta = delto*exp(logi)/masa; // time when we want to stop with goal
	double goalnN3 = 6*xit*pow(delto/cta,2);
	if (iter > 0 ) goalnN3 = min(kCrit*goalnN3,1.0);

	LogOut("[prep] goal nN3 = %f \n", goalnN3);

	int dump = axiona->BckGnd()->ICData().dumpicmeas;
	int iz = 0;
	double trala = 1.;
	fIndex2 = 0 ;
	while ( trala > goalnN3 )
	{
		if (icstudy && !(iz%dump)){
			ninfa.index=fIndex2;
			lm = Measureme (axiona, ninfa);
			fIndex2++;
		} else
			lm.str = strings(axiona);

		dzaux = axiona->dzSize();
		// energy(axiona, eRes, EN_ENE, 0.);
		// float eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);
		// float eMeanS = (eR[5] + eR[6] + eR[7] + eR[8] + eR[9]);
		// LogOut("%f %f %f\n",*axiona->zV(),eMean, eMeanS);
		// MeasDataOut.str = strings(axiona);
		propagate (axiona, dzaux);

		if (*axiona->zV() > cta){
			*axiona->zV() = cta;
			axiona->updateR();
		}

		trala= ((double) lm.str.strDen)/((double) axiona->TotalSize());
		// LogOut("nN3 = %f\n",trala);
		iz++;
	}
}
