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
	// ninfa.mask = spmask;
	// ninfa.rmask = rmask;

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
	initPropagator (pType, axiona, VQCD_1);
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
		// energy(axiona, eRes, false, 0.);
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
	initPropagator (pType, axiona, VQCD_1_DRHO);
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

	double trala = 1.;
	fIndex2 = 0 ;
	while ( trala > goalnN3 )
	{
		if (icstudy){
			ninfa.index=fIndex2;
			lm = Measureme (axiona, ninfa);
			fIndex2++;
		} else
			lm.str = strings(axiona);

		dzaux = axiona->dzSize();
		// energy(axiona, eRes, false, 0.);
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
	}
}


// //--------------------------------------------------
// // prepropagator with relaxing strong damping
// //--------------------------------------------------
// // only if preprop and if z smaller or equal than zInit
// // When z>zInit, it is understood that prepropagation was done
// // NEW it takes the pregam value (if is > 0, otherwise gam )
// if (preprop && ((*axion->zV()) < zInit)) {
// 	//
// 	// LogOut("pppp Preprocessing ... z=%f->%f (VQCDTYPE %d, gam=%.2f pregam=%.2f dwgam=%.2f) \n\n",
// 	// 	(*axion->zV()), zInit, (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_DAMP_RHO, myCosmos.Gamma(),pregammo,dwgammo);
// 	LogOut("pppp Preprocessing ... z=%f->%f (VQCDTYPE %d, gam=%.2f pregam=%.2f dwgam=%.2f) \n\n",
// 		(*axion->zV()), zInit, (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_EVOL_RHO, myCosmos.Gamma(),pregammo,dwgammo);
// 	// gammo is reserved for long-time damping
// 	// use pregammo for prepropagation damping
// 	double gammo_save = myCosmos.Gamma();
// 	double *zaza = axion->zV();
// 	double strdensn;
//
// 	if (pregammo > 0)
// 		myCosmos.SetGamma(pregammo);
//
// 	// prepropagation is always with rho-damping
// 	LogOut("Prepropagator always with damping Vqcd flag %d\n", (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_DAMP_RHO);
// 	initPropagator (pType, axion, (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_DAMP_RHO);
// 	tunePropagator (axion);
//
// 	while (*zaza < zInit){
// 		dzaux = axion->dzSize(zInit)/2.;
// 		//myCosmos.SetGamma(gammo_save*pow(abs(1.0 - (*zaza)/zInit)/(1. - 1./prepcoe),1.5));
//
// 		// obs?
// 		printsample(file_samp, axion, myCosmos.Lambda(), idxprint, lm.str.strDen, lm.maxTheta);
// 		//printsample(file_samp, axion, myCosmos.Lambda(), idxprint, nstrings_globale, maximumtheta);
// 		if (icstudy){
//
// 			// lm = Measureme (axion, index, MEAS_STRING | MEAS_ENERGY | MEAS_2DMAP);
// 			ninfa.index=index;
// 			ninfa.measdata=MEAS_ALLBIN | MEAS_STRING | MEAS_STRINGMAP |
// 			MEAS_ENERGY | MEAS_2DMAP | MEAS_SPECTRUM ;
// 			lm = Measureme (axion, ninfa, MEAS_ALLBIN | MEAS_STRING | MEAS_STRINGMAP |
// 			MEAS_ENERGY | MEAS_2DMAP | MEAS_SPECTRUM);
// 			index++;
//
// 		} else{
// 			// LogOut("z %f (gamma %f)\n", *zaza, myCosmos.Gamma());
// 			LogOut(".");
// 		}
// 		propagate (axion, dzaux);
// 	}
//
// 	myCosmos.SetGamma(gammo_save);
// }
// LogOut("\n");
