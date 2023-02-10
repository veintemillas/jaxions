#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "propagator/allProp.h"
#include "energy/energy.h"
#include "utils/utils.h"
#include "io/readWrite.h"
#include "comms/comms.h"
#include "map/map.h"
#include "strings/strings.h"
#include "powerCpu.h"
#include "scalar/scalar.h"
#include "spectrum/spectrum.h"
#include "meas/measa.h"

#include "WKB/WKB.h"
#include "gravity/potential.h"

using namespace std;
using namespace AxionWKB;

double find_saturation (Scalar *axion);

int	main (int argc, char *argv[])
{

	double zendWKB = 10. ;
	Cosmos myCosmos = initAxions(argc, argv);

	if (nSteps==0)
	return 0 ;

	//--------------------------------------------------
	//       AUX STUFF
	//--------------------------------------------------

	void *eRes, *str;			// Para guardar la energia
	trackAlloc(&eRes, 128);
	memset(eRes, 0, 128);
	double *eR = static_cast<double *> (eRes);

	double  *binarray	 ;
	trackAlloc((void**) (&binarray),  10000*sizeof(size_t));
	double *bA = static_cast<double *> (binarray);
	size_t sliceprint = 0 ; // sizeN/2;



	commSync();
	LogOut("\n-------------------------------------------------\n");
	LogOut("\n           WKB EVOLUTION to %f                   \n", zFinl);
	LogOut("\n-------------------------------------------------\n");

	LogOut("\n-------------------------------------------------\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	Scalar *axion;

	LogOut ("Reading conf axion.%05d ...", fIndex);
	readConf(&myCosmos, &axion, fIndex);
	if (axion == NULL)
	{
		LogOut ("Error reading HDF5 file\n");
		exit (0);
	}
	LogOut ("\n");

	if (axion->Field() != FIELD_AXION)
	{
		LogOut ("Error: WKV only works in axion mode\n");
		exit (0);
	}
	LogOut ("\n");


	double z_now = (*axion->zV())	;
	LogOut("--------------------------------------------------\n");
	LogOut("           INITIAL CONDITIONS                     \n\n");

	LogOut("Length =  %2.2f\n", myCosmos.PhysSize());
	LogOut("nQCD   =  %2.2f\n", myCosmos.QcdExp());
	LogOut("N      =  %ld\n",   axion->Length());
	LogOut("Nz     =  %ld\n",   axion->Depth());
	LogOut("zGrid  =  %ld\n",   zGrid);
	LogOut("z      =  %2.2f\n", z_now);
	LogOut("zthr   =  %3.3f\n", myCosmos.ZThRes());
	LogOut("zres   =  %3.3f\n", myCosmos.ZRestore());
	LogOut("mass   =  %3.3f\n", axion->AxionMass());

	if (axion->Precision() == FIELD_SINGLE)
		LogOut("precis =  SINGLE(%d)\n",FIELD_SINGLE);
	else
		LogOut("precis =  DOUBLE(%d)\n",FIELD_DOUBLE);

	LogOut("--------------------------------------------------\n");

	//--------------------------------------------------
	//       MEASUREMENT
	//--------------------------------------------------
	//- Measurement
	MeasData lm;
	//- number of plaquetes pierced by strings
	lm.str.strDen = 0 ;
	//- Info to measurement
	MeasInfo ninfa = deninfa;
	//- information needs to be passed onto measurement files
	// ninfa.sliceprint = sliceprint;
	// ninfa.idxprint = 0 ;
	ninfa.index = fIndex;
	ninfa.redmap = endredmap;
	//
	// // default measurement type is parsed
	// ninfa.measdata = defaultmeasType;
	// ninfa.mask = spmask;
	// ninfa.rmask = rmask;

	//--------------------------------------------------
	//       WKB
	//--------------------------------------------------

	// LogOut ("creating new axion2 ... FIELD_TYPE(%d) ", FIELD_WKB );
	// Scalar *axion2;
	// axion2 = new Scalar (&myCosmos, sizeN, sizeZ, sPrec, cDev, z_now, true, zGrid, FIELD_WKB, LAMBDA_FIXED, CONF_NONE, 0. , 0. );
	// LogOut ("done !\n");
	// WKB wonka(axion, axion2);
	initGravity(axion);

	if (zFinl < z_now)
		zFinl = z_now	;

	LogOut("from z1=%f to z2=%f in %d log-time steps\n\n", z_now, zFinl, nSteps);

	for (int i = 1; i < nSteps+1; i++)
	{
		ninfa.index++;

		ninfa.measdata = defaultmeasType;
		lm = Measureme (axion, ninfa);

		// double zco = z_now + i*(zFinl-z_now)/nSteps	;
		double zco = z_now*std::exp( std::log(zFinl/z_now)*i/nSteps )	;
		{
			double ct_sat = 1e300;
			
			// TO IMPLEMENT
			//calculateGraviPotential();
			//ct_sat = find_saturation(axion);
			
			LogOut ("WKBing to %.4f ... ", zco);
			WKB wonka(axion, axion);
			wonka(zco) 	;
			LogOut ("done!\n");
		}

	}

	//--------------------------------------------------
	//       SAVE DATA
	//--------------------------------------------------

	ninfa.index++;

	LogOut ("\n\n Dumping configuration %05d ...", ninfa.index);
	writeConf(axion, ninfa.index);
	LogOut ("Done!\n\n");


	LogOut ("Printing FINAL measurement file %05d ... ", ninfa.index);

	Measureme (axion, ninfa);

	endAxions();

	return 0;
}

/*  
	AUXILIARY FUNCTIONS
*/  

double find_saturation(Scalar *axion)
{
	double fff = axion->BckGnd()->Frw();

	if (axion->BckGnd()->Indi3() > 0.0 && (fff > 0.0)){

	LogMsg(VERB_NORMAL,"[VAX find_saturation_ct] we first find max gradient of potential (only along z to avoid folding math)");
	double grad_max  = 0;
	double grad_mean = 0;
	if (axion->Precision() == FIELD_SINGLE) {
		float* fieldo = static_cast<float *> (axion->m2Start());
		float grad = 0;
		#pragma omp parallel for schedule(static) reduction(+:grad_mean) reduction(max:grad_max)
		for(size_t idx =0; idx < axion->Size()-axion->Surf();idx++){
			grad = abs(fieldo[idx+axion->Surf()] - fieldo[idx]);
			if (grad > grad_max)
				grad_max = grad;
			grad_mean += grad;
		}
	} else {
		double* fieldo = static_cast<double *> (axion->m2Start());
		double grad = 0;
		#pragma omp parallel for schedule(static) reduction(+:grad_mean)
		for(size_t idx =0; idx < axion->Size()-axion->Surf();idx++){
			grad = abs(fieldo[idx+axion->Surf()] - fieldo[idx]);
			if (grad > grad_max)
				grad_max = grad;
			grad_mean += grad;
		}
	}
	grad_mean /= axion->Size()-axion->Surf();
	LogMsg(VERB_NORMAL,"[VAX find_saturation_ct] grad max %.2e grad_mean %.2e",grad_max, grad_mean);
	//LogOut("[VAX find_saturation_ct] grad max %.2e grad_mean %.2e\n",grad_max, grad_mean);

	double ct = *axion->zV();
	double phi12 = axion->BckGnd()->ICData().grav*grad_max;
	// double phi12 = axion->BckGnd()->ICData().grav*grad_mean;
	LogMsg(VERB_NORMAL,"[VAX find_saturation_ct] phi12 %.5e",phi12);
	// double fun0 = axion->AxionMass(ct)*ct*phi12*log(zFinl/ct);
	double fun0 = axion->AxionMass(ct)*ct*phi12;
	double fun = fun0;
	double ct2 = ct*1.1;
	// double fun2 = axion->AxionMass(ct2)*ct2*phi12*log(zFinl/ct2);
	double fun2 = axion->AxionMass(ct2)*ct2*phi12;
	double k = 1;
	double a = 1;
	double meas = std::abs(fun2 - 1);;
	LogMsg(VERB_NORMAL,"[VAX find_saturation_ct] frw %f indi3 %f ct %e ", fff, axion->BckGnd()->Indi3(), ct );LogFlush();

	while (meas > 0.00001)
	{
		/* Assume power law
		fun ~ k ct^a
		then fun2/fun = (ct2/ct)^a
		a = log(fun2/fun)/log(ct2/ct)
		k = fun2/ct2^a
		fun = 1 -> ct3 = 1/k^{1/a}, ct2 = ct
		*/
		a = std::log(fun2/fun)/std::log(ct2/ct);
		k = fun2/std::pow(ct2,a);
		// LogOut("ct %e DWfun %e ct2 %e DWfun2 %e meas %e k %e a %e\n", ct, DWfun, ct2, DWfun2, meas, k ,a );
		if ((a == 0) && (fun2 > fun0) ){
			LogMsg(VERB_PARANOID,"[VAX find_saturation_ct] flat slope between ct %e ct2 %e", ct, ct2 );
			if (fun2 > 1) {
				LogMsg(VERB_PARANOID,"[VAX findzdoom] Jump back!");
				ct  = ct2;
				ct2 = std::sqrt(*axion->zV()*ct2);
			}
			else {
				LogMsg(VERB_PARANOID,"[VAX find_saturation_ct] DWfun will never reach 1");
				return INFINITY;
			}
		} else {
			ct  = ct2;
			ct2 = std::pow(k,-1./a);
		}
		fun = fun2;
		// fun2 = axion->AxionMass(ct2)*ct2*phi12*log(zFinl/ct2);
		fun2 = axion->AxionMass(ct2)*ct2*phi12;
		meas = std::abs(fun2 - 1);
		LogMsg(VERB_NORMAL,"ct2 %e fun2 %e meas %e k %e a %e", ct2, fun2, meas, k ,a );
	}
	//LogOut("ct2 %e DWfun2 %e meas %e k %e a %e\n", ct2, DWfun2, meas, k ,a );
	LogOut("Current saturation time: %.2e\n",ct2);
	LogMsg(VERB_NORMAL,"[VAX find_saturation_ct] Saturation time %f ", ct2 );LogFlush();
	return ct2 ;
} else {
	return -1 ; }
}
