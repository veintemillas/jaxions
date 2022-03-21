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

double find_saturation_ct(Scalar *axion);

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
	LogOut("\n           PAXION EVOLUTION to %f                   \n", zFinl);
	LogOut("\n-------------------------------------------------\n");

	LogOut("\n-------------------------------------------------\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	Scalar *axion;

	LogOut ("reading conf %d ...", fIndex);
	readConf(&myCosmos, &axion, fIndex);
	if (axion == NULL)
	{
		LogOut ("Error reading HDF5 file\n");
		exit (0);
	}
	LogOut ("\n");

	if (axion->Field() != FIELD_AXION)
	{
		LogOut ("Error: Paxion only works in axion mode\n");
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
	LogOut("mass   =  %3.3f\n\n", axion->AxionMass());

	if (axion->Precision() == FIELD_SINGLE)
		LogOut("precis = SINGLE(%d)\n",FIELD_SINGLE);
	else
		LogOut("precis = DOUBLE(%d)\n",FIELD_DOUBLE);

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
	ninfa.index = fIndex;
	ninfa.redmap = endredmap;

	initPropagator (pType, axion, myCosmos.QcdPot(),Nng);
	tunePropagator (axion);

	LogOut("-----------------------\n TRANSITION TO PAXION \n");
	thetaToPaxion (axion);
	resetPropagator(axion);
	// for (size_t aaaa = 0; aaaa < axion->Surf(); aaaa++){
	// 	static_cast<float*>(axion->vCpu())[aaaa] = aaaa;
	// 	static_cast<float*>(axion->vStart())[aaaa+axion->Size()] = aaaa;
	// }


	int counter = 0;
	int index ;
	double dzaux;
	int i_meas = 0;
	bool measrightnow = false;

	ninfa.index=index;
	// ninfa.measdata |= MEAS_3DMAP;
	// lm = Measureme (axion, ninfa);
	// ninfa.measdata ^= MEAS_3DMAP;
	LogOut("-----------------------\n");
	index++;
	tunePropagator(axion);


	/* Saturation */
	double ct_sat = 1e300;
	if (axion->BckGnd()->ICData().grav > 0)
		ct_sat = find_saturation_ct(axion);
	/*typical value of phi*/
	double typ_phi = 1;

	if (axion->BckGnd()->ICData().grav>0.0){
		initGravity(axion);
		LogOut ("Switch on gravity! (grav %..2e)\n",axion->BckGnd()->ICData().grav);


		/* Conversion of Units ?
			we could convert
			from ct=eta/eta_1 to eta/etaeq
			from R = a/a1 to x = R/Req
			from L = L_c H1R1 to L_c HeqReq
			from mA = m_A/H1 to m_A/Heq, etc...

			we can renormalise the cpax to be <|cpax|^2> = 1

			In ADM units,
			grav = 3/4 (HeqReq/H1R1)^2 1/R

			The R = R(eta) relation close to eq is

			R/Req =  eta/etaeq /sqrt{2} + 0.25 (eta/etaeq /sqrt{2})^2

			eta_eq = 1/Heq R_eq (by definition here)
			eta_1  = 1/H1 R1 (... exact?)

			-----------------------------------------

			grav   = 5.12e-10 (50ueV/m_A)^0.167
			R1/Req = 1.46747-10 (50ueV/m_A)^0.172
			e1/eeq = 3.1690110e-10 (50ueV/m_A)^0.167

			resolution limit ...
			mA R < (pi/delta) / grav

			we force it by using msa ?
			decrease mA in time ... > no expansion of the Universe?
			mA R delta = msa <pi/grav>
			mA = msa <pi/grav>/ R delta

			R/R1 we do an ugly trick assuming no change in DOF

			R/R1 = ct +  0.25 Req/R1(ct eta/etaeq /sqrt{2})^2
			R/R1 = ct +  0.25 8.55437*10^-11 ct**2
			*/

			/* Calculate gravitational time scale */
			calculateGraviPotential();
			Binner<3000,float> contBin(static_cast<float *>(axion->m2Start()), axion->Size(),
							[] (float x) -> float { return (double) ( x ) ;});
			contBin.find();
			LogOut("max %.3f min %.3f\n\n",contBin.max(),contBin.min());

			typ_phi = max(contBin.max(),-contBin.min());

	}
	bool sat = false;


	LogOut ("Start redshift loop\n\n");
	for (int iz = 0; iz < nSteps; iz++)
	{

		dzaux = (uwDz) ? axion->dzSize() : (zFinl-zInit)/nSteps ;

		/* normalise dynamical graavity time-step?
		dpax/dct = ... +  mA * gravi * phi
		assuming phi~O(1)  ... TODO calculate with if (axion->m2Status()=M2_POT)
		dct < 1/gravi mA
		*/
		{
			double dzg = 1./axion->BckGnd()->ICData().grav/axion->AxionMass()/typ_phi;
			if (dzaux > dzg)
				dzaux = dzg;
		}

		/*sat the sat*/
		if (*axion->zV() < ct_sat && *axion->zV() + dzaux > ct_sat)
			dzaux = ct_sat - *axion->zV();

		/* decrease saturation time to improve */
		if (!sat && *axion->zV() > ct_sat)
		// if (!sat && axion->AxionMass()*(*axion->RV()) > 1/axion->Delta()/axion->BckGnd()->ICData().grav)
		{
			/*change definition of mA, nqcd */
			axion->BckGnd()->SetZRestore(*axion->RV());
			axion->BckGnd()->SetQcdExpr(-2.0);
			// need to renormalise also indi3
			// have to make aMass = indi3*indi3*pow(zThRes, nQcd)_old= indi3*indi3*pow(zThRes, -1)
			LogOut("Resolution limit achived! setting Rrestore = %e and n = -1.0\n",*axion->RV()); fflush(stdout);
			sat = true;
		}


		if (!(iz%dump)){
			measrightnow = true;
		}

		propagate (axion, dzaux);
		counter++;

		// Break the loop when we are done
		if ( (*axion->zV()) >= zFinl ){
			LogOut("zf reached! ENDING ... \n"); fflush(stdout);
			break;
		}
		if ( abs((*axion->zV())-zFinl) < 1.0e-10 ){
			LogOut("zf approximately reached! ENDING ... \n"); fflush(stdout);
			break;
		}

		// Partial analysis
		if(measrightnow){

			ninfa.index=index;
			lm = Measureme (axion, ninfa);
			index++;
			i_meas++ ;
			measrightnow = false;
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


/* Finds the time at which the gravitational term becomes large
  phase difference between two points
	int dct mAc [(phi)2-(phi)1] ~ pi
	or ...
	int dct mAc grad (phi) delta ~ pi
	using grad phi ~ k0 phi

	int dct mAc phi /k0 delta ~ pi
	~ n/2 ct mAc gravi delta * k0 ~ pi
	I will use k0 the base grid (can be much worse)
	ct mAc gravi = 1
*/


double find_saturation_ct(Scalar *axion)
{

	double fff = axion->BckGnd()->Frw();

	if (axion->BckGnd()->Indi3() > 0.0 && (fff > 0.0)){
	double ct = *axion->zV();
	double fun0 = axion->AxionMass(ct)*ct*axion->BckGnd()->ICData().grav;
	double fun = fun0;
	double ct2 = ct*1.1;
	double fun2 = axion->AxionMass(ct2)*ct2*axion->BckGnd()->ICData().grav;
	double k = 1;
	double a = 1;
	double meas = std::abs(fun2 - 1);;
	LogMsg(VERB_NORMAL,"[VAX find_saturation_ct] frw %f indi3 %f ct %e ", fff, axion->BckGnd()->Indi3(), ct );LogFlush();
	while (meas > 0.001)
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
		fun2 = axion->AxionMass(ct2)*ct2*axion->BckGnd()->ICData().grav;
		meas = std::abs(fun2 - 1);
		LogMsg(VERB_PARANOID,"ct2 %e fun2 %e meas %e k %e a %e", ct2, fun2, meas, k ,a );
	}
	//LogOut("ct2 %e DWfun2 %e meas %e k %e a %e\n", ct2, DWfun2, meas, k ,a );
	LogMsg(VERB_NORMAL,"[VAX find_saturation_ct] Saturation time %f ", ct2 );LogFlush();
	return ct2 ;
} else {
	return -1 ; }
}
