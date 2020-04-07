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

using namespace std;
using namespace AxionWKB;

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
			double dzg = 1./axion->BckGnd()->ICData().grav/axion->AxionMass();
			if (dzaux > dzg)
				dzaux = dzg;
		}

		if (!sat && axion->AxionMass()*(*axion->RV()) > 3.141597/axion->Delta()/axion->BckGnd()->ICData().grav)
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
