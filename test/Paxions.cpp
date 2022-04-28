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

double find_saturation_ct(Scalar *axion, double zf);

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

	/*typical value of phi*/
	double typ_phi = 1;

	if (axion->BckGnd()->ICData().grav>0.0){

		LogOut ("Switch on gravity! (grav %..2e)\n",axion->BckGnd()->ICData().grav);
		initGravity(axion);

		if (axion->BckGnd()->ICData().grav_hyb){
			LogOut ("Tunning hybrid method\n");
			tuneGravityHybrid	();

		}


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
			LogOut("Phi max %.3f min %.3f\n",contBin.max(),contBin.min());

			typ_phi = max(contBin.max(),-contBin.min());

			/* Assumes gravitational field in m2start */
			ct_sat = find_saturation_ct(axion,zFinl);
			LogOut("Saturation time %.2f\n",ct_sat);

	}
	bool sat = false;


	LogOut ("Start redshift loop\n\n");
	for (int iz = 0; iz < nSteps; iz++)
	{

		dzaux = (uwDz) ? axion->dzSize() : (zFinl-zInit)/nSteps ;

		/* normalise dynamical graavity time-step?
		Option 1, (Naive) allow only phase~1 per iteration in the point with the largest grav-pot.
		there is really not need because our integrator is exact in V as we alternate V and K Kick operators. */
		// {
		// 	double dzg = 1./axion->BckGnd()->ICData().grav/axion->AxionMass()/typ_phi;
		// 	if (dzaux > dzg)
		// 		dzaux = dzg;
		// }

		/* Maximum, logarithmic time steps */
		if (dzaux > *axion->zV())
			dzaux = *axion->zV();

		/* If time step accross grav-saturation, cut on grav-saturation time */
		if (*axion->zV() < ct_sat && *axion->zV() + dzaux > ct_sat)
			dzaux = ct_sat - *axion->zV();

		/* If time step accross axion mass growth, shorten it to nail it */
		if (*axion->zV() < axion->BckGnd()->ZThRes() && *axion->zV() + dzaux > axion->BckGnd()->ZThRes())
			dzaux = axion->BckGnd()->ZThRes() - *axion->zV();


		/* decrease saturation time to improve */
		if (axion->BckGnd()->ICData().grav_sat)
			if (!sat && *axion->zV() == ct_sat)
			{
				/* We want to saturate the conformal axion mass to be constant from this moment on
				this means that mA R = (mA R)_now
				in power-law cosmology R=ct^frw
				mA^2 = 1/R^2frw, which we can achieve with nqcd = -2frw
				nqcd is only active below zthreshold or above zrestore
				if ct<Rc, then set Rc=R_restore=Rnow
				if ct>Rc, set R_restore=R_now
				if ct>Rrestore>Rc, need to change indi3 (because of the way Rthres is implemented)
				 */
				/*change definition of mA, nqcd */
				if (ninfa.printconf & PRINTCONF_PAXIONSAT)
					writeConf(axion,index); 
				
				double R0 = *axion->RV();
				double Rc = axion->BckGnd()->ZThRes();
				double Rr = axion->BckGnd()->ZRestore();
				double aa = 2*axion->BckGnd()->Frw();
				if (R0 <= Rc){
					axion->BckGnd()->SetZThRes(R0);
					axion->BckGnd()->SetZRestore(R0);
					axion->BckGnd()->SetQcdExpr(-aa);
					LogOut("Linear gravitaty resolution limit achived! SET Rc=Rr = %e and n = %f\n",R0,-aa);
				} else if (R0 > Rc && R0 <= Rr) {
					axion->BckGnd()->SetZRestore(R0);
					axion->BckGnd()->SetQcdExpr(-aa);
					LogOut("Linear gravitaty resolution limit achived! SET Rr = %e (Rc = %e) and n = %f\n",R0,Rc,-aa);
				} else if (R0 > Rc && R0 > Rr) {
					// we will keep Rc, Rr, but will change indi3 to have the same conformal mass with the new nqcd = -2*frw
					// aMassSq = indi3**2*(Rc)**nqcd * (R/Rr)**nqcd = indi3**2*(Rc R / Rr)**nqcd
					// indi3_neq = aMass(nqcd)/[(Rc R / Rr)]**(-frw)
					double newindi3 = axion->AxionMass()/pow(Rc*R0/Rr,-aa);
					axion->BckGnd()->SetIndi3(-aa);
					axion->BckGnd()->SetQcdExpr(-aa);
					LogOut("Linear gravitaty resolution limit achived! RESET indi3 = %e (Rc = %e, Rr = %e) and n = %f\n",newindi3,Rc,Rr,-aa);
				}
				// need to renormalise also indi3
				// have to make aMass = indi3*indi3*pow(zThRes, nQcd)_old= indi3*indi3*pow(zThRes, -1)

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
	int dct mA [(phi)2-(phi)1] ~ pi

	int dct mA phi12 ~ pi
	~ 2/n ct mA phi12 ~ pi
	I will calculate
	ct mA gravi phi12_code = 1
*/


double find_saturation_ct(Scalar *axion, double zFinl)
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
	LogOut("[VAX find_saturation_ct] grad max %.2e grad_mean %.2e\n",grad_max, grad_mean);

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
	LogMsg(VERB_NORMAL,"[VAX find_saturation_ct] Saturation time %f ", ct2 );LogFlush();
	return ct2 ;
} else {
	return -1 ; }
}
