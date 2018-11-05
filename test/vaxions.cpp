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
#include "scalar/scaleField.h"
#include "spectrum/spectrum.h"
#include "scalar/mendTheta.h"
#include "projector/projector.h"

#include "meas/measa.h"
#include "WKB/WKB.h"

using namespace std;
using namespace AxionWKB;


// vaxions3d definitions

void	printsample  (FILE *fichero, Scalar *axion,            double LLL, size_t idxprint, size_t nstrings_global, double maximumtheta);
void	printsample_p(FILE *fichero, Scalar *axion, double zz, double LLL, size_t idxprint, size_t nstrings_global, double maximumtheta);
double	findzdoom(Scalar *axion);
void	checkTime (Scalar *axion, int index);

// template<typename Float>
// MeasData	Measureme  (Scalar *axiona,  int indexa, MeasureType measa);
//
// MeasData	Measureme  (Scalar *axiona,  int indexa, MeasureType measa);

//-point to print
size_t idxprint = 0 ;
//- z-coordinate of the slice that is printed as a 2D map
size_t sliceprint = 0 ;


/* Program */


int	main (int argc, char *argv[])
{
	Cosmos myCosmos = initAxions(argc, argv);

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	commSync();

	LogOut("\n-------------------------------------------------\n");
	LogOut("\n          VAXION 3D!                             \n\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	start = std::chrono::high_resolution_clock::now();

	//-grids
	Scalar *axion;

	//-records time given in the command line
	double zInit_save = zInit;
	//-prepropagator initial time (goes from zInit to zpreInit)
	double zpreInit = zInit;
	if(preprop)
		zpreInit = zInit/prepcoe;

	if ((fIndex == -1) && (cType == CONF_NONE) && (!restart_flag))
		LogOut("Error: Neither initial conditions nor configuration to be loaded selected. Empty field.\n");
	else
	{
		if ( (fIndex == -1) && !restart_flag)
		{
			//This generates initial conditions
			LogOut("Generating scalar ... ");
			axion = new Scalar (&myCosmos, sizeN, sizeZ, sPrec, cDev, zpreInit, lowmem, zGrid, fTypeP, lType, cType, parm1, parm2);

			LogOut("Done! \n");
		}
		else
		{

			//This reads from an axion.00000 file
			readConf(&myCosmos, &axion, fIndex, restart_flag);

			if (axion == NULL)
			{
				LogOut ("Error reading HDF5 file\n");
				exit (0);
			}
			// prepropagation tends to mess up reading initial conditions
			// configurations are saved before prepropagation and have z<zInit, which readConf reverses
			// the following line fixes the issue, but a more elegant solution could be devised
			if( (preprop) && !restart_flag) {
				zInit = zInit_save;
				*axion->zV() = zpreInit;
			}
		}
	}
	LogOut ("z %f zInit %f zInit_save %f zpreInit %f\n",*axion->zV(),zInit,zInit_save,zpreInit);
	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
	LogOut("ICtime %f min\n",elapsed.count()*1.e-3/60.);

	//--------------------------------------------------
	// USEFUL VARIABLES
	//--------------------------------------------------
	//- Measurement
	MeasData lm;
	//- number of plaquetes pierced by strings
	nstrings_globale = 0;
	lm.str.strDen = 0 ;
	//- Info to measurement
	MeasInfo ninfa;
	//- information needs to be passed onto measurement files
	ninfa.sliceprint = sliceprint;
	ninfa.idxprint = 0 ;
	ninfa.index = 0;
	ninfa.measdata = MEAS_NOTHING;
	ninfa.mask = spmask;

	//-maximum value of the theta angle in the simulation
	double maximumtheta = M_PI;
	lm.maxTheta = M_PI;

	//-output txt file
	FILE *file_samp ;
	file_samp = NULL;
	if (!restart_flag)
		file_samp = fopen("out/sample.txt","w+");
		else
		file_samp = fopen("out/sample.txt","a+"); // if restart append in file

		// //-point to print
		// size_t idxprint = 0 ;
		// //- z-coordinate of the slice that is printed as a 2D map
		// size_t sliceprint = 0 ;

	//-current conformal time
	double z_now ;
	//-current axion mass
	double axmass_now;
	//-grid spacing [obs?]
	double delta = axion->Delta();
	//-time step
	double dz;
	//-?? [obs?]
	double dzaux;
	//-llphys = LL or LL/z^2 in LAMBDA_Z2 mode
	double llphys = myCosmos.Lambda();

	//-set default dz [obs?]
	if (nSteps == 0)
		dz = 0.;
	else
		dz = (zFinl - zInit)/((double) nSteps);

	///-for reduced map Redondo version [obs?]
	int siN = (int) sizeN;
	if (endredmap > siN)
	{
		LogOut("[Error:1] Reduced map dimensions (%d) set to %d\n ", endredmap,siN);
		endredmap = siN;
	}

	if (siN%endredmap != 0 )
	{
		int schei =  siN/endredmap;
		endredmap = siN/schei;
		LogOut("[Error:2] Reduced map dimensions set to %d\n ", endredmap);
	}

	//-control flag to activate damping only once
	bool coD = true;

	//-number of iterations with 0 strings; used to switch to theta mode
	int strcount = 0;

	// obs?
	//- saves string data
	StringData rts ;
	//- obsolete
	double	strDen;

	//-LL at z=1, used for Z2 mode
	double LL1 = myCosmos.Lambda();

	LogOut("--------------------------------------------------\n");
	if (!restart_flag)
	LogOut("           STARTING COMPUTATION                   \n");
	else
	LogOut("           CONTINUE COMPUTATION                   \n");
	LogOut("--------------------------------------------------\n");

	//--------------------------------------------------
	//   THE TIME ITERATION LOOP
	//--------------------------------------------------

	//-block counter
	int counter = 0;
	//-used to label measurement files [~block, but with exceptions]
	int index = 0;

	//-stores energy [obs? move up]
	void *eRes;
	trackAlloc(&eRes, 128);
	memset(eRes, 0, 128);
	double *eR = static_cast<double *> (eRes);

	commSync();

	// if icstudy you might want to save this configuration
	if (icstudy)
	{
		if (fIndex == -1){
			if (prinoconfo%2 == 1 ){
				LogOut ("Dumping configuration %05d ...", index);
				writeConf(axion, index);
				LogOut ("Done!\n");
			}
			else{
				LogOut ("Bypass configuration writting!\n");
			}
		}
		else
			index = fIndex;
	}


	//-stores shift in real part of PQ and cPQ (conformal field) [obs? move up]
	double saskia = 0.0;
	double shiftz = 0.0;

//	--------------------------------------------------
//	--------------------------------------------------

	Folder munge(axion);

	if (cDev != DEV_GPU){
		LogOut ("Folding configuration ... ");
		munge(FOLD_ALL);
	}

	if (cDev != DEV_CPU){
		LogOut ("Transferring configuration to device\n");
		axion->transferDev(FIELD_MV);
	}
	LogOut ("Done! \n");

	if (dump > nSteps)
		dump = nSteps;

	int nLoops;

	if (dump == 0)
		nLoops = 0;
	else
		nLoops = (int)(nSteps/dump);


	LogOut("--------------------------------------------------\n");
	LogOut("           PARAMETERS  						                \n\n");
	LogOut("Length =  %2.2f\n", myCosmos.PhysSize());
	LogOut("nQCD   =  %2.2f\n", myCosmos.QcdExp());
	LogOut("indi3  =  %2.2f\n", myCosmos.Indi3());

	if (myCosmos.ZRestore() > myCosmos.ZThRes())
		LogOut("       =  0 in (%3.3f, %3.3f)   \n", myCosmos.ZThRes(), myCosmos.ZRestore());

	LogOut("N      =  %ld\n",   axion->Length());
	LogOut("Nz     =  %ld\n",   axion->Depth());
	LogOut("zGrid  =  %ld\n",   zGrid);
	LogOut("dx     =  %2.5f\n", axion->Delta());
	LogOut("dz     =  %2.2f/FREQ\n", wDz);

	if (LAMBDA_FIXED == axion->Lambda()){
		LogOut("LL     =  %.0f (msa=%1.2f-%1.2f in zInit,3)\n\n", myCosmos.Lambda(),
		sqrt(2.*myCosmos.Lambda())*zInit*axion->Delta(),sqrt(2.*myCosmos.Lambda())*3*axion->Delta());
	}
	else
		LogOut("LL     =  %1.3e/z^2 Set to make ms*delta =%.2f \n\n", myCosmos.Lambda(), axion->Msa());

	if	((myCosmos.QcdPot() & VQCD_TYPE) == VQCD_1)
		LogOut("VQCD1PQ1,shift,continuous theta  \n\n");
	else if	((myCosmos.QcdPot() & VQCD_TYPE) == VQCD_2)
		LogOut("VQCD2PQ1,no shift, continuous theta  \n\n");
	else if	((myCosmos.QcdPot() & VQCD_TYPE) == VQCD_1_PQ_2)
		LogOut("VQCD1PQ2,shift, continuous theta  \n\n");

	LogOut("Vqcd flag %d\n", myCosmos.QcdPot());
	LogOut("Damping %d gam = %f\n", myCosmos.QcdPot() & VQCD_DAMP, myCosmos.Gamma());
	LogOut("--------------------------------------------------\n\n");
	LogOut("           ESTIMATES\n\n");

	double z_doom2;

	if (myCosmos.Indi3()>0.0 && coSwitch2theta ){

	// if ((myCosmos.QcdPot() & VQCD_TYPE) == VQCD_1_PQ_2)
	// 	z_doom = pow(2.0*0.1588*axion->Msa()/axion->Delta(), 2./(myCosmos.QcdExp()+2.));
	// else
	// 	z_doom = pow(    0.1588*axion->Msa()/axion->Delta(), 2./(myCosmos.QcdExp()+2.));


	z_doom2 = findzdoom(axion);
	double z_axiq = pow(1.00/axion->Delta(), 2./(myCosmos.QcdExp()+2.));
	double z_NR   = pow(3.46/axion->Delta(), 2./(myCosmos.QcdExp()+2.));
	LogOut("z_doomsday %f(%f) \n", z_doom2,z_doom2);
	LogOut("z_axiquenc %f \n", z_axiq);
	LogOut("z_NR       %f \n", z_NR);
	;}
	LogOut("--------------------------------------------------\n\n");

	commSync();

	//--------------------------------------------------
	// prepropagator with relaxing strong damping
	//--------------------------------------------------
	// only if preprop and if z smaller or equal than zInit
	// When z>zInit, it is understood that prepropagation was done
	// NEW it takes the pregam value (if is > 0, otherwise gam )
	if (preprop && ((*axion->zV()) < zInit)) {
		//
		// LogOut("pppp Preprocessing ... z=%f->%f (VQCDTYPE %d, gam=%.2f pregam=%.2f dwgam=%.2f) \n\n",
		// 	(*axion->zV()), zInit, (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_DAMP_RHO, myCosmos.Gamma(),pregammo,dwgammo);
		LogOut("pppp Preprocessing ... z=%f->%f (VQCDTYPE %d, gam=%.2f pregam=%.2f dwgam=%.2f) \n\n",
			(*axion->zV()), zInit, (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_EVOL_RHO, myCosmos.Gamma(),pregammo,dwgammo);
		// gammo is reserved for long-time damping
		// use pregammo for prepropagation damping
		double gammo_save = myCosmos.Gamma();
		double *zaza = axion->zV();
		double strdensn;

		if (pregammo > 0)
			myCosmos.SetGamma(pregammo);

		// prepropagation is always with rho-damping
		LogOut("Prepropagator always with damping Vqcd flag %d\n", (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_DAMP_RHO);
		initPropagator (pType, axion, (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_DAMP_RHO);
		tunePropagator (axion);

		while (*zaza < zInit){
			dzaux = axion->dzSize(zInit)/2.;
			//myCosmos.SetGamma(gammo_save*pow(abs(1.0 - (*zaza)/zInit)/(1. - 1./prepcoe),1.5));

			// obs?
			printsample(file_samp, axion, myCosmos.Lambda(), idxprint, lm.str.strDen, lm.maxTheta);
			//printsample(file_samp, axion, myCosmos.Lambda(), idxprint, nstrings_globale, maximumtheta);
			if (icstudy){

				// lm = Measureme (axion, index, MEAS_STRING | MEAS_ENERGY | MEAS_2DMAP);
				ninfa.index=index;
				ninfa.measdata=MEAS_ALLBIN | MEAS_STRING | MEAS_STRINGMAP |
				MEAS_ENERGY | MEAS_2DMAP | MEAS_SPECTRUM ;
				lm = Measureme (axion, ninfa, MEAS_ALLBIN | MEAS_STRING | MEAS_STRINGMAP |
				MEAS_ENERGY | MEAS_2DMAP | MEAS_SPECTRUM);
				index++;

			} else{
				// LogOut("z %f (gamma %f)\n", *zaza, myCosmos.Gamma());
				LogOut(".");
			}
			propagate (axion, dzaux);
		}

		myCosmos.SetGamma(gammo_save);
	}
	LogOut("\n");

	// if icstudy you might want to save this configuration
	if (!icstudy)
	{
		if (fIndex == -1){
			if (prinoconfo%2 == 1 ){
				LogOut ("Dumping configuration (after prep) %05d ...", index);
				writeConf(axion, index);
				LogOut ("Done!\n");
			}
			else{
				LogOut ("Bypass configuration writting!\n");
			}
		}
		else
			index = fIndex;
	}

	if (!restart_flag && (fIndex == -1)){
		LogOut("First measurement file %d \n",index);
		ninfa.index=index;
		ninfa.measdata=MEAS_ALLBIN | MEAS_STRING | MEAS_STRINGMAP |
		MEAS_ENERGY | MEAS_2DMAP | MEAS_SPECTRUM;
		lm = Measureme (axion, ninfa, MEAS_ALLBIN | MEAS_STRING | MEAS_STRINGMAP |
		MEAS_ENERGY | MEAS_2DMAP | MEAS_SPECTRUM);
	}
	else{
		LogOut("last measurement file was %d \n",index);
	}

	LogOut("Running ...\n\n");
	LogOut("Init propagator Vqcd flag %d\n", myCosmos.QcdPot());
	initPropagator (pType, axion, myCosmos.QcdPot());
	tunePropagator (axion);

	LogOut ("Start redshift loop\n\n");
	fflush (stdout);

	start = std::chrono::high_resolution_clock::now();
	old = start;

	//--------------------------------------------------
	// THE TIME ITERATION LOOP
	//--------------------------------------------------

	for (int zloop = 0; zloop < nLoops; zloop++)
	{
		//--------------------------------------------------
		// THE TIME ITERATION SUB-LOOP
		//--------------------------------------------------

		index++;

		for (int zsubloop = 0; zsubloop < dump; zsubloop++)
		{
			z_now = (*axion->zV());
			old = std::chrono::high_resolution_clock::now();

			dzaux = axion->dzSize();

			// PROPAGATOR
			propagate (axion, dzaux);

			// SIMPLE OUTPUT CHECK
			z_now = (*axion->zV());
			//obs
			// printsample(file_samp, axion, myCosmos.Lambda(), idxprint, nstrings_globale, maximumtheta);
			printsample(file_samp, axion, myCosmos.Lambda(), idxprint, lm.str.strDen, lm.maxTheta);

			// CHECKS IF SAXION
			if ((axion->Field() == FIELD_SAXION ) && coSwitch2theta)
			{
				// IF FEW STRINGS COMPUTES THE NUMBER EVERY ITERATION [obs with damping]
				// if (nstrings_globale < 1000)
				if (lm.str.strDen < 1000 )
				{
					// rts = strings(axion);
					// nstrings_globale = rts.strDen ;
					// LogOut("  str extra check (string = %d, wall = %d)\n",rts.strDen, rts.wallDn);

					lm.str = strings(axion);
					// LogOut("  str extra check (string = %d, wall = %d)\n",lm.str.strDen, lm.str.wallDn);
				}

				// BEFORE UNPPHYSICAL DW DESTRUCTION, ACTIVATES DAMPING TO DAMP SMALL DW'S
				// DOMAIN WALL KILLER NUMBER
				//if (((*axion->zV()) > z_doom2*0.95) && (coD) && ((myCosmos.QcdPot() & VQCD_DAMP) != VQCD_NONE ))
				if (((*axion->zV()) > z_doom2*0.95) && (coD) && dwgammo > 0.)
				{
					myCosmos.SetGamma(dwgammo);
					LogOut("-----------------------------------------\n");
					LogOut("DAMPING ON (gam = %f, z ~ 0.95*z_doom %f)\n", myCosmos.Gamma(), 0.95*z_doom2);
					LogOut("-----------------------------------------\n");

					//initPropagator (pType, axion, myCosmos.QcdPot());   // old option, required --gam now it is activated with --pregam
					LogOut("Re-Init propagator Vqcd flag %d\n", (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_DAMP_RHO);
					initPropagator (pType, axion, (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_DAMP_RHO);
					coD = false ;
					// possible problem!! if gamma is needed later, as it is written pregammo will stay
				}

				// TRANSITION TO THETA COUNTER
				// obs?
				// if (nstrings_globale == 0) {
				// 	LogOut("  no st counter %d\n", strcount);
				// 	strcount++;
				// }
				if (lm.str.strDen == 0) {
					LogOut("  no st counter %d\n", strcount);
					strcount++;
				}

				// IF CONF_SAXNOISE we do not ever switch to theta to follow the evolution of saxion field
				if (smvarType != CONF_SAXNOISE)
					// if (nstrings_globale == 0 && strcount > safest0)
					if (lm.str.strDen == 0 && strcount > safest0 && coSwitch2theta)
					{
						if (axion->Lambda() == LAMBDA_Z2)
							llphys = myCosmos.Lambda()/(z_now*z_now);

						axmass_now = axion->AxionMass();
						saskia	   = axion->Saskia();
						shiftz	   = z_now * saskia;

						ninfa.index=index;
						ninfa.measdata=MEAS_ALLBIN | MEAS_STRING | MEAS_STRINGMAP |
						MEAS_ENERGY | MEAS_2DMAP | MEAS_SPECTRUM;
						lm = Measureme (axion, ninfa, MEAS_ALLBIN | MEAS_STRING | MEAS_STRINGMAP |
						MEAS_ENERGY | MEAS_2DMAP | MEAS_SPECTRUM);
						// lm = Measureme (axion, index, MEAS_2DMAP | MEAS_ENERGY | MEAS_ALLBIN ) ;
						index++;

						// if (!coD){
						// 	myCosmos.SetGamma(gammo);
						// }
						LogOut("--------------------------------------------------\n");
						LogOut(" TRANSITION TO THETA (z=%.4f)\n",z_now);
						LogOut(" shift = %f \n", saskia);

						cmplxToTheta (axion, shiftz);

						//IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS
						ninfa.index=index;
						ninfa.measdata=MEAS_ALLBIN | MEAS_STRING | MEAS_STRINGMAP |
						MEAS_ENERGY | MEAS_2DMAP | MEAS_SPECTRUM;
						lm = Measureme (axion, ninfa, MEAS_ALLBIN | MEAS_STRING | MEAS_STRINGMAP |
						MEAS_ENERGY | MEAS_2DMAP | MEAS_SPECTRUM);
						// Measureme (axion, index, MEAS_2DMAP | MEAS_ENERGY | MEAS_ALLBIN ) ;
						index++;
						LogOut("--------------------------------------------------\n");

						tunePropagator (axion);
					}
			}

			current = std::chrono::high_resolution_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);

			counter++;

			if ((*axion->zV()) > zFinl)
			{
				LogOut("zf reached! ENDING ... \n"); fflush(stdout);
				break;
			}
			checkTime(axion, index);
		} // ZSUBLOOP

		//--------------------------------------------------
		// PARTIAL ANALISIS
		//--------------------------------------------------

		if ((*axion->zV()) > zFinl)
		{
			// THE LAST MEASURE IS DONE AT THE END
			LogOut("zf reached! ENDING FINALLY... \n");
			break;
		}
		ninfa.index=index;
		ninfa.measdata=MEAS_ALLBIN | MEAS_STRING | MEAS_STRINGMAP |
		MEAS_ENERGY | MEAS_2DMAP | MEAS_SPECTRUM;
		lm = Measureme (axion, ninfa, MEAS_ALLBIN | MEAS_STRING | MEAS_STRINGMAP |
		MEAS_ENERGY | MEAS_2DMAP | MEAS_SPECTRUM);

		checkTime(axion, index);
	} // ZLOOP

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	LogOut("\n");
	LogOut("--------------------------------------------------\n");
	LogOut("              EVOLUTION FINISHED \n");
	LogOut("--------------------------------------------------\n");
	fflush(stdout);

	LogOut ("Final measurement file is: %05d \n", index);
	LogOut("Unfold ... ");
	munge(UNFOLD_ALL);
	LogOut("| ");

	//index++	; // LAST MEASUREMENT IS NOT PRINTED INSIDE THE LOOP, IT IS DONE HERE INSTEAD
	z_now = (*axion->zV());

	if (axion->Field() == FIELD_AXION)
	{
		MeasureType mesa = MEAS_2DMAP | MEAS_ALLBIN | MEAS_SPECTRUM | MEAS_ENERGY;

		if ((prinoconfo >= 2) && (wkb2z < 0)) {
			LogOut ("Dumping final configuration %05d ...", index);
			mesa = mesa | MEAS_3DMAP  ;
		}
		if (pconfinal)
			mesa = mesa | MEAS_ENERGY3DMAP ;

		if ( endredmap > 0)
			mesa = mesa | MEAS_REDENE3DMAP ;
		ninfa.index=index;
		ninfa.measdata=mesa;
		Measureme (axion, ninfa, mesa);


		//--------------------------------------------------
		// FINAL WKB
		//--------------------------------------------------

		if (wkb2z >= zFinl) {
			WKB wonka(axion, axion);

			LogOut ("WKBing %d (z=%.4f) to %d (%.4f) ... ", index, z_now, index+1, wkb2z);

			wonka(wkb2z);
			z_now = (*axion->zV());
			LogOut (" done! (z=%.4f)\n", z_now);

			index++;

			MeasureType mesa = MEAS_2DMAP | MEAS_ALLBIN | MEAS_SPECTRUM | MEAS_ENERGY;

			if (prinoconfo >= 2) {
				LogOut ("Dumping final WKBed configuration %05d ...", index);
				mesa = mesa | MEAS_3DMAP  ;
			}

			if (pconfinalwkb)
				mesa = mesa | MEAS_ENERGY3DMAP ;
			// 	writeEDens(axion);

			if ( endredmap > 0)
				mesa = mesa | MEAS_REDENE3DMAP ;

			ninfa.index=index;
			ninfa.measdata=mesa;
			Measureme (axion, ninfa, mesa);
		}
	}
	else
	{
		if ((prinoconfo >= 2)) {
			LogOut ("Dumping final Saxion onfiguration %05d ...", index);
			writeConf(axion, index);
			LogOut ("Done!\n");
		}
	}
	LogOut("z_final = %f\n", *axion->zV());
	LogOut("#_steps = %i\n", counter);
	LogOut("#_prints = %i\n", index);
	LogOut("Total time: %2.3f min\n", elapsed.count()*1.e-3/60.);
	LogOut("Total time: %2.3f h\n", elapsed.count()*1.e-3/3600.);

	trackFree(eRes);
	fclose(file_samp);

	delete axion;

	endAxions();

	return 0;
}

void printsample(FILE *fichero, Scalar *axion, double LLL, size_t idxprint, size_t nstrings_global, double maximumtheta)
{
	double z_now = (*axion->zV());
	double llphys = LLL;
	if (axion->Lambda() == LAMBDA_Z2)
		llphys = LLL/(z_now*z_now);

	size_t S0 = sizeN*sizeN ;
	if (commRank() == 0 && sPrec == FIELD_SINGLE) {
		if (axion->Field() == FIELD_SAXION) {
			double axmass_now = axion->AxionMass();
			double saskia = axion->Saskia();

			fprintf(fichero,"%f %f %f %f %f %f %f %ld %f %e\n", z_now, axmass_now, llphys,

			static_cast<complex<float> *> (axion->mCpu())[idxprint + S0].real(),
			static_cast<complex<float> *> (axion->mCpu())[idxprint + S0].imag(),
			static_cast<complex<float> *> (axion->vCpu())[idxprint].real(),
			static_cast<complex<float> *> (axion->vCpu())[idxprint].imag(),
			nstrings_global, maximumtheta, saskia);
		} else {
			fprintf(fichero,"%f %f %f %f %f\n", z_now, axion->AxionMass(),
			static_cast<float *> (axion->mCpu())[idxprint + S0],
			static_cast<float *> (axion->vCpu())[idxprint], maximumtheta);
		}

		fflush(fichero);
	}
}

double findzdoom(Scalar *axion)
{
	double ct = zInit ;
	double DWfun;
	double meas ;
	while (meas < 0.001)
	{
		DWfun = 40*axion->AxionMassSq(ct)/(2.0*axion->BckGnd()->Lambda()) ;
		if (axion->Lambda() == LAMBDA_Z2)
			DWfun *= ct*ct;
		meas = DWfun - 1 ;
		ct += 0.001 ;
	}
	LogOut("Real z_doom %f ", ct );
	return ct ;
}

void	checkTime (Scalar *axion, int index) {
	auto	cTime = Timer();
	int	cSize = commSize();
	int	flag  = 0;
	std::vector<int> allFlags(cSize);

	bool	done  = false;

	if (wTime <= cTime)
		flag = 1;

	FILE *capa = nullptr;
	if (!((capa  = fopen("./stop", "r")) == nullptr)){
		flag = 2;
		fclose (capa);
	}


	MPI_Allgather(&flag, 1, MPI_INT, allFlags.data(), 1, MPI_INT, MPI_COMM_WORLD);

	for (const int &val : allFlags) {
		if (val > 0) {
			done = true;
			flag = val;
			break;
		}
	}

	if (done) {
		if (cDev == DEV_GPU)
			axion->transferCpu(FIELD_MV);
		if (flag ==2){
			LogMsg(VERB_NORMAL, "[checkTime %d] stop file detected! stopping ... ",index);
			LogOut ("Interrupted manually with stop file ...");
		}
		if (flag ==1){
			LogMsg(VERB_NORMAL, "[checkTime %d] Walltime reached ",index);
			LogOut ("Walltime reached, dumping configuration...");
		}

		writeConf(axion, index, 1);
		LogOut ("Done!\n");

		LogOut("z Final = %f\n", *axion->zV());
		LogOut("nPrints = %i\n", index);

		LogOut("Total time: %2.3f min\n", cTime*1.e-6/60.);
		LogOut("Total time: %2.3f h\n", cTime*1.e-6/3600.);
//		trackFree(eRes);	FIXME!!!!

		delete axion;

		endAxions();

		exit(0);
	}
}
