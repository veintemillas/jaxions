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
#include "axiton/tracker.h"

#include <iostream>
#include <fstream>
#include <string>
#include <map>

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_device_runtime_api.h>
#endif


using namespace std;
using namespace AxionWKB;


// vaxions3d definitions

void    printsample  (FILE *fichero, Scalar *axion, size_t idxprint, size_t nstrings_global, double maximumtheta);
double  findzdoom(Scalar *axion);
void    checkTime (Scalar *axion, int index);
void    printposter (Scalar *axion);
void    readmeasfile (Scalar *axion, DumpType *dumpmode_p, MeasFileParms *measfilepar, MeasInfo *ninfa);
void    mysplit (std::string *str, std::vector<std::string> *result);
int     readheader (std::string *str, std::vector<int> *perm);
int     readmeasline2 (std::string *str, double *ctime, std::vector<int> *lint, std::vector<int> *perm, int n_max);
void 		loadmeasfromlist(MeasFileParms *mfp, MeasInfo *info, int i_meas);
size_t  unfoldidx(size_t idx, Scalar *axion);
//-point to print
size_t idxprint = 66065 ;
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
	LogOut("\n--               VAXION 3D!                    --\n");
	LogOut("\n-------------------------------------------------\n\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	start = std::chrono::high_resolution_clock::now();

	//-grids
	Scalar *axion;

	if ((fIndex == -1) && (myCosmos.ICData().cType == CONF_NONE) && (!restart_flag))
		LogOut("Error: Neither initial conditions nor configuration to be loaded selected. Empty field.\n");
	else
	{
		if ( (fIndex == -1) && !restart_flag)
		{
			LogOut("Generating scalar ... ");
			axion = new Scalar (&myCosmos, sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fTypeP, lType, myCosmos.ICData().Nghost);
			LogOut("Done! \n");
		}
		else
		{
			LogOut("Reading initial conditions from file ... ");
			readConf(&myCosmos, &axion, fIndex, restart_flag);

			// temporary test!! FIX ME! allows to kick the initial configuration
			if ( !(myCosmos.ICData().kickalpha == 0.0) )
				scaleField (axion, FIELD_V, 1.0+myCosmos.ICData().kickalpha);

			if (axion == NULL)
			{
				LogOut ("Error reading HDF5 file\n");
				exit (0);
			}
			LogOut("Done! \n");
		}
	}
	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
	LogOut("ICtime %f min\n",elapsed.count()*1.e-3/60.);


	//-------------------------------------------------
	// PRINT SUMMARY
	//-------------------------------------------------

	printposter(axion);

	//--------------------------------------------------
	// USEFUL VARIABLES
	//--------------------------------------------------

	//-output txt file
	char out2Name[2048];
	sprintf (out2Name, "%s/../sample.txt", outDir);
	FILE *file_samp ;
	file_samp = NULL;
	if (!restart_flag){
		file_samp = fopen(out2Name,"w+");
	} else{
		file_samp = fopen(out2Name,"a+"); // if restart append in file
	}


  //- time when axion mass^2 is 1/40 of saxion mass^2
	double 	z_doom2 = findzdoom(axion);
	//time intervac
	double dzaux;
	//-llphys = LL or LL/z^2 in LAMBDA_Z2 mode
	double llphys = myCosmos.Lambda();
	///-for reduced map Redondo version [obs?]

	//-control flag to activate damping only once
	bool coD = true;
	//-number of iterations with 0 strings; used to switch to theta mode
	int strcount = 0;

	//--------------------------------------------------
	// MEASUREMENTS, DUMP
	//--------------------------------------------------

	//- Measurement
	MeasData lm;
	//- number of plaquetes pierced by strings
	lm.str.strDen = 0 ;
	//- Info to measurement
	MeasInfo ninfa = deninfa;
	// //- information needs to be passed onto measurement files

	//-maximum value of the theta angle in the simulation
	double maximumtheta = M_PI;
	lm.maxTheta = M_PI;

	// dump decision function
	DumpType dumpmode = DUMP_EVERYN ;
	int i_meas = 0;
	bool measrightnow = false;
	double mesi;
	int meastype ;
	MeasFileParms measfilepar;

	readmeasfile (axion, &dumpmode, &measfilepar, &ninfa);

	i_meas=0;

	LogOut("\n");

	LogOut("--------------------------------------------------\n");
	if (!restart_flag)
	LogOut("           STARTING COMPUTATION                   \n");
	else
	LogOut("           CONTINUE COMPUTATION                   \n");
	LogOut("--------------------------------------------------\n");

	//-block counter
	int counter = 0;
	//-used to label measurement files [~block, but with exceptions]
	int index ;

	commSync();

	if (cDev != DEV_CPU){
		LogOut ("Transferring configuration to device\n");
		axion->transferDev(FIELD_MV);
	}
	LogOut ("Done! \n");


	commSync();


	//--------------------------------------------------
	// INITIAL MEASUREMENT
	//--------------------------------------------------


	ninfa.measdata |= MEAS_NNSPEC ;
	if (!restart_flag && (fIndex == -1)){
		index = fIndex2;
		LogOut("First measurement file %d \n",index);
		ninfa.index=index;
		if (ninfa.printconf & PRINTCONF_INITIAL)
			ninfa.measdata |= MEAS_3DMAP ;
		lm = Measureme (axion, ninfa);
	}
	else if (restart_flag)	{
		index = fIndex -1 ;
		LogOut("last measurement file was %d \n",index);
	}
	else if (!restart_flag && (fIndex > -1)){
		index = fIndex;
		LogOut("First measurement from read file %d \n",index);
		ninfa.index=index;
		lm = Measureme (axion, ninfa);
	}

	index++;
	if ( (dumpmode == DUMP_FROMLIST) ){
			LogOut("time %f and %d-measurement %lf\n",*axion->zV(),i_meas,measfilepar.ct[i_meas]);
		if (abs(1.0 -(*axion->zV())/measfilepar.ct[i_meas])<0.0001){
				i_meas++;
				LogOut("i_meas++ initial conditions coincided with 1st measurement\n");
		}

	}

	// SIMPLE OUTPUT CHECK
	printsample(file_samp, axion, idxprint, lm.str.strDen, lm.maxTheta);

	//--------------------------------------------------
	// Axiton TRACKER (if THETA)
	//--------------------------------------------------

		initTracker(axion);
		searchAxitons();

	//--------------------------------------------------
	// TIME ITERATION LOOP
	//--------------------------------------------------

	LogOut("Running ...\n\n");
	LogOut("Init propagator Vqcd flag %d\n", myCosmos.QcdPot());
	if (Nng>0)
		LogOut(" Laplacian with (Nng=%d) neighbours",Nng);
	LogOut("\n");
	initPropagator (pType, axion, myCosmos.QcdPot(),Nng);
	tunePropagator (axion);


	LogOut ("Start redshift loop\n\n");
	for (int iz = 0; iz < nSteps; iz++)
	{

		// time step
		// if ((axion->Field() == FIELD_AXION ) || (axion->Field() == FIELD_SAXION ))
		//  dzaux = 0.0;
		//  else
		 dzaux = (uwDz) ? axion->dzSize() : (zFinl-zInit)/nSteps ;

		//will we dump? and when?
		switch(dumpmode)
			{
				case DUMP_EVERYN:
				if (!(iz%dump)){
					measrightnow = true;
					// meastype = ninfa.measdata;
				}
				break;

				case DUMP_FROMLIST:
				if (*axion->zV() > measfilepar.ct[i_meas])
				{
					for (int i =i_meas; i< measfilepar.ct.size(); i++){
						if (*axion->zV() > measfilepar.ct[i])
							i_meas++;
							LogMsg(VERB_NORMAL,"[VAX] Time jumped over measurement! jumping once!");
					}
				}

				if ( (*axion->zV())+dzaux >= measfilepar.ct[i_meas] && (*axion->zV()) < measfilepar.ct[i_meas]){
					LogMsg(VERB_NORMAL,"[VAX] dct adjusted from %e",dzaux);
					dzaux = measfilepar.ct[i_meas] - (*axion->zV());
					LogMsg(VERB_NORMAL,"                   to   %e",dzaux);
					measrightnow = true;
					loadmeasfromlist(&measfilepar, &ninfa, i_meas);
					defaultmeasType = ninfa.measdata;
					// actually, if this is the last measurement, do not measure!
					if ( (i_meas == measfilepar.ct.size()-1) ){
						LogMsg(VERB_NORMAL,"[VAX] last measurement, do not measure and pass END!",dzaux);
						measrightnow = false;
					}
				}

				break;
			}

			LogFlush();
			// PROPAGATOR
			propagate (axion, dzaux);
			counter++;

 			LogFlush();

			// SIMPLE OUTPUT CHECK
			printsample(file_samp, axion, idxprint, lm.str.strDen, lm.maxTheta);

			// CHECKS IF SAXION
			if ((axion->Field() == FIELD_SAXION ) && coSwitch2theta)
			{
				if (lm.str.strDen < 1000 )
					lm.str = strings(axion);

				// BEFORE UNPPHYSICAL DW DESTRUCTION, ACTIVATES DAMPING TO DAMP SMALL DW'S
				if ((z_doom2 > 0.0) && ((*axion->zV()) > z_doom2*0.95) && (coD) && dwgammo > 0.)
				{
					myCosmos.SetGamma(dwgammo);
					LogOut("-----------------------------------------\n");
					LogOut("DAMPING ON (gam = %f, z ~ 0.95*z_doom %f)\n", myCosmos.Gamma(), 0.95*z_doom2);
					LogOut("-----------------------------------------\n");

					//initPropagator (pType, axion, myCosmos.QcdPot());   // old option, required --gam now it is activated with --pregam
					LogOut("Re-Init propagator Vqcd flag %d\n", (myCosmos.QcdPot() & V_TYPE) | V_DAMP_RHO);
					initPropagator (pType, axion, (myCosmos.QcdPot() & V_TYPE) | V_DAMP_RHO, Nng);
					coD = false ;
					// possible problem!! if gamma is needed later, as it is written pregammo will stay
				}

				// TRANSITION TO THETA COUNTER
				if (lm.str.strDen == 0) {
					LogOut("  no st counter %d/%d\n", strcount,safest0);
					strcount++;
				}

				// IF CONF_SAXNOISE we do not ever switch to theta to follow the evolution of saxion field
				if (smvarType != CONF_SAXNOISE)
					if (lm.str.strDen == 0 && strcount > safest0 && coSwitch2theta)
					{

							// Measurement before switching to theta
							// customize?
							ninfa.index=index;
							ninfa.measdata = rho2thetameasType;
							ninfa.maty     = deninfa.maty;
							ninfa.mask     = deninfa.mask;
							if (ninfa.printconf & PRINTCONF_PRE2THETA)
								ninfa.measdata |= MEAS_3DMAP  ;
							lm = Measureme (axion, ninfa);
							index++;
							if (ninfa.printconf & PRINTCONF_PRE2THETA)
								ninfa.measdata ^= MEAS_3DMAP  ;

						LogOut("--------------------------------------------------\n");
						LogOut(" TRANSITION TO THETA (z=%.4f R=%.4f)\n",(*axion->zV()), (*axion->RV()));
						LogOut(" shift = %f \n", axion->Saskia());

						double shiftz = axion->Saskia()*(*axion->RV());
						cmplxToTheta (axion, shiftz);
						// NAXIONTEMP

						// Measurement after switching to theta
						ninfa.index=index;
						if (ninfa.printconf & PRINTCONF_POST2THETA)
							ninfa.measdata |= MEAS_3DMAP  ;
						lm = Measureme (axion, ninfa);
						index++;
						LogOut("--------------------------------------------------\n");

						tunePropagator (axion);

						/* Axiton tracker */
						initTracker(axion);
						searchAxitons();
					}
			}

			// Break the loop when we are done
			if ( (*axion->zV()) >= zFinl ){
				LogOut("zf %f reached! ENDING ... \n", zFinl); fflush(stdout);
				break;
			}
			if ( abs((*axion->zV())-zFinl) < 1.0e-10 ){
				LogOut("zf %f approximately reached! ENDING ... \n", zFinl); fflush(stdout);
				break;
			}

			/* read axitons?*/
			readAxitons();

			// Partial analysis
			if(measrightnow){

				/* checks for more axitons */

				searchAxitons ();

				ninfa.index=index;
				// in case theta transitioned, the meas was saved as the default
				ninfa.measdata = defaultmeasType;
				ninfa.cTimesec = (double) Timer()*1.0e-6;
				ninfa.propstep = iz;
				// if (axion->Field() == FIELD_PAXION )
				// 		ninfa.measdata |= MEAS_3DMAP;

				lm = Measureme (axion, ninfa);
				index++;
				i_meas++ ;
				//reset flag
				measrightnow = false;
			}
			// after every measurement we check walltime > need update
			checkTime(axion, index);

	} // time loop's over

	LogOut("\n");
	LogOut("--------------------------------------------------\n");
	LogOut("              EVOLUTION FINISHED \n");
	LogOut("--------------------------------------------------\n");
	fflush(stdout);

	LogOut ("Final measurement file is: %05d \n", index);

	//index++	; // LAST MEASUREMENT IS NOT PRINTED INSIDE THE LOOP, IT IS DONE HERE INSTEAD
	// migth be a problem here... double measurement?

	MeasureType mesa = defaultmeasType;

	if ((ninfa.printconf & PRINTCONF_FINAL) ) {
		mesa = mesa | MEAS_3DMAP  ;
	}
	if (pconfinal)
		mesa = mesa | MEAS_ENERGY3DMAP ;

	if ( endredmap > 0)
		mesa = mesa | MEAS_REDENE3DMAP ;
	ninfa.index=index;
	ninfa.measdata=mesa;
	Measureme (axion, ninfa);

	if (axion->Field() == FIELD_AXION)
	{
		//--------------------------------------------------
		// FINAL WKB
		//--------------------------------------------------

		if (wkb2z >= zFinl) {
			WKB wonka(axion, axion);

			LogOut ("WKBing %d (z=%.4f) to %d (%.4f) ... ", index, 	(*axion->zV()), index+1, wkb2z);

			wonka(wkb2z);
			LogOut (" done! (z=%.4f)\n", (*axion->zV()));

			index++;

			/* last measurement after WKB */
			MeasureType mesa = defaultmeasType;

			if (ninfa.printconf & PRINTCONF_WKB) {
				LogOut ("Dumping final WKBed configuration %05d ...", index);
				mesa = mesa | MEAS_3DMAP  ;
			}

			if (pconfinalwkb)
				mesa = mesa | MEAS_ENERGY3DMAP ;
			// 	writeEDens(axion);

			if ( endredmap > 0 )
				mesa = mesa | MEAS_REDENE3DMAP ;

			if ( endredmapwkb > 0 ){
				mesa = mesa | MEAS_REDENE3DMAP ;
				ninfa.redmap=endredmapwkb;
			}

			ninfa.index=index;
			ninfa.measdata=mesa;
			Measureme (axion, ninfa);
		}
	}

	printAxitons();

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	LogOut("z_final = %f\n", *axion->zV());
	LogOut("#_steps = %i\n", counter);
	LogOut("#_prints = %i\n", index);
	LogOut("Total time: %2.3f min\n", elapsed.count()*1.e-3/60.);
	LogOut("Total time: %2.3f h\n", elapsed.count()*1.e-3/3600.);


	fclose(file_samp);

	delete axion;

	endAxions();

	return 0;
}












void printsample(FILE *fichero, Scalar *axion,  size_t idxprint, size_t nstrings_global, double maximumtheta)
{
	double z_now = (*axion->zV());
	double R_now = (*axion->RV());
	double llphys = axion->LambdaP();

	/* unfold if needed */
	size_t idxp = unfoldidx(idxprint, axion);
	size_t S0 = sizeN*sizeN ;
	if (commRank() == 0){
		if (sPrec == FIELD_SINGLE) {
			if (axion->Field() == FIELD_SAXION) {
				double axmass_now = axion->AxionMass();
				double saskia = axion->Saskia();
				float buff[4];
#ifdef USE_GPU
				if (axion->Device() == DEV_GPU) {
					cudaMemcpy(buff, &(static_cast<float*>(axion->mGpuStart())[2*idxprint]),2*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(&(buff[2]), &(static_cast<float*>(axion->vGpu())[2*idxprint]),2*sizeof(float),cudaMemcpyDeviceToHost);
				} else
#endif
				{
					memcpy(buff,&(static_cast<float*> (axion->mStart())[2*idxp]),2*sizeof(float));
					memcpy(&(buff[2]),&(static_cast<float*> (axion->vStart())[2*idxp]),2*sizeof(float));
				}
				fprintf(fichero,"%f %f %f %f %f %f %f %f %ld %f %e\n", z_now, R_now, axmass_now, llphys,
				buff[0], buff[1], buff[2], buff[3],
				nstrings_global, maximumtheta, saskia);
			} else {
				fprintf(fichero,"%f %f %f %f %f %f\n", z_now, R_now, axion->AxionMass(),
				static_cast<float *> (axion->mStart())[idxp],
				static_cast<float *> (axion->vStart())[idxp], maximumtheta);
			}
			fflush(fichero);
		} else if (sPrec == FIELD_DOUBLE){
			if (axion->Field() == FIELD_SAXION) {
				double axmass_now = axion->AxionMass();
				double saskia = axion->Saskia();

				fprintf(fichero,"%f %f %f %f %f %f %f %f %ld %f %e\n", z_now, R_now, axmass_now, llphys,
				static_cast<complex<double> *> (axion->mStart())[idxp].real(),
				static_cast<complex<double> *> (axion->mStart())[idxp].imag(),
				static_cast<complex<double> *> (axion->vStart())[idxp].real(),
				static_cast<complex<double> *> (axion->vStart())[idxp].imag(),
				nstrings_global, maximumtheta, saskia);
			} else {
				fprintf(fichero,"%f %f %f %f %f %f\n", z_now, R_now, axion->AxionMass(),
				static_cast<double *> (axion->mStart())[idxp],
				static_cast<double *> (axion->vStart())[idxp], maximumtheta);
			}
		}
	}
}


double findzdoom(Scalar *axion)
{

	double fff = axion->BckGnd()->Frw();

	if (axion->BckGnd()->Indi3() > 0.0 && (fff > 0.0)){
	double ct = *axion->zV();
	double DWfun0 = 40*axion->AxionMassSq(ct)/(2.0*axion->BckGnd()->LambdaP(ct));
	double DWfun = DWfun0;
	double ct2 = ct*1.1;
	double DWfun2 = 40*axion->AxionMassSq(ct2)/(2.0*axion->BckGnd()->LambdaP(ct2));
	double k = 1;
	double a = 1;
	double meas = std::abs(DWfun2 - 1);;
	LogMsg(VERB_NORMAL,"[VAX findzdoom] frw %f indi3 %f ct %e ", fff, axion->BckGnd()->Indi3(), ct );LogFlush();
	while (meas > 0.001)
	{

		/* Assume power law
		DWfun ~ k ct^a
		then DWfun2/DWfun = (ct2/ct)^a
		a = log(DWfun2/DWfun)/log(ct2/ct)
		k = DWfun2/ct2^a
		DWfun = 1 -> ct3 = 1/k^{1/a}, ct2 = ct
		*/
		a = std::log(DWfun2/DWfun)/std::log(ct2/ct);
		k = DWfun2/std::pow(ct2,a);
		// LogOut("ct %e DWfun %e ct2 %e DWfun2 %e meas %e k %e a %e\n", ct, DWfun, ct2, DWfun2, meas, k ,a );
		if ((a == 0) && (DWfun2 > DWfun0) ){
			LogMsg(VERB_PARANOID,"[VAX findzdoom] flat slope between ct %e ct2 %e", ct, ct2 );
			if (DWfun2 > 1) {
				LogMsg(VERB_PARANOID,"[VAX findzdoom] Jump back!");
				ct  = ct2;
				ct2 = std::sqrt(*axion->zV()*ct2);
			}
			else {
				LogMsg(VERB_PARANOID,"[VAX findzdoom] DWfun will never reach 1");
				return INFINITY;
			}
		} else {
			ct  = ct2;
			ct2 = std::pow(k,-1./a);
		}
		DWfun = DWfun2;
		DWfun2 = 40*axion->AxionMassSq(ct2)/(2.0*axion->BckGnd()->LambdaP(ct2));
		meas = std::abs(DWfun2 - 1);
		LogMsg(VERB_PARANOID,"ct2 %e DWfun2 %e meas %e k %e a %e", ct2, DWfun2, meas, k ,a );
	}
	//LogOut("ct2 %e DWfun2 %e meas %e k %e a %e\n", ct2, DWfun2, meas, k ,a );
	LogMsg(VERB_NORMAL,"[VAX findzdoom] Real z_doom %f ", ct2 );LogFlush();
	return ct2 ;
} else {
	return -1 ; }
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

	FILE *cape = nullptr;
	if (!((cape  = fopen("./abort", "r")) == nullptr)){
		flag = 3;
		fclose (cape);
	}

	FILE *capo = nullptr;
	if (!((capo  = fopen("./savejaxconf", "r")) == nullptr)){
		flag = 4;
		fclose (capo);
		if( remove( "./savejaxconf" ) != 0 ){
			LogOut("savejaxconf file cannot be deleted. Danger!\n");
		}
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
		if (flag ==1){
			LogMsg(VERB_NORMAL, "[VAX checkTime %d] Walltime reached ",index);
			LogOut ("Walltime reached, dumping configuration...");
			writeConf(axion, index, 1);
			LogOut ("Done!\n");

			LogOut("z Final = %f\n", *axion->zV());
			LogOut("nPrints = %i\n", index);

			LogOut("Total time: %2.3f min\n", cTime*1.e-6/60.);
			LogOut("Total time: %2.3f h\n", cTime*1.e-6/3600.);

			delete axion;

			endAxions();

			exit(0);

		}
		if (flag ==2){
			LogMsg(VERB_NORMAL, "[VAX checkTime %d] stop file detected! stopping ... ",index);
			LogOut ("Interrupted manually with stop file ...");
			writeConf(axion, index, 1);
			LogOut ("Done!\n");

			LogOut("z Final = %f\n", *axion->zV());
			LogOut("nPrints = %i\n", index);

			LogOut("Total time: %2.3f min\n", cTime*1.e-6/60.);
			LogOut("Total time: %2.3f h\n", cTime*1.e-6/3600.);

			delete axion;

			endAxions();

			exit(0);

		}
		if (flag == 3){
			LogMsg(VERB_NORMAL, "[VAX checkTime %d] abort file detected! aborting! ",index);
			LogOut ("Aborting ...");
			delete axion;

			endAxions();

			exit(0);

		}
		if (flag ==4){
			LogMsg(VERB_NORMAL, "[VAX checkTime %d] save file detected! saving ... ",index);
			writeConf(axion, index);
			commSync();
			remove( "./save" );
		}

	}
}

void printposter(Scalar *axion)
{
	LogOut("--------------------------------------------------\n");
	LogOut("        SIMULATION (%d x %d x %d) ", axion->Length(), axion->Length(), axion->Depth());
	if (zGrid>1)
		LogOut(" x %d \n\n", zGrid);
	else
		LogOut("      \n\n");

	LogOut("Box Length [1/R1H1]      =  %2.2f\n", axion->BckGnd()->PhysSize());
	LogOut("dx                       =  %2.5f\n", axion->Delta());
	LogOut("dz                       =  %2.2f/FREQ\n\n", wDz);

	LogOut("FRW scale factor (R)     =  z^%1.2f \n\n", axion->BckGnd()->Frw());

	LogOut("Saxion self-cp. Lambda\n");
	if (LAMBDA_FIXED == axion->LambdaT()){
	LogOut("LL                       =  %.0f \n        (msa=%1.2f-%1.2f in zInit,3)\n\n", axion->BckGnd()->Lambda(),
		sqrt(2.0 * axion->BckGnd()->Lambda())*zInit*axion->Delta(),sqrt(2.0 * axion->BckGnd()->Lambda())*3*axion->Delta());
	}
	else{
	LogOut("LL                       =  %1.3e/z^2\n", axion->BckGnd()->Lambda());
	LogOut("msa                      =  %.2f \n\n", axion->Msa());
	}
	if (axion->BckGnd()->Indi3() > 0.0){
	LogOut("Axion mass^2 [H1^2]      = indi3 x R^nQCD \n");
	LogOut("indi3                    =  %2.2f\n", axion->BckGnd()->Indi3());
	LogOut("nQCD                     =  %2.2f\n", axion->BckGnd()->QcdExp());
	if (axion->BckGnd()->ZRestore() > axion->BckGnd()->ZThRes())
		LogOut("                       =  0 in (%.3e, %.3e) \n", axion->BckGnd()->ZThRes(), axion->BckGnd()->ZRestore());

		switch(axion->BckGnd()->QcdPot() & V_QCD){
			case V_QCD0:
			case V_NONE:
				LogOut("V_QCD0, massless axions\n");
				break;
				case V_QCD1:
					LogOut("V_QCD1, shift\n");
					break;
					case V_QCDV:
						LogOut("V_QCDV variant, no shift\n");
						break;
						case V_QCD2:
							LogOut("V_QCD2, N=2, domain wall problem, shift\n");
							break;
							case V_QCDL:
								LogOut("V_QCDL, quadratic potential (in axion mode) VQCD1  (in saxion mode) shift\n");
								break;
								case V_QCDC:
									LogOut("V_QCDC, 1-cos potential, 1/0 problem, no shift\n");
									break;
		}
		switch(axion->BckGnd()->QcdPot() & V_PQ){
			case V_NONE:
				LogOut("V_PQ0, massless saxions!\n");
				break;
				case V_PQ1:
					LogOut("V_PQ1,mexican hat!\n");
					break;
					case V_PQ2:
						LogOut("V_PQ2, top hat !\n");
						break;
		}

		LogOut("Vqcd flag %d\n", axion->BckGnd()->QcdPot());
		LogOut("Damping flag %d 		     \n", axion->BckGnd()->QcdPot() & V_DAMP);
		LogOut("gam                    = %lf \n", axion->BckGnd()->Gamma());
		LogOut("--------------------------------------------------\n\n");
		LogOut("           TIME SCALES ESTIMATES\n\n");

		double 	z_doom2 = findzdoom(axion);
		// if (myCosmos.Indi3()>0.0 && coSwitch2theta ){

		double z_axiq = pow(1.00/axion->Delta(), 2./(axion->BckGnd()->QcdExp()+2.));
		double z_NR   = pow(3.46/axion->Delta(), 2./(axion->BckGnd()->QcdExp()+2.));
		LogOut("mA^2/mS^2 = 1/40  at ctime %lf \n", z_doom2);
		LogOut("mA^2 = mS^2       at ctime %lf \n", z_axiq);
		LogOut("Fastest axions NR at ctime %lf \n", z_NR);
		;
		LogOut("--------------------------------------------------\n\n");
	} else {
		LogOut("Massless axion!!!\n\n");
	}
}


void mysplit (std::string *str, std::vector<std::string> *result)
{
	std::istringstream iss(*str);
	(*result).clear();
	for(std::string s; iss >> s; )
			(*result).push_back(s);
}

int readheader (std::string *str, std::vector<int> *perm)
{
	int n = 0;
	std::vector<std::string> result;
	mysplit(str, &result);

	/* build the permutation to have the order
	0 cout
	1 meas
	2 map
	3 mask */
	for (int k =0; k < result.size(); k++)
		{
			if (result[k] == "ct")
				{(*perm)[k] = 0;n++;}
			else if (result[k] == "meas")
			{(*perm)[k] = 1;n++;}
			else if (result[k] == "map")
			{(*perm)[k] = 2;n++;}
			else if (result[k] == "mask")
			{(*perm)[k] = 3;n++;}
			else if (result[k] == "kgv")
			{(*perm)[k] = 4;n++;}
		}

	return n;
}

int readmeasline2 (std::string *str, double *ctime, std::vector<int> *lint, std::vector<int> *perm, int n_max)
{
	/* reads str and uses the permutation to fill the lint list */
	std::vector<std::string> result;
	mysplit(str, &result);
	(*lint).clear();
	(*lint) = {0, 0, 0, 0};
	int kmax = result.size();
	LogMsg(VERB_PARANOID,"kmax %d n_max %d",kmax,n_max);
	if (kmax > n_max)
		kmax = n_max;

	for (int k = 0; k < kmax; k++)
	{
		if ( (*perm)[k] == 0)
			*ctime = std::stof( result[k]);
		else
		 	(*lint)[(*perm)[k]-1] = std::stoi(result[k]);
	}
	/* return discarded */
	return result.size()-n_max;
}

void readmeasfile (Scalar *axion, DumpType *dumpmode_p, MeasFileParms *measfilepar, MeasInfo *ninfa)
{
		FILE *cacheFile = nullptr;
		if (((cacheFile  = fopen("./measfile.dat", "r")) == nullptr)){
			LogMsg(VERB_NORMAL,"[VAX] No measfile.dat ! Use linear dump mode by default");
		}
		else
		{
			LogOut("Reading measurement files from measfile.dat \n");

			*dumpmode_p = DUMP_FROMLIST;
			LogMsg(VERB_NORMAL,"[VAX] Reading measurement files from list");
			double mesi;
			int meastype ;
			int maptype = 0 ;
			int masktype = 0;
			int kgvtype = 0;
			int i_meas = 0;

			int n_header;

			std::ifstream file("./measfile.dat");
		  std::string str;
			std::vector<int> lint;
			std::vector<int> perm = {0,1,2,3,4};

			/* First line header, create the map */
			std::getline(file, str);
			n_header = readheader (&str, &perm);
			if (n_header==0){
				LogMsg(VERB_NORMAL,"[VAX] old measfile.dat format");
				file.seekg(ios::beg);
				n_header = 2;
			} else {
				LogMsg(VERB_NORMAL,"[VAX] header: %s", str.c_str());
				}
			LogMsg(VERB_NORMAL,"[VAX] perm %d %d %d %d %d", perm[0], perm[1], perm[2], perm[3], perm[4]);

			/* Now read the file */
			while (std::getline(file, str)) {
				/* */
				int n_disc = readmeasline2 (&str, &mesi, &lint, &perm, n_header);

				/* negative int is the signal for a default measurement type */
				if (lint[0] < 0)
					lint[0] = defaultmeasType;

				/* maps  passed in the commandline will be added to all measurements */
				lint[1] |= deninfa.maty;
				/* masks passed in the commandline will be added to all measurements */
				lint[2] |= deninfa.mask;
				/* nRuntype passed in the commandline will be added to all measurements */
				lint[3] |= deninfa.nrt;

				if (mesi < *axion->zV()){
					LogMsg(VERB_NORMAL,"[VAX] read z=%f < current time (z=%f) > DISCARDED",mesi,*axion->zV());
				}
				else {
					(*measfilepar).ct.push_back(mesi);
					(*measfilepar).meas.push_back(lint[0]);
					(*measfilepar).map.push_back(lint[1]);
					(*measfilepar).mask.push_back(lint[2]);
					(*measfilepar).nrt.push_back(lint[3]);
					LogMsg(VERB_NORMAL,"[VAX] i_meas=%d read z=%f meas=%d map=%d mask=%d nrt=%d",
					i_meas, (*measfilepar).ct[i_meas], (*measfilepar).meas[i_meas], (*measfilepar).map[i_meas], (*measfilepar).mask[i_meas], (*measfilepar).nrt[i_meas]);
					i_meas++ ;
				}
			}

			for (int i =i_meas-1; i>0;i--)
				if ((*measfilepar).ct[i] == (*measfilepar).ct[i-1]){
					LogMsg(VERB_NORMAL,"[VAX] merge %d %d at t %f with %d %d > %d", i, i-1,
						(*measfilepar).ct[i], (*measfilepar).meas[i], (*measfilepar).meas[i-1],
							(*measfilepar).meas[i]| (*measfilepar).meas[i-0]);

					(*measfilepar).meas[i-1] |= (*measfilepar).meas[i];
					(*measfilepar).ct.erase( (*measfilepar).ct.begin()+i);
					(*measfilepar).meas.erase( (*measfilepar).meas.begin()+i);
					(*measfilepar).map.erase( (*measfilepar).map.begin()+i);
					(*measfilepar).mask.erase( (*measfilepar).mask.begin()+i);
					(*measfilepar).nrt.erase( (*measfilepar).nrt.begin()+i);
				}



			LogOut("List dump mode! number of measurements = %d (=%d)\n", (*measfilepar).meas.size(),i_meas);
			/*zFinl is a global variable */
			zFinl = (*measfilepar).ct[(*measfilepar).ct.size()-1];
			LogOut("zFinl overwritten to last measurement %lf\n",zFinl);
			(*ninfa).measdata |= (MeasureType) (*measfilepar).meas[0];
			(*ninfa).maty     |= (SliceType) (*measfilepar).map[0];
			(*ninfa).mask     |= (SpectrumMaskType) (*measfilepar).mask[0];
			(*ninfa).nrt      |= (nRunType) (*measfilepar).nrt[0];
			LogOut("First measurement set to %d %d %d %d\n", (*ninfa).measdata, (*ninfa).maty, (*ninfa).mask, (*ninfa).nrt);
			LogOut("- . - . - . - . - . - . - . - . - . - . - . - . -\n");
		}
}

void 		loadmeasfromlist(MeasFileParms *mfp, MeasInfo *info, int i_meas)
{
	(*info).measdata = (MeasureType) (*mfp).meas[i_meas];
	(*info).maty     = (SliceType) (*mfp).map[i_meas];
	(*info).mask     = (SpectrumMaskType) (*mfp).mask[i_meas];
	(*info).nrt     = (nRunType) (*mfp).nrt[i_meas];
}

void axitontracker(Scalar *axion)
{
}

size_t unfoldidx(size_t idx, Scalar *axion)
{
		if (axion->Folded()){
			size_t X[3];
			indexXeon::idx2Vec(idxprint,X,axion->Length());
			size_t v_length = axion->DataAlign()/axion->DataSize();
			size_t XC = axion->Length()*v_length;
			size_t YC = axion->Length()/v_length;
			size_t iiy = X[1]/YC;
			size_t iv  = X[1]-iiy*YC;
			return X[2]*axion->Surf() + iv*XC + X[0]*v_length + iiy;
		} else
			return idx;
}
