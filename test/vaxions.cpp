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

void    printsample  (FILE *fichero, Scalar *axion,            double LLL, size_t idxprint, size_t nstrings_global, double maximumtheta);
void    printsampleS  (FILE *fichero, Scalar *axion,            double LLL, size_t idxprint, size_t nstrings_global, double maximumtheta);
void    printsample_p(FILE *fichero, Scalar *axion, double zz, double LLL, size_t idxprint, size_t nstrings_global, double maximumtheta);
double  findzdoom(Scalar *axion);
void    checkTime (Scalar *axion, int index);
void    printposter (Scalar *axion);

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
	LogOut("\n--               VAXION 3D!                    --\n");
	LogOut("\n-------------------------------------------------\n\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	start = std::chrono::high_resolution_clock::now();

	//-grids
	Scalar *axion;

	if ((fIndex == -1) && (cType == CONF_NONE) && (!restart_flag))
		LogOut("Error: Neither initial conditions nor configuration to be loaded selected. Empty field.\n");
	else
	{
		if ( (fIndex == -1) && !restart_flag)
		{
			LogOut("Generating scalar ... ");
			axion = new Scalar (&myCosmos, sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fTypeP, lType, cType, parm1, parm2);
			LogOut("Done! \n");
		}
		else
		{
			LogOut("Reading initial conditions from file ... ");
			readConf(&myCosmos, &axion, fIndex, restart_flag);
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
	FILE *file_samp ;
	FILE *file_sams ;
	file_samp = NULL;
	file_sams = NULL;
	if (!restart_flag){
		file_samp = fopen("out/sample.txt","w+");
		file_sams = fopen("out/samplS.txt","w+");
	} else{
		file_samp = fopen("out/sample.txt","a+"); // if restart append in file
		file_sams = fopen("out/samplS.txt","a+"); // if restart append in file
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
	MeasInfo ninfa;
	//- information needs to be passed onto measurement files
	ninfa.sliceprint = sliceprint;
	ninfa.idxprint = 0 ;
	ninfa.index = 0;
	ninfa.redmap = endredmap;

	// default measurement type is parsed
	ninfa.measdata = defaultmeasType;
	ninfa.mask = spmask;
	ninfa.rmask = rmask;
	ninfa.strmeas = strmeas;

	//-maximum value of the theta angle in the simulation
	double maximumtheta = M_PI;
	lm.maxTheta = M_PI;

	// dump decision function
	DumpType dumpmode = DUMP_EVERYN ;
	int i_meas = 0;
	bool measrightnow = false;
	double mesi;
	int meastype ;
	std::vector<double>	meas_zlist;
	std::vector<int>	meas_typelist;

	{
			FILE *cacheFile = nullptr;
			if (((cacheFile  = fopen("./measfile.dat", "r")) == nullptr)){
				LogMsg(VERB_NORMAL,"[VAX] No measfile.dat ! Use linear dump mode by default");
			}
			else
			{
				LogOut("\n- . - . - . - . - . - . - . - . - . - . - . - . -\n");
				LogOut("Measurement list : \n");

				dumpmode = DUMP_FROMLIST;
				LogMsg(VERB_NORMAL,"[VAX] Reading measurement files from list");
				fscanf (cacheFile ,"%lf %d", &mesi, &meastype);
				while(!feof(cacheFile)){
					if (meastype < 0)
						meastype = defaultmeasType;
					//if (mesi < *axion->zV()){
					if ((mesi < *axion->zV()) && (abs(1.0 -(*axion->zV())/mesi)>0.0001)){
					// the initial measurement can be occationally discarded here if we set the first measurement time as the initial time of the simulation. (issue?)
					// we allow some fluctuation in order to avoid that.
						LogMsg(VERB_NORMAL,"[VAX] read z=%f < current time (z=%f) > DISCARDED",mesi,*axion->zV());
					}
					else {
						meas_zlist.push_back(mesi);
						meas_typelist.push_back(meastype);
						LogMsg(VERB_NORMAL,"[VAX] i_meas=%d read z=%f meas=%d", i_meas, meas_zlist[i_meas], meas_typelist[i_meas]);
						i_meas++ ;
					}
					fscanf (cacheFile ,"%lf %d", &mesi, &meastype);
				}
				/* check for no repetitions */
				for (int i =i_meas-1; i>0;i--)
					if (meas_zlist[i] == meas_zlist[i-1]){
						LogMsg(VERB_NORMAL,"[VAX] merge %d %d at t %f with %d %d > %d", i, i-1, meas_zlist[i], meas_typelist[i], meas_typelist[i-1], meas_typelist[i]|meas_typelist[i-0]);
						meas_typelist[i-1] |= meas_typelist[i];
						meas_zlist.erase(meas_zlist.begin()+i);
						meas_typelist.erase(meas_typelist.begin()+i);
					}



				LogOut("List dump mode! number of measurements = %d (=%d)\n",meas_zlist.size(),i_meas);
				zFinl = meas_zlist[meas_zlist.size()-1];
				LogOut("zFinl overwritten to last measurement %lf\n",zFinl);
				ninfa.measdata |= (MeasureType) meas_typelist[0];
				LogOut("First measurement set to %d\n", ninfa.measdata);
				LogOut("- . - . - . - . - . - . - . - . - . - . - . - . -\n");
			}
	}
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
	// prepropagator is been moved away
	//--------------------------------------------------

	//--------------------------------------------------
	// INITIAL MEASUREMENT
	//--------------------------------------------------


	if (!restart_flag && (fIndex == -1)){
		index = fIndex2;
		LogOut("First measurement file %d \n",index);
		ninfa.index=index;
		if (prinoconfo & PRINTCONF_INITIAL)
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
			LogOut("time %f and %d-measurement %lf\n",*axion->zV(),i_meas,meas_zlist[i_meas]);
		if (abs(1.0 -(*axion->zV())/meas_zlist[i_meas])<0.0001){
				i_meas++;
				LogOut("i_meas++ initial conditions coincided with 1st measurement\n");
		}

	}

	// SIMPLE OUTPUT CHECK
	printsample(file_samp, axion, myCosmos.Lambda(), idxprint, lm.str.strDen, lm.maxTheta);
	printsampleS(file_sams, axion, myCosmos.Lambda(), idxprint, lm.str.strDen, lm.maxTheta);



	//--------------------------------------------------
	// TIME ITERATION LOOP
	//--------------------------------------------------

	LogOut("Running ...\n\n");
	LogOut("Init propagator Vqcd flag %d\n", myCosmos.QcdPot());
	initPropagator (pType, axion, myCosmos.QcdPot());
	tunePropagator (axion);


	LogOut ("Start redshift loop\n\n");
	for (int iz = 0; iz < nSteps; iz++)
	{

		// time step
		dzaux = axion->dzSize();

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
				if (*axion->zV() > meas_zlist[i_meas])
				{
					for (int i =i_meas; i< meas_zlist.size(); i++){
						if (*axion->zV() > meas_zlist[i])
							i_meas++;
							LogMsg(VERB_NORMAL,"[VAX] Time jumped over measurement! jumping once!");
					}
				}

				if ( (*axion->zV())+dzaux >= meas_zlist[i_meas] && (*axion->zV()) < meas_zlist[i_meas]){
					dzaux = meas_zlist[i_meas] - (*axion->zV());
					measrightnow = true;
					ninfa.measdata = (MeasureType) meas_typelist[i_meas];
					defaultmeasType = ninfa.measdata;
					// actually, if this is the last measurement, do not measure!
					if ( (i_meas == meas_zlist.size()-1) ){
						measrightnow = false;
					}
				}

				break;
			}

			LogFlush();
			// PROPAGATOR
			propagate (axion, dzaux);
			counter++;


			// SIMPLE OUTPUT CHECK
			printsample(file_samp, axion, myCosmos.Lambda(), idxprint, lm.str.strDen, lm.maxTheta);
			printsampleS(file_sams, axion, myCosmos.Lambda(), idxprint, lm.str.strDen, lm.maxTheta);

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
					LogOut("Re-Init propagator Vqcd flag %d\n", (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_DAMP_RHO);
					initPropagator (pType, axion, (myCosmos.QcdPot() & VQCD_TYPE) | VQCD_DAMP_RHO);
					coD = false ;
					// possible problem!! if gamma is needed later, as it is written pregammo will stay
				}

				// TRANSITION TO THETA COUNTER
				if (lm.str.strDen == 0) {
					LogOut("  no st counter %d\n", strcount);
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
							lm = Measureme (axion, ninfa);
							index++;


						LogOut("--------------------------------------------------\n");
						LogOut(" TRANSITION TO THETA (z=%.4f R=%.4f)\n",(*axion->zV()), (*axion->RV()));
						LogOut(" shift = %f \n", axion->Saskia());

						double shiftz = axion->Saskia()*(*axion->RV());
						cmplxToTheta (axion, shiftz);

						// Measurement after switching to theta
						ninfa.index=index;
						lm = Measureme (axion, ninfa);
						index++;
						LogOut("--------------------------------------------------\n");

						tunePropagator (axion);
					}
			}

			// Break the loop when we are done
			if ( (*axion->zV()) >= zFinl ){
				LogOut("zf reached! ENDING ... \n"); fflush(stdout);
				break;
			}

			// Partial analysis
			if(measrightnow){

				ninfa.index=index;
				// in case theta transitioned, the meas was saved as the default
				ninfa.measdata = defaultmeasType;
				lm = Measureme (axion, ninfa);
				index++;
				i_meas++ ;
				//reset flag
				measrightnow = false;
				// after every measurement we check walltime > need update
				checkTime(axion, index);
			}


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

	if ((prinoconfo & PRINTCONF_FINAL) ) {
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

			if (prinoconfo & PRINTCONF_WKB) {
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

void printsample(FILE *fichero, Scalar *axion, double LLL, size_t idxprint, size_t nstrings_global, double maximumtheta)
{
	double z_now = (*axion->zV());
	double R_now = (*axion->RV());
	double llphys = LLL;
	if (axion->Lambda() == LAMBDA_Z2)
		llphys = LLL/(R_now*R_now);

	// LogOut("z %f R %f\n",z_now, R_now);
	size_t S0 = sizeN*sizeN ;
	if (commRank() == 0){
		if (sPrec == FIELD_SINGLE) {
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
		} else if (sPrec == FIELD_DOUBLE){
			if (axion->Field() == FIELD_SAXION) {
				double axmass_now = axion->AxionMass();
				double saskia = axion->Saskia();

				fprintf(fichero,"%f %f %f %f %f %f %f %ld %f %e\n", z_now, axmass_now, llphys,
				static_cast<complex<double> *> (axion->mCpu())[idxprint + S0].real(),
				static_cast<complex<double> *> (axion->mCpu())[idxprint + S0].imag(),
				static_cast<complex<double> *> (axion->vCpu())[idxprint].real(),
				static_cast<complex<double> *> (axion->vCpu())[idxprint].imag(),
				nstrings_global, maximumtheta, saskia);
			} else {
				fprintf(fichero,"%f %f %f %f %f\n", z_now, axion->AxionMass(),
				static_cast<double *> (axion->mCpu())[idxprint + S0],
				static_cast<double *> (axion->vCpu())[idxprint], maximumtheta);
			}
		}
	}
}

void printsampleS(FILE *fichero, Scalar *axion, double LLL, size_t idxprint, size_t nstrings_global, double maximumtheta)
{
	double z_now = (*axion->zV());
	double R_now = (*axion->RV());
	double llphys = LLL;
	if (axion->Lambda() == LAMBDA_Z2)
		llphys = LLL/(R_now*R_now);

	// LogOut("z %f R %f\n",z_now, R_now);
	size_t S0 = sizeN*sizeN ;
	if (commRank() == 0){
		if (sPrec == FIELD_SINGLE) {
			if (axion->Field() == FIELD_SAXION) {
				double axmass_now = axion->AxionMass();
				double saskia = axion->Saskia();
				double inte = axion->IAxionMassSqn(0,z_now,3);
				double iinte = axion->IIAxionMassSqn(0,z_now,3);

				fprintf(fichero,"%f %f %f %f %f %f %f %f %f %ld %f %e %f %f\n", z_now, axmass_now, llphys,
				static_cast<complex<float> *> (axion->mCpu())[idxprint + S0].real(),
				static_cast<complex<float> *> (axion->mCpu())[idxprint + S0].imag(),
				static_cast<complex<float> *> (axion->vCpu())[idxprint].real(),
				static_cast<complex<float> *> (axion->vCpu())[idxprint].imag(),
				static_cast<complex<float> *> (axion->m2Cpu())[idxprint].real(),
				static_cast<complex<float> *> (axion->m2Cpu())[idxprint].imag(),
				nstrings_global, maximumtheta, saskia, inte, iinte);
			} else {
				fprintf(fichero,"%f %f %f %f %f\n", z_now, axion->AxionMass(),
				static_cast<float *> (axion->mCpu())[idxprint + S0],
				static_cast<float *> (axion->vCpu())[idxprint], maximumtheta);
			}
			fflush(fichero);
		} else if (sPrec == FIELD_DOUBLE){
			if (axion->Field() == FIELD_SAXION) {
				double axmass_now = axion->AxionMass();
				double saskia = axion->Saskia();

				fprintf(fichero,"%f %f %f %f %f %f %f %ld %f %e\n", z_now, axmass_now, llphys,
				static_cast<complex<double> *> (axion->mCpu())[idxprint + S0].real(),
				static_cast<complex<double> *> (axion->mCpu())[idxprint + S0].imag(),
				static_cast<complex<double> *> (axion->vCpu())[idxprint].real(),
				static_cast<complex<double> *> (axion->vCpu())[idxprint].imag(),
				nstrings_global, maximumtheta, saskia);
			} else {
				fprintf(fichero,"%f %f %f %f %f\n", z_now, axion->AxionMass(),
				static_cast<double *> (axion->mCpu())[idxprint + S0],
				static_cast<double *> (axion->vCpu())[idxprint], maximumtheta);
			}
		}
	}
}


double findzdoom(Scalar *axion)
{
	if (axion->BckGnd()->Indi3() > 0.0){
	double ct = zInit ;
	double DWfun;
	double meas ;
	double fff = axion->BckGnd()->Frw();
	while (meas < 0.001)
	{
		DWfun = 40*axion->AxionMassSq(ct)/(2.0*axion->BckGnd()->Lambda()) ;
		if (axion->Lambda() == LAMBDA_Z2)
			DWfun *= pow(ct,2*fff);
		meas = DWfun - 1 ;
		ct += 0.001 ;
	}
	LogMsg(VERB_NORMAL,"[VAX findzdoom] Real z_doom %f ", ct );
	return ct ;
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
			LogMsg(VERB_NORMAL, "[VAX checkTime %d] stop file detected! stopping ... ",index);
			LogOut ("Interrupted manually with stop file ...");
		}
		if (flag ==1){
			LogMsg(VERB_NORMAL, "[VAX checkTime %d] Walltime reached ",index);
			LogOut ("Walltime reached, dumping configuration...");
		}

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
	if (LAMBDA_FIXED == axion->Lambda()){
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
		LogOut("                       =  0 in (%3.3f, %3.3f) \n", axion->BckGnd()->ZThRes(), axion->BckGnd()->ZRestore());

	if	((axion->BckGnd()->QcdPot() & VQCD_TYPE) == VQCD_1)
		LogOut("VQCD1PQ1,shift,continuous theta  \n\n");
	else if	((axion->BckGnd()->QcdPot() & VQCD_TYPE) == VQCD_2)
		LogOut("VQCD2PQ1,no shift, continuous theta  \n\n");
	else if	((axion->BckGnd()->QcdPot() & VQCD_TYPE) == VQCD_1_PQ_2)
		LogOut("VQCD1PQ2,shift, continuous theta  \n\n");
	else if	((axion->BckGnd()->QcdPot() & VQCD_TYPE) == VQCD_1N2)
		LogOut("VQCD1PQ1,NDW=2, no shift!, continuous theta \n\n");

		LogOut("Vqcd flag %d\n", axion->BckGnd()->QcdPot());
		LogOut("Damping flag %d 		     \n", axion->BckGnd()->QcdPot() & VQCD_DAMP);
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
