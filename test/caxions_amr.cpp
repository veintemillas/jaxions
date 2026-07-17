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


static inline bool sign_change(double a, double b);
double find_R(Scalar *axion, int *ncross);

template<class Float>
void expander_cubic(Scalar *axion, int i_rho_base);
void caxion_compress_mpi_serial(Scalar *axion);
template<class Float>
inline void get_src_bc(const Float *mCpu,
                       int ir, int iz,
                       int Nr, int Nz, int Ng,
                       int rank,
                       double &re, double &im);
template<class Float>
inline void cubic_weights(double x, int &i, double w[4]);

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
size_t idxprint = 0 ;
//- z-coordinate of the slice that is printed as a 2D map
size_t sliceprint = 0 ;
static int  amr_count = 0 ;   // number of AMR refinements done so far
static bool refine    = true; // whether AMR is still active

//- Append the relevant AMR params to the restart file
static void caxion_save_amr_restart (void)
{
	commSync();
	if (commRank() == 0) {
		char rp[1200];
		sprintf(rp, "%s/%s.restart", outDir, outName);
		hid_t fid = H5Fopen(rp, H5F_ACC_RDWR, H5P_DEFAULT);
		if (fid >= 0) {
			int refine_i = refine ? 1 : 0;
			writeAttribute(fid, &amr_count, "amr_count",  H5T_NATIVE_INT);
			writeAttribute(fid, &refine_i,  "amr_refine", H5T_NATIVE_INT);
			H5Fclose(fid);
		} else
			LogOut("[cax restart] WARNING: could not reopen %s to store AMR state\n", rp);
	}
	commSync();
}


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

#ifndef USE_2DCYL
	ninfa.measdata |= MEAS_NNSPEC;
#endif
	if (!restart_flag && (fIndex == -1)){
		index = fIndex2;
		LogOut("First measurement file %d \n",index);
		ninfa.index=index;
		if (ninfa.printconf & PRINTCONF_INITIAL)
			ninfa.measdata |= MEAS_3DMAP ;
		LogMsg(VERB_NORMAL,"[VAX DEBUG] Before Measureme: ninfa.strmeas = %d", (int)ninfa.strmeas);
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
		LogMsg(VERB_NORMAL,"[VAX DEBUG] Before Measureme: ninfa.strmeas = %d", (int)ninfa.strmeas);
		lm = Measureme (axion, ninfa);
	}

	index++;
	if ( (dumpmode == DUMP_FROMLIST) ){
			LogOut("time %f and %d-measurement %lf\n",*axion->zV(),i_meas,measfilepar.ct[i_meas]);
		if (!restart_flag && (abs(1.0 -(*axion->zV())/measfilepar.ct[i_meas])<0.0001)){
				i_meas++;
				LogOut("i_meas++ initial conditions coincided with 1st measurement\n");
		}

	}

	// SIMPLE OUTPUT CHECK
	printsample(file_samp, axion, ninfa.idxprint, lm.str.strDen, lm.maxTheta);

	// Creation ends here
	if (myCosmos.ICData().nSteps == 0){
		LogOut("--------------------------------------------------\n");
		LogOut("           END CREATION                   \n");
		LogOut("--------------------------------------------------\n");
		fclose(file_samp);
		delete axion;
		endAxions();
		return 0;
	}

	double r_amr = -1.0;
	if (myCosmos.ICData().kcr < 0.5)
	{
		LogOut("--------------------------------------------------\n");
		LogOut("           Poor mans AMR when R_loop = %d/%.2f          \n",axion->TZ(),1/myCosmos.ICData().kcr);
		LogOut("--------------------------------------------------\n");
		r_amr = 1.0/myCosmos.ICData().kcr;
	}



	//--------------------------------------------------
	// Axiton TRACKER (if THETA)
	//--------------------------------------------------

		//initTracker(axion);
		//searchAxitons();

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

	double L = axion->BckGnd()->PhysSize();       // refined box restored by readConf on --restart
	double delta = axion->BckGnd()->PhysSize()/ ((double) axion->TZ());
	double radius_save = L;
	refine     = true;                                // file-scope; default for a fresh run
	amr_count  = 0;                                   // file-scope; number of AMR refinements done
	int  amr_max   = myCosmos.ICData().maxamr;        // cap (-1 = unlimited), from --maxamr

	/* On restart, recover the exact AMR bookkeeping (amr_count, refine) from the
	   attributes we stored on the restart file in checkTime(). L/delta already
	   come back correct via readConf. Missing attributes (old restart / never
	   refined) -> keep the fresh defaults. */
	if (restart_flag)
	{
		char rp[1200];
		sprintf(rp, "%s/%s.restart", outDir, outName);
		hid_t fid = H5Fopen(rp, H5F_ACC_RDONLY, H5P_DEFAULT);
		if (fid >= 0) {
			int refine_i = 1;
			if (readAttribute(fid, &amr_count, "amr_count",  H5T_NATIVE_INT) >= 0 &&
			    readAttribute(fid, &refine_i,  "amr_refine", H5T_NATIVE_INT) >= 0) {
				refine = (refine_i != 0);
				LogOut("[cax restart] recovered amr_count=%d refine=%d, L=%f delta=%f\n",
				       amr_count, (int)refine, L, delta);
			}
			H5Fclose(fid);
		}
	}


	LogOut ("Start redshift loop (steps %lu)\n\n", myCosmos.ICData().nSteps);
	for (int iz = 0; iz < myCosmos.ICData().nSteps; iz++)
	{

		// time step
		// if ((axion->Field() == FIELD_AXION ) || (axion->Field() == FIELD_SAXION ))
		//  dzaux = 0.0;
		//  else
		LogMsg(VERB_HIGH,"[vax] propagation loop %d / %d",iz,nSteps);
		 dzaux = (uwDz) ? axion->dct_Adaptive() : (zFinl-zInit)/nSteps ;

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
							LogMsg(VERB_NORMAL,"[VAX] Time jumped over measurement! jumping once!");LogFlush();
					}
				}

				if ( (*axion->zV())+dzaux >= measfilepar.ct[i_meas] && (*axion->zV()) < measfilepar.ct[i_meas]){
					LogMsg(VERB_NORMAL,"[VAX] dct adjusted from %e",dzaux);LogFlush();
					dzaux = measfilepar.ct[i_meas] - (*axion->zV());
					LogMsg(VERB_NORMAL,"                   to   %e",dzaux);LogFlush();
					measrightnow = true;
					loadmeasfromlist(&measfilepar, &ninfa, i_meas);
					defaultmeasType = ninfa.measdata;
					// actually, if this is the last measurement, do not measure!
					if ( (i_meas == measfilepar.ct.size()-1) ){
						LogMsg(VERB_NORMAL,"[VAX] last measurement, do not measure and pass END!",dzaux);LogFlush();
						measrightnow = false;
					}
				}

				break;
			}

			// PROPAGATOR

			propagate (axion, dzaux);
			counter++;

 			LogFlush();

			// SIMPLE OUTPUT CHECK
			printsample(file_samp, axion, ninfa.idxprint, lm.str.strDen, lm.maxTheta);

			// AMR
#ifdef USE_2DCYL
			if ((axion->Field() == FIELD_SAXION ))
			{
				axion->exchangeGhosts(FIELD_M);
				// find_R reads the field on the host (axion->mStart()). On GPU the
				// live field is on the device, so bring it back exactly like
				// vaxions/Measureme do before any host-side read (vaxions.cpp:704,
				// measa.cpp:44) or find_R measures a stale configuration.
				if (cDev == DEV_GPU)
					axion->transferCpu(FIELD_MV);
				int cross = 0 ;
				double r = find_R(axion,&cross) * delta;
				if (r < radius_save)
					radius_save = r;
				if (cross >= 2 && refine){
					LogOut("Double crossing. No more AMR\n");
					refine = false;
				}
				// refinement condition, when loop has shrunk enough
				// UNLESS it is in the focused phase
				// I define it with the double zero 
				if (refine && (r < L/r_amr))
					{
						{
							ninfa.index=index;
							ninfa.measdata = defaultmeasType;
							ninfa.cTimesec = (double) Timer()*1.0e-6;
							ninfa.propstep = iz;
							lm = Measureme (axion, ninfa);
							index++;
							i_meas++ ;
							//reset flag
							measrightnow = false;
						}

						// On a GPU run, do the whole compression host-side: switch
						// to CPU mode so the exchangeGhosts calls below stay host-only
						// and don't clobber the half-compressed host buffer with stale
						// device ghosts. The host copy is current here (both find_R
						// and Measureme just transferCpu'd it).
						if (cDev == DEV_GPU)
							axion->setDev(DEV_CPU);

						Folder munge(axion);
						munge(UNFOLD_ALL);

						LogOut("[cax] 1/4 Expansion! ! \n");
						// expands m
						axion->exchangeGhosts(FIELD_M);

						caxion_compress_mpi_serial(axion);
						// expands v
						memmove(axion->m2Cpu(),axion->mStart(),axion->DataSize()*axion->Size());
						memmove(axion->mStart(),axion->vCpu(),axion->DataSize()*axion->Size());
						memmove(axion->vCpu(),axion->m2Cpu(),axion->DataSize()*axion->Size());
						axion->exchangeGhosts(FIELD_M);
						caxion_compress_mpi_serial(axion);

						memmove(axion->m2Cpu(),axion->mStart(),axion->DataSize()*axion->Size());
						memmove(axion->mStart(),axion->vCpu(),axion->DataSize()*axion->Size());
						memmove(axion->vCpu(),axion->m2Cpu(),axion->DataSize()*axion->Size());

						measrightnow = true;
						axion->setFolded(false);

						// Back to GPU mode and push the freshly compressed host
						// configuration to the device, or the next propagate (and
						// every find_R after it) runs on the old, uncompressed field.
						if (cDev == DEV_GPU) {
							axion->setDev(DEV_GPU);
							axion->transferDev(FIELD_MV);
						}

						L     /= 2.0;
						delta /= 2.0;
						axion->BckGnd()->SetPhysSize(L);

						// Cap refinement levels (--maxamr): stop AMR after amr_max
						// refinements so delta (and the adaptive dt) can't spiral to 0.
						amr_count++;
						if (amr_max >= 0 && amr_count >= amr_max){
							LogOut("Max AMR levels (%d) reached. No more AMR\n", amr_max);
							refine = false;
						}

						// double la = axion->BckGnd()->Lambda()/std::sqrt(2.0);
						// axion->BckGnd()->SetLambda(la);
					}
			}
#endif 
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

				//searchAxitons ();

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












void printsample(FILE *fichero, Scalar *axion,  size_t idxprint_global, size_t nstrings_global, double maximumtheta)
{
	double z_now = (*axion->zV());
	double R_now = (*axion->RV());
	double llphys = axion->LambdaP();

	/* unfold if needed */

	size_t S0 = sizeN*sizeN ;

	/* Calculate rank */
	size_t zidx = idxprint_global/axion->Surf();
	int rankprint = zidx/axion->Depth();
	size_t idxprinta = idxprint_global - rankprint*axion->Size();
	size_t idxp = unfoldidx(idxprinta, axion);

	LogMsg(VERB_HIGH,"printsample [global idx %lu ] [local idx=%lu] [folded %lu] from rank %d", idxprint_global,	idxprint, idxp, rankprint);
	if (commRank() == rankprint){
		if (sPrec == FIELD_SINGLE) {
			float buff[4];
			if (axion->Field() == FIELD_SAXION) {
				double axmass_now = axion->AxionMass();
				double saskia = axion->Saskia();
#ifdef USE_GPU
				if (axion->Device() == DEV_GPU) {
					cudaMemcpy(buff, &(static_cast<float*>(axion->mGpuStart())[2*idxprinta]),2*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(&(buff[2]), &(static_cast<float*>(axion->vGpu())[2*idxprinta]),2*sizeof(float),cudaMemcpyDeviceToHost);
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
			caxion_save_amr_restart();          // persist amr_count/refine onto the restart file
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
			caxion_save_amr_restart();          // persist amr_count/refine onto the restart file
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
	LogOut("        SIMULATION (%d x %d x %d) ", axion->NX(), axion->NY(), axion->NZ());
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
			indexXeon::idx2Vec(idx,X,axion->Length());
			size_t v_length = axion->DataAlign()/axion->DataSize();
			size_t XC = axion->Length()*v_length;
			size_t YC = axion->Length()/v_length;
			size_t iiy = X[1]/YC;
			size_t iv  = X[1]-iiy*YC;
			return X[2]*axion->Surf() + iv*XC + X[0]*v_length + iiy;
		} else
			return idx;
}

#include <mpi.h>
#include <cmath>
#include <algorithm>

static inline bool sign_change(double a, double b)
{
    return ((a <= 0.0 && b >  0.0) ||
            (a >= 0.0 && b <  0.0));
}

double find_R(Scalar *axion, int *ncross)
{
    const size_t Nrho   = axion->NZ();
    const size_t stride = axion->NX();

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const size_t rho0 = static_cast<size_t>(rank) * Nrho;

    double Rloc = -1.0;

    double first = 0.0;
    double last  = 0.0;
	int    nloc   = 0;

	/* the last rank wraps unphysically back to 0, should not check!*/
	size_t Nrho_tocheck = rank == size-1? Nrho-1 : Nrho;

    if (axion->Precision() == FIELD_SINGLE)
    {
        float *phi = static_cast<float*>(axion->mStart());

        first = static_cast<double>(phi[0]);
        last  = static_cast<double>(phi[2*stride*(Nrho-1)]);

        for (size_t i = 0; i < Nrho_tocheck; ++i)
        {
            const double f0 = static_cast<double>(phi[2*stride*i]);
            const double f1 = static_cast<double>(phi[2*stride*(i+1)]);

            if (sign_change(f0, f1)){
                Rloc = static_cast<double>(rho0 + i) + 0.5;
				nloc++;
			}
        }
    }
    else
    {
        double *phi = static_cast<double*>(axion->mStart());

        first = phi[0];
        last  = phi[2*stride*(Nrho-1)];

        for (size_t i = 0; i + 1 < Nrho_tocheck; ++i)
        {
            const double f0 = phi[2*stride*i];
            const double f1 = phi[2*stride*(i+1)];

            if (sign_change(f0, f1)){
                Rloc = static_cast<double>(rho0 + i) + 0.5;
				nloc++;
			}
        }
    }

    // Check crossing between this rank's last point and next rank's first point.
    double next_first = 0.0;

    if (size > 1)
    {
        MPI_Sendrecv(&first,      1, MPI_DOUBLE, rank-1 >= 0 ? rank-1 : MPI_PROC_NULL, 771,
                     &next_first, 1, MPI_DOUBLE, rank+1 < size ? rank+1 : MPI_PROC_NULL, 771,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (rank + 1 < size)
        {
            if (sign_change(last, next_first))
                Rloc = std::max(Rloc, static_cast<double>(rho0 + Nrho - 1) + 0.5);
        }
    }

    double Rglob = -1.0;
    MPI_Allreduce(&Rloc, &Rglob, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	int nglob;
	MPI_Allreduce(&nloc, &nglob, 1,
                  MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	*ncross = nglob;
    return Rglob;
}



template<class Float>
static inline MPI_Datatype caxion_mpi_type();

template<>
inline MPI_Datatype caxion_mpi_type<float>() { return MPI_FLOAT; }

template<>
inline MPI_Datatype caxion_mpi_type<double>() { return MPI_DOUBLE; }

template<class Float>
inline void cubic_weights(double x, int &i, double w[4])
{
    i = (int) floor(x);
    double t = x - (double)i;

    // stencil: i-1, i, i+1, i+2
    w[0] = -t*(t-1.0)*(t-2.0)/6.0;
    w[1] =  (t+1.0)*(t-1.0)*(t-2.0)/2.0;
    w[2] = -(t+1.0)*t*(t-2.0)/2.0;
    w[3] =  (t+1.0)*t*(t-1.0)/6.0;
}

template<class Float>
inline void get_src_bc(const Float *mCpu,
                       int ir, int iz,
                       int Nr, int Nz, int Ng,
                       int rank,
                       double &re, double &im)
{
    // rho axis only reflects on global rank 0
    if (rank == 0 && ir < 0)
        ir = -ir;

    // z=0: phi(r,-z)=phi*(r,z)
    bool conjz = false;
    if (iz < 0) {
        iz = -iz;
        conjz = true;
    }

    // emergency guards
    if (ir < -Ng) ir = -Ng;
    if (ir >= Nr + Ng) ir = Nr + Ng - 1;
    if (iz >= Nz) iz = Nz - 1;

    size_t idx = (size_t)(ir + Ng)*Nz + iz;

    re = (double)mCpu[2*idx    ];
    im = (double)mCpu[2*idx + 1];

    if (conjz)
        im = -im;
}

template<class Float>
void expander_cubic(Scalar *axion, int i_rho_base)
{
    const Float *m  = static_cast<const Float*>(axion->mCpu());   // ghosted source
    Float       *m2 = static_cast<Float*>(axion->m2Cpu());        // refined output

    const int Nr = axion->NZ();       // local rho physical
    const int Nz = axion->NX();       // z fast
    const int Ng = axion->getNg();

    const int rank = commRank();

    #pragma omp parallel for collapse(2)
    for (int irf = 0; irf < Nr; ++irf) {
        for (int izf = 0; izf < Nz; ++izf) {

            double xold = (double)i_rho_base + 0.5*(double)irf;
            double zold = 0.5*(double)izf;

            int ix, iz;
            double wx[4], wz[4];

            cubic_weights<Float>(xold, ix, wx);
            cubic_weights<Float>(zold, iz, wz);

            double re = 0.0;
            double im = 0.0;

            for (int a = 0; a < 4; ++a) {
                int ira = ix + a - 1;

                for (int b = 0; b < 4; ++b) {
                    int izb = iz + b - 1;

                    double rr, ii;
                    get_src_bc<Float>(m, ira, izb, Nr, Nz, Ng, rank, rr, ii);

                    double ww = wx[a]*wz[b];

                    re += ww*rr;
                    im += ww*ii;
                }
            }

            size_t id = (size_t)irf*Nz + izf;

            m2[2*id    ] = (Float)re;
            m2[2*id + 1] = (Float)im;
        }
    }
}

template<class Float>
void caxion_compress_array_serial(
    Scalar *axion,
    Float *dstStart,      // mStart or vStart
    int tag,
    bool expand_velocity)
{
    int rank, nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    const int Nr = axion->NZ();   // local rho
    const int Nz = axion->NX();   // fast z

    const size_t count = (size_t)Nr * Nz * 2;
    const size_t bytes = count * sizeof(Float);

    MPI_Datatype T = caxion_mpi_type<Float>();

    Float *work = static_cast<Float*>(axion->m2Cpu());

    for (int r = nRanks - 1; r >= 0; --r) {

        const int q     = r / 2;   // source old rank
        const int chunk = r % 2;   // 0 or 1
        const int i_rho_base = chunk * (Nr/2);

        if (rank == q) {

			expander_cubic<Float>(axion, i_rho_base); // writes m-refined block into m2Cpu

            if (r == q) {
                std::memcpy(dstStart, work, bytes);
            } else {
                MPI_Send(work, count, T, r, tag, MPI_COMM_WORLD);
            }
        }

        if (rank == r && r != q) {
            MPI_Recv(dstStart, count, T, q, tag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
}


void caxion_compress_mpi_serial(Scalar *axion)
{
    if (axion->Precision() == FIELD_SINGLE) {

        float *mStart = static_cast<float*>(axion->mStart());
        float *vStart = static_cast<float*>(axion->vStart());

        caxion_compress_array_serial<float>(
            axion, mStart, 6100, false
        );

        // caxion_compress_array_serial<float>(
        //     axion, vStart, 6200, true
        // );

    } else {

        double *mStart = static_cast<double*>(axion->mStart());
        double *vStart = static_cast<double*>(axion->vStart());

        caxion_compress_array_serial<double>(
            axion, mStart, 6100, false
        );

        // caxion_compress_array_serial<double>(
        //     axion, vStart, 6200, true
        // );
    }

    MPI_Barrier(MPI_COMM_WORLD);
}



























// #include <mpi.h>
// #include <cstdio>
// #include <cstdlib>
// #include <cstring>
// #include <cmath>

// inline int ref(int i)
// {
// 	return (i < 0) ? -i : i;
// }

// // expands from a given i_rho a Nrho/2,Nz/2 slice into m2

// template<class cFloat>
// void expander(Scalar *axion,int i_rho_base){
	
// 	// we create arefined array into m2

// 	Float *m  = static_cast<cFloat*>(axion->mCpu());
// 	Float *m2 = static_cast<cFloat*>(axion->m2Cpu());
// 	size_t Nrho = axion->NZ();
// 	size_t Nz = axion->NX();
// 	size_t B  = axion->getNg()*Nz;

// 	int rank = commRank();
// 	// interpolate in rho 
// 	// make sure ghosts are in place
// 	#pragma omp parallel for
// 	for (size_t i_Z = 0 ; i_Z < Z/2; i_Z++){
// 		// I interpolate centrals with 4 points at each side in place
// 		size_t offset = B+i_rho_base*Nz + iZ;
// 		int i0=-1,i1=0,i2=1,i3=2;
// 		// 1st read 
// 		cFloat f0 = m[offset-i0*Nz],f1=m[offset+i1*Nz],f2=m[offset+i2*Nz],f3=m[offset+i3*Nz];
// 		if (rank == 0 && i_rho_base+i0<0)
// 			f0 = conj(f0);
// 		while (i1 < i_rho_base+Nrho/2){
// 			//build z interpolation
// 			f15 = (-f0 + 9.0*f1 + 9.0*f2 - f3)/16.0;
// 			//write
// 			m2[2*offset+2*i1*Nz]   = f1;
// 			m2[2*offset+2*i1*Nz+1] = f15;
// 			//roll 
// 			i0=i1;i1=i2;i2=i3;i3=i3+1;
// 			// new read
// 			f0=f1;f1=f2,f2=f3;f3=m[offset+i3*Nz];
// 		}	
	

// 	// now z lines
// 	#pragma omp parallel for
// 	for (size_t i_rho = 0 ; i_rho < Nrho/2; i_rho++){
// 		// I interpolate centrals with 4 points at each side 
// 		size_t offset = B+i_rho*Nz;
// 		size_t off    = i_rho*Nz;
// 		int i0=-1,i1=0,i2=1,i3=2;
// 		// 1st read 
// 		cFloat f0 = m[offset+ref(i0)],f1=m[offset+i1],f2=m[offset+i2],f3=m[offset+i3];

// 		while (i1 < Nz/2){
// 			//build z interpolation
// 			f15 = (-f0 + 9.0*f1 + 9.0*f2 - f3)/16.0;
// 			//write
// 			m2[2*off+2*i1]   = f1;
// 			m2[2*off+2*i1+1] = f15;
// 			//roll 
// 			i0=i1;i1=i2;i2=i3;i3=i3+1;
// 			// new read
// 			f0=f1;f1=f2,f2=f3;f3=m[offset+i3];
// 		}	
// 	}
// 	// Zlines performed




// 	}



// /***********************************************************************
//  *  caxion MPI /2 compressor
//  *
//  *  Layout assumed:
//  *
//  *      field = (Re Im)(Re Im)... with Z fast, RHO slow
//  *
//  *      complex index: idx = ir*Nz + iz
//  *      raw index:     field[2*idx], field[2*idx+1]
//  *
//  *  mCpu/vCpu include rho ghosts:
//  *
//  *      [ Ng ghost rho slices ][ physical rho slices ][ upper ghosts ... ]
//  *
//  *  mStart/vStart point to first physical rho slice.
//  *
//  *  Compression:
//  *
//  *      old rank q produces refined blocks for new ranks 2q and 2q+1.
//  *      Each block is interpolated first on source rank, then sent serially.
//  *
//  ***********************************************************************/

// template<class Float>
// static inline MPI_Datatype caxion_mpi_type();

// template<>
// inline MPI_Datatype caxion_mpi_type<float>() { return MPI_FLOAT; }

// template<>
// inline MPI_Datatype caxion_mpi_type<double>() { return MPI_DOUBLE; }


// static inline void caxion_quad_weights(double x, int &i0, double w[3])
// {
//     i0 = (int) std::floor(x);

//     const double t = x - (double)i0;

//     // Lagrange through points i0-1, i0, i0+1
//     w[0] = 0.5*t*(t - 1.0);
//     w[1] = 1.0 - t*t;
//     w[2] = 0.5*t*(t + 1.0);
// }


// template<class Float>
// static inline void caxion_get_complex_bc(
//     const Float *srcCpu,
//     int ir,
//     int iz,
//     int Nr,
//     int Nz,
//     int Ng,
//     double &re,
//     double &im)
// {
//     // rho axis: cylindrical even extension
//     if (ir < 0)
//         ir = -ir;

//     // z=0 symmetry: phi(rho,-z) = phi*(rho,z)
//     bool conjugate = false;

//     if (iz < 0) {
//         iz = -iz;
//         conjugate = true;
//     }

//     // Upper z should not be reached in this /2 zoom, but guard anyway.
//     if (iz >= Nz)
//         iz = Nz - 1;

//     // Allow upper rho ghosts. Clamp only as emergency protection.
//     if (ir < -Ng)
//         ir = -Ng;

//     if (ir >= Nr + Ng)
//         ir = Nr + Ng - 1;

//     const size_t idx = ((size_t)(ir + Ng))*((size_t)Nz) + (size_t)iz;

//     re = (double)srcCpu[2*idx    ];
//     im = (double)srcCpu[2*idx + 1];

//     if (conjugate)
//         im = -im;
// }


// template<class Float>
// static inline void caxion_interp_quad_complex_bc(
//     const Float *srcCpu,
//     double x,
//     double z,
//     int Nr,
//     int Nz,
//     int Ng,
//     double &re,
//     double &im)
// {
//     int ix, iz;
//     double wx[3], wz[3];

//     caxion_quad_weights(x, ix, wx);
//     caxion_quad_weights(z, iz, wz);

//     re = 0.0;
//     im = 0.0;

//     for (int a = 0; a < 3; ++a) {
//         const int irs = ix + a - 1;

//         for (int b = 0; b < 3; ++b) {
//             const int izs = iz + b - 1;

//             double rr, ii;
//             caxion_get_complex_bc(srcCpu, irs, izs, Nr, Nz, Ng, rr, ii);

//             const double ww = wx[a]*wz[b];

//             re += ww*rr;
//             im += ww*ii;
//         }
//     }
// }


// template<class Float>
// static inline void caxion_set_complex_phys(
//     Float *dstStart,
//     int ir,
//     int iz,
//     int Nz,
//     double re,
//     double im)
// {
//     const size_t idx = ((size_t)ir)*((size_t)Nz) + (size_t)iz;

//     dstStart[2*idx    ] = (Float)re;
//     dstStart[2*idx + 1] = (Float)im;
// }


// template<class Float>
// void caxion_build_refined_half_block_quad(
//     const Float *srcCpu,
//     Float *dstBlock,
//     int Nr,
//     int Nz,
//     int Ng,
//     int half_id)
// {
//     // half_id = 0 -> old local rho [0, Nr/2]
//     // half_id = 1 -> old local rho [Nr/2, Nr]
//     //
//     // destination block has full physical size Nr x Nz.
//     //
//     // new local ir maps to old source coordinate:
//     //
//     //     xold = ir/2 + half_id*Nr/2
//     //
//     // z always maps as:
//     //
//     //     zold = iz/2

//     const double rho_offset = 0.5*(double)(half_id*Nr);

//     for (int ir = 0; ir < Nr; ++ir) {
//         const double xold = 0.5*(double)ir + rho_offset;

//         for (int iz = 0; iz < Nz; ++iz) {
//             const double zold = 0.5*(double)iz;

//             double re, im;

//             caxion_interp_quad_complex_bc(
//                 srcCpu,
//                 xold,
//                 zold,
//                 Nr,
//                 Nz,
//                 Ng,
//                 re,
//                 im
//             );

//             caxion_set_complex_phys(dstBlock, ir, iz, Nz, re, im);
//         }
//     }
// }


// template<class Float>
// void caxion_compress_array_serial(
//     const Float *srcCpu,
//     Float *dstStart,
//     Float *work,
//     int Nr,
//     int Nz,
//     int Ng,
//     int tag)
// {
//     int rank, nRanks;

//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

//     if (nRanks % 2 != 0) {
//         if (rank == 0)
//             std::fprintf(stderr, "[caxion] compressor needs even number of MPI ranks\n");
//         MPI_Abort(MPI_COMM_WORLD, 1);
//     }

//     if ((Nr % 2) || (Nz % 2)) {
//         if (rank == 0)
//             std::fprintf(stderr, "[caxion] compressor needs even Nr and Nz\n");
//         MPI_Abort(MPI_COMM_WORLD, 1);
//     }

//     if (Ng < 1) {
//         if (rank == 0)
//             std::fprintf(stderr, "[caxion] compressor needs at least Ng >= 1\n");
//         MPI_Abort(MPI_COMM_WORLD, 1);
//     }

//     const size_t count = (size_t)Nr * (size_t)Nz * 2u;
//     const size_t bytes = count * sizeof(Float);

//     MPI_Datatype T = caxion_mpi_type<Float>();

//     // Serial high-to-low ordering.
//     //
//     // This prevents an old source rank q from being overwritten before
//     // it has produced data for destination ranks 2q+1 and 2q.
//     //
//     // new rank r receives from old rank q=r/2, chunk=r%2.

//     for (int r = nRanks - 1; r >= 0; --r) {

//         const int q     = r/2;
//         const int chunk = r%2;

//         if (rank == q) {

//             caxion_build_refined_half_block_quad(
//                 srcCpu,
//                 work,
//                 Nr,
//                 Nz,
//                 Ng,
//                 chunk
//             );

//             if (r == q) {
//                 // Self case, e.g. final rank 0.
//                 std::memcpy(dstStart, work, bytes);
//             } else {
//                 MPI_Send(work, count, T, r, tag, MPI_COMM_WORLD);
//             }
//         }

//         if (rank == r && r != q) {
//             MPI_Recv(dstStart, count, T, q, tag,
//                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         }

//         MPI_Barrier(MPI_COMM_WORLD);
//     }
// }


// /***********************************************************************
//  *  Main wrapper.
//  *
//  *  Adapt only these accessors if jAxions uses different names:
//  *
//  *      axion->NZ()      local rho physical sites
//  *      axion->NX()      z sites, fast axis
//  *      axion->getNg()   rho ghost slices
//  *      axion->mCpu()    ghosted field base
//  *      axion->vCpu()    ghosted velocity base
//  *      axion->mStart()  physical field base
//  *      axion->vStart()  physical velocity base
//  *      axion->m2Cpu()   work buffer with at least Nr*Nz*2 real numbers
//  *
//  ***********************************************************************/

// void caxion_compress_mpi_serial(Scalar *axion)
// {
//     const int Nr = axion->NZ();       // local rho physical count
//     const int Nz = axion->NX();       // z count, fast axis
//     const int Ng = axion->getNg();    // replace if needed

//     int rank;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//     if (axion->Precision() == FIELD_SINGLE) {

//         const float *mCpu = static_cast<const float*>(axion->mCpu());
//         const float *vCpu = static_cast<const float*>(axion->vCpu());

//         float *mStart = static_cast<float*>(axion->mStart());
//         float *vStart = static_cast<float*>(axion->vStart());

//         float *work = static_cast<float*>(axion->m2Cpu());

//         caxion_compress_array_serial<float>(
//             mCpu, mStart, work,
//             Nr, Nz, Ng,
//             6100
//         );

//         // Important: after compressing m, src vCpu must still contain old data.
//         // If vCpu ghosts are updated independently, this is fine.
//         caxion_compress_array_serial<float>(
//             vCpu, vStart, work,
//             Nr, Nz, Ng,
//             6200
//         );

//     } else {

//         const double *mCpu = static_cast<const double*>(axion->mCpu());
//         const double *vCpu = static_cast<const double*>(axion->vCpu());

//         double *mStart = static_cast<double*>(axion->mStart());
//         double *vStart = static_cast<double*>(axion->vStart());

//         double *work = static_cast<double*>(axion->m2Cpu());

//         caxion_compress_array_serial<double>(
//             mCpu, mStart, work,
//             Nr, Nz, Ng,
//             6100
//         );

//         caxion_compress_array_serial<double>(
//             vCpu, vStart, work,
//             Nr, Nz, Ng,
//             6200
//         );
//     }

//     MPI_Barrier(MPI_COMM_WORLD);

//     if (rank == 0)
//         std::fprintf(stdout, "[caxion] MPI /2 compression finished\n");
// }






















// static constexpr int TAG0 = 3200;
// static constexpr int TAG1 = 3201;

// template<class Float>
// static inline MPI_Datatype mpi_type();

// template<>
// inline MPI_Datatype mpi_type<float>() { return MPI_FLOAT; }

// template<>
// inline MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }


// template<class Float>
// void pack_half_rho_zhalf(Float *src, Float *buf,
//                          int Nr, int Nz, int stride,
//                          int half_id)
// {
//     const int Nrh = Nr/2;
//     const int Nzh = Nz/2;
//     const int r0  = half_id * Nrh;

//     size_t p = 0;

//     for (int ir = 0; ir < Nrh; ++ir) {
//         for (int iz = 0; iz < Nzh; ++iz) {
//             const size_t idx = (size_t)(r0 + ir)*stride + iz;

//             buf[p++] = src[2*idx    ]; // Re
//             buf[p++] = src[2*idx + 1]; // Im
//         }
//     }
// }

// template<class Float>
// void unpack_to_lower_corner(Float *dst, Float *buf,
//                             int Nr, int Nz, int stride)
// {
//     const int Nrh = Nr/2;
//     const int Nzh = Nz/2;

//     size_t p = 0;

//     for (int ir = 0; ir < Nrh; ++ir) {
//         for (int iz = 0; iz < Nzh; ++iz) {
//             const size_t idx = (size_t)ir*stride + iz;

//             dst[2*idx    ] = buf[p++];
//             dst[2*idx + 1] = buf[p++];
//         }
//     }
// }

// template<class Float>
// void mpi_reduce_half_array(Float *field,
//                            int Nr, int Nz, int stride,
//                            int tag_offset)
// {
//     int rank, nRanks;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

//     if (nRanks % 2 != 0) {
//         if (rank == 0)
//             fprintf(stderr, "mpi_reduce_half_array requires even number of ranks\n");
//         MPI_Abort(MPI_COMM_WORLD, 1);
//     }

//     if ((Nr % 2) || (Nz % 2)) {
//         if (rank == 0)
//             fprintf(stderr, "mpi_reduce_half_array requires even Nr and Nz\n");
//         MPI_Abort(MPI_COMM_WORLD, 1);
//     }

//     const int Nrh = Nr/2;
//     const int Nzh = Nz/2;

//     const size_t count = (size_t)Nrh * Nzh * 2;

//     Float *recvbuf = (Float *) malloc(count * sizeof(Float));

//     if (recvbuf == nullptr) {
//         fprintf(stderr, "rank %d: malloc recvbuf failed\n", rank);
//         MPI_Abort(MPI_COMM_WORLD, 1);
//     }

//     Float *send0 = nullptr;
//     Float *send1 = nullptr;

//     MPI_Request reqs[2];
//     int nreq = 0;

//     const int source = rank/2;
//     const int chunk  = rank%2;

//     const int tag0 = TAG0 + tag_offset;
//     const int tag1 = TAG1 + tag_offset;

//     if (rank < nRanks/2) {
//         send0 = (Float *) malloc(count * sizeof(Float));
//         send1 = (Float *) malloc(count * sizeof(Float));

//         if (send0 == nullptr || send1 == nullptr) {
//             fprintf(stderr, "rank %d: malloc sendbuf failed\n", rank);
//             MPI_Abort(MPI_COMM_WORLD, 1);
//         }

//         pack_half_rho_zhalf(field, send0, Nr, Nz, stride, 0);
//         pack_half_rho_zhalf(field, send1, Nr, Nz, stride, 1);

//         const int dst0 = 2*rank;
//         const int dst1 = 2*rank + 1;

//         if (dst0 == rank) {
//             for (size_t i = 0; i < count; ++i)
//                 recvbuf[i] = send0[i];
//         } else {
//             MPI_Isend(send0, count, mpi_type<Float>(),
//                       dst0, tag0, MPI_COMM_WORLD, &reqs[nreq++]);
//         }

//         MPI_Isend(send1, count, mpi_type<Float>(),
//                   dst1, tag1, MPI_COMM_WORLD, &reqs[nreq++]);
//     }

//     if (!(source == rank && chunk == 0)) {
//         MPI_Recv(recvbuf, count, mpi_type<Float>(),
//                  source, chunk == 0 ? tag0 : tag1,
//                  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//     }

//     if (nreq > 0)
//         MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

//     unpack_to_lower_corner(field, recvbuf, Nr, Nz, stride);

//     free(recvbuf);
//     if (send0) free(send0);
//     if (send1) free(send1);

//     MPI_Barrier(MPI_COMM_WORLD);
// }

// void mpi_reduce_half(Scalar *axion)
// {
// 	const int Nr = axion->NZ();  // rho-local count, unfortunately named?
// 	const int Nz = axion->NX();  // z count, fast axis
// 	const int stride = Nz;

//     if (axion->Precision() == FIELD_SINGLE) {
//         auto *m = static_cast<float  *>(axion->mStart());
//         auto *v = static_cast<float  *>(axion->vStart());

//         mpi_reduce_half_array<float>(m, Nr, Nz, stride, 0);
//         mpi_reduce_half_array<float>(v, Nr, Nz, stride, 10);
//     } else {
//         auto *m = static_cast<double *>(axion->mStart());
//         auto *v = static_cast<double *>(axion->vStart());

//         mpi_reduce_half_array<double>(m, Nr, Nz, stride, 0);
//         mpi_reduce_half_array<double>(v, Nr, Nz, stride, 10);
//     }
// }

// static inline void quad_weights(double x, int &i0, double w[3])
// {
//     i0 = (int) floor(x);
//     const double t = x - (double)i0;

//     // stencil: i0-1, i0, i0+1
//     w[0] = 0.5*t*(t - 1.0);
//     w[1] = 1.0 - t*t;
//     w[2] = 0.5*t*(t + 1.0);
// }

// template<class Float>
// static inline void interp_quad_complex_bc(
//     const Float *src,
//     double x,
//     double z,
//     int Nr,
//     int Nz,
//     int Ng,
//     double &re,
//     double &im)
// {
//     int ix, iz;
//     double wx[3], wz[3];

//     quad_weights(x, ix, wx);
//     quad_weights(z, iz, wz);

//     re = 0.0;
//     im = 0.0;

//     for (int a = 0; a < 3; ++a) {
//         const int irs = ix + a - 1;

//         for (int b = 0; b < 3; ++b) {
//             const int izs = iz + b - 1;

//             double rr, ii;
//             get_complex_bc(src, irs, izs, Nr, Nz, Ng, rr, ii);

//             const double w = wx[a] * wz[b];

//             re += w * rr;
//             im += w * ii;
//         }
//     }
// }

// template<class Float>
// void build_refined_half_block_quad_bc(
//     const Float *src_ghost,
//     Float *dst,
//     int Nr,
//     int Nz,
//     int Ng,
//     int half_id)
// {
//     const double rho_offset = 0.5 * (double)(half_id * Nr);

//     for (int ir = 0; ir < Nr; ++ir) {
//         const double xold = 0.5*(double)ir + rho_offset;

//         for (int iz = 0; iz < Nz; ++iz) {
//             const double zold = 0.5*(double)iz;

//             double re, im;

//             interp_quad_complex_bc(
//                 src_ghost,
//                 xold, zold,
//                 Nr, Nz, Ng,
//                 re, im
//             );

//             const size_t idx = ((size_t)ir)*Nz + iz;

//             dst[2*idx    ] = (Float)re;
//             dst[2*idx + 1] = (Float)im;
//         }
//     }
// }

// template<class Float>
// void compress_array_serial(Float *field, Float *work,
//                            int Nr, int Nz, int Ng,
//                            int tag)
// {
//     int rank, nRanks;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

//     const size_t count = (size_t)Nr * Nz * 2;
//     const size_t bytes = count * sizeof(Float);

//     MPI_Datatype T = mpi_type<Float>();

//     for (int r = nRanks-1; r >= 0; --r) {

//         const int q     = r/2;   // old/source rank
//         const int chunk = r%2;   // 0 or 1 half

//         if (rank == q) {
//             build_refined_half_block_quad_bc(
//                 field,      // source with ghosts
//                 work,       // destination full block
//                 Nr, Nz, Ng,
//                 chunk
//             );

//             if (r == q) {
//                 memcpy(field, work, bytes);
//             } else {
//                 MPI_Send(work, count, T, r, tag, MPI_COMM_WORLD);
//             }
//         }

//         if (rank == r && r != q) {
//             MPI_Recv(field, count, T, q, tag,
//                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         }

//         MPI_Barrier(MPI_COMM_WORLD);
//     }
// }
