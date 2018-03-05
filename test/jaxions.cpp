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

#include "WKB/WKB.h"

//#include<mpi.h>

using namespace std;
using namespace AxionWKB;

void printsample(FILE *fichero, Scalar *axion, double LLL, size_t idxprint, size_t nstrings_global, double maximumtheta);
void printsample_p(FILE *fichero, Scalar *axion, double zz, double LLL, size_t idxprint, size_t nstrings_global, double maximumtheta);

int	main (int argc, char *argv[])
{
	initAxions(argc, argv);

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	commSync();
	LogOut("\n-------------------------------------------------\n");
	LogOut("\n          JAXION 3D!                             \n\n");

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

	if ((fIndex == -1) && (cType == CONF_NONE))
		LogOut("Error: Neither initial conditions nor configuration to be loaded selected. Empty field.\n");
	else
	{
		if (fIndex == -1)
		{
			//This generates initial conditions
			LogOut("Generating scalar ... ");
			axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zpreInit, lowmem, zGrid, fTypeP, lType, cType, parm1, parm2);

			LogOut("Done! \n");
		}
		else
		{
			//This reads from an axion.00000 file
			readConf(&axion, fIndex);
			if (axion == NULL)
			{
				LogOut ("Error reading HDF5 file\n");
				exit (0);
			}
			// prepropagation tends to mess up reading initial conditions
			// configurations are saved before prepropagation and have z<zInit, which readConf reverses
			// the following line fixes the issue, but a more elegant solution could be devised
			if(preprop) {
				zInit = zInit_save;
				*axion->zV() = zpreInit;
			}
		}
	}

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
	LogOut("ICtime %f min\n",elapsed.count()*1.e-3/60.);

	//--------------------------------------------------
	// USEFUL VARIABLES
	//--------------------------------------------------
	//- number of plaquetes pierced by strings
	size_t nstrings_global = 0;

	//-maximum value of the theta angle in the simulation
	double maximumtheta = 3.141597;

	//-output txt file
	FILE *file_samp ;
	file_samp = NULL;
	file_samp = fopen("out/sample.txt","w+");

	//-point to print
	size_t idxprint = 0 ;
	//- z-coordinate of the slice that is printed as a 2D map
	size_t sliceprint = 0 ;

	//-current conformal time
	double z_now ;
	//-current axion mass
	double axmass_now;
	//-grid spacing [obs?]
	double delta = sizeL/sizeN;
	//-time step
	double dz;
	//-?? [obs?]
	double dzaux;
	//-llphys = LL or LL/z^2 in LAMBDA_Z2 mode
	double llphys = LL ;

	//-set default dz [obs?]
	if (nSteps == 0)
		dz = 0.;
	else
		dz = (zFinl - zInit)/((double) nSteps);

	///-for reduced map Redondo version [obs?]
	if (endredmap > sizeN)
	{
		endredmap = sizeN;
		LogOut("[Error:1] Reduced map dimensions set to %d\n ", endredmap);
	}
	if (sizeN%endredmap != 0 )
	{
		int schei =  sizeN/endredmap;
		endredmap = sizeN/schei;
		LogOut("[Error:2] Reduced map dimensions set to %d\n ", endredmap);
	}

	//-conformal time at which axion mass is switched off and on again
	zthres 	 = 100.0 ;
	zrestore = 100.0 ;

	//-control flag to activate damping only once
	bool coD = true;

	//-number of iterations with 0 strings; used to switch to theta mode
	int strcount = 0;

	//- saves string data
	StringData rts ;
	//- obsolete
	double	strDen;

	//-LL at z=1, used for Z2 mode
	double LL1 = LL ;

	LogOut("--------------------------------------------------\n");
	LogOut("           STARTING COMPUTATION                   \n");
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

	if (fIndex == -1)
	{
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

	//-stores shift in real part of PQ and cPQ (conformal field) [obs? move up]
	double saskia = 0.0;
	double shiftz = 0.0;

//	--------------------------------------------------
//	--------------------------------------------------

	Folder munge(axion);

	if (cDev != DEV_GPU)
	{
		LogOut ("Folding configuration ... ");
		munge(FOLD_ALL);
	}

	if (cDev != DEV_CPU)
	{
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
	LogOut("Length =  %2.2f\n", sizeL);
	LogOut("nQCD   =  %2.2f\n", nQcd);
	LogOut("N      =  %ld\n",   sizeN);
	LogOut("Nz     =  %ld\n",   sizeZ);
	LogOut("zGrid  =  %ld\n",   zGrid);
	LogOut("dx     =  %2.5f\n", delta);
	LogOut("dz     =  %2.2f/FREQ\n", wDz);
	if (LAMBDA_FIXED == axion->Lambda()){
	LogOut("LL     =  %.0f \n\n", LL);	}
	else {
	LogOut("LL     =  %1.3e/z^2 Set to make ms*delta =%.2f \n\n", LL1, msa); }
	if ((vqcdType & VQCD_TYPE) == VQCD_1)
		LogOut("VQCD1PQ1,shift,continuous theta  \n\n");
	else if((vqcdType & VQCD_TYPE) == VQCD_2)
		LogOut("VQCD2PQ1,no shift, continuous theta  \n\n");
	else if((vqcdType & VQCD_TYPE) == VQCD_1_PQ_2)
		LogOut("VQCD1PQ2,shift, continuous theta  \n\n");
	LogOut("Vqcd flag %d\n", vqcdType);
	LogOut("Damping %d gam = %f\n", vqcdType & VQCD_DAMP, gammo);
	LogOut("--------------------------------------------------\n\n");
	LogOut("           ESTIMATES  						                \n\n");
	double z_doom;
	if ((vqcdType & VQCD_TYPE) == VQCD_1_PQ_2)
	z_doom = pow(0.1588*2.0*msa/delta,2./(nQcd+2.))	;
	else
	z_doom = pow(0.1588*msa/delta,2./(nQcd+2.))	;
	double z_axiq = pow(1./delta,2./(nQcd+2.))					;
	double z_NR   = pow(3.46/delta,2./(nQcd+2.))					;
	LogOut("z_doomsday %f \n", z_doom);
	LogOut("z_axiquenc %f \n", z_axiq);
	LogOut("z_NR       %f \n", z_NR);
	LogOut("--------------------------------------------------\n\n");

	commSync();

	//--------------------------------------------------
	// prepropagator with relaxing strong damping
	//--------------------------------------------------
	// only if preprop and if z smaller or equal than zInit
	// When z>zInit, it is understood that prepropagation was done
	if (preprop && ((*axion->zV()) < zInit)) {
		LogOut("pppp Preprocessing ... z=%.f->%.f (VQCDTYPE %d, gam=%.f) \n\n", (*axion->zV()), zInit,
					(vqcdType & VQCD_TYPE) | VQCD_DAMP_RHO, gammo);
		double gammo_save = gammo ;
		double *zaza = axion->zV();
		double strdensn ;
		initPropagator (pType, axion, nQcd, delta, LL, gammo, (vqcdType & VQCD_TYPE) | VQCD_DAMP_RHO);
		tunePropagator (axion);

		while ( *zaza < zInit )
		{
			dzaux = dzSize(zInit, axion->Field(), axion->Lambda(),vqcdType)/2.;
			gammo = gammo_save*pow(abs(1.0 - (*zaza)/zInit)/(1. - 1./prepcoe),1.5);

			printsample(file_samp, axion, LL, idxprint, nstrings_global, maximumtheta);

			if (icstudy)
			{
				// string control
				rts = strings(axion);
				nstrings_global = rts.strDen;
				strdensn = 0.75*delta*nstrings_global*(*zaza)*(*zaza)/(sizeL*sizeL*sizeL);
				LogOut("z %f strings %ld [Lz^2/V] %f (gammo %f)\n", *zaza, nstrings_global, strdensn, gammo);

			if (axion->Lambda() == LAMBDA_Z2)
				llphys = LL1/((*zaza)*(*zaza));
			axmass_now = axionmass(z_now,nQcd,zthres, zrestore);
			saskia = saxionshift(axmass_now, llphys, vqcdType);
			shiftz = z_now * saskia;

			createMeas(axion, index);
					if ( axion->Field() == FIELD_SAXION)
					{
							writeString(axion, rts, false);
							energy(axion, eRes, false, delta, nQcd, LL, vqcdType, shiftz);
					}
					if(p2dmapo)
						writeMapHdf5s(axion,sliceprint);
					writeEnergy(axion, eRes);
			destroyMeas();
			index++;
			}
			else
			{
				LogOut("z %f (gammo %f)\n", *zaza, gammo);
			}
			propagate (axion, dzaux);

		}

	gammo = gammo_save = gammo ;

	}

	LogOut("First measurement file %d \n",index);
	createMeas(axion, index);
							rts = strings(axion);
							nstrings_global = rts.strDen;
							writeString(axion, rts, false);
							energy(axion, eRes, false, delta, nQcd, LL, vqcdType, shiftz);
							writeEnergy(axion, eRes);

							if(p2dmapo)
								writeMapHdf5s (axion,sliceprint);
							{
								float z_now = *axion->zV();
								Binner<100,complex<float>> rhoBin(static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
													[z=z_now] (complex<float> x) { return (double) abs(x)/z; } );
								rhoBin.run();
								writeBinner(rhoBin, "/bins", "rhoB");

								Binner<100,complex<float>> thBin(static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
												 [] (complex<float> x) { return (double) arg(x); });
								thBin.run();
								writeBinner(thBin, "/bins", "thetaB");
							}
	destroyMeas();


	// damping only from zst1000
	LogOut("Running ...\n\n");
	initPropagator (pType, axion, nQcd, delta, LL, gammo, vqcdType & VQCD_TYPE);
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

	dzaux = dzSize(z_now, axion->Field(), axion->Lambda(),vqcdType);

	// PROPAGATOR
	propagate (axion, dzaux);

	// SIMPLE OUTPUT CHECK
	z_now = (*axion->zV());
	printsample(file_samp, axion, LL1, idxprint, nstrings_global, maximumtheta);

	// CHECKS IF SAXION
	if (axion->Field() == FIELD_SAXION)
	{
		// IF FEW STRINGS COMPUTES THE NUMBER EVERY ITERATION [obs with damping]
		if (nstrings_global < 1000)
		{
			rts = strings(axion);
			nstrings_global = rts.strDen ;
			LogOut("  str extra check (string = %d, wall = %d)\n",rts.strDen, rts.wallDn);
		}

		// BEFORE UNPPHYSICAL DW DESTRUCTION, ACTIVATES DAMPING TO DAMP SMALL DW'S
		if ( ((*axion->zV()) > z_doom*0.95) && (coD) && ( (vqcdType & VQCD_DAMP) != VQCD_NONE ) )
		{
			LogOut("-----------------------------------------\n");
			LogOut("DAMPING ON (gam = %f, z ~ 0.95*z_doom %f)\n", gammo, 0.95*z_doom);
			LogOut("-----------------------------------------\n");
			initPropagator (pType, axion, nQcd, delta, LL, gammo, vqcdType );
			coD = false ;
		}

		// TRANSITION TO THETA COUNTER
		if (nstrings_global == 0) {
			LogOut("  no st counter %d\n", strcount);
			strcount++;
		}
		// IF CONF_SAXNOISE we do not ever switch to theta to follow the evolution of saxion field
		if (smvarType != CONF_SAXNOISE)
		if (nstrings_global == 0 && strcount > safest0)
		{
			if (axion->Lambda() == LAMBDA_Z2)
				llphys = LL1/((*axion->zV())*(*axion->zV()));
			axmass_now = axionmass(z_now,nQcd,zthres, zrestore);
			saskia = saxionshift(axmass_now, llphys, vqcdType);
			shiftz = z_now * saskia;

			createMeas(axion, 10000);
				// IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS
				if(p2dmapo)
				writeMapHdf5s (axion,sliceprint);
				//ENERGY
				energy(axion, eRes, false, delta, nQcd, LL, vqcdType, shiftz);
				writeEnergy(axion, eRes);
				// BIN RHO+THETA
				float z_now = *axion->zV();
				float shiftzf = shiftz ;
				Binner<100,complex<float>> rhoBin(static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
								  [z=z_now,ss=shiftzf] (complex<float> x) { return (double) abs(x-ss)/z; });
				rhoBin.run();
				writeBinner(rhoBin, "/bins", "rhoB");

				Binner<100,complex<float>> thBin(static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
								 [ss=shiftzf] (complex<float> x) { return (double) arg(x-ss); });
				thBin.run();
				writeBinner(thBin, "/bins", "thetaB");
			destroyMeas();

			LogOut("--------------------------------------------------\n");
			LogOut(" TRANSITION TO THETA (z=%.4f)\n",z_now);
			LogOut(" shift = %f \n", saskia);

			cmplxToTheta (axion, shiftz);

			//IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS

			createMeas(axion, 10001);
			// IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS
				if(p2dmapo)
				  	writeMapHdf5s (axion,sliceprint);
						//ENERGY
				energy(axion, eRes, false, delta, nQcd, 0., vqcdType, 0.);
				writeEnergy(axion, eRes);
			// BIN THETA
				Binner<100,float> thBin2(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(),
							 [z=z_now] (float x) -> float { return (float) (x/z); });
				thBin2.run();
				writeBinner(thBin2, "/bins", "thetaB");
			destroyMeas();

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

} // ZSUBLOOP

	//--------------------------------------------------
	// PARTIAL ANALISIS
	//--------------------------------------------------

	z_now = (*axion->zV());
	if (axion->Lambda() == LAMBDA_Z2)
		llphys = LL1/(z_now*z_now);
	axmass_now = axionmass(z_now,nQcd,zthres, zrestore);
	saskia = saxionshift(axmass_now, llphys, vqcdType);
	shiftz = z_now * saskia;

	if ((*axion->zV()) > zFinl)
	{
		// THE LAST MEASURE IS DONE AT THE END
		LogOut("zf reached! ENDING FINALLY... \n");
		break;
	}

	createMeas(axion, index);

		if ( axion->Field() == FIELD_SAXION)
		{
			// BIN RHO+THETA
			float z_now = *axion->zV();
			float shiftzf = shiftz ;
			Binner<100,complex<float>> rhoBin(static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
								[z=z_now,ss=shiftzf] (complex<float> x) { return (double) abs(x-ss)/z; });
			rhoBin.run();
			writeBinner(rhoBin, "/bins", "rhoB");

			Binner<100,complex<float>> thBin(static_cast<complex<float> *>(axion->mCpu()) + axion->Surf(), axion->Size(),
							 [ss=shiftzf] (complex<float> x) { return (double) arg(x-ss); });
			thBin.run();
			writeBinner(thBin, "/bins", "thetaB");
		//ENERGY
			energy(axion, eRes, false, delta, nQcd, LL, vqcdType, shiftz);
		//DOMAIN WALL KILLER NUMBER
			double maa = 40*axionmass2(z_now,nQcd,zthres, zrestore)/(2*llphys);
			if (axion->Lambda() == LAMBDA_Z2 )
				maa = maa*z_now*z_now;
		//STRINGS
			rts = strings(axion);
			nstrings_global = rts.strDen;
			if (p3DthresholdMB/((double) nstrings_global) > 1.)
				writeString(axion, rts, true);
			else
				writeString(axion, rts, false);

			LogOut("%d/%d | z=%f | dz=%.3e | LLaux=%.3e | 40ma2/ms2=%.3e ", zloop, nLoops, (*axion->zV()), dzaux, llphys, maa );
			LogOut("strings %ld [Lt^2/V] %f\n", nstrings_global, 0.75*delta*nstrings_global*z_now*z_now/(sizeL*sizeL*sizeL));
		}
		else //( axion->Field() == FIELD_AXION)
		{
			//BIN THETA
			Binner<100,float> thBin2(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(),
						 [z=z_now] (float x) -> float { return (float) (x/z);});
			thBin2.run();
			writeBinner(thBin2, "/bins", "thetaB");
			maximumtheta = max(abs(thBin2.min()),thBin2.max());

			LogOut("%d/%d | z=%f | dz=%.3e | maxtheta=%f | ", zloop, nLoops, (*axion->zV()), dzaux, maximumtheta);
			fflush(stdout);

			LogOut("DensMap");

			SpecBin specAna(axion, (pType & PROP_SPEC) ? true : false);

			// computes energy and creates map
			energy(axion, eRes, true, delta, nQcd, 0., vqcdType, 0.);
			{
				double *eR = static_cast<double*>(eRes);
				float eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);
				Binner<3000,float> contBin(static_cast<float *>(axion->m2Cpu()), axion->Size(),
							    [eMean = eMean] (float x) -> float { return (double) (log10(x/eMean) );});
				contBin.run();
				writeBinner(contBin, "/bins", "contB");
			}
			//computes power spectrum
			specAna.pRun();
			writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");

			specAna.nRun();
			writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
			writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");
			writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");


			 LogOut("| \n");
			fflush(stdout);

		}
		if(p2dmapo)
			writeMapHdf5s(axion,sliceprint);
		writeEnergy(axion, eRes);
	destroyMeas();

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

		if ( (prinoconfo >= 2) && (wkb2z < 0)  ){
					LogOut ("Dumping final configuration %05d ...", index);
					writeConf(axion, index);
					LogOut ("Done!\n");
		}


		createMeas(axion, index);
			if(p2dmapo)
				writeMapHdf5s(axion,sliceprint);

			LogOut("n Spectrum ... ");
			SpecBin specAna(axion, (pType & PROP_SPEC) ? true : false);
			specAna.nRun();
			writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
			writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");
			writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");
			LogOut("DensMap ... ");
			energy(axion, eRes, true, delta, nQcd, 0., vqcdType, 0.);
			{
				float eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);
				Binner<3000,float> contBin(static_cast<float *>(axion->m2Cpu()), axion->Size(),
							    [eMean = eMean] (float x) -> float { return (double) (log10(x/eMean) );});
				contBin.run();
				writeBinner(contBin, "/bins", "contB");
			}
			if (pconfinal)
				writeEDens(axion);
			writeEnergy(axion, eRes);

			LogOut("p Spectrum ... ");
			specAna.pRun();
			writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");

			if ( endredmap > 0){
				LogOut("filtering to reduce map with %d neighbours ... ", sizeN/endredmap);
				int nena = sizeN/endredmap ;
				specAna.filter(nena);
				writeEDensReduced(axion, index, endredmap, endredmap/zGrid);
			}

			LogOut("|  ");
			LogOut("Theta bin ... ");

			double z_now = *axion->zV();
			Binner<100,float> thBin2(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(),
						 [z=z_now] (float x) -> float { return (float) (x/z); });
			thBin2.run();
			//writeArray(thBin2.data(), 100, "/bins", "testTh");
			writeBinner(thBin2, "/bins", "thetaB");

		destroyMeas();



	//--------------------------------------------------
	// FINAL WKB
	//--------------------------------------------------

		if (wkb2z >= zFinl)
		{
				WKB wonka(axion, axion);

				LogOut ("WKBing %d (z=%.4f) to %d (%.4f) ... ", index, z_now, index+1, wkb2z);

				wonka(wkb2z) 	;
				z_now = (*axion->zV());
				LogOut (" done! (z=%.4f)\n", z_now);

				index++			;
				if ( (prinoconfo >= 2) ){
							LogOut ("Dumping final WKBed configuration %05d ...", index);
							writeConf(axion, index);
							LogOut ("Done!\n");
					}

				LogOut ("Printing measurement file %05d ... ", index);

				createMeas(axion, index);
					SpecBin specAna(axion, (pType & PROP_SPEC) ? true : false);

					LogOut("theta ");
					Binner<100,float> thBin2(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(),
								 [z=z_now] (float x) -> float { return (float) (x/z);});
					thBin2.run();
					//writeArray(thBin2.data(), 100, "/bins", "testTh");
					writeBinner(thBin2, "/bins", "thetaB");

					LogOut ("spec ");
					specAna.nRun();
					writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
					writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");
					writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");
					if(p2dmapo){
						LogOut ("2D ");
						writeMapHdf5s(axion,sliceprint);
						LogOut ("Done!\n");}

					// computes energy and creates map
					LogOut ("en ");
					energy(axion, eRes, true, delta, nQcd, 0., vqcdType, 0.);
					{
						float eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);
						Binner<3000,float> contBin(static_cast<float *>(axion->m2Cpu()), axion->Size(),
									    [eMean = eMean] (float x) -> float { return (double) (log10(x/eMean) );});
						contBin.run();
						writeBinner(contBin, "/bins", "contB");
					}

					if (pconfinalwkb) {
						LogOut ("MAP ");
						writeEDens(axion);}

					LogOut ("tot ");
					writeEnergy(axion, eRes);
					//computes power spectrum
					LogOut ("pow ");
					specAna.pRun();
					writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");

					if ( endredmap > 0 ){
						LogOut("redmap ");
						int nena = sizeN/endredmap ;
						specAna.filter(nena);
						writeEDensReduced(axion, index, endredmap, endredmap/zGrid);
					}
					destroyMeas();
			}

  }
	//else{} if field is saxion NOTHING IS DONE

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
					double axmass_now = axionmass(z_now,nQcd,zthres, zrestore);
					double saskia = saxionshift(axmass_now, llphys, vqcdType);
					fprintf(fichero,"%f %f %f %f %f %f %f %ld %f %e\n", z_now, axmass_now, llphys,
					static_cast<complex<float> *> (axion->mCpu())[idxprint + S0].real(),
					static_cast<complex<float> *> (axion->mCpu())[idxprint + S0].imag(),
					static_cast<complex<float> *> (axion->vCpu())[idxprint].real(),
					static_cast<complex<float> *> (axion->vCpu())[idxprint].imag(),
					nstrings_global, maximumtheta, saskia);
			} else {
					fprintf(fichero,"%f %f %f %f %f\n", z_now, axionmass(z_now,nQcd,zthres, zrestore),
					static_cast<float *> (axion->mCpu())[idxprint + S0],
					static_cast<float *> (axion->vCpu())[idxprint], maximumtheta);
				} fflush(fichero);}
}

// void printmeasure(int index, Scalar *axion)
// {
//
// }

void printsample_p(FILE *fichero, Scalar *axion, double zz, double LLL, size_t idxprint, size_t nstrings_global, double maximumtheta)
{
	double z_now = (*axion->zV());
	double llphys = LLL;
	if (axion->Lambda() == LAMBDA_Z2)
		llphys = LLL/(z_now*z_now);
	size_t S0 = sizeN*sizeN ;
	if (commRank() == 0 && sPrec == FIELD_SINGLE) {
			if (axion->Field() == FIELD_SAXION) {
					double axmass_now = axionmass(z_now,nQcd,zthres, zrestore);
					double saskia = saxionshift(axmass_now, llphys, vqcdType);
					fprintf(fichero,"%f %f %f %f %f %f %f %ld %f %e\n", zz, axmass_now, llphys,
					static_cast<complex<float> *> (axion->mCpu())[idxprint + S0].real()*zz/z_now,
					static_cast<complex<float> *> (axion->mCpu())[idxprint + S0].imag()*zz/z_now,
					static_cast<complex<float> *> (axion->vCpu())[idxprint].real()*zz/z_now,
					static_cast<complex<float> *> (axion->vCpu())[idxprint].imag()*zz/z_now,
					nstrings_global, maximumtheta, saskia);
			} else {
					fprintf(fichero,"%f %f %f %f %f\n", z_now, axionmass(z_now,nQcd,zthres, zrestore),
					static_cast<float *> (axion->mCpu())[idxprint + S0],
					static_cast<float *> (axion->vCpu())[idxprint], maximumtheta);
				} fflush(fichero);}
}
