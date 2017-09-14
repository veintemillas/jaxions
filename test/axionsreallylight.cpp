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

//#include<mpi.h>

using namespace std;

#ifdef	USE_XEON
	__declspec(target(mic)) char *mX, *vX, *m2X;
#endif

int	main (int argc, char *argv[])
{
	initAxions(argc, argv);

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	//msa = 1.7 ;
	//wDz = 0.8 ;

	commSync();
	LogOut("\n-------------------------------------------------\n");
	LogOut("\n          CREATING MINICLUSTERS!                \n\n");



	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	start = std::chrono::high_resolution_clock::now();

	Scalar *axion;
	char fileName[256];

	if ((fIndex == -1) && (cType == CONF_NONE))
		LogOut("Error: Neither initial conditions nor configuration to be loaded selected. Empty field.\n");
	else
	{
		if (fIndex == -1)
		{
			//This generates initial conditions
			LogOut("Generating scalar ... ");
			axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fType, cType, parm1, parm2);
			LogOut("Done! \n");
		}
		else
		{
			//This reads from an Axion.00000 file
			readConf(&axion, fIndex);
			if (axion == NULL)
			{
				LogOut ("Error reading HDF5 file\n");
				exit (0);
			}
		}
	}

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
	LogOut("ICtime %f min\n",elapsed.count()*1.e-3/60.);

	//--------------------------------------------------
	//          OUTPUTS FOR CHECKING
	//--------------------------------------------------

	double Vr, Vt, Kr, Kt, Grz, Gtz;
	size_t nstrings = 0 ;
	size_t nstrings_global = 0 ;

  double nstringsd = 0. ;
	double nstringsd_global = 0. ;
	double maximumtheta = 3.141597;
	size_t sliceprint = 0 ; // sizeN/2;

	// Axion spectrum
	const int kmax = axion->Length()/2 -1;
	int powmax = floor(1.733*kmax)+2 ;

	double  *spectrumK ;
	double  *spectrumG ;
	double  *spectrumV ;
	double  *binarray	 ;
	size_t  *axitonarray	 ;
	trackAlloc((void**) (&spectrumK), 8*powmax);
	trackAlloc((void**) (&spectrumG), 8*powmax);
	trackAlloc((void**) (&spectrumV), 8*powmax);
	trackAlloc((void**) (&binarray),  10000*sizeof(size_t));
	trackAlloc((void**) (&axitonarray),  100*sizeof(size_t));
	LogOut("Bins allocated! \n");

	// double *sK = static_cast<double *> (spectrumK);
	// double *sG = static_cast<double *> (spectrumG);
	// double *sV = static_cast<double *> (spectrumV);
	double *bA = static_cast<double *> (binarray);
	//double *bAd = static_cast<double *> (binarray);

	double *sK = static_cast<double *> (axion->mCpu());
	double *sG = static_cast<double *> (axion->mCpu())+powmax;
	double *sV = static_cast<double *> (axion->mCpu())+2*powmax;


	double z_now ;

	//--------------------------------------------------
	//          SETTING BASE PARAMETERS
	//--------------------------------------------------

	double delta = sizeL/sizeN;
	double dz;
	double dzaux;
	double llaux;
	double llprint;

	if (nSteps == 0)
		dz = 0.;
	else
		dz = (zFinl - zInit)/((double) nSteps);

	// LogOut("--------------------------------------------------\n");
	// LogOut("           BASE INITIAL CONDITIONS                \n\n");
	//
	// LogOut("Length =  %2.5f\n", sizeL);
	// LogOut("N      =  %ld\n",   sizeN);
	// LogOut("Nz     =  %ld\n",   sizeZ);
	// LogOut("zGrid  =  %ld\n",   zGrid);
	// LogOut("dx     =  %2.5f\n", delta);
	// LogOut("dz     =  %2.5f\n", dz);
	// LogOut("LL     =  %2.5f\n", LL);
	// LogOut("--------------------------------------------------\n");

	const size_t S0 = sizeN*sizeN;
	const size_t SF = sizeN*sizeN*(sizeZ+1)-1;
	const size_t V0 = 0;
	const size_t VF = axion->Size()-1;

	//	--------------------------------------------------
	//	TRICK PARAMETERS TO RESPECT STRINGS
	//	--------------------------------------------------

		// WE USE LAMDA_Z2 WITH msa = 1.5 so
		// zthres = z at which we reach ma^2/ms^2 =1/80=1/9*9

		//msa = 1.7 ;
		zthres 	 = 100.0 ;
		zrestore = 100.0 ;
	  double llconstantZ2 = 0.5/pow(delta/msa,2.);
		LL = pow(msa/(delta*zthres),2.)/2. ;
		LogOut ("llconstantZ2 = %f - LL will be set to llconstantZ2/Z^2 \n", llconstantZ2);

		bool coZ = 1;
	  bool coS = 1;
		bool coA = 1;
		int strcount = 0;

		int numaxiprint = 10 ;
		StringData rts ;

		axion->SetLambda(LAMBDA_Z2)	;
		if (LAMBDA_FIXED == axion->Lambda())
		{ 	LogOut ("Lambda in FIXED mode\n"); 	}
		else
		{		LogOut ("Lambda in Z2 mode\n"); 		}

	//--------------------------------------------------
	//   THE TIME ITERATION LOOP
	//--------------------------------------------------

	LogOut("--------------------------------------------------\n");
	LogOut("           STARTING REALLIGHT COMPUTATION         \n");
	LogOut("--------------------------------------------------\n");


	int counter = 0;
	int index = 0;

	commSync();

	void *eRes, *str;			// Para guardar la energia
	trackAlloc(&eRes, 128);
	memset(eRes, 0, 128);
	double *eR = static_cast<double *> (eRes);

#ifdef	__MIC__
	alignAlloc(&str, 64, (axion->Size()));
#elif defined(__AVX__)
	alignAlloc(&str, 32, (axion->Size()));
#else
	alignAlloc(&str, 16, (axion->Size()));
#endif
	memset(str, 0, axion->Size()/2);

	commSync();


	if (fIndex == -1)
	{
		LogOut ("Dumping configuration %05d ...", index);
		writeConf(axion, index);
		LogOut ("Done!\n");
		//LogOut ("Bypass configuration writting!\n");
	}
	else
		index = fIndex;

	double saskia;

//	--------------------------------------------------
//	--------------------------------------------------

	Folder munge(axion);

	if (cDev != DEV_GPU)
	{
		LogOut ("Folding configuration ... ");
		munge(FOLD_ALL);
	}
	LogOut ("Done! \n");

	if (cDev != DEV_CPU)
	{
		LogOut ("Transferring configuration to device\n");
		axion->transferDev(FIELD_MV);
	}

//	if (cDev != DEV_GPU)
	{
		double	strDen;

		if (commRank() == 0)
		{
		munge(UNFOLD_SLICE, sliceprint);
		writeMap (axion, index);
		}
	}

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
	LogOut("LL     =  %1.3e/z^2 Set to make ms*delta =%f \n\n", llconstantZ2, msa);
	LogOut("VQCD1,shift,con_thres=100, continuous theta  \n", llconstantZ2, msa);
	LogOut("--------------------------------------------------\n");

	LogOut ("Start redshift loop\n\n");
	fflush (stdout);

	commSync();

	initPropagator (pType, axion, nQcd, delta, llconstantZ2, VQCD_1);

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

			old = std::chrono::high_resolution_clock::now();

			z_now = (*axion->zV());
			//--------------------------------------------------
			// DYAMICAL deltaz
			//--------------------------------------------------

			 double masi = z_now*axionmass(z_now, nQcd, zthres, zrestore);
			 double mfre = sqrt(masi*masi + 12./(delta*delta));
			 dzaux = wDz/mfre ;

			//If SAXION_MODE
			if (axion->Field() == FIELD_SAXION && coZ)  // IF SAXION and Z2 MODE
			{
				llaux = llconstantZ2;
				llprint = llaux/(z_now*z_now); //physical value
				double mfre = sqrt( msa*msa + 12.)/delta;
				dzaux = min(dzaux,wDz/mfre)  ;
			}

			//--------------------------------------------------
			// PROPAGATOR
			//--------------------------------------------------

				propagate (axion, dzaux);

			if (axion->Field() == FIELD_SAXION)
			{
				// compute this 500 qith an educated guess
				if (nstrings_global < 500)
				{
					rts = strings(axion, str);
					nstrings_global = rts.strDen ;
					maximumtheta = axion->maxtheta();
					LogOut("  str extra check (%d) (maxth = %f)\n",nstrings_global,maximumtheta);
				}

				//--------------------------------------------------
				// TRANSITION TO THETA
				//--------------------------------------------------


				if (nstrings_global == 0)
				{

					z_now = (*axion->zV());
					llprint = llaux/(z_now*z_now); //physical value
					saskia = z_now * saxionshift(z_now, nQcd, zthres, zrestore, llprint);

								createMeas(axion, 10000);
								// IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS
										munge(UNFOLD_SLICE, sliceprint);
								  	writeMapHdf5 (axion);
								//ENERGY
							  		energy(axion, eRes, false, delta, nQcd, llaux, VQCD_1, saskia);
										writeEnergy(axion, eRes);
								// BIN THETA
										Binner<float,100> thBin(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(), z_now);
										thBin.run();
										writeArray(thBin.data(), 100, "/bins", "testTh");
								destroyMeas();

					// TRANSITION TO THETA
					LogOut("--------------------------------------------------\n");
					LogOut("              TRANSITION TO THETA \n");
					cmplxToTheta (axion, saskia);

					// SHIFTS THETA TO A CONTINUOUS FIELD
					// REQUIRED UNFOLDED FIELDS
					munge(UNFOLD_ALL);
					axion->mendtheta();
					munge(FOLD_ALL);

								//IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS

								createMeas(axion, 10001);
								// IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS
										munge(UNFOLD_SLICE, sliceprint);
								  	writeMapHdf5 (axion);
								//ENERGY
										energy(axion, eRes, false, delta, nQcd, 0., VQCD_1, 0.);
										writeEnergy(axion, eRes);
								// BIN THETA
										thBin.run();
										writeArray(thBin.data(), 100, "/bins", "testTh");
								destroyMeas();


								LogOut("--------------------------------------------------\n");

								destroyMeas();
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

//      LogOut("1IT %.3fs ETA %.3fh ",elapsed.count()*1.e-3,((nLoops-index)*dump)*elapsed.count()/(1000*60*60.));


			z_now = (*axion->zV());
			llprint = llaux/(z_now*z_now); //physical value
			saskia = saxionshift(z_now, nQcd, zthres, zrestore, llprint);

			createMeas(axion, index);

			if ( axion->Field() == FIELD_SAXION)
			{
								// if (axion->LowMem())
								// 	profiler::printMiniStats(*static_cast<double*>(axion->zV()), rts, PROF_PROP, std::string("RKN4 Saxion Lowmem"));
								// else
								// 	profiler::printMiniStats(*static_cast<double*>(axion->zV()), rts, PROF_PROP, std::string("RKN4 Saxion"));
					//ENERGY
						energy(axion, eRes, false, delta, nQcd, llaux, VQCD_1, saskia);
					//DOMAIN WALL KILLER NUMBER
						double maa = 40*axionmass(z_now,nQcd,zthres, zrestore)/(2*llaux);
						if (axion->Lambda() == LAMBDA_Z2 )
						maa = maa*z_now*z_now;
					//STRINGS
						rts = strings(axion, str);
						nstrings_global = rts.strDen ;
						LogOut("%d/%d | z=%f | dz=%.3e | LLaux=%.3e | 40ma2/ms2=%.3e ", zloop, nLoops, (*axion->zV()), dzaux, llaux, maa );
						LogOut("strings ", zloop, nLoops, (*axion->zV()), dzaux, llaux);
						LogOut("(G)= %ld \n", nstrings_global);
			}
			else
			{
									profiler::printMiniStats(*static_cast<double*>(axion->zV()), rts, PROF_PROP, std::string("RKN4 Axion"));

				LogOut("%d/%d | z=%f | dz=%.3e | maxtheta=%f | ", zloop, nLoops, (*axion->zV()), dzaux, maximumtheta);
				fflush(stdout);

				LogOut("DensMap ... ");

				SpecBin specAna(axion, (pType & PROP_SPEC) ? true : false);

				// computes energy and creates map
				energy(axion, eRes, true, delta, nQcd, 0., VQCD_1, 0.);
				//bins density
				axion->writeMAPTHETA( (*(axion->zV() )) , index, binarray, 10000)		;
				//write binned distribution
				writeArray(bA, 10000, "/bins", "cont");
				//computes power spectrum
				specAna.pRun();
				writeArray(specAna.data(SPECTRUM_P), powmax, "/pSpectrum", "sP");

				specAna.nRun();
				writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
				writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");
				writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");

				//double *eR = static_cast<double *>(eRes);

				 LogOut("| \N");
				fflush(stdout);

			}

			writeMapHdf5(axion);
			writeEnergy(axion, eRes);
			destroyMeas();




			if ((*axion->zV()) > zFinl)
			{
				LogOut("zf reached! ENDING FINALLY... \n");
				break;
			}




	} // ZLOOP

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	LogOut("\n");
	LogOut("--------------------------------------------------\n");
	LogOut("              EVOLUTION FINISHED \n");
	LogOut("--------------------------------------------------\n");
	fflush(stdout);

	LogOut("Unfold ... ");
	munge(UNFOLD_ALL);
	LogOut("| ");

	writeConf(axion, index);

	if (axion->Field() == FIELD_AXION)
	{
		createMeas(axion, index+1);
		writeMapHdf5(axion);

		printf("n Spectrum ... %d", commRank());
		/*	Test		*/
		SpecBin specAna(axion, (pType & PROP_SPEC) ? true : false);
		specAna.nRun();
		writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
		writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");
		writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");

		//double *eR = static_cast<double *>(eRes);

		LogOut("|  ");
		LogOut("DensMap ... ");

		energy(axion, eRes, true, delta, nQcd, 0., VQCD_1, 0.);
		//bins density
		axion->writeMAPTHETA( (*(axion->zV() )) , index, binarray, 10000)		;
		//write binned distribution
		writeArray(bA, 10000, "/bins", "cont");

		printf("p Spectrum ... %d", commRank());
		specAna.pRun();
		writeArray(specAna.data(SPECTRUM_P), powmax, "/pSpectrum", "sP");

		LogOut("|  ");
		LogOut("Theta bin ... ");

		double zNow = *axion->zV();
		Binner<float,100> thBin(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(), zNow);
		thBin.run();
		writeArray(thBin.data(), 100, "/bins", "testTh");

		/*	Fin test	*/

		destroyMeas();

	}

	if (cDev != DEV_GPU)
	{
		//axion->unfoldField();
		//munge(UNFOLD_ALL);
	}

	LogOut("z_final = %f\n", *axion->zV());
	LogOut("#_steps = %i\n", counter);
	LogOut("#_prints = %i\n", index);
	LogOut("Total time: %2.3f min\n", elapsed.count()*1.e-3/60.);
	LogOut("Total time: %2.3f h\n", elapsed.count()*1.e-3/3600.);

	trackFree(&eRes, ALLOC_TRACK);
	trackFree(&str,  ALLOC_ALIGN);
	trackFree((void**) (&spectrumK),  ALLOC_TRACK);
	trackFree((void**) (&spectrumG),  ALLOC_TRACK);
	trackFree((void**) (&spectrumV),  ALLOC_TRACK);
	trackFree((void**) (&binarray),  ALLOC_TRACK);
	trackFree((void**) (&axitonarray),  ALLOC_TRACK);

	delete axion;

	endAxions();

	return 0;
}
