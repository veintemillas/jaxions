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
			axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fTypeP, lType, cType, parm1, parm2);
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

	double  *binarray	 ;
	trackAlloc((void**) (&binarray),  10000*sizeof(size_t));
	double *bA = static_cast<double *> (binarray);

	complex<float> *mC = static_cast<complex<float> *> (axion->mCpu());
	complex<float> *vC = static_cast<complex<float> *> (axion->mCpu());
	float *m = static_cast<float *> (axion->mCpu());
	float *v = static_cast<float *> (axion->vCpu());

	FILE *file_samp ;
	file_samp = NULL;
	file_samp = fopen("out/sample.txt","w+");
	size_t idxprint = 0 ;

	LogOut("Bins allocated! \n");




	double z_now ;

	//--------------------------------------------------
	//          SETTING BASE PARAMETERS
	//--------------------------------------------------

	double delta = sizeL/sizeN;
	double dz;
	double dzaux;
	double llphys;

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
		LogOut ("llconstantZ2 = %f - LL will be set to llphys=llconstantZ2/Z^2 \n", llconstantZ2);

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
		double	strDen;

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

			z_now = (*axion->zV());

			if (commRank() == 0 && sPrec == FIELD_SINGLE) {
					if (axion->Field() == FIELD_SAXION) {
						// LAMBDA_Z2 MODE assumed!
							llphys = llconstantZ2/(z_now*z_now);
							saskia = saxionshift(z_now, nQcd, zthres, zrestore, llphys);
							fprintf(file_samp,"%f %f %f %f %f %f %f %ld %f %e\n", z_now, axionmass(z_now,nQcd,zthres, zrestore), llphys,
							mC[idxprint + S0].real(), mC[idxprint + S0].imag(), vC[idxprint].real(), vC[idxprint].imag(), nstrings_global, maximumtheta, saskia);
					} else {
							fprintf(file_samp,"%f %f %f %f %f\n", z_now, axionmass(z_now,nQcd,zthres, zrestore),
							m[idxprint + S0], v[idxprint], maximumtheta);
						} fflush(file_samp);}


			old = std::chrono::high_resolution_clock::now();


			//--------------------------------------------------
			// DYAMICAL deltaz
			//--------------------------------------------------

			dzaux = dzSize(z_now, axion->Field(), axion->Lambda());

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
					LogOut("  str extra check (string = %d, wall = %d)\n",rts.strDen, rts.wallDn);
				}

				//--------------------------------------------------
				// TRANSITION TO THETA
				//--------------------------------------------------


				if (nstrings_global == 0)
				{

					z_now = (*axion->zV());
					llphys = llconstantZ2/(z_now*z_now); //physical value
					saskia = z_now * saxionshift(z_now, nQcd, zthres, zrestore, llphys);

								createMeas(axion, 10000);
								// IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS
								  	writeMapHdf5s (axion,sliceprint);
								//ENERGY
							  		energy(axion, eRes, false, delta, nQcd, llphys, VQCD_1, saskia);
										writeEnergy(axion, eRes);
								// BIN THETA
										Binner<float,100> thBin(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(), z_now);
										thBin.run();
										writeArray(thBin.data(), 100, "/bins", "testTh");
								destroyMeas();

					// TRANSITION TO THETA
					LogOut("--------------------------------------------------\n");
					LogOut("              TRANSITION TO THETA \n");
					LogOut("              shift = %f 			\n", saskia);
					cmplxToTheta (axion, saskia);

					// SHIFTS THETA TO A CONTINUOUS FIELD
					// REQUIRED UNFOLDED FIELDS
					munge(UNFOLD_ALL);
					axion->mendtheta();
					munge(FOLD_ALL);

								//IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS

								createMeas(axion, 10001);
								// IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS
								  	writeMapHdf5s (axion,sliceprint);
								//ENERGY
										energy(axion, eRes, false, delta, nQcd, 0., VQCD_1, 0.);
										writeEnergy(axion, eRes);
								// BIN THETA
										Binner<float,100> thBin2(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(), z_now);
										thBin2.run();
										writeArray(thBin2.data(), 100, "/bins", "testTh");
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


			// z_now = (*axion->zV());
			// llprint = llaux/(z_now*z_now); //physical value
			// saskia = saxionshift(z_now, nQcd, zthres, zrestore, llprint);
			z_now = (*axion->zV());
			llphys = llconstantZ2/(z_now*z_now); //physical value
			saskia = z_now * saxionshift(z_now, nQcd, zthres, zrestore, llphys);

			createMeas(axion, index);

			if ( axion->Field() == FIELD_SAXION)
			{
					//ENERGY
						energy(axion, eRes, false, delta, nQcd, llphys, VQCD_1, saskia);
					//DOMAIN WALL KILLER NUMBER
						double maa = 40*axionmass2(z_now,nQcd,zthres, zrestore)/(2*llphys);
						if (axion->Lambda() == LAMBDA_Z2 )
							maa = maa*z_now*z_now;
					//STRINGS
						rts = strings(axion, str);
						nstrings_global = rts.strDen;
						if (nstrings_global < 10000)
							writeString(str, rts, true);
						else
							writeString(str, rts, false);
						LogOut("%d/%d | z=%f | dz=%.3e | LLaux=%.3e | 40ma2/ms2=%.3e ", zloop, nLoops, (*axion->zV()), dzaux, llphys, maa );
						LogOut("strings %ld \n", nstrings_global);
			}
			else
			{
				//temp comment
				//BIN THETA
				//Binner<float,100> thBin2(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(), z_now);
				//thBin2.run();
				//writeArray(thBin2.data(), 100, "/bins", "testTh");
				// maximumtheta = thBin2.t1	???? ;
				LogOut("%d/%d | z=%f | dz=%.3e | maxtheta=%f | ", zloop, nLoops, (*axion->zV()), dzaux, maximumtheta);
				fflush(stdout);

				LogOut("DensMap");

				SpecBin specAna(axion, (pType & PROP_SPEC) ? true : false);

				// computes energy and creates map
				energy(axion, eRes, true, delta, nQcd, 0., VQCD_1, 0.);
				//bins density
				axion->writeMAPTHETA( (*(axion->zV() )) , index, binarray, 10000)		;
				//write binned distribution
				writeArray(bA, 10000, "/bins", "cont");
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

			writeMapHdf5s(axion,sliceprint);
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

	index++	;
	writeConf(axion, index);

	if (axion->Field() == FIELD_AXION)
	{
		createMeas(axion, index);

		writeMapHdf5s(axion,sliceprint);

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
		writeEDens(axion, index);
		writeEnergy(axion, eRes);

		printf("p Spectrum ... %d", commRank());
		specAna.pRun();
		writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");

		LogOut("|  ");
		LogOut("Theta bin ... ");

		double zNow = *axion->zV();
		Binner<float,100> thBin2(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(), zNow);
		thBin2.run();
		writeArray(thBin2.data(), 100, "/bins", "testTh");

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
	trackFree((void**) (&binarray),  ALLOC_TRACK);
	fclose(file_samp);


	delete axion;

	endAxions();

	return 0;
}
