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

#include "WKB/WKB.h"

//#include<mpi.h>

using namespace std;
using namespace AxionWKB;

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
	complex<float> *vC = static_cast<complex<float> *> (axion->vCpu());
	//Pointers
	float *m = static_cast<float *> (axion->mCpu());
	float *v = static_cast<float *> (axion->mCpu())+axion->eSize();


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
	double llphys = LL ;

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

		bool coZ = 1;
	  bool coS = 1;
		bool coA = 1;
		int strcount = 0;

		int numaxiprint = 10 ;
		StringData rts ;

		//double llconstantZ2 = 0.5/pow(delta/msa,2.);
		//axion->SetLambda(LAMBDA_Z2)	;

		// in Z2 mode LL = 0.5/pow(delta/msa,2.);
		double llconstantZ2 = LL ;

		if (LAMBDA_FIXED == axion->Lambda())
		{ 	LogOut ("Lambda in FIXED mode\n"); 	}
		else
		{		LogOut ("Lambda in Z2 mode\n");
		LogOut ("llconstantZ2 = %f - LL = llconstantZ2/Z^2 \n", llconstantZ2);
		}

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

	double saskia;
	double shiftz;

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
	if (LAMBDA_FIXED == axion->Lambda()){
	LogOut("LL     =  %f \n\n", LL);	}
	else {
	LogOut("LL     =  %1.3e/z^2 Set to make ms*delta =%f \n\n", llconstantZ2, msa); }
	LogOut("VQCD1,shift,con_thres=100, continuous theta  \n\n");
	LogOut("--------------------------------------------------\n\n");
	LogOut("           ESTIMATES  						                \n\n");
	double z_doom = pow(0.1588*msa/delta,2./(nQcd+2.))	;
	double z_axiq = pow(1./delta,2./(nQcd+2.))					;
	double z_NR   = pow(3.46/delta,2./(nQcd+2.))					;
	LogOut("z_doomsday %f \n", z_doom);
	LogOut("z_axiquenc %f \n", z_axiq);
	LogOut("z_NR       %f \n", z_NR);
	LogOut("--------------------------------------------------\n\n");


	createMeas(axion, index);
							if(p2dmapo)
								writeMapHdf5s (axion,sliceprint);
							maximumtheta = axion->thetaDIST(100, binarray); // note that bins rho 100-200
							writeArray(bA, 100, "/bins", "theta");
							writeArray(bA+100, 100, "/bins", "rho");
							writeBinnerMetadata (maximumtheta, 0., 100, "/bins");
	destroyMeas();


	LogOut ("Start redshift loop\n\n");
	fflush (stdout);

	commSync();

	// LL is LL      in FIXED MODE
	// LL is LL(z=1) in Z2 MODE (computed from msa in parse.cpp)
	initPropagator (pType, axion, nQcd, delta, LL, VQCD_1);

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


			//--------------------------------------------------
			// DYAMICAL deltaz
			//--------------------------------------------------

			dzaux = dzSize(z_now, axion->Field(), axion->Lambda());

			//--------------------------------------------------
			// PROPAGATOR
			//--------------------------------------------------

				propagate (axion, dzaux);

				if (commRank() == 0 && sPrec == FIELD_SINGLE) {
						if (axion->Field() == FIELD_SAXION) {
							// LAMBDA_Z2 MODE assumed!
								if (axion->Lambda() == LAMBDA_Z2)
									llphys = llconstantZ2/(z_now*z_now);
								saskia = saxionshift(z_now, nQcd, zthres, zrestore, llphys);
								fprintf(file_samp,"%f %f %f %f %f %f %f %ld %f %e\n", z_now, axionmass(z_now,nQcd,zthres, zrestore), llphys,
								mC[idxprint + S0].real(), mC[idxprint + S0].imag(), vC[idxprint].real(), vC[idxprint].imag(), nstrings_global, maximumtheta, saskia);
						} else {
								fprintf(file_samp,"%f %f %f %f %f\n", z_now, axionmass(z_now,nQcd,zthres, zrestore),
								m[idxprint + S0], v[idxprint], maximumtheta);
							} fflush(file_samp);}

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
					if (axion->Lambda() == LAMBDA_Z2)
						llphys = llconstantZ2/(z_now*z_now);
					saskia = saxionshift(z_now, nQcd, zthres, zrestore, llphys);
					double shiftz = z_now * saskia;

								createMeas(axion, 10000);
								// IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS
									if(p2dmapo)
										writeMapHdf5s (axion,sliceprint);
								//ENERGY
							  		energy(axion, eRes, false, delta, nQcd, llphys, VQCD_1, shiftz);
										writeEnergy(axion, eRes);
								// BIN THETA
														// new program to be adapted

														//Binner<float,100> thBin(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(), z_now);
														//thBin.run();
																//writeArray(thBin.data(), 100, "/bins", "testTh");
														//writeBinner(thBin, "/bins", "testTh");
														// old shit still works
														maximumtheta = axion->thetaDIST(100, binarray); // note that bins rho 100-200
														writeArray(bA, 100, "/bins", "theta");
														writeArray(bA+100, 100, "/bins", "rho");
														writeBinnerMetadata (maximumtheta, 0., 100, "/bins");
										destroyMeas();

					// TRANSITION TO THETA
					LogOut("--------------------------------------------------\n");
					LogOut("              TRANSITION TO THETA \n");
					LogOut("              shift = %f 			\n", saskia);
					cmplxToTheta (axion, shiftz);

					// SHIFTS THETA TO A CONTINUOUS FIELD
					// REQUIRED UNFOLDED FIELDS
					munge(UNFOLD_ALL);
					axion->mendtheta();
					munge(FOLD_ALL);


								//IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS

								createMeas(axion, 10001);
								// IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS
									if(p2dmapo)
									  	writeMapHdf5s (axion,sliceprint);
								//ENERGY
										energy(axion, eRes, false, delta, nQcd, 0., VQCD_1, 0.);
										writeEnergy(axion, eRes);
								// BIN THETA
										Binner<float,100> thBin2(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(), z_now);
										thBin2.run();
										//writeArray(thBin2.data(), 100, "/bins", "testTh");
										writeBinner(thBin2, "/bins", "testTh");
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
			if (axion->Lambda() == LAMBDA_Z2)
				llphys = llconstantZ2/(z_now*z_now);
			shiftz = z_now * saxionshift(z_now, nQcd, zthres, zrestore, llphys);

			createMeas(axion, index);

			if ( axion->Field() == FIELD_SAXION)
			{
					//THETA
						maximumtheta = axion->thetaDIST(100, binarray); // note that bins rho 100-200
						writeArray(bA, 100, "/bins", "theta");
						writeArray(bA+100, 100, "/bins", "rho");
						writeBinnerMetadata (maximumtheta, 0., 100, "/bins");
					//ENERGY
						energy(axion, eRes, false, delta, nQcd, llphys, VQCD_1, shiftz);
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
						LogOut("strings %ld [Lt^2/V] %f\n", nstrings_global, 1.5*delta*nstrings_global*z_now*z_now/(sizeL*sizeL*sizeL));
			}
			else
			{
				//temp comment
				//BIN THETA
				Binner<float,100> thBin2(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(), z_now);
				thBin2.run();
				//writeArray(thBin2.data(), 100, "/bins", "testTh");
				writeBinner(thBin2, "/bins", "testTh");
				maximumtheta = max(abs(thBin2.min()),thBin2.max());

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
			if(p2dmapo)
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
		if (pconfinal)
			writeEDens(axion, index);
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

		double zNow = *axion->zV();
		Binner<float,100> thBin2(static_cast<float *>(axion->mCpu()) + axion->Surf(), axion->Size(), zNow);
		thBin2.run();
		//writeArray(thBin2.data(), 100, "/bins", "testTh");
		writeBinner(thBin2, "/bins", "testTh");

		/*	Fin test	*/

		destroyMeas();



			//--------------------------------------------------
			// FINAL WKB
			//--------------------------------------------------

			if (wkb2z >= zFinl)
			{
						WKB wonka(axion, axion);

						LogOut ("WKBing %d to %.4f ... ", index, wkb2z);

						wonka(wkb2z) 	;

						LogOut (" done!\n", zFinl);

						index++			;
						if ( (prinoconfo >= 2) ){
									LogOut ("Dumping final WKBed configuration %05d ...", index);
									writeConf(axion, index);
									LogOut ("Done!\n");
							}


							LogOut ("Printing measurement file %05d ... ", index);
							createMeas(axion, index);
									SpecBin specAna(axion, (pType & PROP_SPEC) ? true : false);

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
									energy(axion, eRes, true, delta, nQcd, 0., VQCD_1, 0.);
									//bins density
									LogOut ("con ");
									axion->writeMAPTHETA( (*(axion->zV() )) , index, binarray, 10000)		;
									//write binned distribution
									LogOut ("bin ");
									writeArray(bA, 10000, "/bins", "cont");
									if (pconfinalwkb) {
										LogOut ("MAP ");
										writeEDens(axion, index);}

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


			// --------------------------------------------------
			// FINAL REDUCE MAP (note it reads from file)
			// --------------------------------------------------
			//
			if ( endredmap > 0)
			{
				// LogOut ("Reducing map %d to %d^3 ... ", index, endredmap);
				// 	char mirraa[128] ;
				// 	strcpy (mirraa, outName);
				// 	strcpy (outName, "./out/m/axion\0");
				// 	reduceEDens(index, endredmap, endredmap) ;
				// 	strcpy (outName, mirraa);
				// LogOut ("Done!\n");

				createMeas(axion, index+1);
				writeEDensReduced(axion, index+1, endredmap, endredmap/zGrid);
				destroyMeas();

			}

  }
	//else{} if field is saxion




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
