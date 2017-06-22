#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "propagator/allProp.h"
#include "energy/energy.h"
#include "energy/energyMap.h"
#include "enum-field.h"
#include "utils/utils.h"
#include "io/readWrite.h"
#include "comms/comms.h"
#include "map/map.h"
#include "strings/strings.h"
#include "powerCpu.h"
#include "scalar/scalar.h"


#include<mpi.h>

using namespace std;

#ifdef	USE_XEON
	__declspec(target(mic)) char *mX, *vX, *m2X;
#endif

#define printMpi(...) do {		\
	if (!commRank()) {		\
	  printf(__VA_ARGS__);  	\
	  fflush(stdout); }		\
}	while (0)


int	main (int argc, char *argv[])
{
	parseArgs(argc, argv);

	if (initComms(argc, argv, zGrid, cDev) == -1)
	{
		printf ("Error initializing devices and Mpi\n");
		return 1;
	}

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	commSync();
	printMpi("\n-------------------------------------------------\n");
	printMpi("\n          CREATING MINICLUSTERS!                \n\n");



	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	FlopCounter *fCount = new FlopCounter;

	start = std::chrono::high_resolution_clock::now();

	Scalar *axion;
	char fileName[256];

	if ((initFile == NULL) && (fIndex == -1) && (cType == CONF_NONE))
		printMpi("Error: Neither initial conditions nor configuration to be loaded selected. Empty field.\n");
	else
	{
		if (fIndex == -1)
		{
			//This generates initial conditions
			printMpi("Generating scalar ... ");
			axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fType, cType, parm1, parm2, fCount);
			printMpi("Done! \n");
		}
		else
		{
			//This reads from an Axion.00000 file
			readConf(&axion, fIndex);
			if (axion == NULL)
			{
				printMpi ("Error reading HDF5 file\n");
				exit (0);
			}
		}
	}

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
	printMpi("ICtime %f min\n",elapsed.count()*1.e-3/60.);

	//--------------------------------------------------
	//          OUTPUTS FOR CHECKING
	//--------------------------------------------------

	FILE *file_sample ;
	file_sample = NULL;

	FILE *file_energy ;
	file_energy = NULL;

	//energy 2//
	//FILE *file_energy2 ;
	//energy 2//
	//file_energy2 = NULL;

	FILE *file_spectrum ;
	file_spectrum = NULL;

	FILE *file_power ;
	file_power = NULL;

	FILE *file_thetabin ;
	file_thetabin = NULL;

	FILE *file_contbin ;
	file_contbin = NULL;


	if (commRank() == 0)
	{
		file_sample = fopen("out/sample.txt","w+");
		//fprintf(file_sample,"%f %f %f\n",z, creal(m[0]), cimag(m[0]));
		file_energy = fopen("out/energy.txt","w+");
		//fprintf(file_sample,"%f %f %f\n",z, creal(m[0]), cimag(m[0]));
		//energy 2//
		//file_energy2 = fopen("out/energy2.txt","w+");
		file_spectrum = fopen("out/spectrum.txt","w+");
		file_power = fopen("out/power.txt","w+");
		file_thetabin = fopen("out/thetabin.txt","w+");
		file_contbin = fopen("out/contbin.txt","w+");
	}
	printMpi("Files prepared! \n");

	double Vr, Vt, Kr, Kt, Grz, Gtz;
	size_t nstrings = 0 ;
	size_t nstrings_global = 0 ;

  double nstringsd = 0. ;
	double nstringsd_global = 0. ;
	double maximumtheta = 3.141597;
	size_t sliceprint = 1;

	// Axion spectrum
	const int kmax = axion->Length()/2 -1;
	int powmax = floor(1.733*kmax)+2 ;

	double  *spectrumK ;
	double  *spectrumG ;
	double  *spectrumV ;
	double  *binarray	 ;
	trackAlloc((void**) (&spectrumK), 8*powmax);
	trackAlloc((void**) (&spectrumG), 8*powmax);
	trackAlloc((void**) (&spectrumV), 8*powmax);
	trackAlloc((void**) (&binarray),  10000*sizeof(size_t));
	printMpi("Bins allocated! \n");

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

	printMpi("--------------------------------------------------\n");
	printMpi("           BASE INITIAL CONDITIONS                \n\n");

	printMpi("Length =  %2.5f\n", sizeL);
	printMpi("N      =  %ld\n",   sizeN);
	printMpi("Nz     =  %ld\n",   sizeZ);
	printMpi("zGrid  =  %ld\n",   zGrid);
	printMpi("dx     =  %2.5f\n", delta);
	printMpi("dz     =  %2.5f\n", dz);
	printMpi("LL     =  %2.5f\n", LL);
	printMpi("--------------------------------------------------\n");

	const size_t S0 = sizeN*sizeN;
	const size_t SF = sizeN*sizeN*(sizeZ+1)-1;
	const size_t V0 = 0;
	const size_t VF = axion->Size()-1;

	double saskia;


	// printMpi("INITIAL CONDITIONS LOADED\n");
	// if (sPrec != FIELD_DOUBLE)
	// {
	// 	printMpi("Example mu: m[0] = %f + %f*I, m[N3-1] = %f + %f*I\n", ((complex<float> *) axion->mCpu())[S0].real(), ((complex<float> *) axion->mCpu())[S0].imag(),
	// 								        ((complex<float> *) axion->mCpu())[SF].real(), ((complex<float> *) axion->mCpu())[SF].imag());
	// 	printMpi("Example  v: v[0] = %f + %f*I, v[N3-1] = %f + %f*I\n", ((complex<float> *) axion->vCpu())[V0].real(), ((complex<float> *) axion->vCpu())[V0].imag(),
	// 								        ((complex<float> *) axion->vCpu())[VF].real(), ((complex<float> *) axion->vCpu())[VF].imag());
	// }
	// else
	// {
	// 	printMpi("Example mu: m[0] = %lf + %lf*I, m[N3-1] = %lf + %lf*I\n", ((complex<double> *) axion->mCpu())[S0].real(), ((complex<double> *) axion->mCpu())[S0].imag(),
	// 									    ((complex<double> *) axion->mCpu())[SF].real(), ((complex<double> *) axion->mCpu())[SF].imag());
	// 	printMpi("Example  v: v[0] = %lf + %lf*I, v[N3-1] = %lf + %lf*I\n", ((complex<double> *) axion->vCpu())[V0].real(), ((complex<double> *) axion->vCpu())[V0].imag(),
	// 									    ((complex<double> *) axion->vCpu())[VF].real(), ((complex<double> *) axion->vCpu())[VF].imag());
	// }

	//JAVIER commented next
	//printMpi("Ez     =  %ld\n",    axion->eDepth());


	// for (i=0; i<100;i++)
	// {
	// 	printMpi("%f",saxionshift(z_now, nQcd, 0, 3., LL);)
	// }


	//--------------------------------------------------
	//   THE TIME ITERATION LOOP
	//--------------------------------------------------

	printMpi("--------------------------------------------------\n");
	printMpi("           STARTING COMPUTATION                   \n");
	printMpi("--------------------------------------------------\n");


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
		//printMpi ("Dumping configuration %05d ...", index);
		//writeConf(axion, index);
		//printMpi ("Done!\n");
		printMpi ("Bypass configuration writting!\n");
		fflush (stdout);
	}
	else
		index = fIndex + 1;

	//JAVIER commented next
	//printf ("Process %d reached syncing point\n", commRank());
	//fflush (stdout);
//	commSync();

//	--------------------------------------------------
//	TRICK PARAMETERS TO RESPECT STRINGS
//	--------------------------------------------------

	//THIS STRATEGY HAS THREE STEPS
	// WE USE LAMDA_Z2 (LL/Z^2) WITH msa = 1.5
	// 	THIS FIXES A NEW LL
	// 	AS THE AXION MASS INCREASES, THERE WILL BE A POINT WHEN ma^2/ms^2 =1/80=1/9*9

	// AT THIS zthreshold, WE SWITCH OFF THE GROWTH OF AXION MASS AND THE DECREASE OF LAMBDA
	// 	zthres = z at which we reach ma^2/ms^2 =1/80=1/9*9
	//
	// WHEN STRINGS HAVE ALL DECAYED WE TURN ON THE MASS INCREASE AGAIN
	// ABOVE zrestore


	double msa = 2.0 ;
	double llconstantZ2 = 0.5/pow(delta/msa,2.);
	printMpi ("llconstantZ2 in Z2 mode set to %f\n",  llconstantZ2);

	printMpi ("zth-zres (%f,%f) changed to ", zthres, zrestore);
	zthres = pow(msa*sizeN/(indi3*sizeL*9.),2./(nQcd+2.)) ;
	zrestore = 3.0 ;

	printMpi ("(%f,%f)  \n", zthres, zrestore);

	LL = llconstantZ2/pow(zthres,2.) ;
	printMpi ("LL reset to %f \n", LL);

	bool coZ = 1;
  bool coS = 1;
	int strcount = 0;


	axion->SetLambda(LAMBDA_Z2)	;
	if (LAMBDA_FIXED == axion->Lambda())
	{ 	printMpi ("Lambda in FIXED mode\n"); 	}
	else
	{		printMpi ("Lambda in Z2 mode\n"); 		}



//	--------------------------------------------------
//	--------------------------------------------------


	Folder munge(axion);

	if (cDev != DEV_GPU)
	{
		printMpi ("Folding configuration ... ");
		munge(FOLD_ALL);
	}
	printMpi ("Done! \n");

	if (cDev != DEV_CPU)
	{
		printMpi ("Transferring configuration to device\n");
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


	printMpi("--------------------------------------------------\n");
	printMpi("           START LOOP  						                \n\n");
	printMpi("Length =  %2.5f\n", sizeL);
	printMpi("N      =  %ld\n",   sizeN);
	printMpi("Nz     =  %ld\n",   sizeZ);
	printMpi("zGrid  =  %ld\n",   zGrid);
	printMpi("dx     =  %2.5f\n", delta);
	printMpi("dz     =  variable min[1/d,1/m_a,1/m_s]/2\n", dz);
	printMpi("LL     =  FM_variable ms a =%f\n",msa);
	printMpi("--------------------------------------------------\n");

	printMpi ("Start redshift loop\n\n");
	fflush (stdout);

	commSync();

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


			//--------------------------------------------------
			// DYAMICAL deltaz
			//--------------------------------------------------

			z_now = (*axion->zV());

			//dzaux = min(delta,1./(3.*pow((*axion->zV()),1.+nQcd/2.)));
			dzaux = min(delta/1.5,1./(z_now*axionmass(z_now,nQcd,zthres, zrestore)));
			if (axion->Field() == FIELD_SAXION && coZ)  // IF SAXION and Z2 MODE
			{
				// llaux is the (ARGUMENT) I have to feed the propagator that uses LL=(ARGUMENT)/z^2 IN LAMBDA_Z2 mode
				llaux = llconstantZ2;
				llprint = llaux/(z_now*z_now); //physical value
				dzaux = min(dzaux,delta);
			}

			//printMpi("(dz0,dz1,dz2)= (%f,%f,%f) ", delta, 1./(sqrt(LL)*(*axion->zV())) ,1./(9.*pow((*axion->zV()),nQcd)));
			// HERE I CHANGE THE SAXION MASS ABOVE z_thres
			// NOTE THAT I DEFINED zthr as LL = llconstantZ2/pow(zthr,2.)
			if ((axion->Field() == FIELD_SAXION) && (z_now > zthres) && coZ )
			{
				axion->SetLambda(LAMBDA_FIXED)	;
				printMpi("Lambda Fixed transition at %f \n", (*axion->zV()));
				coZ = 0;
			}

			if ( axion->Field() == FIELD_SAXION && (!coZ) ) // IF SAXION and LAMBDA MODE
			{
				// llaux is the (ARGUMENT) I have to feed the propagator that uses LL=(ARGUMENT) IN LAMBDA_FIXED mode
				llaux = LL;
				llprint = LL; //physical value
        dzaux = min(dzaux,1./(sqrt(2.*LL)*z_now));
				printMpi("*");
			}
        dzaux = dzaux/2.0;




				//--------------------------------------------------
				// PRINT POINT
				//--------------------------------------------------


				if (commRank() == 0)
					{

						if (axion->Field() == FIELD_SAXION)
						{

							 saskia = saxionshift(z_now, nQcd, zthres, zrestore, llprint);

							if (sPrec == FIELD_DOUBLE) {
								fprintf(file_sample,"%f %f %f %f %f %f %f %ld %f %lf\n",z_now, axionmass(z_now,nQcd,zthres, zrestore), llprint,
								static_cast<complex<double> *> (axion->mCpu())[sliceprint*S0+S0].real(), static_cast<complex<double> *> (axion->mCpu())[sliceprint*S0+S0].imag(),
								static_cast<complex<double> *> (axion->vCpu())[sliceprint*S0].real(), static_cast<complex<double> *> (axion->vCpu())[sliceprint*S0].imag(),
								nstrings_global, maximumtheta, saskia );
							} else {
								fprintf(file_sample,"%f %f %f %f %f %f %f %ld %f %lf\n",z_now, axionmass(z_now,nQcd,zthres, zrestore), llprint,
								static_cast<complex<float>  *> (axion->mCpu())[sliceprint*S0+S0].real(), static_cast<complex<float>  *> (axion->mCpu())[sliceprint*S0+S0].imag(),
								static_cast<complex<float>  *> (axion->vCpu())[sliceprint*S0].real(), static_cast<complex<float>  *> (axion->vCpu())[sliceprint*S0].imag(),
								nstrings_global, maximumtheta, saskia);
							}
						}
						else
						{
							if (sPrec == FIELD_DOUBLE) {
								fprintf(file_sample,"%f %f %f %f %f\n", z_now, axionmass(z_now,nQcd,zthres, zrestore),
								static_cast<double*> (axion->mCpu())[sliceprint*S0+S0], static_cast<double*> (axion->vCpu())[sliceprint*S0],
								maximumtheta);
							} else {
								fprintf(file_sample,"%f %f %f %f %f\n",z_now, axionmass(z_now,nQcd,zthres, zrestore),
								static_cast<float*> (axion->mCpu())[sliceprint*S0+S0], static_cast<float*> (axion->vCpu())[sliceprint*S0],
								maximumtheta);
								// fprintf(file_sample,"%f %f ",static_cast<float*> (axion->mCpu())[S0+1], static_cast<float*> (axion->vCpu())[S0+1]);
								// fprintf(file_sample,"%f %f\n", static_cast<float*> (axion->mCpu())[S0+2], static_cast<float*> (axion->vCpu())[S0+2]);
							}
						}
						fflush(file_sample);

					}

			//--------------------------------------------------
			// PROPAGATOR
			//--------------------------------------------------

			//printMpi("dzaux, dz= %f, %f | llaux, LL = %f, %f\n", dzaux, dz, llaux*pow((*axion->zV()),2.), LL );
//			if (axion->Field() == FIELD_SAXION)
//			{
				propagate (axion, fCount, dzaux, delta, nQcd, llaux, VQCD_1);

                if (nstrings_global < 500)
                {
                  //nstrings_global = analyzeStrFoldedNP(axion, index);
                  //MPI_Allreduce(&nstrings, &nstrings_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
									nstrings_global = strings(axion, str, fCount);
									maximumtheta = axion->maxtheta();
									printMpi("  str extra check (%d) (maxth = %f)\n",nstrings_global,maximumtheta);
                  //printMpi("%ld (%d) %ld - ", nstrings, coS, nstrings_global); fflush(stdout);
                }
								//printMpi("%d (%d) %f -> %d", nstrings, coS, (*axion->zV()),
								//( (nstrings <1) && (!coS) && ((*axion->zV()) > 0.6))); fflush(stdout);
                if ( (nstrings_global == 0) && ((*axion->zV()) > 0.1) && axion->Field() == FIELD_SAXION )
                {
										strcount += 1;
										printMpi("  str countdown (%d/20) (maxth = %f)\n",strcount,maximumtheta);
										if ((strcount >5 ) )
										 {
											printMpi("\n");
	                    printMpi("--------------------------------------------------\n");
	                    printMpi("              TRANSITION TO THETA \n");
											fflush(stdout);

											z_now = (*axion->zV());
											llprint = max(LL,llconstantZ2/(z_now*z_now)); //physical value

											saskia = z_now*saxionshift(z_now, nQcd, zthres, zrestore, llprint);

	                    cmplxToTheta (axion, fCount, saskia);
											zrestore = z_now;
	                    printMpi("--------------------------------------------------\n");
										 }
                }
//			}
//			else
//			{
//				propTheta	(axion, dzaux, nQcd, delta, fCount);
//			}


			current = std::chrono::high_resolution_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);

			fCount->addTime(elapsed.count()*1.e-3);

			counter++;

			if ((*axion->zV()) > zFinl)
			{
				printMpi("zf reached! ENDING ... \n"); fflush(stdout);
				break;
			}

			//ENERGY EVERY TIME STEP
			// energy(axion, fCount, eRes, delta, nQcd, LL);
			// fprintf(file_energy,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %d %+lf\n",
			// (*axion->zV()), eR[0], eR[1], eR[2], eR[3], eR[4], eR[5], eR[6], eR[7], eR[8], eR[9], nstrings, maximumtheta);

		} // ZSUBLOOP

		//--------------------------------------------------
		// PARTIAL ANALISIS
		//--------------------------------------------------

      printMpi("1IT %.3fs ETA %.3fh ",elapsed.count()*1.e-3,((nLoops-index)*dump)*elapsed.count()/(1000*60*60.));
			fflush(stdout);

			z_now = (*axion->zV());
			llprint = max(LL,llconstantZ2/(z_now*z_now)); //physical value

			saskia = z_now*saxionshift(z_now, nQcd, zthres, zrestore, llprint);
			// ENERGY NEEDS, axion, llaux (autocorrectes Z2 mode), nQCD?, delta, ..., shift of conformal field = z*shift_physical)
			energy(axion, fCount, eRes, false, delta, nQcd, llaux, VQCD_1, saskia);

			// fprintf(file_energy,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %d %+lf\n",
			// (*axion->zV()), eR[0], eR[1], eR[2], eR[3], eR[4], eR[5], eR[6], eR[7], eR[8], eR[9], nstrings, maximumtheta);
			// fflush(file_energy);

			fprintf(file_energy,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %d %+lf\n",
			(*axion->zV()), eR[TH_GRX], eR[TH_GRY], eR[TH_GRZ], eR[TH_POT], eR[TH_KIN], eR[RH_GRX], eR[RH_GRY], eR[RH_GRZ], eR[RH_POT], eR[RH_KIN], nstrings_global, maximumtheta);
			fflush(file_energy);



			if ( axion->Field() == FIELD_SAXION)
			{
				printMpi("%d/%d | z=%f | dz=%.3e | LLaux=%.3e ", zloop, nLoops, (*axion->zV()), dzaux, llaux);
				printMpi("strings ", zloop, nLoops, (*axion->zV()), dzaux, llaux);

										//nstrings_global = strings(axion, str, fCount);
										nstrings_global =	analyzeStrFolded(axion, index);
										//MPI_Allreduce(&nstrings, &nstrings_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
										//nstrings = (int) nstringsd_global ;

										if ((*axion->zV()) > 0.6)
										{
//										 createMeas(axion, index);
//										 printMpi("[mc-");
//										 writeString	( str , nstrings_global);
//									 printMpi("sw-");
//										 destroyMeas();
	//								 printMpi("d] ");
										}

										printMpi("(G)= %ld \n", nstrings_global);

			}
			else
			{
				maximumtheta = axion->maxtheta();
				printMpi("%d/%d | z=%f | dz=%.3e | maxtheta=%f\n", zloop, nLoops, (*axion->zV()), dzaux, maximumtheta);
				fflush(stdout);
				// munge(UNFOLD_ALL);
				// writeConf(axion, index);
				// munge(FOLD_ALL);



				//IF USING DENSITY FROM ALEX
				//energyMap(axion, LL, nQcd, delta, fCount, VQCD_1, 0.);
				//printMpi("bineando\n", zloop, nLoops, (*axion->zV()), dzaux, maximumtheta);
				//fflush(stdout);
				//axion->writeMAPTHETA( (*(axion->zV() )) , index, binarray, 10000)		;

				//IF USING DENSITY FROM JAVI
				munge(UNFOLD_ALL);
				axion->writeMAPTHETA( (*(axion->zV() )) , index, binarray, 10000)		;


				if (commRank() == 0)
				{
				fprintf(file_contbin,"%f ", (*(axion->zV() )));
				// first three numbers are dens average, max contrast and maximum of the binning
				for(int i = 0; i<10000; i++) {	fprintf(file_contbin, "%e ", (float) bA[i]);}
				fprintf(file_contbin, "\n");
				fflush(file_contbin);
				}

				// // BIN THETA
				// maximumtheta = axion->thetaDIST(100, binarray);
				// if (commRank() == 0)
				// {
				// 	fprintf(file_thetabin,"%f %f ", (*(axion->zV() )), maximumtheta );
				// 	for(int i = 0; i<100; i++) {	fprintf(file_thetabin, "%f ", bA[i]);} fprintf(file_thetabin, "\n");
				// }

				// axion->writeMAPTHETA( (*(axion->zV() )) , index, binarray, 10000)		;
				// printMpi("| ");
				//
				// if (commRank() == 0)
				// {
				// fprintf(file_contbin,"%f ", (*(axion->zV() )));
				// // first three numbers are dens average, max contrast and maximum of the binning
				// for(int i = 0; i<10000; i++) {	fprintf(file_contbin, "%f ", (float) bA[i]);}
				// fprintf(file_contbin, "\n");
				// fflush(file_contbin);
				// }
				// commSync();

				munge(FOLD_ALL);
			}

			 if (commRank() == 0)
			 {
			 munge(UNFOLD_SLICE, sliceprint);
			 writeMap (axion, index);
			 }

			// SAVE FILE OUTPUT


			if ((*axion->zV()) > zFinl)
			{
				printMpi("zf reached! ENDING FINALLY... \n");
				break;
			}

	} // ZLOOP

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	printMpi("\n");
	printMpi("--------------------------------------------------\n");
	printMpi("              EVOLUTION FINISHED \n");
	printMpi("--------------------------------------------------\n");
	fflush(stdout);

	printMpi("Unfold ... ");
	munge(UNFOLD_ALL);
	printMpi("| ");

	if (axion->Field() == FIELD_AXION)
	{
		createMeas(axion, index+1);

		printMpi("nSpec ... ");
		//NUMBER SPECTRUM
		//spectrumUNFOLDED(axion, spectrumK, spectrumG, spectrumV);
		spectrumUNFOLDED(axion);

		//printf("sp %f %f %f ...\n", (float) sK[0]+sG[0]+sV[0], (float) sK[1]+sG[1]+sV[1], (float) sK[2]+sG[2]+sV[2]);
		printMpi("| ");
		if (commRank() == 0)
		{
		fprintf(file_spectrum,  "%lf ", (*axion->zV()));
		for(int i = 0; i<powmax; i++) {	fprintf(file_spectrum, "%lf ", sK[i]);} fprintf(file_spectrum, "\n");
		fprintf(file_spectrum,  "%lf ", (*axion->zV()));
		for(int i = 0; i<powmax; i++) {	fprintf(file_spectrum, "%lf ", sG[i]);} fprintf(file_spectrum, "\n");
		fprintf(file_spectrum,  "%lf ", (*axion->zV()));
		for(int i = 0; i<powmax; i++) {	fprintf(file_spectrum, "%lf ", sV[i]);} fprintf(file_spectrum, "\n");
		//axion->foldField();
		}
		commSync();

		writeSpectrum(axion, sK, sG, sV, powmax, false);

		printMpi("DensMap ... ");
		axion->writeMAPTHETA( (*(axion->zV() )) , index, binarray, 10000)		;
		printMpi("| ");

		if (commRank() == 0)
		{
		fprintf(file_contbin,"%f ", (*(axion->zV() )));
		// first three numbers are dens average, max contrast and maximum of the binning
		for(int i = 0; i<10000; i++) {	fprintf(file_contbin, "%f ", (float) bA[i]);}
		fprintf(file_contbin, "\n");
		fflush(file_contbin);
		}
		commSync();
		writeArray(axion, bA, 10000, "/bins", "cont");


		//POWER SPECTRUM

		printMpi("pSpec ... ");

		powerspectrumUNFOLDED(axion, fCount);
		if (commRank() == 0)
		{
		printf("sp %f ...\n", sK[0]);
		fprintf(file_power,  "%f ", (*axion->zV()));
		for(int i = 0; i<powmax; i++) {	fprintf(file_power, "%f ", sK[i]);} fprintf(file_power, "\n");
		}
		printMpi("| ");

		//writeArray(axion, bA, 10000, "/bins", "cont");
		//writeSpectrum(axion, sK, sG, sV, powmax, true);



		// BIN THETA
		maximumtheta = axion->thetaDIST(100, binarray);
		if (commRank() == 0)
		{
			fprintf(file_thetabin,"%f %f ", (*(axion->zV() )), maximumtheta );
			for(int i = 0; i<100; i++) {	fprintf(file_thetabin, "%f ", bA[i]);} fprintf(file_thetabin, "\n");
		}


		writeArray(axion, binarray, 100, "/bins", "theta");

		// printMpi("dens2m ... ");
		// axion->denstom();
		// printMpi("| ");

		destroyMeas();

		//munge(FOLD_ALL);
		fflush(file_power);
		fflush(file_spectrum);
	}

	if (cDev != DEV_GPU)
	{
		//axion->unfoldField();
		//munge(UNFOLD_ALL);
	}

	if (axion->Field() == FIELD_AXION)
	{
	if (nSteps > 0)
	writeConf(axion, index);
	}

	printMpi("z_final = %f\n", *axion->zV());
	printMpi("#_steps = %i\n", counter);
	printMpi("#_prints = %i\n", index);
	printMpi("Total time: %2.3f min\n", elapsed.count()*1.e-3/60.);
	printMpi("Total time: %2.3f h\n", elapsed.count()*1.e-3/3600.);
	printMpi("GFlops: %.3f\n", fCount->GFlops());
	printMpi("GBytes: %.3f\n", fCount->GBytes());

	trackFree(&eRes, ALLOC_TRACK);
	trackFree(&str,  ALLOC_ALIGN);
	trackFree((void**) (&spectrumK),  ALLOC_TRACK);
	trackFree((void**) (&spectrumG),  ALLOC_TRACK);
	trackFree((void**) (&spectrumV),  ALLOC_TRACK);
	trackFree((void**) (&binarray),  ALLOC_TRACK);

	delete fCount;
	delete axion;

	endComms();

	printMemStats();

	//JAVIER
	if (commRank() == 0)
	{
		fclose (file_sample);
		fclose (file_energy);
		fclose (file_spectrum);
		fclose (file_power);
		fclose (file_thetabin);
		fclose (file_contbin);
		//energy 2//	fclose (file_energy2);
	}

	return 0;
}