#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "propagator/allProp.h"
#include "energy/energy.h"
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

	msa =1.7 ;

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

	//energy 2//	FILE *file_energy2 ;
	//energy 2//	file_energy2 = NULL;

	FILE *file_spectrum ;
	file_spectrum = NULL;

	FILE *file_power ;
	file_power = NULL;

	FILE *file_thetabin ;
	file_thetabin = NULL;

	FILE *file_contbin ;
	file_contbin = NULL;

	FILE *file_axiton ;
	file_axiton = NULL;


	if (commRank() == 0)
	{
		file_sample = fopen("out/sample.txt","w+");
		//fprintf(file_sample,"%f %f %f\n",z, creal(m[0]), cimag(m[0]));
		file_energy = fopen("out/energy.txt","w+");
		//fprintf(file_sample,"%f %f %f\n",z, creal(m[0]), cimag(m[0]));
		//energy 2//	file_energy2 = fopen("out/energy2.txt","w+");
		file_spectrum = fopen("out/spectrum.txt","w+");
		file_power = fopen("out/power.txt","w+");
		file_thetabin = fopen("out/thetabin.txt","w+");
		file_contbin = fopen("out/contbin.txt","w+");
		file_axiton = fopen("out/axiton.txt","w+");
	}
	printMpi("Files prepared! \n");

	double Vr, Vt, Kr, Kt, Grz, Gtz;
	size_t nstrings = 0 ;
	size_t nstrings_global = 0 ;

  double nstringsd = 0. ;
	double nstringsd_global = 0. ;
	double maximumtheta = 3.141597;
	size_t sliceprint = sizeN/2;

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

	// printMpi("--------------------------------------------------\n");
	// printMpi("           BASE INITIAL CONDITIONS                \n\n");
	//
	// printMpi("Length =  %2.5f\n", sizeL);
	// printMpi("N      =  %ld\n",   sizeN);
	// printMpi("Nz     =  %ld\n",   sizeZ);
	// printMpi("zGrid  =  %ld\n",   zGrid);
	// printMpi("dx     =  %2.5f\n", delta);
	// printMpi("dz     =  %2.5f\n", dz);
	// printMpi("LL     =  %2.5f\n", LL);
	// printMpi("--------------------------------------------------\n");

	const size_t S0 = sizeN*sizeN;
	const size_t SF = sizeN*sizeN*(sizeZ+1)-1;
	const size_t V0 = 0;
	const size_t VF = axion->Size()-1;

	//	--------------------------------------------------
	//	TRICK PARAMETERS TO RESPECT STRINGS
	//	--------------------------------------------------

		// WE USE LAMDA_Z2 WITH msa = 1.5 so
		// zthres = z at which we reach ma^2/ms^2 =1/80=1/9*9

		msa = 1.7 ;
		zthres 	 = 100.0 ;
		zrestore = 100.0 ;
	  double llconstantZ2 = 0.5/pow(delta/msa,2.);
		LL = pow(msa/(delta*zthres),2.)/2. ;
		printMpi ("llconstantZ2 = %f - LL will be set to llconstantZ2/Z^2 \n", llconstantZ2);

		bool coZ = 1;
	  bool coS = 1;
		bool coA = 1;
		int strcount = 0;

		int numaxiprint = 10 ;

		axion->SetLambda(LAMBDA_Z2)	;
		if (LAMBDA_FIXED == axion->Lambda())
		{ 	printMpi ("Lambda in FIXED mode\n"); 	}
		else
		{		printMpi ("Lambda in Z2 mode\n"); 		}




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

	double saskia;




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
	printMpi("           PARAMETERS  						                \n\n");
	printMpi("Length =  %2.2f\n", sizeL);
	printMpi("nQCD   =  %2.2f\n", nQcd);
	printMpi("N      =  %ld\n",   sizeN);
	printMpi("Nz     =  %ld\n",   sizeZ);
	printMpi("zGrid  =  %ld\n",   zGrid);
	printMpi("dx     =  %2.5f\n", delta);
	printMpi("dz     =  %2.2f/FREQ\n", 1.2);
	printMpi("LL     =  %1.3e/z^2 Set to make ms*delta =%f \n\n", llconstantZ2, msa);
	printMpi("VQCD1,shift,con_thres=100, continuous theta  \n", llconstantZ2, msa);
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

			z_now = (*axion->zV());

				//--------------------------------------------------
				// PRINT POINT
				//--------------------------------------------------

				// give idx folded! NOW SHIT
				//size_t idxprint = sliceprint*S0  + (sizeN/2)*sizeN + sizeN/2 ;
				size_t idxprint = 0 ;

				if (commRank() == 0)
					{

						if (axion->Field() == FIELD_SAXION)
						{

							llprint = max(LL , llaux/pow(z_now,2.));
							saskia = saxionshift(z_now, nQcd, zthres, zrestore, llprint);

							if (sPrec == FIELD_DOUBLE) {
								fprintf(file_sample,"%f %f %f %f %f %f %f %ld %f %e\n",
								z_now,
								axionmass(z_now,nQcd,zthres, zrestore),
								llprint,
								static_cast<complex<double> *> (axion->mCpu())[idxprint + S0].real(),
								static_cast<complex<double> *> (axion->mCpu())[idxprint + S0].imag(),
								static_cast<complex<double> *> (axion->vCpu())[idxprint].real(),
								static_cast<complex<double> *> (axion->vCpu())[idxprint].imag(),
								nstrings_global,
								maximumtheta,
								saskia );
							} else {
								fprintf(file_sample,"%f %f %f %f %f %f %f %ld %f %e\n",
								z_now,
								axionmass(z_now,nQcd,zthres, zrestore),
								llprint,
								static_cast<complex<float>  *> (axion->mCpu())[idxprint + S0].real(),
								static_cast<complex<float>  *> (axion->mCpu())[idxprint + S0].imag(),
								static_cast<complex<float>  *> (axion->vCpu())[idxprint].real(),
								static_cast<complex<float>  *> (axion->vCpu())[idxprint].imag(),
								nstrings_global, maximumtheta, saskia);
							}
						}
						else
						{
							if (sPrec == FIELD_DOUBLE) {
								fprintf(file_sample,"%f %f %f %f %f\n",
								z_now,
								axionmass(z_now,nQcd,zthres, zrestore),
								static_cast<double*> (axion->mCpu())[idxprint + S0],
								static_cast<double*> (axion->vCpu())[idxprint],
								maximumtheta);
							} else {
								fprintf(file_sample,"%f %f %f %f %f\n",
								z_now,
								axionmass(z_now,nQcd,zthres, zrestore),
								static_cast<float*> (axion->mCpu())[idxprint + S0],
								static_cast<float*> (axion->vCpu())[idxprint],
								maximumtheta);
								// fprintf(file_sample,"%f %f ",static_cast<float*> (axion->mCpu())[S0+1], static_cast<float*> (axion->vCpu())[S0+1]);
								// fprintf(file_sample,"%f %f\n", static_cast<float*> (axion->mCpu())[S0+2], static_cast<float*> (axion->vCpu())[S0+2]);
							}
						}
						fflush(file_sample);

						// AXITONS

						if (!coA)
						{
							fprintf(file_axiton,"%f ", z_now);
							for (int i =0; i <numaxiprint ; i++)
							{		// IF FOLDED!!
									fprintf(file_axiton,"%f ", static_cast<float*> (axion->mCpu())[axitonarray[i] + S0]);
							}
							fprintf(file_axiton,"\n ");
							fflush(file_axiton);
						}


					}
			//--------------------------------------------------
			// DYAMICAL deltaz
			//--------------------------------------------------

			//Set dz to gradients and axion mass
			//dzaux = min(delta,1./(z_now*axionmass(z_now,nQcd,zthres, zrestore)));
			 double masi = z_now*axionmass(z_now,nQcd,zthres, zrestore);
			 double mfre = sqrt(masi*masi + 12./(delta*delta));
			 dzaux = 1.2/mfre ;

			//If SAXION_MODE
			if (axion->Field() == FIELD_SAXION && coZ)  // IF SAXION and Z2 MODE
			{
				llaux = llconstantZ2;
				llprint = llaux/(z_now*z_now); //physical value
				// dzaux = min(dzaux,delta/1.5);
				double mfre = sqrt( msa*msa + 12.)/delta;
				dzaux = min(dzaux,1.2/mfre)  ;
			}

      //dzaux = dzaux/1.5 ;

			//--------------------------------------------------
			// PROPAGATOR
			//--------------------------------------------------

			//printMpi("dzaux, dz= %f, %f | llaux, LL = %f, %f\n", dzaux, dz, llaux*pow((*axion->zV()),2.), LL );
//			if (axion->Field() == FIELD_SAXION)
//			{
				propagate (axion, fCount, dzaux, delta, nQcd, llaux, VQCD_1);
			if (axion->Field() == FIELD_SAXION)
			{

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
				if ( (nstrings_global == 0) && ((*axion->zV()) > 0.2) )
				{
					 strcount += 1;
					 printMpi("  str countdown (%d/20) (maxth = %f)\n",strcount,maximumtheta);

					 // BIN THETA
 					maximumtheta = axion->thetaDIST(100, binarray);
 					if (commRank() == 0)
 					{
 						fprintf(file_thetabin,"%f %f ", (*(axion->zV() )), maximumtheta );
 						for(int i = 0; i<200; i++) {	fprintf(file_thetabin, "%f ", bA[i]);} fprintf(file_thetabin, "\n");
 						fflush(file_thetabin);
 					}


				if ((strcount >10 ) )
				{



					printMpi("\n");

					z_now = (*axion->zV());
					llprint = llaux/(z_now*z_now); //physical value
					saskia = z_now * saxionshift(z_now, nQcd, zthres, zrestore, llprint);
					printMpi("--------------------------------------------------\n");
					printMpi("              TRANSITION TO THETA \n");
					// BIN THETA
					maximumtheta = axion->thetaDIST(100, binarray);
					if (commRank() == 0)
					{
						fprintf(file_thetabin,"%f %f ", (*(axion->zV() )), maximumtheta );
						for(int i = 0; i<200; i++) {	fprintf(file_thetabin, "%f ", bA[i]);} fprintf(file_thetabin, "\n");
					}
					//IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS
					 munge(UNFOLD_SLICE, sliceprint);
					  	writeMap (axion, 100000);
					cmplxToTheta (axion, fCount, saskia);

					// SHIFTS THETA TO A CONTINUOUS FIELD
					// REQUIRED UNFOLDED FIELDS
					munge(UNFOLD_ALL);
					axion->mendtheta();
					munge(FOLD_ALL);

					//IF YOU WANT A MAP TO CONTROL THE TRANSITION TO THETA UNCOMMENT THIS
					munge(UNFOLD_SLICE, sliceprint);
					writeMap (axion, 100001);


					printMpi("--------------------------------------------------\n");
				}
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

		} // ZSUBLOOP

		//--------------------------------------------------
		// PARTIAL ANALISIS
		//--------------------------------------------------

      printMpi("1IT %.3fs ETA %.3fh ",elapsed.count()*1.e-3,((nLoops-index)*dump)*elapsed.count()/(1000*60*60.));


			z_now = (*axion->zV());
			llprint = llaux/(z_now*z_now); //physical value
			saskia = saxionshift(z_now, nQcd, zthres, zrestore, llprint);


			if ( axion->Field() == FIELD_SAXION)
			{
				energy(axion, fCount, eRes, false, delta, nQcd, llaux, VQCD_1, saskia);

				double maa = 40*axionmass(z_now,nQcd,zthres, zrestore)/(2*llaux);
				if (axion->Lambda() == LAMBDA_Z2 )
				maa = maa*z_now*z_now;

				printMpi("%d/%d | z=%f | dz=%.3e | LLaux=%.3e | 40ma2/ms2=%.3e ", zloop, nLoops, (*axion->zV()), dzaux, llaux, maa );
				printMpi("strings ", zloop, nLoops, (*axion->zV()), dzaux, llaux);

										//nstrings_global = strings(axion, str, fCount);
										nstrings_global =	analyzeStrFolded(axion, index);
										//MPI_Allreduce(&nstrings, &nstrings_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
										//nstrings = (int) nstringsd_global ;
										// if ((*axion->zV()) > 0.6)
										// {
										//  createMeas(axion, index);
										//  writeString	( str , nstrings_global);
										//  destroyMeas();
										// }
										printMpi("(G)= %ld \n", nstrings_global);

			}
			else
			{
				//BINS THETA
				maximumtheta = axion->thetaDIST(100, binarray);
				if (commRank() == 0)
				{
					fprintf(file_thetabin,"%f %f ", (*(axion->zV() )), maximumtheta );
					for(int i = 0; i<200; i++) {	fprintf(file_thetabin, "%f ", bA[i]);} fprintf(file_thetabin, "\n");
					fflush(file_thetabin);
				}

				printMpi("%d/%d | z=%f | dz=%.3e | maxtheta=%f | ", zloop, nLoops, (*axion->zV()), dzaux, maximumtheta);
				fflush(stdout);

				printMpi("DensMap ... ");

				//OLD VERSION, NEEDS UNFOLD
				//munge(UNFOLD_ALL);
				//axion->writeMAPTHETA( (*(axion->zV() )) , index, binarray, 10000)		;

				// NEW VERSION
				// computes energy and creates map
				energy(axion, fCount, eRes, true, delta, nQcd, 0., VQCD_1, 0.);
				//bins density
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

				// writeConf(axion, index);
				//munge(FOLD_ALL);

				if (coA && maximumtheta > 3.14 && z_now > 3.0)
				{
					// threshold ; pointer ; number of axitons
					axion->writeAXITONlist(100. , axitonarray, numaxiprint) ;
					for(int i = 0; i<numaxiprint; i++)
					{
						printMpi("(%ld)", axitonarray[i]);
					}
					printMpi("\n");
					if (axitonarray[numaxiprint-1]>0)
					coA = 0 ;
				}
				printMpi("\n");
				// if (!coA)
				// {
				// 	fprintf(file_axiton,"%f ", z_now);
				// 	for (int i =0; i <10 ; i++)
				// 	{		// IF FOLDED!!
				// 			fprintf(file_axiton,"%f ", static_cast<float*> (axion->mCpu())[axitonarray[i] + S0]);
				// 	}
				// 	fprintf(file_axiton,"\n ");
				// 	fflush(file_axiton);
				// }

			}

			 if (commRank() == 0)
			 {
			 munge(UNFOLD_SLICE, sliceprint);
			 writeMap (axion, index);

			 //Prints energy computed before
			 fprintf(file_energy,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %d %+lf\n",
			 (*axion->zV()),
			 eR[TH_GRX], eR[TH_GRY], eR[TH_GRZ],
			 eR[TH_POT], eR[TH_KIN],
			 eR[RH_GRX], eR[RH_GRY], eR[RH_GRZ],
			 eR[RH_POT], eR[RH_KIN],
			 nstrings_global, maximumtheta);
			 fflush(file_energy);


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

		energy(axion, fCount, eRes, true, delta, nQcd, 0., VQCD_1, 0.);
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
			for(int i = 0; i<200; i++) {	fprintf(file_thetabin, "%f ", bA[i]);} fprintf(file_thetabin, "\n");
		}


		writeArray(axion, sK, 100, "/bins", "theta");

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
	trackFree((void**) (&axitonarray),  ALLOC_TRACK);

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
		fclose (file_axiton);
		//energy 2//	fclose (file_energy2);
	}

	return 0;
}
