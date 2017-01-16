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
	}
	printMpi("Files prepared! \n");

	double Vr, Vt, Kr, Kt, Grz, Gtz;
	int nstrings = 201 ;

  double nstringsd = 201. ;
	double nstringsd_global = 201 ;
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

	double *sK = static_cast<double *> (spectrumK);
	double *sG = static_cast<double *> (spectrumG);
	double *sV = static_cast<double *> (spectrumV);
	double *bA = static_cast<double *> (binarray);
	//double *bAd = static_cast<double *> (binarray);

	//--------------------------------------------------
	//          SETTING BASE PARAMETERS
	//--------------------------------------------------

	double delta = sizeL/sizeN;
	double dz;
	double dzaux;
	double llaux;

	if (nSteps == 0)
		dz = 0.;
	else
		dz = (zFinl - zInit)/((double) nSteps);

	printMpi("--------------------------------------------------\n");
	printMpi("           INITIAL CONDITIONS                     \n\n");

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




	printMpi("INITIAL CONDITIONS LOADED\n");
	if (sPrec != FIELD_DOUBLE)
	{
		printMpi("Example mu: m[0] = %f + %f*I, m[N3-1] = %f + %f*I\n", ((complex<float> *) axion->mCpu())[S0].real(), ((complex<float> *) axion->mCpu())[S0].imag(),
									        ((complex<float> *) axion->mCpu())[SF].real(), ((complex<float> *) axion->mCpu())[SF].imag());
		printMpi("Example  v: v[0] = %f + %f*I, v[N3-1] = %f + %f*I\n", ((complex<float> *) axion->vCpu())[V0].real(), ((complex<float> *) axion->vCpu())[V0].imag(),
									        ((complex<float> *) axion->vCpu())[VF].real(), ((complex<float> *) axion->vCpu())[VF].imag());
	}
	else
	{
		printMpi("Example mu: m[0] = %lf + %lf*I, m[N3-1] = %lf + %lf*I\n", ((complex<double> *) axion->mCpu())[S0].real(), ((complex<double> *) axion->mCpu())[S0].imag(),
										    ((complex<double> *) axion->mCpu())[SF].real(), ((complex<double> *) axion->mCpu())[SF].imag());
		printMpi("Example  v: v[0] = %lf + %lf*I, v[N3-1] = %lf + %lf*I\n", ((complex<double> *) axion->vCpu())[V0].real(), ((complex<double> *) axion->vCpu())[V0].imag(),
										    ((complex<double> *) axion->vCpu())[VF].real(), ((complex<double> *) axion->vCpu())[VF].imag());
	}

	//JAVIER commented next
	//printMpi("Ez     =  %ld\n",    axion->eDepth());

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

	bool coZ = 1;
    bool coS = 1;

	axion->SetLambda(LAMBDA_Z2)	;
	if (LAMBDA_FIXED == axion->Lambda())
	{
	printMpi ("Lambda in FIXED mode\n");
	}
	else
	{
		printMpi ("Lambda in Z2 mode\n");
	}

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

		munge(UNFOLD_SLICE, sliceprint);
		writeMap (axion, index);

	}

	if (dump > nSteps)
		dump = nSteps;

	int nLoops;

	if (dump == 0)
		nLoops = 0;
	else
		nLoops = (int)(nSteps/dump);

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

			if (commRank() == 0)
			{

				if (axion->Field() == FIELD_SAXION)
				{
					if (sPrec == FIELD_DOUBLE) {
						fprintf(file_sample,"%f %f %f %f %f %d\n",(*(axion->zV() )), static_cast<complex<double> *> (axion->mCpu())[sliceprint*S0].real(), static_cast<complex<double> *> (axion->mCpu())[sliceprint*S0].imag(),
							static_cast<complex<double> *> (axion->vCpu())[sliceprint*S0].real(), static_cast<complex<double> *> (axion->vCpu())[sliceprint*S0].imag(),nstrings);
					} else {
						fprintf(file_sample,"%f %f %f %f %f %d\n",(*(axion->zV() )), static_cast<complex<float>  *> (axion->mCpu())[sliceprint*S0].real(), static_cast<complex<float>  *> (axion->mCpu())[sliceprint*S0].imag(),
							static_cast<complex<float>  *> (axion->vCpu())[sliceprint*S0].real(), static_cast<complex<float>  *> (axion->vCpu())[sliceprint*S0].imag(),nstrings);
					}
				}
				else
				{
					if (sPrec == FIELD_DOUBLE) {
						fprintf(file_sample,"%f %f %f\n",(*(axion->zV() )), static_cast<double*> (axion->mCpu())[sliceprint*S0], static_cast<double*> (axion->vCpu())[sliceprint*S0]);
					} else {
						fprintf(file_sample,"%f %f %f\n",(*(axion->zV() )), static_cast<float*> (axion->mCpu())[sliceprint*S0], static_cast<float*> (axion->vCpu())[sliceprint*S0]);
						// fprintf(file_sample,"%f %f ",static_cast<float*> (axion->mCpu())[S0+1], static_cast<float*> (axion->vCpu())[S0+1]);
						// fprintf(file_sample,"%f %f\n", static_cast<float*> (axion->mCpu())[S0+2], static_cast<float*> (axion->vCpu())[S0+2]);
					}
				}
				fflush(file_sample);

			}

			old = std::chrono::high_resolution_clock::now();


			//--------------------------------------------------
			// DYAMICAL deltaz
			//--------------------------------------------------

			dzaux = min(delta,1./(3.*pow((*axion->zV()),1.+nQcd/2.)));
			if (axion->Field() == FIELD_SAXION && coZ)  // IF SAXION and Z2 MODE
			{
				llaux = 1./pow(2.*delta,2.);
			}

			//printMpi("(dz0,dz1,dz2)= (%f,%f,%f) ", delta, 1./(sqrt(LL)*(*axion->zV())) ,1./(9.*pow((*axion->zV()),nQcd)));
			if (axion->Field() == FIELD_SAXION && LL*pow((*axion->zV()),2.) > llaux && coZ )
			{
				axion->SetLambda(LAMBDA_FIXED)	;
				printMpi("Lambda Fixed transition at %f \n", (*axion->zV()));
				coZ = 0;
			}
			if ( !coZ )
			{
				llaux = LL;
        dzaux = min(dzaux,1./(sqrt(2.*LL)*(*axion->zV())));
			}
        dzaux = dzaux/2.;

			//--------------------------------------------------
			// PROPAGATOR
			//--------------------------------------------------

			//printMpi("dzaux, dz= %f, %f | llaux, LL = %f, %f\n", dzaux, dz, llaux*pow((*axion->zV()),2.), LL );
			if (axion->Field() == FIELD_SAXION)
			{
				propagate (axion, dzaux, llaux, nQcd, delta, cDev, fCount, VQCD_1);

                if (nstrings < 200)
                {
                  //nstrings = analyzeStrFoldedNP(axion, index);
                  nstringsd = strings(axion, cDev, str, fCount);
                  MPI_Allreduce(&nstringsd, &nstringsd_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
									nstrings = (int) nstringsd_global ;

                  //printMpi("%d (%d) %f ", nstrings, coS, (*axion->zV())); fflush(stdout);
                }
                if ( (nstrings ==0) && (!coZ) && ((*axion->zV()) > 0.6) )
                {
                    printMpi("\n");
                    printMpi("--------------------------------------------------\n");
                    printMpi("              TRANSITION TO THETA \n");
                    cmplxToTheta	(axion, fCount);
                    printMpi("--------------------------------------------------\n");
                    fflush(stdout);
                }
			}
			else
			{
				propTheta	(axion, dzaux,     nQcd, delta, cDev, fCount);
			}

			current = std::chrono::high_resolution_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);

			fCount->addTime(elapsed.count()*1.e-3);

			counter++;

			if ((*axion->zV()) > zFinl)
			{
				printMpi("zf reached! ENDING ... \n");
				break;
			}

		} // ZSUBLOOP

		//--------------------------------------------------
		// PARTIAL ANALISIS
		//--------------------------------------------------

      printMpi("IT %.3f ETA %.3f ",elapsed.count()*1.e-3*dump,((nLoops-index)*dump)*elapsed.count()/(1000*60.));

			if ( axion->Field() == FIELD_SAXION)
			{
				printMpi("%d/%d | z=%f | dz=%.3e | LLaux=%.3e ", zloop, nLoops, (*axion->zV()), dzaux, llaux);
				if ((*axion->zV()) > 0.2)
				{
					printMpi("strings (z>0.2) ", zloop, nLoops, (*axion->zV()), dzaux, llaux);
					fflush (stdout);
										//nstrings = analyzeStrFolded(axion, index);
                    //printMpi("= %d ", nstrings);
                    //nstringsd = strings(axion, cDev, str, fCount);
                    //nstrings = (int) nstringsd ;
										//printMpi("= %d ", nstrings);
										nstringsd = strings(axion, cDev, str, fCount);
										MPI_Allreduce(&nstringsd, &nstringsd_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
										nstrings = (int) nstringsd_global ;
										printMpi("= %d ", nstrings);
					fflush (stdout);
                    if (nstrings < 200)
                    {
                        coS = 0;
                        printMpi("Low string density! coS=0");
                    }
				}
				printMpi("\n");
			}
			else
			{
				printMpi("%d/%d | z=%f | dz=%.3e \n", zloop, nLoops, (*axion->zV()), dzaux);
				fflush (stdout);
			}

			munge(UNFOLD_SLICE, sliceprint);
			writeMap (axion, index);

//
//	if ( axion->Field() == FIELD_SAXION && nstrings == 0 && (*axion->zV()) > 0.6 )
//	{
//		printMpi("\n");
//		printMpi("--------------------------------------------------\n");
//		printMpi("              TRANSITION TO THETA \n");
//		cmplxToTheta	(axion, fCount);
//		printMpi("--------------------------------------------------\n");
//		fflush(stdout);
//	}

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

	if (axion->Field() == FIELD_AXION)
	{
		printMpi("Unfold | ");
		munge(UNFOLD_ALL);

		printMpi("nSpec | ");
		//NUMBER SPECTRUM
		spectrumUNFOLDED(axion, spectrumK, spectrumG, spectrumV);
		//printf("sp %f %f %f ...\n", (float) sK[0]+sG[0]+sV[0], (float) sK[1]+sG[1]+sV[1], (float) sK[2]+sG[2]+sV[2]);
		if (commRank() == 0)
		{
		fprintf(file_spectrum,  "%f ", (*axion->zV()));
		for(int i = 0; i<powmax; i++) {	fprintf(file_spectrum, "%f ", (float) sK[i]);} fprintf(file_spectrum, "\n");
		fprintf(file_spectrum,  "%f ", (*axion->zV()));
		for(int i = 0; i<powmax; i++) {	fprintf(file_spectrum, "%f ", (float) sG[i]);} fprintf(file_spectrum, "\n");
		fprintf(file_spectrum,  "%f ", (*axion->zV()));
		for(int i = 0; i<powmax; i++) {	fprintf(file_spectrum, "%f ", (float) sV[i]);} fprintf(file_spectrum, "\n");
		//axion->foldField();
		}

		printMpi("DensMap | ");
		axion->writeMAPTHETA( (*(axion->zV() )) , index, binarray, 10000)		;


		if (commRank() == 0)
		{
		fprintf(file_contbin,"%f ", (*(axion->zV() )));
		// first three numbers are dens average, max contrast and maximum of the binning
		for(int i = 0; i<10000; i++) {	fprintf(file_contbin, "%f ", (float) bA[i]);}
		fprintf(file_contbin, "\n");
		fflush(file_contbin);
		}
		// BIN THETA
		maximumtheta = axion->thetaDIST(100, spectrumK);
		if (commRank() == 0)
		{
			fprintf(file_thetabin,"%f %f ", (*(axion->zV() )), maximumtheta );
			for(int i = 0; i<100; i++) {	fprintf(file_thetabin, "%f ", (float) sK[i]);} fprintf(file_thetabin, "\n");
		}

		printMpi("dens2m | ");
		axion->denstom();

		printMpi("pSpec | ");
		//POWER SPECTRUM
		powerspectrumUNFOLDED(axion, spectrumK, spectrumG, spectrumV, fCount);
		printf("sp %f %f %f ...\n", (float) sK[0]+sG[0]+sV[0], (float) sK[1]+sG[1]+sV[1], (float) sK[2]+sG[2]+sV[2]);
		fprintf(file_power,  "%f ", (*axion->zV()));
		for(int i = 0; i<powmax; i++) {	fprintf(file_power, "%f ", (float) sK[i]);} fprintf(file_power, "\n");
		fprintf(file_power,  "%f ", (*axion->zV()));
		for(int i = 0; i<powmax; i++) {	fprintf(file_power, "%f ", (float) sG[i]);} fprintf(file_power, "\n");
		fprintf(file_power,  "%f ", (*axion->zV()));
		for(int i = 0; i<powmax; i++) {	fprintf(file_power, "%f ", (float) sV[i]);} fprintf(file_power, "\n");

		//munge(FOLD_ALL);
		fflush(file_power);
		fflush(file_spectrum);
	}

	if (cDev != DEV_GPU)
	{
		//axion->unfoldField();
		//munge(UNFOLD_ALL);
	}

	if (nSteps > 0)
	writeConf(axion, index);

	printMpi("z_final = %f\n", *axion->zV());
	printMpi("#_steps = %i\n", counter);
	printMpi("#_prints = %i\n", index);
	printMpi("Total time: %2.3f s\n", elapsed.count()*1.e-3);
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
