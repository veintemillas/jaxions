#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "code3DCpu.h"
#include "scalarField.h"
#include "propagator.h"
#include "propSimple.h"
#include "energy.h"
#include "strings.h"
#include "enum-field.h"
#include "index.h"
#include "parse.h"
#include "readWrite.h"
#include "comms.h"
#include "flopCounter.h"
#include "map.h"
#include "memAlloc.h"
#include "strings.h"

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

	printMpi("\n-------------------------------------------------\n");
	printMpi("\n          CREATING MINICLUSTERS!                \n\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------


	Scalar *axion;
	char fileName[256];

	if ((initFile == NULL) && (fIndex == -1) && (cType == CONF_NONE))
	{
		if (sPrec != FIELD_DOUBLE)
			sprintf(fileName, "data/initial_conditions_m_single.txt");
		else
			sprintf(fileName, "data/initial_conditions_m.txt");
		//This prepares the axion field from default files
		axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, fileName, lowmem, zGrid, CONF_NONE, 0, 0);
		printMpi("Eo\n");
	}
	else
	{
		if (fIndex == -1)
			//This generates initial conditions
			axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, initFile, lowmem, zGrid, cType, parm1, parm2);
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

	//--------------------------------------------------
	//          OUTPUTS FOR CHECKING
	//--------------------------------------------------

	FILE *file_sample ;
	file_sample = NULL;

	FILE *file_energy ;
	file_energy = NULL;


	FILE *file_energy2 ;
	file_energy2 = NULL;

	if (commRank() == 0)
	{
		file_sample = fopen("out/s_sample.txt","w+");
		//fprintf(file_sample,"%f %f %f\n",z, creal(m[0]), cimag(m[0]));

		file_energy = fopen("out/s_energy.txt","w+");
		//fprintf(file_sample,"%f %f %f\n",z, creal(m[0]), cimag(m[0]));

		file_energy2 = fopen("out/energy2.txt","w+");
	}

	double Vr, Vt, Kr, Kt, Grz, Gtz;

	//--------------------------------------------------
	//          SETTING BASE PARAMETERS
	//--------------------------------------------------

	double delta = sizeL/sizeN;
	double dz;

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

	//--------------------------------------------------
	//   THE TIME ITERATION LOOP
	//--------------------------------------------------

	printMpi("--------------------------------------------------\n");
	printMpi("           STARTING COMPUTATION                   \n");
	printMpi("--------------------------------------------------\n");

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	int counter = 0;
	int index = 0;

	FlopCounter *fCount = new FlopCounter;

	commSync();

	void *eRes, *str;			// Para guardar la energia
	trackAlloc(&eRes, 128);
	memset(eRes, 0, 128);
#ifdef	__MIC__
	alignAlloc(&str, 64, (axion->Size()/2));
#elif defined(__AVX__)
	alignAlloc(&str, 32, (axion->Size()/2));
#else
	alignAlloc(&str, 16, (axion->Size()/2));
#endif
	memset(str, 0, axion->Size()/2);

	commSync();

	if (fIndex == -1)
	{
		printMpi ("Dumping configuration %05d ...", index);
		writeConf(axion, index);
		printMpi ("Done!\n");
		fflush (stdout);
	}
	else
		index = fIndex + 1;

	if (cDev != DEV_GPU)
	{
		printMpi ("Avoid the folding\n");
		//axion->foldField();
	}

	if (cDev != DEV_CPU)
	{
		printMpi ("Transferring configuration to device\n");
		axion->transferDev(FIELD_MV);
	}

	if (cDev != DEV_GPU)
	{
		printMpi("Strings...");
		//analyzeStrFolded(axion, index);
		//analyzeStrUNFolded(axion, index);
		printMpi(" Done!");
		memcpy   (axion->mCpu(), static_cast<char *> (axion->mCpu()) + S0*sizeZ*axion->dataSize(), S0*axion->dataSize());
		//axion->unfoldField2D(sizeZ-1);
		writeMap (axion, index);
		//energy(axion, LL, nQcd, delta, cDev, eRes, fCount);
		axion->writeENERGY ((*(axion->zV() )),file_energy, Grz, Gtz, Vr, Vt, Kr, Kt);


		if (commRank() == 0)
		{
			if (axion->Precision() == FIELD_DOUBLE)
			{
				double *eR = static_cast<double *> (eRes);
				fprintf(file_energy2,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf\n", (*axion->zV()), Vr, Vt, Kr, Kt, Grz, Gtz);
			}
			else
			{
				double *eR = static_cast<double *> (eRes);
				fprintf(file_energy2,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf\n", (*axion->zV()), Vr, Vt, Kr, Kt, Grz, Gtz);
			}
		}

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

	for (int zloop = 0; zloop < nLoops; zloop++)
	{
		//--------------------------------------------------
		// THE TIME ITERATION SUB-LOOP
		//--------------------------------------------------

		index++;

		for (int zsubloop = 0; zsubloop < dump; zsubloop++)
		{
			if (commRank() == 0) {
				if (sPrec == FIELD_DOUBLE) {
					fprintf(file_sample,"%f %f %f %f %f\n",(*(axion->zV() )), static_cast<complex<double> *> (axion->mCpu())[S0].real(), static_cast<complex<double> *> (axion->mCpu())[S0].imag(),
						static_cast<complex<double> *> (axion->vCpu())[S0].real(), static_cast<complex<double> *> (axion->vCpu())[S0].imag());
				} else {
					fprintf(file_sample,"%f %f %f %f %f\n",(*(axion->zV() )), static_cast<complex<float>  *> (axion->mCpu())[S0].real(), static_cast<complex<float>  *> (axion->mCpu())[S0].imag(),
						static_cast<complex<float>  *> (axion->vCpu())[S0].real(), static_cast<complex<float>  *> (axion->vCpu())[S0].imag());
				}
			}

			old = std::chrono::high_resolution_clock::now();
//			propagate (axion, dz, LL, nQcd, delta, cDev, fCount);
			propagateSimple (axion, dz, LL, nQcd, delta, 2);

			current = std::chrono::high_resolution_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);

			fCount->addTime(elapsed.count()*1.e-3);

			counter++;
		} // zsubloop

		fflush (stdout);
		axion->transferCpu(FIELD_MV);

/*	TODO

	2. Fix writeMap so it reads data from the first slice of m
*/

		if (cDev != DEV_GPU)
		{
			//double Grz, Gtz, Vr, Vt, Kr, Kt;
//			writeConf(axion, index);
			//if (axion->Precision() == FIELD_DOUBLE)
			if ((*axion->zV()) > 0.4 )
			{
				//printMpi("Strings (if %f>0.4) ... ", (*axion->zV()));
				fflush (stdout);
				//analyzeStrFolded(axion, index);
				//analyzeStrUNFolded(axion, index);
			}


			//axion->unfoldField2D(sizeZ-1);
			writeMap (axion, index);
			//energy(axion, LL, nQcd, delta, cDev, eRes, fCount);
			axion->writeENERGY ((*(axion->zV() )),file_energy, Grz, Gtz, Vr, Vt, Kr, Kt);

			if (commRank() == 0)
			{
				if (axion->Precision() == FIELD_DOUBLE)
				{
					double *eR = static_cast<double *> (eRes);
					fprintf(file_energy2,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf\n", (*axion->zV()), Vr, Vt, Kr, Kt, Grz, Gtz);
				}
				else
				{
					double *eR = static_cast<double *> (eRes);
					fprintf(file_energy2,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf\n", (*axion->zV()), Vr, Vt, Kr, Kt, Grz, Gtz);
				}
			}
		}
	} // zloop

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	printMpi("\n PROGRAMM FINISHED\n");

	// if (cDev != DEV_GPU)
	// 	axion->unfoldField();

	if (nSteps > 0)
		writeConf(axion, index);

	printMpi("z_final = %f\n", *axion->zV());
	printMpi("#_steps = %i\n", counter);
	printMpi("#_prints = %i\n", index);
	printMpi("Total time: %2.3f s\n", elapsed.count()*1.e-3);
	printMpi("GFlops: %.3f\n", fCount->GFlops());
	printMpi("GBytes: %.3f\n", fCount->GBytes());
	printMpi("--------------------------------------------------\n");

	trackFree(&eRes, ALLOC_TRACK);
	trackFree(&str,  ALLOC_ALIGN);

	delete fCount;
	delete axion;

	endComms();

	printMemStats();

	//JAVIER
	if (commRank() == 0)
	{
		fclose (file_sample);
		fclose (file_energy);
		fclose (file_energy2);
	}

	return 0;
}
