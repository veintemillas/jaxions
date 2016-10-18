#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "code3DCpu.h"
#include "scalarField.h"
#include "propagator.h"
#include "enum-field.h"
#include "index.h"
#include "parse.h"
#include "readWrite.h"
#include "comms.h"
#include "flopCounter.h"
#include "map.h"
#include "memAlloc.h"

using namespace std;

#ifdef	USE_XEON
	__declspec(target(mic)) char *mX, *vX, *m2X;
#endif

#define printMpi(...) do {		\
	if (!commRank()) {		\
	  printf(__VA_ARGS__);  }	\
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
			//This reads from an Axion.00000 file
			readConf(&axion, fIndex);
	}

	//--------------------------------------------------
	//          OUTPUTS FOR CHECKING
	//--------------------------------------------------
	FILE *file_sample ;
	file_sample = NULL;
	file_sample = fopen("out/sample.txt","w+");
	//fprintf(file_sample,"%f %f %f\n",z, creal(m[0]), cimag(m[0]));

	//--------------------------------------------------
	//          SETTING BASE PARAMETERS
	//--------------------------------------------------

	double dz = (zFinl - zInit)/((double) nSteps);
	double delta = sizeL/sizeN;

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

	const uint S0 = sizeN*sizeN;
	const uint SF = sizeN*sizeN*(sizeZ+1)-1;
	const uint V0 = 0;
	const uint VF = axion->Size()-1;

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

	printMpi("Ez     =  %ld\n",    axion->eDepth());

	//--------------------------------------------------
	//   THE TIME ITERATION LOOP
	//--------------------------------------------------

	printMpi("--------------------------------------------------\n");
	printMpi("           STARTING COMPUTATION                   \n");
	printMpi("--------------------------------------------------\n");

	std::chrono::high_resolution_clock::time_point start, current, old;

	int counter = 0;
	int index = 0;

	printMpi ("Dumping configuration %05d...\n", index);
	fflush (stdout);
	writeConf(axion, index);

	if (cDev != DEV_GPU)
	{
		memcpy   (axion->mCpu(), static_cast<char *> (axion->mCpu()) + 2*S0*sizeZ*axion->dataSize(), 2*S0*axion->dataSize());
		writeMap (axion, index);
	}
	else
	{
		memcpy   (axion->mCpu(), static_cast<char *> (axion->mCpu()) + 2*S0*sizeZ*axion->dataSize(), 2*S0*axion->dataSize());
		writeMap (axion, index);
	}

// TEST
/*

		axion->foldField();
	index++;
	writeConf(axion, index);
		axion->unfoldField();

	index++;

	writeConf(axion, index);
	if (cDev != DEV_GPU)
	{
		memcpy   (axion->mCpu(), static_cast<char *> (axion->mCpu()) + 2*S0*sizeZ*axion->dataSize(), 2*S0*axion->dataSize());
		writeMap (axion, index);
	}
	else
	{
		memcpy   (axion->mCpu(), static_cast<char *> (axion->mCpu()) + 2*S0*sizeZ*axion->dataSize(), 2*S0*axion->dataSize());
		writeMap (axion, index);
	}
*/
// FIN

	if (dump > nSteps)
		dump = nSteps;

	int nLoops = (int)(nSteps/dump);

	FlopCounter *fCount = new FlopCounter;

	if (cDev != DEV_GPU)
	{
		printMpi ("Folding configuration\n");
		axion->foldField();
	}

	if (cDev != DEV_CPU)
	{
		printMpi ("Transferring configuration to device\n");
		axion->transferDev(FIELD_MV);
	}

	printMpi ("Start redshift loop\n");
	fflush (stdout);

	commSync();

	start = std::chrono::high_resolution_clock::now();
	old = start;
	std::chrono::milliseconds elapsed;

	for (int zloop = 0; zloop < nLoops; zloop++)
	{
		//--------------------------------------------------
		// THE TIME ITERATION SUB-LOOP
		//--------------------------------------------------

		index++;

		for (int zsubloop = 0; zsubloop < dump; zsubloop++)
		{
			if (sPrec == FIELD_DOUBLE)
			{		fprintf(file_sample,"%f %f %f %f %f\n",(*(axion->zV() )), static_cast<complex<double> *> (axion->mCpu())[S0].real(), static_cast<complex<double> *> (axion->mCpu())[S0].imag(),static_cast<complex<double> *> (axion->vCpu())[S0].real(), static_cast<complex<double> *> (axion->vCpu())[S0].imag());
			} else
			{ 	fprintf(file_sample,"%f %f %f %f %f\n",(*(axion->zV() )), static_cast<complex<float>  *> (axion->mCpu())[S0].real(), static_cast<complex<float>  *> (axion->mCpu())[S0].imag(),static_cast<complex<float>  *> (axion->vCpu())[S0].real(), static_cast<complex<float>  *> (axion->vCpu())[S0].imag());
			}
			old = std::chrono::high_resolution_clock::now();
			propagate (axion, dz, LL, nQcd, delta, cDev, fCount);

			current = std::chrono::high_resolution_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);

			fCount->addTime(elapsed.count()*1.e-3);
			//JAVIER commented next line
			//verbose? YEAH
			//printMpi("%2d - %2d: z = %lf elapsed time =  %2.3lf s\n", zloop, zsubloop, *(axion->zV()), fCount->DTime());

			counter++;
		} // zsubloop

		printMpi ("Generating 2D map...");
		fflush (stdout);
		axion->transferCpu(FIELD_MV);

/*	TODO

	2. Fix writeMap so it reads data from the first slice of m
*/

		if (cDev != DEV_GPU)
		{
			axion->unfoldField2D(sizeZ-1);
//			writeConf(axion, index);
			writeMap (axion, index);
		}
		else
		{
//			writeConf(axion, index);
			memcpy   (axion->mCpu(), static_cast<char *> (axion->mCpu()) + 2*S0*sizeZ*axion->dataSize(), 2*S0*axion->dataSize());
			writeMap (axion, index);
		}
	} // zloop

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	printMpi("\n PROGRAMM FINISHED\n");

	if (cDev != DEV_GPU)
		axion->unfoldField();

	if (sPrec == FIELD_DOUBLE)
	{
		printMpi("\n Examples m: m[0]= %le + %le*I, m[N3-1]= %le + %le*I\n",static_cast<complex<double> *> (axion->mCpu())[S0].real(), static_cast<complex<double> *> (axion->mCpu())[S0].imag(),
		 								  static_cast<complex<double> *> (axion->mCpu())[SF].real(), static_cast<complex<double> *> (axion->mCpu())[SF].imag());
		printMpi("\n Examples v: v[0]= %le + %le*I, v[N3-1]= %le + %le*I\n",static_cast<complex<double> *> (axion->vCpu())[V0].real(), static_cast<complex<double> *> (axion->vCpu())[V0].imag(),
									 	  static_cast<complex<double> *> (axion->vCpu())[VF].real(), static_cast<complex<double> *> (axion->vCpu())[VF].imag());
	}
	else
	{
		printMpi("\n Examples m: m[0]= %f + %f*I, m[N3-1]= %f + %f*I\n",  static_cast<complex<float> *> (axion->mCpu())[S0].real(), static_cast<complex<float> *> (axion->mCpu())[S0].imag(),
										  static_cast<complex<float> *> (axion->mCpu())[SF].real(), static_cast<complex<float> *> (axion->mCpu())[SF].imag());
		printMpi("\n Examples v: v[0]= %le + %le*I, v[N3-1]= %le + %le*I\n\n",static_cast<complex<float> *> (axion->vCpu())[V0].real(), static_cast<complex<float> *> (axion->vCpu())[V0].imag(),
										  static_cast<complex<float> *> (axion->vCpu())[VF].real(), static_cast<complex<float> *> (axion->vCpu())[VF].imag());
	}

	printMpi("z_final = %f\n", *axion->zV());
	printMpi("#_steps = %i\n", counter);
	printMpi("#_prints = %i\n", index);
	printMpi("Total time: %2.3f s\n", elapsed.count()*1.e-3);
	printMpi("GFlops: %.3f\n", fCount->GFlops());
	printMpi("GBytes: %.3f\n", fCount->GBytes());
	printMpi("--------------------------------------------------\n");

	delete fCount;
	delete axion;

	endComms();

	printMemStats();


	//JAVIER
	fclose(file_sample);
	
	return 0;
}
