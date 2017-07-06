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
#include<omp.h>

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

	if (initComms(argc, argv, zGrid, cDev, verb) == -1)
	{
		printf ("Error initializing devices and Mpi\n");
		return 1;
	}

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	printMpi("\n-------------------------------------------------\n");
	printMpi("\n          TESTING CODE STUFF!                \n\n");


	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	FlopCounter *fCount = new FlopCounter;

	start = std::chrono::high_resolution_clock::now();

	Scalar *axion;
	char fileName[256];

	if (fIndex == -1) {
		//This generates initial conditions
		printMpi("Generating scalar... ");
		axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fType, cType, parm1, parm2, fCount);
		printMpi("Done! \n");
	} else {
		//This reads from an Axion.$fIndex file
		printMpi("Reading from file... ");
		readConf(&axion, fIndex);
		if (axion == NULL)
		{
			printMpi ("Error reading HDF5 file\n");
			exit (0);
		}
		printMpi("Done! \n");
	}

	axion->transferDev(FIELD_MV);

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
	printMpi("ICtime %f min\n",elapsed.count()*1.e-3/60.);

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
	printMpi("           STARTING TEST                   \n");
	printMpi("--------------------------------------------------\n");


	int counter = 0;
	int index = 0;

	commSync();

	void	*eRes, *str;
	trackAlloc(&eRes, 256);
	memset(eRes, 0, 256);
#if	defined(__MIC__) || defined(__AVX512F__)
	alignAlloc(&str, 64, (axion->Size()));
#elif	defined(__AVX__)
	alignAlloc(&str, 32, (axion->Size()));
#else
	alignAlloc(&str, 16, (axion->Size()));
#endif
	memset(str, 0, axion->Size());

	commSync();

	axion->SetLambda(LAMBDA_Z2)	;
	if (LAMBDA_FIXED == axion->Lambda())
		printMpi ("Lambda in FIXED mode\n");
	else
		printMpi ("Lambda in Z2 mode\n");

//	writeConf(axion, index);

	auto strDen = strings(axion, str);

	fCount->addTime(elapsed.count()*1.e-3);

	if (axion->LowMem())
		energy(axion, fCount, eRes, false, delta, nQcd, LL);
	else
		energy(axion, fCount, eRes, true,  delta, nQcd, LL);

	printMpi("Nstrings %lu\n", strDen.strDen);
	printMpi("Chiral   %ld\n", strDen.strChr);
	printMpi("Nwalls   %lu\n", strDen.wallDn);
	printMpi("GFlops %lf\tGBytes %lf\n", fCount->GFlops(), fCount->GBytes());

	auto S = axion->Surf();
	auto V = axion->Size();

	if	(axion->Precision() == FIELD_SINGLE) {
		double *eR = static_cast<double *>(eRes);
		printMpi("Energy: %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", eR[0], eR[1], eR[2], eR[3], eR[4], eR[5], eR[6], eR[7], eR[8], eR[9]);
		printMpi("Punto: %lf %lf\n", static_cast<float *>(axion->mCpu())[2*S], static_cast<float *>(axion->mCpu())[2*S+1]);
		printMpi("Punto: %lf %lf\n", static_cast<float *>(axion->mCpu())[2*(V+S)-2], static_cast<float *>(axion->mCpu())[2*(V+S)-1]);
	}	else	{
		double *eR = static_cast<double *>(eRes);
		printMpi("Energy: %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", eR[0], eR[1], eR[2], eR[3], eR[4], eR[5], eR[6], eR[7], eR[8], eR[9]);
		printMpi("Punto: %lf %lf\n", static_cast<double *>(axion->mCpu())[2*S], static_cast<double *>(axion->mCpu())[2*S+1]);
		printMpi("Punto: %lf %lf\n", static_cast<double *>(axion->mCpu())[2*(V+S)-2], static_cast<double *>(axion->mCpu())[2*(V+S)-1]);
	}

	createMeas(axion, index);
	writeString(str, strDen);
	writeEnergy(axion, eRes);
	writePoint(axion);
	destroyMeas();
/*
	printMpi("--------------------------------------------------\n");
	printMpi("TO THETA\n");
	cmplxToTheta (axion, fCount);
	fflush(stdout);
	printMpi("--------------------------------------------------\n");
	index++;

	writeConf(axion, index);

	energy(axion, fCount, eRes, true, delta);

	createMeas(axion, index);
	writeEnergy(axion, eRes);
	writePoint(axion);
	destroyMeas();
*/
	trackFree(&eRes, ALLOC_TRACK);
	trackFree(&str,  ALLOC_ALIGN);

	delete fCount;
	delete axion;

	endComms();

	printMemStats();

	return 0;
}
