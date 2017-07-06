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
	printMpi("\n          TESTING SIZE LIMIT!                \n\n");


	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	FlopCounter *fCount = new FlopCounter;

	start = std::chrono::high_resolution_clock::now();

	Scalar *axion;
	char fileName[256];

	printMpi("Generating scalar ... ");
	axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fType, cType, parm1, parm2, fCount);
	printMpi("Done! \n");

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
	printMpi("ICtime %f min\n",elapsed.count()*1.e-3/60.);


	old = std::chrono::high_resolution_clock::now();
	printMpi("Comlex to theta\n");
	cmplxToTheta (axion, fCount);
	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);
	printMpi("Complex2Theta time %f min\n",elapsed.count()*1.e-3/60.);

	old = std::chrono::high_resolution_clock::now();
	printMpi("Write conf\n");
	writeConf(axion, 0);
	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);
	printMpi("Write time time %f min\n",elapsed.count()*1.e-3/60.);

	delete fCount;
	delete axion;

	endComms();

	printMemStats();


	return 0;
}
