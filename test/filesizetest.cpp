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

#include<mpi.h>
#include<omp.h>

using namespace std;

#ifdef	USE_XEON
	__declspec(target(mic)) char *mX, *vX, *m2X;
#endif

int	main (int argc, char *argv[])
{
	initAxions(argc, argv);

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	LogOut("\n-------------------------------------------------\n");
	LogOut("\n          TESTING SIZE LIMIT!                \n\n");


	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	start = std::chrono::high_resolution_clock::now();

	Scalar *axion;
	char fileName[256];

	LogOut("Generating scalar ... ");
	axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fType, lType, cType, parm1, parm2);
	LogOut("Done! \n");

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
	LogOut("ICtime %f min\n",elapsed.count()*1.e-3/60.);


	old = std::chrono::high_resolution_clock::now();
	LogOut("Complex to theta\n");
	cmplxToTheta (axion);
	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);
	LogOut("Complex2Theta time %f min\n",elapsed.count()*1.e-3/60.);

	old = std::chrono::high_resolution_clock::now();
	LogOut("Write conf\n");
	writeConf(axion, 0);
	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);
	LogOut("Write time time %f min\n",elapsed.count()*1.e-3/60.);

	delete axion;

	endAxions();

	return 0;
}
