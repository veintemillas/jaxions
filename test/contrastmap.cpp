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

	if (initComms(argc, argv, zGrid, cDev, verb) == -1)
	{
		printf ("Error initializing devices and Mpi\n");
		return 1;
	}

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	commSync();
	printMpi("\n-------------------------------------------------\n");
	printMpi("\n   CREATING DENSITY CONTRAST MAP!(%d)           \n",fIndex);
	printMpi("\n-------------------------------------------------\n");

	printMpi("\n-------------------------------------------------\n");

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
			printMpi("No file selected!");
			//axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fType, cType, parm1, parm2, fCount);
			//printMpi("Done! \n");
		}
		else
		{
			//This reads from an Axion.00000 file
			printMpi ("reading conf %d ...", fIndex);
			readConf(&axion, fIndex);
			if (axion == NULL)
			{
				printMpi ("Error reading HDF5 file\n");
				exit (0);
			}
			else{
			printMpi ("Done!\n", fIndex);
			}
		}
	}

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
	printMpi("Reading time %f min\n",elapsed.count()*1.e-3/60.);

	void *eRes, *str;			// Para guardar la energia
	trackAlloc(&eRes, 128);
	memset(eRes, 0, 128);
	double *eR = static_cast<double *> (eRes);
	double  *binarray	 ;
	trackAlloc((void**) (&binarray),  10000*sizeof(size_t));
	double *bA = static_cast<double *> (binarray);

	double delta = sizeL/sizeN;
	double z_now = (*(axion->zV() ));
	int indexa = 10901 ;
	// creates energy map
	// posible problems with zthreshold, etc... but if mass was simple powerlaw, ok
	// version for theta only
	printMpi("ene \n");

	printMpi("%f %f %f %f \n",
	z_now,
	axionmass(z_now,nQcd,zthres, zrestore),
	static_cast<float*> (axion->mCpu())[sizeN*sizeN],
	static_cast<float*> (axion->vCpu())[0]
	);

	energy(axion, fCount, eRes, true, delta, nQcd, 0., VQCD_1, 0.);
	// bins density
	printMpi("bin \n");
	axion->writeMAPTHETA( (*(axion->zV() )) , indexa, binarray, 10000)		;
	// complex to real
	printMpi("auto \n");
	axion->autodenstom2() ;
	// Writes contrast map
	printMpi("write \n");
	writeEDens (axion, indexa) ;

	printMpi("z_final = %f\n", *axion->zV());
	printMpi("Total time: %2.3f min\n", elapsed.count()*1.e-3/60.);
	printMpi("Total time: %2.3f h\n", elapsed.count()*1.e-3/3600.);

	trackFree(&eRes, ALLOC_TRACK);
	trackFree((void**) (&binarray),  ALLOC_TRACK);

	delete fCount;
	delete axion;

	endComms();

	printMemStats();


	return 0;
}