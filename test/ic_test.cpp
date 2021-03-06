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

#include<mpi.h>

using namespace std;

int	main (int argc, char *argv[])
{
	Cosmos myCosmos = initAxions(argc, argv);

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	commSync();
	LogOut("\n-------------------------------------------------\n");
	LogOut("\n   IC TEST !(%d)           \n",fIndex);
	LogOut("\n-------------------------------------------------\n");

	LogOut("\n-------------------------------------------------\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	start = std::chrono::high_resolution_clock::now();

	Scalar *axion;
	char fileName[256];

	if ((fIndex == -1) && (myCosmos.ICData().cType == CONF_NONE))
		LogOut("Error: Neither initial conditions nor configuration to be loaded selected. Empty field.\n");
	else
	{
		if (fIndex == -1)
		{
			//This generates initial conditions
			LogOut("Generating scalar ... ");
			axion = new Scalar (&myCosmos, sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fTypeP, lType);
			LogOut("Done! \n");
		}
		else
		{
			//This reads from an Axion.00000 file
			readConf(&myCosmos, &axion, fIndex);
			if (axion == NULL)
			{
				LogOut ("Error reading HDF5 file\n");
				exit (0);
			}
		}
	}

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
	LogOut("Reading/Creating time %f min\n",elapsed.count()*1.e-3/60.);

	void *eRes, *str;			// Para guardar la energia
	trackAlloc(&eRes, 128);
	memset(eRes, 0, 128);
	double *eR = static_cast<double *> (eRes);
	double  *binarray	 ;
	trackAlloc((void**) (&binarray),  10000*sizeof(size_t));
	double *bA = static_cast<double *> (binarray);

	double *sK = static_cast<double *> (axion->mCpu());

	double delta = axion->Delta();
	double z_now = (*(axion->zV() ));
	int indexa = 10901 ;

	// spectrum

	//--------------------------------------------------
	//       OUTPUT FILE
	//--------------------------------------------------

	FILE *file_thetis ;
	file_thetis = NULL;

	file_thetis = fopen("out/thetis.txt","w+");
	//--------------------------------------------------
	// 			COMPUTES THETA STATISTICS
	//--------------------------------------------------


	LogOut("ene \n");
	LogOut("%f %f %f %f \n", z_now, axion->AxionMass(z_now), static_cast<float*> (axion->mCpu())[sizeN*sizeN], static_cast<float*> (axion->vCpu())[0]);

	Folder munge(axion);
	munge(UNFOLD_ALL);

	if (commRank() == 0)
	{
		munge(UNFOLD_SLICE, 0);
//		writeMap (axion, 0);	DEPRECATED, USE writeMapHdf5 INSTEAD
	}

	LogOut("spexth \n");
	SpecBin spectrum = SpecBin(axion, false);
	spectrum.pRun();
	//powerspectrumexpitheta(axion) ;	DEPRECATED, USE SPECTRUM INSTEAD
	LogOut("write \n");

	if (commRank() == 0)
	{
		fprintf(file_thetis,  "%lf ", (*axion->zV()));

		for(int i = 0; i<spectrum.PowMax(); i++)
			fprintf(file_thetis, "%lf ", spectrum.data(SPECTRUM_P)[i]);

		fprintf(file_thetis, "\n");
	}

	if (commRank() == 0)
		for(int i = 0; i<10; i++)
			printf("%lf ", sK[i]);


	delete axion;

	endAxions();


	LogOut("\n End \n");
	return 0;
}
