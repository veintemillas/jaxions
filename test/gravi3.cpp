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
#include "meas/measa.h"

#include"fft/fftCode.h"
#include "gravity/potential.h"

using namespace std;

int	main (int argc, char *argv[])
{

	Cosmos myCosmos = initAxions(argc, argv);

	//--------------------------------------------------
	//       AUX STUFF
	//--------------------------------------------------

	commSync();
	LogOut("\n-------------------------------------------------\n");
	LogOut("\n Worksheet for gravitational potential solver    \n");
	LogOut("\n-------------------------------------------------\n");

	LogOut("\n-------------------------------------------------\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	Scalar *axion;

	LogOut ("reading conf %d ...", fIndex);
	readConf(&myCosmos, &axion, fIndex);
	if (axion == NULL)
	{
		LogOut ("Error reading HDF5 file\n");
		exit (0);
	}
	LogOut ("\n");


	double z_now = (*axion->zV())	;
	LogOut("--------------------------------------------------\n");
	LogOut("           READ CONDITIONS                     \n\n");

	LogOut("Length =  %2.2f\n", myCosmos.PhysSize());
	LogOut("nQCD   =  %2.2f\n", myCosmos.QcdExp());
	LogOut("N      =  %ld\n",   axion->Length());
	LogOut("Nz     =  %ld\n",   axion->Depth());
	LogOut("zGrid  =  %ld\n",   zGrid);
	LogOut("z      =  %2.2f\n", z_now);
	LogOut("zthr   =  %3.3f\n", myCosmos.ZThRes());
	LogOut("zres   =  %3.3f\n", myCosmos.ZRestore());
	LogOut("mass   =  %3.3f\n\n", axion->AxionMass());

	if (axion->Precision() == FIELD_SINGLE)
		LogOut("precis = SINGLE(%d)\n",FIELD_SINGLE);
	else
		LogOut("precis = DOUBLE(%d)\n",FIELD_DOUBLE);

	LogOut("--------------------------------------------------\n");

	//--------------------------------------------------
	//       MEASUREMENT
	//--------------------------------------------------

	int index = 0;

	/* Works in PAXION (and perhaps in AXION) */
	LogOut("> Loading Paxion \n");
	if (axion->Field() == FIELD_AXION)
		thetaToPaxion(axion);

	LogOut("> Paxion loaded \n");

	//--------------------------------------------------
	//       MEASUREMENT
	//--------------------------------------------------

	// size_t redfft  = axion->BckGnd()->ICData().kMax;
	// size_t smsteps = redfft*redfft;
	// size_t smstep2 = axion->BckGnd()->ICData().siter;

	// FILE *cacheFile = nullptr;
	// if (((cacheFile  = fopen("./red.dat", "r")) == nullptr)){
	// 	LogMsg(VERB_NORMAL,"No red.dat file use defaults kmax, kmax**2, siter");
	// } else {
	// 	fscanf (cacheFile ,"%lu ", &redfft);
	// 	fscanf (cacheFile ,"%lu ", &smsteps);
	// 	fscanf (cacheFile ,"%lu ", &smstep2);
	// 	LogOut("[gravi] red.dat file used \n");
	// 	LogOut("        redfft %d \n", redfft);
	// 	LogOut("        smsteps %d \n",smsteps);
	// 	LogOut("        smstep2 %d \n",smstep2);
	// }

	Folder munge(axion);

	if (!axion->Folded()){
		munge(FOLD_ALL);
	}

	axion->setM2(M2_ENERGY);
	InitGravity(axion);
	tuneGravity	( (unsigned int) 2048, (unsigned int) 32, (unsigned int) 8);
	tuneGravityHybrid	();

	// setHybridMode	(true);
	calculateGraviPotential	();

	// setHybridMode	(false);
	// calculateGraviPotential	();

	munge(UNFOLD_M2);
		createMeas(axion, 156);
		axion->setM2(M2_ENERGY);
			memmove(axion->m2Cpu(), axion->m2Start(), axion->Precision()*axion->Size());
			writeEMapHdf5s (axion,0);
			writeEDens(axion, MAP_THETA);
		destroyMeas();
		index++;


	LogOut ("Done!\n\n");

	endAxions();

	return 0;
}
