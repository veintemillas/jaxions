#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "propagator/allProp.h"
#include "energy/energy.h"
#include "energy/dContrast.h"
#include "utils/utils.h"
#include "io/readWrite.h"
#include "comms/comms.h"
#include "strings/strings.h"
#include "scalar/scalar.h"

#include <iostream>

#define	ScaleFactor 1.5

using namespace std;

int	main (int argc, char *argv[])
{
	Cosmos myCosmos = initAxions(argc, argv);

	LogOut("\n-------------------------------------------------\n");
	LogOut("\n          CREATING MINICLUSTERS!                \n\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	Scalar *axion;
	char fileName[256];

	std::cout << zInit << std::endl;

	if ((fIndex == -1) && (cType == CONF_NONE)) {
		LogOut("Error: Neither initial conditions nor configuration to be loaded selected. Empty field.\n");
	} else {
		if (fIndex == -1)
			//This generates initial conditions
			axion = new Scalar (&myCosmos, sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fTypeP, lType, cType, parm1, parm2);
		else
		{
			//This reads from an Axion.$fIndex file
			readConf(&myCosmos, &axion, fIndex);
			if (axion == nullptr)
			{
				LogOut ("Error reading HDF5 file\n");
				exit (0);
			}
		}
	}

	//--------------------------------------------------
	//          SETTING BASE PARAMETERS
	//--------------------------------------------------

	double delta = axion->Delta();
	double dz;

	if (nSteps == 0)
		dz = 0.;
	else
		dz = (zFinl - zInit)/((double) nSteps);

	LogOut("--------------------------------------------------\n");
	LogOut("           INITIAL CONDITIONS                     \n\n");

	LogOut("Length =  %2.5f\n", myCosmos.PhysSize());
	LogOut("N      =  %ld\n",   axion->Length());
	LogOut("Nz     =  %ld\n",   axion->Depth());
	LogOut("zGrid  =  %ld\n",   zGrid);
	LogOut("dx     =  %2.5f\n", delta);
	LogOut("dz     =  %2.5f\n", dz);
	LogOut("LL     =  %2.5f\n", myCosmos.Lambda());
	LogOut("--------------------------------------------------\n");

	const size_t S0 = axion->Surf();
	const size_t SF = axion->Size()-1+S0;
	const size_t V0 = 0;
	const size_t VF = axion->Size()-1;


	LogOut("INITIAL CONDITIONS LOADED\n");
	if (sPrec != FIELD_DOUBLE)
	{
		LogOut("Example mu: m[0] = %f + %f*I, m[N3-1] = %f + %f*I\n", ((complex<float> *) axion->mCpu())[S0].real(), ((complex<float> *) axion->mCpu())[S0].imag(),
									        ((complex<float> *) axion->mCpu())[SF].real(), ((complex<float> *) axion->mCpu())[SF].imag());
		LogOut("Example  v: v[0] = %f + %f*I, v[N3-1] = %f + %f*I\n", ((complex<float> *) axion->vCpu())[V0].real(), ((complex<float> *) axion->vCpu())[V0].imag(),
									        ((complex<float> *) axion->vCpu())[VF].real(), ((complex<float> *) axion->vCpu())[VF].imag());
	}
	else
	{
		LogOut("Example mu: m[0] = %lf + %lf*I, m[N3-1] = %lf + %lf*I\n", ((complex<double> *) axion->mCpu())[S0].real(), ((complex<double> *) axion->mCpu())[S0].imag(),
										    ((complex<double> *) axion->mCpu())[SF].real(), ((complex<double> *) axion->mCpu())[SF].imag());
		LogOut("Example  v: v[0] = %lf + %lf*I, v[N3-1] = %lf + %lf*I\n", ((complex<double> *) axion->vCpu())[V0].real(), ((complex<double> *) axion->vCpu())[V0].imag(),
										    ((complex<double> *) axion->vCpu())[VF].real(), ((complex<double> *) axion->vCpu())[VF].imag());
	}

	//--------------------------------------------------
	//   THE TIME ITERATION LOOP
	//--------------------------------------------------

	LogOut("--------------------------------------------------\n");
	LogOut("           STARTING COMPUTATION                   \n");
	LogOut("--------------------------------------------------\n");

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	int counter = 0;
	int index = 0;

	commSync();

	void *eRes, *str;			// Para guardar la energia
	trackAlloc(&eRes, 128);
	memset(eRes, 0, 128);

	commSync();

	if (fIndex == -1)
	{
		LogOut ("Dumping configuration %05d ...", index);
		writeConf(axion, index);
		LogOut ("Done!\n");
	}
	else
		index = fIndex;

	if (LAMBDA_FIXED == axion->Lambda())
		LogOut ("Lambda in FIXED mode\n");
	else
		LogOut ("Lambda in Z2 mode\n");

	Folder munge(axion);

	LogOut ("Folding configuration\n");
	munge(FOLD_ALL);

	if (cDev != DEV_CPU) {
		LogOut ("Transferring configuration to device\n");
		axion->transferDev(FIELD_MV);
	}

	if (dump > nSteps)
		dump = nSteps;

	int nLoops;

	if (dump == 0)
		nLoops = 0;
	else
		nLoops = (int)(nSteps/dump);

	LogOut ("Start redshift loop\n");

	commSync();

	start = std::chrono::high_resolution_clock::now();
	old = start;

	initPropagator (pType, axion, myCosmos.QcdPot());

	LogOut("--------------------------------------------------\n");
	LogOut("            TUNING PROPAGATOR                     \n");
	LogOut("--------------------------------------------------\n");

	tunePropagator (axion);

	for (int zloop = 0; zloop < nLoops; zloop++)
	{
		//--------------------------------------------------
		// THE TIME ITERATION SUB-LOOP
		//--------------------------------------------------

		index++;

		for (int zsubloop = 0; zsubloop < dump; zsubloop++)
			propagate (axion, dz);

		auto strDen = strings(axion);

		energy(axion, eRes, true);

		profiler::Profiler &prof = profiler::getProfiler(PROF_PROP);

		auto pFler = prof.Prof().cbegin();
		auto pName = pFler->first;

		profiler::printMiniStats(*static_cast<double*>(axion->zV()), strDen, PROF_PROP, pName);

		createMeas  (axion, index);
		writeMapHdf5(axion);
		writeEDens  (axion, MAP_ALL);
		writeString (axion, strDen);
		writeEnergy (axion, eRes);
		writePoint  (axion);
		cDensityMap (axion);
		destroyMeas ();

		// Test reduced strings
		createMeas  (axion, index+20000);
		axion->setReduced(true, axion->Length()/ScaleFactor, axion->Depth()/ScaleFactor);
		strDen = strings(axion);
		writeString (axion, strDen);
		axion->setReduced(false);
		destroyMeas ();
	} // zloop

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	LogOut("\n PROGRAMM FINISHED\n");

	munge(UNFOLD_ALL);

	if (cDev != DEV_CPU) {
		LogOut ("Transferring configuration to host\n");
		axion->transferCpu(FIELD_MV);
	}

	writeConf(axion, index);

	LogOut("z_final = %f\n", *axion->zV());
	LogOut("#_steps = %i\n", counter);
	LogOut("#_prints = %i\n", index);
	LogOut("Total time: %2.3f s\n", elapsed.count()*1.e-3);

	trackFree(eRes);

	delete axion;

	endAxions();

	return 0;
}
