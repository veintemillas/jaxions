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

using namespace std;

int	main (int argc, char *argv[])
{
	initAxions(argc, argv);

	LogOut("\n-------------------------------------------------\n");
	LogOut("\n          CREATING MINICLUSTERS!                \n\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	Scalar *axion;
	char fileName[256];

	if ((fIndex == -1) && (cType == CONF_NONE)) {
		LogOut("Error: Neither initial conditions nor configuration to be loaded selected. Empty field.\n");
	} else {
		if (fIndex == -1)
			//This generates initial conditions
			axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fType, lType, cType, parm1, parm2);
		else
		{
			//This reads from an Axion.$fIndex file
			readConf(&axion, fIndex);
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

	double delta = sizeL/sizeN;
	double dz;

	if (nSteps == 0)
		dz = 0.;
	else
		dz = (zFinl - zInit)/((double) nSteps);

	LogOut("--------------------------------------------------\n");
	LogOut("           INITIAL CONDITIONS                     \n\n");

	LogOut("Length =  %2.5f\n", sizeL);
	LogOut("N      =  %ld\n",   sizeN);
	LogOut("Nz     =  %ld\n",   sizeZ);
	LogOut("zGrid  =  %ld\n",   zGrid);
	LogOut("dx     =  %2.5f\n", delta);
	LogOut("dz     =  %2.5f\n", dz);
	LogOut("LL     =  %2.5f\n", LL);
	LogOut("--------------------------------------------------\n");

	const size_t S0 = sizeN*sizeN;
	const size_t SF = sizeN*sizeN*(sizeZ+1)-1;
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
#ifdef	__MIC__
	alignAlloc(&str, 64, (axion->Size()));
#elif defined(__AVX__)
	alignAlloc(&str, 32, (axion->Size()));
#else
	alignAlloc(&str, 16, (axion->Size()));
#endif
	memset(str, 0, axion->Size());

	commSync();

	if (fIndex == -1)
	{
		LogOut ("Dumping configuration %05d ...", index);
		writeConf(axion, index);
		LogOut ("Done!\n");
	}
	else
		index = fIndex;

	axion->SetLambda(LAMBDA_Z2);

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

	initPropagator (pType, axion, nQcd, delta, LL, VQCD_1);

	start = std::chrono::high_resolution_clock::now();
	old = start;

	int cnt = 0;

	for (int zloop = 0; zloop < nLoops; zloop++)
	{
		//--------------------------------------------------
		// THE TIME ITERATION SUB-LOOP
		//--------------------------------------------------

		index++;

		for (int zsubloop = 0; zsubloop < dump; zsubloop++)
			propagate (axion, dz);

		auto strDen = strings(axion, str);

		energy(axion, eRes, false, delta, nQcd, LL);

		profiler::Profiler &prof = profiler::getProfiler(PROF_PROP);

		auto pFler = prof.Prof().cbegin();
		auto pName = pFler->first;

		profiler::printMiniStats(*static_cast<double*>(axion->zV()), strDen, PROF_PROP, pName);

		createMeas(axion, index);
		writeEDens(axion, index);

		if (axion->Field() == FIELD_SAXION) {
			writeString(str, strDen);
		}

		writeMapHdf5(axion);
		writeEnergy(axion, eRes);
		writePoint(axion);
		destroyMeas();

		if (strDen.strDen == 0 && axion->Field() == FIELD_SAXION && strDen.wallDn == 0)
			cnt++;
		else
			cnt = 0;

		if (cnt == 40) {
			LogOut("--------------------------------------------------\n");
			LogOut("              TRANSITION TO THETA \n");
			LogOut("--------------------------------------------------\n");

			munge(UNFOLD_ALL);
			writeConf(axion, index);

			double saskia = 0.0;

			cmplxToTheta (axion, saskia);
		}
	} // zloop

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	LogOut("\n PROGRAMM FINISHED\n");

	munge(UNFOLD_ALL);
	writeConf(axion, index);

	LogOut("z_final = %f\n", *axion->zV());
	LogOut("#_steps = %i\n", counter);
	LogOut("#_prints = %i\n", index);
	LogOut("Total time: %2.3f s\n", elapsed.count()*1.e-3);

	trackFree(&eRes, ALLOC_TRACK);
	trackFree(&str,  ALLOC_ALIGN);

	delete axion;

	endAxions();

	return 0;
}
