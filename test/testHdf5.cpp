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
#include "spectrum/spectrum.h"
#include "scalar/scalar.h"
#include "reducer/reducer.h"

#include<mpi.h>
#include<omp.h>

#define	ScaleSize 2

using namespace std;

#ifdef	USE_XEON
	__declspec(target(mic)) char *mX, *vX, *m2X;
#endif

int	main (int argc, char *argv[])
{
	Cosmos myCosmos = initAxions(argc, argv);

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	LogOut("\n-------------------------------------------------\n");
	LogOut("\n          TESTING CODE STUFF!                \n\n");


	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	start = std::chrono::high_resolution_clock::now();

	Scalar *axion;
	char fileName[256];
	int index = 0;

	if (fIndex == -1) {
		//This generates initial conditions
		LogOut("Generating scalar... ");
		axion = new Scalar (&myCosmos, sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fTypeP, lType);
		LogOut("Done! \n");
	} else {
		//This reads from an Axion.$fIndex file
		LogOut("Reading from file... ");
		readConf(&myCosmos, &axion, fIndex);
		if (axion == nullptr)
		{
			LogOut ("Error reading HDF5 file\n");
			exit (0);
		}
		LogOut("Done! \n");

		index = fIndex + 1000;
	}

	axion->transferDev(FIELD_MV);

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
	LogOut("ICtime %f min\n",elapsed.count()*1.e-3/60.);

	//--------------------------------------------------
	//          SETTING BASE PARAMETERS
	//--------------------------------------------------

	double dz;
	double dzaux;
	double llaux;
	double llprint;

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
	LogOut("dx     =  %2.5f\n", axion->Delta());
	LogOut("dz     =  %2.5f\n", dz);
	LogOut("LL     =  %2.5f\n", myCosmos.Lambda());
	LogOut("--------------------------------------------------\n");

	const size_t S0 = axion->Surf();
	const size_t SF = axion->Size()-1+S0;
	const size_t V0 = 0;
	const size_t VF = axion->Size()-1;

	//--------------------------------------------------
	//   THE TIME ITERATION LOOP
	//--------------------------------------------------

	LogOut("--------------------------------------------------\n");
	LogOut("           STARTING TEST                   \n");
	LogOut("--------------------------------------------------\n");


	int counter = 0;

	commSync();

	writeConf(axion, index);

	void	*eRes;
	trackAlloc(&eRes, 256);
	memset(eRes, 0, 256);

	if (axion->Field() == FIELD_SAXION) {
		if (LAMBDA_FIXED == axion->LambdaT())
			LogOut ("Lambda in FIXED mode\n");
		else
			LogOut ("Lambda in Z2 mode\n");
	}

	commSync();

	if (axion->LowMem())
		energy(axion, eRes, false);
	else
		energy(axion, eRes, true);

	auto S = axion->Surf();
	auto V = axion->Size();

	double *eR = static_cast<double *>(eRes);

	double eMean = eR[0] + eR[1] + eR[2] + eR[3] + eR[4];

	if (axion->Field() == FIELD_SAXION)
		eMean += eR[5] + eR[6] + eR[7] + eR[8] + eR[9];

	if (axion->Field() == FIELD_SAXION)
		LogOut("Energy: %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf --> %lf\n", eR[0], eR[1], eR[2], eR[3], eR[4], eR[5], eR[6], eR[7], eR[8], eR[9], eMean);
	else
		LogOut("Energy: %lf %lf %lf %lf %lf -- > %lf\n", eR[0], eR[1], eR[2], eR[3], eR[4], eMean);

	createMeas(axion, index);

	if (axion->Field() == FIELD_SAXION) {
		auto strDen = strings(axion);

		LogOut("Nstrings %lu\n", strDen.strDen);
		LogOut("Chiral   %ld\n", strDen.strChr);
		LogOut("Nwalls   %lu\n", strDen.wallDn);

		writeString(axion, strDen);
	}

	writeEnergy(axion, eRes);
	writePoint(axion);

	if (!axion->LowMem()) {
		writeEDens(axion, MAP_ALL);
		if (axion->Precision() == FIELD_DOUBLE) {
			Binner<3000,double>contBin(static_cast<double*>(axion->m2Cpu()), axion->Size(),
						   [eMean = eMean] (double x) -> double { return (double) (log10(x/eMean) );});
			contBin.run();
			writeBinner(contBin, "/bins", "contB");
		} else {
			Binner<3000,float> contBin(static_cast<float *>(axion->m2Cpu()), axion->Size(),
						   [eMean = eMean] (float x) -> float { return (double) (log10(x/eMean) );});
			contBin.run();
			writeBinner(contBin, "/bins", "contB");
		}

		SpecBin specAna(axion, (pType & PROP_SPEC) ? true : false);
		specAna.pRun();
		writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");
		specAna.nRun();
		writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
		writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");
		writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");
	}

	destroyMeas();

//	LogOut("--------------------------------------------------\n");
//	LogOut("TO THETA\n");
//	cmplxToTheta (axion, 0., aMod);
//	fflush(stdout);
//	LogOut("--------------------------------------------------\n");
//	index++;
//
//	writeConf(axion, index);
//
//	energy(axion, eRes, true, delta);
//
//	createMeas(axion, index);
//	writeEnergy(axion, eRes);
//	writePoint(axion);
//	destroyMeas();

	Scalar *reduced;

	// This is equivalent to Javi's filter
	double eFc  = 0.5*M_PI*M_PI*(ScaleSize*ScaleSize)/((double) axion->Surf());
	double nFc  = 1.;
	int    kMax = axion->Length()/ScaleSize;


	if (!axion->LowMem() && axion->Depth()/ScaleSize >= 2) {
		if (axion->Precision() == FIELD_DOUBLE) {
			reduced = reduceField(axion, axion->Length()/ScaleSize, axion->Depth()/ScaleSize, FIELD_MV,
				  [eFc = eFc, nFc = nFc] (int px, int py, int pz, complex<double> x) -> complex<double> { return x*((double) nFc*exp(-eFc*(px*px + py*py + pz*pz))); }, false);
			energy(axion, eRes, true);
			//reduceField(axion, axion->Length()/ScaleSize, axion->Depth()/ScaleSize, FIELD_M2,
			//	  [eFc = eFc, nFc = nFc] (int px, int py, int pz, complex<double> x) -> complex<double> { return x*((double) nFc*exp(-eFc*(px*px + py*py + pz*pz))); });
			reduceField(axion, axion->Length()/ScaleSize, axion->Depth()/ScaleSize, FIELD_M2,
				  [kMax = kMax] (int px, int py, int pz, complex<double> x) -> complex<double> { return ((px*px + py*py + pz*pz) <= kMax*kMax) ? x : complex<double>(0.,0.); });
		} else {
			reduced = reduceField(axion, axion->Length()/ScaleSize, axion->Depth()/ScaleSize, FIELD_MV,
				  [eFc = eFc, nFc = nFc] (int px, int py, int pz, complex<float>  x) -> complex<float>  { return x*((float)  (nFc*exp(-eFc*(px*px + py*py + pz*pz)))); }, false);
			energy(axion, eRes, true);
			//reduceField(axion, axion->Length()/ScaleSize, axion->Depth()/ScaleSize, FIELD_M2,
			//	  [eFc = eFc, nFc = nFc] (int px, int py, int pz, complex<float>  x) -> complex<float>  { return x*((float)  (nFc*exp(-eFc*(px*px + py*py + pz*pz)))); });
			reduceField(axion, axion->Length()/ScaleSize, axion->Depth()/ScaleSize, FIELD_M2,
				  [kMax = kMax] (int px, int py, int pz, complex<float> x) -> complex<float> { return ((px*px + py*py + pz*pz) <= kMax*kMax) ? x : complex<float>(0.,0.); });
		}

		writeConf(reduced, index+100);
		delete reduced;

		auto str = strings(axion);
		createMeas (axion, index+100);
		writeEnergy(axion, eRes);
		writeString(axion, str);
		writeEDens (axion, MAP_ALL);
		destroyMeas();

	} else {
		LogOut ("MPI z dimension too small, skipping reduction...\n");
	}

	trackFree(eRes);

	delete axion;

	endAxions();

	return 0;
}
