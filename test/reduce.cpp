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
#include "scalar/scaleField.h"
#include "spectrum/spectrum.h"
#include "scalar/mendTheta.h"
#include "projector/projector.h"

#include "reducer/reducer.h"

#include "meas/measa.h"
#include "WKB/WKB.h"
#include "axiton/tracker.h"

using namespace std;
using namespace AxionWKB;



//-point to print
size_t idxprint = 0 ;
//- z-coordinate of the slice that is printed as a 2D map
size_t sliceprint = 0 ;


/* Program */

int	main (int argc, char *argv[])
{
	Cosmos myCosmos = initAxions(argc, argv);

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	commSync();

	LogOut("\n-------------------------------------------------\n");
	LogOut("\n--               VAXION 3D!                    --\n");
	LogOut("\n-------------------------------------------------\n\n");

	//-grids
	Scalar *axion;
	Scalar *reduced;


	LogOut("Reading initial conditions from file ... ");
	readConf(&myCosmos, &axion, fIndex, restart_flag);

	//- Measurement
	MeasData lm;
	//- number of plaquetes pierced by strings
	lm.str.strDen = 0 ;
	//- Info to measurement
	MeasInfo ninfa = deninfa;

	//-maximum value of the theta angle in the simulation
	double maximumtheta = M_PI;
	lm.maxTheta = M_PI;

	LogOut("\n");

	LogOut("--------------------------------------------------\n");
	LogOut("           REDUCE AND MEAS                        \n");
	LogOut("--------------------------------------------------\n");

	commSync();

	int ScaleSize = axion->Length()/deninfa.redmap;

	// This is equivalent to Javi's filter
	double eFc  = 0.5*M_PI*M_PI*kCrit*kCrit*(ScaleSize*ScaleSize)/((double) axion->Surf());
	double nFc  = 1.;
	int    kMax = axion->Length()/ScaleSize;



	if (!axion->LowMem() && axion->Depth()/ScaleSize >= 2) {
		if (axion->Precision() == FIELD_DOUBLE) {
			reduced = reduceField(axion, axion->Length()/ScaleSize, axion->Depth()/ScaleSize, FIELD_MV,
					[eFc = eFc, nFc = nFc] (int px, int py, int pz, complex<double> x) -> complex<double> { return x*((double) nFc*exp(-eFc*(px*px + py*py + pz*pz))); }, false);
		} else {
			reduced = reduceField(axion, axion->Length()/ScaleSize, axion->Depth()/ScaleSize, FIELD_MV,
					[eFc = eFc, nFc = nFc] (int px, int py, int pz, complex<float>  x) -> complex<float>  { return x*((float)  (nFc*exp(-eFc*(px*px + py*py + pz*pz)))); }, false);
		}

		writeConf(reduced, fIndex+1000);

	} else {
		LogOut ("MPI z dimension too small, skipping reduction...\n");
	}


	delete reduced;
	delete axion;

	endAxions();

	return 0;
}
