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
	LogOut("--------------------------------------------------\n\n");
	LogOut("Usage:\n");
	LogOut("mpirun -np R redu --index i --zgrid R --size N --depth Z --redmp n --nologmpi --beta f \n");
	LogOut("mpirun -np %d redu --zgrid %d --index %d --size %d --depth %d --redmp %d --beta %f \n", 
			zGrid,zGrid, fIndex, axion->Length(), axion->Depth(), deninfa.redmap, myCosmos.ICData().beta);
	commSync();

	size_t nLx = deninfa.redmap;
	double ScaleSize = axion->Length()/deninfa.redmap;
	size_t nLz = (axion->Depth()*nLx)/axion->Length();

	// This is equivalent to Javi's filter
	double eFc  = 0.5*M_PI*M_PI*myCosmos.ICData().beta*myCosmos.ICData().beta*(ScaleSize*ScaleSize)/((double) axion->Surf());
	double nFc  = 1.;
	int    kMax = axion->Length()/ScaleSize;
	int    kMax2= (nLx/2)*(nLx/2);

	
	if (!axion->LowMem() && nLx >= 2 && nLz >= 2) {
		if (axion->Precision() == FIELD_DOUBLE) {
			reduced = reduceField(axion, nLx, nLz, FIELD_MV,
					[eFc = eFc, nFc = nFc] (int px, int py, int pz, complex<double> x) -> complex<double> { return x*((double) nFc*exp(-eFc*(px*px + py*py + pz*pz))); }, false);
		} else {
			reduced = reduceField(axion, nLx, nLz, FIELD_MV,
					[eFc = eFc, nFc = nFc, kMax2 = kMax2] (int px, int py, int pz, complex<float>  x) -> complex<float>  { return x*((float)  (nFc* ((px*px + py*py + pz*pz > kMax2) ? 0 : 1))); }, false);
		}

		writeConf(reduced, fIndex+1000);
		LogOut("Field reduced in file axion.%05d",fIndex+1000);

	} else {
		LogOut ("MPI z dimension too small, skipping reduction...\n");
	}


	delete reduced;
	delete axion;

	endAxions();

	return 0;
}
