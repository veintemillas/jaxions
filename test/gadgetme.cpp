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
#include "reducer/reducer.h"
#include "gadget/gadget_output.h"
#include "projector/projector.h"

#include "WKB/WKB.h"

using namespace std;
using namespace AxionWKB;

int	main (int argc, char *argv[])
{

	double zendWKB = 10. ;
	Cosmos myCosmos = initAxions(argc, argv);
	size_t nPart = sizeN*sizeN*sizeN;

	if (nSteps==0)
	return 0 ;

	commSync();
	LogOut("\n\nUsage:   mpirun -n RANKS gadgetme --index X --zgrid RANKS --nologmpi --size N --redmp n --gadtype GADTYPE --mapvel\n");
	LogOut("         Creates N^3 particles, reducing the grid first to n^3 if needed\n");
	LogOut("Options: --gadtype [gad/gadmass/gadgrid]\n"); 
	LogOut("         If --part_vel is parsed the configurations contains velocities (to implement)\n");
	LogOut("         If gad is selected, --kcr sigma is the variance of the displacement (default = 1, recommended ~ 0.25)\n\n");

	LogOut("\n----------------------------------------------------------------------\n");
	LogOut("   GADGET axion.m.%05d > %d^3 = %5d particles   \n", fIndex, sizeN, nPart);
	LogOut("----------------------------------------------------------------------\n\n");

	Scalar *axion;
	LogOut ("Reading conf axion.%05d ...", fIndex);
	
	readConf(&myCosmos, &axion, fIndex);
	if (axion == NULL)
	{
		LogOut ("Error reading HDF5 file\n");
		exit (0);
	}
	LogOut ("... done!\n");

	/* Creates axion and reads energy into M2 */
	// double eMean = readEDens	(&myCosmos, &axion, fIndex);

	// LogOut("eMean = %lf\n",eMean);
	// LogOut("N_grid = %lu\n",axion->Length());
	// LogOut("Z_grid = %lu\n",axion->Depth());
	commSync();

	size_t Ngrid = axion->Length();

	if (sizeN < Ngrid)
		endredmap = sizeN*sizeN*sizeN;

	// if (endredmap > 0)
	// {
	// 	LogOut("Reduce\n");
	// 	LogOut("reduced energy map from N=%d to N=%d by smoothing\n",axion->Length(), endredmap);
	// 	double ScaleSize = ((double) axion->Length())/((double) endredmap);
	// 	double eFc  = 0.5*M_PI*M_PI*(ScaleSize*ScaleSize)/((double) axion->Surf());
	// 	size_t nLz = endredmap / commSize();

	// 	if (axion->Precision() == FIELD_DOUBLE) {
	// 		reduceField(axion, endredmap, nLz, FIELD_M2, [eFc = eFc] (int px, int py, int pz, complex<double> x) -> complex<double>
	// 				 { return x*exp(-eFc*(px*px + py*py + pz*pz)); });
	// 	} else {
	// 		reduceField(axion, endredmap, nLz, FIELD_M2, [eFc = eFc] (int px, int py, int pz, complex<float>  x) -> complex<float>
	// 				 { return x*((float) exp(-eFc*(px*px + py*py + pz*pz))); });
	// 	}

	// 	double newmean_local = 0.0 ;
	// 	size_t localsize = endredmap*endredmap*endredmap/commSize();

	// 	#pragma omp parallel for schedule(static) reduction(+:newmean_local)
	// 	for (size_t idx =0; idx < localsize; idx++)
	// 	{
	// 		newmean_local += (double) static_cast<float *> (axion->m2Cpu())[idx] ;
	// 	}
	// 	commSync();
	// 	MPI_Allreduce(&newmean_local, &eMean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	// 	Ngrid = endredmap;
	// 	size_t totalsize = Ngrid*Ngrid*Ngrid;
	// 	eMean /= (double) totalsize;
	// }

	bool map_velocity = false;
	if (axion->BckGnd()->ICData().part_vel)
		map_velocity = true;
	
	axion->exchangeGhosts(FIELD_M);
	axion->exchangeGhosts(FIELD_V);
	
	if (gadType == GAD_GRID)
		createGadget_Grid (axion,Ngrid,nPart,map_velocity);
	else if (gadType == GAD_MASS)
		createGadget_Mass (axion,Ngrid,nPart,map_velocity);
	else if (gadType == GAD)
		LogOut("Not yet implemented...");

	/* Save energy projection to compare with Gadget*/

	createMeas(axion, fIndex+1000);
	LogOut("\n\nSaving projection plot in axion.m.%05d ...",fIndex+1000);
	projectField(axion, [] (float x) -> float { return x ; } );
	writePMapHdf5 (axion);
	LogOut("done!\n\n");
	destroyMeas();

	endAxions();

	return 0;
}
