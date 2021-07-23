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

#include "WKB/WKB.h"

using namespace std;
using namespace AxionWKB;

int	main (int argc, char *argv[])
{

	double zendWKB = 10. ;
	Cosmos myCosmos = initAxions(argc, argv);

	if (nSteps==0)
	return 0 ;

	//--------------------------------------------------
	//       AUX STUFF
	//--------------------------------------------------


	commSync();
	LogOut("\n-------------------------------------------------\n");
	LogOut("\n   GADGET axion.m.%5d > %5d        particles     \n", fIndex, sizeN);
	LogOut("\n-------------------------------------------------\n");
	LogOut(" ~ (%04d)^3                                      \n", pow( (double) sizeN ,1./3.));
	LogOut(" KCrit   = %5f \n",kCrit);
	LogOut("\n-------------------------------------------------\n");

	size_t nPart = sizeN;

	LogOut("(Usage: mpirun -n RANKS gadgetme --index X --zgrid RANKS --nologmpi --size N --redmp n --kcr sigma )\n");
	LogOut("(Usage: creates N^3 particulas reducing the grid first to n^3  )\n");
	LogOut("(Usage: sigma in latice units, default = 1, recommended ~ 0.25  )\n");

	Scalar *axion;

	/* Creates axion and reads energy into M2 */
	double eMean = readEDens	(&myCosmos, &axion, fIndex);

	LogOut("eMean = %lf\n",eMean);
	LogOut("N_grid = %lu\n",axion->Length());
	LogOut("Z_grid = %lu\n",axion->Depth());
	commSync();

	// for (int i=0; i< 10; i++)
	// 	printf("energy[%d,%d] = %f\n",commRank(),i,static_cast<float *> (axion->m2Cpu())[i]);


	size_t Ngrid = axion->Length();

	if (sizeN < Ngrid)
		endredmap = sizeN;

	if (endredmap > 0)
	{
		LogOut("Reduce\n");
		LogOut("reduced energy map from N=%d to N=%d by smoothing\n",axion->Length(), endredmap);
		double ScaleSize = ((double) axion->Length())/((double) endredmap);
		double eFc  = 0.5*M_PI*M_PI*(ScaleSize*ScaleSize)/((double) axion->Surf());
		size_t nLz = endredmap / commSize();

		if (axion->Precision() == FIELD_DOUBLE) {
			reduceField(axion, endredmap, nLz, FIELD_M2, [eFc = eFc] (int px, int py, int pz, complex<double> x) -> complex<double>
					 { return x*exp(-eFc*(px*px + py*py + pz*pz)); });
		} else {
			reduceField(axion, endredmap, nLz, FIELD_M2, [eFc = eFc] (int px, int py, int pz, complex<float>  x) -> complex<float>
					 { return x*((float) exp(-eFc*(px*px + py*py + pz*pz))); });
		}

		double newmean_local = 0.0 ;
		size_t localsize = endredmap*endredmap*endredmap/commSize();

		#pragma omp parallel for schedule(static) reduction(+:newmean_local)
		for (size_t idx =0; idx < localsize; idx++)
		{
			newmean_local += (double) static_cast<float *> (axion->m2Cpu())[idx] ;
		}
		commSync();
		MPI_Allreduce(&newmean_local, &eMean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		Ngrid = endredmap;
		size_t totalsize = Ngrid*Ngrid*Ngrid;
		eMean /= (double) totalsize;
	}

	LogOut("Ready to Gadget %lu!\n",nPart);
	writeGadget(axion,eMean,Ngrid,nPart,kCrit);

	//create measurement spectrum
	if ( !(defaultmeasType == MEAS_NOTHING) )
	{
		MeasInfo ninfa = deninfa;
		ninfa.index=fIndex+1;
		ninfa.measdata = defaultmeasType;
		ninfa.cTimesec = (double) Timer()*1.0e-6;
		ninfa.propstep = 1;
		MeasData lm = Measureme (axion, ninfa);
	}

	endAxions();

	return 0;
}
