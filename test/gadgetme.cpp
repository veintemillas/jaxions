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
	LogOut("\n   GADGET axion.m.%5d > %5d^3      particles     \n", fIndex, sizeN);
	LogOut("\n-------------------------------------------------\n");
	LogOut(" KCrit   = %5f \n",kCrit);
	LogOut("\n-------------------------------------------------\n");


	Scalar *axion;

	double eMean = readEDens	(&myCosmos, &axion, fIndex);

	LogOut("eMean = %lf\n",eMean);
	LogOut("N_grid = %lu\n",axion->Length());
	LogOut("Z_grid = %lu\n",axion->Depth());
	commSync();

	// for (int i=0; i< 10; i++)
	// 	printf("energy[%d,%d] = %f\n",commRank(),i,static_cast<float *> (axion->m2Cpu())[i]);

	size_t sizeNreducedgrid = axion->Length();

	if (sizeN < axion->Length())
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

		sizeNreducedgrid = endredmap;
		size_t totalsize = endredmap*endredmap*endredmap;
		eMean /= (double) totalsize;
	}


	size_t nPart = sizeNreducedgrid*sizeNreducedgrid*sizeNreducedgrid;

	LogOut("Ready to Gadget!\n");
	writeGadget(axion,eMean,sizeNreducedgrid,nPart,kCrit);

	//create measurement spectrum
	{
		createMeas(axion, fIndex+1);
		// writeEDens(axion);
		SpecBin specAna(axion, false); // no spectral flag
		specAna.pRun();
		writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");
		destroyMeas();
	}
	
	endAxions();

	return 0;
}
