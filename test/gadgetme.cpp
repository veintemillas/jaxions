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
#include "WKB/WKB.h"
#include "strings/strings.h"
#include "powerCpu.h"
#include "scalar/scalar.h"
#include "spectrum/spectrum.h"
#include "meas/measa.h"
#include "reducer/reducer.h"
#include "gadget/gadget_output.h"
#include "projector/projector.h"

using namespace std;
using namespace AxionWKB;

int	main (int argc, char *argv[])
{
    double zendWKB = 10.;
    Cosmos myCosmos = initAxions(argc, argv);
    size_t nPart = sizeN*sizeN*sizeN;
    double sigma = kCrit;

    commSync();
    
    LogOut("\n\nUsage:   mpirun -n <RANKS> gadgetme --index <X> --zgrid <RANKS> --nologmpi --size <N> --gadtype <GADTYPE> --part_vel --sm_vel --part_disp --kcr <DISP>\n");
    LogOut("         Creates N^3 particles, with options --gadtype [halo/void]\n");
    LogOut("--------------------------------------------------------------------");
    LogOut("\nGADGET axion.m.%05d > %d^3 = %5d particles     \n\n", fIndex, sizeN, nPart);

    Scalar *axion, *vaxion;

    /* axion is read from Paxion file */
    LogOut ("Reading conf axion.%05d ...", fIndex);
    readConf(&myCosmos, &axion, fIndex);
    if (axion == NULL) { LogOut ("Error reading HDF5 file\n"); exit (0); }
    LogOut ("... done!\n");
    
    bool map_velocity = false;
    bool sm_vel = false;
    bool part_disp = false;

    if (gadType == HALO)
    {
        LogOut("HALO mappings selected is %5f \n",sigma);
        LogOut("Displacing particles with sigma  = %5f ...\n",sigma);
    }

    if (gadType == VOID)
    {
        LogOut("VOID mapping selected");
        if (axion->BckGnd()->ICData().part_vel) map_velocity = true;
        if (axion->BckGnd()->ICData().sm_vel) sm_vel = true;
        if (axion->BckGnd()->ICData().part_disp) sm_vel = true;
    }
    
    double z_now = *axion->zV();
    /* Allocate memory to build velocity fields */
    if (sm_vel)
    {	
        LogOut("\nVelocity smoothing selected: allocating memory for velocity fields ...");
        vaxion = new Scalar(&myCosmos,sizeN,sizeZ,sPrec,cDev,z_now,true,zGrid,FIELD_WKB,LAMBDA_FIXED,CONF_NONE);
    }
    else { vaxion = NULL;}  	

    size_t Ngrid = axion->Length();
    
    /* Implement reduction here*/
    if (sizeN < Ngrid) endredmap = sizeN*sizeN*sizeN;
	
    commSync();
    axion->exchangeGhosts(FIELD_M);
    axion->exchangeGhosts(FIELD_V);
    
    /* Main function */	
    if      (gadType == VOID)  createGadget_Void (axion,vaxion,Ngrid,nPart,map_velocity,sm_vel,part_disp);
    else if (gadType == HALO)  createGadget_Halo (axion,Ngrid,nPart,sigma);

    bool save_check = false;
    if (save_check)
    {
        createMeas(axion, fIndex+1000);
        LogOut("\n\nSaving projection plot in axion.m.%05d ...",fIndex+1000);
        projectField(axion, [] (float x) -> float { return x ; } );
        writePMapHdf5 (axion);
        LogOut("done!\n\n");
        destroyMeas();
    }
    
    endAxions();
    return 0;
}


// TO ADD: REDUCTION OPTIONS 
         
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
