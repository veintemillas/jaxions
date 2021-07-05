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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include "propagator/propGpu.h"

using namespace std;
using namespace AxionWKB;



//-point to print
size_t idxprint = 0 ;
//- z-coordinate of the slice that is printed as a 2D map
size_t sliceprint = 0 ;


void loadparams(PropParms *pipar, Scalar *axion);

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
	LogOut("           TEST PROP GPU                          \n");
	LogOut("--------------------------------------------------\n");
	LogOut(" N %d n %d kCrit %f \n",axion->Length(),deninfa.redmap,myCosmos.ICData().beta);
	commSync();


	/* transfer to device */
	LogOut("Folded? %d (0 false, 1 true) \n",axion->Folded());
	axion->transferDev(FIELD_MV);

	PropParms ppar;
	loadparams(&ppar, axion);

	/* propagate */
	double dz = 30;
	double c1 = 1;
	double d1 = 1;
	size_t uS =  ppar.Ng*axion->Surf();
	size_t uV =  axion->Size();
	size_t ext = uS+uV;

	size_t xBlock = 128 ;
	size_t yBlock = 1 ;
	size_t zBlock = 1 ;
	propagateGpu(axion->mGpu(), axion->vGpu(), axion->m2Gpu(), ppar, dz, c1, d1, 2*uS, uV, V_PQ1|V_QCD1, axion->Precision(), xBlock, yBlock, zBlock,
				((cudaStream_t *)axion->Streams())[2]);
	axion->exchangeGhosts(FIELD_M);
	propagateGpu(axion->mGpu(), axion->vGpu(), axion->m2Gpu(), ppar, dz, c1, d1, uS, 2*uS, V_PQ1|V_QCD1, axion->Precision(), xBlock, yBlock, zBlock,
				((cudaStream_t *)axion->Streams())[0]);
	if (uV>uS)
	propagateGpu(axion->mGpu(), axion->vGpu(), axion->m2Gpu(), ppar, dz, c1, d1, uV,  ext, V_PQ1|V_QCD1, axion->Precision(), xBlock, yBlock, zBlock,
				((cudaStream_t *)axion->Streams())[1]);

	dz = 0;
	propagateGpu(axion->m2Gpu(), axion->vGpu(), axion->mGpu(), ppar, dz, c1, d1, 2*uS, uV, V_PQ1|V_QCD1, axion->Precision(), xBlock, yBlock, zBlock,
				((cudaStream_t *)axion->Streams())[2]);
	axion->exchangeGhosts(FIELD_M);
	propagateGpu(axion->m2Gpu(), axion->vGpu(), axion->mGpu(), ppar, dz, c1, d1, uS, 2*uS, V_PQ1|V_QCD1, axion->Precision(), xBlock, yBlock, zBlock,
				((cudaStream_t *)axion->Streams())[0]);
	if (uV>uS)
	propagateGpu(axion->m2Gpu(), axion->vGpu(), axion->mGpu(), ppar, dz, c1, d1, uV,  ext, V_PQ1|V_QCD1, axion->Precision(), xBlock, yBlock, zBlock,
				((cudaStream_t *)axion->Streams())[1]);

	/* write result */
	axion->transferCpu(FIELD_MV);
	writeConf(axion, 2);

	delete axion;

	endAxions();

	return 0;
}

void loadparams(PropParms *pipar, Scalar *axion)
{
	(*pipar).lambda = axion->LambdaP();
	(*pipar).massA2 = axion->AxionMassSq();
	(*pipar).R      = *axion->RV();
	(*pipar).Rpp    = axion->Rpp();
	(*pipar).Rp     = axion->BckGnd()->Rp(*axion->zV());

	(*pipar).Ng     = axion->getNg();
	(*pipar).Lx     = axion->Length();;
	(*pipar).PC     = axion->getCO();
	(*pipar).ood2a  = 1./(axion->Delta()*axion->Delta());
	(*pipar).gamma  = axion->BckGnd()->Gamma();
	(*pipar).frw    = axion->BckGnd()->Frw();
	(*pipar).dectime= axion->BckGnd()->DecTime();

}
