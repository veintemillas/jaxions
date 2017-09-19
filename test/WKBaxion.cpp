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

//#include "WKB/WKB.h"

using namespace std;

int	main (int argc, char *argv[])
{

	double zendWKB = 10. ;
	initAxions(argc, argv);

	commSync();
	LogOut("\n-------------------------------------------------\n");
	LogOut("\n           WKB EVOLUTION to %f                   \n", zendWKB);
	LogOut("\n-------------------------------------------------\n");

	LogOut("\n-------------------------------------------------\n");


	// Arrange to a list that can be read


	// LogOut("--------------------------------------------------\n");
	// LogOut("           PARSED CONDITIONS                     \n\n");
	//
	// LogOut("Length =  %2.2f\n", sizeL);
	// LogOut("nQCD   =  %2.2f\n", nQcd);
	// LogOut("N      =  %ld\n",   sizeN);
	// LogOut("Nz     =  %ld\n",   sizeZ);
	// LogOut("zGrid  =  %ld\n",   zGrid);
	// LogOut("z      =  %2.2f\n", zInit);
	// LogOut("zthres   =  %3.3f\n", zthres);
	// LogOut("zthres   =  %3.3f\n", zrestore);
	// LogOut("mass   =  %3.3f\n", axionmass(zInit, nQcd, zthres, zrestore));
	// LogOut("--------------------------------------------------\n");


	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------
	Scalar *axion;

	LogOut ("reading conf %d ...", fIndex);
	readConf(&axion, fIndex);
	if (axion == NULL)
	{
		LogOut ("Error reading HDF5 file\n");
		exit (0);
	}
	LogOut ("\n");

	double z_now = (*axion->zV())	;
	LogOut("--------------------------------------------------\n");
	LogOut("           INITIAL CONDITIONS                     \n\n");

	LogOut("Length =  %2.2f\n", sizeL);
	LogOut("nQCD   =  %2.2f\n", nQcd);
	LogOut("N      =  %ld\n",   sizeN);
	LogOut("Nz     =  %ld\n",   sizeZ);
	LogOut("zGrid  =  %ld\n",   zGrid);
	LogOut("z      =  %2.2f\n", z_now);
	LogOut("zthr   =  %3.3f\n", zthres);
	LogOut("zres   =  %3.3f\n", zrestore);
	LogOut("mass   =  %3.3f\n\n", axionmass(z_now, nQcd, zthres, zrestore));
	if (axion->Precision() == FIELD_SINGLE)
	LogOut("precis = SINGLE(%d)\n",sPrec);
		else
	LogOut("precis = DOUBLE(%d)\n",sPrec);
	LogOut("--------------------------------------------------\n");

	//--------------------------------------------------
	//       NEW AXION
	//--------------------------------------------------

	LogOut ("creating new axion ... %d", fType );
// the new axion is always prepared in lowmem
	Scalar *axion2;
	// force lowmem in scalar mode -> to be done!
	axion2 = new Scalar (sizeN, sizeZ, sPrec, cDev, z_now, 1 , zGrid, FIELD_AXION, CONF_NONE, 0. , 0. );
	LogOut ("done !\n");

	WKB* wonka;
	wonka = new WKB(axion, axion2);

	wonka.doWKB(10.);


	endAxions();

	return 0;
}
