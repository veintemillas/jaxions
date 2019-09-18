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
#include "gen/genConf.h"

#include "meas/measa.h"
#include "WKB/WKB.h"

using namespace std;
using namespace AxionWKB;

int	main (int argc, char *argv[])
{
	Cosmos myCosmos = initAxions(argc, argv);

	commSync();

	MeasData lm;
	lm.str.strDen = 0 ;

	MeasInfo ninfa;
	ninfa.sliceprint = 0;
	ninfa.idxprint = 0 ;
	ninfa.index = 0;
	ninfa.measdata = MEAS_NOTHING;


	//-grids


	// interaction problematic issue FIX ME!!
	kCrit = 0.01;
	Scalar *axion;


	axion = new Scalar (&myCosmos, sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fTypeP, lType);


	LogOut("LOOP\n\n\n");

	switch (myCosmos.ICData().cType)
	{
		case 	CONF_VILGOR:
		case 	CONF_VILGORK:
		case 	CONF_VILGORS:
		case 	CONF_KMAX:{
					for(int i=0; i<30;i++)
					{
					// log space from
					myCosmos.ICData().kcr = exp(i*log(1000)/30.);
					for(int j=1; j<3;j++)
					{
					// kCrit changes the variable in the parse.h header read by initial condition generator
					// parm2 is also read by it
					LogOut("par1 %zu par2 %f kmax %zu kCrit %f cType %d (VILGOR %d) \n", parm1, parm2, myCosmos.ICData().kMax, myCosmos.ICData().kcr, myCosmos.ICData().cType, static_cast<int>(CONF_VILGOR));
					genConf	(&myCosmos, axion);
					ninfa.measdata = MEAS_STRING | MEAS_2DMAP | MEAS_STRINGMAP | MEAS_STRINGMAP | MEAS_SPECTRUM;
					lm = Measureme (axion, ninfa);
					ninfa.index += 1;
					}
					}
			}
			break;

		case	CONF_SMOOTH:
					{
						// SMOOTH version
							for(int i=0; i<30;i++)
							{
							// log space from 0 to 1000
							parm1 = exp(i*log(1000)/30.);
							myCosmos.ICData().siter = exp(i*log(1000)/30.);
							for(int j=1; j<3;j++)
							{
							LogOut("par1 %zu par2 %f kmax %zu kCrit %f cType %d (SMOOTH %d) \n", parm1, parm2, myCosmos.ICData().kMax, myCosmos.ICData().kcr, myCosmos.ICData().cType, static_cast<int>(CONF_SMOOTH));
							genConf	(&myCosmos, axion);
							ninfa.measdata = MEAS_STRING | MEAS_2DMAP | MEAS_STRINGMAP | MEAS_STRINGMAP | MEAS_SPECTRUM;
							lm = Measureme (axion, ninfa);
							ninfa.index += 1;
							}
							}
					}
		default:
			LogError ("Error: Invalid IC type. ");
			exit(1);
			break;
	}


	delete axion;
	endAxions();

	exit(0);

}
