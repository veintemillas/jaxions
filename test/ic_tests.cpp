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



	kCrit = 0.01;
	Scalar *axion;


	axion = new Scalar (&myCosmos, sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fTypeP, lType, cType, parm1, parm2);


	LogOut("LOOP\n\n\n");

	switch (cType)
	{
		case 	CONF_VILGOR:
		case 	CONF_VILGORK:
		case 	CONF_VILGORS:
		case 	CONF_KMAX:{
					for(int i=0; i<30;i++)
					{
					// log space from
					kCrit = exp(i*log(1000)/30.);
					for(int j=1; j<3;j++)
					{
					// kCrit changes the variable in the parse.h header read by initial condition generator
					// parm2 is also read by it
					LogOut("par1 %d par2 %f kmax %d kCrit %f cType %d (VILGOR %d) \n",parm1,parm2,kMax,kCrit,cType,CONF_VILGOR);
					genConf	(&myCosmos, axion, cType, parm1, parm2);

					lm = Measureme (axion, ninfa, MEAS_STRING | MEAS_2DMAP | MEAS_STRINGMAP | MEAS_STRINGMAP | MEAS_SPECTRUM);
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
							iter = exp(i*log(1000)/30.);
							for(int j=1; j<3;j++)
							{
							LogOut("par1 %d par2 %f kmax %d kCrit %f cType %d (SMOOTH %d) \n",parm1,parm2,kMax,kCrit,cType,CONF_SMOOTH);
							genConf	(&myCosmos, axion, cType, parm1, parm2);

							lm = Measureme (axion, ninfa, MEAS_STRING | MEAS_2DMAP | MEAS_STRINGMAP | MEAS_STRINGMAP | MEAS_SPECTRUM);
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
