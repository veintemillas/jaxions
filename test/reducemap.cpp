#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "propagator/allProp.h"
#include "energy/energy.h"
#include "enum-field.h"
#include "utils/utils.h"
#include "utils/misc.h"
#include "utils/logger.h"
#include "io/readWrite.h"
#include "comms/comms.h"
#include "map/map.h"
#include "strings/strings.h"
#include "powerCpu.h"
#include "scalar/scalar.h"

#include<mpi.h>

using namespace std;

int	main (int argc, char *argv[])
{
	initAxions(argc, argv);

	commSync();
	LogOut("\n-------------------------------------------------\n");
	LogOut("\n   REDUCING CONTRAST MAP FROM ?? TO 128          \n",fIndex);
	LogOut("\n-------------------------------------------------\n");

	LogOut("\n-------------------------------------------------\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------


	if (fIndex == -1)
	{
		LogOut("Error: configuration to be loaded not selected\n");
		return 0 ;
	}

	reduceEDens(fIndex, 128, 128);

	endAxions();

	return 0;
}
