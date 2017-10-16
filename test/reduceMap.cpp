#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "utils/utils.h"
#include "io/readWrite.h"

using namespace std;

int	main (int argc, char *argv[])
{
	initAxions(argc, argv);

	commSync();
	LogOut("\n-------------------------------------------------\n");
	LogOut("\n           REDUCING CONTRAST MAP TO %d           \n", sizeN);
	LogOut("\n             (CUTTING MODES OF FFT)              \n");
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

	reduceEDens(fIndex, sizeN, sizeZ);

	LogOut ("Map reduced\n");

	endAxions();

	return 0;
}
