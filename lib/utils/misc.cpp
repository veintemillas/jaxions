#include"utils/logger.h"
#include"utils/profiler.h"
#include"utils/parse.h"
#include"utils/memAlloc.h"

using namespace profiler;

Cosmos	initAxions(int argc, char *argv[]) {
	parseArgs(argc, argv);

	Cosmos myCosmos = createCosmos();

        if (initComms(argc, argv, zGrid, cDev, logMpi, verb) == -1)
	{
        	LogOut ("Error initializing devices and Mpi\n");
                exit(1);
	}

	if (commRank() == 0)
		createOutput();

	LogMsg (VERB_NORMAL, "Output folder set to %s", outDir);
	LogMsg (VERB_NORMAL, "FFTW wisdom folder set to %s", wisDir);

	initProfilers();

	return	myCosmos;
}

void	endAxions() {
	printMemStats();
	printProfStats();
	endComms();

	return;
}
