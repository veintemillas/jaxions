#include"utils/logger.h"
#include"utils/profiler.h"
#include"utils/parse.h"
#include"utils/memAlloc.h"

using namespace profiler;

int	initAxions(int argc, char *argv[]) {
	parseArgs	(argc, argv);

        if (initComms(argc, argv, zGrid, cDev, logMpi, verb) == -1)
	{
        	LogOut ("Error initializing devices and Mpi\n");
                return	1;
	}

	LogMsg (VERB_NORMAL, "Output folder set to %s", outDir);
	LogMsg (VERB_NORMAL, "FFTW wisdom folder set to %s", wisDir);

	initProfilers();

	return	0;
}

void	endAxions() {
	printMemStats();
	printProfStats();
	endComms();

	return;
}
