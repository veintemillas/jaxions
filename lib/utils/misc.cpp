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

	initProfilers();

	return	0;
}

void	endAxions() {
	printMemStats();
	printProfStats();
	endComms();

	return;
}
