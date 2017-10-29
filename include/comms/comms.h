#ifndef	_COMMS_
	#define	_COMMS_

	#include "enum-field.h"

	int	initComms (int argc, char *argv[], int size, DeviceType dev, LogMpi logMpi, VerbosityLevel verb);
	void	endComms();
	int	commRank();
	int	commThreads();
	int	commSize();
	int	commAcc();
	void	commSync();
	size_t	gpuMemAvail();
#endif
