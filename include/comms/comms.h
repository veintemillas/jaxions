#ifndef	_COMMS_
	#define	_COMMS_

	#include "enum-field.h"

	int	initComms (int argc, char *argv[], int size, DeviceType dev);
	void	endComms();
	int	commRank();
	int	commAcc();
	void	commSync();
#endif
