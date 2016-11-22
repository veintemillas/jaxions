#ifndef	_ENERGY_MAP_
	#define	_ENERGY_MAP_

	#include "scalar/scalarField.h"
	#include "utils/flopCounter.h"

	void	energy	(Scalar *field, const double nQcd, const double delta, DeviceType dev, FlopCounter *fCount);
#endif
