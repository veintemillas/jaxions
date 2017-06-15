#ifndef	_ENERGY_MAP_
	#define	_ENERGY_MAP_

	#include "scalar/scalarField.h"
	#include "utils/flopCounter.h"

	void	energyMap	(Scalar *field, const double LL, const double nQcd, const double delta, DeviceType dev, void *eRes, FlopCounter *fCount, const VqcdType pot=VQCD_1, const double sh=0);
#endif
