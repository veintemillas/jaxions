#ifndef	_ENERGY_BASE_
	#define	_ENERGY_BASE_

	#include "scalar/scalarField.h"
	#include "utils/flopCounter.h"

	void	energy	(Scalar *field, const double LL, const double nQcd, const double delta, DeviceType dev, void *eRes, FlopCounter *fCount, VqcdType pot=VQCD_2, const double shift=0);
#endif
