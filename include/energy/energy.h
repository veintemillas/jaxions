#ifndef	_ENERGY_BASE_
	#define	_ENERGY_BASE_

	#include "scalar/scalarField.h"
	#include "utils/flopCounter.h"

	void	energy	(Scalar *field, FlopCounter *fCount, void *eRes, const double delta, const double nQcd=7., const double LL=15000., VqcdType pot=VQCD_1, const double shift=0);
#endif
