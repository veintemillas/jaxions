#ifndef	_ENERGY_BASE_
	#define	_ENERGY_BASE_

	#include "scalar/scalarField.h"
	#include "utils/flopCounter.h"

	void	energy	(Scalar *field, void *eRes, const bool map, const double shift=0);
#endif
