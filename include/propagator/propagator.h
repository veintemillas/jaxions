#ifndef	_PROPAGATOR_
	#define	_PROPAGATOR_

	#include "scalar/scalarField.h"
	#include "utils/flopCounter.h"

	double	propagate	(Scalar *field, const double dz, const double LL, const double nQcd, const double delta, FlopCounter *fCount, VqcdType pot=VQCD_2);
#endif
