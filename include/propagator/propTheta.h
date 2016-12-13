#ifndef	_PROPAGATOR_
	#define	_PROPAGATOR_

	#include "scalar/scalarField.h"
	#include "utils/flopCounter.h"

	double	propTheta	(Scalar *field, const double dz, const double LL, const double nQcd, const double delta, DeviceType dev, FlopCounter *fCount);
#endif
