#ifndef	_PROP_THETA_
	#define	_PROP_THETA_

	#include "scalar/scalarField.h"
	#include "utils/flopCounter.h"

	double	propTheta	(Scalar *field, const double dz, const double nQcd, const double delta, FlopCounter *fCount);
#endif
