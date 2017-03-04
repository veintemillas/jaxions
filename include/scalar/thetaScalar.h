#ifndef	_THETA_SCALAR_
	#define	_THETA_SCALAR_

	#include "scalar/scalarField.h"
	#include "utils/flopCounter.h"

	void	cmplxToTheta	(Scalar *field, FlopCounter *fCount, const double shift=0);
#endif
