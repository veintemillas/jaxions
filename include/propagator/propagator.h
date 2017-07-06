#ifndef	_PROPAGATOR_
	#define	_PROPAGATOR_

	#include "scalar/scalarField.h"
	#include "utils/flopCounter.h"

	//double	propagate	(Scalar *field, FlopCounter *fCount, const double dz, const double delta, const double nQcd=7., const double LL=15000., VqcdType pot=VQCD_2);
	double	propagate	(Scalar *field, const double dz, const double delta, const double nQcd=7., const double LL=15000., VqcdType pot=VQCD_2);
#endif
