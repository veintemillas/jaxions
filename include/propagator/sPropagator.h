#ifndef	_SPROPAGATOR_
	#define	_SPROPAGATOR_

	#include "scalar/scalarField.h"

	void	sPropagate	(Scalar *field, const double dz, const double delta, const double nQcd=7., const double LL=15000., VqcdType pot=VQCD_2);
#endif
