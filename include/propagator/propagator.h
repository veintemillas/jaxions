#ifndef	_PROPAGATOR_
	#define	_PROPAGATOR_

	#include "scalar/scalarField.h"

	//void	propagate	(Scalar *field, const double dz, const double delta, const double nQcd=7., const double LL=15000., VqcdType pot=VQCD_2);
	void	initPropagator	(PropType pType, Scalar *field, const double nQcd=7., const double delta=0.0, const double LL=15000., VqcdType pot=VQCD_1);
	void	propagate	(Scalar *field, const double dz);
#endif
