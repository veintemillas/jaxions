#ifndef	_PROPAGATOR_
	#define	_PROPAGATOR_

	#include "scalar/scalarField.h"

 	void	initPropagator	(PropType pType, Scalar *field, VqcdType pot, int Ng=-1);
	void	propagate	(Scalar *field, const double dz);
	void	resetPropagator	(Scalar *field);
	void	tunePropagator	(Scalar *field);
  void	initGravity	(Scalar *field);
#endif
