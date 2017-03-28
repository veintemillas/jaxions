#ifndef	_PROP_THETA_CPU_
	#define	_PROP_THETA_CPU_

	#include "scalar/scalarField.h"

	void	propThetaXeon	(Scalar *axionField, const double dz, const double delta2, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision);
	void	propThetaCpu	(Scalar *axionField, const double dz, const double delta2, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision);
#endif
