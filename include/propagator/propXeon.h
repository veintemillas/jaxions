#ifndef	_PROP_CPU_
	#define	_PROP_CPU_

	#include "scalar/scalarField.h"

	void	propagateXeon	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision);
	void	propagateCpu	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision);
	void	propLowMemXeon	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision);
	void	propLowMemCpu	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision);
#endif
