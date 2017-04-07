#ifndef	_PROP_CPU_
	#define	_PROP_CPU_

	#include "scalar/scalarField.h"

	void	propagateXeon	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, const VqcdType VQcd);
	void	propagateCpu	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, const VqcdType VQcd);
	void	propLowMemXeon	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, const VqcdType VQcd);
	void	propLowMemCpu	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, const VqcdType VQcd);
#endif
