#ifndef	_SPROP_CPU_
	#define	_SPROP_CPU_

	#include "scalar/scalarField.h"

	void	propSpecXeon	(Scalar *axionField, const double dz, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, const VqcdType VQcd);
	void	propSpecCpu	(Scalar *axionField, const double dz, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, const VqcdType VQcd);
#endif
