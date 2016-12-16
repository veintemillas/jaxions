#ifndef	_ENERGY_XEON_
	#define	_ENERGY_XEON_

	#include"scalar/scalarField.h"

	void	energyXeon	(Scalar *axionField, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, const size_t Vt, FieldPrecision precision, void *eRes);
	void	energyCpu	(Scalar *axionField, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, const size_t Vt, FieldPrecision precision, void *eRes);
	void	energyXeonV2	(Scalar *axionField, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, const size_t Vt, FieldPrecision precision, void *eRes);
	void	energyCpuV2	(Scalar *axionField, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, const size_t Vt, FieldPrecision precision, void *eRes);
#endif
