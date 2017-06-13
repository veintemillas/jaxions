#ifndef	_ENERGY_XEON_
	#define	_ENERGY_XEON_

	#include"scalar/scalarField.h"

	void	energyXeon	(Scalar *axionField, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, void *eRes, const double shift, const VqcdType VQcd);
	void	energyCpu	(Scalar *axionField, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, void *eRes, const double shift, const VqcdType VQcd);

	void	energyThetaXeon	(Scalar *axionField, const double delta2, const double nQcd, const size_t Lx, const size_t V, const size_t S, void *eRes, const bool map=false);
	void	energyThetaCpu	(Scalar *axionField, const double delta2, const double nQcd, const size_t Lx, const size_t V, const size_t S, void *eRes, const bool map=false);
#endif
