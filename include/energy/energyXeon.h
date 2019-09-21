#ifndef	_ENERGY_XEON_
	#define	_ENERGY_XEON_

	#include"scalar/scalarField.h"

	void	energyCpu	(Scalar *axionField, const double delta2, const double LL, const double nQcd, void *eRes, const double shift, const VqcdType VQcd, const EnType map=EN_ENE);

	void	energyThetaCpu	(Scalar *axionField, const double delta2, const double nQcd, void *eRes, const bool map=false, const bool wMod=false);
#endif
