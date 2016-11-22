#ifndef	_ENERGY_MAP_XEON_
	#define	_ENERGY_MAP_XEON_

	#include"scalar/scalarField.h"

	void	energyMapXeon	(Scalar *axionField, const double delta2, const double nQcd, const size_t Lx, const size_t V, const size_t S);
	void	energyMapCpu	(Scalar *axionField, const double delta2, const double nQcd, const size_t Lx, const size_t V, const size_t S);
#endif
