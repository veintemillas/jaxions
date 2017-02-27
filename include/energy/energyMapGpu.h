#ifndef	_ENERGY_MAP_GPU_
	#define	_ENERGY_MAP_GPU_

	#include"scalar/scalarField.h"

	void    energyMapGpu    (const void * __restrict__ m, const void * __restrict__ v, void * __restrict__ m2, double *z, const double delta2, const double nQcd,
	                         const uint Lx, const uint Lz, const uint V, const uint S, FieldPrecision precision, cudaStream_t &stream);
#endif
