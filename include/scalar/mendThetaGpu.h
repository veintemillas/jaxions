#ifndef	_MEND_GPU_
	#define	_MEND_GPU_

	#include "scalar/scalarField.h"
	#include "enum-field.h"

	uint	mendThetaGpu	(void * __restrict__ m, void * __restrict__ v, const double z, const uint Lx, const uint Vo, const uint Vf, FieldPrecision precision, cudaStream_t &stream);
#endif
