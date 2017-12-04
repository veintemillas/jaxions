#ifndef	_STRINGS_GPU_
	#define	_STRINGS_GPU_

	uint3	stringGpu	(const void * __restrict__ m, const uint Lx, const uint Lz, const uint rLx, const uint rLz, const uint S, const uint V,
				 FieldPrecision precision, void * __restrict__ str, cudaStream_t &stream);
#endif
