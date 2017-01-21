#ifndef	_STRINGS_GPU_
	#define	_STRINGS_GPU_

	size_t	stringGpu	(const void * __restrict__ m, const uint Lx, const uint V, const uint S, FieldPrecision precision, void * __restrict__ str, cudaStream_t &stream);
#endif
