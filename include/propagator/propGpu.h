#ifndef	_PROP_GPU_
	#define	_PROP_GPU_

	void	propagateGpu	(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, PropParms ppar, const double dz, const double c, const double d,
		const uint Vo, const uint Vf, const VqcdType VQcd, FieldPrecision precision, const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream);

	void	updateMGpu	(void * __restrict__ m, const void * __restrict__ v, const double dz, const double d, const uint Lx, const uint Vo, const uint Vf, FieldPrecision precision,
				 const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream, FieldType fType = FIELD_SAXION);

	void	updateVGpu	(const void * __restrict__ m, void * __restrict__ v, PropParms ppar, const double dz, const double c, const uint Vo, const uint Vf, const VqcdType VQcd, FieldPrecision precision,
				 const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream);
#endif
