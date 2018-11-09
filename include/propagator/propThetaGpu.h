#ifndef	_PROP_THETA_GPU_
	#define	_PROP_THETA_GPU_

	void	propThetaGpu	(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, double *R, const double dz, const double c, const double d, const double delta2,
				 const double nQcd, const uint Lx, const uint Lz, const uint Vo, const uint Vf, FieldPrecision precision, const int xBlock, const int yBlock, const int zBlock,
				 cudaStream_t &stream, const bool wMod);
#endif
