#ifndef	_PROP_THETA_GPU_
	#define	_PROP_THETA_GPU_

	void	propThetaGpu	(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, PropParms ppar, const double dz, const double c, const double d,
		const uint Vo, const uint Vf, FieldPrecision precision, const int xBlock, const int yBlock, const int zBlock,
				 cudaStream_t &stream, const bool wMod);
#endif
