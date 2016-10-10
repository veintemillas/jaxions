void	propagateGpu	(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, double *z, const double dz, const double c, const double d, const double delta2,
			 const double LL, const double nQcd, const int Lx, const int Lz, const int Vo, const int Vf, FieldPrecision precision, cudaStream_t &stream);

void	updateMGpu	(void * __restrict__ m, const void * __restrict__ v, const double dz, const double d, const int Lx, const int Vo, const int Vf, FieldPrecision precision,
			 cudaStream_t &stream);

void	updateVGpu	(const void * __restrict__ m, void * __restrict__ v, double *z, const double dz, const double c, const double delta2, const double LL, const double nQcd,
			 const int Lx, const int Lz, const int Vo, const int Vf, FieldPrecision precision, cudaStream_t &stream);
