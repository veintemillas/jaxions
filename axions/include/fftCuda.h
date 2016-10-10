int	initCudaFFT	(const int size, const int Lz, FieldPrecision prec);
int	runCudaFFT	(void *data, int sign);
void	closeCudaFFT	();
