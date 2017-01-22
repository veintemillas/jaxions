#ifndef	_FFT_CPU_
	#define	_FFT_CPU_

	void	initFFT	(void *m, void *m2, const size_t n1, const size_t Tz, FieldPrecision prec, bool lowmem);
	void	runFFT	(int sign);
	void	closeFFT();

	void	initFFTSpectrum	(void *m2, const size_t n1, const size_t Tz, FieldPrecision prec, bool lowmem);
	void	runFFTSpectrum	(int sign);
	void	closeFFTSpectrum();

	void	initFFThalo	(void *m, void *v, const size_t n1, const size_t Lz, FieldPrecision prec);
	void	runFFThalo(int sign);
	void	closeFFThalo();
#endif
