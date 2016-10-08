#ifndef	_SCALAR_CLASS_
	#define	_SCALAR_CLASS_

	#include<vector>
	#include<complex>
	#include"enum-field.h"

	class	Scalar
	{
		private:

		const int n1;
		const int n2;
		const int n3;

		const int Lz;
		const int Ez;
		const int v3;

		const int nSplit;

		void	*m,   *v,   *m2;			// Cpu data
		void	*m_d, *v_d, *m2_d;			// Gpu data
		double	*z;

		void	*sStreams;

		const bool lowmem;

		FieldPrecision	precision;
		int	fSize;


		void	recallGhosts(FieldIndex fIdx);		// Move the fileds that will become ghosts from the Cpu to the Gpu
		void	transferGhosts(FieldIndex fIdx);	// Copy back the ghosts to the Gpu

		public:

				 Scalar(const int nLx, const int nLz, FieldPrecision prec, const double zI, bool lowmem, const int nSp);
				 Scalar(const int nLx, const int nLz, FieldPrecision prec, const double zI, char fileName[], bool lowmem, const int nSp);
				~Scalar();

		void		*mCpu() { return m; }
		const void	*mCpu() const { return m; }
		void		*vCpu() { return v; }
		const void	*vCpu() const { return v; }
		void		*m2Cpu() { return m2; }
		const void	*m2Cpu() const { return m2; }

		void		*mGpu() { return m_d; }
		const void	*mGpu() const { return m_d; }
		void		*vGpu() { return v_d; }
		const void	*vGpu() const { return v_d; }
		void		*m2Gpu() { return m2_d; }
		const void	*m2Gpu() const { return m2_d; }

		int		Size()   { return n3; }
		int		Surf()   { return n2; }
		int		Length() { return n1; }
		int		Depth()  { return Lz; }
		int		eDepth() { return Ez; }
		int		eSize()  { return v3; }

		FieldPrecision	Precision() { return precision; }

		int		dataSize() { return fSize; }

		double		*zV() { return z; }
		const double	*zV() const { return z; }

		void		setZ(const double newZ) { *z = newZ; }

		void	transferGpu(FieldIndex fIdx);		// Move data to Gpu
		void	transferCpu(FieldIndex fIdx);		// Move data to Cpu

		void	sendGhosts(FieldIndex fIdx);		// Send the ghosts in the Cpu using MPI, use this to exchange ghosts with Cpus
		void	exchangeGhosts(FieldIndex fIdx);	// Transfer ghosts from neighbouring ranks, use this to exchange ghosts with Gpus

		void	fftCpu();				// Fast Fourier Transform in the Cpu
		void	fftGpu();				// Fast Fourier Transform in the Gpu

		void	prepareCpu(int *window);		// Sets the field for a FFT, prior to analysis

		void	squareGpu();				// Squares the m2 field in the Gpu
		void	squareCpu();				// Squares the m2 field in the Cpu

		void	*Streams() { return sStreams; }
	};
#endif
