#ifndef	_SCALAR_CLASS_
	#define	_SCALAR_CLASS_

	#include"enum-field.h"

	#ifdef	USE_GPU
//		#include<cuda.h>
		#include<cuda_runtime.h>
	#endif

	#ifdef	USE_XEON
		#include "xeonDefs.h"
	#endif

	extern double	kCrit;
	extern int	kMax;

	class	Scalar
	{
		private:

		const int n1;
		const int n2;
		const int n3;

		const int Lz;
		const int Tz;
		const int Ez;
		const int v3;

		const int nSplit;

		const bool lowmem;

		DeviceType	device;
		FieldPrecision	precision;

		int	fSize;
		int	mAlign;

		double	*z;

		void	*m,   *v,   *m2;			// Cpu data
#ifdef	USE_GPU
		void	*m_d, *v_d, *m2_d;			// Gpu data

		void	*sStreams;
#endif
		void	recallGhosts(FieldIndex fIdx);		// Move the fileds that will become ghosts from the Cpu to the Gpu
		void	transferGhosts(FieldIndex fIdx);	// Copy back the ghosts to the Gpu

		void	addZmom(int pz, int oPz, void *data, int sign);

		public:

//				 Scalar(const int nLx, const int nLz, FieldPrecision prec, DeviceType dev, const double zI, bool lowmem, const int nSp);
				 Scalar(const int nLx, const int nLz, FieldPrecision prec, DeviceType dev, const double zI, char fileName[], bool lowmem, const int nSp);
				~Scalar();

		void		genConf();

		void		*mCpu() { return m; }
		const void	*mCpu() const { return m; }
		void		*vCpu() { return v; }
		const void	*vCpu() const { return v; }
		void		*m2Cpu() { return m2; }
		const void	*m2Cpu() const { return m2; }

#ifdef	USE_XEON
		__attribute__((target(mic))) void	*mXeon() { return mX; }
		__attribute__((target(mic))) const void	*mXeon() const { return mX; }
		__attribute__((target(mic))) void	*vXeon() { return vX; }
		__attribute__((target(mic))) const void	*vXeon() const { return vX; }
		__attribute__((target(mic))) void	*m2Xeon() { return m2X; }
		__attribute__((target(mic))) const void	*m2Xeon() const { return m2X; }
#endif

#ifdef	USE_GPU
		void		*mGpu() { return m_d; }
		const void	*mGpu() const { return m_d; }
		void		*vGpu() { return v_d; }
		const void	*vGpu() const { return v_d; }
		void		*m2Gpu() { return m2_d; }
		const void	*m2Gpu() const { return m2_d; }
#endif
		bool		LowMem()  { return lowmem; }

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

		void	foldField	();
		void	unfoldField	();

		void	transferDev(FieldIndex fIdx);		// Move data to device (Gpu or Xeon)
		void	transferCpu(FieldIndex fIdx);		// Move data to Cpu

		//void	sendGhosts(FieldIndex fIdx);		// Send the ghosts in the Cpu using MPI, use this to exchange ghosts with Cpus
		void	sendGhosts(FieldIndex fIdx, CommOperation commOp);		// Send the ghosts in the Cpu using MPI, use this to exchange ghosts with Cpus
		//void	recvGhosts(FieldIndex fIdx);		// Send the ghosts in the Cpu using MPI, use this to exchange ghosts with Cpus
		//void	waitGhosts();				// Send the ghosts in the Cpu using MPI, use this to exchange ghosts with Cpus
		void	exchangeGhosts(FieldIndex fIdx);	// Transfer ghosts from neighbouring ranks, use this to exchange ghosts with Gpus

		void	fftCpu(int sign);			// Fast Fourier Transform in the Cpu
		void	fftCpu2(int sign);			// Fast Fourier Transform in the Cpu
		void	fftGpu(int sign);			// Fast Fourier Transform in the Gpu


		void	prepareCpu(int *window);		// Sets the field for a FFT, prior to analysis

		void	squareGpu();				// Squares the m2 field in the Gpu
		void	squareCpu();				// Squares the m2 field in the Cpu

		void	genConf	(const int kMax);
#ifdef	USE_GPU
		void	*Streams() { return sStreams; }
#endif
	};
#endif
