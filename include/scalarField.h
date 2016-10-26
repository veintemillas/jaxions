#ifndef	_SCALAR_CLASS_
	#define	_SCALAR_CLASS_

	#include"enum-field.h"

//	#ifdef	USE_GPU
//		#include<cuda_runtime.h>
//	#endif

	#ifdef	USE_XEON
		#include "xeonDefs.h"
	#endif

	class	Scalar
	{
		private:

		const size_t n1;
		const size_t n2;
		const size_t n3;

		const size_t Lz;
		const size_t Tz;
		const size_t Ez;
		const size_t v3;

		const int  nSplit;
		const bool lowmem;

		DeviceType	device;
		FieldPrecision	precision;

		size_t	fSize;
		size_t	mAlign;

		double	*z;

		void	*m,   *v,   *m2;			// Cpu data
#ifdef	USE_GPU
		void	*m_d, *v_d, *m2_d;			// Gpu data

		void	*sStreams;
#endif
		void	recallGhosts(FieldIndex fIdx);		// Move the fileds that will become ghosts from the Cpu to the Gpu
		void	transferGhosts(FieldIndex fIdx);	// Copy back the ghosts to the Gpu

		void	scaleField(FieldIndex fIdx, double factor);
		void	randConf();
		void	smoothConf(const size_t iter, const double alpha);

		template<typename Float>
		void	iteraField(const size_t iter, const Float alpha);

		//JAVIER
		template<typename Float>
		void	normaCOREField(const Float alpha);
		template<typename Float>
		void	ENERGY(const Float zz, FILE *enWrite);//, Float &Grho1, Float &Gtheta1, Float &Vrho1, Float &Vtheta1, Float &Krho1, Float &Ktheta1);

		template<typename Float>
		void	momConf(const size_t kMax, const Float kCrit);

		public:

				 Scalar(const size_t nLx, const size_t nLz, FieldPrecision prec, DeviceType dev, const double zI, char fileName[], bool lowmem, const int nSp,
					ConfType cType, const size_t parm1, const double parm2);
				~Scalar();

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

		size_t		TotalSize() { return n3*nSplit; }
		size_t		Size()      { return n3; }
		size_t		Surf()      { return n2; }
		size_t		Length()    { return n1; }
		size_t		Depth()     { return Lz; }
		size_t		eDepth()    { return Ez; }
		size_t		eSize()     { return v3; }

		FieldPrecision	Precision() { return precision; }

		size_t		dataSize() { return fSize; }

		double		*zV() { return z; }
		const double	*zV() const { return z; }

		void		setZ(const double newZ) { *z = newZ; }

		void	foldField	();
		void	unfoldField	();
		void	unfoldField2D	(const size_t sZ);	// Just for the maps

		void	transferDev(FieldIndex fIdx);		// Move data to device (Gpu or Xeon)
		void	transferCpu(FieldIndex fIdx);		// Move data to Cpu

		//void	sendGhosts(FieldIndex fIdx);		// Send the ghosts in the Cpu using MPI, use this to exchange ghosts with Cpus
		void	sendGhosts(FieldIndex fIdx, CommOperation commOp);		// Send the ghosts in the Cpu using MPI, use this to exchange ghosts with Cpus
		//void	recvGhosts(FieldIndex fIdx);		// Send the ghosts in the Cpu using MPI, use this to exchange ghosts with Cpus
		//void	waitGhosts();				// Send the ghosts in the Cpu using MPI, use this to exchange ghosts with Cpus
		void	exchangeGhosts(FieldIndex fIdx);	// Transfer ghosts from neighbouring ranks, use this to exchange ghosts with Gpus

		void	fftCpu(int sign);			// Fast Fourier Transform in the Cpu
		void	fftGpu(int sign);			// Fast Fourier Transform in the Gpu


		void	prepareCpu(int *window);		// Sets the field for a FFT, prior to analysis

		void	squareGpu();				// Squares the m2 field in the Gpu
		void	squareCpu();				// Squares the m2 field in the Cpu

		void	genConf	(ConfType cType, const size_t parm1, const double parm2);

		void	writeENERGY (double zzz, FILE *enwrite);//, double &Gfr, double &Gft, double &Vfr, double &Vft, double &Kfr, double &Kft);
#ifdef	USE_GPU
		void	*Streams() { return sStreams; }
#endif
	};
#endif
