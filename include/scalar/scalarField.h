#ifndef	_SCALAR_CLASS_
	#define	_SCALAR_CLASS_

	#include"enum-field.h"
	#include"utils/utils.h"
	#include"utils/tunable.h"

	class	Scalar : public Tunable
	{
		private:

		const size_t n1;
		const size_t n2;
		const size_t n3;

		const size_t Lz;
		const size_t Tz;
		const size_t Ez;
		const size_t v3;

		bool eReduced;
		size_t rLx;
		size_t rLz;
		size_t rTz;

		const int  nSplit;

		DeviceType	device;
		FieldPrecision	precision;
		FieldType	fieldType;
		LambdaType	lambdaType;

		size_t	fSize;
		size_t	mAlign;
		int	shift;
		bool	folded;
		bool	lowmem;

		double	*z;

		void	*m,   *v,   *m2,   *str;			// Cpu data
#ifdef	USE_GPU
		void	*m_d, *v_d, *m2_d;				// Gpu data

		void	*sStreams;
#endif
		void	recallGhosts(FieldIndex fIdx);		// Move the fileds that will become ghosts from the Cpu to the Gpu
		void	transferGhosts(FieldIndex fIdx);	// Copy back the ghosts to the Gpu

		/* Eliminar */

		template<typename Float>
		void axitonfinder(const Float contrastthreshold, void *idxbin, const int numaxitons); // TEST

		/* Fin eliminar */
		/* Falta: axitonfinder */

		public:

				 Scalar(const size_t nLx, const size_t nLz, FieldPrecision prec, DeviceType dev, const double zI, bool lowmem, const int nSp,
					FieldType newType, LambdaType lType, ConfType cType, const size_t parm1, const double parm2);
				~Scalar();

		void		*mCpu()  { return m; }
		const void	*mCpu()  const { return m; }
		void		*vCpu()  { return v; }
		const void	*vCpu()  const { return v; }
		void		*m2Cpu() { return m2; }
		const void	*m2Cpu() const { return m2; }

		void		*sData() { return str; }
		const void	*sData() const { return str; }
#ifdef	USE_GPU
		void		*mGpu() { return m_d; }
		const void	*mGpu() const { return m_d; }
		void		*vGpu() { return v_d; }
		const void	*vGpu() const { return v_d; }
		void		*m2Gpu() { return m2_d; }
		const void	*m2Gpu() const { return m2_d; }
#endif
		bool		LowMem()    		  { return lowmem; }
		void		setLowMem(const bool nLm) { lowmem = nLm; }

		size_t		TotalSize()  { return n3*nSplit; }
		size_t		Size()       { return n3; }
		size_t		Surf()       { return n2; }
		size_t		Length()     { return n1; }
		size_t		TotalDepth() { return Lz*nSplit; }
		size_t		Depth()      { return Lz; }
		size_t		rLength()    { return eReduced ? rLx : n1; }
		size_t		rTotalDepth(){ return eReduced ? rLz*nSplit : Lz*nSplit; }
		size_t		rDepth()     { return eReduced ? rLz : Lz; }
		size_t		rSize()      { return eReduced ? (rLx*rLx*rLz) : n3; }
		size_t		eDepth()     { return Ez; }
		size_t		eSize()      { return v3; }

		FieldPrecision	Precision()  { return precision; }
		DeviceType	Device()     { return device; }
		LambdaType	Lambda()     { return lambdaType; }
		FieldType	Field()      { return fieldType; }

		void		setLambda(LambdaType newLambda) { lambdaType = newLambda; }

		size_t		DataSize ()  { return fSize; }
		size_t		DataAlign()  { return mAlign; }
		int		Shift()      { return shift; }
		bool		Folded()     { return folded; }
		bool		Reduced()    { return eReduced; }


		double		*zV()        { return z; }
		const double	*zV() const  { return z; }

		void		setZ(const double newZ) { *z = newZ; }

		void	setField	(FieldType field);
		void	setFolded	(bool foli);
		void	setReduced	(bool eRed, size_t nLx = 0, size_t nLz = 0);

		void	transferDev(FieldIndex fIdx);		// Move data to device (Gpu or Xeon)
		void	transferCpu(FieldIndex fIdx);		// Move data to Cpu

		void	sendGhosts(FieldIndex fIdx, CommOperation cOp);	// Send the ghosts in the Cpu using MPI, use this to exchange ghosts with Cpus
		void	exchangeGhosts(FieldIndex fIdx);	// Transfer ghosts from neighbouring ranks, use this to exchange ghosts with Gpus

		/*	Eliminar	*/

		void	writeAXITONlist (double contrastthreshold, void *idxbin, int numaxitons);

		/*	Fin eliminar	*/

#ifdef	USE_GPU
		void	*Streams() { return sStreams; }
#endif
	};
#endif
