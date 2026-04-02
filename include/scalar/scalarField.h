#ifndef	_SCALAR_CLASS_
	#define	_SCALAR_CLASS_

	#include"enum-field.h"
	#include"cosmos/cosmos.h"
	//#include"utils/utils.h"
	#include"utils/tunable.h"

	class	Scalar : public Tunable
	{
		private:

		Cosmos	*bckgnd;

		/* Geometry
		Nx Ny Nz in each rank,
		nSplit ranks
		total sites along z , Nzt = Nz*nSplit
		ghost only along z, Nx Ny Nz_g
		*/

		size_t Nx;
		size_t Ny;
		size_t Nz;
		size_t Nz_g;
		size_t Nxy;
		size_t Nxyz;

		size_t Tz;
		size_t Nxyz_g;	// volume with ghost

		bool eReduced;
		size_t rNx;		// reduced fields
		size_t rNy;		// reduced fields
		size_t rNz;
		size_t rTz;

		const int  nSplit;

		DeviceType	device;
		FieldPrecision	precision;
		FieldType	fieldType;
		LambdaType	lambdaType;
		StatusM2	statusM2;
		StatusM2	statusM2h;
		StatusSD  statusSD;

		size_t	fSize;
		size_t	mAlign;
		int	shift;
		bool	mmomspace;
		bool	vmomspace;
		bool folded, M2folded;
		bool	lowmem;
		bool    lowmemgpu;
		size_t Ng;
		size_t lap;

		size_t nmodes;

		// conformal time
		double	*z;
		// scale factor
		double	*R;
		// string core parameter
		double	msa;

		// propagation constants //FIX ME place in propClass?
		std::vector<double>	co;
		std::vector<double>	cop;




		// CPU data
		void *m   = nullptr, *v   = nullptr, *m2  = nullptr, *str = nullptr;
		//FAXION data
		void *rho = nullptr, *vho = nullptr, *g   = nullptr;
		//Aux data
		void *m_a  = nullptr, *v_a  = nullptr, *m2_a = nullptr;
		void *k_a  = nullptr, *k2_a = nullptr, *g_a  = nullptr;

#ifdef	USE_GPU
		void	*m_d, *v_d, *m2_d;				// Gpu data

		void	*sStreams;
#endif


		void	recallGhosts(FieldIndex fIdx);		// Move the fileds that will become ghosts from the Cpu to the Gpu
		void	transferGhosts(FieldIndex fIdx);	// Copy back the ghosts to the Gpu

		double	 rsvPQ2	    (const double z) { auto z2 = z*z; auto z3 = z2*z; auto z4 = z2*z2;
						       return (0.125*z + 0.30676113886283973*z2 + 0.20762392505082639*z3 + 0.03303541390146716*z4)/
						       (1 + 3.0165891109027165*z + 2.857822775289389*z2 + 0.8969613324856603*z3 + 0.05640260585369341*z4); }
		/* Eliminar */

		template<typename Float>
		void axitonfinder(const Float contrastthreshold, void *idxbin, const int numaxitons); // TEST

		/* Fin eliminar */
		/* Falta: axitonfinder */

		public:

				 Scalar(Cosmos *cm, const size_t nLx, const size_t nLz, FieldPrecision prec, DeviceType dev, const double zI, bool lowmem, const int nSp,
					FieldType newType, LambdaType lType, size_t Ngg=1);
				~Scalar();

		Cosmos		*BckGnd(){ return bckgnd; }

		/* Field pointers */
		void		*mCpu()  { return m; }
		const void	*mCpu()  const { return m; }
		void		*mStart      () { return static_cast<void *>(static_cast<char *>(m)  + fSize*(Nxy)*Ng); }
		void		*mFrontGhost () { return m; }
		void		*mBackGhost  () { return static_cast<void *>(static_cast<char *>(m)  + fSize*(Nxy*Ng+Nxyz)); }

		/* Velocity pointers */
		void		*vGhost () { return static_cast<void *>(static_cast<char *>(v) + fSize*(Nxyz)); }
		void		*vCpu()  { return v; }
		const void	*vCpu()  const { return v; }
		void		*vStart      () { return (fieldType == FIELD_PAXION) ? (static_cast<void *>(static_cast<char *>(v)  + fSize*(Nxy)*Ng)) : v; }
		void		*vFrontGhost () { return v; }
		void		*vBackGhost  () { return static_cast<void *>(static_cast<char *>(v)  + fSize*(Nxy*Ng+Nxyz)); }

		/* Auxiliary field pointers */
		void		*m2Cpu() { return m2; }
		const void	*m2Cpu() const { return m2; }
		void		*m2Start     () { return static_cast<void *>(static_cast<char *>(m2) + fSize*(Nxy)*Ng); }
		void		*m2FrontGhost() { return m2; }
		void		*m2BackGhost () { return static_cast<void *>(static_cast<char *>(m2) + fSize*(Nxy*Ng+Nxyz)); }
		void		*m2half      () { return static_cast<void *>(static_cast<char *>(m2) + (Nxyz_g)*precision); }
		/* m2h plus a ghost, used when fSize=precision because in complex mode fSize=2precision and the grid does not fit in m2h */
		void		*m2hStart    () { return static_cast<void *>(static_cast<char *>(m2) + (Nxyz_g)*precision + fSize*(Nxy)*Ng); }

		/* Auxiliary++ field pointers for linear mode evolution no ghost*/

		void		*m_aCpu()  { return m_a; }
		const void	*m_aCpu()  const { return m_a; }
		void		*v_aCpu()  { return v_a; }
		const void	*v_aCpu()  const { return v_a; }
		void		*m2_aCpu() { return m2_a; }
		const void	*m2_aCpu() const { return m2_a; }
		void		*g_aCpu() { return g_a; }
		const void	*g_aCpu() const { return g_a; }
		void		*k_Cpu()  { return k_a; }
		const void	*k_Cpu() const { return k_a; }
		void		*k2_Cpu()  { return k2_a; }
		const void	*k2_Cpu() const { return k2_a; }
		size_t		NModes ()  { return nmodes; }
		void	setNModes	(size_t niw) {nmodes = niw;} ;

		/* Faxion rho, vho, gx, gy, gx*/
		void		*rhoCpu       () { return                                         rho; }
		void		*rhoStart     () { return static_cast<void *>(static_cast<char *>(rho) + fSize*(Nxy)*Ng); }
		void		*rhoFrontGhost() { return                                         rho; }
		void		*rhoBackGhost () { return static_cast<void *>(static_cast<char *>(rho) + fSize*(Nxy*Ng+Nxyz)); }

		void		*vhoCpu       () { return                                         vho; }
		void		*vhoStart     () { return static_cast<void *>(static_cast<char *>(vho) + fSize*(Nxy)*Ng); }
		void		*vhoFrontGhost() { return                                         vho; }
		void		*vhoBackGhost () { return static_cast<void *>(static_cast<char *>(vho) + fSize*(Nxy*Ng+Nxyz)); }

		void		*gxCpu       () { return                                         g; }
		void		*gxStart     () { return static_cast<void *>(static_cast<char *>(g) + fSize*(Nxy)*Ng); }
		void		*gxFrontGhost() { return                                         g; }
		void		*gxBackGhost () { return static_cast<void *>(static_cast<char *>(g) + fSize*(Nxy*Ng+Nxyz)); }

		void		*gyCpu       () { return static_cast<void *>(static_cast<char *>(g) + (Nxyz_g)*precision                ); }
		void		*gyStart     () { return static_cast<void *>(static_cast<char *>(g) + (Nxyz_g)*precision + fSize*(Nxy)*Ng); }
		void		*gyFrontGhost() { return static_cast<void *>(static_cast<char *>(g) + (Nxyz_g)*precision                ); }
		void		*gyBackGhost () { return static_cast<void *>(static_cast<char *>(g) + (Nxyz_g)*precision + fSize*(Nxy*Ng+Nxyz)); }

		void		*gzCpu       () { return static_cast<void *>(static_cast<char *>(g) + 2*(Nxyz_g)*precision                ); }
		void		*gzStart     () { return static_cast<void *>(static_cast<char *>(g) + 2*(Nxyz_g)*precision + fSize*(Nxy)*Ng); }
		void		*gzFrontGhost() { return static_cast<void *>(static_cast<char *>(g) + 2*(Nxyz_g)*precision                ); }
		void		*gzBackGhost () { return static_cast<void *>(static_cast<char *>(g) + 2*(Nxyz_g)*precision + fSize*(Nxy*Ng+Nxyz)); }

		void		*sData() { return str; }
		const void	*sData() const { return str; }

#ifdef	USE_GPU
		void		*mGpu() { return m_d; }
		void		*mGpuStart() { return static_cast<void *>(static_cast<char *>(m_d)  + fSize*(Nxy)*Ng); }
		const void	*mGpu() const { return m_d; }
		void		*vGpu() { return v_d; }
		const void	*vGpu() const { return v_d; }
		void		*m2Gpu() { return m2_d; }
		const void	*m2Gpu() const { return m2_d; }
#endif
		bool		LowMem()    		  { return lowmem; }
		bool            LowMemGPU()                  { return lowmemgpu; }
		void		setLowMem(const bool nLm) { lowmem = nLm; }

		// OUTDATED
		size_t		TotalSize()  { return Nxyz*nSplit; }
		size_t		Size()       { return Nxyz; }
		size_t		Surf()       { return Nxy; }
		size_t		Length()     { return Nx; }
		size_t		TotalDepth() { return Nz*nSplit; }
		size_t		Depth()      { return Nz; }

		size_t		rLength()    { return eReduced ? rNx : Nx; }
		size_t		rTotalDepth(){ return eReduced ? rNz*nSplit : Nz*nSplit; }
		size_t		rDepth()     { return eReduced ? rNz : Nz; }
		size_t		rSize()      { return eReduced ? (rNx*rNy*rNz) : Nxyz; }
		size_t		eDepth()     { return Nz_g; }
		size_t		eSize()      { return Nxyz_g; }
		size_t    vSize()      { return Nxy*(Nz+2); }

		// TO STAY
		size_t		NX()         { return Nx; }
		size_t		NY()         { return Ny; }
		size_t		NZ()         { return Nz; }
		size_t		TZ()         { return Nz*nSplit; }
		size_t		NXY()        { return Nxy; }
		size_t		NXYZ()       { return Nxyz; }
		size_t		TXYZ()       { return Nxyz*nSplit; }

		size_t		rNX()        { return eReduced ? rNx : Nx; }
		size_t		rNY()        { return eReduced ? rNy : Ny; }
		size_t		rNZ()     { return eReduced ? rNz : Nz; }
		size_t		rTZ()        { return eReduced ? rNz*nSplit : Nz*nSplit; }
		size_t		rNXYZ()      { return eReduced ? (rNx*rNy*rNz) : Nxyz; }
		size_t		NZg()        { return Nz_g; }
		size_t		NXYZg()      { return Nxyz_g; }



		FieldPrecision	Precision()  { return precision; }
		DeviceType	Device()     { return device; }
		LambdaType	LambdaT()     { return lambdaType; }
		FieldType	Field()      { return fieldType; }
		StatusM2	m2Status()   { return statusM2; }
		StatusM2	m2hStatus()   { return statusM2h; }
		StatusSD  sDStatus()   { return statusSD;}
		void  setLambdaT (LambdaType newLambda) { lambdaType = newLambda; }

		size_t		DataSize ()  { return fSize; }
		size_t		DataAlign()  { return mAlign; }
		int			Shift()      { return shift; }
		bool		Folded()     { return folded; }
		bool		M2Folded()     { return M2folded; }
		bool		MMomSpace()     { return mmomspace; }
		bool		VMomSpace()     { return vmomspace; }
		bool		Reduced()    { return eReduced; }


		double		Delta()      { return bckgnd->PhysSize()/((double) Nx); }
		// double		Msa()        { return msa; } //sqrt(2.*bckgnd->Lambda())*Delta(); }

		/*	Overloading	*/
		double		AxionMass  ();
		double		AxionMassSq();
		double		IIAxionMassSqn(double z0, double z, int nn);
		double		IAxionMassSqn(double z0, double z, int nn);
		double		SaxionMassSq();
		double		HubbleMassSq();
		double		Rpp();
		double    HubbleConformal();
		double		SaxionShift();
		double		Saskia     ();
		double		dzSize     ();
		double		AxionMass  (const double zNow);
		double		AxionMassSq(const double zNow);
		double		SaxionMassSq(const double zNow);
		double		HubbleMassSq(const double zNow);
		double		SaxionShift(const double ct);
		double		Saskia     (const double zNow);
		double		dzSize     (const double zNow);
		double		Rfromct    (const double ct);
		double		LambdaP   (); // Returns the value of Lambda with the 1/z2 included IF needed
		double		Msa();

		double		*zV()        { return z; }
		const double	*zV() const  { return z; }

		double		*RV()        { return R; }
		const double	*RV() const  { return R; }

		void	setZ (const double newZ)    { *z = newZ; }
		void	setM2(const StatusM2 newM2) { statusM2 = newM2; }
		void	setM2h(const StatusM2 newM2) { statusM2h = newM2; }
		void	setSD(const StatusSD newSD) { statusSD = newSD; }

		void	setField	(FieldType field);
		void	setFolded	  (bool foli) { folded = foli; };
		void	setM2Folded	(bool foli) { M2folded = foli; };
		void	setMMomSpace	(bool foli);
		void	setVMomSpace	(bool foli);
		void	updateR ();
		void	setReduced	(bool eRed, size_t nLx = 0, size_t nLz = 0);
		void	setReduced	(bool eRed, size_t nLx, size_t nLy, size_t nLz);
		void	setDims	(size_t newnLx, size_t newnLz);
		void	setDims	(size_t newnLx, size_t newnLy, size_t newnLz);

		void	transferDev(FieldIndex fIdx);		// Move data to device (Gpu or Xeon)
		void	transferCpu(FieldIndex fIdx);		// Move data to Cpu

		/* Function to exchange ghosts
		sendGeneral allows exchange of two general slices
		sendGhost2 exchanges usual ghost slices of m,v, or m2, with option to ng<Ng
		exchangeGhosts does the standard exchange
		sendGhost is a previous version that works for the standard exchange
		sendghost 3 exchages stringData ghost
		exchangeStringGhost wraps send and receive*/

		void	sendGeneral(CommOperation opComm, size_t count, MPI_Datatype dataType, void* sendBufferB, void* receiveBufferF, void* sendBufferF, void* receiveBufferB);
		void	sendGhosts2(FieldIndex fIdx, CommOperation opComm, int ng = -1);
 		void	sendGhosts(FieldIndex fIdx, CommOperation cOp);	// Send the ghosts in the Cpu using MPI, use this to exchange ghosts with Cpus
		void	exchangeGhosts(FieldIndex fIdx);	// Transfer ghosts from neighbouring ranks, use this to exchange ghosts with Gpus

		void	sendGhosts3(CommOperation opComm);
		void	exchangeStringGhost();


		void    setModes ();

		size_t  getNg() {return Ng;}
		size_t  getLap() {return lap;}
		void	  setCO	(size_t newN);
		double  *getCO() {return &(co[0]); };
		double  *getCOp() {return &(cop[0]); };

		void  setDev(DeviceType newdev) { device = newdev; }
		/*	Eliminar	*/

		void	writeAXITONlist (double contrastthreshold, void *idxbin, int numaxitons);

		/*	Fin eliminar	*/

#ifdef	USE_GPU
		void	*Streams() { return sStreams; }
#endif
	};
#endif
