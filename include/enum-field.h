#ifndef	_ENUM_FIELD_
	#define _ENUM_FIELD_
	#include<mpi.h>
	#include<vector>

	typedef	unsigned int uint;

	namespace	AxionEnum {
		typedef enum	FieldType_s
		{
			FIELD_SAXION	= 1,
			FIELD_AXION	= 2,
			FIELD_AXION_MOD	= 130,
			FIELD_WKB	= 6,
			FIELD_MOD	= 128,
			FIELD_SX_RD	= 257,
			FIELD_AX_RD	= 258,
			FIELD_AX_MOD_RD	= 386,
			FIELD_REDUCED	= 256,
		}	FieldType;

		typedef	enum	FieldIndex_s
		{
			FIELD_M   = 1,
			FIELD_V   = 2,
			FIELD_MV  = 3,
			FIELD_M2  = 4,
			FIELD_MM2 = 5,
			FIELD_M2V = 6,
			FIELD_ALL = 7,
			FIELD_MTOM2 = 16, // option for FFTs
			FIELD_M2TOM2 = 32, // option for FFTs
		}	FieldIndex;

		typedef enum	OrderType_s
		{
			EO_TO_LEX,
			LEX_TO_EO,
		}	OrderType;

		typedef enum	ParityType_s
		{
			PARITY_EVEN = 0,
			PARITY_ODD  = 1,
		}	ParityType;

		typedef enum	FieldPrecision_s
		{
			FIELD_DOUBLE = 8,
			FIELD_SINGLE = 4,
			FIELD_NONE   = 0,
	//		FIELD_HALF,
		}	FieldPrecision;

		typedef enum	FoldType_s
		{
			FOLD_ALL,
			UNFOLD_ALL,
			UNFOLD_SLICE,
			UNFOLD_SLICEYZ,
		}	FoldType;

		typedef	enum	StringType_s
		{
			STRING_XY_POSITIVE = 1,
			STRING_YZ_POSITIVE = 2,
			STRING_ZX_POSITIVE = 4,
			STRING_XY_NEGATIVE = 8,
			STRING_YZ_NEGATIVE = 16,
			STRING_ZX_NEGATIVE = 32,
			STRING_XY          = 9,
			STRING_YZ          = 18,
			STRING_ZX          = 36,
			STRING_ONLY        = 63, //9+18+36
			STRING_WALL	       = 64,
			STRING_MASK	       = 128,     //used to exclude spectra and energy sums
		}	StringType;

		typedef	enum	LambdaType_s
		{
			LAMBDA_FIXED       = 0,
			LAMBDA_Z2          = 1,
		}	LambdaType;

		typedef	enum	AxionMassType_s
		{
			AXIONMASS_POWERLAW,
			AXIONMASS_ZIGZAG,
		}	AxionMassType;


		typedef	enum	VqcdType_s
		{
			VQCD_1            = 1,		// QCD1 potential chi(1-RePhi/fa), PQ1 potential lambda(|Phi|^2-fa^2)^2/4
			VQCD_2            = 2,		// QCD2 potential chi(1-RePhi/fa)^2/2 + chi(ImPhi/fa)^2/2, PQ1 potential
			VQCD_1_PQ_2       = 4,		// QCD1 potential, PQ2 potential lambda(|Phi|^4-fa^4)^2/4
			VQCD_1N2          = 8,		// QCD1 [N=2] potential chi[(1-(RePhi/v)^2+(RePhi/v)^2], PQ1 potential
			VQCD_QUAD         = 16,		// QCD1 [N=2] potential chi[(1-(RePhi/v)^2+(RePhi/v)^2], PQ1 potential
			VQCD_0            = 32,  // cosine potential

			VQCD_1_RHO        = 8193,		// First version QCD potential, only rho evolution
			VQCD_2_RHO        = 8194,		// Second version QCD potential, only rho evolution
			VQCD_1_PQ_2_RHO   = 8196,		// PQ version QCD potential, only rho evolution
			VQCD_1N2_RHO      = 8200,
			VQCD_0_RHO        = 8216,

			VQCD_1_DRHO       = 16385,		// First version QCD potential, rho damping
			VQCD_2_DRHO       = 16386,		// Second version QCD potential, rho damping
			VQCD_1_PQ_2_DRHO  = 16388,		// PQ version QCD potential, rho damping
			VQCD_1N2_DRHO     = 16392,
			VQCD_0_DRHO       = 16408,

			VQCD_1_DALL       = 32769,		// First version QCD potential, full damping
			VQCD_2_DALL       = 32770,		// Second version QCD potential, full damping
			VQCD_1_PQ_2_DALL  = 32772,		// PQ version QCD potential, full damping
			VQCD_1N2_DALL     = 32776,		// PQ version QCD potential, full damping
			VQCD_0_DALL       = 32792,		// PQ version QCD potential, full damping

			VQCD_1_DRHO_RHO	     = 24577,	// First version QCD potential, rho damping and only rho evolution
			VQCD_2_DRHO_RHO	     = 24578,	// Second version QCD potential, rho damping and only rho evolution
			VQCD_1_PQ_2_DRHO_RHO = 24580,	// PQ version QCD potential, rho damping and only rho evolution
			VQCD_1N2_DRHO_RHO    = 24584,
			VQCD_0_DRHO_RHO      = 24600,

			VQCD_1_DALL_RHO	     = 40961,	// First version QCD potential, full damping and only rho evolution
			VQCD_2_DALL_RHO	     = 40962,	// Second version QCD potential, full damping and only rho evolution
			VQCD_1_PQ_2_DALL_RHO = 40964,	// PQ version QCD potential, full damping and only rho evolution
			VQCD_1N2_DALL_RHO    = 40968,
			VQCD_0_DALL_RHO      = 40984,

			/*	VQCD Masks	*/
			VQCD_TYPE	      = 63,		// Masks base potential
			VQCD_DAMP	      = 49152,	// Masks damping mode 16384+32768

			/*	VQCD Flags	*/
			VQCD_EVOL_RHO	      = 8192,
			VQCD_DAMP_RHO	      = 16384,
			VQCD_DAMP_ALL	      = 32768,
			VQCD_NONE	      = 0,
		}	VqcdType;

		typedef enum	ConfType_s
		{
			CONF_NONE       = 0,
			CONF_SMOOTH     = 1,
			CONF_KMAX       = 2,
			CONF_VILGOR     = 4,
			CONF_VILGORK    = 8,
			CONF_VILGORS    = 16,
			CONF_TKACHEV    = 32,
			CONF_READ       = 64,
			CONF_LOLA       = 128, // Clean VILGOR
			CONF_COLE       = 256,
		}	ConfType;

		typedef enum	ConfsubType_s
		{
		/* Why do we need explicit numbers here??? */
			CONF_RAND         = 0,
			CONF_STRINGXY     = 1,
			CONF_STRINGYZ     = 2,
			CONF_MINICLUSTER0 = 3,
			CONF_MINICLUSTER  = 4,
			CONF_AXNOISE      = 5,
			CONF_SAXNOISE     = 6,
			CONF_AX1MODE      = 7,
			CONF_PARRES       = 8,
		}	ConfsubType;

		typedef	enum	MomConfType_s
		{
			MOM_MFLAT    = 0,  // white noise prepared for M
			MOM_MSIN     = 1,  // Gaussian with a sinc amplitude
			MOM_MVSINCOS = 2,  // Tkachev sinc into M and its derivative into V
			MOM_MEXP     = 4,  // Gaussian with exp(-k/k_c) mean amplitude
			MOM_MEXP2    = 8,  // Gaussian with exp(-(k/k_c)^2) mean amplitude
			MOM_COLE     = 16,  // 1/sqrt(1+kt2) EXP
			MOM_KK       = 1024, 	// extra momentum factor
			MOM_KCOLE   = 1040,  //
		}	MomConfType;

		typedef enum	DeviceType_s
		{
			DEV_CPU,
			DEV_GPU,
		}	DeviceType;

		typedef enum	CommOperation_s
		{
			COMM_SEND = 1 ,
			COMM_RECV = 2,
			COMM_SDRV = 4,
			COMM_TESTS = 8,
			COMM_TESTR = 16,
			COMM_TEST  = 24,
			COMM_WAIT = 32,
		}	CommOperation;

		typedef enum	AllocType_s
		{
			ALLOC_TRACK = 0,
			ALLOC_ALIGN = 1,
		}	AllocType;

		typedef	enum	Int_EnergyIdx_s
		{
			TH_GRX  = 0,
			TH_GRY  = 1,
			TH_GRZ  = 2,
			TH_KIN  = 3,
			TH_POT  = 4,
			RH_GRX  = 5,
			RH_GRY  = 6,
			RH_GRZ  = 7,
			RH_KIN  = 8,
			RH_POT  = 9,
                        RH_RHO  = 10, // Average value of rho
			//masked values
			TH_GRXM = 11,
			TH_GRYM = 12,
			TH_GRZM = 13,
			TH_KINM = 14,
			TH_POTM = 15,
			RH_GRXM = 16,
			RH_GRYM = 17,
			RH_GRZM = 18,
			RH_KINM = 19,
			RH_POTM = 20,
			//aux values
			RH_RHOM = 21,
			MM_NUMM = 22,
		}	EnergyIdx;

		typedef	enum	LogLevel_s
		{
			LOG_MSG   = 1048576,
			LOG_DEBUG = 2097152,
			LOG_ERROR = 4194304,
		}	LogLevel;

		typedef	enum	LogMpi_s
		{
			ALL_RANKS,
			ZERO_RANK,
		}	LogMpi;

		typedef	enum	ProcessorBrand_s
		{
			GENUINE_INTEL,
			AUTHENTIC_AMD,
			UNKNOWN_BRAND,
		}	ProcessorBrand;

		typedef	enum	ProfType_s
		{
			PROF_SCALAR,
			PROF_GENCONF,
			PROF_PROP,
			PROF_TUNER,
			PROF_STRING,
			PROF_ENERGY,
			PROF_FOLD,
			PROF_HDF5,
			PROF_REDUCER,
			PROF_PROJECTOR,
			PROF_SPECTRUM_FILLBINS,
			PROF_SPECTRUM_NRUNLOOP,
			PROF_SPECTRUM_FFTM2,
			PROF_MEAS,
			PROF_FTFIELD,
		}	ProfType;

		typedef	enum	VerbosityLevel_s
		{
			VERB_SILENT    = 0,
			VERB_NORMAL    = 1,
			VERB_HIGH      = 2,
			VERB_DEBUG     = 3,
			VERB_PARANOID  = 4,
		}	VerbosityLevel;

		typedef	enum	PrintConf_s
		{
			PRINTCONF_NONE    = 0,
			PRINTCONF_INITIAL = 1,
			PRINTCONF_FINAL   = 2,
			PRINTCONF_WKB   	= 4,
			PRINTCONF_BOTH    = 3,
		}	PrintConf;

		typedef	struct	StringData_v
		{
			size_t				strDen;
			long long int	strChr;
			size_t				wallDn;
			double        strLen;
			double        strDeng;
			double        strVel;
			double        strVel2;
			double        strGam;
			size_t				strDen_local;
			long long int	strChr_local;
			size_t				wallDn_local;
			double        strLen_local;
			double        strDeng_local;
		}	StringData;

		typedef	struct	StringEnergyData_v
		{
			double rho_str;
			double rho_a;
			double rho_s;
			double rho_str_Vil;
			double rho_a_Vil;
			double rho_s_Vil;
			size_t nout;
			double rmask;
		}	StringEnergyData;

		typedef	enum	FFTtype_s {
			FFT_NOTYPE         = 0,
			FFT_CtoC_MtoM      = 1,
			FFT_CtoC_M2toM2    = 2,
			FFT_CtoC_MtoM2     = 3,
			FFT_CtoC_VtoM2     = 4,
			FFT_SPSX           = 5,
			FFT_SPAX           = 6,
			FFT_PSPEC_SX       = 7,
			FFT_PSPEC_AX       = 8,
			FFT_RDSX_M         = 9,
			FFT_RDSX_V         = 10,
			FFT_RHO_SX         = 11,
			FFT_RtoC_MtoM_WKB  = 12,
			FFT_RtoC_VtoV_WKB  = 13,
			FFT_RtoC_M2toM2_WKB= 14,
			FFT_CtoC_VtoV      = 15,
			FFT_CtoC_M2toM     = 16,
		}	FFTtype;

		typedef	enum	FFTdir_s {
			FFT_NONE   = 0,
			FFT_FWD    = 1,
			FFT_BCK    = 2,
			FFT_FWDBCK = 3,
		}	FFTdir;

		typedef	enum	PropcType_s {
			PROPC_NONE	    = 0,		// For parsing
			PROPC_BASE	    = 1,		// Optimised proppagator 1 neighbour
			PROPC_SPEC	    = 2,		// Spectral flag
			PROPC_FSPEC     = 4,		// Full Spectral flag
			PROPC_NNEIG     = 8,		// vectorised multi Ng laplacian propagator
		} PropcType;

		typedef	enum	PropType_s {
			PROP_NONE     = 0,		// For parsing
			PROP_BASE     = 1,    // Optimised propagator 1 neighbour
			PROP_SPEC     = 2,		// Spectral flag
			PROP_FSPEC    = 4,		// Full Spectral flag
			PROP_NNEIG    = 8,		// vectorised multi Ng laplacian propagator

			PROP_LEAP     = 16,
			PROP_OMELYAN2	= 32,
			PROP_OMELYAN4	= 64,
			PROP_RKN4     = 128,
			PROP_MLEAP    = 256,

			PROP_BLEAP     = 17,
			PROP_BOMELYAN2	= 33,
			PROP_BOMELYAN4	= 65,
			PROP_BRKN4     = 129,
			PROP_BMLEAP    = 257,

			PROP_SLEAP     = 18,
			PROP_SOMELYAN2	= 34,
			PROP_SOMELYAN4	= 66,
			PROP_SRKN4     = 130,
			PROP_SMLEAP     = 258,

			PROP_FSLEAP     = 20,
			PROP_FSOMELYAN2 = 36,
			PROP_FSOMELYAN4 = 68,
			PROP_FSRKN4     = 132,
			PROP_FSMLEAP    = 260,

			PROP_NLEAP      = 24,
			PROP_NOMELYAN2  = 40,
			PROP_NOMELYAN4  = 72,
			PROP_NRKN4      = 136,
			PROP_NMLEAP     = 264,

			PROP_MASK	      = 496,// 16+32+64+128+256 So far... Masks the integrator type, removing the spectral/N flags flag
			PROP_LAPMASK	  = 15, // Masks the laplacian type, removing the integrator time
		}	PropType;

		typedef	enum	SpectrumType_s {
			SPECTRUM_K	= 1,
			SPECTRUM_G	= 2,
			SPECTRUM_V	= 4,
			SPECTRUM_GV	= 6,
			SPECTRUM_P	= 8,
			SPECTRUM_GaS    = 16,
			SPECTRUM_GaSadd= 32,
			SPECTRUM_KS	= 65,
			SPECTRUM_GS	= 66,
			SPECTRUM_VS	= 68,
			SPECTRUM_GVS= 70,
			SPECTRUM_PS	= 72,
			SPECTRUM_NN	= 128,
			SPECTRUM_AK = 256,
			SPECTRUM_KK = 512,
			SPECTRUM_GG = 1024,
			SPECTRUM_VV	= 2048,
		}	SpectrumType;

		typedef	enum	SpectrumMaskType_s {
			SPMASK_FLAT	= 1,
			SPMASK_VIL	= 2,
			SPMASK_VIL2	= 4,
			SPMASK_REDO	= 8,
			SPMASK_SAXI	= 16,
			SPMASK_AXIT	= 32,
			SPMASK_AXIT2= 64,
		}	SpectrumMaskType;

		typedef	enum	StringMeasureType_s {
			STRMEAS_STRING = 0,
			STRMEAS_LENGTH = 1,
			STRMEAS_GAMMA  = 2,
			STRMEAS_ENERGY = 4,
		}	StringMeasureType;

// 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304

		typedef enum	DumpType_s
		{
		  DUMP_EVERYN	  = 1,
		  DUMP_FROMLIST	= 2,
		  DUMP_LOG	    = 4,	//not yet implemented
		}	DumpType;

		typedef	enum	FindType_s {
			FIND_MAX,
			FIND_MIN,
		}	FindType;

		typedef	enum	MapType_s {
			MAP_RHO   = 1,
			MAP_THETA = 2,
			MAP_ALL   = 3,
			MAP_NONE  = 0,
		}	MapType;

		typedef	enum	StatusM2_s {
			M2_ENERGY      = 0,
			M2_ENERGY_FFT  = 1,
			M2_STRINGMAP   = 2,
			M2_STRINGCOO   = 4,
			M2_DIRTY       = 8,
			M2_MASK_TEST   = 16,
			M2_ENERGY_RED  = 32,
			M2_ENERGY_AXI  = 64,
			M2_ENERGY_MASK_AXI_FFT = 128,
			M2_MASK_AXI2_FFT = 256,
			M2_MASK        = 512,    // Points outside string
			M2_ANTIMASK    = 1024,   // Points inside the string region (duplicated to mask complex energy)
		}	StatusM2;

		typedef	enum	StatusSD_s {
			SD_DIRY        = 0,
			SD_MAP         = 1,
			SD_STRINGCOORD = 2,
			SD_MASK        = 4,
			SD_MAPMASK     = 5,
			SD_AXITONMASK  = 6,
		}	StatusSD;

		// analysis functions to be called inside a measurement
		typedef	enum	MeasureType_s {
			MEAS_NOTHING			= 0,
			MEAS_BINTHETA     = 1,
			MEAS_BINRHO       = 2,
			MEAS_BINLOGTHETA2 = 4,
			MEAS_BINDELTA     = 8,
			MEAS_ALLBIN       = 15,
			// MEAS_BIN...	  = 16,
			MEAS_STRING	      = 32,
			MEAS_STRINGMAP    = 64,
			MEAS_STRINGCOO    = 128,

			MEAS_ENERGY       = 256,  // energy
			MEAS_ENERGY3DMAP  = 512,  // 3D energy map to reduced dimensions
			MEAS_REDENE3DMAP  = 1024, // 3D energy map to reduced dimensions
			MEAS_2DMAP        = 2048, // slice of m, v
			MEAS_3DMAP        = 4096, // 3D configuration

			MEAS_MASK         = 8192, // experimental
			MEAS_PSP_A        = 16384,
			MEAS_PSP_S        = 32768,
			MEAS_NSP_A        = 65536,
			MEAS_NSP_S        = 131072,
			MEAS_NNSPEC       = 262144, 	// number of modes per bin for normalisation purposes
			// MASK for any spectrum
			// MEAS_SPECTRUM     = 507904, 		//  245760, 	// 16384 + 32768 + 65536 + 131072 (any of the spectra)
			MEAS_SPECTRUM     = 516096, 		//  245760, 	// 16384 + 32768 + 65536 + 131072 (any of the spectra)

			MEAS_SPECTRUMA    = 81920, 	  // 16384  + 65536  (any of the axion spectra)
			// MASK for those that require energy
			MEAS_NEEDENERGY   = 50952,				// 8 + 256 + 512 + 1024 + 16384 + 32768
			// MASK for those that require energy saved in m2
			MEAS_NEEDENERGYM2 = 50696,				// 8 + 512 + 1024 + 16384 + 32768
		}	MeasureType;

//Used when energy is called
		typedef enum	EnType_s
		{
			EN_NO         = 0,
			EN_ENE        = 1,
			EN_ONLYMAP    = 2, // used for maskinh
			EN_MAP        = 3, // total energy = 2
			EN_MASK       = 4, // masked energy from axions->SData() = masked
			EN_ENEMASK    = 5, // total + masked energies
			EN_MAPMASK    = 6, // map of masked energy
			EN_ENEMAPMASK = 7, // energy, masked energy and total map
			// EN_AMASK = 8, // Antimask
			// EN_AMAMA = 10, // Antimask
		}	EnType;

		// data given to measurement function (includes labels and analyses)
		typedef	struct	MeasInfo_v
		{
			int	index;
			size_t sliceprint	;
			size_t idxprint	;
			MeasureType measdata ;
			SpectrumMaskType mask ;
			double rmask; 					// a radius to mask
			std::vector<double> rmask_tab;
			int i_rmask;
			int redmap;
			StringMeasureType strmeas;
		}	MeasInfo;

		// data output by measurement function to program
		typedef	struct	MeasData_v
		{
			StringData	str;
			StringEnergyData strE;
			double			maxTheta;
			double 			eA;
			double 			eS;
		}	MeasData;

		// data output by measurement function to program
		typedef	struct	IcData_v
		{
			size_t        Nghost;
			bool          icdrule;
			bool          preprop;
			bool          icstudy;
			double        prepstL;
			double        prepcoe;
			double        pregammo;
			double        prelZ2e;
			VqcdType      prevtype;
			bool          normcore;
			double        alpha;
			size_t        siter;
			size_t        kMax ;
			double        kcr;
			double        mode0;
			double        zi;
			double        logi;
			double        kickalpha;
			double        extrav;
			ConfType      cType;
			ConfsubType   smvarType;
			MomConfType   mocoty;
		}	IcData;

		typedef	struct	MomParms_v
		{
			size_t        kMax;
			double        kCrt;
			MomConfType   mocoty;
			double        mass2;
			FieldType     ftype;
		}	MomParms;

#ifdef	__NVCC__
	#define	Attr	inline constexpr __host__ __device__
#else
	#define	Attr	inline constexpr
#endif
		template<typename enumFlag>
		Attr enumFlag  operator &  (enumFlag  lhs, const enumFlag rhs) { return static_cast<enumFlag>(static_cast<int>(lhs) & static_cast<int>(rhs)); }
		template<typename enumFlag>
		Attr enumFlag& operator &= (enumFlag &lhs, const enumFlag rhs) { lhs  = static_cast<enumFlag>(static_cast<int>(lhs) & static_cast<int>(rhs)); return lhs; }
		template<typename enumFlag>
		Attr enumFlag  operator |  (enumFlag  lhs, const enumFlag rhs) { return static_cast<enumFlag>(static_cast<int>(lhs) | static_cast<int>(rhs)); }
		template<typename enumFlag>
		Attr enumFlag& operator |= (enumFlag &lhs, const enumFlag rhs) { lhs  = static_cast<enumFlag>(static_cast<int>(lhs) | static_cast<int>(rhs)); return lhs; }
		template<typename enumFlag>
		Attr enumFlag  operator ^  (enumFlag  lhs, const enumFlag rhs) { return static_cast<enumFlag>(static_cast<int>(lhs) ^ static_cast<int>(rhs)); }
		template<typename enumFlag>
		Attr enumFlag& operator ^= (enumFlag &lhs, const enumFlag rhs) { lhs  = static_cast<enumFlag>(static_cast<int>(lhs) ^ static_cast<int>(rhs)); return lhs; }
#undef	Attr
	}	// End namespace

	using namespace AxionEnum;
#endif
