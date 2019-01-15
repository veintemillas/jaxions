#ifndef	_ENUM_FIELD_
	#define _ENUM_FIELD_
	#include<mpi.h>

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
			LAMBDA_FIXED,
			LAMBDA_Z2,
		}	LambdaType;

		typedef	enum	AxionMassType_s
		{
			AXIONMASS_POWERLAW,
			AXIONMASS_ZIGZAG,
		}	AxionMassType;


		typedef	enum	VqcdType_s
		{
			VQCD_1		       = 1,		// QCD1 potential chi(1-RePhi/fa), PQ1 potential lambda(|Phi|^2-fa^2)^2/4
			VQCD_2		       = 2,		// QCD2 potential chi(1-RePhi/fa)^2/2 + chi(ImPhi/fa)^2/2, PQ1 potential
			VQCD_1_PQ_2	     = 4,		// QCD1 potential, PQ2 potential lambda(|Phi|^4-fa^4)^2/4
			VQCD_1N2	       = 8,		// QCD1 [N=2] potential chi[(1-(RePhi/v)^2+(RePhi/v)^2], PQ1 potential
			VQCD_QUAD	       = 16,		// QCD1 [N=2] potential chi[(1-(RePhi/v)^2+(RePhi/v)^2], PQ1 potential

			VQCD_1_RHO	     = 8193,		// First version QCD potential, only rho evolution
			VQCD_2_RHO	     = 8194,		// Second version QCD potential, only rho evolution
			VQCD_1_PQ_2_RHO	 = 8196,		// PQ version QCD potential, only rho evolution
			VQCD_1N2_RHO     = 8200,

			VQCD_1_DRHO	     = 16385,		// First version QCD potential, rho damping
			VQCD_2_DRHO	     = 16386,		// Second version QCD potential, rho damping
			VQCD_1_PQ_2_DRHO = 16388,		// PQ version QCD potential, rho damping
			VQCD_1N2_DRHO    = 16392,

			VQCD_1_DALL	     = 32769,		// First version QCD potential, full damping
			VQCD_2_DALL	     = 32770,		// Second version QCD potential, full damping
			VQCD_1_PQ_2_DALL = 32772,		// PQ version QCD potential, full damping
			VQCD_1N2_DALL    = 32776,		// PQ version QCD potential, full damping

			VQCD_1_DRHO_RHO	     = 24577,	// First version QCD potential, rho damping and only rho evolution
			VQCD_2_DRHO_RHO	     = 24578,	// Second version QCD potential, rho damping and only rho evolution
			VQCD_1_PQ_2_DRHO_RHO = 24580,	// PQ version QCD potential, rho damping and only rho evolution
			VQCD_1N2_DRHO_RHO    = 24584,

			VQCD_1_DALL_RHO	     = 40961,	// First version QCD potential, full damping and only rho evolution
			VQCD_2_DALL_RHO	     = 40962,	// Second version QCD potential, full damping and only rho evolution
			VQCD_1_PQ_2_DALL_RHO = 40964,	// PQ version QCD potential, full damping and only rho evolution
			VQCD_1N2_DALL_RHO    = 40968,

			/*	VQCD Masks	*/
			VQCD_TYPE		= 15,		// Masks base potential
			VQCD_DAMP		= 49152,	// Masks damping mode 16384+32768

			/*	VQCD Flags	*/
			VQCD_EVOL_RHO		= 8192,
			VQCD_DAMP_RHO		= 16384,
			VQCD_DAMP_ALL		= 32768,
			VQCD_NONE		= 0,
		}	VqcdType;

		typedef enum	ConfType_s
		{
			CONF_KMAX,
			CONF_VILGOR,
			CONF_VILGORK,
			CONF_VILGORS,
			CONF_TKACHEV,
			CONF_SMOOTH,
			CONF_READ,
			CONF_NONE,
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
		}	ConfsubType;

		typedef	enum	MomConfType_s
		{
			MOM_MFLAT    = 0,  // white noise prepared for M
			MOM_MSIN     = 1,  // Gaussian with a sinc amplitude
			MOM_MVSINCOS = 2,  // Tkachev sinc into M and its derivative into V
			MOM_MEXP     = 4,  // Gaussian with exp(-k/k_c) mean amplitude
			MOM_MEXP2    = 8,  // Gaussian with exp(-(k/k_c)^2) mean amplitude
		}	MomConfType;

		typedef enum	DeviceType_s
		{
			DEV_CPU,
			DEV_GPU,
		}	DeviceType;

		typedef enum	CommOperation_s
		{
			COMM_SEND,
			COMM_RECV,
			COMM_SDRV,
			COMM_WAIT,
		}	CommOperation;

		typedef enum	AllocType_s
		{
			ALLOC_TRACK = 0,
			ALLOC_ALIGN = 1,
		}	AllocType;

		typedef	enum	Int_EnergyIdx_s
		{
			TH_GRX = 0,
			TH_GRY = 1,
			TH_GRZ = 2,
			TH_KIN = 3,
			TH_POT = 4,
			RH_GRX = 5,
			RH_GRY = 6,
			RH_GRZ = 7,
			RH_KIN = 8,
			RH_POT = 9,
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
		}	ProfType;

		typedef	enum	VerbosityLevel_s
		{
			VERB_SILENT = 0,
			VERB_NORMAL = 1,
			VERB_HIGH   = 2,
			VERB_DEBUG  = 3,
		}	VerbosityLevel;

		typedef	enum	PrintConf_s
		{
			PRINTCONF_NONE    = 0,
			PRINTCONF_INITIAL = 1,
			PRINTCONF_FINAL   = 2,
			PRINTCONF_BOTH    = 3,
		}	PrintConf;

		typedef	struct	StringData_v
		{
			size_t				strDen;
			long long int	strChr;
			size_t				wallDn;
			double        strLen;
			size_t				strDen_local;
			long long int	strChr_local;
			size_t				wallDn_local;
			double        strLen_local;
		}	StringData;

		typedef	enum	FFTtype_s {
			FFT_CtoC_MtoM,
			FFT_CtoC_M2toM2,
			FFT_CtoC_MtoM2,
			FFT_CtoC_VtoM2,
			FFT_SPSX,
			FFT_SPAX,
			FFT_PSPEC_SX,
			FFT_PSPEC_AX,
			FFT_RDSX_M,
			FFT_RDSX_V,
			FFT_RHO_SX,
			FFT_RtoC_MtoM_WKB,
			FFT_RtoC_VtoV_WKB,
			FFT_RtoC_M2toM2_WKB,
			FFT_NOTYPE,
		}	FFTtype;

		typedef	enum	FFTdir_s {
			FFT_NONE   = 0,
			FFT_FWD    = 1,
			FFT_BCK    = 2,
			FFT_FWDBCK = 3,
		}	FFTdir;

		typedef	enum	PropType_s {
			PROP_NONE	= 0,		// For parsing
			PROP_SPEC	= 1,		// Spectral flag
			PROP_LEAP	= 2,
			PROP_OMELYAN2	= 4,
			PROP_OMELYAN4	= 8,
			PROP_RKN4	= 16,
			PROP_SLEAP	= 3,
			PROP_SOMELYAN2	= 5,
			PROP_SOMELYAN4	= 9,
			PROP_SRKN4	= 17,
			PROP_MLEAP      = 32,
			PROP_SMLEAP     = 33,
			PROP_MASK	= 62,		// 2+4+8+16+32 So far... Masks the integrator type, removing the spectral flag
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
		}	SpectrumType;

		typedef	enum	SpectrumMaskType_s {
			SPMASK_FLAT	= 1,
			SPMASK_VIL	= 2,
			SPMASK_VIL2	= 4,
			SPMASK_TEST	= 8,
			SPMASK_SAXI	= 16,
		}	SpectrumMaskType;

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
		}	StatusM2;

		typedef	enum	StatusSD_s {
			SD_DIRY        = 0,
			SD_MAP         = 1,
			SD_STRINGCOORD = 2,
			SD_MASK        = 4,
			SD_MAPMASK     = 5,
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

			MEAS_ENERGY       = 256,
			MEAS_ENERGY3DMAP  = 512,
			MEAS_REDENE3DMAP  = 1024,
			MEAS_2DMAP        = 2048,
			MEAS_3DMAP        = 4096,

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

		// data given to measurement function (includes labels and analyses)
		typedef	struct	MeasInfo_v
		{
			int	index;
			size_t sliceprint	;
			size_t idxprint	;
			MeasureType measdata ;
			SpectrumMaskType mask ;
			double rmask; 					// a radius to mask
			int redmap;
		}	MeasInfo;

		// data output by measurement function to program
		typedef	struct	MeasData_v
		{
			StringData	str;
			double			maxTheta;
			double 			eA;
			double 			eS;
		}	MeasData;



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
