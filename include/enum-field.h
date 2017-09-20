#ifndef	_ENUM_FIELD_
	#define _ENUM_FIELD_

	typedef	unsigned int uint;

	typedef enum	FieldType_s
	{
		FIELD_SAXION,
		FIELD_AXION,
		FIELD_WKB,
	}	FieldType;

	typedef	enum	FieldIndex_s
	{
		FIELD_M   = 1,
		FIELD_V   = 2,
		FIELD_MV  = 3,
		FIELD_M2  = 4,
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
		PARITY_ODD = 1,
	}	ParityType;

	typedef enum	FieldPrecision_s
	{
		FIELD_DOUBLE,
		FIELD_SINGLE,
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
		STRING_WALL	   = 64,
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
		VQCD_1,
		VQCD_2,
		VQCD_NONE,
	}	VqcdType;

	typedef enum	ConfType_s
	{
		CONF_KMAX,
		CONF_TKACHEV,
		CONF_SMOOTH,
		CONF_READ,
		CONF_NONE,
	}	ConfType;

	typedef enum	ConfsubType_s
	{
		CONF_RAND = 0,
		CONF_STRINGXY = 1,
		CONF_STRINGYZ = 2,
		CONF_MINICLUSTER0 = 3,
		CONF_MINICLUSTER = 4,
	}	ConfsubType;

	typedef enum	DeviceType_s
	{
		DEV_CPU,
		DEV_GPU,
		DEV_XEON,
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
		LOG_MSG   = 0,
		LOG_DEBUG = 1,
		LOG_ERROR = 2,
	}	LogLevel;

	typedef	enum	LogMpi_s
	{
		ALL_RANKS,
		ZERO_RANK,
	}	LogMpi;

	typedef	enum	ProfType_s
	{
		PROF_SCALAR,
		PROF_GENCONF,
		PROF_PROP,
		PROF_STRING,
		PROF_ENERGY,
		PROF_FOLD,
		PROF_HDF5,
	}	ProfType;

	typedef	enum	VerbosityLevel_s
	{
		VERB_SILENT=0,
		VERB_NORMAL=1,
		VERB_HIGH=2,
	}	VerbosityLevel;

	typedef	struct	StringData_v
	{
		size_t	strDen;
		size_t	strChr;
		size_t	wallDn;
	}	StringData;

	typedef	enum	FFTtype_s {
		FFT_CtoC_MtoM,
		FFT_CtoC_M2toM2,
		FFT_CtoC_MtoM2,
		FFT_CtoC_VtoM2,
		FFT_SPSX,
		FFT_SPAX,
		FFT_RtoC_M2toM2_AXION,
		FFT_RtoC_M2toM2_SAXION,
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

	inline FFTdir	operator &  (FFTdir  lhs, const FFTdir rhs) { return static_cast<FFTdir>(static_cast<int>(lhs) & static_cast<const int>(rhs)); }
	inline FFTdir&	operator &= (FFTdir &lhs, const FFTdir rhs) { lhs  = static_cast<FFTdir>(static_cast<int>(lhs) & static_cast<const int>(rhs)); return lhs; }
	inline FFTdir	operator |  (FFTdir  lhs, const FFTdir rhs) { return static_cast<FFTdir>(static_cast<int>(lhs) | static_cast<const int>(rhs)); }
	inline FFTdir&	operator |= (FFTdir &lhs, const FFTdir rhs) { lhs  = static_cast<FFTdir>(static_cast<int>(lhs) | static_cast<const int>(rhs)); return lhs; }

	typedef	enum	PropType_s {
		PROP_NONE	= 0,		// For parsing
		PROP_SPEC	= 1,		// Spectral flag
		PROP_LEAP	= 2,
		PROP_OMELYAN2	= 4,
		PROP_OMELYAN4	= 8,
		PROP_RKN4	= 16,
		PROP_MASK	= 30,		// So far... Masks the integrator type, removing the spectral flag
		PROP_SLEAP	= 3,
		PROP_SOMELYAN2	= 5,
		PROP_SOMELYAN4	= 9,
		PROP_SRKN4	= 17,
	}	PropType;

	inline PropType		operator &  (PropType  lhs, const PropType rhs) { return static_cast<PropType>(static_cast<int>(lhs) & static_cast<const int>(rhs)); }
	inline PropType&	operator &= (PropType &lhs, const PropType rhs) { lhs  = static_cast<PropType>(static_cast<int>(lhs) & static_cast<const int>(rhs)); return lhs; }
	inline PropType		operator |  (PropType  lhs, const PropType rhs) { return static_cast<PropType>(static_cast<int>(lhs) | static_cast<const int>(rhs)); }
	inline PropType&	operator |= (PropType &lhs, const PropType rhs) { lhs  = static_cast<PropType>(static_cast<int>(lhs) | static_cast<const int>(rhs)); return lhs; }


	typedef	enum	SpectrumType_s {
		SPECTRUM_K	= 1,
		SPECTRUM_G	= 2,
		SPECTRUM_V	= 4,
		SPECTRUM_GV	= 6,
		SPECTRUM_P	= 8,
	}	SpectrumType;

	inline SpectrumType	operator &  (SpectrumType  lhs, const FFTdir rhs) { return static_cast<SpectrumType>(static_cast<int>(lhs) & static_cast<const int>(rhs)); }
	inline SpectrumType&	operator &= (SpectrumType &lhs, const FFTdir rhs) { lhs  = static_cast<SpectrumType>(static_cast<int>(lhs) & static_cast<const int>(rhs)); return lhs; }
	inline SpectrumType	operator |  (SpectrumType  lhs, const FFTdir rhs) { return static_cast<SpectrumType>(static_cast<int>(lhs) | static_cast<const int>(rhs)); }
	inline SpectrumType&	operator |= (SpectrumType &lhs, const FFTdir rhs) { lhs  = static_cast<SpectrumType>(static_cast<int>(lhs) | static_cast<const int>(rhs)); return lhs; }

	typedef	enum	FindType_s {
		FIND_MAX,
		FIND_MIN,
	}	FindType;
#endif
