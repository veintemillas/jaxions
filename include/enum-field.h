#ifndef	_ENUM_FIELD_
	#define _ENUM_FIELD_

	typedef	unsigned int uint;

	typedef enum	FieldType_s
	{
		FIELD_SAXION,
		FIELD_AXION,
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
	}	VqcdType;

	typedef enum	ConfType_s
	{
		CONF_KMAX,
		CONF_TKACHEV,
		CONF_SMOOTH,
		CONF_READ,
		CONF_NONE,
	}	ConfType;

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
#endif
