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

	typedef	enum	StringType_s
	{
		STRING_XY_POSITIVE = 1,
		STRING_YZ_POSITIVE = 2,
		STRING_ZX_POSITIVE = 4,
		STRING_XY_NEGATIVE = 8,
		STRING_YZ_NEGATIVE = 16,
		STRING_ZX_NEGATIVE = 32,
	}	StringType;

	typedef	enum	LambdaType_s
	{
		LAMBDA_FIXED,
		LAMBDA_Z2,
	}	LambdaType;

	typedef enum	ConfType_s
	{
		CONF_KMAX,
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
#endif
