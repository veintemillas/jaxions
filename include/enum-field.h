#ifndef	_ENUM_FIELD_
	#define _ENUM_FIELD_
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
#endif
