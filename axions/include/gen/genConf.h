#ifndef	_GEN_CONF_
	#define	_GEN_CONF_

	#include "scalar/scalarField.h"
	#include "utils/flopCounter.h"

	void	genConf	(Scalar *field, ConfType cType, size_t parm1, double parm2, FlopCounter *fCount);
#endif
