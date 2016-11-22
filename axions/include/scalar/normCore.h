#ifndef	_NORM_CORE_FIELD_
	#define	_NORM_CORE_FIELD_

	#include "scalar/scalarField.h"
	#include "enum-field.h"
	#include "utils/flopCounter.h"

	void	normCoreField	(Scalar *field, const double alpha, FlopCounter *fCount);
#endif
