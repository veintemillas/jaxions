#ifndef	_NORM_FIELD_
	#define	_NORM_FIELD_

	#include "scalar/scalarField.h"
	#include "enum-field.h"
	#include "utils/flopCounter.h"

	void	normaliseField	(Scalar *field, const FieldIndex fIdx, FlopCounter *fCount);
#endif
