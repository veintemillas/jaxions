#ifndef	_SCALE_FIELD_
	#define	_SCALE_FIELD_

	#include "scalar/scalarField.h"
	#include "enum-field.h"
	#include "utils/flopCounter.h"

	void	scaleField	(Scalar *field, const FieldIndex fIdx, const double factor, FlopCounter *fCount);
#endif
