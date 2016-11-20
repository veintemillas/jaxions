#ifndef	SCALE_FIELD
	#define	SCALE_FIELD

	#include "scalarField.h"
	#include "enum-field.h"
	#include "flopCounter.h"

	void	scaleField	(Scalar *field, const FieldIndex fIdx, const double factor, FlopCounter *fCount);
#endif
