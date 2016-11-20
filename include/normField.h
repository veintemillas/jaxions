#ifndef	NORM_FIELD
	#define	NORM_FIELD

	#include "scalarField.h"
	#include "enum-field.h"
	#include "flopCounter.h"

	void	normaliseField	(Scalar *field, const FieldIndex fIdx, FlopCounter *fCount);
#endif
