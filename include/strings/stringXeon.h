#ifndef	_STRINGS_CPU_
	#define	_STRINGS_CPU_

	#include"scalar/scalarField.h"

	size_t	stringXeon	(Scalar *axionField, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, void *string);
	size_t	stringCpu	(Scalar *axionField, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, void *string);
#endif
