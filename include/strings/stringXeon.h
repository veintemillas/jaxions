#ifndef	_STRINGS_CPU_
	#define	_STRINGS_CPU_

	#include"scalar/scalarField.h"

	StringData	stringXeon	(Scalar *axionField, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, void *string);
	StringData	stringCpu	(Scalar *axionField, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, void *string);
#endif
