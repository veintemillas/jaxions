#ifndef	_SPROP_CPU_
	#define	_SPROP_CPU_

	#include "enum-field.h"
	void	propSpecKernelXeon(void * m_, void * __restrict__ v_, const void * __restrict__ m2_, double *z, const double dz, const double c, const double d, const double LL,
				   const double nQcd, const double fMom, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision, const VqcdType VQcd);
/*

	#include "scalar/scalarField.h"

	void	propSpecXeon	(Scalar *axionField, const double dz, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, const VqcdType VQcd);
	void	propSpecCpu	(Scalar *axionField, const double dz, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, const VqcdType VQcd);
*/
#endif
