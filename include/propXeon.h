#include"scalarField.h"

void	propagateXeon	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const uint Lx, const uint V, const uint S, FieldPrecision precision);
void	propagateCpu	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const uint Lx, const uint V, const uint S, FieldPrecision precision);
void	propLowMemXeon	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const uint Lx, const uint V, const uint S, FieldPrecision precision);
void	propLowMemCpu	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const uint Lx, const uint V, const uint S, FieldPrecision precision);
