#include"scalarField.h"

void	propagateXeon	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const int Lx, const int V, const int S, FieldPrecision precision);
void	propagateCpu	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const int Lx, const int V, const int S, FieldPrecision precision);
void	propLowMemXeon	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const int Lx, const int V, const int S, FieldPrecision precision);
void	propLowMemCpu	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const int Lx, const int V, const int S, FieldPrecision precision);
