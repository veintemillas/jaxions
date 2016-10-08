#include"scalarField.h"

void	propagate	(Scalar *field, const double dz, const double LL, const double nQcd, const double delta, bool gpu);
void	propLowMem	(Scalar *field, const double dz, const double LL, const double nQcd, const double delta, bool gpu);
