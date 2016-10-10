#include "scalarField.h"
#include "flopCounter.h"

double	propagate	(Scalar *field, const double dz, const double LL, const double nQcd, const double delta, DeviceType dev, FlopCounter *fCount);
