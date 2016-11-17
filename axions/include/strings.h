#include "scalarField.h"
#include "flopCounter.h"

double	strings	(Scalar *field, DeviceType dev, void *string, FlopCounter *fCount);

void	analyzeStrFolded	(Scalar *axion, const int index);

int	analyzeStrUNFolded	(Scalar *axion, const int index);
