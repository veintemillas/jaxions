#ifndef	_STRINGS_
	#define	_STRINGS_

	#include "scalar/scalarField.h"
	#include "utils/flopCounter.h"

	double	strings	(Scalar *field, DeviceType dev, void *string, FlopCounter *fCount);

	int	analyzeStrFolded	(Scalar *axion, const int index);

	int	analyzeStrUNFolded	(Scalar *axion, const int index);
#endif
