#ifndef	_STRINGS_
	#define	_STRINGS_

	#include "scalar/scalarField.h"
	#include "utils/flopCounter.h"
	#include <vector>

	//StringData	strings	(Scalar *field, void *string, FlopCounter *fCount);
	StringData	strings	(Scalar *field, void *string);
	std::vector<std::vector<size_t>>  strToCoords     (char *strData, size_t Lx, size_t V);

	int	analyzeStrFolded	(Scalar *axion, const int index);
	int	analyzeStrFoldedNP	(Scalar *axion, const int index);
	int	analyzeStrUNFolded	(Scalar *axion, const int index);
#endif
