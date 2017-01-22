#ifndef	_MAP_SCALAR_
	#define	_MAP_SCALAR_

	#include "scalar/scalarField.h"

	void	writeMap	(Scalar *axion, const int index);
//	void	writeDensityMap3D	(Scalar *axion, const int index)

	void	writeMapAt	 (Scalar *axion, const int index);
	void	writeMapRho	 (Scalar *axion, const int index);
	void	writeMapDens (Scalar *axion, const int index);

#endif
