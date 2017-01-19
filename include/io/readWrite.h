#ifndef	_IO_HDF5_
	#define	_IO_HDF5_

	#include "scalar/scalarField.h"

	void	writeConf	(Scalar  *axion, int index);
	void	readConf	(Scalar **axion, int index);

	void	createMeas	(Scalar *axion, int index);
	void	destroyMeas	();

	void	writeString	(void *strData, double strDen);

#endif
