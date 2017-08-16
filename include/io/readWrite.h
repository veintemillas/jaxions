#ifndef	_IO_HDF5_
	#define	_IO_HDF5_

	#include "scalar/scalarField.h"

	void	writeConf	(Scalar  *axion, int index);
	void	readConf	(Scalar **axion, int index);

	void	createMeas	(Scalar *axion, int index);
	void	destroyMeas	();

	void	writeString	(void *strData, StringData strDat);
	void	writeEnergy	(Scalar *axion, void *eData);
	void	writeEDens	(Scalar *axion, int index);
	void	writeMapHdf5	(Scalar *axion);

	void	reduceEDens	(int index, uint newLx, uint newLz);

	void	writePoint	(Scalar *axion);
	void    writeSpectrum 	(Scalar *axion, void *spectrumK, void *spectrumG, void *spectrumV, size_t powMax, bool power);
	void    writeArray	(double *array, size_t aSize, const char *group, const char *dataName);
#endif
