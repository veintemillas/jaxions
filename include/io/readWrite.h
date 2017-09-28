#ifndef	_IO_HDF5_
	#define	_IO_HDF5_

	#include "scalar/scalarField.h"

	void	writeConf	(Scalar  *axion, int index);
	void	readConf	(Scalar **axion, int index);

	void	createMeas	(Scalar *axion, int index);
	void	destroyMeas	();

	void	writeString	(void *strData, StringData strDat, const bool rData=true);
	void	writeEnergy	(Scalar *axion, void *eData);
	void	writeEDens	(Scalar *axion, int index);

	void	writeMapHdf5	(Scalar *axion);
	void	writeMapHdf5s	(Scalar *axion, int slicenumbertoprint);

	void	reduceEDens	(int index, uint newLx, uint newLz);

	void	writePoint	(Scalar *axion);
	void    writeSpectrum 	(Scalar *axion, void *spectrumK, void *spectrumG, void *spectrumV, size_t powMax, bool power);
	void    writeArray	(double *array, size_t aSize, const char *group, const char *dataName);



	void	writeBinnerMetadata (double max, double min, size_t N, const char *group);

	template<typename cFloat, const size_t N>
	void	writeBinner	(Binner<cFloat,N> bins, const char *group, const char *dataName) {

		if (sizeof(cFloat) == 4) {
			std::array<double,N> data;

			#pragma omp parallel for schedule(static)	// Data conversion, I'm not accepting complex datatypes
			for (size_t i=0; i<N; i++) {
				data[i] = bins[i];
			}
			writeArray (data.data(), N, group, dataName);
		} else {
			writeArray (bins.data(), N, group, dataName);
		}

		double max = bins.max();
		double min = bins.min();

		std::string	baseName(group);

		baseName += std::string("/") + std::string(dataName);

		writeBinnerMetadata (max, min, N, baseName.c_str());
	}
#endif
