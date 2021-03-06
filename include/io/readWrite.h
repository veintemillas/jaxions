#ifndef	_IO_HDF5_
	#define	_IO_HDF5_

	#include "scalar/scalarField.h"

	// void	writeConf	(Scalar  *axion, int index);
	// void	readConf	(Cosmos *myCosmos, Scalar **axion, int index);
	void	writeConf (Scalar *axion, int index, const bool restart=false);
	void	readConf	(Cosmos *myCosmos, Scalar **axion, int index, const bool restart=false);

	void	createMeas	(Scalar *axion, int index);
	void	destroyMeas	();

	void	writeString	(Scalar *axion, StringData strDat, const bool rData=true);
	void	writeStringCo	(Scalar *axion, StringData strDat, const bool rData=true);
	void	writeStringEnergy	(Scalar *axion, StringEnergyData strEDat);
	void	writeEnergy	(Scalar *axion, void *eData, double rmask=-1.0);
	void	writeEDens	(Scalar *axion, MapType fMap=MAP_THETA);
	void	writeDensity	(Scalar *axion, MapType fMap, double eMax, double eMin);
	void	writeEDensReduced	(Scalar *axion, int index, int newNx, int newNz);

	void	writeMapHdf5	(Scalar *axion);
	void	writeMapHdf5s	(Scalar *axion, int slicenumbertoprint);
	void	writeMapHdf5s2	(Scalar *axion, int slicenumbertoprint);
	void	writeEMapHdf5	(Scalar *axion);
	void	writeEMapHdf5s	(Scalar *axion, int slicenumbertoprint=0, char *eCh="/map/E");
	void	writePMapHdf5	(Scalar *axion);

	void	reduceEDens	(int index, uint newLx, uint newLz);

	void	writePoint	(Scalar *axion);
	void    writeSpectrum 	(Scalar *axion, void *spectrumK, void *spectrumG, void *spectrumV, size_t powMax, bool power);
	void    writeArray	(double *array, size_t aSize, const char *group, const char *dataName);



	void	writeBinnerMetadata (double max, double min, size_t N, const char *group);

	template<const size_t N, typename cFloat>
	void	writeBinner	(Binner<N,cFloat> bins, const char *group, const char *dataName) {

		writeArray (bins.data(), N, group, dataName);

		double max = bins.max();
		double min = bins.min();

		std::string	baseName(group);

		baseName += std::string("/") + std::string(dataName);

		writeBinnerMetadata (max, min, N, baseName.c_str());
	}
#endif
