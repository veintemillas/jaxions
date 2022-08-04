#ifndef	_IO_HDF5_
	#define	_IO_HDF5_

	#include "scalar/scalarField.h"
	#include <hdf5.h>
	#include "utils/binner.h"

	// void	writeConf	(Scalar  *axion, int index);
	// void	readConf	(Cosmos *myCosmos, Scalar **axion, int index);
	void	writeConf (Scalar *axion, int index, const bool restart=false);
	void	readConf	(Cosmos *myCosmos, Scalar **axion, int index, const bool restart=false);
	double	readEDens	(Cosmos *myCosmos, Scalar **axion, int index);

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
	void	writePMapHdf5s	(Scalar *axion, char *eCh);
	void	writeGadget	(Scalar *axion, double eMean, size_t realN=0, size_t nParts=0, double sigma = 1.0);

	void	reduceEDens	(int index, uint newLx, uint newLz);

	void	writePoint	(Scalar *axion);
	void    writeSpectrum 	(Scalar *axion, void *spectrumK, void *spectrumG, void *spectrumV, size_t powMax, bool power);
	void    writeArray	(double *array, size_t aSize, const char *group, const char *dataName, int rank = 0);
	void    writeAttribute	(double *data, const char *name);
	void    writeAttribute	(void *data, const char *name, hid_t h5_Type);
	void    writeAttributeg	(void *data, const char *group, const char *name, hid_t h5_Type);
	herr_t	writeAttribute  (hid_t file_id, void *data, const char *name, hid_t h5_type);

	void	writeBinnerMetadata (double max, double min, size_t N, const char *group);

	void	writeGadget	(Scalar *axion);
	void	writeConfNyx (Scalar *axion, int index);

	template<const size_t N, typename cFloat>
	void	writeBinner	(Binner<N,cFloat> bins, const char *group, const char *dataName) {

	LogMsg(VERB_PARANOID,"[wB] Writting binner");LogFlush();

		writeArray (bins.data(), N, group, dataName);

		double max = bins.max();
		double min = bins.min();

		std::string	baseName(group);

		baseName += std::string("/") + std::string(dataName);

		writeBinnerMetadata (max, min, N, baseName.c_str());
	}
#endif
