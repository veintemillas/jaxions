#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "propagator/allProp.h"
#include "energy/energy.h"
#include "utils/utils.h"
#include "io/readWrite.h"
#include "comms/comms.h"
#include "map/map.h"
#include "strings/strings.h"
#include "powerCpu.h"
#include "scalar/scalar.h"
#include "spectrum/spectrum.h"

#include "WKB/WKB.h"

using namespace std;
using namespace AxionWKB;

int	main (int argc, char *argv[])
{

	double zendWKB = 10. ;
	initAxions(argc, argv);

	if (nSteps==0)
	return 0 ;

	//--------------------------------------------------
	//       AUX STUFF
	//--------------------------------------------------

	void *eRes, *str;			// Para guardar la energia
	trackAlloc(&eRes, 128);
	memset(eRes, 0, 128);
	double *eR = static_cast<double *> (eRes);

	double  *binarray	 ;
	trackAlloc((void**) (&binarray),  10000*sizeof(size_t));
	double *bA = static_cast<double *> (binarray);
	size_t sliceprint = 0 ; // sizeN/2;



	commSync();
	LogOut("\n-------------------------------------------------\n");
	LogOut("\n           WKBonce EVOLUTION to %f               \n", zFinl);
	LogOut("\n-------------------------------------------------\n");

	LogOut("\n-------------------------------------------------\n");


	// Arrange to a list that can be read


	// LogOut("--------------------------------------------------\n");
	// LogOut("           PARSED CONDITIONS                     \n\n");
	//
	// LogOut("Length =  %2.2f\n", sizeL);
	// LogOut("nQCD   =  %2.2f\n", nQcd);
	// LogOut("N      =  %ld\n",   sizeN);
	// LogOut("Nz     =  %ld\n",   sizeZ);
	// LogOut("zGrid  =  %ld\n",   zGrid);
	// LogOut("z      =  %2.2f\n", zInit);
	// LogOut("zthres   =  %3.3f\n", zthres);
	// LogOut("zthres   =  %3.3f\n", zrestore);
	// LogOut("mass   =  %3.3f\n", axionmass(zInit, nQcd, zthres, zrestore));
	// LogOut("--------------------------------------------------\n");





	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	Scalar *axion;

	LogOut ("reading conf %d ...", fIndex);
	readConf(&axion, fIndex);
	if (axion == NULL)
	{
		LogOut ("Error reading HDF5 file\n");
		exit (0);
	}
	LogOut ("\n");

	// Axion spectrum
	const int kmax = axion->Length()/2 -1;
	int powmax = floor(1.733*kmax)+2 ;
	double delta = sizeL/sizeN;

	double z_now = (*axion->zV())	;
	LogOut("--------------------------------------------------\n");
	LogOut("           INITIAL CONDITIONS                     \n\n");

	LogOut("Length =  %2.2f\n", sizeL);
	LogOut("nQCD   =  %2.2f\n", nQcd);
	LogOut("N      =  %ld\n",   sizeN);
	LogOut("Nz     =  %ld\n",   sizeZ);
	LogOut("zGrid  =  %ld\n",   zGrid);
	LogOut("z      =  %2.2f\n", z_now);
	LogOut("zthr   =  %3.3f\n", zthres);
	LogOut("zres   =  %3.3f\n", zrestore);
	LogOut("mass   =  %3.3f\n\n", axionmass(z_now, nQcd, zthres, zrestore));
	if (axion->Precision() == FIELD_SINGLE)
	LogOut("precis = SINGLE(%d)\n",FIELD_SINGLE);
		else
	LogOut("precis = DOUBLE(%d)\n",FIELD_DOUBLE);
	LogOut("--------------------------------------------------\n");

	//--------------------------------------------------
	//       NEW AXION
	//--------------------------------------------------
	// if (axion->Field() == FIELD_SAXION)
	// {
	// 		LogOut ("Not ready for SAXION!");
	// 		return 0;
	// }

	//--------------------------------------------------
	//       WKB
	//--------------------------------------------------

	WKB wonka(axion, axion);

	int index = fIndex ;

	LogOut ("WKBing %d to %.4f ... ", index, zFinl);

	wonka(zFinl) 	;

	LogOut (" done!\n", zFinl);

	index++			;

	LogOut ("\n\n Dumping configuration %05d ...", index);
	writeConf(axion, index);
	LogOut ("Done!\n\n");


		LogOut ("Printing measurement file %05d ... ", index);
		createMeas(axion, index);
				SpecBin specAna(axion, (pType & PROP_SPEC) ? true : false);
				// computes energy and creates map
				LogOut ("en ");
				energy(axion, eRes, true, delta, nQcd, 0., VQCD_1, 0.);
				//bins density
				LogOut ("con ");
				axion->writeMAPTHETA( (*(axion->zV() )) , index, binarray, 10000)		;
				//write binned distribution
				LogOut ("bin ");
				writeArray(bA, 10000, "/bins", "cont");
				LogOut ("MAP ");
				writeEDens(axion, index);
				LogOut ("tot ");
				writeEnergy(axion, eRes);
				//computes power spectrum
				LogOut ("pow ");
				specAna.pRun();
				writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");
				LogOut ("spec ");
				specAna.nRun();
				writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
				writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");
				writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");
				LogOut ("2D ");
				writeMapHdf5s(axion,sliceprint);
				LogOut ("Done!\n");

			destroyMeas();



	endAxions();

	return 0;
}
