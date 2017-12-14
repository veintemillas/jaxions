#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "utils/utils.h"
#include "io/readWrite.h"
#include "spectrum/spectrum.h"
#include "energy/energy.h"

using namespace std;

int	main (int argc, char *argv[])
{
	initAxions(argc, argv);

	commSync();

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


	LogOut("\n-------------------------------------------------\n");
	LogOut("\n          GEN CONTRAST MAP FROM FILE             \n", sizeN);
	LogOut("\n-------------------------------------------------\n");

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
	double z_now = (*(axion->zV() ));

	LogOut("--------------------------------------------------\n");
	LogOut(" PARAMETERS READ in file index = %d               \n\n");

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
	LogOut("--------------------------------------------------\n\n");

	int index = fIndex;

	if ( (endredmap > 0) )
	{
	LogOut ("REDUCED map %d will be generated in out/m/redaxion.m.%05d\n\n", index, index);
		char mirraa[128] ;
		strcpy (mirraa, outName);
		strcpy (outName, "redaxion\0");
		//reduceEDens(index, endredmap, endredmap) ;
		//strcpy (outName, mirraa);
	}
	else
	{
		LogOut ("CONTRAST map %d will be generated in out/m/axion.m.%5d\n\n", index, index+1);
		index += 1 ;
	}

	LogOut ("Printing measurement file %05d ... ", index);
	createMeas(axion, index);
			SpecBin specAna(axion, (pType & PROP_SPEC) ? true : false);

			LogOut ("spec ");
			specAna.nRun();
			writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
			writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");
			writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");

			// computes energy and creates map
			LogOut ("en ");
			energy(axion, eRes, true, delta, nQcd, 0., VQCD_1, 0.);
			//bins density
			LogOut ("con ");
			//write binned distribution
			LogOut ("bin ");
			{
				double *eR = static_cast<double*>(eRes);
				if (axion->Precision() == FIELD_SINGLE) {
					float eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);
					Binner<10000,float> contBin(static_cast<float *>(axion->m2Cpu()), axion->Size(),
								    [eMean = eMean] (double x) -> double { return (double) (log10(x/eMean) );});
					contBin.run();
					writeBinner(contBin, "/bins", "contB");
				} else {
					double eMean = (eR[0] + eR[1] + eR[2] + eR[3] + eR[4]);
					Binner<10000,double>contBin(static_cast<double*>(axion->m2Cpu()), axion->Size(),
								    [eMean = eMean] (double x) -> double { return (double) (log10(x/eMean) );});
					contBin.run();
					writeBinner(contBin, "/bins", "contB");
				}
			}

			if ( (endredmap <= 0) || (endredmap >= sizeN) )
			{
				LogOut ("MAP ");
				writeEDens(axion);
			}
			LogOut ("tot ");
			writeEnergy(axion, eRes);

			//computes power spectrum
			LogOut ("pow ");
			specAna.pRun();
			writeArray(specAna.data(SPECTRUM_P), specAna.PowMax(), "/pSpectrum", "sP");

			if ( endredmap > 0)
			{
				LogOut("redmap ");
				int nena = sizeN/endredmap ;
				specAna.filter(nena);
				// this must be called immediately after pRun
				writeEDensReduced(axion, index, endredmap, endredmap/zGrid);
			}
			LogOut("Done! ");


		destroyMeas();
		LogOut("and closed file!\n\n ");



	endAxions();


	return 0;
}
