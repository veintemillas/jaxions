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
	LogOut("\n           WKB EVOLUTION to %f                   \n", zendWKB);
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
	if (axion->Field()==FIELD_SAXION)
	{
			LogOut ("Not ready for SAXION!");
			return 0;
	}

	LogOut ("creating new axion2 ... FIELD_TYPE(%d) ", FIELD_WKB );
// the new axion is always prepared in lowmem
	Scalar *axion2;
	// force lowmem in scalar mode -> to be done!
	axion2 = new Scalar (sizeN, sizeZ, sPrec, cDev, z_now, true, zGrid, FIELD_WKB, LAMBDA_FIXED, CONF_NONE, 0. , 0. );
	LogOut ("done !\n");

	//--------------------------------------------------
	//       WKB
	//--------------------------------------------------

	
	WKB wonka(axion, axion2);

	// float *mm = static_cast<float *> (axion->mCpu())+axion->Surf();
	// float *vv = static_cast<float *> (axion->vCpu());
	// float *aa = static_cast<float *> (axion->m2Cpu())+axion->Surf();
	//
	// float *mm2 = static_cast<float *> (axion2->mCpu());
	// float *vv2 = static_cast<float *> (axion2->vCpu());
	// float *aa2 = static_cast<float *> (axion2->m2Cpu());
	//
	// for (int i = 1; i < 10; i++)
	// LogOut("%f %f %f vs %f %f %f\n", mm[i], vv[i], aa[i], mm2[i], vv2[i], aa2[i]);
	//
	// LogOut("\n");
	//
	// for (int i = axion->Size()-1; i > axion->Size()-10; i--)
	// LogOut("%f %f %f vs %f %f %f\n", mm[i], vv[i], aa[i], mm2[i], vv2[i], aa2[i]);


	int index = fIndex ;

	if (zFinl < z_now)
		zFinl = z_now	;

	LogOut("from z1=%f to z2=%f in %d time steps\n\n",z_now,zFinl,nSteps);

	for (int i = 0; i < nSteps; i++)
	{

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
								LogOut ("tot ");
								writeEnergy(axion, eRes);
								//computes power spectrum
								LogOut ("pow ");
								specAna.pRun();
								writeArray(specAna.data(SPECTRUM_P), powmax, "/pSpectrum", "sP");
								LogOut ("spec ");
								specAna.nRun();
								writeArray(specAna.data(SPECTRUM_K), specAna.PowMax(), "/nSpectrum", "sK");
								writeArray(specAna.data(SPECTRUM_G), specAna.PowMax(), "/nSpectrum", "sG");
								writeArray(specAna.data(SPECTRUM_V), specAna.PowMax(), "/nSpectrum", "sV");
								LogOut ("2D ");
								writeMapHdf5s(axion,sliceprint);
								LogOut ("Done!\n");
		destroyMeas();

		double zco = z_now + i*(zFinl-z_now)/nSteps	;
		wonka(zco) 	;
		index++			;
	}
	//--------------------------------------------------
	//       SAVE DATA
	//--------------------------------------------------


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
				writeArray(specAna.data(SPECTRUM_P), powmax, "/pSpectrum", "sP");
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
