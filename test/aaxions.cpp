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

using namespace std;

#ifdef	USE_XEON
	__declspec(target(mic)) char *mX, *vX, *m2X;
#endif


int	main (int argc, char *argv[])
{
	initAxions(argc, argv);

	LogOut("\n-------------------------------------------------\n");
	LogOut("\n          CREATING MINICLUSTERS!                \n\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	Scalar *axion;
	char fileName[256];

	if ((fIndex == -1) && (cType == CONF_NONE)) {
		LogOut("Error: Neither initial conditions nor configuration to be loaded selected. Empty field.\n");
	} else {
		if (fIndex == -1)
			//This generates initial conditions
			axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fType, cType, parm1, parm2);
		else
		{
			//This reads from an Axion.$fIndex file
			readConf(&axion, fIndex);
			if (axion == nullptr)
			{
				LogOut ("Error reading HDF5 file\n");
				exit (0);
			}
		}
	}

	//--------------------------------------------------
	//          SETTING BASE PARAMETERS
	//--------------------------------------------------

	double delta = sizeL/sizeN;
	double dz;

	if (nSteps == 0)
		dz = 0.;
	else
		dz = (zFinl - zInit)/((double) nSteps);

	LogOut("--------------------------------------------------\n");
	LogOut("           PARAMETERS                             \n\n");

	LogOut("Length =  %2.1f\n", sizeL);
	LogOut("N      =  %ld\n",   sizeN);
	LogOut("Nz     =  %ld\n",   sizeZ);
	LogOut("zGrid  =  %ld\n",   zGrid);
	LogOut("dx     =  %2.5f\n", delta);
	LogOut("LL     =  %2.1f\n", LL);
	LogOut("t1     =  %2.2f\n", zInit);
	LogOut("wDz    =  %2.2f\n", wDz);
	LogOut("---------------\n", wDz);
	LogOut("msa_I  =  %2.2f\n", sqrt(2.*LL)*zInit*delta);
	LogOut("msa_3  =  %2.2f\n", sqrt(2.*LL)*3.0*delta);
	LogOut("--------------------------------------------------\n");

	const size_t S0 = sizeN*sizeN;
	const size_t SF = sizeN*sizeN*(sizeZ+1)-1;
	const size_t V0 = 0;
	const size_t VF = axion->Size()-1;

	//--------------------------------------------------
	//   FILES
	//--------------------------------------------------

	FILE *file_sample ;
	file_sample = NULL;
	if (commRank() == 0)
	{
		file_sample = fopen("out/sample.txt","w+");
	}
	LogOut("Files prepared! \n");

	size_t nstrings_global = 0 ;
	double maximumtheta = 3.141597;
	double saskia;
	StringData rts ;

	//--------------------------------------------------
	//   THE TIME ITERATION LOOP
	//--------------------------------------------------

	LogOut("--------------------------------------------------\n");
	LogOut("           STARTING COMPUTATION                   \n");
	LogOut("--------------------------------------------------\n");

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	int counter = 0;
	int index = 0;

	commSync();

	void *eRes, *str;			// Para guardar la energia
	trackAlloc(&eRes, 128);
	memset(eRes, 0, 128);
#ifdef	__MIC__
	alignAlloc(&str, 64, (axion->Size()));
#elif defined(__AVX__)
	alignAlloc(&str, 32, (axion->Size()));
#else
	alignAlloc(&str, 16, (axion->Size()));
#endif
	memset(str, 0, axion->Size());

	commSync();

	if (fIndex == -1)
	{
		LogOut ("Dumping configuration %05d ...", index);
		writeConf(axion, index);
		LogOut ("Done!\n");
	}
	else
		index = fIndex;

	axion->SetLambda(LAMBDA_FIXED);

	if (LAMBDA_FIXED == axion->Lambda())
		LogOut ("Lambda in FIXED mode\n");
	else
		LogOut ("Lambda in Z2 mode\n");

	Folder munge(axion);

	LogOut ("Folding configuration\n");
	munge(FOLD_ALL);

	if (cDev != DEV_CPU) {
		LogOut ("Transferring configuration to device\n");
		axion->transferDev(FIELD_MV);
	}

	if (dump > nSteps)
		dump = nSteps;

	int nLoops;

	if (dump == 0)
		nLoops = 0;
	else
		nLoops = (int)(nSteps/dump);

		double masi ;
		double mfreA ;
		double mfreS ;
		double mfre ;
		double dzaux ;
		double z_now ;

	LogOut ("Start redshift loop\n");

	commSync();

	start = std::chrono::high_resolution_clock::now();
	old = start;

	initPropagator (pType, axion, nQcd, delta, LL, VQCD_1);

	for (int zloop = 0; zloop < nLoops; zloop++)
	{

		//--------------------------------------------------
		// THE TIME ITERATION SUB-LOOP
		//--------------------------------------------------

		index++;

		for (int zsubloop = 0; zsubloop < dump; zsubloop++)
        {
      	z_now = (*axion->zV());

				size_t idxprint = 0 ;

				if (commRank() == 0)
					{
						if (axion->Field() == FIELD_SAXION)
						{
							saskia = saxionshift(z_now, nQcd, zthres, zrestore, LL);

							if (sPrec == FIELD_DOUBLE) {
								fprintf(file_sample,"%f %f %f %f %f %f %f %ld %f %e\n",
								z_now,
								axionmass(z_now,nQcd,zthres, zrestore),
								LL,
								static_cast<complex<double> *> (axion->mCpu())[idxprint + S0].real(),
								static_cast<complex<double> *> (axion->mCpu())[idxprint + S0].imag(),
								static_cast<complex<double> *> (axion->vCpu())[idxprint].real(),
								static_cast<complex<double> *> (axion->vCpu())[idxprint].imag(),
								nstrings_global,
								maximumtheta,
								saskia );
							} else {
								fprintf(file_sample,"%f %f %f %f %f %f %f %ld %f %e\n",
								z_now,
								axionmass(z_now,nQcd,zthres, zrestore),
								LL,
								static_cast<complex<float>  *> (axion->mCpu())[idxprint + S0].real(),
								static_cast<complex<float>  *> (axion->mCpu())[idxprint + S0].imag(),
								static_cast<complex<float>  *> (axion->vCpu())[idxprint].real(),
								static_cast<complex<float>  *> (axion->vCpu())[idxprint].imag(),
								nstrings_global, maximumtheta, saskia);
							}
						}
						else
						{
							if (sPrec == FIELD_DOUBLE) {
								fprintf(file_sample,"%f %f %f %f %f\n",
								z_now,
								axionmass(z_now,nQcd,zthres, zrestore),
								static_cast<double*> (axion->mCpu())[idxprint + S0],
								static_cast<double*> (axion->vCpu())[idxprint],
								maximumtheta);
							} else {
								fprintf(file_sample,"%f %f %f %f %f\n",
								z_now,
								axionmass(z_now,nQcd,zthres, zrestore),
								static_cast<float*> (axion->mCpu())[idxprint + S0],
								static_cast<float*> (axion->vCpu())[idxprint],
								maximumtheta);
								// fprintf(file_sample,"%f %f ",static_cast<float*> (axion->mCpu())[S0+1], static_cast<float*> (axion->vCpu())[S0+1]);
								// fprintf(file_sample,"%f %f\n", static_cast<float*> (axion->mCpu())[S0+2], static_cast<float*> (axion->vCpu())[S0+2]);
							}
						}
						fflush(file_sample);
					}

	      //--------------------------------------------------
				// DYAMICAL deltaz
				//--------------------------------------------------
	  		//Set dz to gradients and axion mass or scalar mass
				 masi = z_now*axionmass(z_now,nQcd,zthres, zrestore);
				 mfreA = sqrt(masi*masi + 12./(delta*delta));
				 masi = sqrt(2.*LL)*z_now ;
				 mfreS = sqrt(masi*masi + 12./(delta*delta));
				 mfre = min(mfreA,mfreS);
				 dzaux = wDz/mfre ;

				 propagate (axion, dz);
        }
		auto strDen = strings(axion, str);
		nstrings_global = rts.strDen ;
		maximumtheta = axion->maxtheta();

		energy(axion, eRes, false, delta, nQcd, LL);

		profiler::Profiler &prof = profiler::getProfiler(PROF_PROP);

		auto pFler = prof.Prof().cbegin();
		auto pName = pFler->first;

		profiler::printMiniStats(*static_cast<double*>(axion->zV()), strDen, PROF_PROP, pName);

		createMeas(axion, index);
		//writeEDens(axion, index);
		writeString(str, strDen);
		writeEnergy(axion, eRes);
		writeMapHdf5(axion);
		writePoint(axion);
		destroyMeas();

	} // zloop

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	LogOut("\n PROGRAMM FINISHED\n");

	munge(UNFOLD_ALL);
	writeConf(axion, index);

	LogOut("z_final = %f\n", *axion->zV());
	LogOut("#_steps = %i\n", counter);
	LogOut("#_prints = %i\n", index);
	LogOut("Total time: %2.3f s\n", elapsed.count()*1.e-3);

	trackFree(&eRes, ALLOC_TRACK);
	trackFree(&str,  ALLOC_ALIGN);

	delete axion;

	endAxions();

	return 0;
}
