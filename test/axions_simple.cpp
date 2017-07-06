#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "scalar/scalar.h"
#include "propagator/allProp.h"
#include "energy/energy.h"
#include "enum-field.h"
#include "utils/utils.h"
#include "comms/comms.h"
#include "io/readWrite.h"
#include "map/map.h"
#include "strings/strings.h"
#include "powerCpu.h"

using namespace std;

#ifdef	USE_XEON
	__declspec(target(mic)) char *mX, *vX, *m2X;
#endif

#define printMpi(...) do {		\
	if (!commRank()) {		\
	  printf(__VA_ARGS__);  	\
	  fflush(stdout); }		\
}	while (0)


int	main (int argc, char *argv[])
{
	parseArgs(argc, argv);

	if (initComms(argc, argv, zGrid, cDev, verb) == -1)
	{
		printf ("Error initializing devices and Mpi\n");
		return 1;
	}

	printMpi("\n-------------------------------------------------\n");
	printMpi("\n          CREATING MINICLUSTERS!                \n\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	FlopCounter *fCount = new FlopCounter;

	Scalar *axion;
	char fileName[256];

	if ((initFile == NULL) && (fIndex == -1) && (cType == CONF_NONE))
		printMpi("Error: Neither initial conditions nor configuration to be loaded selected. Empty field.\n");
	else
	{
		if (fIndex == -1)
			//This generates initial conditions
			axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fType, cType, parm1, parm2, fCount);
		else
		{
			//This reads from an Axion.00000 file
			readConf(&axion, fIndex);
			if (axion == NULL)
			{
				printMpi ("Error reading HDF5 file\n");
				exit (0);
			}
		}
	}

	//--------------------------------------------------
	//          OUTPUTS FOR CHECKING
	//--------------------------------------------------

	FILE *file_sample ;
	file_sample = NULL;

	FILE *file_energy ;
	file_energy = NULL;

	FILE *file_energy2 ;
	file_energy2 = NULL;

	FILE *file_spectrum ;
	file_spectrum = NULL;


	if (commRank() == 0)
	{
		file_sample = fopen("out/s_sample.txt","w+");
		//fprintf(file_sample,"%f %f %f\n",z, creal(m[0]), cimag(m[0]));

		file_energy = fopen("out/s_energy.txt","w+");
		//fprintf(file_sample,"%f %f %f\n",z, creal(m[0]), cimag(m[0]));

		file_energy2 = fopen("out/energy2.txt","w+");

		file_spectrum = fopen("out/spectrum.txt","w+");
	}

	// Energy 2
	double Vr, Vt, Kr, Kt, Grz, Gtz;
	// Strings
	int nstrings = 0 ;

	// Axion spectrum
	const int kmax = axion->Length()/2 -1;
	int powmax = floor(1.733*kmax)+2 ;

	double  *spectrumK ;
	double  *spectrumG ;
	double  *spectrumV ;
	trackAlloc((void**) (&spectrumK), 8*powmax);
	trackAlloc((void**) (&spectrumG), 8*powmax);
	trackAlloc((void**) (&spectrumV), 8*powmax);


	//--------------------------------------------------
	//          SETTING BASE PARAMETERS
	//--------------------------------------------------

	double delta = sizeL/sizeN;
	double dz;

	if (nSteps == 0)
		dz = 0.;
	else
		dz = (zFinl - zInit)/((double) nSteps);

	printMpi("--------------------------------------------------\n");
	printMpi("           INITIAL CONDITIONS                     \n\n");

	printMpi("Length =  %2.5f\n", sizeL);
	printMpi("N      =  %ld\n",   sizeN);
	printMpi("Nz     =  %ld\n",   sizeZ);
	printMpi("zGrid  =  %ld\n",   zGrid);
	printMpi("dx     =  %2.5f\n", delta);
	printMpi("dz     =  %2.5f\n", dz);
	printMpi("LL     =  %2.5f\n", LL);
	printMpi("Ng     =  %d  \n", Ng);
	printMpi("--------------------------------------------------\n");

	const size_t S0 = sizeN*sizeN;
	const size_t SF = sizeN*sizeN*(sizeZ+1)-1;
	const size_t V0 = 0;
	const size_t VF = axion->Size()-1;

	//--------------------------------------------------
	//   THE TIME ITERATION LOOP
	//--------------------------------------------------

	printMpi("--------------------------------------------------\n");
	printMpi("           STARTING COMPUTATION                   \n");
	printMpi("--------------------------------------------------\n");

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	int counter = 0;
	int index = 0;

	commSync();

	void *eRes, *str;			// Para guardar la energia
	trackAlloc(&eRes, 128);
	memset(eRes, 0, 128);
#ifdef	__MIC__
	alignAlloc(&str, 64, (axion->Size()/2));
#elif defined(__AVX__)
	alignAlloc(&str, 32, (axion->Size()/2));
#else
	alignAlloc(&str, 16, (axion->Size()/2));
#endif
	memset(str, 0, axion->Size()/2);

	commSync();

	if (fIndex == -1)
	{
		printMpi ("Dumping configuration %05d ...", index);
		writeConf(axion, index);
		printMpi ("Done!\n");
		fflush (stdout);
	}
	else
		index = fIndex + 1;

	if (cDev != DEV_CPU)
	{
		printMpi ("Transferring configuration to device\n");
		axion->transferDev(FIELD_MV);
	}

	if (dump > nSteps)
		dump = nSteps;

	int nLoops;

	if (dump == 0)
		nLoops = 0;
	else
		nLoops = (int)(nSteps/dump);

		if (cDev != DEV_GPU)
		{
			//printMpi("Strings...");
			//analyzeStrFolded(axion, index);
			//analyzeStrUNFolded(axion, index);
			//printMpi(" Done!");
			memcpy   (axion->mCpu(), static_cast<char *> (axion->mCpu()) + S0*sizeZ*axion->DataSize(), S0*axion->DataSize());
			writeMap (axion, index);
			//energy(axion, LL, nQcd, delta, cDev, eRes, fCount);
			axion->writeENERGY ((*(axion->zV() )),file_energy, Grz, Gtz, Vr, Vt, Kr, Kt);
			fprintf(file_energy2,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf %d\n", (*axion->zV()), Vr, Vt, Kr, Kt, Grz, Gtz, nstrings);
			printf("%d/%d | z = %lf | st = %d | Vr %+lf Vt %+lf Kr %+lf Kt %+lf Grz %+lf Gtz %+lf\n", index, nLoops, (*axion->zV()), nstrings, Vr, Vt, Kr, Kt, Grz, Gtz);
		}

	printMpi ("Start redshift loop\n\n");
	fflush (stdout);

	commSync();

	start = std::chrono::high_resolution_clock::now();
	old = start;

	for (int zloop = 0; zloop < nLoops; zloop++)
	{
		//--------------------------------------------------
		// THE TIME ITERATION SUB-LOOP
		//--------------------------------------------------

		index++;

		for (int zsubloop = 0; zsubloop < dump; zsubloop++)
		{
			if (commRank() == 0) {
				if (sPrec == FIELD_DOUBLE) {
					fprintf(file_sample,"%f %f %f %f %f\n",(*(axion->zV() )), static_cast<complex<double> *> (axion->mCpu())[S0].real(), static_cast<complex<double> *> (axion->mCpu())[S0].imag(),
						static_cast<complex<double> *> (axion->vCpu())[V0].real(), static_cast<complex<double> *> (axion->vCpu())[V0].imag());
				} else {
					fprintf(file_sample,"%f %f %f %f %f\n",(*(axion->zV() )), static_cast<complex<float>  *> (axion->mCpu())[S0].real(), static_cast<complex<float>  *> (axion->mCpu())[S0].imag(),
						static_cast<complex<float>  *> (axion->vCpu())[V0].real(), static_cast<complex<float>  *> (axion->vCpu())[V0].imag());
				}
			}

			old = std::chrono::high_resolution_clock::now();
//			propagate (axion, dz, LL, nQcd, delta, cDev, fCount);
			propagateSimple (axion, dz, LL, nQcd, delta, Ng);

			current = std::chrono::high_resolution_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);

			fCount->addTime(elapsed.count()*1.e-3);

			counter++;
		} // zsubloop

		fflush (stdout);
		axion->transferCpu(FIELD_MV);

/*	TODO

	2. Fix writeMap so it reads data from the first slice of m
*/

		if (cDev != DEV_GPU)
		{
			//double Grz, Gtz, Vr, Vt, Kr, Kt;
//			writeConf(axion, index);
			//if (axion->Precision() == FIELD_DOUBLE)
			if ((*axion->zV()) > 0.4 )
			{
				//printMpi("Strings (if %f>0.4) ... ", (*axion->zV()));
				fflush (stdout);
				//analyzeStrFolded(axion, index);
				nstrings = analyzeStrUNFolded(axion, index);
			}

			memcpy   (axion->mCpu(), static_cast<char *> (axion->mCpu()) + S0*sizeZ*axion->DataSize(), S0*axion->DataSize());
			writeMap (axion, index);
			//energy(axion, LL, nQcd, delta, cDev, eRes, fCount);
			axion->writeENERGY ((*(axion->zV() )),file_energy, Grz, Gtz, Vr, Vt, Kr, Kt);
			fprintf(file_energy2,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf %d\n", (*axion->zV()), Vr, Vt, Kr, Kt, Grz, Gtz, nstrings);
			printf("%d/%d | z = %lf | st = %d | Vr %+lf Vt %+lf Kr %+lf Kt %+lf Grz %+lf Gtz %+lf\n", index, nLoops, (*axion->zV()), nstrings, Vr, Vt, Kr, Kt, Grz, Gtz);

			if ((*axion->zV()) > 0.4 )
			{
				double *sK = static_cast<double *> (spectrumK);
				double *sG = static_cast<double *> (spectrumG);
				double *sV = static_cast<double *> (spectrumV);
				spectrumUNFOLDED(axion);
				//spectrumUNFOLDED(axion, spectrumK, spectrumG, spectrumV);
				printf("sp %f %f %f ...\n", (float) sK[0]+sG[0]+sV[0], (float) sK[1]+sG[1]+sV[1], (float) sK[2]+sG[2]+sV[2]);
				fprintf(file_spectrum,  "%f ", (*axion->zV()));
				for(int i = 0; i<powmax; i++) {	fprintf(file_spectrum, "%f ", (float) sK[i]);} fprintf(file_spectrum, "\n");
				fprintf(file_spectrum,  "%f ", (*axion->zV()));
				for(int i = 0; i<powmax; i++) {	fprintf(file_spectrum, "%f ", (float) sG[i]);} fprintf(file_spectrum, "\n");
				fprintf(file_spectrum,  "%f ", (*axion->zV()));
				for(int i = 0; i<powmax; i++) {	fprintf(file_spectrum, "%f ", (float) sV[i]);} fprintf(file_spectrum, "\n");
			}


		}
	} // zloop

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	printMpi("\n PROGRAMM FINISHED\n");

	if (nSteps > 0)
		writeConf(axion, index);

	printMpi("z_final = %f\n", *axion->zV());
	printMpi("#_steps = %i\n", counter);
	printMpi("#_prints = %i\n", index);
	printMpi("Total time: %2.3f s\n", elapsed.count()*1.e-3);
	printMpi("GFlops: %.3f\n", fCount->GFlops());
	printMpi("GBytes: %.3f\n", fCount->GBytes());
	printMpi("--------------------------------------------------\n");

	trackFree(&eRes, ALLOC_TRACK);
	trackFree(&str,  ALLOC_ALIGN);
	trackFree((void**) (&spectrumK),  ALLOC_TRACK);
	trackFree((void**) (&spectrumG),  ALLOC_TRACK);
	trackFree((void**) (&spectrumV),  ALLOC_TRACK);

	delete fCount;
	delete axion;

	endComms();

	printMemStats();

	//JAVIER
	if (commRank() == 0)
	{
		fclose (file_sample);
		fclose (file_energy);
		fclose (file_energy2);
		fclose (file_spectrum);
	}

	return 0;
}
