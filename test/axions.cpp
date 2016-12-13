#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "code3DCpu.h"
#include "scalar/scalarField.h"
#include "propagator/propagator.h"
#include "propagator/propSimple.h"
#include "energy/energy.h"
#include "enum-field.h"
#include "utils/index.h"
#include "utils/parse.h"
#include "utils/flopCounter.h"
#include "utils/memAlloc.h"
#include "io/readWrite.h"
#include "comms/comms.h"
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

	if (initComms(argc, argv, zGrid, cDev) == -1)
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
	{
		if (sPrec != FIELD_DOUBLE)
			sprintf(fileName, "data/initial_conditions_m_single.txt");
		else
			sprintf(fileName, "data/initial_conditions_m.txt");
		//This prepares the axion field from default files
		axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, fileName, lowmem, zGrid, CONF_NONE, 0, 0, fCount);
		printMpi("Eo\n");
	}
	else
	{
		if (fIndex == -1)
			//This generates initial conditions
			axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, initFile, lowmem, zGrid, cType, parm1, parm2, fCount);
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


	//energy 2//	FILE *file_energy2 ;
	//energy 2//	file_energy2 = NULL;

	FILE *file_spectrum ;
	file_spectrum = NULL;
	FILE *file_power ;
	file_power = NULL;

	if (commRank() == 0)
	{
		file_sample = fopen("out/sample.txt","w+");
		//fprintf(file_sample,"%f %f %f\n",z, creal(m[0]), cimag(m[0]));

		file_energy = fopen("out/energy.txt","w+");
		//fprintf(file_sample,"%f %f %f\n",z, creal(m[0]), cimag(m[0]));

		//energy 2//	file_energy2 = fopen("out/energy2.txt","w+");

		file_spectrum = fopen("out/spectrum.txt","w+");
		file_power = fopen("out/power.txt","w+");
	}

	double Vr, Vt, Kr, Kt, Grz, Gtz;
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
	printMpi("--------------------------------------------------\n");

	const size_t S0 = sizeN*sizeN;
	const size_t SF = sizeN*sizeN*(sizeZ+1)-1;
	const size_t V0 = 0;
	const size_t VF = axion->Size()-1;

	printMpi("INITIAL CONDITIONS LOADED\n");
	if (sPrec != FIELD_DOUBLE)
	{
		printMpi("Example mu: m[0] = %f + %f*I, m[N3-1] = %f + %f*I\n", ((complex<float> *) axion->mCpu())[S0].real(), ((complex<float> *) axion->mCpu())[S0].imag(),
									        ((complex<float> *) axion->mCpu())[SF].real(), ((complex<float> *) axion->mCpu())[SF].imag());
		printMpi("Example  v: v[0] = %f + %f*I, v[N3-1] = %f + %f*I\n", ((complex<float> *) axion->vCpu())[V0].real(), ((complex<float> *) axion->vCpu())[V0].imag(),
									        ((complex<float> *) axion->vCpu())[VF].real(), ((complex<float> *) axion->vCpu())[VF].imag());
	}
	else
	{
		printMpi("Example mu: m[0] = %lf + %lf*I, m[N3-1] = %lf + %lf*I\n", ((complex<double> *) axion->mCpu())[S0].real(), ((complex<double> *) axion->mCpu())[S0].imag(),
										    ((complex<double> *) axion->mCpu())[SF].real(), ((complex<double> *) axion->mCpu())[SF].imag());
		printMpi("Example  v: v[0] = %lf + %lf*I, v[N3-1] = %lf + %lf*I\n", ((complex<double> *) axion->vCpu())[V0].real(), ((complex<double> *) axion->vCpu())[V0].imag(),
										    ((complex<double> *) axion->vCpu())[VF].real(), ((complex<double> *) axion->vCpu())[VF].imag());
	}

	//JAVIER commented next
	//printMpi("Ez     =  %ld\n",    axion->eDepth());

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
		//printMpi ("Dumping configuration %05d ...", index);
		//writeConf(axion, index);
		//printMpi ("Done!\n");
		printMpi ("Bypass configuration writting!\n");
		fflush (stdout);
	}
	else
		index = fIndex + 1;

	//JAVIER commented next
	//printf ("Process %d reached syncing point\n", commRank());
	//fflush (stdout);
//	commSync();

	if (cDev != DEV_GPU)
	{
		printMpi ("Folding configuration\n");
		axion->foldField();
	}

	if (cDev != DEV_CPU)
	{
		printMpi ("Transferring configuration to device\n");
		axion->transferDev(FIELD_MV);
	}

//	if (cDev != DEV_GPU)
	{
		double	strDen;

		if ((*axion->zV()) > 1.2 )
		{
			printMpi("Strings...");
			analyzeStrFolded(axion, index);
			printMpi("Vector Strings...");
			//strDen = strings(axion, cDev, str, fCount);
			//printMpi(" Done! String density %lf\n", strDen);
		}

		memcpy   (axion->mCpu(), static_cast<char *> (axion->mCpu()) + S0*sizeZ*axion->DataSize(), S0*axion->DataSize());
		//copy v unfolded into last slice
		//memcpy   (axion->mCpu(), static_cast<char *> (axion->mCpu()) + S0*sizeZ*axion->DataSize(), S0*axion->DataSize());
		//axion->unfoldField2D(sizeZ-1);
		axion->unfoldField2D(0);
		writeMap (axion, index);
		energy(axion, LL, nQcd, delta, cDev, eRes, fCount);

		//energy 2//	axion->writeENERGY ((*(axion->zV() )),file_energy, Grz, Gtz, Vr, Vt, Kr, Kt);


		if (commRank() == 0)
		{
			if (axion->Precision() == FIELD_DOUBLE)
			{
				double *eR = static_cast<double *> (eRes);
				fprintf(file_energy,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf\n", (*axion->zV()), eR[6], eR[7], eR[8], eR[9], eR[0], eR[2], eR[4], eR[1], eR[3], eR[5]);

				//energy 2//	fprintf(file_energy2,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf\n", (*axion->zV()), Vr, Vt, Kr, Kt, Grz, Gtz);
				printMpi("??/?? - - - ENERGY Vr=%lf Va=%lf Kr=%lf Ka=%lf Gr=%lf Ga=%lf \n", eR[6], eR[7], eR[8], eR[9], eR[0] + eR[2] + eR[4], eR[1] + eR[3] + eR[5]);
				//printMpi("ENERGY & PRINTED - - - Vr=%lf Va=%lf Kr=%lf Ka=%lf Gr=%lf Ga=%lf \n", eR[6], eR[7], eR[8], eR[9], eR[0] + eR[2] + eR[4], eR[1] + eR[3] + eR[5]);
			}
			else
			{
				double *eR = static_cast<double *> (eRes);
				fprintf(file_energy,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf\n", (*axion->zV()), eR[6], eR[7], eR[8], eR[9], eR[0], eR[2], eR[4], eR[1], eR[3], eR[5]);
				//float *eR = static_cast<float *> (eRes);
				//fprintf(file_energy,  "%+f %+f %+f %+f %+f %+f %+f %+f %+f %+f %+f\n", (*axion->zV()), eR[6], eR[7], eR[8], eR[9], eR[0], eR[2], eR[4], eR[1], eR[3], eR[5]);

				//energy 2//	fprintf(file_energy2,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf\n", (*axion->zV()), Vr, Vt, Kr, Kt, Grz, Gtz);
				//printMpi("ENERGY & PRINTED - - - Vr=%f Va=%f Kr=%f Ka=%f Gr=%f Ga=%f \n", eR[6], eR[7], eR[8], eR[9], eR[0] + eR[2] + eR[4], eR[1] + eR[3] + eR[5]);
				printMpi("??/?? - - - ENERGY Vr=%lf Va=%lf Kr=%lf Ka=%lf Gr=%lf Ga=%lf \n", eR[6], eR[7], eR[8], eR[9], eR[0] + eR[2] + eR[4], eR[1] + eR[3] + eR[5]);
			}
		}

/*	TEST  CON LA ENERGIA MODIFICADA PARA VER EL RENDIMIENTO, SE PUEDE BORRAR	*/

		// double Grz, Gtz, Vr, Vt, Kr, Kt;
		// 	old = std::chrono::high_resolution_clock::now();
		// axion->writeENERGY ((*(axion->zV() )),file_energy, Grz, Gtz, Vr, Vt, Kr, Kt);
		// 	current = std::chrono::high_resolution_clock::now();
		// 	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);
		//
		// 	printMpi("Elapsed %2.3lf\n", elapsed.count()*1.e-3);
		//
		// 	old = std::chrono::high_resolution_clock::now();
		// energy(axion, LL, nQcd, delta, cDev, eRes, fCount);
		// 	current = std::chrono::high_resolution_clock::now();
		// 	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);
		//
		// 	printMpi("Elapsed %2.3lf\n", elapsed.count()*1.e-3);
		//
		// if (axion->Precision() == FIELD_SINGLE)
		// {
		// 	printf ("Gxrho %+e +%le\nGxth %+e +%le\nGyrho %+e +%le\nGyth %+e +%le\nGzrho %+e +%le\nGzth %+e +%le\n",
		// 	(static_cast<float*>(eRes))[0], 0., (static_cast<float*>(eRes))[1], 0.,  (static_cast<float*>(eRes))[2], 0.,
		// 	(static_cast<float*>(eRes))[3], 0., (static_cast<float*>(eRes))[4], Grz, (static_cast<float*>(eRes))[5], Gtz);
		//
		// 	printf ("Vrho %+e +%le\nVth %+e +%le\nKrho %+e +%le\nKth %+e +%le\n",
		// 	(static_cast<float*>(eRes))[6], Vr, (static_cast<float*>(eRes))[7], Vt, (static_cast<float*>(eRes))[8], Kr,
		// 	(static_cast<float*>(eRes))[9], Kt);
		// 	fflush(stdout);
		// } else {
		//
		// 	printf ("Gxrho %+le +%le\nGxth %+e +%e\nGyrho %+le +%le\nGyth %+le +%le\nGzrho %+le +%le\nGzth %+le +%le\n",
		// 	(static_cast<double*>(eRes))[0], 0., (static_cast<double*>(eRes))[1], 0.,  (static_cast<double*>(eRes))[2], 0.,
		// 	(static_cast<double*>(eRes))[3], 0., (static_cast<double*>(eRes))[4], Grz, (static_cast<double*>(eRes))[5], Gtz);
		//
		// 	printf ("Vrho %+le +%e\nVth %+le +%le\nKrho %+le +%le\nKth %+e +%e\n",
		// 	(static_cast<double*>(eRes))[6], Vr, (static_cast<double*>(eRes))[7], Vt, (static_cast<double*>(eRes))[8], Kr,
		// 	(static_cast<double*>(eRes))[9], Kt);
		// 	fflush(stdout);
		// }

	}

	if (dump > nSteps)
		dump = nSteps;

	int nLoops;

	if (dump == 0)
		nLoops = 0;
	else
		nLoops = (int)(nSteps/dump);

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
						static_cast<complex<double> *> (axion->vCpu())[S0].real(), static_cast<complex<double> *> (axion->vCpu())[S0].imag());
				} else {
					fprintf(file_sample,"%f %f %f %f %f\n",(*(axion->zV() )), static_cast<complex<float>  *> (axion->mCpu())[S0].real(), static_cast<complex<float>  *> (axion->mCpu())[S0].imag(),
						static_cast<complex<float>  *> (axion->vCpu())[S0].real(), static_cast<complex<float>  *> (axion->vCpu())[S0].imag());
				}
			}

			old = std::chrono::high_resolution_clock::now();
			propagate (axion, dz, LL, nQcd, delta, cDev, fCount);
//			propagateSimple (axion, dz, LL, nQcd, delta);

			current = std::chrono::high_resolution_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);

			fCount->addTime(elapsed.count()*1.e-3);
			//JAVIER commented next line
			//verbose? YEAH
			//printMpi("%2d - %2d: z = %lf elapsed time =  %2.3lf s\n", zloop, zsubloop, *(axion->zV()), fCount->DTime());

			counter++;
		} // zsubloop

		//printMpi ("Transfer to CPU ...");
		fflush (stdout);
		axion->transferCpu(FIELD_MV);

/*	TODO

	2. Fix writeMap so it reads data from the first slice of m
*/

//		if (cDev != DEV_GPU)
		{
			//double Grz, Gtz, Vr, Vt, Kr, Kt;
//			writeConf(axion, index);
			//if (axion->Precision() == FIELD_DOUBLE)
			if ((*axion->zV()) > 1.2 )
			{
				printMpi("Strings (if %f>1.2) ... ", (*axion->zV()));
				fflush (stdout);
				nstrings = analyzeStrFolded(axion, index);
				printMpi("stLength = %d ", nstrings);
				fflush (stdout);

				if (nstrings == 0 )
				{
					//POWER SPECTRUM
					double *sK = static_cast<double *> (spectrumK);
					double *sG = static_cast<double *> (spectrumG);
					double *sV = static_cast<double *> (spectrumV);
					axion->unfoldField();
					powerspectrumUNFOLDED(axion, spectrumK, spectrumG, spectrumV, fCount);
					//printf("sp %f %f %f ...\n", (float) sK[0]+sG[0]+sV[0], (float) sK[1]+sG[1]+sV[1], (float) sK[2]+sG[2]+sV[2]);
					fprintf(file_power,  "%f ", (*axion->zV()));
					for(int i = 0; i<powmax; i++) {	fprintf(file_power, "%f ", (float) sK[i]);} fprintf(file_power, "\n");
					fprintf(file_power,  "%f ", (*axion->zV()));
					for(int i = 0; i<powmax; i++) {	fprintf(file_power, "%f ", (float) sG[i]);} fprintf(file_power, "\n");
					fprintf(file_power,  "%f ", (*axion->zV()));
					for(int i = 0; i<powmax; i++) {	fprintf(file_power, "%f ", (float) sV[i]);} fprintf(file_power, "\n");
					//NUMBER SPECTRUM
					spectrumUNFOLDED(axion, spectrumK, spectrumG, spectrumV);
					//printf("sp %f %f %f ...\n", (float) sK[0]+sG[0]+sV[0], (float) sK[1]+sG[1]+sV[1], (float) sK[2]+sG[2]+sV[2]);
					fprintf(file_spectrum,  "%f ", (*axion->zV()));
					for(int i = 0; i<powmax; i++) {	fprintf(file_spectrum, "%f ", (float) sK[i]);} fprintf(file_spectrum, "\n");
					fprintf(file_spectrum,  "%f ", (*axion->zV()));
					for(int i = 0; i<powmax; i++) {	fprintf(file_spectrum, "%f ", (float) sG[i]);} fprintf(file_spectrum, "\n");
					fprintf(file_spectrum,  "%f ", (*axion->zV()));
					for(int i = 0; i<powmax; i++) {	fprintf(file_spectrum, "%f ", (float) sV[i]);} fprintf(file_spectrum, "\n");
					axion->foldField();
				}
			}



			//axion->unfoldField2D(sizeZ-1);
			axion->unfoldField2D(0);
			writeMap (axion, index);
//			axion->writeENERGY ((*(axion->zV() )),file_energy, Grz, Gtz, Vr, Vt, Kr, Kt);
			energy(axion, LL, nQcd, delta, cDev, eRes, fCount);

			//energy 2// 	axion->writeENERGY ((*(axion->zV() )),file_energy, Grz, Gtz, Vr, Vt, Kr, Kt);

			if (commRank() == 0)
			{
				if (axion->Precision() == FIELD_DOUBLE)
				{
					double *eR = static_cast<double *> (eRes);
					fprintf(file_energy,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf\n", (*axion->zV()), eR[6], eR[7], eR[8], eR[9], eR[0], eR[2], eR[4], eR[1], eR[3], eR[5]);
					//energy 2// 	fprintf(file_energy2,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf\n", (*axion->zV()), Vr, Vt, Kr, Kt, Grz, Gtz);
					printMpi("\r%d/%d - - - ENERGY Vr=%lf Va=%lf Kr=%lf Ka=%lf Gr=%lf Ga=%lf \n", index, nLoops, eR[6], eR[7], eR[8], eR[9], eR[0] + eR[2] + eR[4], eR[1] + eR[3] + eR[5]);
				}
				else
				{
					double *eR = static_cast<double *> (eRes);
					fprintf(file_energy,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf %+lf\n", (*axion->zV()), eR[6], eR[7], eR[8], eR[9], eR[0], eR[2], eR[4], eR[1], eR[3], eR[5]);
		//			float *eR = static_cast<float *> (eRes);
		//			fprintf(file_energy,  "%+f %+f %+f %+f %+f %+f %+f %+f %+f %+f %+f\n", (*axion->zV()), eR[6], eR[7], eR[8], eR[9], eR[0], eR[2], eR[4], eR[1], eR[3], eR[5]);
					//energy 2//	fprintf(file_energy2,  "%+lf %+lf %+lf %+lf %+lf %+lf %+lf\n", (*axion->zV()), Vr, Vt, Kr, Kt, Grz, Gtz);
					printMpi("\r%d/%d - - - ENERGY Vr=%f Va=%f Kr=%f Ka=%f Gr=%f Ga=%f \n", index, nLoops, eR[6], eR[7], eR[8], eR[9], eR[0] + eR[2] + eR[4], eR[1] + eR[3] + eR[5]);
				}
			}
		}
	} // zloop

	// LAST DENSITY MAP
	//energyMap	(Scalar *field, const double LL, const double nQcd, const double delta, DeviceType dev, FlopCounter *fCount)
	//energyMap	(Scalar *field, const double LL, const double nQcd, const double delta, DeviceType dev, FlopCounter *fCount)



	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	printMpi("\n PROGRAMM FINISHED\n");

	if (cDev != DEV_GPU)
		axion->unfoldField();

	//if (nSteps > 0)
	//	writeConf(axion, index);

	if (sPrec == FIELD_DOUBLE)
	{
		printMpi("\n Examples m: m[0]= %le + %le*I, m[N3-1]= %le + %le*I\n",static_cast<complex<double> *> (axion->mCpu())[S0].real(), static_cast<complex<double> *> (axion->mCpu())[S0].imag(),
		 								  static_cast<complex<double> *> (axion->mCpu())[SF].real(), static_cast<complex<double> *> (axion->mCpu())[SF].imag());
		printMpi("\n Examples v: v[0]= %le + %le*I, v[N3-1]= %le + %le*I\n",static_cast<complex<double> *> (axion->vCpu())[V0].real(), static_cast<complex<double> *> (axion->vCpu())[V0].imag(),
									 	  static_cast<complex<double> *> (axion->vCpu())[VF].real(), static_cast<complex<double> *> (axion->vCpu())[VF].imag());
	}
	else
	{
		printMpi("\n Examples m: m[0]= %e + %e*I, m[N3-1]= %e + %e*I\n",  static_cast<complex<float> *> (axion->mCpu())[S0].real(), static_cast<complex<float> *> (axion->mCpu())[S0].imag(),
										  static_cast<complex<float> *> (axion->mCpu())[SF].real(), static_cast<complex<float> *> (axion->mCpu())[SF].imag());
		printMpi("\n Examples v: v[0]= %e + %e*I, v[N3-1]= %e + %e*I\n\n",static_cast<complex<float> *> (axion->vCpu())[V0].real(), static_cast<complex<float> *> (axion->vCpu())[V0].imag(),
										  static_cast<complex<float> *> (axion->vCpu())[VF].real(), static_cast<complex<float> *> (axion->vCpu())[VF].imag());
	}

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
		fclose (file_spectrum);
		fclose (file_power);
		//energy 2//	fclose (file_energy2);
	}

	return 0;
}