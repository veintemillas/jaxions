#include <cmath>
#include <cstring>
#include <chrono>

#include <complex>
#include <vector>

#include "propagator/allProp.h"
#include "energy/energy.h"
#include "enum-field.h"
#include "utils/utils.h"
#include "io/readWrite.h"
#include "comms/comms.h"
#include "map/map.h"
#include "strings/strings.h"
#include "powerCpu.h"
#include "scalar/scalar.h"

#include<mpi.h>
#include<omp.h>

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

	std::chrono::high_resolution_clock::time_point start, current, old;
	std::chrono::milliseconds elapsed;

	printMpi("\n-------------------------------------------------\n");
	printMpi("\n          TESTING CODE STUFF!                \n\n");


	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	FlopCounter *fCount = new FlopCounter;

	start = std::chrono::high_resolution_clock::now();

	Scalar *axion;
	char fileName[256];

	printMpi("Generating scalar ... ");
	axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, fType, cType, parm1, parm2, fCount);
	printMpi("Done! \n");

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
	printMpi("ICtime %f min\n",elapsed.count()*1.e-3/60.);


	//--------------------------------------------------
	//          OUTPUTS FOR CHECKING
	//--------------------------------------------------

	double Vr, Vt, Kr, Kt, Grz, Gtz;
	size_t nstrings = 0 ;
	size_t nstrings_global = 0 ;

  double nstringsd = 0. ;
	double nstringsd_global = 0. ;
	double maximumtheta = 3.141597;
	size_t sliceprint = 1;

	// Axion spectrum
	const int kmax = axion->Length()/2 -1;
	int powmax = floor(1.733*kmax)+2 ;

	double  *spectrumK ;
	double  *spectrumG ;
	double  *spectrumV ;
	double  *binarray	 ;
	trackAlloc((void**) (&spectrumK), 8*powmax);
	trackAlloc((void**) (&spectrumG), 8*powmax);
	trackAlloc((void**) (&spectrumV), 8*powmax);
	trackAlloc((void**) (&binarray),  10000*sizeof(size_t));
	printMpi("Bins allocated! \n");

	double *sK = static_cast<double *> (spectrumK);
	double *sG = static_cast<double *> (spectrumG);
	double *sV = static_cast<double *> (spectrumV);
	double *bA = static_cast<double *> (binarray);
	//double *bAd = static_cast<double *> (binarray);

 // complex<float> *mSf = static_cast<complex<float>*> (axion->mCpu());
 // complex<float> *vSf = static_cast<complex<float>*> (axion->vCpu());
 // complex<double> *mSd = static_cast<complex<double>*> (axion->mCpu());
 // complex<double> *vSd = static_cast<complex<double>*> (axion->vCpu());
 //
 // float *mTf = static_cast<float*> (axion->mCpu());
 // float *vTf = static_cast<float*> (axion->vCpu());
 // double *mTd = static_cast<double*> (axion->mCpu());
 // double *vTd = static_cast<double*> (axion->vCpu());

	double z_now ;


	///////////////////////////////

	//const int kmax = axion->Length()/2 -1;
	//int powmax = floor(1.733*kmax)+2 ;
	const int n1 = axion->Length();


	#pragma omp parallel default(shared)
	{

		int tid = omp_get_thread_num();


		double spectrumK_private[powmax];
		double spectrumG_private[powmax];
		double spectrumV_private[powmax];

		printf("th%d alloc, ",tid);fflush(stdout);

		for (int i=0; i < powmax; i++)
		{
			spectrumK_private[i] = 0.0;
			spectrumG_private[i] = 0.0;
			spectrumV_private[i] = 0.0;
		}

	size_t idx, midx;
	int bin;
	int kz, ky, kx ;
	size_t iz, nz, iy, ny, ix, nx;
	double k2, w;
	complex<float> ftk, ftmk;


	#pragma omp for schedule(static)
	for (int kz = 0; kz<kmax + 0; kz++)
	{

		iz = (n1+kz)%n1 ;
		nz = (n1-kz)%n1 ;

		for (int ky = -kmax; ky<kmax + 1; ky++)
		{
			iy = (n1+ky)%n1 ;
			ny = (n1-ky)%n1 ;

			for	(int kx = -kmax; kx<kmax + 1; kx++)
			{
				ix = (n1+kx)%n1 ;
				nx = (n1-kx)%n1 ;

				k2 =	kx*kx + ky*ky + kz*kz;
				bin  = (int) floor(sqrt(k2)) 	;

				//CONTINUUM DEFINITION
				//k2 =	(39.47842/(sizeL*sizeL)) * k2;
				//double w = (double) sqrt(k2 + mass2);
				//LATICE DEFINITION
				//this first instance of w is aux
				//k2 =	(minus1costab[abs(kx)]+minus1costab[abs(ky)]+minus1costab[abs(kz)]);
				//w = sqrt(k2 + mass2);
				//k2 =	(39.47841760435743/(sizeL*sizeL)) * k2;

				idx  = ix+iy*n1+iz*n1*n1;
				midx = nx+ny*n1+nz*n1*n1;

				ftk = static_cast<complex<float>  *> (axion->mCpu())[idx];
				ftmk = static_cast<complex<float>  *> (axion->mCpu())[midx];

				if(!(kz==0||kz==kmax+1))
				{
				// -k is in the negative kx volume
				// it not summed in the for loop so include a factor of 2
				spectrumK_private[bin] += 2.*pow(abs(ftk - ftmk),2);
				spectrumG_private[bin] += 2.*pow(abs(ftk + ftmk),2);		//mass2 is included
				spectrumV_private[bin] += 2.*pow(abs(ftk + ftmk),2);								//mass2 is included
				}
				else
				{
				// -k is in the kz=0 so both k and -k will be summed in the loop
				spectrumK_private[bin] += pow(abs(ftk - ftmk),2);
				spectrumG_private[bin] += pow(abs(ftk + ftmk),2);		//mass2 is included
				spectrumV_private[bin] += pow(abs(ftk + ftmk),2);								//mass2 is included
				}
			}//x

		}//y
	}//z

	#pragma omp critical
	{
		for(int n=0; n<powmax; n++)
		{
			static_cast<double*>(spectrumK)[n] += spectrumK_private[n];
			static_cast<double*>(spectrumG)[n] += spectrumG_private[n];
			static_cast<double*>(spectrumV)[n] += spectrumV_private[n];
		}
	}

}










	//--------------------------------------------------
	//          SETTING BASE PARAMETERS
	//--------------------------------------------------

	double delta = sizeL/sizeN;
	double dz;
	double dzaux;
	double llaux;
	double llprint;

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

	//--------------------------------------------------
	//   THE TIME ITERATION LOOP
	//--------------------------------------------------

	printMpi("--------------------------------------------------\n");
	printMpi("           STARTING TEST                   \n");
	printMpi("--------------------------------------------------\n");


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

	bool coZ = 1;
  bool coS = 1;
	int strcount = 0;

	axion->SetLambda(LAMBDA_Z2)	;
	if (LAMBDA_FIXED == axion->Lambda())
	{
	printMpi ("Lambda in FIXED mode\n");
	}
	else
	{
		printMpi ("Lambda in Z2 mode\n");
	}

	Folder munge(axion);

	if (cDev != DEV_GPU)
	{
		printMpi ("Folding configuration ... ");
		munge(FOLD_ALL);
	}
	printMpi ("Done! \n");

	int nLoops;

	if (dump == 0)
		nLoops = 0;
	else
		nLoops = (int)(nSteps/dump);

	commSync();

	start = std::chrono::high_resolution_clock::now();
	old = start;

	printMpi("--------------------------------------------------\n");
	printMpi("TO THETA\n");
	cmplxToTheta (axion, fCount);
	fflush(stdout);
	printMpi("--------------------------------------------------\n");

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - old);


  munge(UNFOLD_SLICE, sliceprint);
	writeMap (axion, index);

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	printMpi("Unfold ... ");
	munge(UNFOLD_ALL);
	printMpi("| ");

	if (axion->Field() == FIELD_AXION)
	{

		printMpi("nSpec ... ");
		//NUMBER SPECTRUM
		spectrumUNFOLDED(axion, spectrumK, spectrumG, spectrumV);
		//printf("sp %f %f %f ...\n", (float) sK[0]+sG[0]+sV[0], (float) sK[1]+sG[1]+sV[1], (float) sK[2]+sG[2]+sV[2]);
		printMpi("| ");

		printMpi("DensMap ... ");
		axion->writeMAPTHETA( (*(axion->zV() )) , index, binarray, 10000)		;
		printMpi("| ");

		if (commRank() == 0)
		{
		printf("%f ", (*(axion->zV() )));
		// first three numbers are dens average, max contrast and maximum of the binning
		for(int i = 0; i<10000; i++) {	printf("%f ", (float) bA[i]);}
		printf("\n");

		}
		// BIN THETA
		maximumtheta = axion->thetaDIST(100, spectrumK);
		if (commRank() == 0)
		{
			printf("%f %f ", (*(axion->zV() )), maximumtheta );
			for(int i = 0; i<100; i++) {	printf("%f ", (float) sK[i]);} printf("\n");
		}

		printMpi("dens2m ... ");
		axion->denstom();
		printMpi("| ");

		printMpi("pSpec ... ");
		//POWER SPECTRUM
		if (commRank() == 0)
		{
		powerspectrumUNFOLDED(axion, spectrumK, spectrumG, spectrumV, fCount);
		printf("sp %f %f %f ...\n", (float) sK[0]+sG[0]+sV[0], (float) sK[1]+sG[1]+sV[1], (float) sK[2]+sG[2]+sV[2]);
		printf("%f ", (*axion->zV()));
		for(int i = 0; i<powmax; i++) {	printf("%f ", (float) sK[i]);} printf("\n");
		printf("%f ", (*axion->zV()));
		for(int i = 0; i<powmax; i++) {	printf("%f ", (float) sG[i]);} printf("\n");
		printf("%f ", (*axion->zV()));
		for(int i = 0; i<powmax; i++) {	printf("%f ", (float) sV[i]);} printf("\n");
		}
		printMpi("| ");

		//munge(FOLD_ALL);
	}

	if (nSteps > 0)
	writeConf(axion, index);

	trackFree(&eRes, ALLOC_TRACK);
	trackFree(&str,  ALLOC_ALIGN);
	trackFree((void**) (&spectrumK),  ALLOC_TRACK);
	trackFree((void**) (&spectrumG),  ALLOC_TRACK);
	trackFree((void**) (&spectrumV),  ALLOC_TRACK);
	trackFree((void**) (&binarray),  ALLOC_TRACK);

	delete fCount;
	delete axion;

	endComms();

	printMemStats();


	return 0;
}
