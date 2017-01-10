//
//          RungeKuttaNyström version
//          created on 20.11.2015
//          simple gradient to accelerate calculations

#include<cmath>
#include<chrono>

#include<complex>
#include<vector>

#include"code3DCpu.h"
#include"scalar/scalar.h"
#include "propagator/allProp.h"
#include"enum-field.h"
#include"utils/utils.h"
#include"io/readWrite.h"
#include"comms/comms.h"

using namespace std;

/*	TODO

	2. Generate configurations
*/


#ifdef	USE_XEON
	__declspec(target(mic)) char *mX, *vX, *m2X;
#endif

#define printMpi(...) do {		\
	if (!commRank()) {		\
	  printf(__VA_ARGS__);  }	\
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

	axion = new Scalar (sizeN, sizeZ, sPrec, cDev, zInit, lowmem, zGrid, FIELD_SAXION, CONF_NONE, 0, 0, NULL);
	readConf(&axion, 0);

	//--------------------------------------------------
	//          SETTING BASE PARAMETERS
	//-------------------------------------------------- 

	double dz = (zFinl - zInit)/((double) nSteps);
	double delta = sizeL/sizeN;

	printMpi("--------------------------------------------------\n");
	printMpi("           INITIAL CONDITIONS                     \n\n");

	printMpi("Length =  %2.5f\n", sizeL);
	printMpi("N      =  %d\n",    sizeN);
	printMpi("Nz     =  %d\n",    sizeZ);
	printMpi("zGrid  =  %d\n",    zGrid);
	printMpi("dx     =  %2.5f\n", delta);  
	printMpi("dz     =  %2.5f\n", dz);
	printMpi("LL     =  %2.5f\n", LL);
	printMpi("--------------------------------------------------\n");

	const int S0 = sizeN*sizeN;
	const int SF = sizeN*sizeN*(sizeZ+1)-1;
	const int V0 = 0;
	const int VF = axion->Size()-1;

	printMpi("INITIAL CONDITIONS LOADED\n");
	if (sPrec != FIELD_DOUBLE)
	{
		printMpi("Example mu: m[0] = %f + %f*I, m[N3-1] = %f + %f*I\n", ((complex<float> *) axion->mCpu())[S0].real()/zInit, ((complex<float> *) axion->mCpu())[S0].imag()/zInit,
									        ((complex<float> *) axion->mCpu())[SF].real()/zInit, ((complex<float> *) axion->mCpu())[SF].imag()/zInit);
		printMpi("Example  v: v[0] = %f + %f*I, v[N3-1] = %f + %f*I\n", ((complex<float> *) axion->vCpu())[V0].real(), ((complex<float> *) axion->vCpu())[V0].imag(),
									        ((complex<float> *) axion->vCpu())[VF].real(), ((complex<float> *) axion->vCpu())[VF].imag());
	}
	else
	{
		printMpi("Example mu: m[0] = %lf + %lf*I, m[N3-1] = %lf + %lf*I\n", ((complex<double> *) axion->mCpu())[S0].real()/zInit, ((complex<double> *) axion->mCpu())[S0].imag()/zInit,
										    ((complex<double> *) axion->mCpu())[SF].real()/zInit, ((complex<double> *) axion->mCpu())[SF].imag()/zInit);
		printMpi("Example  v: v[0] = %lf + %lf*I, v[N3-1] = %lf + %lf*I\n", ((complex<double> *) axion->vCpu())[V0].real(), ((complex<double> *) axion->vCpu())[V0].imag(),
										    ((complex<double> *) axion->vCpu())[VF].real(), ((complex<double> *) axion->vCpu())[VF].imag());
	}

	printMpi("Ez     =  %d\n",    axion->eDepth());

	//--------------------------------------------------
	//   THE TIME ITERATION LOOP
	//--------------------------------------------------  

	printMpi("--------------------------------------------------\n");
	printMpi("           STARTING COMPUTATION                   \n");
	printMpi("--------------------------------------------------\n");

	std::chrono::high_resolution_clock::time_point start, current, old;

	int counter = 0;
	int index = 0;

	//printMpi ("Dumping configuration %05d...\n", index);
	//fflush (stdout);
	//writeConf(axion, index);


	printMpi ("Start FFT\n");
	fflush (stdout);

	commSync();

	start = std::chrono::high_resolution_clock::now();
	old = start;
	std::chrono::milliseconds elapsed;

	axion->fftCpu(1);

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);

	index++;
	writeConf(axion, index);

	printMpi("\n PROGRAMM FINISHED\n");

	if (sPrec == FIELD_DOUBLE)
	{
		printMpi("\n Examples m: m[0]= %f + %f*I, m[N3-1]= %f + %f*I\n",  ((complex<double> *) axion->mCpu())[S0].real(), ((complex<double> *) axion->mCpu())[S0].imag(),
		 								  ((complex<double> *) axion->mCpu())[SF].real(), ((complex<double> *) axion->mCpu())[SF].imag());
		printMpi("\n Examples v: v[0]= %f + %f*I, v[N3-1]= %f + %f*I\n\n",((complex<double> *) axion->vCpu())[V0].real(), ((complex<double> *) axion->vCpu())[V0].imag(),
									 	  ((complex<double> *) axion->vCpu())[VF].real(), ((complex<double> *) axion->vCpu())[VF].imag());
	}
	else
	{
		printMpi("\n Examples m: m[0]= %f + %f*I, m[N3-1]= %f + %f*I\n",  ((complex<float> *) axion->mCpu())[S0].real(), ((complex<float> *) axion->mCpu())[S0].imag(),
										  ((complex<float> *) axion->mCpu())[SF].real(), ((complex<float> *) axion->mCpu())[SF].imag());
		printMpi("\n Examples v: v[0]= %f + %f*I, v[N3-1]= %f + %f*I\n\n",((complex<float> *) axion->vCpu())[V0].real(), ((complex<float> *) axion->vCpu())[V0].imag(),
										  ((complex<float> *) axion->vCpu())[VF].real(), ((complex<float> *) axion->vCpu())[VF].imag());
	}

	printMpi("z_final = %f\n", *axion->zV());
	printMpi("Total time: %2.3f s\n", elapsed.count()*1.e-3);
	printMpi("--------------------------------------------------\n");

	delete fCount;

	endComms();
    
	return 0;
}
