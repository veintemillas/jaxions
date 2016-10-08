//
//          RungeKuttaNyström version
//          created on 20.11.2015
//          simple gradient to accelerate calculations

#include<cmath>
#include<ctime>

#include<complex>
#include<vector>
#include<omp.h>

#include"code3DCpu.h"
#include"scalarField.h"
#include"propagator.h"
#include"enum-field.h"
#include"index.h"
#include"parse.h"
#include"readWrite.h"
#include"comms.h"

#ifdef	PROFILE
#include<cuda_profiler_api.h>
#endif

using namespace std;

/*	TODO

	2. Generate configurations
	3. MPI y comms
	4. Overlapp comms y calc?
*/


#define printMpi(...) do {		\
	if (!commRank()) {		\
	  printf(__VA_ARGS__);  }	\
}	while (0)

int	main (int argc, char *argv[])
{
	parseArgs(argc, argv);

	if (initCudaComms(argc, argv, zGrid) == -1)
	{
		printf ("Error initializing Cuda and Mpi\n");
		return 1;
	}

	printMpi("\n-------------------------------------------------\n");
	printMpi("\n          CREATING MINICLUSTERS!                \n\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS       
	//--------------------------------------------------


	Scalar *axion;
	char fileName[256];

	if ((initFile == NULL) && (fIndex == -1))
	{
		if (sPrec != FIELD_DOUBLE)
			sprintf(fileName, "data/initial_conditions_m_single.txt");
		else
			sprintf(fileName, "data/initial_conditions_m.txt");

		axion = new Scalar (sizeN, sizeZ, sPrec, zInit, fileName, lowmem, zGrid);
	}
	else
	{
		if (fIndex == -1)
			axion = new Scalar (sizeN, sizeZ, sPrec, zInit, initFile, lowmem, zGrid);
		else
			readConf(&axion, fIndex);
	}


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

	axion->transferGpu(FIELD_MV);
	axion->transferCpu(FIELD_MV);

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

	clock_t start, current;

	int counter = 0;
	int index = 0;

	printMpi ("Dumping configuration %05d...\n", index);
	fflush (stdout);
	writeConf(axion, index);

	if (dump > nSteps)
		dump = nSteps;

	int nLoops = (int)(nSteps/dump);

	printMpi ("Start redshift loop\n");
	fflush (stdout);
	start = clock();

	for (int zloop = 0; zloop < nLoops; zloop++)
	{
		//--------------------------------------------------
		// THE TIME ITERATION SUB-LOOP
		//--------------------------------------------------

		index++;

		#ifdef PROFILE
		cudaProfilerStart();
		#endif

		for (int zsubloop = 0; zsubloop < dump; zsubloop++)
		{
			if (!lowmem)
				propagate (axion, dz, LL, nQcd, delta, true);
			else
				propLowMem(axion, dz, LL, nQcd, delta, true);

			current = clock();
			printMpi("%2d - %2d: z = %lf elapsed time =  %2.3lf s\n", zloop, zsubloop, *(axion->zV()), (double)(current-start)/((double)CLOCKS_PER_SEC*24.));
//			fflush(stdout);
			counter++;
		} // zsubloop

		#ifdef PROFILE
		cudaProfilerStop();
		#endif

		printMpi ("Dumping configuration %05d...\n", index);
		fflush (stdout);
		axion->transferCpu(FIELD_MV);
		writeConf(axion, index);
	} // zloop

	current = clock();

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
	printMpi("#_steps = %i\n", counter);
	printMpi("#_prints = %i\n", index);
	printMpi("Elapsed time: %.2f s\n ",(double)(current-start)/(double)CLOCKS_PER_SEC);
	printMpi("--------------------------------------------------\n");

	endComms();
    
	return 0;
}
