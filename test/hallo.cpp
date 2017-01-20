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

#include<omp.h>
#include<mpi.h>

using namespace std;

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

	printf("\n-------------------------------------------------\n");
	printf("\n          HALOO !                \n\n");

	//--------------------------------------------------
	//       READING INITIAL CONDITIONS
	//--------------------------------------------------

	start = std::chrono::high_resolution_clock::now();

	Scalar *axion;
	char fileName[256];

	//This reads from an Axion.00000 file
	readConf(&axion, fIndex);
	if (axion == NULL)
			{
				printf ("Error reading HDF5 file\n");
				exit (0);
	  	}


	const size_t S0 = sizeN*sizeN;
	const size_t SF = sizeN*sizeN*(sizeZ+1)-1;
	const size_t V0 = 0;
	const size_t VF = axion->Size()-1;
	const size_t n1 = axion->Length();
	const size_t Lz = axion->Depth();

	double delta = sizeL/sizeN;

	current = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
	printf("Reading time %f min\n",elapsed.count()*1.e-3/60.);


		printMpi("--------------------------------------------------\n");
		printMpi("Length =  %2.2\n", sizeL);
		printMpi("N      =  %ld\n",   sizeN);
		printMpi("Nz     =  %ld\n",   Lz);
		printMpi("zGrid  =  %ld\n",   zGrid);
		printMpi("dx     =  %2.5f\n", delta);
		printMpi("--------------------------------------------------\n");



	float *mD = static_cast<float*> (axion->mCpu());
	float lala ;

	//TEST PRINT
	// for (size_t j =0; j<Lzz; j++)
	// {
	// 	for (size_t i =0; i<S0; i++)
	// 	{
	// 		lala = mD[S0*(j+1)+i];
	// 	}
	// 	printMpi("m[S0*%d] = %f\n", j+1, lala);
	// }
	printMpi("Searching for local maxima \n");

	axion->exchangeGhosts(FIELD_M);
	size_t Nmaxima = 0;
	size_t reads = 0;

	#pragma omp parallel for default(shared) schedule(static) reduction(+:Nmaxima,reads)
	for (size_t iz=0; iz < Lz; iz++)
	{
		//int tid = omp_get_thread_num();
		//printMpi("thread %d gets iz=%d\n",tid, iz);
		size_t idx, idaux ;
		for (size_t iy=0; iy < n1; iy++)
		{
			int iyP = (iy+1)%n1;
			int iyM = (iy-1+n1)%n1;
			for (size_t ix=0; ix < n1; ix++)
			{
				reads++;
				idx = ix + iy*n1+(iz+1)*S0 ;
				if (mD[idx]<5)
				continue;

				int ixP = (ix+1)%n1;
				int ixM = (ix-1+n1)%n1;

				idaux = ixP + iy*n1+(iz+1)*S0 ;
				if (mD[idaux]-mD[idx]<0)
				continue;

				idaux = ixM + iy*n1+(iz+1)*S0 ;
				if (mD[idaux]-mD[idx]<0)
				continue;

				idaux = ix + iyP*n1+(iz+1)*S0 ;
				if (mD[idaux]-mD[idx]<0)
				continue;

				idaux = ix + iyM*n1+(iz+1)*S0 ;
				if (mD[idaux]-mD[idx]<0)
				continue;

				if (mD[idx+S0]-mD[idx]<0)
				continue;

				if (mD[idx-S0]-mD[idx]<0)
				continue;
				//printMpi("(iz,iy,ix)=(%d,%d,%d)r=%f\n ",iz,iy,ix,mD[idx]);
				Nmaxima++ ;
			} //END X LOOP
		} //END Y LOOP
	} //END Z LOOP





	printMpi("n1=%d,Lz=%d Nmaxima=%lu  out of %lu  reads \n",n1, Lz, Nmaxima, reads);



	printMpi("FISHING ...\n");








	// FINISHING
	// FINISHING


	delete axion;

	endComms();

	printMemStats();



	return 0;
}
