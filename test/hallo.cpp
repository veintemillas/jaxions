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

#include <fftw3.h>
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
	const size_t n3 = axion->Size();
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
	printMpi("z      =  %2.5f\n", (*axion->zV()));
	printMpi("--------------------------------------------------\n");

	float *mD = static_cast<float*> (axion->mCpu());
	float *vD = static_cast<float*> (axion->vCpu());
	complex<float> *kD = static_cast<complex<float>*> (axion->vCpu());
	float lala ;


	// //TEST PRINT
	// for (size_t j =0; j<Lzz; j++)
	// {
	// 	for (size_t i =0; i<S0; i++)
	// 	{
	// 		lala = mD[S0*(j+1)+i];
	// 	}
	// 	printMpi("m[S0*%d] = %f\n", j+1, lala);
	// }
	// printMpi("Searching for local maxima \n");

	//TEST PRINT

		// for (size_t i =0; i<5; i++)
		// {
		// 	lala = mD[i];
		// 	printMpi("m[%d] = %f ", i, lala);
		// }
		// printMpi("\n");
		// for (size_t i =0; i<5; i++)
		// {
		// 	lala = mD[S0+i];
		// 	printMpi("m[S0+%d] = %f ", i, lala);
		// }
		// printMpi("\n");

	// COPY initial m(dens) into v [it will be rewriten]
	#pragma omp parallel for default(shared) schedule(static)
	for (size_t idx=S0; idx < n3+S0; idx++)
	{
		vD[idx] = mD[idx-S0];
	}

	printMpi("1 - Search for local maxima \n");
	printMpi("---------------------------\n");

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
				if (mD[idx]<1)
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

	printMpi("n1=%d,Lz=%d Nmaxima=%lu  out of %f  reads [%f permil in volume]\n",
	n1, Lz, Nmaxima, reads/pow(sizeN,3),(float) 1000*Nmaxima/reads);

	//printMpi("Folded? %d\n",axion->Folded());

	Folder munge(axion);

	// #pragma omp parallel for default(shared) schedule(static)
	// for (size_t idx=S0; idx < n3+S0; idx++)
	// {
	// 	size_t ix = idx%n1;
	// 	mD[idx] = ix%128;
	// }

	printMpi("Print dens in at\n");
	for (int sliceprint =0; sliceprint<10; sliceprint++)
	{
		munge(UNFOLD_SLICE, sliceprint);
		writeMapAt (axion, sliceprint);
	}

	//axion->loadHalo();
	//printMpi("\n");


	if (!fftw_init_threads())
	{
		printf ("Error initializing FFT with threads\n");
		fflush (stdout);
	} else {
		int nThreads = omp_get_max_threads();
		printf ("Using %d threads for the FFTW\n", nThreads);
		fflush (stdout);
		fftw_plan_with_nthreads(nThreads);
	}

	fftwf_complex *FTFT ;
	fftwf_plan planr2c, planc2r;

	FTFT = (fftwf_complex*) fftwf_malloc(sizeN*sizeN*(sizeN/2+1) * sizeof(fftwf_complex));

	planr2c = fftwf_plan_dft_r2c_3d(Lz, n1, n1, mD, FTFT, FFTW_ESTIMATE );
	planc2r = fftwf_plan_dft_c2r_3d(Lz, n1, n1, FTFT, mD,  FFTW_ESTIMATE );

	fftwf_execute(planr2c);

	//axion->fftCpuHalo(-1);
	// for (size_t idx=0; idx < 10; idx++)
	// {
	// 		printMpi("Print v[(n1/2+1)*idx]=%f %f*I -- ",idx,kD[(n1/2+1)*idx].real(), kD[(n1/2+1)*idx].imag());
	// 		printMpi("Print v[(n1/2+1)*idx]=%f %f*I -- ",idx,kD[(n1/2+1)*idx].real(), kD[(n1/2+1)*idx].imag());
	// 		printMpi("Print v[S0*+%d]=%f %f*I\n",  idx,kD[n1+2+idx].real(), kD[n1+2+idx].imag());
	// }
	//printMpi("Print v[last]=%f %f*I\n",kD[n3].real(), kD[n3].imag());

	size_t n21 = n1/2+1;
	double filterGA = (double) sizeN ;
	double filterTH = 200. ;
	printMpi("Filter within %f (Gauss)- %f (TopHat)\n",filterGA, filterTH);

#pragma omp parallel
{
	size_t az,ay,ax;
	int kx, ky, kz;
	double control;

	#pragma omp parallel for default(shared) schedule(static)
	for (size_t idk = 0; idk < S0*n21; idk++)
		{
			ax = idk%n21;
			kx = (int) ax;
			ay = ((idk-ax)/n21)%n1;
			ky = (int) ay ;
			if (ay>n1/2) {ky = (int) ay-n1;}
			az = idk/(n1*n21);
			kz = (int) az ;
			if (az>n1/2) {kz = (int) az-n1;}

			control = sqrt((double) (kx*kx+ky*ky+kz*kz))/filterGA;
			// if (control > 1)
			// {
			// 	FTFT[idk][0] = 0.f ;
			// 	FTFT[idk][1] = 0.f ;
			// }
			 FTFT[idk][0] *= exp(-control) ;
			 FTFT[idk][1] *= exp(-control)	;
		}
}
	//axion->fftCpuHalo(+1);
	fftwf_execute(planc2r);

	#pragma omp parallel for default(shared) schedule(static)
	for (size_t idx=S0; idx < n3+S0; idx++)
	{
		mD[idx] = mD[idx]/((float) n3);
	}

	printMpi("Check average ... \n");
	double ave = 0.;

	#pragma omp parallel for default(shared) schedule(static) reduction(+:ave)
	for (size_t idx=S0; idx < n3+S0; idx++)
	{
		ave += (double) mD[idx] ;
	}
	printMpi("%f \n", ave/n3);

	printMpi("Print dens in at\n");
	for (int sliceprint =0; sliceprint<10; sliceprint++)
	{
		munge(UNFOLD_SLICE, sliceprint);
		writeMapAt (axion, sliceprint+10);
	}

	Nmaxima = 0;
	reads= 0;
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
				if (mD[idx]<1)
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

	printMpi("n1=%d,Lz=%d Nmaxima=%lu  out of %f  reads [%f permil in volume]\n",
	n1, Lz, Nmaxima, reads/pow(sizeN,3),(float) 1000*Nmaxima/reads);


	size_t Listmax[Nmaxima];
	Nmaxima=0;
	#pragma omp parallel for default(shared) schedule(static)
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
				idx = ix + iy*n1+(iz+1)*S0 ;
				if (mD[idx]<1)
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

				#pragma omp critical
				{
					Listmax[Nmaxima] = idx;
					Nmaxima++ ;
				}
			} //END X LOOP
		} //END Y LOOP
	} //END Z LOOP



	if (Nmaxima <1000)
	{
		for (size_t n=0; n<Nmaxima; n++)
		{
			printf("Max[%d][r]= %f ", n, mD[Listmax[n]+S0]);
		}
	}
	else
	{
		printMpi("Too many maxima to print\n");
	}

	printMpi("- Outputbin \n");
	printMpi("---------------------------\n");

	char stoHal[256];
	sprintf(stoHal,   "out/halo.txt");
	FILE *HalWrite = NULL;


	if ((HalWrite  = fopen(stoHal, "w+")) == NULL)
	{
		printf ("Couldn't open file %s for writing\n", stoHal);
	}
	else
	{
		for(size_t n=0; n<Nmaxima; n++)
		{
			fprintf(HalWrite, "%f ", mD[Listmax[n]+S0]) ;
		}
		fprintf(HalWrite,"\n");
	}



	printMpi("--------------------------------------------------\n");
	printMpi("   FISHING ...\n");
	printMpi("--------------------------------------------------\n");



	fftwf_destroy_plan(planc2r);
	fftwf_destroy_plan(planr2c);
	fftwf_free(FTFT);



	// FINISHING
	// FINISHING


	delete axion;

	endComms();

	printMemStats();



	return 0;
}
