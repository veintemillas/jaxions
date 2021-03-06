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

#include <fftw3.h>
#include <omp.h>
#include <mpi.h>

using namespace std;

#define printMpi(...) do {		\
	if (!commRank()) {		\
	  printf(__VA_ARGS__);  	\
	  fflush(stdout); }		\
}	while (0)

int	main (int argc, char *argv[])
{
	initAxions(argc, argv);

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
	printMpi("n3     =  %ld\n",   n3);
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
		// 	lala = mD[S0*n1+i];
		// 	printMpi("m[S0+%d] = %f ", i, lala);
		// }
		// printMpi("\n");



	// COPY initial m(dens) into v [it will be rewriten]

	#pragma omp parallel for default(shared) schedule(static)
	for (size_t idx=0; idx < n3; idx++)
	{
		vD[idx] = mD[idx+S0];
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

	// PREPARE FFT

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

	// EXECUTE FFT DENS INTO FTFT

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

	// INVERSE FFT

	//axion->fftCpuHalo(+1);
	fftwf_execute(planc2r);


	// NORMALISE


	#pragma omp parallel for default(shared) schedule(static)
	for (size_t idx=S0; idx < n3+S0; idx++)
	{
		mD[idx] = mD[idx]/((float) n3);
	}


	// CHECK THAT THE AVERAGE STAYS AT 1

	printMpi("Check average ... \n");
	double ave = 0.;

	#pragma omp parallel for default(shared) schedule(static) reduction(+:ave)
	for (size_t idx=S0; idx < n3+S0; idx++)
	{
		ave += (double) mD[idx] ;
	}
	printMpi("%f \n", ave/n3);


	// PRINT SLICES OF THE FILTERED DATA FOR COMPARISON
	// USES WRITE AT, WHICH ASUMES IS c_THETA, SO DIVIDES BY Z 
	printMpi("Print dens in at\n");
	for (int sliceprint =0; sliceprint<10; sliceprint++)
	{
		munge(UNFOLD_SLICE, sliceprint);
		writeMapAt (axion, sliceprint+10);
	}


	// FIND NEW MAXIMA OF THE FILTERED DENS MAP

	// FIND NUMBER

	axion->exchangeGhosts(FIELD_M);
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

	//CREATE ARRAY OF ADRESSES TO CONTAIN MAXIMA
	size_t Listmax[Nmaxima];
	Nmaxima=0;


	// FILL THE ARRAY WITH MAXIMA

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


	// PRINT FOR JOY IF NOT A BURDEN

	if (Nmaxima <1000)
	{
		for (size_t n=0; n<Nmaxima; n++)
		{
			printf("Max[%d][r]= %f ", n, mD[Listmax[n]]);
		}
	}
	else
	{
		printMpi("Too many maxima to print\n");
	}


	// OUTPUT THE NUMBER OF MAXIMA

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
			fprintf(HalWrite, "%f ", mD[Listmax[n]]) ;
		}
		fprintf(HalWrite,"\n");
	}


	//CHOOSE the BIG HALO
	size_t biggy = 0;
	float maxi = 0.;
	for (size_t n=0; n<Nmaxima; n++)
	{
		if (mD[Listmax[n]] > maxi )
		{
			printMpi("biggy n=%d, r=%f, candidate= %f\n", biggy, maxi, mD[Listmax[n]]);
			maxi = mD[Listmax[n]];
			biggy = n;
		}
	}


	//EXTRACT ITS MASS DISTRIBUTION
	//ELLIPTICITY?

	int numi = 50;
	size_t cubeside = numi*2+1;
	size_t cubeside2 = cubeside*cubeside;
	size_t center = Listmax[biggy];
	float localcube[cubeside*cubeside*cubeside];
	float binned[cubeside];
	float binned2[cubeside];
	int   nparts[cubeside];

	size_t cz, cy, cx;
	cx = center%n1;
	cy = ((center-cx)/n1)%n1;
	cz = center/S0 -1 ;

	printMpi("cube centered at (%d,%d,%d)[%d], side = %d sites\n ",
	cz, cy, cx, Listmax[biggy], cubeside);


//#pragma omp parallel
//{

	size_t idx;
	size_t pz, py, px;
	size_t lz, ly, lx;
	size_t bin;
	double COMz =0.;
	double COMy =0.;
	double COMx =0.;
	printMpi("checs\n");
//	#pragma omp for schedule(static)
	for (int iz=-numi; iz < numi+1; iz++)
	{
		pz = (cz+iz)%n1 ;
		lz = (iz+numi);
		for (int iy=-numi; iy < numi+1; iy++)
		{
			py = (cy+iy)%n1 ;
			ly = (iy+numi);
			for (int ix=-numi; ix < numi+1; ix++)
			{
				px = (cx+ix)%n1 ;
				lx = ix+numi ;
				idx = px + py*n1 + pz*S0;
				bin= (int) sqrt((double) iz*iz+iy*iy+ix*ix);
				//printMpi("[%d](%d,%d,%d) %d %f",idx, pz,py,px, bin, mD[idx+S0]);
				localcube[lx+cubeside*ly+cubeside2*lz] = mD[idx+S0];
				if (bin<numi+1)
				{
					binned[bin] +=  mD[idx+S0];
					binned2[bin] +=  pow(mD[idx+S0],2);
					nparts[bin] += 1 ;
					COMz += (double) mD[idx+S0]*iz	;
					COMy += (double) mD[idx+S0]*iy	;
					COMx += (double) mD[idx+S0]*iz	;
				}
			}
		}
	}
//}
for(size_t n=0; n<numi+1; n++)
{
	fprintf(HalWrite, "%f ", binned[n]) ;
}
fprintf(HalWrite,"\n");
for(size_t n=0; n<numi+1; n++)
{
	fprintf(HalWrite, "%f ", binned2[n]) ;
}
fprintf(HalWrite,"\n");
for(size_t n=0; n<numi+1; n++)
{
	fprintf(HalWrite, "%d ", nparts[n]) ;
}
fprintf(HalWrite,"\n");
fprintf(HalWrite, "%f %f %f\n", COMz, COMy, COMx) ;

printMpi("Print dens around biggy\n");
for (int sliceprint = 0; sliceprint<cubeside; sliceprint++)
{
	pz = (cz+(sliceprint-numi))%n1 ;
	munge(UNFOLD_SLICE, pz);
	writeMapAt (axion, sliceprint+20);
}



	printMpi("--------------------------------------------------\n");
	printMpi("   FISHING ...\n");
	printMpi("--------------------------------------------------\n");

	fclose(HalWrite);

	fftwf_destroy_plan(planc2r);
	fftwf_destroy_plan(planr2c);
	fftwf_free(FTFT);



	// FINISHING
	// FINISHING


	delete axion;

	endAxions();

	return 0;
}
