#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <climits>

#include "enum-field.h"
#include "utils/memAlloc.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
#endif

#include <omp.h>
#include <sched.h>

#ifndef __USE_POSIX
	#define HOST_NAME_MAX   128
#endif

static int rank = -1;
static int idxAcc = -1;
static int commSz = 0;
static char hostname[HOST_NAME_MAX];

static size_t gpuMem = 0;

int	commRank()
{
	return rank;
}

int	commSize()
{
	return commSz;
}

int	commAcc()
{
	return idxAcc;
}

char	*commHost()
{
	return hostname;
}

void	commSync()
{
	MPI_Barrier(MPI_COMM_WORLD);
}

size_t	gpuMemAvail()
{
	return	gpuMem;
}

int	initComms (int argc, char *argv[], int size, DeviceType dev)
{
	int nAccs = 0;
	int realSize = 1;
	int tProv;
	char *allHosts;

	MPI_Init_thread (&argc, &argv, MPI_THREAD_FUNNELED, &tProv);

	if (tProv != MPI_THREAD_FUNNELED)
		printf ("Error: Requested MPI_THREAD_FUNNELED could not be satisfied. Got %d\nOpenMP behavior undefined!!", tProv);

	MPI_Comm_size (MPI_COMM_WORLD, &realSize);

	if (realSize != size)
	{
		printf ("Error: Requested %d processes, got %d. Adjust you command line parameters\n", size, realSize);
		MPI_Finalize();
		return -1;
	}

	commSz = realSize;

	gethostname(hostname, HOST_NAME_MAX);
	hostname[HOST_NAME_MAX-1] = '\0';		// gethostname no termina la cadena con \0

	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	MPI_Barrier(MPI_COMM_WORLD);

	switch (dev)
	{
		case DEV_GPU:
		{
#ifdef	USE_GPU
			cudaError_t cErr = cudaGetDeviceCount(&nAccs);

			if (cErr != cudaSuccess)
			{
				printf("Rank %d CUDA error (host %s):\n", rank, hostname);
				printf("%s\n", cudaGetErrorString(cErr));
				MPI_Finalize();
				return -1;
			}
#else
			printf ("Gpu support not built\n");
			exit   (1);
#endif
			break;
		}

		case DEV_XEON:
			nAccs = 2;	//EVIL
			break;

		default:
		case DEV_CPU:
			nAccs = 0;
			break;
	}

	if (dev != DEV_CPU)
	{
		MPI_Barrier(MPI_COMM_WORLD);

		if (!nAccs)
		{
			printf ("Error: There are no visible accelerators");
			return 0;
		}

		trackAlloc((void **) &allHosts, sizeof(char)*HOST_NAME_MAX*size);

		MPI_Allgather(hostname, HOST_NAME_MAX, MPI_CHAR, allHosts, HOST_NAME_MAX, MPI_CHAR, MPI_COMM_WORLD);

		idxAcc = 0;

		printf("Rank %d got %d accelerators\n", rank, nAccs);
		fflush(stdout);

		for (int i=0; i<rank; i++)
		{
			if (!strncmp(hostname, &allHosts[HOST_NAME_MAX*i], HOST_NAME_MAX))
				idxAcc++;
		}

		printf("Rank %d got accid %d\n", rank, idxAcc);
		fflush(stdout);

		trackFree((void **) &allHosts, ALLOC_TRACK);

#ifdef	USE_GPU
		if (dev == DEV_GPU)
		{
			cudaSetDevice(idxAcc);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

			cudaDeviceProp gpuProp;

			cudaGetDeviceProperties(&gpuProp, idxAcc);

			printf("  Peak Memory Bandwidth of Gpu %d (GB/s): %f\n\n", idxAcc, 2.0*gpuProp.memoryClockRate*(gpuProp.memoryBusWidth/8)/1.0e6);
			gpuMem = gpuProp.totalGlobalMem;
		}
#endif
		printf ("Rank %d reporting from host %s: Found %d accelerators, using accelerator %d\n\n", rank, hostname, nAccs, idxAcc);
	}

#ifdef	USE_XEON
	if (dev == DEV_XEON)
	{
		int nprocs, nthreads;

		#pragma offload target(mic)
		#pragma omp parallel
		{
			nprocs = omp_get_num_procs();
			nthreads = omp_get_num_threads();
		}

		printf ("Rank %d Xeon Phi will use %d threads for %d processors\n", rank, nthreads, nprocs);
	}
#endif

	if (dev == DEV_CPU)
	{
		int nprocs, nthreads, mthreads;

		#pragma omp parallel
		{
			nprocs = omp_get_num_procs();
			nthreads = omp_get_num_threads();
			mthreads = omp_get_max_threads();
		}

		printf ("Rank %d Cpu will use %d threads for %d processors (max %d)\n", rank, nthreads, nprocs, mthreads);
	}

	return nAccs;
}

void	endComms()
{
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}
