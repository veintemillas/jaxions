#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <climits>

#include "enum-field.h"
#include "utils/memAlloc.h"
#include "utils/logger.h"

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

int	initComms (int argc, char *argv[], int size, DeviceType dev, VerbosityLevel verb)
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

	createLogger	(0, ZERO_RANK, verb);

	switch (dev)
	{
		case DEV_GPU:
		{
#ifdef	USE_GPU
			cudaError_t cErr = cudaGetDeviceCount(&nAccs);

			if (cErr != cudaSuccess)
			{
				LogError ("Rank %d CUDA error (host %s): %s", rank, hostname, cudaGetErrorString(cErr));
				MPI_Finalize();
				return -1;
			}
#else
			LogError ("Gpu support not built");
			exit   (1);
#endif
			break;
		}

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
			LogError ("Error: There are no visible accelerators");
			return 0;
		}

		trackAlloc((void **) &allHosts, sizeof(char)*HOST_NAME_MAX*size);

		MPI_Allgather(hostname, HOST_NAME_MAX, MPI_CHAR, allHosts, HOST_NAME_MAX, MPI_CHAR, MPI_COMM_WORLD);

		idxAcc = 0;

		LogMsg (VERB_NORMAL, "Rank %d got %d accelerators", rank, nAccs);

		for (int i=0; i<rank; i++)
		{
			if (!strncmp(hostname, &allHosts[HOST_NAME_MAX*i], HOST_NAME_MAX))
				idxAcc++;
		}

		LogMsg (VERB_NORMAL, "Rank %d got accid %d", rank, idxAcc);

		trackFree((void **) &allHosts, ALLOC_TRACK);

#ifdef	USE_GPU
		if (dev == DEV_GPU)
		{
			cudaSetDevice(idxAcc);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

			cudaDeviceProp gpuProp;

			cudaGetDeviceProperties(&gpuProp, idxAcc);

			LogMsg (VERB_NORMAL, "  Peak Memory Bandwidth of Gpu %d (GB/s): %f", idxAcc, 2.0*gpuProp.memoryClockRate*(gpuProp.memoryBusWidth/8)/1.0e6);
			gpuMem = gpuProp.totalGlobalMem;
		}
#endif
		LogMsg (VERB_NORMAL, "Rank %d reporting from host %s: Found %d accelerators, using accelerator %d", rank, hostname, nAccs, idxAcc);
	}

	if (dev == DEV_CPU)
	{
		int nprocs, nthreads, mthreads;

		#pragma omp parallel
		{
			nprocs = omp_get_num_procs();
			nthreads = omp_get_num_threads();
			mthreads = omp_get_max_threads();
		}

		LogMsg (VERB_NORMAL, "Rank %d Cpu will use %d threads for %d processors (max %d)", rank, nthreads, nprocs, mthreads);
	}

	return nAccs;
}

void	endComms()
{
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}
