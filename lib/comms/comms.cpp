#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <climits>

#include "enum-field.h"
#include "utils/memAlloc.h"
#include "utils/logger.h"
#include "utils/misc.h"

#ifdef	USE_GPU
	#include <cuda.h>
	#include <cuda_runtime.h>
#endif

#include <omp.h>
#include <sched.h>

#ifndef __USE_POSIX
	#define HOST_NAME_MAX   128
#endif

// TODO Compute ranks per processor with hwloc (total cache size)

static int rank          =  0;
static int ranksPerNode  =  0;
static int nThreads      =  1;
static int idxAcc        = -1;
static int commSz        =  0;
static char hostname[HOST_NAME_MAX];

static int tPerBlock	 = 0;
static int maxThreads[3] = { 0, 0, 0 };
static int maxGrid[3]    = { 0, 0, 0 };

static size_t gpuMem     = 0;

int	commRank()
{
	return rank;
}

int	commRanksPerNode()
{
	return ranksPerNode;
}

int	commSize()
{
	return commSz;
}

int	commThreads()
{
	return nThreads;
}

int	commAcc()
{
	return idxAcc;
}

int	maxThreadsPerBlock() {
	return	tPerBlock;
}

int	maxThreadsPerDim(const int dim) {
	return	(dim < 3) ? maxThreads[dim] : 0;
}

int	maxGridSize(const int dim) {
	return	(dim < 3) ? maxGrid[dim] : 0;
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

int	initComms (int argc, char *argv[], int size, DeviceType dev, LogMpi logMpi, VerbosityLevel verb)
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

	createLogger	(0, logMpi, verb);

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
			endAxions();
			exit   (1);
#endif
			break;
		}

		default:
		case DEV_CPU:
			nAccs = 0;
			break;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (!nAccs & (dev == DEV_GPU))
	{
		LogError ("Error: There are no visible accelerators");
		endAxions();
		return 0;
	}

	trackAlloc((void **) &allHosts, sizeof(char)*HOST_NAME_MAX*size);

	MPI_Allgather(hostname, HOST_NAME_MAX, MPI_CHAR, allHosts, HOST_NAME_MAX, MPI_CHAR, MPI_COMM_WORLD);

#ifdef	USE_GPU
	if (dev == DEV_GPU)
	{
		idxAcc = 0;

		for (int i=0; i<rank; i++) {
			if (!strncmp(hostname, &allHosts[HOST_NAME_MAX*i], HOST_NAME_MAX))
				idxAcc++;
		}

		LogMsg (VERB_NORMAL, "Rank %d got %d accelerators", rank, nAccs);
		LogMsg (VERB_NORMAL, "Rank %d got accid %d", rank, idxAcc);

		cudaSetDevice(idxAcc);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

		cudaDeviceProp gpuProp;

		cudaGetDeviceProperties(&gpuProp, idxAcc);

		LogMsg (VERB_NORMAL, "  Peak Memory Bandwidth of Gpu %d (GB/s): %f", idxAcc, 2.0*gpuProp.memoryClockRate*(gpuProp.memoryBusWidth/8)/1.0e6);
		gpuMem	      = gpuProp.totalGlobalMem;
		tPerBlock     = gpuProp.maxThreadsPerBlock;
		maxThreads[0] = gpuProp.maxThreadsDim[0];
		maxThreads[1] = gpuProp.maxThreadsDim[1];
		maxThreads[2] = gpuProp.maxThreadsDim[2];
		maxGrid[0]    = gpuProp.maxGridSize[0];
		maxGrid[1]    = gpuProp.maxGridSize[1];
		maxGrid[2]    = gpuProp.maxGridSize[2];

		int  accCount  = 0;

		for (int i=0; i<nAccs; i++) {
			if (i==idxAcc)
				continue;

			int canAccessPeerFw, canAccessPeerBw;
			cudaDeviceCanAccessPeer(&canAccessPeerFw, idxAcc, i);
			cudaDeviceCanAccessPeer(&canAccessPeerFw, i, idxAcc);

			if (canAccessPeerFw & canAccessPeerBw)
				accCount++;
		}

		if (accCount == nAccs - 1) {
			for (int i=0; i<nAccs; i++)
				cudaDeviceEnablePeerAccess(i, 0);

			LogMsg (VERB_NORMAL, "Rank %d reporting from host %s: P2P enabled in GPUs", rank, hostname);
		}
	}
	LogMsg (VERB_NORMAL, "Rank %d reporting from host %s: Found %d accelerators, using accelerator %d", rank, hostname, nAccs, idxAcc);
#endif

	for (int i=0; i<size; i++) {
		if (!strncmp(hostname, &allHosts[HOST_NAME_MAX*i], HOST_NAME_MAX))
			ranksPerNode++;
	}

	int nProcs, mThreads;

	#pragma omp parallel
	{
		nProcs   = omp_get_num_procs();
		nThreads = omp_get_num_threads();
		mThreads = omp_get_max_threads();
	}

	LogMsg (VERB_NORMAL, "Rank %d Cpu will use %d threads for %d processors (max %d)", rank, nThreads, nProcs, mThreads);

	trackFree((void **) allHosts);

	return nAccs;
}

void	endComms()
{
	AxionsLog::myLog->stop();
	AxionsLog::myLog->flushLog();
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}
