#include<unistd.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<mpi.h>

static int rank = -1;
static int idxGpu = -1;
static char hostname[HOST_NAME_MAX];

int	commRank()
{
	return rank;
}

int	commGpu()
{
	return idxGpu;
}

char	*commHost()
{
	return hostname;
}

int	initCudaComms (int argc, char *argv[], int size)
{
	int nGpus = 0;
	int realSize = 1;
	char *allHosts;

	MPI_Init (&argc, &argv);
	MPI_Comm_size (MPI_COMM_WORLD, &realSize);

	if (realSize != size)
	{
		printf ("Error: Requested %d processes, got %d. Adjust you command line parameters\n", size, realSize);
		MPI_Finalize();
		return -1;
	}

	gethostname(hostname, HOST_NAME_MAX);
	hostname[HOST_NAME_MAX-1] = '\0';		// gethostname no termina la cadena con \0

	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	MPI_Barrier(MPI_COMM_WORLD);

	cudaError_t cErr = cudaGetDeviceCount(&nGpus);

	if (cErr != cudaSuccess)
	{
		printf("Rank %d CUDA error (host %s):\n", rank, hostname);
		printf("%s\n", cudaGetErrorString(cErr));
		MPI_Finalize();
		return -1;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (!nGpus)
	{
		printf ("Error: There are no visible gpus");
		return 0;
	}

	allHosts = (char *) malloc(sizeof(char)*HOST_NAME_MAX*size);

	MPI_Allgather(hostname, HOST_NAME_MAX, MPI_CHAR, allHosts, HOST_NAME_MAX, MPI_CHAR, MPI_COMM_WORLD);

	idxGpu = 0;

	printf("Rank %d got %d gpus\n", rank, nGpus);
	fflush(stdout);

	for (int i=0; i<rank; i++)
	{
		if (!strncmp(hostname, &allHosts[HOST_NAME_MAX*i], HOST_NAME_MAX))
			idxGpu++;
	}

	printf("Rank %d got gpuid %d\n", rank, idxGpu);
	fflush(stdout);

	free(allHosts);

	cudaSetDevice(idxGpu);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	printf ("Rank %d reporting from host %s: Found %d gpus, using gpu %d\n\n", rank, hostname, nGpus, idxGpu);

	return nGpus;
}

void	endComms()
{
	MPI_Finalize();
}
