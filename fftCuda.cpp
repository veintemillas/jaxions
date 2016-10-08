#include<cstdio>
#include<cublas.h>
#include<cufft.h>

using namespace std;

cufftHandle	fftPlan;
cudaStream_t	streamCuFFT;

int	initCudaFFT	(const int size)
{
	cudaStreamCreate        (&streamCuFFT);

	if      (cufftPlan3d(&fftPlan, size, size, size, CUFFT_Z2Z) != CUFFT_SUCCESS)
	{
		printf  ("Error in the FFT!!!\n");
		return 1;
	}

	cufftSetCompatibilityMode       (fftPlan, CUFFT_COMPATIBILITY_NATIVE);
	cufftSetStream                  (fftPlan, streamCuFFT);

	if      (cudaDeviceSynchronize() != cudaSuccess)
	{
		printf  ("Error synchronizing!!!\n");
		return 1;
	}

	return  0;
}


int	runCudaFFT(void *data)
{
	if      (cufftExecZ2Z(fftPlan, (double2 *) data, (double2 *) data, CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		printf  ("Error executing FFT!!!\n");
		return 1;
	}
}
            
void	closeCudaFFT	()
{
	printf	("Destroying cuFFT\n");
	fflush	(stdout);
	cufftDestroy (fftPlan);
	cudaStreamDestroy (streamCuFFT);
}
