#include<cstdio>
#include<cstdlib>

#include"enum-field.h"

#ifdef	USE_GPU
	#include<cublas.h>
	#include<cufft.h>

cufftHandle	fftPlan;
cudaStream_t	streamCuFFT;

bool		singPr;

#endif

using namespace std;

int	initCudaFFT	(const int size, const int Lz, FieldPrecision prec)
{
#ifdef	USE_GPU
	int nD[2] = { size, size };
	int Vol   = size*size;

//	cudaStreamCreate        (&streamCuFFT);

	switch (prec)
	{
		case FIELD_DOUBLE:

//		if      (cufftPlanMany(&fftPlan, 2, nD, nD, 1, Vol, nD, 1, Vol, CUFFT_Z2Z, Lz) != CUFFT_SUCCESS)
		{
			printf  ("Error in the FFT!!!\n");
			return 1;
		}
		singPr = false;
		break;

		case FIELD_SINGLE:

//		if      (cufftPlanMany(&fftPlan, 2, nD, nD, 1, Vol, nD, 1, Vol, CUFFT_C2C, Lz) != CUFFT_SUCCESS)
		{
			printf  ("Error in the FFT!!!\n");
			return 1;
		}
		singPr = true;
		break;
	}

//	cufftSetCompatibilityMode       (fftPlan, CUFFT_COMPATIBILITY_NATIVE);
//	cufftSetStream                  (fftPlan, streamCuFFT);

//	if      (cudaDeviceSynchronize() != cudaSuccess)
	{
		printf  ("Error synchronizing!!!\n");
		return 1;
	}

	return  0;
#else
	printf  ("Gpu support not built\n");
	exit	(1);
#endif
}


int	runCudaFFT(void *data, int sign)
{
#ifdef	USE_GPU
	if (singPr)
	{
//		if      (cufftExecC2C(fftPlan, (float2 *) data, (float2 *) data, sign) != CUFFT_SUCCESS)
		{
			printf  ("Error executing FFT!!!\n");
			return 1;
		}
	} else {
//		if      (cufftExecZ2Z(fftPlan, (double2 *) data, (double2 *) data, sign) != CUFFT_SUCCESS)
		{
			printf  ("Error executing FFT!!!\n");
			return 1;
		}
	}
	return	0;
#else
	printf  ("Gpu support not built\n");
	exit	(1);
#endif
}
            
void	closeCudaFFT	()
{
#ifdef	USE_GPU
	printf	("Destroying cuFFT\n");
	fflush	(stdout);
//	cufftDestroy (fftPlan);
//	cudaStreamDestroy (streamCuFFT);
#else
	printf  ("Gpu support not built\n");
	exit	(1);
#endif
}

