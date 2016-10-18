#include "kernelParms.cuh"
#include "complexGpu.cuh"
#include "enum-field.h"

template<typename Float>
__global__ void	squareKernel (gpuCu::complex<Float> *m, const int Vol)
{
	int idx = (threadIdx.x + blockIdx.x*blockDim.x) + gridDim.x*blockDim.x*((threadIdx.y + blockIdx.y*blockDim.y) + gridDim.y*blockDim.y*(threadIdx.z + blockIdx.z*blockDim.z));

	if	(idx >= Vol)
		return;

	m[idx] = gpuCu::pow(abs(m[idx]/((Float) Vol)), 2);
}

void	square (void *m, const int len, const int lz, const int Vol, FieldPrecision prec)
{
        dim3    gridSize((len+BSIZE-1)/BSIZE,(len+BSIZE-1)/BSIZE,(lz+BSIZE-1)/BSIZE);
        dim3    blockSize(BSIZE,BSIZE,BSIZE);

	if (prec == FIELD_DOUBLE)
	        squareKernel<<<gridSize,blockSize>>> ((gpuCu::complex<double> *) m, Vol);
	else
	        squareKernel<<<gridSize,blockSize>>> ((gpuCu::complex<float> *) m, Vol);

        cudaDeviceSynchronize();
}
