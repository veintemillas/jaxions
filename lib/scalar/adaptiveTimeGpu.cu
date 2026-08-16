#include <cuda_runtime.h>
#include "kernelParms.cuh"
#include "complexGpu.cuh"
#include "scalar/adaptiveTimeGpu.h"
#include "cudaErrors.h"

using namespace gpuCu;

template<typename Float>
__global__ void adaptiveLineMaxKernel(const complex<Float> *field,
                                      const complex<Float> *velocity,
                                      size_t length, double *result)
{
    extern __shared__ double work[];
    double *phi2 = work;
    double *vmax = work + blockDim.x;
    double localPhi2 = 0.0, localVmax = 0.0;
    for (size_t i = threadIdx.x; i < length; i += blockDim.x) {
        const double re = field[i].real(), im = field[i].imag();
        localPhi2 = max(localPhi2, re*re + im*im);
        localVmax = max(localVmax, max(abs((double) velocity[i].real()),
                                      abs((double) velocity[i].imag())));
    }
    phi2[threadIdx.x] = localPhi2;
    vmax[threadIdx.x] = localVmax;
    __syncthreads();
    for (unsigned int stride = blockDim.x/2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            phi2[threadIdx.x] = max(phi2[threadIdx.x], phi2[threadIdx.x+stride]);
            vmax[threadIdx.x] = max(vmax[threadIdx.x], vmax[threadIdx.x+stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { result[0] = phi2[0]; result[1] = vmax[0]; }
}

void adaptiveLineMaxGpu(const void *field, const void *velocity, size_t length,
                        bool singlePrecision, void *streamPtr,
                        double &phi2Max, double &velocityMax)
{
    constexpr unsigned int threads = 256;
    static double *deviceResult = nullptr;
    if (deviceResult == nullptr) cudaMalloc(&deviceResult, 2*sizeof(double));
    cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);
    if (singlePrecision)
        adaptiveLineMaxKernel<<<1, threads, 2*threads*sizeof(double), stream>>>(
            static_cast<const complex<float> *>(field),
            static_cast<const complex<float> *>(velocity), length, deviceResult);
    else
        adaptiveLineMaxKernel<<<1, threads, 2*threads*sizeof(double), stream>>>(
            static_cast<const complex<double> *>(field),
            static_cast<const complex<double> *>(velocity), length, deviceResult);
    double hostResult[2];
    cudaMemcpyAsync(hostResult, deviceResult, sizeof(hostResult),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    CudaCheckError();
    phi2Max = hostResult[0]; velocityMax = hostResult[1];
}
