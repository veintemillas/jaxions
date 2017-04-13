#include "cub/cub.cuh"

__device__ uint bCount = 0;

template <const int bSize, typename Float, const unsigned int eSize>
__device__ inline void reduction(Float * __restrict__ eRes, const Float * __restrict__ tmp, Float *partial)
{
	typedef cub::BlockReduce<Float, bSize, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduce;
	const int blockSurf = gridDim.x*gridDim.y;

	__shared__ bool isLastBlockDone;
	__shared__ typename BlockReduce::TempStorage cub_tmp[eSize];

	Float tmpA[eSize];

	#pragma unroll
	for (int j=0; j<eSize; j++)
		tmpA[j] = BlockReduce(cub_tmp[j]).Sum(tmp[j]);

	if (threadIdx.x == 0)
	{
		const int bIdx = blockIdx.x + gridDim.x*blockIdx.y;

		#pragma unroll
		for (int j=0; j<eSize; j++)
			partial[bIdx + j*blockSurf] = tmpA[j];

		__threadfence();

		unsigned int cBlock = atomicInc(&bCount, blockSurf);
		isLastBlockDone = (cBlock == (blockSurf-1));
	}

	__syncthreads();

	// finish the reduction if last block
	if (isLastBlockDone)
	{
		uint i = threadIdx.x;

		#pragma unroll
		for (int j=0; j<eSize; j++)
			tmpA[j] = 0;

		while (i < blockSurf)
		{
			#pragma unroll
			for (int j=0; j<eSize; j++)
				tmpA[j] += partial[i + j*blockSurf];

			i += bSize;
		}

		#pragma unroll
		for (int j=0; j<eSize; j++)
			tmpA[j] = BlockReduce(cub_tmp[j]).Sum(tmpA[j]);

		if (threadIdx.x == 0)
		{
			#pragma unroll
			for (int j=0; j<eSize; j++)
				eRes[j] = tmpA[j];

			bCount = 0;
		}
	}
}

