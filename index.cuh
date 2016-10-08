#pragma once

namespace indexHelper
{
	static __host__ __device__ __forceinline__ void idx2Vec(int idx, int x[3], const int Lx)
	{
		const int &LX = Lx;

		x[0] = idx%LX;
		x[2] = idx/(LX*LX);
		x[1] = (idx - x[2]*(LX*LX))/LX;
	}

	static __host__ __device__ __forceinline__ int vec2Idx(int x[3], const int Lx)
	{
		return (x[0] + x[1]*Lx + x[2]*(Lx*Lx));
	}
}
