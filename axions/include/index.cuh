#pragma once

namespace indexHelper
{
	static __host__ __device__ inline void idx2Vec(int idx, int x[3], const int Lx)
	{
		int tmp = idx/Lx;

		x[2] = tmp/Lx;
		x[1] = tmp - x[2]*Lx;
		x[0] = idx - tmp*Lx;
	}

	static __host__ __device__ inline int vec2Idx(int x[3], const int Lx)
	{
		return (x[0] + Lx*(x[1] + Lx*x[2]));
	}
}
