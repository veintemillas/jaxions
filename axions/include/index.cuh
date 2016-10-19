#pragma once

namespace indexHelper
{
	static __host__ __device__ inline void idx2Vec(uint idx, uint x[3], const uint Lx)
	{
		uint tmp = idx/Lx;

		x[2] = tmp/Lx;
		x[1] = tmp - x[2]*Lx;
		x[0] = idx - tmp*Lx;
	}

	static __host__ __device__ inline uint vec2Idx(uint x[3], const uint Lx)
	{
		return (x[0] + Lx*(x[1] + Lx*x[2]));
	}
}
