#pragma once

namespace indexHelper
{

static __host__ __device__ inline void idx2Vec(uint idx, uint x[3], const uint Lx, const uint Ly)
{
    const uint tmp = idx / Lx;   // y + Ly*z
    const uint z   = tmp / Ly;

    x[2] = z;
    x[1] = tmp - z * Ly;
    x[0] = idx - tmp * Lx;
}

	static __host__ __device__ inline void idx2Vec(uint idx, uint x[3], const uint Lx)
{
	idx2Vec(idx, x, Lx, Lx);
}

static __host__ __device__ inline uint vec2Idx(uint x[3], const uint Lx, const uint Ly)
{
    return x[0] + Lx * (x[1] + Ly * x[2]);
}

static __host__ __device__ inline uint vec2Idx(uint x[3], const uint Lx)
{
	    return vec2Idx(x, Lx, Lx);
}

}
