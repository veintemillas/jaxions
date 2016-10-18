#pragma once

namespace indexXeon
{
#ifdef	USE_XEON
	__attribute__((target(mic)))
#endif
	static inline void idx2Vec(uint idx, uint x[3], const uint Lx)
	{
			uint tmp = idx/Lx;

			x[2] = tmp/Lx;
			x[1] = tmp - x[2]*Lx;
			x[0] = idx - tmp*Lx;
	}

#ifdef	USE_XEON
	__attribute__((target(mic)))
#endif
	static inline uint vec2Idx(uint x[3], const uint Lx)
	{
		return (x[0] + Lx*(x[1] + Lx*x[2]));
	}
}
