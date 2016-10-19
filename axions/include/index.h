#pragma once

namespace indexXeon
{
#ifdef	USE_XEON
	__attribute__((target(mic)))
#endif
	static inline void idx2Vec(size_t idx, size_t x[3], const size_t Lx)
	{
			size_t tmp = idx/Lx;

			x[2] = tmp/Lx;
			x[1] = tmp - x[2]*Lx;
			x[0] = idx - tmp*Lx;
	}

#ifdef	USE_XEON
	__attribute__((target(mic)))
#endif
	static inline size_t vec2Idx(size_t x[3], const size_t Lx)
	{
		return (x[0] + Lx*(x[1] + Lx*x[2]));
	}
}
