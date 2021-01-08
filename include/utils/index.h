#pragma once

namespace indexXeon
{
	static inline void idx2Vec(size_t idx, size_t x[3], const size_t Lx)
	{
			size_t tmp = idx/Lx;

			x[2] = tmp/Lx;
			x[1] = tmp - x[2]*Lx;
			x[0] = idx - tmp*Lx;
	}

	static inline void idx2VecPad(size_t idx, size_t x[3], const size_t Lx)
	{
		/* idx = x[0] + (Lx+2)(x[1] + Lx*x[2]) */
			size_t tmp = idx/(Lx+2); //  y + Lx z

			x[2] = tmp/Lx;           // z
			x[1] = tmp - x[2]*Lx;    // y
			x[0] = idx - tmp*(Lx+2); // x
	}

	static inline size_t vec2Idx(size_t x[3], const size_t Lx)
	{
		return (x[0] + Lx*(x[1] + Lx*x[2]));
	}

	static inline void idx2VecNeigh(size_t idx, size_t X[3], size_t O[4], const size_t L)
	{
			size_t tmp = idx/L;
			size_t   S = L*L;

			X[2] = tmp/L;
			X[1] = tmp - X[2]*L;
			X[0] = idx - tmp*L;

			// O = iPx, iMx, iPy, iMy
			if (X[0] == 0) {
				O[0] = idx + 1; // iPx
				O[1] = idx + L - 1;
			} else {
				if (X[0] == L - 1) {
					O[0] = idx - L + 1;
					O[1] = idx - 1;
				} else {
					O[0] = idx + 1;
					O[1] = idx - 1;
				}
			}
			if (X[1] == 0) {
				O[2] = idx + L;
				O[3] = idx + S - L;
			} else {
				if (X[1] == L - 1) {
					O[2] = idx - S + L;
					O[3] = idx - L;
				} else {
					O[2] = idx + L;
					O[3] = idx - L;
				}
			}


	}

}
