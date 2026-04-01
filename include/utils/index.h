#pragma once

namespace indexXeon
{

	static inline void idx2Vec(size_t idx, size_t x[3], const size_t Lx, const size_t Ly)
	{
	    size_t tmp = idx / Lx;   // y + Ly*z

	    x[2] = tmp / Ly;         // z
	    x[1] = tmp - x[2] * Ly;  // y
	    x[0] = idx - tmp * Lx;   // x
	}

	static inline size_t vec2Idx(size_t x[3], const size_t Lx, const size_t Ly)
	{
	    return x[0] + Lx * (x[1] + Ly * x[2]);
	}

	static inline void idx2VecPad(size_t idx, size_t x[3], const size_t Lx, const size_t Ly)
	{
	    size_t tmp = idx / (Lx + 2); // y + Ly*z

	    x[2] = tmp / Ly;             // z
	    x[1] = tmp - x[2] * Ly;      // y
	    x[0] = idx - tmp * (Lx + 2); // x
	}

	static inline void idx2Vec(size_t idx, size_t x[3], const size_t Lx)
	{
			return	idx2Vec(idx, x, Lx, Lx);
	}

	static inline size_t vec2Idx(size_t x[3], const size_t Lx)
	{
	    return vec2Idx(x, Lx, Lx);
	}

	static inline void idx2VecPad(size_t idx, size_t x[3], const size_t Lx)
	{
			return idx2VecPad(idx, x, Lx, Lx);
	}

	static inline void idx2VecNeigh(size_t idx, size_t X[3], size_t O[4], const size_t Lx, const size_t Ly)
	{
	    size_t tmp = idx / Lx;
	    size_t   S = Lx * Ly;

	    X[2] = tmp / Ly;
	    X[1] = tmp - X[2] * Ly;
	    X[0] = idx - tmp * Lx;

	    // O = iPx, iMx, iPy, iMy
	    if (X[0] == 0) {
	        O[0] = idx + 1;
	        O[1] = idx + Lx - 1;
	    } else {
	        if (X[0] == Lx - 1) {
	            O[0] = idx - Lx + 1;
	            O[1] = idx - 1;
	        } else {
	            O[0] = idx + 1;
	            O[1] = idx - 1;
	        }
	    }

	    if (Ly == 1) {
	        O[2] = idx;
	        O[3] = idx;
	    } else {
	        if (X[1] == 0) {
	            O[2] = idx + Lx;
	            O[3] = idx + S - Lx;
	        } else {
	            if (X[1] == Ly - 1) {
	                O[2] = idx - S + Lx;
	                O[3] = idx - Lx;
	            } else {
	                O[2] = idx + Lx;
	                O[3] = idx - Lx;
	            }
	        }
	    }
	}

	static inline void idx2VecNeigh(size_t idx, size_t X[3], size_t O[4], const size_t Lx)
	{
			idx2VecNeigh(idx, X, O, Lx, Lx);
	}

}
