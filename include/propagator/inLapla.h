#include <cstdio>
#include <cmath>
#include "scalar/scalarField.h"
#include "enum-field.h"

#include "utils/triSimd.h"

#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

#include <immintrin.h>

#ifdef	__AVX512F__
	#define	Align 64
	#define	_PREFIX_ _mm512
#else
	#if not defined(__AVX__) and not defined(__AVX2__)
		#define	Align 16
		#define	_PREFIX_ _mm
	#else
		#define	Align 32
		#define	_PREFIX_ _mm256
	#endif
#endif

#ifdef	__AVX512F__
	#define	_MData_ __m512d
	#define	step  8
	#define	cStep 4
#elif	defined(__AVX__)
	#define	_MData_ __m256d
	#define	step  4
	#define	cStep 2
#else
	#define	_MData_ __m128d
	#define	step  2
	#define	cStep 1
#endif

template<const bool wMod>
inline	_MData_	lap_pd(const void * __restrict__ m_, size_t idx, size_t NN, double *PC, const double ood2, const size_t Lx, const size_t Sf)
{
	const double * __restrict__ m	= (const double * __restrict__) __builtin_assume_aligned (m_, Align);

	_MData_ COV[5];
	for (size_t nv = 0; nv < NN ; nv++)
		COV[nv]  = opCode(set1_pd, PC[nv]*ood2);

#ifdef	__AVX512F__
	const size_t XC = (Lx<<3);
	const size_t YC = (Lx>>3);

	const long long int __attribute__((aligned(Align))) shfRg[8] = { 7, 0, 1, 2, 3, 4, 5, 6 };
	const long long int __attribute__((aligned(Align))) shfLf[8] = { 1, 2, 3, 4, 5, 6, 7, 0 };
#elif	defined(__AVX__)
	const size_t XC = (Lx<<2);
	const size_t YC = (Lx>>2);
#else
	const size_t XC = (Lx<<1);
	const size_t YC = (Lx>>1);
#endif
	const _MData_ d2Vec  = opCode(set1_pd, ood2);

#ifdef	__AVX512F__
	const auto vShRg  = opCode(load_si512, shfRg);
	const auto vShLf  = opCode(load_si512, shfLf);
#endif

	size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz;
	size_t idx = zC*(YC*XC) + yC*XC + xC;

	if (idx >= Vf)
		continue;

	// FIXME Take xyz as an input, so we don't need to compute them again
	size_t tmi = idx/XC, itp;
	itp = tmi/YC;
	X[1] = tmi - itp*YC;
	X[0] = idx - tmi*XC;

	mel = opCode(load_pd, &m[idx]);
	lap = opCode(set1_pd, 0.0); // for the laplacian

	for (size_t nv=1; nv < NN+1; nv++)
	{
		if (X[0] < nv*step)
			idxMx = ( idx + XC - nv*step );
		else
			idxMx = ( idx - nv*step );
		//x+
		if (X[0] + nv*step >= XC)
			idxPx = ( idx - XC + nv*step );
		else
			idxPx = ( idx + nv*step );

		if (X[1] < nv )
		{
			idxMy = ( idx + Sf - nv*XC );
			idxPy = ( idx + nv*XC );

			mPy = opCode(load_pd, &m[idxPy]);
			#ifdef	__AVX512F__
				mMy = opCode(permutexvar_pd, vShRg, opCode(load_pd, &m[idxMy]));
			#elif	defined(__AVX2__)
				mMy = opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_pd, &m[idxMy])), opCode(setr_epi32, 6,7,0,1,2,3,4,5)));
			#elif	defined(__AVX__)
				acu = opCode(permute_pd, opCode(load_pd, &m[idxMy]), 0b00000101);
				vel = opCode(permute2f128_pd, acu, acu, 0b00000001);
				mMy = opCode(blend_pd, acu, vel, 0b00000101);
			#else
				acu = opCode(load_pd, &m[idxMy]);
				mMy = opCode(shuffle_pd, acu, acu, 0x00000001);
			#endif
		} else {
			idxMy = ( idx - nv*XC );
			mMy = opCode(load_pd, &m[idxMy]);

			if (X[1] + nv >= YC)
			{
				idxPy = ( idx + nv*XC - Sf );
				#ifdef	__AVX512F__
					mPy = opCode(permutexvar_pd, vShLf, opCode(load_pd, &m[idxPy]));
				#elif	defined(__AVX2__)
					mPy = opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_pd, &m[idxPy])), opCode(setr_epi32, 2,3,4,5,6,7,0,1)));
				#elif	defined(__AVX__)
					acu = opCode(permute_pd, opCode(load_pd, &m[idxPy]), 0b00000101);
					vel = opCode(permute2f128_pd, acu, acu, 0b00000001);
					mPy = opCode(blend_pd, acu, vel, 0b00001010);
				#else
					vel = opCode(load_pd, &m[idxPy]);
					mPy = opCode(shuffle_pd, vel, vel, 0x00000001);
				#endif
			} else {
				idxPy = ( idx + nv*XC );
				mPy = opCode(load_pd, &m[idxPy]);
			}
		}

		idxPz = idx+nv*Sf;
		idxMz = idx-nv*Sf;

		if (wMod) {
			/*	idxPx	*/

			vel = opCode(sub_pd, opCode(load_pd, &m[idxPx]), mel);
			acu = opCode(mod_pd, vel, tpVec);

			/*	idxMx	*/

			vel = opCode(sub_pd, opCode(load_pd, &m[idxMx]), mel);
			acu = opCode(add_pd, opCode(mod_pd, vel, tpVec), acu);

			/*	idxPz	*/

			vel = opCode(sub_pd, opCode(load_pd, &m[idxPz]), mel);
			acu = opCode(add_pd, opCode(mod_pd, vel, tpVec), acu);

			/*	idxMz	*/

			vel = opCode(sub_pd, opCode(load_pd, &m[idxMz]), mel);
			acu = opCode(add_pd, opCode(mod_pd, vel, tpVec), acu);

			/*	idxPy	*/

			vel = opCode(sub_pd, mPy, mel);
			acu = opCode(add_pd, opCode(mod_pd, vel, tpVec), acu);

			/*	idxMy	*/

			vel = opCode(sub_pd, mMy, mel);
			acu = opCode(add_pd, opCode(mod_pd, vel, tpVec), acu);

		} else {
			acu = 	opCode(add_pd,
					opCode(add_pd,
						opCode(add_pd, mPy, mMy),
						opCode(add_pd,
							opCode(add_pd, opCode(load_pd, &m[idxPx]), opCode(load_pd, &m[idxMx])),
								opCode(add_pd, opCode(load_pd, &m[idxPz]), opCode(load_pd, &m[idxMz]))
						)
					),
					opCode(mul_pd, mel, opCode(set1_pd, -6.0)));
		}

		lap = opCode(add_pd, lap, opCode(mul_pd,acu,COV[nv-1]));

	} // End neighbour loop

	return	lap;
}

inline	_MData_	slap_pd(const void * __restrict__ m_, size_t idx, size_t NN, double *PC, const double ood2, const size_t Lx, const size_t Sf)
{
	const double * __restrict__ m	= (const double * __restrict__) __builtin_assume_aligned (m_, Align);

	_MData_ COV[5];
	for (size_t nv = 0; nv < NN ; nv++)
		COV[nv]  = opCode(set1_pd, PC[nv]*ood2);

#ifdef	__AVX512F__
	const size_t XC = (Lx<<2);
	const size_t YC = (Lx>>2);

	const long long int __attribute__((aligned(Align))) shfRg[8] = { 6, 7, 0, 1, 2, 3, 4, 5 };
	const long long int __attribute__((aligned(Align))) shfLf[8] = { 2, 3, 4, 5, 6, 7, 0, 1 };
#elif	defined(__AVX__)
	const size_t XC = (Lx<<1);
	const size_t YC = (Lx>>1);
#else
	const size_t XC = Lx;
	const size_t YC = Lx;
#endif
	const _MData_ d2Vec  = opCode(set1_pd, ood2);

#ifdef	__AVX512F__
	const auto vShRg  = opCode(load_si512, shfRg);
	const auto vShLf  = opCode(load_si512, shfLf);
#endif

	size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz;

	// FIXME Take xyz as an input, so we don't need to compute them again
	size_t tmi = idx/XC, itp;
	itp = tmi/YC;
	X[1] = tmi - itp*YC;
	X[0] = idx - tmi*XC;

	idxP0 = idx << 1

	mel = opCode(load_pd, &m[idxP0]);
	lap = opCode(set1_pd, 0.0); // for the laplacian

	for (size_t nv=1; nv < NN+1; nv++)
	{
		if (X[0] < nv*cStep)
			idxMx = (idx + XC - nv*cStep) << 1;
		else
			idxMx = (idx - nv*cStep) << 1;
		//x+
		if (X[0] + nv*cStep >= XC)
			idxPx = (idx - XC + nv*cStep) << 1;
		else
			idxPx = (idx + nv*cStep) << 1;

		if (X[1] < nv )
		{
			idxMy = (idx + Sf - nv*XC) << 1;
			idxPy = (idx + nv*XC) << 1;
			#ifdef	__AVX512F__
				acu = opCode(add_pd, opCode(permutexvar_pd, vShRg, opCode(load_pd, &m[idxMy])), opCode(load_pd, &m[idxPy]));
			#elif	defined(__AVX__)
				mPy = opCode(load_pd, &m[idxMy]);
				acu = opCode(add_pd, opCode(permute2f128_pd, mPy, mPy, 0b00000001), opCode(load_pd, &m[idxPy]));
			#else
				acu = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
			#endif
		} else {
			idxMy = ( idx - nv*XC );
			mMy = opCode(load_pd, &m[idxMy]);

			if (X[1] + nv >= YC)
			{
				idxPy = ((idx + nv*XC - Sf) << 1);
				#ifdef	__AVX512F__
					acu = opCode(add_pd, opCode(permutexvar_pd, vShLf, opCode(load_pd, &m[idxPy])), opCode(load_pd, &m[idxMy]));
				#elif	defined(__AVX__)
					mPy = opCode(load_pd, &m[idxPy]);
					acu = opCode(add_pd, opCode(permute2f128_pd, mPy, mPy, 0b00000001), opCode(load_pd, &m[idxMy]));
				#else
					acu = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
				#endif
			} else {
				idxPy = (idx + nv*XC) << 1;
				acu = opCode(add_pd, opCode(load_pd, &m[idxPy]), opCode(load_pd, &m[idxMy]));
			}
		}

		idxPz = (idx+nv*Sf) << 1;
		idxMz = (idx-nv*Sf) << 1;

		acu = 	opCode(add_pd,
				opCode(add_pd, acu, opCode(add_pd,
								opCode(add_pd, opCode(load_pd, &m[idxPx]), opCode(load_pd, &m[idxMx])),
								opCode(add_pd, opCode(load_pd, &m[idxPz]), opCode(load_pd, &m[idxMz]))
				)
			),
			opCode(mul_pd, mel, opCode(set1_pd, -6.0)));

		lap = opCode(add_pd, lap, opCode(mul_pd,acu,COV[nv-1]));

	} // End neighbour loop

	return	lap;
}

#undef	_MData_
#undef	cStep
#undef	step

#ifdef	__AVX512F__
	#define	_MData_ __m512
	#define	step  16
	#define	cStep 16
#elif	defined(__AVX__)
	#define	_MData_ __m256
	#define	step  8
	#define	cStep 8
#else
	#define	_MData_ __m128
	#define	step  4
	#define	cStep 4
#endif

template<const bool wMod>
inline	_MData_	lap_ps(const void * __restrict__ m_, size_t idx, size_t NN, double *PC, const double ood2, const size_t Lx, const size_t Sf)
{
	const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_, Align);

	_MData_ COV[5];
	for (size_t nv = 0; nv < NN ; nv++)
		COV[nv]  = opCode(set1_ps, PC[nv]*ood2);

#ifdef	__AVX512F__
	const size_t XC = (Lx<<4);
	const size_t YC = (Lx>>4);

	const int    __attribute__((aligned(Align))) shfRg[16] = {15,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14};
	const int    __attribute__((aligned(Align))) shfLf[16] = { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0};

	const auto vShRg  = opCode(load_si512, shfRg);
	const auto vShLf  = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
	const size_t XC = (Lx<<3);
	const size_t YC = (Lx>>3);
#else
	const size_t XC = (Lx<<2);
	const size_t YC = (Lx>>2);
#endif
	size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz;

	size_t tmi = idx/XC, itp;

	itp = tmi/YC;
	X[1] = tmi - itp*YC;
	X[0] = idx - tmi*XC;

	mel = opCode(load_ps, &m[idx]);
	lap = opCode(set1_ps, 0.f); // for the laplacian

	for (size_t nv=1; nv < NN+1; nv++)
	{
		if (X[0] < nv*step)
			idxMx = ( idx + XC - nv*step );
		else
			idxMx = ( idx - nv*step );
		//x+
		if (X[0] + nv*step >= XC)
			idxPx = ( idx - XC + nv*step );
		else
			idxPx = ( idx + nv*step );

		if (X[1] < nv )
		{
			idxMy = ( idx + Sf - nv*XC );
			idxPy = ( idx + nv*XC );

			mPy = opCode(load_ps, &m[idxPy]);
			#ifdef	__AVX512F__
				mMy = opCode(permutexvar_ps, vShRg, opCode(load_ps, &m[idxMy]));
			#elif	defined(__AVX2__)
				mMy = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 7,0,1,2,3,4,5,6));
			#elif	defined(__AVX__)
				tmp = opCode(permute_ps, opCode(load_ps, &m[idxMy]), 0b10010011);
				vel = opCode(permute2f128_ps, tmp, tmp, 0b00000001);
				mMy = opCode(blend_ps, tmp, vel, 0b00010001);
			#else
				tmp = opCode(load_ps, &m[idxMy]);
				mMy = opCode(shuffle_ps, tmp, tmp, 0b10010011);
			#endif
		} else {
			idxMy = ( idx - nv*XC );
			mMy = opCode(load_ps, &m[idxMy]);

			if (X[1] + nv >= YC)
			{
				idxPy = ( idx + nv*XC - Sf );
				#ifdef	__AVX512F__
					mPy = opCode(permutexvar_ps, vShLf, opCode(load_ps, &m[idxPy]));
				#elif	defined(__AVX2__)
					mPy = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 1,2,3,4,5,6,7,0));
				#elif	defined(__AVX__)
					tmp = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b00111001);
					vel = opCode(permute2f128_ps, tmp, tmp, 0b00000001);
					mPy = opCode(blend_ps, tmp, vel, 0b10001000);
				#else
					vel = opCode(load_ps, &m[idxPy]);
					mPy = opCode(shuffle_ps, vel, vel, 0b00111001);
				#endif
			} else {
				idxPy = ( idx + nv*XC );
				mPy = opCode(load_ps, &m[idxPy]);
			}
		}

		idxPz = idx+nv*Sf;
		idxMz = idx-nv*Sf;

		if (wMod) {
			/*	idxPx	*/

			vel = opCode(sub_ps, opCode(load_ps, &m[idxPx]), mel);
			acu = opCode(mod_ps, vel, tpVec);

			/*	idxMx	*/

			vel = opCode(sub_ps, opCode(load_ps, &m[idxMx]), mel);
			acu = opCode(add_ps, opCode(mod_ps, vel, tpVec), acu);

			/*	idxPz	*/

			vel = opCode(sub_ps, opCode(load_ps, &m[idxPz]), mel);
			acu = opCode(add_ps, opCode(mod_ps, vel, tpVec), acu);

			/*	idxMz	*/

			vel = opCode(sub_ps, opCode(load_ps, &m[idxMz]), mel);
			acu = opCode(add_ps, opCode(mod_ps, vel, tpVec), acu);

			/*	idxPy	*/

			vel = opCode(sub_ps, mPy, mel);
			acu = opCode(add_ps, opCode(mod_ps, vel, tpVec), acu);

			/*	idxMy	*/

			vel = opCode(sub_ps, mMy, mel);
			acu = opCode(add_ps, opCode(mod_ps, vel, tpVec), acu);

		} else {
			acu = opCode(add_ps,
				opCode(add_ps,
					opCode(add_ps, mPy, mMy),
					opCode(add_ps,
						opCode(add_ps, opCode(load_ps, &m[idxPx]), opCode(load_ps, &m[idxMx])),
						opCode(add_ps, opCode(load_ps, &m[idxPz]), opCode(load_ps, &m[idxMz]))
					)
				),
				opCode(mul_ps, mel, opCode(set1_ps, -6.f)));
		}

		lap = opCode(add_ps, lap, opCode(mul_ps, acu, COV[nv-1]));

	} // End neighbour loop

	return	lap;
}

inline	_MData_	slap_ps(const void * __restrict__ m_, size_t idx, size_t NN, double *PC, const double ood2, const size_t Lx, const size_t Sf)
{
	const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_, Align);

	_MData_ COV[5];
	for (size_t nv = 0; nv < NN ; nv++)
		COV[nv]  = opCode(set1_ps, PC[nv]*ood2);

#ifdef	__AVX512F__
	const size_t XC = (Lx<<3);
	const size_t YC = (Lx>>3);

	const int    __attribute__((aligned(Align))) shfRg[16] = {14, 15,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13 };
	const int    __attribute__((aligned(Align))) shfLf[16] = { 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1 };

	const auto vShRg  = opCode(load_si512, shfRg);
	const auto vShLf  = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
	const size_t XC = (Lx<<2);
	const size_t YC = (Lx>>2);
#else
	const size_t XC = (Lx<<1);
	const size_t YC = (Lx>>1);
#endif
	size_t X[2], idxP0, idxMx, idxPx, idxMy, idxPy, idxMz, idxPz;

	size_t tmi = idx/XC, itp;

	itp = tmi/YC;
	X[1] = tmi - itp*YC;
	X[0] = idx - tmi*XC;

	idxP0 = idx << 1;

	mel = opCode(load_ps, &m[idxP0]);
	lap = opCode(set1_ps, 0.f); // for the laplacian

	for (size_t nv=1; nv < NN+1; nv++)
	{
		if (X[0] < nv*cStep)
			idxMx = (idx + XC - nv*cStep) << 1;
		else
			idxMx = (idx - nv*cStep) << 1;
		//x+
		if (X[0] + nv*cStep >= XC)
			idxPx = (idx - XC + nv*s\cStep) << 1;
		else
			idxPx = (idx + nv*cStep) << 1;

		if (X[1] < nv)
		{
			idxMy = (idx + Sf - nv*XC) << 1;
			idxPy = (idx + nv*XC) << 1;

			mPy = opCode(load_ps, &m[idxPy]);
			#if     defined(__AVX512F__)
				acu = opCode(add_ps, opCode(permutexvar_ps, vShRg, opCode(load_ps, &m[idxMy])), opCode(load_ps, &m[idxPy]));
			#elif   defined(__AVX2__)
				acu = opCode(add_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 6,7,0,1,2,3,4,5)),  opCode(load_ps, &m[idxPy]));
			#elif   defined(__AVX__)
				mMx = opCode(permute_ps, opCode(load_ps, &m[idxMy]), 0b01001110);
				mPx = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
				acu = opCode(add_ps, opCode(blend_ps, mMx, mPx, 0b00110011), opCode(load_ps, &m[idxPy]));
			#else
				mMx = opCode(load_ps, &m[idxMy]);
				acu = opCode(add_ps, opCode(shuffle_ps, mMx, mMx, 0b01001110), opCode(load_ps, &m[idxPy]));
			#endif
		} else {
			idxMy = (idx - nv*XC) << 1;
			mMy = opCode(load_ps, &m[idxMy]);

			if (X[1] + nv >= YC)
			{
				idxPy = (idx + nv*XC - Sf) << 1;
				#ifdef	__AVX512F__
					acu = opCode(add_ps, opCode(permutexvar_ps, vShLf, opCode(load_ps, &m[idxPy])), opCode(load_ps, &m[idxMy]));
				#elif   defined(__AVX2__)       //AVX2
					acu = opCode(add_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 2,3,4,5,6,7,0,1)), opCode(load_ps, &m[idxMy]));
				#elif   defined(__AVX__)        //AVX
					mMx = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b01001110);
					mPx = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
					acu = opCode(add_ps, opCode(blend_ps, mMx, mPx, 0b11001100), opCode(load_ps, &m[idxMy]));
				#else
                                        mPx = opCode(load_ps, &m[idxPy]);
                                        acu = opCode(add_ps, opCode(shuffle_ps, mPx, mPx, 0b01001110), opCode(load_ps, &m[idxMy]));
				#endif
			} else {
				idxPy = (idx + nv*XC) << 1;
				acu = opCode(load_ps, &m[idxPy], &m[idxMy]);
			}
		}

		idxPz = (idx+nv*Sf) << 1;
		idxMz = (idx-nv*Sf) << 1;

		acu = opCode(add_ps,
			opCode(add_ps, acu,
				opCode(add_ps, mPy, mMy),
				opCode(add_ps,
					opCode(add_ps, opCode(load_ps, &m[idxPx]), opCode(load_ps, &m[idxMx])),
					opCode(add_ps, opCode(load_ps, &m[idxPz]), opCode(load_ps, &m[idxMz]))
				)
			),
			opCode(mul_ps, mel, opCode(set1_ps, -6.f)));
		}

		lap = opCode(add_ps, lap, opCode(mul_ps, acu, COV[nv-1]));

	} // End neighbour loop

	return	lap;
}

#undef	opCode
#undef	opCode_N
#undef	opCode_P
#undef	Align
#undef	_PREFIX_
