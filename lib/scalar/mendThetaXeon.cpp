#include <cstdio>
#include <cmath>
#include "scalar/scalarField.h"
#include "enum-field.h"
#include "scalar/varNQCD.h"
#include "utils/parse.h"

#include "utils/triSimd.h"

#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

#include <immintrin.h>

#ifdef  __AVX512F__
	#define Align 64
	#define _PREFIX_ _mm512
#else
	#if not defined(__AVX__) and not defined(__AVX2__)
		#define Align 16
		#define _PREFIX_ _mm
	#else
		#define Align 32
		#define _PREFIX_ _mm256
	#endif
#endif

//----------------------------------------------------------------------
//		CHECK JUMPS
//----------------------------------------------------------------------

//	THIS FUNCTION CHECKS THETA IN ORDER AND NOTES DOWN POSITIONS WITH JUMPS OF 2 PI
//  MARKS THEM DOWN INTO THE ST BIN ARRAY AS POSSIBLE PROBLEMATIC POINTS WITH GRADIENTS
//  TRIES TO MEND THE THETA DISTRIBUTION INTO MANY RIEMMAN SHEETS TO HAVE A CONTINUOUS FIELD

inline  size_t	mendThetaKernelXeon(void * __restrict__ m_, void * __restrict__ v_, const double z, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision)
{
        const size_t Sf = Lx*Lx;
	size_t count = 0;

        if (precision == FIELD_DOUBLE)
        {
#ifdef  __AVX512F__
	#define _MData_ __m512d
	#define step 8
#elif   defined(__AVX__)
	#define _MData_ __m256d
	#define step 4
#else
	#define _MData_ __m128d
	#define step 2
#endif

		double * __restrict__ m = (double * __restrict__) __builtin_assume_aligned (m_, Align);
		double * __restrict__ v = (double * __restrict__) __builtin_assume_aligned (v_, Align);

		const double zP = M_PI*z;

#ifdef  __AVX512F__
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const long long int __attribute__((aligned(Align))) shfLf[8] = { 1, 2, 3, 4, 5, 6, 7, 0 };
#elif   defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);
#endif
		const _MData_ pVec  = opCode(set1_pd, +zP);
		const _MData_ mVec  = opCode(set1_pd, -zP);
		const _MData_ vVec  = opCode(set1_pd, 2.*M_PI);
		const _MData_ cVec  = opCode(set1_pd, zP*2.);

#ifdef  __AVX512F__
                const auto vShLf  = opCode(load_si512, shfLf);
#endif

                #pragma omp parallel default(shared)
                {
                        _MData_ mel, mDf, mDp, mDm, mPx, mPy, mPz, vPx, vPy, vPz;

                        #pragma omp for schedule(static)
                        for (size_t idx = Vo; idx < Vf; idx += step)
                        {
				size_t X[2], idxPx, idxPy, idxPz = idx + Sf, idxVx, idxVy, idxVz = idx;

				{
					size_t tmi = idx/XC, tpi;

				        tpi = tmi/YC;
					X[1] = tmi - tpi*YC;
					X[0] = idx - tmi*XC;
				}

				if (X[0] == XC-step)
					idxPx = idx - XC + step;
				else
					idxPx = idx + step;

				idxVx = idxPx - Sf;

				if (X[1] == YC-1)
				{
					idxPy = idx - Sf + XC;
					idxVy = idxPy - Sf;
#ifdef  __AVX512F__
					mPy = opCode(permutexvar_pd, vShLf, opCode(load_pd, &m[idxPy]));
					vPy = opCode(permutexvar_pd, vShLf, opCode(load_pd, &v[idxVy]));
#elif   defined(__AVX2__)
					mPy = opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_pd, &m[idxPy])), opCode(setr_epi32, 2,3,4,5,6,7,0,1)));
					vPy = opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_pd, &v[idxVy])), opCode(setr_epi32, 2,3,4,5,6,7,0,1)));
#elif   defined(__AVX__)
					mPx = opCode(permute_pd, opCode(load_pd, &m[idxPy]), 0b00000101);
					vPx = opCode(permute_pd, opCode(load_pd, &v[idxVy]), 0b00000101);
					mPz = opCode(permute2f128_pd, mPx, mPx, 0b00000001);
					vPz = opCode(permute2f128_pd, vPx, vPx, 0b00000001);
					mPy = opCode(blend_pd, mPx, mPz, 0b00001010);
					vPy = opCode(blend_pd, vPx, vPz, 0b00001010);
#else
					mPx = opCode(load_pd, &m[idxPy]);
					vPx = opCode(load_pd, &v[idxVy]);
					mPy = opCode(shuffle_pd, mPx, mPx, 0x00000001);
					vPy = opCode(shuffle_pd, vPx, vPx, 0x00000001);
#endif
				} else {
					idxPy = idx + XC;
					idxVy = idxPy - Sf;
					mPy = opCode(load_pd, &m[idxPy]);
					vPy = opCode(load_pd, &v[idxVy]);
				}

				mel = opCode(load_pd, &m[idx]);
				mPx = opCode(load_pd, &m[idxPx]);
				mPz = opCode(load_pd, &v[idxPz]);
				vPx = opCode(load_pd, &m[idxVx]);
				vPz = opCode(load_pd, &v[idxVz]);

				/*	X-Direction	*/

				mDf = opCode(sub_pd, mPx, mel);
#ifdef	__AVX512__
				auto pMask = opCode(cmp_pd_mask, mDf, pVec, _CMP_GE_OQ);
				auto mMask = opCode(cmp_pd_mask, mDf, mVec, _CMP_LT_OQ);

				mPx = opCode(mask_sub_pd, mPx, pMask, mPx, cVec);
				mPx = opCode(mask_add_pd, mPx, mMask, mPx, cVec);
				vPx = opCode(mask_sub_pd, vPx, pMask, vPx, vVec);
				vPx = opCode(mask_add_pd, vPx, mMask, vPx, vVec);

				pMask |= mMask;

				for (int i=1, int k=0; k<step; i<<=1, k++)
					count += (pMask & i) >> k;
#else	// AVX and SSE4.1

#ifdef	__AVX__
				mDp = opCode(cmp_pd, mDf, pVec, _CMP_GE_OQ);
				mDm = opCode(cmp_pd, mDf, mVec, _CMP_LT_OQ);
#else
				mDp = opCode(cmpge_pd, mDf, pVec);
				mDm = opCode(cmplt_pd, mDf, mVec);
#endif
				mPx = opCode(add_pd, mPx,
					opCode(sub_pd,
						opCode(and_pd, mDp, cVec),
						opCode(and_pd, mDm, cVec)));

				vPx = opCode(add_pd, vPx,
					opCode(sub_pd,
						opCode(and_pd, mDp, vVec),
						opCode(and_pd, mDm, vVec)));

				mDp = opCode(or_pd, mDp, mDm);

				for (int k=0; k<step; k++)
					count += reinterpret_cast<size_t&>(mDp[k]) & 1;
#endif	// AVX and SSE4.1

				/*	Y-Direction	*/

				mDf = opCode(sub_pd, mPy, mel);
#ifdef	__AVX512__
				pMask = opCode(cmp_pd_mask, mDf, pVec, _CMP_GE_OQ);
				mMask = opCode(cmp_pd_mask, mDf, mVec, _CMP_LT_OQ);

				mPy = opCode(mask_sub_pd, mPy, pMask, mPy, cVec);
				mPy = opCode(mask_add_pd, mPy, mMask, mPy, cVec);
				vPy = opCode(mask_sub_pd, vPy, pMask, vPy, vVec);
				vPy = opCode(mask_add_pd, vPy, mMask, vPy, vVec);

				pMask |= mMask;

				for (int i=1, int k=0; k<step; i<<=1, k++)
					count += (pMask & i) >> k;
#else	// AVX and SSE4.1

#ifdef	__AVX__
				mDp = opCode(cmp_pd, mDf, pVec, _CMP_GE_OQ);
				mDm = opCode(cmp_pd, mDf, mVec, _CMP_LT_OQ);
#else
				mDp = opCode(cmpge_pd, mDf, pVec);
				mDm = opCode(cmplt_pd, mDf, mVec);
#endif
				mPy = opCode(add_pd, mPy,
					opCode(sub_pd,
						opCode(and_pd, mDp, cVec),
						opCode(and_pd, mDm, cVec)));

				vPy = opCode(add_pd, vPy,
					opCode(sub_pd,
						opCode(and_pd, mDp, vVec),
						opCode(and_pd, mDm, vVec)));

				mDp = opCode(or_pd, mDp, mDm);

				for (int k=0; k<step; k++)
					count += reinterpret_cast<size_t&>(mDp[k]) & 1;
#endif	// AVX and SSE4.1

				/*	Z-Direction	*/

				mDf = opCode(sub_pd, mPz, mel);
#ifdef	__AVX512__
				pMask = opCode(cmp_pd_mask, mDf, pVec, _CMP_GE_OQ);
				mMask = opCode(cmp_pd_mask, mDf, mVec, _CMP_LT_OQ);

				mPz = opCode(mask_sub_pd, mPz, pMask, mPz, cVec);
				mPz = opCode(mask_add_pd, mPz, mMask, mPz, cVec);
				vPz = opCode(mask_sub_pd, vPz, pMask, vPz, vVec);
				vPz = opCode(mask_add_pd, vPz, mMask, vPz, vVec);

				pMask |= mMask;

				for (int i=1, int k=0; k<step; i<<=1, k++)
					count += (pMask & i) >> k;
#else	// AVX and SSE4.1

#ifdef	__AVX__
				mDp = opCode(cmp_pd, mDf, pVec, _CMP_GE_OQ);
				mDm = opCode(cmp_pd, mDf, mVec, _CMP_LT_OQ);
#else
				mDp = opCode(cmpge_pd, mDf, pVec);
				mDm = opCode(cmplt_pd, mDf, mVec);
#endif
				mPz = opCode(add_pd, mPz,
					opCode(sub_pd,
						opCode(and_pd, mDp, cVec),
						opCode(and_pd, mDm, cVec)));

				vPz = opCode(add_pd, vPz,
					opCode(sub_pd,
						opCode(and_pd, mDp, vVec),
						opCode(and_pd, mDm, vVec)));

				mDp = opCode(or_pd, mDp, mDm);

				for (int k=0; k<step; k++)
					count += reinterpret_cast<size_t&>(mDp[k]) & 1;
#endif	// AVX and SSE4.1
				opCode(store_pd, &v[idxVx], vPx);
				opCode(store_pd, &v[idxVy], vPy);
				opCode(store_pd, &v[idxVz], vPz);
				opCode(store_pd, &m[idxPx], mPx);
				opCode(store_pd, &m[idxPy], mPy);
				opCode(store_pd, &m[idxPz], mPz);
			}
		}
#undef  _MData_
#undef  step
	}
	else if (precision == FIELD_SINGLE)
	{
#ifdef  __AVX512F__
	#define _MData_ __m512
	#define step 16
#elif   defined(__AVX__)
	#define _MData_ __m256
	#define step 8
#else
	#define _MData_ __m128
	#define step 4
#endif
		float * __restrict__ m = (float * __restrict__) __builtin_assume_aligned (m_, Align);
		float * __restrict__ v = (float * __restrict__) __builtin_assume_aligned (v_, Align);

		const float zP = M_PI*z;
#ifdef  __AVX512F__
		const size_t XC = (Lx<<4);
		const size_t YC = (Lx>>4);

		const int    __attribute__((aligned(Align))) shfLf[16] = { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0};

		const auto vShLf  = opCode(load_si512, shfLf);
#elif   defined(__AVX__)
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);
#else
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);
#endif
		const _MData_ pVec  = opCode(set1_ps, +zP);
		const _MData_ mVec  = opCode(set1_ps, -zP);
		const _MData_ vVec  = opCode(set1_ps, 2.*M_PI);
		const _MData_ cVec  = opCode(set1_ps, zP*2.);

		#pragma omp parallel default(shared) reduction(+:count)
		{
                        _MData_ mel, mDf, mDp, mDm, mPx, mPy, mPz, vPx, vPy, vPz;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxPx, idxPy, idxPz = idx + Sf, idxVx, idxVy, idxVz = idx;

				{
					size_t tmi = idx/XC, tpi;

				        tpi = tmi/YC;
					X[1] = tmi - tpi*YC;
					X[0] = idx - tmi*XC;
				}

				if (X[0] == XC-step)
					idxPx = idx - XC + step;
				else
					idxPx = idx + step;

				idxVx = idxPx - Sf;

				if (X[1] == YC-1)
				{
					idxPy = idx - Sf + XC;
					idxVy = idxPy - Sf;
#ifdef  __AVX512F__
					mPy = opCode(permutexvar_ps, vShLf, opCode(load_ps, &m[idxPy]));
					vPy = opCode(permutexvar_ps, vShLf, opCode(load_ps, &v[idxVy]));
#elif   defined(__AVX2__)
					mPy = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 1,2,3,4,5,6,7,0));
					vPy = opCode(permutevar8x32_ps, opCode(load_ps, &v[idxVy]), opCode(setr_epi32, 1,2,3,4,5,6,7,0));
#elif   defined(__AVX__)
					mPx = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b00111001);
					vPx = opCode(permute_ps, opCode(load_ps, &v[idxVy]), 0b00111001);
					mPz = opCode(permute2f128_ps, mPx, mPx, 0b00000001);
					vPz = opCode(permute2f128_ps, vPx, vPx, 0b00000001);
					mPy = opCode(blend_ps, mPx, mPz, 0b10001000);
					vPy = opCode(blend_ps, vPx, vPz, 0b10001000);
#else
					mPx = opCode(load_ps, &m[idxPy]);
					vPx = opCode(load_ps, &m[idxVy]);
					mPy = opCode(shuffle_ps, mPx, mPx, 0b00111001);
					vPy = opCode(shuffle_ps, vPx, vPx, 0b00111001);
#endif
				}
				else
				{
					idxPy = idx + XC;
					idxVy = idxPy - Sf;
					mPy = opCode(load_ps, &m[idxPy]);
					vPy = opCode(load_ps, &m[idxVy]);
				}

				mel = opCode(load_ps, &m[idx]);
				mPx = opCode(load_ps, &m[idxPx]);
				mPz = opCode(load_ps, &v[idxPz]);
				vPx = opCode(load_ps, &m[idxVx]);
				vPz = opCode(load_ps, &v[idxVz]);

				/*	X-Direction	*/

				mDf = opCode(sub_ps, mPx, mel);
#ifdef	__AVX512__
				auto pMask = opCode(cmp_pd_mask, mDf, pVec, _CMP_GE_OQ);
				auto mMask = opCode(cmp_pd_mask, mDf, mVec, _CMP_LT_OQ);

				mPx = opCode(mask_sub_ps, mPx, pMask, mPx, cVec);
				mPx = opCode(mask_add_ps, mPx, mMask, mPx, cVec);
				vPx = opCode(mask_sub_ps, vPx, pMask, vPx, vVec);
				vPx = opCode(mask_add_ps, vPx, mMask, vPx, vVec);

				pMask |= mMask;

				for (int i=1, int k=0; k<step; i<<=1, k++)
					count += (pMask & i) >> k;
#else	// AVX and SSE4.1

#ifdef	__AVX__
				mDp = opCode(cmp_ps, mDf, pVec, _CMP_GE_OQ);
				mDm = opCode(cmp_ps, mDf, mVec, _CMP_LT_OQ);
#else
				mDp = opCode(cmpge_ps, mDf, pVec);
				mDm = opCode(cmplt_ps, mDf, mVec);
#endif
				mPx = opCode(add_ps, mPx,
					opCode(sub_ps,
						opCode(and_ps, mDp, cVec),
						opCode(and_ps, mDm, cVec)));

				vPx = opCode(add_ps, vPx,
					opCode(sub_ps,
						opCode(and_ps, mDp, vVec),
						opCode(and_ps, mDm, vVec)));

				mDp = opCode(or_ps, mDp, mDm);

				for (int k=0; k<step; k++)
					count += reinterpret_cast<int&>(mDp[k]) & 1;
#endif	// AVX and SSE4.1

				/*	Y-Direction	*/

				mDf = opCode(sub_ps, mPy, mel);
#ifdef	__AVX512__
				pMask = opCode(cmp_ps_mask, mDf, pVec, _CMP_GE_OQ);
				mMask = opCode(cmp_ps_mask, mDf, mVec, _CMP_LT_OQ);

				mPy = opCode(mask_sub_ps, mPy, pMask, mPy, cVec);
				mPy = opCode(mask_add_ps, mPy, mMask, mPy, cVec);
				vPy = opCode(mask_sub_ps, vPy, pMask, vPy, vVec);
				vPy = opCode(mask_add_ps, vPy, mMask, vPy, vVec);

				pMask |= mMask;

				for (int i=1, int k=0; k<step; i<<=1, k++)
					count += (pMask & i) >> k;
#else	// AVX and SSE4.1

#ifdef	__AVX__
				mDp = opCode(cmp_ps, mDf, pVec, _CMP_GE_OQ);
				mDm = opCode(cmp_ps, mDf, mVec, _CMP_LT_OQ);
#else
				mDp = opCode(cmpge_ps, mDf, pVec);
				mDm = opCode(cmplt_ps, mDf, mVec);
#endif
				mPy = opCode(add_ps, mPy,
					opCode(sub_ps,
						opCode(and_ps, mDp, cVec),
						opCode(and_ps, mDm, cVec)));

				vPy = opCode(add_ps, vPy,
					opCode(sub_ps,
						opCode(and_ps, mDp, vVec),
						opCode(and_ps, mDm, vVec)));

				mDp = opCode(or_ps, mDp, mDm);

				for (int k=0; k<step; k++)
					count += reinterpret_cast<int&>(mDp[k]) & 1;
#endif	// AVX and SSE4.1

				/*	Z-Direction	*/

				mDf = opCode(sub_ps, mPz, mel);
#ifdef	__AVX512__
				pMask = opCode(cmp_ps_mask, mDf, pVec, _CMP_GE_OQ);
				mMask = opCode(cmp_ps_mask, mDf, mVec, _CMP_LT_OQ);

				mPz = opCode(mask_sub_ps, mPz, pMask, mPz, cVec);
				mPz = opCode(mask_add_ps, mPz, mMask, mPz, cVec);
				vPz = opCode(mask_sub_ps, vPz, pMask, vPz, vVec);
				vPz = opCode(mask_add_ps, vPz, mMask, vPz, vVec);

				pMask |= mMask;

				for (int i=1, int k=0; k<step; i<<=1, k++)
					count += (pMask & i) >> k;
#else	// AVX and SSE4.1

#ifdef	__AVX__
				mDp = opCode(cmp_ps, mDf, pVec, _CMP_GE_OQ);
				mDm = opCode(cmp_ps, mDf, mVec, _CMP_LT_OQ);
#else
				mDp = opCode(cmpge_ps, mDf, pVec);
				mDm = opCode(cmplt_ps, mDf, mVec);
#endif
				mPz = opCode(add_ps, mPz,
					opCode(sub_ps,
						opCode(and_ps, mDp, cVec),
						opCode(and_ps, mDm, cVec)));

				vPz = opCode(add_ps, vPz,
					opCode(sub_ps,
						opCode(and_ps, mDp, vVec),
						opCode(and_ps, mDm, vVec)));

				mDp = opCode(or_ps, mDp, mDm);

				for (int k=0; k<step; k++)
					count += reinterpret_cast<int&>(mDp[k]) & 1;
#endif	// AVX and SSE4.1
				opCode(store_ps, &v[idxVx], vPx);
				opCode(store_ps, &v[idxVy], vPy);
				opCode(store_ps, &v[idxVz], vPz);
				opCode(store_ps, &m[idxPx], mPx);
				opCode(store_ps, &m[idxPy], mPy);
				opCode(store_ps, &m[idxPz], mPz);
			}
		}
#undef  _MData_
#undef  step
	}
}

template<typename Float>
inline  size_t	mendThetaSingle(Float * __restrict__ m, Float * __restrict__ v, const double z, const size_t Lx, const size_t Sf, const size_t Vo, const size_t Vf, const int step)
{
	const double zP = M_PI*z;

	Float mDf, mel[step], mPx[step], vPx[step], mPy[step], vPy[step], mPz[step], vPz[step];

	int shf = 0, cnt = step;

	while (cnt != 1) {
		cnt >>= 1;
		shf++;
	}
		
	const size_t XC = (Lx<<shf);
	const size_t YC = (Lx>>shf);

	for (size_t idx = Vo; idx < Vf; idx += step)
	{
		memcpy (&mel[0], &m[idx], step*sizeof(Float));

		size_t X[2], idxPx, idxPy, idxPz = idx + Sf, idxVx, idxVy, idxVz = idx;

		{
			size_t tmi = idx/XC, tpi;

		        tpi = tmi/YC;
			X[1] = tmi - tpi*YC;
			X[0] = idx - tmi*XC;
		}

		if (X[0] == XC-step)
			idxPx = idx - XC + step;
		else
			idxPx = idx + step;

		idxVx = idxPx - Sf;

		if (X[1] == YC-1)
		{
			idxPy = idx - Sf + XC;
			idxVy = idxPy - Sf;

			memcpy (&mPy[0], &m[idxPy], step*sizeof(Float));
			memcpy (&vPy[0], &v[idxVy], step*sizeof(Float));

			Float mSave = mPy[0];
			Float vSave = vPy[0];

			for (int i = 0; i < step-1; i++) {
				mPy[i] = mPy[i+1];
				vPy[i] = vPy[i+1];
			}

			mPy[step-1] = mSave;
			vPy[step-1] = vSave;
		} else {
			idxPy = idx + XC;
			idxVy = idxPy - Sf;

			memcpy (&mPy[0], &m[idxPy], step*sizeof(Float));
			memcpy (&vPy[0], &v[idxVy], step*sizeof(Float));
		}

		memcpy (&mPx[0], &m[idxPx], step*sizeof(Float));
		memcpy (&vPx[0], &v[idxVx], step*sizeof(Float));
		memcpy (&mPz[0], &m[idxPz], step*sizeof(Float));
		memcpy (&vPz[0], &v[idxVz], step*sizeof(Float));

		/*	Vector loop	*/
		for (int i=0; i<step; i++) {

			/*	X-Direction	*/

			mDf = mPx[i] - mel[i];

			if (mDf > zP) {
				mPx[i] -= zP;
				vPx[i] -= 2.*M_PI;
				memcpy (&m[idxPx], &mPx[0], step*sizeof(Float));
				memcpy (&v[idxVx], &vPx[0], step*sizeof(Float));
			} else if (mDf < -zP) {
				mPx[i] += zP;
				vPx[i] += 2.*M_PI;
				memcpy (&m[idxPx], &mPx[0], step*sizeof(Float));
				memcpy (&v[idxVx], &vPx[0], step*sizeof(Float));
			}

			/*	Y-Direction	*/

			mDf = mPy[i] - mel[i];

			if (mDf > zP) {
				mPy[i] -= zP;
				vPy[i] -= 2.*M_PI;
				memcpy (&m[idxPy], &mPy[0], step*sizeof(Float));
				memcpy (&v[idxVy], &vPy[0], step*sizeof(Float));
			} else if (mDf < -zP) {
				mPy[i] += zP;
				vPy[i] += 2.*M_PI;
				memcpy (&m[idxPy], &mPy[0], step*sizeof(Float));
				memcpy (&v[idxVy], &vPy[0], step*sizeof(Float));
			}

			/*	Z-Direction	*/

			mDf = mPz[i] - mel[i];

			if (mDf > zP) {
				mPz[i] -= zP;
				vPz[i] -= 2.*M_PI;
				memcpy (&m[idxPz], &mPz[0], step*sizeof(Float));
				memcpy (&v[idxVz], &vPz[0], step*sizeof(Float));
			} else if (mDf < -zP) {
				mPz[i] += zP;
				vPz[i] += 2.*M_PI;
				memcpy (&m[idxPz], &mPz[0], step*sizeof(Float));
				memcpy (&v[idxVz], &vPz[0], step*sizeof(Float));
			}
		}
	}
}

bool	mendSliceXeon (Scalar *field, size_t slice)
{
	const double z  = *(field->zV());
	const size_t Sf =   field->Surf();

	size_t tJmps = 0;
	bool   wJmp  = false;

	switch (field->Precision()) {
		case FIELD_DOUBLE:

		// Run the first slice single core, no vectorization, until it's uniform
		do {
			field->exchangeGhosts(FIELD_M);
			tJmps = mendThetaSingle<double>(static_cast<double*>(field->mCpu()), static_cast<double*>(field->vCpu()), z, field->Length(), Sf, slice*Sf, (slice+1)*Sf, Align/sizeof(double));

			if (tJmps)
				wJmp = true;
		}	while	(tJmps != 0);

		break;

		case FIELD_SINGLE:

		// Run the first slice single core, no vectorization, until it's uniform
		do {
			field->exchangeGhosts(FIELD_M);
			tJmps = mendThetaSingle<float> (static_cast<float *>(field->mCpu()), static_cast<float *>(field->vCpu()), z, field->Length(), Sf, slice*Sf, (slice+1)*Sf, Align/sizeof(float));

			if (tJmps)
				wJmp = true;
		}	while	(tJmps != 0);

		break;
	}

	return	wJmp;
}

bool	mendThetaXeon (Scalar *field)
{
	const double	z     = *(field->zV());
	size_t		tJmps = 0;
	size_t		cIdx  = 0;
	bool		wJmp  = false;

	wJmp = mendSliceXeon (field, 0);	// Slice 0 belongs to the ghosts, it updates the first usable slice and we won't about the last boundary

	// Parallelize and vectorize over subsequent slices

	for (size_t i = 0; i<field->Depth()-1; i++) {
		cIdx += field->Surf();
	
		do {
			field->exchangeGhosts(FIELD_M);
			tJmps = mendThetaKernelXeon(field->mCpu(), field->vCpu(), z, field->Length(), cIdx, cIdx + field->Surf(), field->Precision());

			if (tJmps)
				wJmp = true;
		}	while	(tJmps != 0);
	}

	return	wJmp;
}

