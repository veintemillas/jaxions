#include<cstdio>
#include<cmath>
#include"scalarField.h"
#include"enum-field.h"

#ifdef USE_XEON
	#include"comms.h"
	#include"xeonDefs.h"
#endif


#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

#include <immintrin.h>


#ifdef	__MIC__
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

// FUSIONA OPERACIONES

#ifdef	__MIC__
#ifdef USE_XEON
__attribute__((target(mic)))
#endif
inline	void	stringHandD(const __m512d s1, const __m512d s2, int *hand)
{
	__m512d zero = { 0., 0., 0., 0., 0., 0., 0., 0. };
	__m512d conj = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
	__m512d tp2, tp3;
	__mmask16 tpm, tmm, tmp, str;

	tp2 = opCode(mul_pd, s1, s2);
	tmp = opCode(cmp_pd_mask, tp2, zero, _CMP_LT_OS);

	tmp &= 0b1010101010101010;

	tp3 = opCode(mul_pd, s2, conj);
	tp2 = opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tp3), _MM_PERM_BADC));
	tp3 = opCode(mul_pd, s1,  tp2);
	tp2 = opCode(add_pd, tp3, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tp3), _MM_PERM_BADC)));
	tpm = opCode(cmp_pd_mask, tp2, zero, _CMP_GT_OS);
	tmm = opCode(cmp_pd_mask, tp2, zero, _CMP_LE_OS);

	tpm &= tmp;
	tmm &= tmp;

	hand[0] += ((tpm &  2) >> 1) - ((tmm &  2) >> 1);
	hand[1] += ((tpm &  8) >> 3) - ((tmm &  8) >> 3);
	hand[2] += ((tpm & 32) >> 5) - ((tmm & 32) >> 5);
	hand[3] += ((tpm &128) >> 7) - ((tmm &128) >> 7);

	return;
}

#ifdef USE_XEON
__attribute__((target(mic)))
#endif
inline	void	stringHandS(const __m512 s1, const __m512 s2, int *hand)
{
	__m512 zero = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };
	__m512 conj = { 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
	__m512 tp2, tp3;
	__mmask16 tpm, tmm, tmp, str;

	tp2 = opCode(mul_ps, s1, s2);
	tmp = opCode(cmp_ps_mask, tp2, zero, _CMP_LT_OS);

	tmp &= 0b1010101010101010;

	tp3 = opCode(mul_ps, s2, conj);
	tp2 = opCode(swizzle_ps, tp3, _MM_SWIZ_REG_CDAB);
	tp3 = opCode(mul_ps, s1,  tp2);
	tp2 = opCode(add_ps, tp3, opCode(swizzle_ps, tp3, _MM_SWIZ_REG_CDAB));
	tpm = opCode(cmp_ps_mask, tp2, zero, _CMP_GT_OS);
	tmm = opCode(cmp_ps_mask, tp2, zero, _CMP_LE_OS);

	tpm &= tmp;
	tmm &= tmp;

	hand[0] += ((tpm &    2) >> 1) - ((tmm &    2) >> 1);
	hand[1] += ((tpm &    8) >> 3) - ((tmm &    8) >> 3);
	hand[2] += ((tpm &   32) >> 5) - ((tmm &   32) >> 5);
	hand[3] += ((tpm &  128) >> 7) - ((tmm &  128) >> 7);
	hand[4] += ((tpm &  512) >> 9) - ((tmm &  512) >> 9);
	hand[5] += ((tpm & 2048) >>11) - ((tmm & 2048) >>11);
	hand[6] += ((tpm & 8192) >>13) - ((tmm & 8192) >>13);
	hand[7] += ((tpm &32768) >>15) - ((tmm &32768) >>15);

	return;
}

#elif defined(__AVX__)
inline	void	stringHandD(const __m256d s1, const __m256d s2, int *hand)
{
	__m256d tp2, tp3;
	__m256i	str, tmp;

	int __attribute__((aligned(Align))) tmpHand[8];

	tp2 = opCode(mul_pd, s1, s2);
	tmp = opCode(castpd_si256, opCode(cmp_pd, tp2, opCode(setzero_pd), _CMP_LT_OS));
	tp3 = opCode(mul_pd, s2, opCode(set_pd,-1., 1.,-1., 1.));
	tp2 = opCode(permute_pd, tp3, 0b00000101);
	tp3 = opCode(mul_pd, s1,  tp2);
	tp2 = opCode(add_pd, tp3, opCode(permute_pd, tp3, 0b00000101));
	tp3 = opCode(cmp_pd, tp2, opCode(setzero_pd), _CMP_GT_OS);
#ifdef __AVX2__
	str = opCode(sub_epi64, opCode(castpd_si256, opCode(and_pd, tp3, opCode(castsi256_pd, opCode(set_epi64x,2,0,2,0)))), opCode(set_epi64x,1,0,1,0));
	str = opCode(and_si256, str, tmp);
#else
	str = opCode(castpd_si256, opCode(or_pd, opCode(and_pd, tp3, opCode(castsi256_pd, opCode(set_epi64x,2,0,2,0))), opCode(castsi256_pd, opCode(set_epi64x,1,0,1,0))));
	str = opCode(castpd_si256, opCode(and_pd, opCode(castsi256_pd, str), opCode(castsi256_pd, tmp)));
#endif
	opCode(store_si256, static_cast<__m256i*>(static_cast<void*>(tmpHand)), str);

#ifdef __AVX2__
	hand[0] += tmpHand[2];
	hand[1] += tmpHand[6];
#else
	hand[0] += (tmpHand[2] & 2) - (tmpHand[2] & 1);
	hand[1] += (tmpHand[6] & 2) - (tmpHand[6] & 1);
#endif

	return;
}

inline	void	stringHandS(const __m256 s1, const __m256 s2, int *hand)
{
	__m256	tp2, tp3;
	__m256i	str, tmp;

	int __attribute__((aligned(Align))) tmpHand[8];

	tp2 = opCode(mul_ps, s1, s2);
	tmp = opCode(castps_si256, opCode(cmp_ps, tp2, opCode(setzero_ps), _CMP_LT_OS));
	tp3 = opCode(mul_ps, s2, opCode(set_ps,-1.f, 1.f,-1.f, 1.f,-1.f, 1.f,-1.f, 1.f));
	tp2 = opCode(permute_ps, tp3, 0b10110001);
	tp3 = opCode(mul_ps, s1, tp2);
	tp2 = opCode(add_ps, tp3, opCode(permute_ps, tp3, 0b10110001));
	tp3 = opCode(cmp_ps, tp2, opCode(setzero_ps), _CMP_GT_OS);
#ifdef __AVX2__
	str = opCode(sub_epi64, opCode(castps_si256, opCode(and_ps, tp3, opCode(castsi256_ps, opCode(set_epi32,2,0,2,0,2,0,2,0)))), opCode(set_epi32,1,0,1,0,1,0,1,0));
	str = opCode(and_si256, str, tmp);
#else
	str = opCode(castps_si256, opCode(or_ps, opCode(and_ps, tp3, opCode(castsi256_ps, opCode(set_epi32,2,0,2,0,2,0,2,0))), opCode(castsi256_ps, opCode(set_epi32,1,0,1,0,1,0,1,0))));
	str = opCode(castps_si256, opCode(and_ps, opCode(castsi256_ps, str), opCode(castsi256_ps, tmp)));
#endif
	opCode(store_si256, static_cast<__m256i*>(static_cast<void*>(tmpHand)), str);

#ifdef __AVX2__
	hand[0] += tmpHand[1];
	hand[1] += tmpHand[3];
	hand[2] += tmpHand[5];
	hand[3] += tmpHand[7];
#else
	hand[0] += (tmpHand[1] & 2) - (tmpHand[1] & 1);
	hand[1] += (tmpHand[3] & 2) - (tmpHand[3] & 1);
	hand[2] += (tmpHand[5] & 2) - (tmpHand[5] & 1);
	hand[3] += (tmpHand[7] & 2) - (tmpHand[7] & 1);
#endif

	return;
}
#else
inline	void	stringHandD(const __m128d s1, const __m128d s2, int *hand)
{
	__m128d	tp2, tp3;
	__m128i str, tmp;

	int __attribute__((aligned(Align))) tmpHand[4];

	tp2 = opCode(mul_pd, s1, s2);
	tmp = opCode(castpd_si128, opCode(cmplt_pd, tp2, opCode(setzero_pd)));
	tp3 = opCode(mul_pd, s2, opCode(set_pd,-1., 1.));
	tp2 = opCode(shuffle_pd, tp3, tp3, 0b00000001);
	tp3 = opCode(mul_pd, s1, tp2);
	tp2 = opCode(add_pd, tp3, opCode(shuffle_pd, tp3, tp3, 0b00000001));
	tp3 = opCode(cmpgt_pd, tp2, opCode(setzero_pd));
	str = opCode(sub_epi64, opCode(castpd_si128, opCode(and_pd, tp3, opCode(castsi128_pd, opCode(set_epi64x, 2, 0)))), opCode(set_epi64x, 1, 0));
	str = opCode(and_si128, str, tmp);
	opCode(store_si128, static_cast<__m128i*>(static_cast<void*>(tmpHand)), str);
	*hand += tmpHand[2];

	return;
}

inline	void	stringHandS(const __m128 s1, const __m128 s2, int *hand)
{
	__m128	tp2, tp3;
	__m128i	str, tmp;

	int __attribute__((aligned(Align))) tmpHand[4];

	tp2 = opCode(mul_ps, s1, s2);
	tmp = opCode(castps_si128, opCode(cmplt_ps, tp2, opCode(setzero_ps)));
	tp3 = opCode(mul_ps, s2, opCode(set_ps,-1.f, 1.f,-1.f, 1.f));
	tp2 = opCode(shuffle_ps, tp3, tp3, 0b10110001);
	tp3 = opCode(mul_ps, s1, tp2);
	tp2 = opCode(add_ps, tp3, opCode(shuffle_ps, tp3, tp3, 0b10110001));
	tp3 = opCode(cmpgt_ps, tp2, opCode(setzero_ps));
	str = opCode(sub_epi64, opCode(castps_si128, opCode(and_ps, tp3, opCode(castsi128_ps, opCode(set_epi32,2,0,2,0)))), opCode(set_epi32,1,0,1,0));
	str = opCode(and_si128, str, tmp);
	opCode(store_si128, static_cast<__m128i*>(static_cast<void*>(tmpHand)), str);
	hand[0] += tmpHand[1];
	hand[1] += tmpHand[3];

	return;
}
#endif

#ifdef USE_XEON
__attribute__((target(mic)))
#endif
double	stringKernelXeon(const void * __restrict__ m_, const int Lx, const int Vo, const int Vf, FieldPrecision precision, void * __restrict__ strg)
{
	const size_t	Sf = Lx*Lx;
	size_t		nStrings = 0;
	long long int	nChiral = 0;

	if (precision == FIELD_DOUBLE)
	{
#ifdef	__MIC__
	#define	_MData_ __m512d
	#define	step 4
#elif	defined(__AVX__)
	#define	_MData_ __m256d
	#define	step 2
#else
	#define	_MData_ __m128d
	#define	step 1
#endif

#ifdef	USE_XEON
		const double * __restrict__ m	= (const double * __restrict__) m_;
		__assume_aligned(m, Align);
#else
		const double * __restrict__ m	= (const double * __restrict__) __builtin_assume_aligned (m_, Align);
#endif

#ifdef	__MIC__
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		int hand[4] = { 0, 0, 0, 0 };
#elif	defined(__AVX__)
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		int hand[2] = { 0, 0 };
#else
		const size_t XC = Lx;
		const size_t YC = Lx;

		int hand[1] = { 0 };
#endif
		#pragma omp parallel default(shared) private(hand) reduction(+:nStrings,nChiral)
		{
			_MData_ mel, mPx, mPy, mPz, mXY, mYZ, mZX;
			_MData_ str, tmp;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[3], idxPx, idxPy, idxPz, idxXY, idxYZ, idxZX, idxP0, idxMz;

				{
					size_t tmi = idx/XC;

					X[2] = tmi/YC;
					X[1] = tmi - X[2]*YC;
					X[0] = idx - tmi*XC;
				}

				idxP0 = (idx << 1);
				idxPz = ((idx + Sf) << 1);
				idxMz = ((idx - Sf) >> 1);

				if (X[1] == YC-1)
				{
					idxPy = ((idx - Sf + XC) << 1);
					idxYZ = ((idx + XC) << 1);

					if (X[0] == XC-step)
					{
						idxPx = ((idx - XC + step) << 1);
						idxXY = ((idx - Sf + step) << 1);
						idxZX = ((idx + Sf - XC + step) << 1);
					}
					else
					{
						idxPx = ((idx + step) << 1);
						idxXY = ((idx - Sf + XC + step) << 1);
						idxZX = ((idx + Sf + step) << 1);
					}

					mel = opCode(load_pd, &m[idxP0]);
					mPx = opCode(load_pd, &m[idxPx]);
					mPz = opCode(load_pd, &m[idxPz]);
					mZX = opCode(load_pd, &m[idxZX]);
#ifdef	__MIC__
					mPy = opCode(castps_pd, opCode(permute4f128_ps, opCode(castpd_ps, opCode(load_pd, &m[idxPy])), _MM_PERM_ADCB));
					mXY = opCode(castps_pd, opCode(permute4f128_ps, opCode(castpd_ps, opCode(load_pd, &m[idxXY])), _MM_PERM_ADCB));
					mYZ = opCode(castps_pd, opCode(permute4f128_ps, opCode(castpd_ps, opCode(load_pd, &m[idxYZ])), _MM_PERM_ADCB));
#elif	defined(__AVX__)
					tmp = opCode(load_pd, &m[idxPy]);
					str = opCode(load_pd, &m[idxXY]);
					mPy = opCode(load_pd, &m[idxYZ]);
					mYZ = opCode(permute2f128_pd, mPy, mPy, 0b00000001);
					mXY = opCode(permute2f128_pd, str, str, 0b00000001);
					mPy = opCode(permute2f128_pd, tmp, tmp, 0b00000001);
#else
					mPy = opCode(load_pd, &m[idxPy]);
					mXY = opCode(load_pd, &m[idxXY]);
					mYZ = opCode(load_pd, &m[idxYZ]);
#endif
				}
				else
				{
					idxPy = ((idx + XC) << 1);
					idxYZ = ((idx + Sf + XC) << 1);

					if (X[0] == XC-step)
					{
						idxPx = ((idx - XC + step) << 1);
						idxXY = ((idx + step) << 1);
						idxZX = ((idx + Sf - XC + step) << 1);
					}
					else
					{
						idxPx = ((idx + step) << 1);
						idxXY = ((idx + XC + step) << 1);
						idxZX = ((idx + Sf + step) << 1);
					}

					mel = opCode(load_pd, &m[idxP0]);
					mPx = opCode(load_pd, &m[idxPx]);
					mPz = opCode(load_pd, &m[idxPz]);
					mZX = opCode(load_pd, &m[idxZX]);
					mPy = opCode(load_pd, &m[idxPy]);
					mXY = opCode(load_pd, &m[idxXY]);
					mYZ = opCode(load_pd, &m[idxYZ]);
				}

				// Tienes los 7 puntos que definen las 3 plaquetas

				size_t nIdx = (X[0]/step + X[1]*Lx + (X[2]-1)*Sf);

				// Plaqueta XY

				stringHandD (mel, mPx, hand);
				stringHandD (mPx, mXY, hand);
				stringHandD (mXY, mPy, hand);
				stringHandD (mPy, mel, hand);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					size_t tIdx = nIdx + ih*YC*Lx;

					switch (hand[ih])
					{
						case 2:
						{
							static_cast<char *>(strg)[tIdx] |= STRING_XY_POSITIVE;
							nStrings++;
							nChiral++;
							//printf ("Positive string %d %d %d, 0\n", X[0]/step, X[1]+ih*YC, X[2]-1);
							//fflush (stdout);
						}
						break;

						case -2:
						{
							static_cast<char *>(strg)[tIdx] |= STRING_XY_NEGATIVE;
							nStrings++;
							nChiral--;
							//printf ("Negative string %d %d %d, 0\n", X[0]/step, X[1]+ih*YC, X[2]-1);
							//fflush (stdout);
						}
						break;

						default:
						break;
					}
	
					hand[ih] = 0;
				}

				// Plaqueta YZ

				stringHandD (mel, mPy, hand);
				stringHandD (mPy, mYZ, hand);
				stringHandD (mYZ, mPz, hand);
				stringHandD (mPz, mel, hand);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					size_t tIdx = nIdx + ih*YC*Lx;

					switch (hand[ih])
					{
						case 2:
						{
							static_cast<char *>(strg)[tIdx] |= STRING_YZ_POSITIVE;
							nStrings++;
							nChiral++;
							//printf ("Positive string %d %d %d, 1\n", X[0]/step, X[1]+ih*YC, X[2]-1);
							//fflush (stdout);
						}
						break;

						case -2:
						{
							static_cast<char *>(strg)[tIdx] |= STRING_YZ_NEGATIVE;
							nStrings++;
							nChiral--;
							//printf ("Negative string %d %d %d, 1\n", X[0]/step, X[1]+ih*YC, X[2]-1);
							//fflush (stdout);
						}
						break;

						default:
						break;
					}

					hand[ih] = 0;
				}

				// Plaqueta ZX

				stringHandD (mel, mPz, hand);
				stringHandD (mPz, mZX, hand);
				stringHandD (mZX, mPx, hand);
				stringHandD (mPx, mel, hand);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					size_t tIdx = nIdx + ih*YC*Lx;

					switch (hand[ih])
					{
						case 2:
						{
							static_cast<char *>(strg)[tIdx] |= STRING_ZX_POSITIVE;
							nStrings++;
							nChiral++;
							//printf ("Positive string %d %d %d, 2\n", X[0]/step, X[1]+ih*YC, X[2]-1);
							//fflush (stdout);
						}
						break;

						case -2:
						{
							static_cast<char *>(strg)[tIdx] |= STRING_ZX_NEGATIVE;
							nStrings++;
							nChiral--;
							//printf ("Negative string %d %d %d, 2\n", X[0]/step, X[1]+ih*YC, X[2]-1);
							//fflush (stdout);
						}
						break;

						default:
						break;
					}

					hand[ih] = 0;
				}
			}
		}

#undef	_MData_
#undef	step
	}
	else if (precision == FIELD_SINGLE)
	{
#ifdef	__MIC__
	#define	_MData_ __m512
	#define	step 8
#elif	defined(__AVX__)
	#define	_MData_ __m256
	#define	step 4
#else
	#define	_MData_ __m128
	#define	step 2
#endif

#ifdef	USE_XEON
		const float * __restrict__ m	= (const float * __restrict__) m_;
		__assume_aligned(m, Align);
#else
		const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_, Align);
#endif

#ifdef	__MIC__
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		int hand[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		int hand[4] = { 0, 0, 0, 0 };
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		int hand[2] = { 0, 0 };
#endif

		#pragma omp parallel default(shared) private(hand) reduction(+:nStrings,nChiral)
		{
			_MData_ mel, mPx, mPy, mPz, mXY, mYZ, mZX;
			_MData_ str, tmp;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[3], idxPx, idxPy, idxPz, idxXY, idxYZ, idxZX, idxP0, idxMz;

				{
					size_t tmi = idx/XC;

					X[2] = tmi/YC;
					X[1] = tmi - X[2]*YC;
					X[0] = idx - tmi*XC;
				}

				idxP0 = (idx << 1);
				idxPz = ((idx + Sf) << 1);
				idxMz = ((idx - Sf) >> 1);

				if (X[1] == YC-1)
				{
					idxPy = ((idx - Sf + XC) << 1);
					idxYZ = ((idx + XC) << 1);

					if (X[0] == XC-step)
					{
						idxPx = ((idx - XC + step) << 1);
						idxXY = ((idx - Sf + step) << 1);
						idxZX = ((idx + Sf - XC + step) << 1);
					}
					else
					{
						idxPx = ((idx + step) << 1);
						idxXY = ((idx - Sf + XC + step) << 1);
						idxZX = ((idx + Sf + step) << 1);
					}

					mel = opCode(load_ps, &m[idxP0]);
					mPx = opCode(load_ps, &m[idxPx]);
					mPz = opCode(load_ps, &m[idxPz]);
					mZX = opCode(load_ps, &m[idxZX]);
#ifdef	__MIC__
					tmp = opCode(swizzle_ps, opCode(load_ps, &m[idxPy]), _MM_SWIZ_REG_BADC);
					str = opCode(swizzle_ps, opCode(load_ps, &m[idxXY]), _MM_SWIZ_REG_BADC);
					mPy = opCode(swizzle_ps, opCode(load_ps, &m[idxYZ]), _MM_SWIZ_REG_BADC);
					mYZ = opCode(mask_blend_ps, opCode(int2mask, 0b1100110011001100), mPy, opCode(permute4f128_ps, mPy, _MM_PERM_ADCB));
					mXY = opCode(mask_blend_ps, opCode(int2mask, 0b1100110011001100), str, opCode(permute4f128_ps, str, _MM_PERM_ADCB));
					mPy = opCode(mask_blend_ps, opCode(int2mask, 0b1100110011001100), tmp, opCode(permute4f128_ps, tmp, _MM_PERM_ADCB));
#elif	defined(__AVX2__)
					mPy = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 2,3,4,5,6,7,0,1));
					mXY = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxXY]), opCode(setr_epi32, 2,3,4,5,6,7,0,1));
					mYZ = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxYZ]), opCode(setr_epi32, 2,3,4,5,6,7,0,1));
#elif	defined(__AVX__)
					tmp = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b01001110);
					str = opCode(permute_ps, opCode(load_ps, &m[idxXY]), 0b01001110);
					mPy = opCode(permute_ps, opCode(load_ps, &m[idxYZ]), 0b01001110);
					mYZ = opCode(blend_ps, mPy, opCode(permute2f128_ps, mPy, mPy, 0b00000001), 0b11001100);
					mXY = opCode(blend_ps, str, opCode(permute2f128_ps, str, str, 0b00000001), 0b11001100);
					mPy = opCode(blend_ps, tmp, opCode(permute2f128_ps, tmp, tmp, 0b00000001), 0b11001100);
#else
					tmp = opCode(load_ps, &m[idxPy]);
					str = opCode(load_ps, &m[idxXY]);
					mPy = opCode(load_ps, &m[idxYZ]);
					mYZ = opCode(shuffle_ps, mPy, mPy, 0b01001110);
					mXY = opCode(shuffle_ps, str, str, 0b01001110);
					mPy = opCode(shuffle_ps, tmp, tmp, 0b01001110);
#endif
				}
				else
				{
					idxPy = ((idx + XC) << 1);
					idxYZ = ((idx + Sf + XC) << 1);

					if (X[0] == XC-step)
					{
						idxPx = ((idx - XC + step) << 1);
						idxXY = ((idx + step) << 1);
						idxZX = ((idx + Sf - XC + step) << 1);
					}
					else
					{
						idxPx = ((idx + step) << 1);
						idxXY = ((idx + XC + step) << 1);
						idxZX = ((idx + Sf + step) << 1);
					}

					mel = opCode(load_ps, &m[idxP0]);
					mPx = opCode(load_ps, &m[idxPx]);
					mPz = opCode(load_ps, &m[idxPz]);
					mZX = opCode(load_ps, &m[idxZX]);
					mPy = opCode(load_ps, &m[idxPy]);
					mXY = opCode(load_ps, &m[idxXY]);
					mYZ = opCode(load_ps, &m[idxYZ]);
				}

				// Tienes los 7 puntos que definen las 3 plaquetas

				// Plaqueta XY

				stringHandS (mel, mPx, hand);
				stringHandS (mPx, mXY, hand);
				stringHandS (mXY, mPy, hand);
				stringHandS (mPy, mel, hand);

				size_t nIdx = (X[0]/step + X[1]*Lx + (X[2]-1)*Sf);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					size_t tIdx = nIdx + ih*YC*Lx;

					switch (hand[ih])
					{
						case 2:
						{
							static_cast<char *>(strg)[tIdx] |= STRING_XY_POSITIVE;
							nStrings++;
							nChiral++;
							//printf ("Positive string %d %d %d, 0\n", X[0]/step, X[1]+ih*Lx/step, idx/(XC*YC)-1);
							//fflush (stdout);
						}
						break;

						case -2:
						{
							static_cast<char *>(strg)[tIdx] |= STRING_XY_NEGATIVE;
							nStrings++;
							nChiral--;
							//printf ("Negative string %d %d %d, 0\n", X[0]/step, X[1]+ih*Lx/step, idx/(XC*YC)-1);
							//fflush (stdout);
						}
						break;

						default:
						break;
					}
	
					hand[ih] = 0;
				}

				// Plaqueta YZ

				stringHandS (mel, mPy, hand);
				stringHandS (mPy, mYZ, hand);
				stringHandS (mYZ, mPz, hand);
				stringHandS (mPz, mel, hand);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					size_t tIdx = nIdx + ih*YC*Lx;

					switch (hand[ih])
					{
						case 2:
						{
							static_cast<char *>(strg)[tIdx] |= STRING_YZ_POSITIVE;
							nStrings++;
							nChiral++;
							//printf ("Positive string %d %d %d, 1\n", X[0]/step, X[1]+ih*Lx/step, idx/(XC*YC)-1);
							//fflush (stdout);
						}
						break;

						case -2:
						{
							static_cast<char *>(strg)[tIdx] |= STRING_YZ_NEGATIVE;
							nStrings++;
							nChiral--;
							//printf ("Negative string %d %d %d, 1\n", X[0]/step, X[1]+ih*Lx/step, idx/(XC*YC)-1);
							//fflush (stdout);
						}
						break;

						default:
						break;
					}

					hand[ih] = 0;
				}

				// Plaqueta ZX

				stringHandS (mel, mPz, hand);
				stringHandS (mPz, mZX, hand);
				stringHandS (mZX, mPx, hand);
				stringHandS (mPx, mel, hand);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					size_t tIdx = nIdx + ih*YC*Lx;

					switch (hand[ih])
					{
						case 2:
						{
							static_cast<char *>(strg)[tIdx] |= STRING_ZX_POSITIVE;
							nStrings++;
							nChiral++;
							//printf ("Positive string %d %d %d, 2\n", X[0]/step, X[1]+ih*Lx/step, idx/(XC*YC)-1);
							//fflush (stdout);
						}
						break;

						case -2:
						{
							static_cast<char *>(strg)[tIdx] |= STRING_ZX_NEGATIVE;
							nStrings++;
							nChiral--;
							//printf ("Negative string %d %d %d, 2\n", X[0]/step, X[1]+ih*Lx/step, idx/(XC*YC)-1);
							//fflush (stdout);
						}
						break;

						default:
						break;
					}

					hand[ih] = 0;
				}
			}
		}

#undef	_MData_
#undef	step
	}

	printf ("Chirality of configuration %lf (%lld chiral points)\n", ((double) nChiral)/((double) (Vf-Vo)), nChiral);
	printf ("Density of configuration %lf (%llu string points)\n", ((double) nStrings)/((double) (Vf-Vo)), nStrings);
	fflush (stdout);

	return (((double) nStrings)/((double) (Vf-Vo)));
}

double	stringXeon	(Scalar *axionField, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, void *strg)
{
	double	  strDen = 0.;
#ifdef USE_XEON
	const int    micIdx = commAcc();
	const size_t Vh = (V>>1);
	char *str = static_cast<char*>(strg);

	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) out(str : length(Vh) UseX) nocopy(mX : ReUseX)
	{
		strDen = stringKernelXeon(mX, Lx, S, V+S, precision, str);
	}
#endif
	return	strDen;
}

double	stringCpu	(Scalar *axionField, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, void *strg)
{
	axionField->exchangeGhosts(FIELD_M);
	return	(stringKernelXeon(axionField->mCpu(), Lx, S, V+S, precision, strg));
}
