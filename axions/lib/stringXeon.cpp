// ANADIR CONTADOR DE CUERDAS

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

// HACER INLINE
// ACABAR

#ifdef USE_XEON
__attribute__((target(mic)))
#endif
#ifdef	__MIC__

#define	_MData_ __m256d

void	stringHandD(const __m512d s1, const __m512d s2)
{
	str = opCode(mul_pd, mel, mPx);
	tmp =
	str = opCode(mul_pd, mPx, conj);
	tp2 = opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, str), _MM_PERM_BADC));
	tp3 = opCode(mul_pd, tmp, tp2);
	tp2 = opCode(add_pd, tp3, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tp3), _MM_PERM_BADC)));
	tp3 = 
	str = opCode(and_pd, tp2, zero);
#elif defined(__AVX__)
inline	void	stringHandD(const __m256d s1, const __m256d s2, const __m256d conj, long long int *hand)
{
	__m256d	str, tmp, tp2, tp3;

	long long int tmpHand[2];

	str = opCode(mul_pd, s1, s2);
	tmp = opCode(cmp_pd, str, opCode(setzero_pd), _CMP_LT_OS);
	str = opCode(mul_pd, s2, conj);
	tp2 = opCode(permute_pd, str, 0b00000101);
	tp3 = opCode(mul_pd, s1,  tp2);
	tp2 = opCode(add_pd, tp3, opCode(permute_pd, tp3, 0b00000101));
	tp3 = opCode(cmp_pd, tp2, opCode(setzero_pd), _CMP_GT_OS);
	tp2 = opCode(and_pd, tp3, tmp);
#ifdef __AVX2__
	str = opCode(permute4x64_pd, tp2, 0b10001101);
#else
	tmp = opCode(permute2f128_pd, tp2, tp2, 0b00000001);
	tp3 = opCode(permute_pd, tmp, 0b00000101);
	str = opCode(blend_pd, tp3, tp2, 0b00000101);
#endif
	opCode(maskstore_pd, static_cast<double*>(static_cast<void*>(tmpHand)), opCode(setr_epi64x, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 0), str);
	hand[0] = ((tmpHand[0] >> 62) & 2) - 1;
	hand[1] = ((tmpHand[1] >> 62) & 2) - 1;

	return;
}

inline	void	stringHandS(const __m256 s1, const __m256 s2, const __m256 conj, int *hand)
{
	__m256	str, tmp, tp2, tp3;

	int tmpHand[4];

	str = opCode(mul_ps, s1, s2);
	tmp = opCode(cmp_ps, str, opCode(setzero_ps), _CMP_LT_OS);
	str = opCode(mul_ps, s2, conj);
	tp2 = opCode(permute_ps, str, 0b10110001);
	tp3 = opCode(mul_ps, tmp, tp2);
	tp2 = opCode(add_ps, tp3, opCode(permute_ps, tp3, 0b10110001));
	tp3 = opCode(cmp_ps, tp2, opCode(setzero_ps), _CMP_GT_OS);
	tp2 = opCode(and_ps, tp3, tmp);
#ifdef __AVX2__
	str = opCode(permutevar8x32_ps, tp2, opCode(set_epi32, 6, 4, 3, 0, 7, 5, 3, 1));
#else
	tmp = opCode(permute2f128_ps, tp2, tp2, 0b00000001);
	tp3 = opCode(permute_ps, tmp, 0b10110001);
	str = opCode(blend_ps, tp3, tp2, 0b01010101);
#endif
	opCode(maskstore_ps, static_cast<float*>(static_cast<void*>(tmpHand)), opCode(setr_epi64x, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 0), str);

/* Vectoriza esto tb */
	hand[0] += ((tmpHand[0] >> 30) & 2) - 1;
	hand[1] += ((tmpHand[1] >> 30) & 2) - 1;
	hand[2] += ((tmpHand[2] >> 30) & 2) - 1;
	hand[3] += ((tmpHand[3] >> 30) & 2) - 1;

	return;
}
#else
inline	void	stringHandD(const __m128d s1, const __m128d s2, const __m128d conj, long long int *hand)
{
	__m128d	str, tmp, tp2, tp3;

	long long int tmpHand;

	str = opCode(mul_pd, s1, s2);
	tmp = opCode(cmplt_pd, str, opCode(setzero_pd));
	str = opCode(mul_pd, s2, conj);
	tp2 = opCode(shuffle_pd, str, str, 0b00000001);
	tp3 = opCode(mul_pd, tmp, tp2);
	tp2 = opCode(add_pd, tp3, opCode(shuffle_pd, tp3, tp3, 0b00000001));
	tp3 = opCode(cmpgt_pd, tp2, opCode(setzero_pd));
	str = opCode(and_pd,   tp3, tmp);
	opCode(storeh_pd, static_cast<double*>(static_cast<void*>(&tmpHand)), str);
	*hand = ((tmpHand >> 62) & 2) - 1;

	return;
}

inline	void	stringHandS(const __m128 s1, const __m128 s2, const __m128 conj, int *hand)
{
	__m128	str, tmp, tp2, tp3;

	long long int tmpHand[2];

	str = opCode(mul_ps, s1, s2);
	tmp = opCode(cmplt_ps, str, opCode(setzero_ps));
	str = opCode(mul_ps, s2, conj);
	tp2 = opCode(shuffle_ps, str, str, 0b10110001);
	tp3 = opCode(mul_ps, tmp, tp2);
	tp2 = opCode(add_ps, tp3, opCode(shuffle_ps, tp3, tp3, 0b10110001));
	tp3 = opCode(cmpgt_ps, tp2, opCode(setzero_ps));
	tmp = opCode(and_ps,   tp3, tmp);
	tp2 = opCode(shuffle_ps, tmp, tmp, 0b11011000);
	opCode(storeh_pd, static_cast<double*>(static_cast<void*>(tmpHand)), opCode(castps_pd, str));
	hand[0] = ((tmpHand[0] >> 30) & 2) - 1;
	hand[1] = ((tmpHand[1] >> 30) & 2) - 1;

	return;
}
#endif

#ifdef USE_XEON
__attribute__((target(mic)))
#endif
void	stringKernelXeon(const void * __restrict__ m_, const int Lx, const int Vo, const int Vf, FieldPrecision precision, void * __restrict__ string)
{
	const size_t Sf = Lx*Lx;

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

		const double __attribute__((aligned(Align))) zeroAux[8]  = { 0., 0., 0., 0., 0., 0., 0., 0. };
		const double __attribute__((aligned(Align))) conjAux[8]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };

		const _MData_ zero  = opCode(load_pd, zeroAux);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) conjAux[4]  = { 1.,-1., 1.,-1. };

		const _MData_ zero  = opCode(setzero_pd);

		long long int hand[2] = { 0, 0 };
#else
		const size_t XC = Lx;
		const size_t YC = Lx;

		const double __attribute__((aligned(Align))) conjAux[2]  = { 1.,-1. };

		const _MData_ zero = opCode(setzero_pd);
		long long int hand[1] = { 0 };
#endif
		const _MData_ conj = opCode(load_pd, conjAux);

		#pragma omp parallel default(shared) private(hand)
		{
			_MData_ mel, mPx, mPy, mPz, mXY, mYZ, mZX;
			_MData_ str, tmp;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxPx, idxPy, idxPz, idxXY, idxYZ, idxZX, idxP0;

				{
					size_t tmi = idx/XC, tpi;

					tpi = tmi/YC;
					X[1] = tmi - tpi*YC;
					X[0] = idx - tmi*XC;
				}

				idxP0 = (idx << 1);
				idxPz = ((idx + Sf) << 1);

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

				// Tienes los 7 puntos que definen las 3 plaquetas

				// Plaqueta XY

				stringHandD (mel, mPx, conj, hand);
				stringHandD (mPx, mXY, conj, hand);
				stringHandD (mXY, mPy, conj, hand);
				stringHandD (mPy, mel, conj, hand);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					switch (hand[ih])
					{
						case 2:
						{
							int strDf = 1;								// 0b0001
							printf ("Positive string %d %d %d, 0\n", X[0]+ih, X[1], idx/(XC*YC));
							fflush (stdout);
							static_cast<int *>(string)[idx>>3] |= (strDf << (4*ih));
						}
						break;

						case -2:
						{
							int strDf = 9;								// 0b1001
							printf ("Negative string %d %d %d, 0\n", X[0]+ih, X[1], idx/(XC*YC));
							fflush (stdout);
							static_cast<int *>(string)[idx>>3] |= (strDf << (4*ih));
						}
						break;

						default:
						break;
					}
				}

				// Plaqueta YZ

				stringHandD (mel, mPy, conj, hand);
				stringHandD (mPy, mYZ, conj, hand);
				stringHandD (mYZ, mPz, conj, hand);
				stringHandD (mPz, mel, conj, hand);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					switch (hand[ih])
					{
						case 2:
						{
							int strDf = 2;								// 0b0010
							printf ("Positive string %d %d %d, 1\n", X[0]+ih, X[1], idx/(XC*YC));
							fflush (stdout);
							static_cast<int *>(string)[idx>>3] |= (strDf << (4*ih));
						}
						break;

						case -2:
						{
							int strDf = 10;								// 0b1010
							printf ("Negative string %d %d %d, 1\n", X[0]+ih, X[1], idx/(XC*YC));
							fflush (stdout);
							static_cast<int *>(string)[idx>>3] |= (strDf << (4*ih));
						}
						break;

						default:
						break;
					}

					hand[ih] = 0;
				}

				// Plaqueta ZX

				stringHandD (mel, mPz, conj, hand);
				stringHandD (mPz, mZX, conj, hand);
				stringHandD (mZX, mPx, conj, hand);
				stringHandD (mPx, mel, conj, hand);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					switch (hand[ih])
					{
						case 2:
						{
							int strDf = 4;								// 0b0100
							printf ("Positive string %d %d %d, 2\n", X[0]+ih, X[1], idx/(XC*YC));
							fflush (stdout);
							static_cast<int *>(string)[idx>>3] |= (strDf << (4*ih));
						}
						break;

						case -2:
						{
							int strDf = 12;								// 0b1100
							printf ("Negative string %d %d %d, 2\n", X[0]+ih, X[1], idx/(XC*YC));
							fflush (stdout);
							static_cast<int *>(string)[idx>>3] |= (strDf << (4*ih));
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

		const float __attribute__((aligned(Align))) zeroAux[16]  = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };
		const float __attribute__((aligned(Align))) conjAux[16]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1. };

		int hand[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

		const _MData_ zero  = opCode(load_ps, zeroAux);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) conjAux[8]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };

		const _MData_ zero  = opCode(setzero_ps);

		int hand[4] = { 0, 0, 0, 0 };
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const float __attribute__((aligned(Align))) conjAux[4]  = { 1.,-1., 1.,-1. };

		const _MData_ zero  = opCode(setzero_ps);

		int hand[2] = { 0, 0 };
#endif

		const _MData_ conj = opCode(load_ps, conjAux);

		#pragma omp parallel default(shared) private(hand) 
		{
			_MData_ mel, mPx, mPy, mPz, mXY, mYZ, mZX;
			_MData_ str, tmp;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxPx, idxPy, idxPz, idxXY, idxYZ, idxZX, idxP0;

				{
					size_t tmi = idx/XC, tpi;

					tpi = tmi/YC;
					X[1] = tmi - tpi*YC;
					X[0] = idx - tmi*XC;
				}

				idxP0 = (idx << 1);
				idxPz = ((idx + Sf) << 1);

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

				stringHandS (mel, mPx, conj, hand);
				stringHandS (mPx, mXY, conj, hand);
				stringHandS (mXY, mPy, conj, hand);
				stringHandS (mPy, mel, conj, hand);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					switch (hand[ih])
					{
						case 2:
						{
							int strDf = 1;								// 0b0001
							printf ("Positive string %d %d %d, 0\n", X[0]+ih, X[1], idx/(XC*YC));
							fflush (stdout);
							static_cast<int *>(string)[idx>>3] |= (strDf << (4*ih));
						}
						break;

						case -2:
						{
							int strDf = 9;								// 0b1001
							printf ("Negative string %d %d %d, 0\n", X[0]+ih, X[1], idx/(XC*YC));
							fflush (stdout);
							static_cast<int *>(string)[idx>>3] |= (strDf << (4*ih));
						}
						break;

						default:
						break;
					}

					hand[ih] = 0;
				}

				// Plaqueta YZ

				stringHandS (mel, mPy, conj, hand);
				stringHandS (mPy, mYZ, conj, hand);
				stringHandS (mYZ, mPz, conj, hand);
				stringHandS (mPz, mel, conj, hand);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					switch (hand[ih])
					{
						case 2:
						{
							int strDf = 2;								// 0b0010
							printf ("Positive string %d %d %d, 1\n", X[0]+ih, X[1], idx/(XC*YC));
							fflush (stdout);
							static_cast<int *>(string)[idx>>3] |= (strDf << (4*ih));
						}
						break;

						case -2:
						{
							int strDf = 10;								// 0b1010
							printf ("Negative string %d %d %d, 1\n", X[0]+ih, X[1], idx/(XC*YC));
							fflush (stdout);
							static_cast<int *>(string)[idx>>3] |= (strDf << (4*ih));
						}
						break;

						default:
						break;
					}

					hand[ih] = 0;
				}

				// Plaqueta ZX

				stringHandS (mel, mPz, conj, hand);
				stringHandS (mPz, mZX, conj, hand);
				stringHandS (mZX, mPx, conj, hand);
				stringHandS (mPx, mel, conj, hand);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					switch (hand[ih])
					{
						case 2:
						{
							int strDf = 4;								// 0b0100
							printf ("Positive string %d %d %d, 2\n", X[0]+ih, X[1], idx/(XC*YC));
							fflush (stdout);
							static_cast<int *>(string)[idx>>3] |= (strDf << (4*ih));
						}
						break;

						case -2:
						{
							int strDf = 12;								// 0b1100
							printf ("Negative string %d %d %d, 2\n", X[0]+ih, X[1], idx/(XC*YC));
							fflush (stdout);
							static_cast<int *>(string)[idx>>3] |= (strDf << (4*ih));
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
}

void	stringXeon	(Scalar *axionField, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, void *string)
{
#ifdef USE_XEON
	const int  micIdx = commAcc(); 

	int bulk  = 32;

	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		stringKernelXeon(mX, Lx, S, V+S, precision, string);
	}
#endif
}

void	stringCpu	(Scalar *axionField, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, void *string)
{
	axionField->exchangeGhosts(FIELD_M);
	stringKernelXeon(axionField->mCpu(), Lx, S, V+S, precision, string);
}
