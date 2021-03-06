#include<cstdio>
#include<cmath>
#include"scalar/scalarField.h"
#include"enum-field.h"
#include"propagator/RKParms.h"
#include"scalar/varNQCD.h"
#include "utils/parse.h"

#include"utils/triSimd.h"

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

void	propThetaKernelXeon(const void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_, double *z, const double dz, const double c, const double d,
			    const double ood2, const double nQcd, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision)
{
	const size_t Sf = Lx*Lx;

	if (precision == FIELD_DOUBLE)
	{
#ifdef	__AVX512F__
	#define	_MData_ __m512d
	#define	step 8
#elif	defined(__AVX__)
	#define	_MData_ __m256d
	#define	step 4
#else
	#define	_MData_ __m128d
	#define	step 2
#endif

		const double * __restrict__ m	= (const double * __restrict__) __builtin_assume_aligned (m_, Align);
		double * __restrict__ v		= (double * __restrict__) __builtin_assume_aligned (v_, Align);
		double * __restrict__ m2	= (double * __restrict__) __builtin_assume_aligned (m2_, Align);

		const double dzc = dz*c;
		const double dzd = dz*d;
		const double zR = *z;
		const double iZ = 1./zR;
		//const double zQ = 9.*pow(zR, nQcd+3.);
		const double zQ = axionmass2(zR, nQcd, zthres, zrestore)*zR*zR*zR;
		const double tV	= 2.*M_PI*zR;

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
		const _MData_ tpVec  = opCode(set1_pd, tV);
		const _MData_ zQVec  = opCode(set1_pd, zQ);
		const _MData_ izVec  = opCode(set1_pd, iZ);
		const _MData_ d2Vec  = opCode(set1_pd, ood2);
		const _MData_ dzcVec = opCode(set1_pd, dzc);
		const _MData_ dzdVec = opCode(set1_pd, dzd);

#ifdef	__AVX512F__
		const auto vShRg  = opCode(load_si512, shfRg);
		const auto vShLf  = opCode(load_si512, shfLf);
#endif

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, vel, mPy, mMy, acu;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxPx, idxMx, idxPy, idxMy, idxPz, idxMz;

				mel = opCode(load_pd, &m[idx]);

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

				if (X[0] == 0)
					idxMx = idx + XC - step;
				else
					idxMx = idx - step;

				if (X[1] == 0)
				{
					idxMy = idx + Sf - XC;
					idxPy = idx + XC;
					mPy = opCode(load_pd, &m[idxPy]);
#ifdef	__AVX512F__
					mMy = opCode(add_pd, opCode(permutexvar_pd, vShRg, opCode(load_pd, &m[idxMy])), mPy);
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
				}
				else
				{
					idxMy = idx - XC;
					mMy = opCode(load_pd, &m[idxMy]);

					if (X[1] == YC-1)
					{
						idxPy = idx - Sf + XC;
#ifdef	__AVX512F__
						mPy = opCode(add_pd, opCode(permutexvar_pd, vShLf, opCode(load_pd, &m[idxPy])), mMy);
#elif	defined(__AVX2__)	//AVX2
						mPy = opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_pd, &m[idxPy])), opCode(setr_epi32, 2,3,4,5,6,7,0,1)));
#elif	defined(__AVX__)
						acu = opCode(permute_pd, opCode(load_pd, &m[idxPy]), 0b00000101);
						vel = opCode(permute2f128_pd, acu, acu, 0b00000001);
						mPy = opCode(blend_pd, acu, vel, 0b00001010);
#else
						vel = opCode(load_pd, &m[idxPy]);
						mPy = opCode(shuffle_pd, vel, vel, 0x00000001);
#endif
					}
					else
					{
						idxPy = idx + XC;
						mPy = opCode(load_pd, &m[idxPy]);
					}
				}

				idxPz = idx+Sf;
				idxMz = idx-Sf;

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

				/*	Dv	*/

				vel = opCode(sub_pd,
					opCode(mul_pd, acu, d2Vec),
					opCode(mul_pd, zQVec, opCode(sin_pd, opCode(mul_pd, mel, izVec))));
				mPy = opCode(load_pd, &v[idxMz]);

#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_pd, vel, dzcVec, mPy);
				mMy = opCode(fmadd_pd, tmp, dzdVec, mel);
#else
				tmp = opCode(add_pd, mPy, opCode(mul_pd, vel, dzcVec));
				mMy = opCode(add_pd, mel, opCode(mul_pd, tmp, dzdVec));
#endif

				/*	Make sure the result is between -pi and pi	*/

				acu = opCode(mod_pd, mMy, tpVec);

				/*	Store	*/

				opCode(store_pd, &v[idxMz], tmp);
				opCode(store_pd, &m2[idx],  acu);
			}
		}
#undef	_MData_
#undef	step
	}
	else if (precision == FIELD_SINGLE)
	{
#ifdef	__AVX512F__
	#define	_MData_ __m512
	#define	step 16
#elif	defined(__AVX__)
	#define	_MData_ __m256
	#define	step 8
#else
	#define	_MData_ __m128
	#define	step 4
#endif

		const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_, Align);
		float * __restrict__ v		= (float * __restrict__) __builtin_assume_aligned (v_, Align);
		float * __restrict__ m2		= (float * __restrict__) __builtin_assume_aligned (m2_, Align);

		const float dzc = dz*c;
		const float dzd = dz*d;
		const float zR = *z;
		const float iZ = 1./zR;
		//const float zQ = 9.*powf(zR, nQcd+3.);
		const float zQ = (float) axionmass2((double) zR, nQcd, zthres, zrestore)*zR*zR*zR;
		const float tV	= 2.*M_PI*zR;
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
		const _MData_ tpVec  = opCode(set1_ps, tV);
		const _MData_ zQVec  = opCode(set1_ps, zQ);
		const _MData_ izVec  = opCode(set1_ps, iZ);
		const _MData_ d2Vec  = opCode(set1_ps, ood2);
		const _MData_ dzcVec = opCode(set1_ps, dzc);
		const _MData_ dzdVec = opCode(set1_ps, dzd);

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, vel, tpP, mPy, tpM, mMy, v2p, acu, tP2, tM2, sel;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz;

				mel = opCode(load_ps, &m[idx]);

				{
					size_t tmi = idx/XC, itp;

					itp = tmi/YC;
					X[1] = tmi - itp*YC;
					X[0] = idx - tmi*XC;
				}

				if (X[0] == XC-step)
					idxPx = idx - XC + step;
				else
					idxPx = idx + step;

				if (X[0] == 0)
					idxMx = idx + XC - step;
				else
					idxMx = idx - step;

				if (X[1] == 0)
				{
					idxMy = idx + Sf - XC;
					idxPy = idx + XC;
					mPy = opCode(load_ps, &m[idxPy]);
#ifdef	__AVX512F__
					mMy = opCode(permutexvar_ps, vShRg, opCode(load_ps, &m[idxMy]));
#elif	defined(__AVX2__)
					mMy = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 7,0,1,2,3,4,5,6));
#elif	defined(__AVX__)
					acu = opCode(permute_ps, opCode(load_ps, &m[idxMy]), 0b10010011);
					vel = opCode(permute2f128_ps, acu, acu, 0b00000001);
					mMy = opCode(blend_ps, acu, vel, 0b00010001);
#else
					acu = opCode(load_ps, &m[idxMy]);
					mMy = opCode(shuffle_ps, acu, acu, 0b10010011);
#endif
				}
				else
				{
					idxMy = idx - XC;
					mMy = opCode(load_ps, &m[idxMy]);

					if (X[1] == YC-1)
					{
						idxPy = idx - Sf + XC;
#ifdef	__AVX512F__
						mPy = opCode(permutexvar_ps, vShLf, opCode(load_ps, &m[idxPy]));
#elif	defined(__AVX2__)
						mPy = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 1,2,3,4,5,6,7,0));
#elif	defined(__AVX__)
						acu = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b00111001);
						vel = opCode(permute2f128_ps, acu, acu, 0b00000001);
						mPy = opCode(blend_ps, acu, vel, 0b10001000);
#else
						vel = opCode(load_ps, &m[idxPy]);
						mPy = opCode(shuffle_ps, vel, vel, 0b00111001);
#endif
					}
					else
					{
						idxPy = idx + XC;
						mPy = opCode(load_ps, &m[idxPy]);
					}
				}

				idxPz = idx+Sf;
				idxMz = idx-Sf;

				/*	idxPx	*/

				vel = opCode(sub_ps, opCode(load_ps, &m[idxPx]), mel);
				acu = opCode(mod_ps, vel, tpVec);
				/*	idxMx	*/

				vel = opCode(sub_ps, opCode(load_ps, &m[idxMx]), mel);
				acu = opCode(add_ps, acu, opCode(mod_ps, vel, tpVec));

				/*	idxPz	*/

				vel = opCode(sub_ps, opCode(load_ps, &m[idxPz]), mel);
				acu = opCode(add_ps, acu, opCode(mod_ps, vel, tpVec));

				/*	idxMz	*/

				vel = opCode(sub_ps, opCode(load_ps, &m[idxMz]), mel);
				acu = opCode(add_ps, acu, opCode(mod_ps, vel, tpVec));

				/*	idxPy	*/

				vel = opCode(sub_ps, mPy, mel);
				acu = opCode(add_ps, acu, opCode(mod_ps, vel, tpVec));

				/*	idxMy	*/

				vel = opCode(sub_ps, mMy, mel);
				acu = opCode(add_ps, acu, opCode(mod_ps, vel, tpVec));

				/*	Dv	*/

				tpM = opCode(sub_ps,
					opCode(mul_ps, acu, d2Vec),
					opCode(mul_ps, zQVec, opCode(sin_ps, opCode(mul_ps, mel, izVec))));

				mPy = opCode(load_ps, &v[idxMz]);

#if	defined(__MIC__) || defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_ps, tpM, dzcVec, mPy);
				tpP = opCode(fmadd_ps, tmp, dzdVec, mel);
#else
				tmp = opCode(add_ps, mPy, opCode(mul_ps, tpM, dzcVec));
				tpP = opCode(add_ps, mel, opCode(mul_ps, tmp, dzdVec));
#endif
				/*	Make sure the result is between -pi and pi	*/
				acu = opCode(mod_ps, tpP, tpVec);

				/*	Store	*/

				opCode(store_ps, &v[idxMz], tmp);
				opCode(store_ps, &m2[idx],  acu);
			}
		}
#undef	_MData_
#undef	step
	}
}

#undef	opCode
#undef	opCode_N
#undef	opCode_P
#undef	Align
#undef	_PREFIX_
