#include<cstdio>
#include<cmath>
#include"enum-field.h"
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

void	propSpecThetaKernelXeon(void * m_, void * __restrict__ v_, const void * __restrict__ m2_, double *z, const double dz, const double c, const double d,
				const double nQcd, const double fMom, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision)
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

		double *	      m		= (	 double * )		__builtin_assume_aligned (m_,  Align);
		double * __restrict__ v		= (	 double * __restrict__) __builtin_assume_aligned (v_,  Align);
		const double * __restrict__ m2	= (const double * __restrict__) __builtin_assume_aligned (m2_, Align);

		const double dzc = dz*c;
		const double dzd = dz*d;
		const double zR = *z;
		//const double zQ = 9.*pow(zR, nQcd+3.);
		const double zQ = axionmass2(zR, nQcd, zthres, zrestore)*zR*zR*zR;
		const double iz = 1.0/zR;

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
		const _MData_ zQVec  = opCode(set1_pd, zQ);
		const _MData_ izVec  = opCode(set1_pd, iz);
		const _MData_ fMVec  = opCode(set1_pd, fMom);
		const _MData_ dzcVec = opCode(set1_pd, dzc);
		const _MData_ dzdVec = opCode(set1_pd, dzd);

#ifdef	__AVX512F__
		const auto vShRg  = opCode(load_si512, shfRg);
		const auto vShLf  = opCode(load_si512, shfLf);
#endif

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, vel, acu;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t idxMz;

				mel = opCode(load_pd, &m[idx]);
				tmp = opCode(load_pd, &m2[idx]);

				idxMz = idx-Sf;

				acu = opCode(sub_pd,
					opCode(mul_pd, tmp, fMVec),
					opCode(mul_pd, zQVec, opCode(sin_pd, opCode(mul_pd, mel, izVec))));
				vel = opCode(load_pd, &v[idxMz]);

#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_pd, acu, dzcVec, vel);
				acu = opCode(fmadd_pd, tmp, dzdVec, mel);
#else
				tmp = opCode(add_pd, vel, opCode(mul_pd, acu, dzcVec));
				acu = opCode(add_pd, mel, opCode(mul_pd, tmp, dzdVec));
#endif

				/*	Store	*/

				opCode(store_pd, &v[idxMz], tmp);
				opCode(store_pd, &m[idx],   acu);
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

		float *		     m		= (	 float * )		__builtin_assume_aligned (m_,  Align);
		float * __restrict__ v		= (	 float * __restrict__)	__builtin_assume_aligned (v_,  Align);
		const float * __restrict__ m2	= (const float * __restrict__)	__builtin_assume_aligned (m2_, Align);

		const float dzc = dz*c;
		const float dzd = dz*d;
		const float zR = *z;
		//const float zQ = 9.*powf(zR, nQcd+3.);
		const float zQ = (float) axionmass2((double) zR, nQcd, zthres, zrestore)*zR*zR*zR;
		const float iz = 1.f/zR;
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
		const _MData_ zQVec  = opCode(set1_ps, zQ);
		const _MData_ izVec  = opCode(set1_ps, iz);
		const _MData_ fMVec  = opCode(set1_ps, fMom);
		const _MData_ dzcVec = opCode(set1_ps, dzc);
		const _MData_ dzdVec = opCode(set1_ps, dzd);

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, vel, mPy, mMy, acu;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t idxMz;

				mel = opCode(load_ps, &m[idx]);
				tmp = opCode(load_ps, &m2[idx]);

				idxMz = idx-Sf;

				acu = opCode(sub_ps,
					opCode(mul_ps, tmp, fMVec),
					opCode(mul_ps, zQVec, opCode(sin_ps, opCode(mul_ps, mel, izVec))));
				vel = opCode(load_ps, &v[idxMz]);

#if	defined(__MIC__) || defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_ps, acu, dzcVec, vel);
				acu = opCode(fmadd_ps, tmp, dzdVec, mel);
#else
				tmp = opCode(add_ps, vel, opCode(mul_ps, acu, dzcVec));
				acu = opCode(add_ps, mel, opCode(mul_ps, tmp, dzdVec));
#endif
				/*	Store	*/

				opCode(store_ps, &v[idxMz], tmp);
				opCode(store_ps, &m[idx],   acu);
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
