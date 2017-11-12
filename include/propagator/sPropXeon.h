#include<cstdio>
#include<cmath>
#include"scalar/scalarField.h"
#include"enum-field.h"
#include"propagator/laplacian.h"
#include"scalar/varNQCD.h"
#include"utils/utils.h"
#include"fft/fftCode.h"

#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

#include <immintrin.h>

#if	defined(__AVX512F__)
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

template<const VqcdType VQcd>
inline	void	sPropKernelXeon(void * __restrict__ m_, void * __restrict__ v_, const void * __restrict__ m2_, double *z, const double dz, const double c, const double d,
				const double LL, const double nQcd, const double fMom, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision)
{
	const size_t Sf = Lx*Lx;

	if (precision == FIELD_DOUBLE)
	{
#if	defined(__AVX512F__)
	#define	_MData_ __m512d
	#define	step 4
#elif	defined(__AVX__)
	#define	_MData_ __m256d
	#define	step 2
#else
	#define	_MData_ __m128d
	#define	step 1
#endif

		double	     * __restrict__ m	= (double	* __restrict__)	__builtin_assume_aligned (m_,  Align);
		double	     * __restrict__ v	= (double	* __restrict__) __builtin_assume_aligned (v_,  Align);
		const double * __restrict__ m2	= (const double * __restrict__) __builtin_assume_aligned (m2_, Align);

		const double dzc = dz*c;
		const double dzd = dz*d;
		const double zR = *z;
		const double z2 = zR*zR;
		//const double zQ = 9.*pow(zR, nQcd+3.);
		const double zQ = axionmass2(zR, nQcd, zthres, zrestore)*zR*zR*zR;

#if	defined(__AVX512F__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const double    __attribute__((aligned(Align))) zQAux[8] = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };	// Only real part
		const double    __attribute__((aligned(Align))) zRAux[8] = { zR, 0., zR, 0., zR, 0., zR, 0. };	// Only real part
		const long long __attribute__((aligned(Align))) shfRg[8] = {6, 7, 0, 1, 2, 3, 4, 5 };
		const long long __attribute__((aligned(Align))) shfLf[8] = {2, 3, 4, 5, 6, 7, 0, 1 };

		const auto  vShRg = opCode(load_si512, shfRg);
		const auto  vShLf = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) zQAux[4] = { zQ, 0., zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zRAux[4] = { zR, 0., zR, 0. };	// Only real part
#else
		const size_t XC = Lx;
		const size_t YC = Lx;

		const double __attribute__((aligned(Align))) zQAux[2] = { zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zRAux[2] = { zR, 0. };	// Only real part

#endif
		const _MData_ zQVec  = opCode(load_pd, zQAux);
		const _MData_ zRVec  = opCode(load_pd, zRAux);
		const _MData_ fMVec  = opCode(set1_pd, fMom);

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, mPx, mPy;
			size_t idxMz, idxP0 ;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				idxMz = ((idx-Sf) << 1);
				idxP0 = (idx << 1);

				mPx = opCode(load_pd, &m2[idxMz]);
				tmp = opCode(mul_pd, mPx, fMVec);
				mel = opCode(load_pd,  &m[idxP0]);
				mPy = opCode(mul_pd, mel, mel);

#if	defined(__AVX512F__)
				mPx = opCode(add_pd, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, mPy), _MM_PERM_BADC)), mPy);
#elif	defined(__AVX__)
				mPx = opCode(add_pd, opCode(permute_pd, mPy, 0b00000101), mPy);
#else
				mPx = opCode(add_pd, opCode(shuffle_pd, mPy, mPy, 0b00000001), mPy);
#endif

				switch	(VQcd) {
					case	VQCD_1:
						mPx = opCode(add_pd, tmp,
							opCode(sub_pd, zQVec,
							opCode(mul_pd, mel,
								opCode(mul_pd,
									opCode(sub_pd, mPx, opCode(set1_pd, z2)),
									opCode(set1_pd, LL)))));
						break;

					case	VQCD_1_PQ_2:
						mPx = opCode(add_pd, tmp,
							opCode(sub_pd, zQVec,
							opCode(mul_pd, mel,
								opCode(mul_pd,
									opCode(sub_pd, mPx, opCode(set1_pd, z2)),
									opCode(set1_pd, LL)))));
						break;

					case	VQCD_2:
						mPx = opCode(add_pd, tmp,
							opCode(sub_pd,
								opCode(mul_pd, opCode(set1_pd, zQ), opCode(sub_pd, zRVec, mel)),
								opCode(mul_pd, mel,
									opCode(mul_pd,
										opCode(sub_pd, mPx, opCode(set1_pd, z2)),
										opCode(set1_pd, LL)))));
						break;
				}

				mPy = opCode(load_pd, &v[idxMz]);
#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_pd, mPx, opCode(set1_pd, dzc), mPy);
				mPx = opCode(fmadd_pd, tmp, opCode(set1_pd, dzd), mel);
#else
				tmp = opCode(add_pd, mPy, opCode(mul_pd, mPx, opCode(set1_pd, dzc)));
				mPx = opCode(add_pd, mel, opCode(mul_pd, tmp, opCode(set1_pd, dzd)));
#endif
				opCode(store_pd, &v[idxMz], tmp);
				opCode(store_pd, &m[idxP0], mPx);
			}
		}
#undef	_MData_
#undef	step
	}
	else if (precision == FIELD_SINGLE)
	{
#if	defined(__AVX512F__)
	#define	_MData_ __m512
	#define	step 8
#elif	defined(__AVX__)
	#define	_MData_ __m256
	#define	step 4
#else
	#define	_MData_ __m128
	#define	step 2
#endif

		float	    * __restrict__ m	= (      float * __restrict__) __builtin_assume_aligned (m_,  Align);
		float	    * __restrict__ v	= (      float * __restrict__) __builtin_assume_aligned (v_,  Align);
		const float * __restrict__ m2	= (const float * __restrict__) __builtin_assume_aligned (m2_, Align);

		const float dzc = dz*c;
		const float dzd = dz*d;
		const float zR = *z;
		const float z2 = zR*zR;
		//const float zQ = 9.*powf(zR, nQcd+3.);
		const float zQ = (float) axionmass2(*z, nQcd, zthres, zrestore)*zR*zR*zR;

#if	defined(__AVX512F__)
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) zQAux[16] = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zRAux[16] = { zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f };
		const int   __attribute__((aligned(Align))) shfRg[16] = {14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
		const int   __attribute__((aligned(Align))) shfLf[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1};

		const auto  vShRg  = opCode(load_si512, shfRg);
		const auto  vShLf  = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) zQAux[8]  = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zRAux[8]  = { zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f };
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const float __attribute__((aligned(Align))) zQAux[4]  = { zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zRAux[4]  = { zR, 0.f, zR, 0.f };
#endif
		const _MData_ zQVec  = opCode(load_ps, zQAux);
		const _MData_ zRVec  = opCode(load_ps, zRAux);
		const _MData_ fMVec  = opCode(set1_ps, fMom);

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, mPx, mPy, mMx;
			size_t idxMz, idxP0 ;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				idxMz = ((idx-Sf) << 1);
				idxP0 = (idx << 1);

				mPx = opCode(load_ps, &m2[idxMz]);
				tmp = opCode(mul_ps, mPx, fMVec);
				mel = opCode(load_ps, &m[idxP0]);
				mPy = opCode(mul_ps, mel, mel);

#if	defined(__AVX__)// || defined(__AVX512F__)
				mPx = opCode(add_ps, opCode(permute_ps, mPy, 0b10110001), mPy);
#else
				mPx = opCode(add_ps, opCode(shuffle_ps, mPy, mPy, 0b10110001), mPy);
#endif
				switch	(VQcd) {
					case	VQCD_1:
						mMx = opCode(add_ps, tmp,
							opCode(sub_ps, zQVec,
								opCode(mul_ps, mel,
									opCode(mul_ps,
										opCode(sub_ps, mPx, opCode(set1_ps, z2)),
										opCode(set1_ps, LL)))));
						break;

					case	VQCD_1_PQ_2:
						mMx = opCode(add_ps, tmp,
							opCode(sub_ps, zQVec,
								opCode(mul_ps, mel,
									opCode(mul_ps,
										opCode(sub_ps, mPx, opCode(set1_ps, z2)),
										opCode(set1_ps, LL)))));
						break;

					case	VQCD_2:
						mMx = opCode(sub_ps,
							opCode(sub_ps, tmp, opCode(mul_ps, opCode(set1_ps, zQ), opCode(sub_ps, mel, zRVec))),
							opCode(mul_ps,
								opCode(mul_ps, mel,
									opCode(sub_ps, mPx, opCode(set1_ps, z2))),
									opCode(set1_ps, LL)));
						break;
				}

				mPy = opCode(load_ps, &v[idxMz]);

#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_ps, mMx, opCode(set1_ps, dzc), mPy);
				mPx = opCode(fmadd_ps, tmp, opCode(set1_ps, dzd), mel);
#else
				tmp = opCode(add_ps, mPy, opCode(mul_ps, mMx, opCode(set1_ps, dzc)));
				mPx = opCode(add_ps, mel, opCode(mul_ps, tmp, opCode(set1_ps, dzd)));
#endif
				opCode(store_ps, &v[idxMz], tmp);
				opCode(store_ps, &m[idxP0], mPx);
			}
		}
#undef	_MData_
#undef	step
	}
}
