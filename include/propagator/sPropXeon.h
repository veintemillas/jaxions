#include<cstdio>
#include<cmath>
#include"scalar/scalarField.h"
#include"enum-field.h"
#include"propagator/laplacian.h"
//#include"scalar/varNQCD.h"
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
inline	void	sPropKernelXeon(void * __restrict__ m_, void * __restrict__ v_, const void * __restrict__ m2_, double *R, const double dz, const double c, const double d,
				const double ood2, const double LL, const double aMass2, const double gamma, const double fMom, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision)
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
		const double zR = *R;
		const double z2 = zR*zR;
		const double z4 = z2*z2;
		const double zQ = aMass2*z2*zR;

		const double LaLa = LL*2./z4;
		const double GGGG = pow(ood2,0.5)*gamma;
		const double GGiZ = GGGG/zR;
		const double mola = GGGG*dzc/2.;
		const double damp1 = 1./(1.+mola);
		const double damp2 = (1.-mola)*damp1;
		const double epsi = mola/(1.+mola);

#if	defined(__AVX512F__)
//		const size_t XC = (Lx<<2);
//		const size_t YC = (Lx>>2);

		const double    __attribute__((aligned(Align))) zQAux[8] = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };	// Only real part
		const double    __attribute__((aligned(Align))) zRAux[8] = { zR, 0., zR, 0., zR, 0., zR, 0. };	// Only real part
		const long long __attribute__((aligned(Align))) shfRg[8] = {6, 7, 0, 1, 2, 3, 4, 5 };
		const long long __attribute__((aligned(Align))) shfLf[8] = {2, 3, 4, 5, 6, 7, 0, 1 };

		const auto  vShRg = opCode(load_si512, shfRg);
		const auto  vShLf = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
//		const size_t XC = (Lx<<1);
//		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) zQAux[4] = { zQ, 0., zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zRAux[4] = { zR, 0., zR, 0. };	// Only real part
#else
//		const size_t XC = Lx;
//		const size_t YC = Lx;

		const double __attribute__((aligned(Align))) zQAux[2] = { zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zRAux[2] = { zR, 0. };	// Only real part

#endif
		const _MData_ zQVec  = opCode(load_pd, zQAux);
		const _MData_ zRVec  = opCode(load_pd, zRAux);
		const _MData_ fMVec  = opCode(set1_pd, fMom);

		/* begin the calculation */
		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, mPx, mMx, mPy;
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

				switch	(VQcd & VQCD_TYPE) {

					default:
					case	VQCD_1:
						mMx = opCode(add_pd, tmp,
							opCode(sub_pd, zQVec,
							opCode(mul_pd, mel,
								opCode(mul_pd,
									opCode(sub_pd, mPx, opCode(set1_pd, z2)),
									opCode(set1_pd, LL)))));
						break;

					case	VQCD_1_PQ_2:
						mMx = opCode(add_pd, tmp,
							opCode(sub_pd, zQVec,
							opCode(mul_pd, mel,
								opCode(mul_pd,
									opCode(sub_pd, opCode(mul_pd, mPx, mPx), opCode(set1_pd, z4)),
									opCode(set1_pd, LaLa)))));
						break;

					case	VQCD_2:
						mMx = opCode(add_pd, tmp,
							opCode(sub_pd,
								opCode(mul_pd, opCode(set1_pd, zQ), opCode(sub_pd, zRVec, mel)),
								opCode(mul_pd, mel,
									opCode(mul_pd,
										opCode(sub_pd, mPx, opCode(set1_pd, z2)),
										opCode(set1_pd, LL)))));
						break;
				}

				mPy = opCode(load_pd, &v[idxMz]);

				switch  (VQcd & VQCD_DAMP) {

					default:
					case    VQCD_NONE:
#if     defined(__AVX512F__) || defined(__FMA__)
					tmp = opCode(fmadd_pd, mMx, opCode(set1_pd, dzc), mPy);
#else
					tmp = opCode(add_pd, mPy, opCode(mul_pd, mMx, opCode(set1_pd, dzc)));
#endif
					break;

					case    VQCD_DAMP_RHO:
					{
						tmp = opCode(mul_pd, mel, mPy);
#if     defined(__AVX__)// || defined(__AVX512F__)
						auto vecmv = opCode(add_pd, opCode(permute_pd, tmp, 0b00000101), tmp);
#else
						auto vecmv = opCode(add_pd, opCode(shuffle_pd, tmp, tmp, 0b00000001), tmp);
#endif
						tmp = opCode(mul_pd, mel, mMx);
#if     defined(__AVX__)// || defined(__AVX512F__)
						auto vecma = opCode(add_pd, opCode(permute_pd, tmp, 0b00000101), tmp);
#else
						auto vecma = opCode(add_pd, opCode(shuffle_pd, tmp, tmp, 0b00000001), tmp);
#endif

#if     defined(__AVX512F__) || defined(__FMA__)
						auto veca = opCode(fmadd_pd, mel, opCode(set1_pd, GGiZ), mMx);
						tmp = opCode(sub_pd,
							opCode(fmadd_pd, veca, opCode(set1_pd, dzc), mPy),
							opCode(mul_pd, opCode(mul_pd, opCode(set1_pd, epsi), opCode(div_pd, mel, mPx)),
								opCode(fmadd_pd, vecmv, opCode(set1_pd, 2.), opCode(mul_pd, vecma, opCode(set1_pd, dzc)))));
#else
						auto veca = opCode(add_pd, mMx, opCode(mul_pd, mel, opCode(set1_pd, GGiZ)));
						tmp = opCode(sub_pd,
							opCode(add_pd, mPy, opCode(mul_pd, veca, opCode(set1_pd, dzc))),
							opCode(mul_pd, opCode(mul_pd, opCode(set1_pd, epsi), opCode(div_pd, mel, mPx)),
								opCode(add_pd,
									opCode(mul_pd, vecmv, opCode(set1_pd, 2.)),
									opCode(mul_pd, vecma, opCode(set1_pd, dzc)))));
#endif
						break;
					}

					case    VQCD_DAMP_ALL:
#if     defined(__AVX512F__) || defined(__FMA__)
					tmp = opCode(fmadd_pd, mPy, opCode(set1_pd, damp2), opCode(mul_pd, mMx, opCode(set1_pd, damp1*dzc)));
#else
					tmp = opCode(add_pd, opCode(mul_pd, mPy, opCode(set1_pd, damp2)), opCode(mul_pd, mMx, opCode(set1_pd, damp1*dzc)));
#endif
					break;
				}

				if (VQcd & VQCD_EVOL_RHO)
				{
					auto vecmv = opCode(mul_pd, mel, tmp);
#if     defined(__AVX__)// || defined(__AVX512F__)
					auto vecma = opCode(add_pd, opCode(permute_pd, vecmv, 0b00000101), vecmv);
#else
					auto vecma = opCode(add_pd, opCode(shuffle_pd, vecmv, vecmv, 0b00000001), vecmv);
#endif
					tmp = opCode(div_pd, opCode(mul_pd, mel, vecma), mPx);
				}

#if	defined(__AVX512F__) || defined(__FMA__)
				mPx = opCode(fmadd_pd, tmp, opCode(set1_pd, dzd), mel);
#else
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
		const float zR = *R;
		const float z2 = zR*zR;
		const float z4 = z2*z2;
		const float zQ = (float) (aMass2*z2*zR);

		const float LaLa = LL*2.f/z4;
		const float GGGG = pow(ood2,0.5f)*gamma;
		const float GGiZ = GGGG/zR;
		const float mola = GGGG*dzc/2.f;
		const float damp1 = 1.f/(1.f+mola);
		const float damp2 = (1.f-mola)*damp1;
		const float epsi = mola/(1.f+mola);

#if	defined(__AVX512F__)
//		const size_t XC = (Lx<<3);
//		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) zQAux[16] = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zRAux[16] = { zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f };
		const int   __attribute__((aligned(Align))) shfRg[16] = {14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
		const int   __attribute__((aligned(Align))) shfLf[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1};

		const auto  vShRg  = opCode(load_si512, shfRg);
		const auto  vShLf  = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
//		const size_t XC = (Lx<<2);
//		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) zQAux[8]  = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zRAux[8]  = { zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f };
#else
//		const size_t XC = (Lx<<1);
//		const size_t YC = (Lx>>1);

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
				switch	(VQcd & VQCD_TYPE) {

					default:
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
										opCode(sub_ps, opCode(mul_ps, mPx, mPx), opCode(set1_ps, z4)),
										opCode(set1_ps, LaLa)))));
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

				switch (VQcd & VQCD_DAMP) {

					default:
					case    VQCD_NONE:
#if     defined(__AVX512F__) || defined(__FMA__)
					tmp = opCode(fmadd_ps, mMx, opCode(set1_ps, dzc), mPy);
#else
					tmp = opCode(add_ps, mPy, opCode(mul_ps, mMx, opCode(set1_ps, dzc)));
#endif
					break;

					case    VQCD_DAMP_RHO:
					{
						tmp = opCode(mul_ps, mel, mPy);
#if     defined(__AVX__)// || defined(__AVX512F__)
						auto vecmv = opCode(add_ps, opCode(permute_ps, tmp, 0b10110001), tmp);
#else
						auto vecmv = opCode(add_ps, opCode(shuffle_ps, tmp, tmp, 0b10110001), tmp);
#endif

						// vecma
						tmp = opCode(mul_ps, mel, mMx);
#if     defined(__AVX__)// || defined(__AVX512F__)
						auto vecma = opCode(add_ps, opCode(permute_ps, tmp, 0b10110001), tmp);
#else
						auto vecma = opCode(add_ps, opCode(shuffle_ps, tmp, tmp, 0b10110001), tmp);
#endif
						// mPy=V veca=A mPx=|M|^2
						// V = (V+Adt) - (epsi M/|M|^2)(2 MV+ MA*dt)
#if     defined(__AVX512F__) || defined(__FMA__)
						// damping rho direction
						// add GGGG term to acceleration
						auto veca = opCode(fmadd_ps, mel, opCode(set1_ps, GGiZ), mMx);

						tmp = opCode(sub_ps,
							opCode(fmadd_ps, veca, opCode(set1_ps, dzc), mPy),
							opCode(mul_ps, opCode(mul_ps, opCode(set1_ps, epsi), opCode(div_ps, mel, mPx)),
								opCode(fmadd_ps, vecmv, opCode(set1_ps, 2.f), opCode(mul_ps, vecma, opCode(set1_ps, dzc)))));
#else
						// damping rho direction
						// add GGGG term to acceleration
						auto veca = opCode(add_ps, mMx, opCode(mul_ps, mel, opCode(set1_ps, GGiZ)));

						tmp = opCode(sub_ps,
							opCode(add_ps, mPy, opCode(mul_ps, veca, opCode(set1_ps, dzc))),
							opCode(mul_ps, opCode(mul_ps, opCode(set1_ps, epsi), opCode(div_ps, mel, mPx)),
								opCode(add_ps,
									opCode(mul_ps, vecmv, opCode(set1_ps, 2.f)),
									opCode(mul_ps, vecma, opCode(set1_ps, dzc)))));
#endif
					}
					break;

					case    VQCD_DAMP_ALL:
					// damping all directions implementation
#if     defined(__AVX512F__) || defined(__FMA__)
					tmp = opCode(fmadd_ps, mPy, opCode(set1_ps, damp2), opCode(mul_ps, mMx, opCode(set1_ps, damp1*dzc)));
#else
					tmp = opCode(add_ps, opCode(mul_ps, mPy, opCode(set1_ps, damp2)), opCode(mul_ps, mMx, opCode(set1_ps, damp1*dzc)));
#endif
					break;
				}

				// if only evolution along r is desired project v into rho (m) direction in complex space
				// we use vecma = m*v_update
				if (VQcd & VQCD_EVOL_RHO)
				{
					auto vecmv = opCode(mul_ps, mel, tmp);
#if     defined(__AVX__)// || defined(__AVX512F__)
					auto vecma = opCode(add_ps, opCode(permute_ps, vecmv, 0b10110001), vecmv);
#else
					auto vecma = opCode(add_ps, opCode(shuffle_ps, vecmv, vecmv, 0b10110001), vecmv);
#endif
					tmp   = opCode(div_ps, opCode(mul_ps, mel, vecma), mPx);
				}

#if	defined(__AVX512F__) || defined(__FMA__)
				mPx = opCode(fmadd_ps, tmp, opCode(set1_ps, dzd), mel);
#else
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
