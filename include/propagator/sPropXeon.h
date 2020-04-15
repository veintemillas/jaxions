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
inline	void	sPropKernelXeon(void * __restrict__ m_, void * __restrict__ v_, const void * __restrict__ m2_, PropParms ppar, const double dz, const double c, const double d,
				const size_t Vo, const size_t Vf, FieldPrecision precision)
{
	const size_t NN    = ppar.Ng;
	const size_t Lx    = ppar.Lx;
	const size_t Sf    = Lx*Lx;
	const size_t NSf   = Sf*NN;

	const double R     = ppar.R;
	const double ood2  = ppar.ood2a;
	const double mA2   = ppar.massA2;
	const double gamma = ppar.gamma;
	const double LL    = ppar.lambda;
	const double Rpp   = ppar.Rpp;
	const double fMom  = ppar.fMom1;

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
		const double R2 = R*R;
		const double zQ = mA2*R2*R;
		double gasa;
		switch	(VQcd & V_QCD) {
				case	V_QCD2:
				gasa = (mA2*R2)/2;
				break;

				default:
				case	V_QCDC:
				gasa = (mA2*R2*R2);
				break;
		}
		//For V_QCD2 & V_QCDC
		const double zN = gasa;

		const double R4 = R2*R2;
		const double LaLa = LL*2./R4;
		const double GGGG = gamma/R;
		const double mola = GGGG*dzc/2.;
		const double damp1 = 1./(1.+mola);
		const double damp2 = (1.-mola)*damp1;
		const double epsi = mola/(1.+mola);



#if	defined(__AVX512F__)
//		const size_t XC = (Lx<<2);
//		const size_t YC = (Lx>>2);

		const double    __attribute__((aligned(Align))) zQAux[8] = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };	// Only real part
		const double    __attribute__((aligned(Align))) zNAux[8] = { zN,-zN, zN,-zN, zN,-zN, zN,-zN };	// to complex congugate
		const double    __attribute__((aligned(Align))) zRAux[8] = {  R, 0.,  R, 0.,  R, 0.,  R, 0. };	// Only real part
		const long long __attribute__((aligned(Align))) shfRg[8] = {6, 7, 0, 1, 2, 3, 4, 5 };
		const long long __attribute__((aligned(Align))) shfLf[8] = {2, 3, 4, 5, 6, 7, 0, 1 };

		const auto  vShRg = opCode(load_si512, shfRg);
		const auto  vShLf = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
//		const size_t XC = (Lx<<1);
//		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) zQAux[4] = { zQ, 0., zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zNAux[4] = { zN,-zN, zN,-zN };	// to complex congugate
		const double __attribute__((aligned(Align))) zRAux[4] = {  R, 0.,  R, 0. };	// Only real part
#else
//		const size_t XC = Lx;
//		const size_t YC = Lx;

		const double __attribute__((aligned(Align))) zQAux[2] = { zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zNAux[2] = { zN,-zN };	// to complex congugate
		const double __attribute__((aligned(Align))) zRAux[2] = {  R, 0. };	// Only real part

#endif
		const _MData_ zQVec  = opCode(load_pd, zQAux);
		const _MData_ zNVec  = opCode(load_pd, zNAux);
		const _MData_ zRVec  = opCode(load_pd, zRAux);
		const _MData_ fMVec  = opCode(set1_pd, fMom);

		/* begin the calculation */
		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, mPx, mMx, mPy, tmp2;
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

/* mMx = acceleration
			 = lap - Phi *
						( PQ-part
							- R''p/R )
*/
				switch	(VQcd & V_PQ) {
					case V_PQ1:
						mMx = opCode(sub_pd, tmp,
										opCode(mul_pd, mel,
											opCode(sub_pd,
												opCode(mul_pd, opCode(sub_pd, mPx, opCode(set1_pd, R2)),
													opCode(set1_pd, LL)),
												opCode(set1_pd, Rpp))));
					break;
					case V_PQ2:
						mMx = opCode(sub_pd, tmp,
										opCode(mul_pd, mel,
											opCode(sub_pd,
												opCode(mul_pd,
													opCode(sub_pd, opCode(mul_pd, mPx, mPx), opCode(set1_pd, R4)),
														opCode(mul_pd, mPx, opCode(set1_pd, LaLa))),
											opCode(set1_pd, Rpp))));
					break;
				}
				/* mMx = mMx + VQCD part */
				switch	(VQcd & V_QCD) {
					case V_QCD1:
						mMx = opCode(add_pd, mMx, zQVec);
					break;
					case V_QCDV:
						mMx = opCode(add_pd, mMx, opCode(mul_pd, opCode(set1_pd, zQ), opCode(sub_pd, zRVec, mel)));
					break;
					case V_QCD2:
						mMx = opCode(sub_pd, mMx, opCode(mul_pd,zNVec,mel));
					break;
					case V_QCDC:
						tmp2 = opCode(div_pd,
										opCode(vqcd0_pd,mel),
											opCode(sqrt_pd, opCode(mul_pd, mPx, opCode(mul_pd, mPx, mPx) ) ) ); //
						mMx = opCode(add_pd, mMx, opCode(mul_pd, zNVec, tmp2));
					break;
					default:
					case V_QCDL:
					case V_QCD0:
					break;
				}

				mPy = opCode(load_pd, &v[idxMz]);

				switch  (VQcd & V_DAMP) {

					default:
					case    V_NONE:
#if     defined(__AVX512F__) || defined(__FMA__)
					tmp = opCode(fmadd_pd, mMx, opCode(set1_pd, dzc), mPy);
#else
					tmp = opCode(add_pd, mPy, opCode(mul_pd, mMx, opCode(set1_pd, dzc)));
#endif
					break;

					case    V_DAMP_RHO:
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
					tmp = opCode(sub_pd,
						opCode(fmadd_pd, mMx, opCode(set1_pd, dzc), mPy),
						opCode(mul_pd, opCode(mul_pd, opCode(set1_pd, epsi), opCode(div_pd, mel, mPx)),
							opCode(fmadd_pd, opCode(sub_pd,vecmv,opCode(div_pd,mPx,opCode(set1_pd, R))), opCode(set1_pd, 2.0), opCode(mul_pd, vecma, opCode(set1_pd, dzc)))));
#else
					tmp = opCode(sub_pd,
						opCode(add_pd, mPy, opCode(mul_pd, mMx, opCode(set1_pd, dzc))),
						opCode(mul_pd, opCode(mul_pd, opCode(set1_pd, epsi), opCode(div_pd, mel, mPx)),
							opCode(add_pd,
								opCode(mul_pd, opCode(sub_pd,vecmv,opCode(div_pd,mPx,opCode(set1_pd, R))), opCode(set1_pd, 2.0)),
								opCode(mul_pd, vecma, opCode(set1_pd, dzc)))));
#endif
						break;
					}

					case    V_DAMP_ALL:
#if     defined(__AVX512F__) || defined(__FMA__)
					tmp = opCode(fmadd_pd, mPy, opCode(set1_pd, damp2), opCode(mul_pd, mMx, opCode(set1_pd, damp1*dzc)));
#else
					tmp = opCode(add_pd, opCode(mul_pd, mPy, opCode(set1_pd, damp2)), opCode(mul_pd, mMx, opCode(set1_pd, damp1*dzc)));
#endif
					break;
				}

				if (VQcd & V_EVOL_RHO)
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
		const float Rf  = (float) R;
		const float R2 = Rf*Rf;
		const float R4 = R2*R2;
		const float zQ = (float) (mA2*R2* Rf);

		float gasa;
		switch	(VQcd & V_QCD) {
				case	V_QCD2:
				gasa = (float) (mA2*R2)/2;
				break;

				default:
				case	V_QCDC:
				gasa = (float) (mA2*R2*R2);
				break;
		}
		//For V_QCD2 & V_QCDC
		const float zN = gasa;

		const float LaLa = LL*2.f/R4;
		const float GGGG = gamma/Rf;
		const float mola = GGGG*dzc/2.f;
		const float damp1 = 1.f/(1.f+mola);
		const float damp2 = (1.f-mola)*damp1;
		const float epsi = mola/(1.f+mola);

#if	defined(__AVX512F__)
//		const size_t XC = (Lx<<3);
//		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) zQAux[16] = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zNAux[16] = { zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN };
		const float __attribute__((aligned(Align))) zRAux[16] = { Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f };
		const int   __attribute__((aligned(Align))) shfRg[16] = {14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
		const int   __attribute__((aligned(Align))) shfLf[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1};

		const auto  vShRg  = opCode(load_si512, shfRg);
		const auto  vShLf  = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
//		const size_t XC = (Lx<<2);
//		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) zQAux[8]  = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zNAux[8]  = { zN, -zN, zN, -zN, zN, -zN, zN, -zN };
		const float __attribute__((aligned(Align))) zRAux[8]  = { Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f };
#else
//		const size_t XC = (Lx<<1);
//		const size_t YC = (Lx>>1);

		const float __attribute__((aligned(Align))) zQAux[4]  = { zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zNAux[4]  = { zN, -zN, zN, -zN };
		const float __attribute__((aligned(Align))) zRAux[4]  = { Rf, 0.f, Rf, 0.f };
#endif
		const _MData_ zQVec  = opCode(load_ps, zQAux);
		const _MData_ zNVec  = opCode(load_ps, zNAux);
		const _MData_ zRVec  = opCode(load_ps, zRAux);
		const _MData_ fMVec  = opCode(set1_ps, fMom);

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, mPx, mPy, mMx, tmp2;
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
				/* mMx = acceleration
							 = lap - Phi *
										( PQ-part
											- R''p/R )
				*/
				switch	(VQcd & V_PQ) {
					case V_PQ1:
						mMx = opCode(sub_ps, tmp,
										opCode(mul_ps, mel,
											opCode(sub_ps,
												opCode(mul_ps, opCode(sub_ps, mPx, opCode(set1_ps, R2)),
													opCode(set1_ps, LL)),
												opCode(set1_ps, Rpp))));
					break;
					case V_PQ2:
						mMx = opCode(sub_ps, tmp,
										opCode(mul_ps, mel,
											opCode(sub_ps,
												opCode(mul_ps,
													opCode(sub_ps, opCode(mul_ps, mPx, mPx), opCode(set1_ps, R4)),
														opCode(mul_ps, mPx, opCode(set1_ps, LaLa))),
											opCode(set1_ps, Rpp))));
					break;
				}
				/* mMx = mMx + VQCD part */
				switch	(VQcd & V_QCD) {
					case V_QCD1:
						mMx = opCode(add_ps, mMx, zQVec);
					break;
					case V_QCDV:
						mMx = opCode(add_ps, mMx, opCode(mul_ps, opCode(set1_ps, zQ), opCode(sub_ps, zRVec, mel)));
					break;
					case V_QCD2:
						mMx = opCode(sub_ps, mMx, opCode(mul_ps,zNVec,mel));
					break;
					case V_QCDC:
						tmp2 = opCode(div_ps,
										opCode(vqcd0_ps,mel),
											opCode(sqrt_ps, opCode(mul_ps, mPx, opCode(mul_ps, mPx, mPx) ) ) ); //
						mMx = opCode(add_ps, mMx, opCode(mul_ps, zNVec, tmp2));
					break;
					default:
					case V_QCDL:
					case V_QCD0:
					break;
				}

				mPy = opCode(load_ps, &v[idxMz]);

				switch (VQcd & V_DAMP) {

					default:
					case    V_NONE:
#if     defined(__AVX512F__) || defined(__FMA__)
					tmp = opCode(fmadd_ps, mMx, opCode(set1_ps, dzc), mPy);
#else
					tmp = opCode(add_ps, mPy, opCode(mul_ps, mMx, opCode(set1_ps, dzc)));
#endif
					break;

					case    V_DAMP_RHO:
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
// A*dzc + mPy - epsi (M/|M|^2)(2*(vecmv-|M|^2/t) +vecma dzc)
				tmp = opCode(sub_ps,
					opCode(fmadd_ps, mMx, opCode(set1_ps, dzc), mPy),
					opCode(mul_ps, opCode(mul_ps, opCode(set1_ps, epsi), opCode(div_ps, mel, mPx)),
						opCode(fmadd_ps, opCode(sub_ps,vecmv,opCode(div_ps,mPx,opCode(set1_ps, Rf))), opCode(set1_ps, 2.f), opCode(mul_ps, vecma, opCode(set1_ps, dzc)))));
#else
				tmp = opCode(sub_ps,
					opCode(add_ps, mPy, opCode(mul_ps, mMx, opCode(set1_ps, dzc))),
					opCode(mul_ps, opCode(mul_ps, opCode(set1_ps, epsi), opCode(div_ps, mel, mPx)),
						opCode(add_ps,
							opCode(mul_ps, opCode(sub_ps,vecmv,opCode(div_ps,mPx,opCode(set1_ps, Rf))), opCode(set1_ps, 2.f)),
							opCode(mul_ps, vecma, opCode(set1_ps, dzc)))));
#endif
					}
					break;

					case    V_DAMP_ALL:
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
				if (VQcd & V_EVOL_RHO)
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
