#include<cstdio>
#include<cmath>
#include"scalar/scalarField.h"
#include"enum-field.h"
#include"propagator/RKParms.h"

#ifdef USE_XEON
	#include"comms/comms.h"
	#include"utils/xeonDefs.h"
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


#define	tV	(2.*M_PI)
#define	M_PI2	(M_PI *M_PI)
#define	M_PI4	(M_PI2*M_PI2)
#define	M_PI6	(M_PI4*M_PI2)

#ifdef	__MIC__
	#define	_MData_ __m512d
#elif	defined(__AVX__)
	#define	_MData_ __m256d
#else
	#define	_MData_ __m128d
#endif

inline _MData_	opCode(sin_pd, _MData_ x)
{
	_MData_ tmp2, tmp3, tmp5, a, b, c;
	static const double a_s = -0.0415758*4., b_s = 0.00134813*6., c_s = -(1+M_PI2*a_s+M_PI4*b_s)/(M_PI6);

	a = opCode(set1_pd, a_s);
	b = opCode(set1_pd, b_s);
	c = opCode(set1_pd, c_s);

	tmp2 = opCode(mul_pd, x, x);
	tmp3 = opCode(mul_pd, tmp2, x);
	tmp5 = opCode(mul_pd, tmp3, tmp2);
	return opCode(add_pd, x, opCode(add_pd,
		opCode(add_pd,
			opCode(mul_pd, tmp3, a),
			opCode(mul_pd, tmp5, b)),
		opCode(mul_pd, c, opCode(mul_pd, tmp2, tmp5))));
}

#undef	_MData_

#ifdef	__MIC__
	#define	_MData_ __m512
#elif	defined(__AVX__)
	#define	_MData_ __m256
#else
	#define	_MData_ __m128
#endif

inline _MData_	opCode(sin_ps, _MData_ x)
{
	_MData_ tmp2, tmp3, tmp5, a, b, c;
	static const float a_s = -0.0415758f*4.f, b_s = 0.00134813f*6.f, c_s = -(1+M_PI2*a_s+M_PI4*b_s)/(M_PI6);

	a = opCode(set1_ps, a_s);
	b = opCode(set1_ps, b_s);
	c = opCode(set1_ps, c_s);

	tmp2 = opCode(mul_ps, x, x);
	tmp3 = opCode(mul_ps, tmp2, x);
	tmp5 = opCode(mul_ps, tmp3, tmp2);
	return opCode(add_ps, x, opCode(add_ps,
		opCode(add_ps,
			opCode(mul_ps, tmp3, a),
			opCode(mul_ps, tmp5, b)),
		opCode(mul_ps, c, opCode(mul_ps, tmp2, tmp5))));
}

#undef	_MData_

#ifdef USE_XEON
__attribute__((target(mic)))
#endif
void	propThetaKernelXeon(const void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_, double *z, const double dz, const double c, const double d,
			    const double ood2, const double nQcd, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision)
{
	const size_t Sf = Lx*Lx;

	if (precision == FIELD_DOUBLE)
	{
#ifdef	__MIC__
	#define	_MData_ __m512d
	#define	_MInt_  __m512i
	#define	step 8
#elif	defined(__AVX__)
	#define	_MData_ __m256d
	#define	step 4
#else
	#define	_MData_ __m128d
	#define	step 2
#endif

#ifdef	USE_XEON
		const double * __restrict__ m	= (const double * __restrict__) m_;
		double * __restrict__ v		= (double * __restrict__) v_;
		double * __restrict__ m2	= (double * __restrict__) m2_;

		__assume_aligned(m, Align);
		__assume_aligned(v, Align);
		__assume_aligned(m2, Align);
#else
		const double * __restrict__ m	= (const double * __restrict__) __builtin_assume_aligned (m_, Align);
		double * __restrict__ v		= (double * __restrict__) __builtin_assume_aligned (v_, Align);
		double * __restrict__ m2	= (double * __restrict__) __builtin_assume_aligned (m2_, Align);
#endif

		const double dzc = dz*c;
		const double dzd = dz*d;
		const double zR = *z;
		const double iZ = 1./zR;
		const double zQ = 9.*pow(zR, nQcd+3.);

#ifdef	__MIC__
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const double __attribute__((aligned(Align))) tpAux[8]  = { tV, tV, tV, tV, tV, tV, tV, tV };
		const double __attribute__((aligned(Align))) zQAux[8]  = { zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ };
		const double __attribute__((aligned(Align))) izAux[8]  = { iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ };
		const double __attribute__((aligned(Align))) d2Aux[8]  = { ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2 };
		const double __attribute__((aligned(Align))) dzcAux[8] = { dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc };
		const double __attribute__((aligned(Align))) dzdAux[8] = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd };

		const int    __attribute__((aligned(Align))) shfRg[16] = {14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
		const int    __attribute__((aligned(Align))) shfLf[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1};
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const double __attribute__((aligned(Align))) tpAux[4]  = { tV, tV, tV, tV };
		const double __attribute__((aligned(Align))) zQAux[4]  = { zQ, zQ, zQ, zQ };
		const double __attribute__((aligned(Align))) izAux[4]  = { iZ, iZ, iZ, iZ };
		const double __attribute__((aligned(Align))) d2Aux[4]  = { ood2, ood2, ood2, ood2 };
		const double __attribute__((aligned(Align))) dzcAux[4] = { dzc, dzc, dzc, dzc };
		const double __attribute__((aligned(Align))) dzdAux[4] = { dzd, dzd, dzd, dzd };
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) tpAux[2]  = { tV, tV };
		const double __attribute__((aligned(Align))) zQAux[2]  = { zQ, zQ };
		const double __attribute__((aligned(Align))) izAux[2]  = { iZ, iZ };
		const double __attribute__((aligned(Align))) d2Aux[2]  = { ood2, ood2 };
		const double __attribute__((aligned(Align))) dzcAux[2] = { dzc, dzc };
		const double __attribute__((aligned(Align))) dzdAux[2] = { dzd, dzd };

#endif
		const _MData_ tpVec  = opCode(set1_pd, tV);
		const _MData_ zQVec  = opCode(set1_pd, zQ);
		const _MData_ izVec  = opCode(set1_pd, iZ);
		const _MData_ d2Vec  = opCode(set1_pd, ood2);
		const _MData_ dzcVec = opCode(set1_pd, dzc);
		const _MData_ dzdVec = opCode(set1_pd, dzd);

#ifdef __MIC__
		const _MInt_  vShRg  = opCode(load_si512, shfRg);
		const _MInt_  vShLf  = opCode(load_si512, shfLf);
		const _MData_ z0Vec  = opCode(load_ps, z0Aux);
#endif

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, vel, tpP, tpM, mPy, mMy, acu, v2p, tP2, tM2;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxPx, idxMx, idxPy, idxMy, idxPz, idxMz;

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
#ifdef	__MIC__
					mMy = opCode(castsi512_pd, opCode(permutevar_epi32, vShRg, opCode(castpd_si512, opCode(load_pd, &m[idxMy]))));
#elif	defined(__AVX2__)	//AVX2
					mMy = opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_pd, &m[idxMy])), opCode(setr_epi32, 6,7,0,1,2,3,4,5)));
#elif	defined(__AVX__)
					tpM = opCode(permute_pd, opCode(load_pd, &m[idxMy]), 0b00000101);
					tpP = opCode(permute2f128_pd, tpM, tpM, 0b00000001);
					mMy = opCode(blend_pd, tpM, tpP, 0b00000101);
#else
					tpM = opCode(load_pd, &m[idxMy]);
					mMy = opCode(shuffle_pd, tpM, tpM, 0x00000001);
#endif
				}
				else
				{
					idxMy = idx - XC;
					mMy = opCode(load_pd, &m[idxMy]);

					if (X[1] == YC-1)
					{
						idxPy = idx - Sf + XC;
#ifdef	__MIC__
						mPy = opCode(castsi512_pd, opCode(permutevar_epi32, vShLf, opCode(castpd_si512, opCode(load_pd, &m[idxPy]))));
#elif	defined(__AVX2__)	//AVX2
						tmp = opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_pd, &m[idxPy])), opCode(setr_epi32, 2,3,4,5,6,7,0,1)));
#elif	defined(__AVX__)
						tpM = opCode(permute_pd, opCode(load_pd, &m[idxPy]), 0b00000101);
						tpP = opCode(permute2f128_pd, tpM, tpM, 0b00000001);
						mPy = opCode(blend_pd, tpM, tpP, 0b00001010);
#else
						tpP = opCode(load_pd, &m[idxPy]);
						mPy = opCode(shuffle_pd, tpP, tpP, 0x00000001);
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

				mel = opCode(load_pd, &m[idx]);

				/*	idxPx	*/

				vel = opCode(sub_pd, opCode(load_pd, &m[idxPx]), mel);
				tpM = opCode(sub_pd, vel, tpVec);
				tpP = opCode(add_pd, vel, tpVec);
				v2p = opCode(mul_pd, vel, vel);
				tP2 = opCode(mul_pd, tpP, tpP);
				tM2 = opCode(mul_pd, tpM, tpM);
#ifdef	__MIC__
				tmp = opCode(gmin_pd, opCode(gmin_pd, tP2, tM2), v2p);
				acu = opCode(mask_add_pd, z0Vec, opCode(cmp_pd, tmp, tP2, _CMP_EQ_OQ), z0Vec, tpP);
				acu = opCode(mask_add_pd, acu,   opCode(cmp_pd, tmp, tM2, _CMP_EQ_OQ), acu,   tpM);
				acu = opCode(mask_add_pd, acu,   opCode(cmp_pd, tmp, v2p, _CMP_EQ_OQ), acu,   vel);
#elif	defined(__AVX__)
				acu = opCode(setzero_pd);
				tmp = opCode(min_pd, opCode(min_pd, tP2, tM2), v2p);
				acu = opCode(add_pd,
					opCode(add_pd,
						opCode(and_pd, opCode(cmp_pd, tmp, tP2, _CMP_EQ_OQ), tpP),
						opCode(and_pd, opCode(cmp_pd, tmp, tM2, _CMP_EQ_OQ), tpM)),
					opCode(and_pd, opCode(cmp_pd, tmp, v2p, _CMP_EQ_OQ), vel));
#else
				acu = opCode(setzero_pd);
				tmp = opCode(min_pd, opCode(min_pd, tP2, tM2), v2p);
				acu = opCode(add_pd,
					opCode(add_pd,
						opCode(and_pd, opCode(cmpeq_pd, tmp, tP2), tpP),
						opCode(and_pd, opCode(cmpeq_pd, tmp, tM2), tpM)),
					opCode(and_pd, opCode(cmpeq_pd, tmp, v2p), vel));
#endif
				/*	idxMx	*/

				vel = opCode(sub_pd, opCode(load_pd, &m[idxMx]), mel);
				tpM = opCode(sub_pd, vel, tpVec);
				tpP = opCode(add_pd, vel, tpVec);
				v2p = opCode(mul_pd, vel, vel);
				tP2 = opCode(mul_pd, tpP, tpP);
				tM2 = opCode(mul_pd, tpM, tpM);
#ifdef	__MIC__
				tmp = opCode(gmin_pd, opCode(gmin_pd, tP2, tM2), v2p);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, tmp, tP2, _CMP_EQ_OQ), acu, tpP);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, tmp, tM2, _CMP_EQ_OQ), acu, tpM);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, tmp, v2p, _CMP_EQ_OQ), acu, vel);
#elif	defined(__AVX__)
				tmp = opCode(min_pd, opCode(min_pd, tP2, tM2), v2p);
				acu = opCode(add_pd, acu, opCode(add_pd,
					opCode(add_pd,
						opCode(and_pd, opCode(cmp_pd, tmp, tP2, _CMP_EQ_OQ), tpP),
						opCode(and_pd, opCode(cmp_pd, tmp, tM2, _CMP_EQ_OQ), tpM)),
					opCode(and_pd, opCode(cmp_pd, tmp, v2p, _CMP_EQ_OQ), vel)));
#else
				tmp = opCode(min_pd, opCode(min_pd, tP2, tM2), v2p);
				acu = opCode(add_pd, acu, opCode(add_pd,
					 opCode(add_pd,
						opCode(and_pd, opCode(cmpeq_pd, tmp, tP2), tpP),
						opCode(and_pd, opCode(cmpeq_pd, tmp, tM2), tpM)),
					opCode(and_pd, opCode(cmpeq_pd, tmp, v2p), vel)));
#endif
				/*	idxPz	*/

				vel = opCode(sub_pd, opCode(load_pd, &m[idxPz]), mel);
				tpM = opCode(sub_pd, vel, tpVec);
				tpP = opCode(add_pd, vel, tpVec);
				v2p = opCode(mul_pd, vel, vel);
				tP2 = opCode(mul_pd, tpP, tpP);
				tM2 = opCode(mul_pd, tpM, tpM);
#ifdef	__MIC__
				tmp = opCode(gmin_pd, opCode(gmin_pd, tP2, tM2), v2p);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, tmp, tP2, _CMP_EQ_OQ), acu, tpP);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, tmp, tM2, _CMP_EQ_OQ), acu, tpM);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, tmp, v2p, _CMP_EQ_OQ), acu, vel);
#elif	defined(__AVX__)
				tmp = opCode(min_pd, opCode(min_pd, tP2, tM2), v2p);
				acu = opCode(add_pd, acu, opCode(add_pd,
					opCode(add_pd,
						opCode(and_pd, opCode(cmp_pd, tmp, tP2, _CMP_EQ_OQ), tpP),
						opCode(and_pd, opCode(cmp_pd, tmp, tM2, _CMP_EQ_OQ), tpM)),
					opCode(and_pd, opCode(cmp_pd, tmp, v2p, _CMP_EQ_OQ), vel)));
#else
				tmp = opCode(min_pd, opCode(min_pd, tP2, tM2), v2p);
				acu = opCode(add_pd, acu, opCode(add_pd,
					 opCode(add_pd,
						opCode(and_pd, opCode(cmpeq_pd, tmp, tP2), tpP),
						opCode(and_pd, opCode(cmpeq_pd, tmp, tM2), tpM)),
					opCode(and_pd, opCode(cmpeq_pd, tmp, v2p), vel)));
#endif
				/*	idxMz	*/

				vel = opCode(sub_pd, opCode(load_pd, &m[idxMz]), mel);
				tpM = opCode(sub_pd, vel, tpVec);
				tpP = opCode(add_pd, vel, tpVec);
				v2p = opCode(mul_pd, vel, vel);
				tP2 = opCode(mul_pd, tpP, tpP);
				tM2 = opCode(mul_pd, tpM, tpM);
#ifdef	__MIC__
				tmp = opCode(gmin_pd, opCode(gmin_pd, tP2, tM2), v2p);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, tmp, tP2, _CMP_EQ_OQ), acu, tpP);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, tmp, tM2, _CMP_EQ_OQ), acu, tpM);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, tmp, v2p, _CMP_EQ_OQ), acu, vel);
#elif	defined(__AVX__)
				tmp = opCode(min_pd, opCode(min_pd, tP2, tM2), v2p);
				acu = opCode(add_pd, acu, opCode(add_pd,
					opCode(add_pd,
						opCode(and_pd, opCode(cmp_pd, tmp, tP2, _CMP_EQ_OQ), tpP),
						opCode(and_pd, opCode(cmp_pd, tmp, tM2, _CMP_EQ_OQ), tpM)),
					opCode(and_pd, opCode(cmp_pd, tmp, v2p, _CMP_EQ_OQ), vel)));
#else
				tmp = opCode(min_pd, opCode(min_pd, tP2, tM2), v2p);
				acu = opCode(add_pd, acu, opCode(add_pd,
					 opCode(add_pd,
						opCode(and_pd, opCode(cmpeq_pd, tmp, tP2), tpP),
						opCode(and_pd, opCode(cmpeq_pd, tmp, tM2), tpM)),
					opCode(and_pd, opCode(cmpeq_pd, tmp, v2p), vel)));
#endif
				/*	idxPy	*/

				vel = opCode(sub_pd, mPy, mel);
				tpM = opCode(sub_pd, vel, tpVec);
				tpP = opCode(add_pd, vel, tpVec);
				v2p = opCode(mul_pd, vel, vel);
				tP2 = opCode(mul_pd, tpP, tpP);
				tM2 = opCode(mul_pd, tpM, tpM);
#ifdef	__MIC__
				tmp = opCode(gmin_pd, opCode(gmin_pd, tP2, tM2), v2p);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, tmp, tP2, _CMP_EQ_OQ), acu, tpP);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, tmp, tM2, _CMP_EQ_OQ), acu, tpM);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, tmp, v2p, _CMP_EQ_OQ), acu, vel);
#elif	defined(__AVX__)
				tmp = opCode(min_pd, opCode(min_pd, tP2, tM2), v2p);
				acu = opCode(add_pd, acu, opCode(add_pd,
					opCode(add_pd,
						opCode(and_pd, opCode(cmp_pd, tmp, tP2, _CMP_EQ_OQ), tpP),
						opCode(and_pd, opCode(cmp_pd, tmp, tM2, _CMP_EQ_OQ), tpM)),
					opCode(and_pd, opCode(cmp_pd, tmp, v2p, _CMP_EQ_OQ), vel)));
#else
				tmp = opCode(min_pd, opCode(min_pd, tP2, tM2), v2p);
				acu = opCode(add_pd, acu, opCode(add_pd,
					 opCode(add_pd,
						opCode(and_pd, opCode(cmpeq_pd, tmp, tP2), tpP),
						opCode(and_pd, opCode(cmpeq_pd, tmp, tM2), tpM)),
					opCode(and_pd, opCode(cmpeq_pd, tmp, v2p), vel)));
#endif
				/*	idxMy	*/

				vel = opCode(sub_pd, mMy, mel);
				tpM = opCode(sub_pd, vel, tpVec);
				tpP = opCode(add_pd, vel, tpVec);
				v2p = opCode(mul_pd, vel, vel);
				tP2 = opCode(mul_pd, tpP, tpP);
				tM2 = opCode(mul_pd, tpM, tpM);
#ifdef	__MIC__
				tmp = opCode(gmin_pd, opCode(gmin_pd, tP2, tM2), v2p);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, tmp, tP2, _CMP_EQ_OQ), acu, tpP);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, tmp, tM2, _CMP_EQ_OQ), acu, tpM);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, tmp, v2p, _CMP_EQ_OQ), acu, vel);
#elif	defined(__AVX__)
				tmp = opCode(min_pd, opCode(min_pd, tP2, tM2), v2p);
				acu = opCode(add_pd, acu, opCode(add_pd,
					opCode(add_pd,
						opCode(and_pd, opCode(cmp_pd, tmp, tP2, _CMP_EQ_OQ), tpP),
						opCode(and_pd, opCode(cmp_pd, tmp, tM2, _CMP_EQ_OQ), tpM)),
					opCode(and_pd, opCode(cmp_pd, tmp, v2p, _CMP_EQ_OQ), vel)));
#else
				tmp = opCode(min_pd, opCode(min_pd, tP2, tM2), v2p);
				acu = opCode(add_pd, acu, opCode(add_pd,
					 opCode(add_pd,
						opCode(and_pd, opCode(cmpeq_pd, tmp, tP2), tpP),
						opCode(and_pd, opCode(cmpeq_pd, tmp, tM2), tpM)),
					opCode(and_pd, opCode(cmpeq_pd, tmp, v2p), vel)));
#endif

//#ifdef	__MIC__		NO SE SI VA A FUNCIONAR PORQUE CREO QUE EL SENO NO ESTA DEFINIDO EN KNC

//#else
				tpM = opCode(sub_pd,
					opCode(mul_pd, acu, d2Vec),
					opCode(mul_pd, zQVec, opCode(sin_pd, opCode(mul_pd, mel, izVec))));
//#endif
				mPy = opCode(load_pd, &v[idxMz]);
#if	defined(__MIC__) || defined(__FMA__)
				tmp = opCode(fmadd_pd, tpM, dzcVec, mPy);
				tpP = opCode(fmadd_pd, tmp, dzdVec, mel);
#else
				tmp = opCode(add_pd, mPy, opCode(mul_pd, tpM, dzcVec));
				tpP = opCode(add_pd, mel, opCode(mul_pd, tmp, dzdVec));
#endif

				/* Make sure the result is between -pi and pi	*/

				mMy = opCode(sub_pd, tpP, tpVec);
				mPy = opCode(add_pd, tpP, tpVec);
				v2p = opCode(mul_pd, tpP, tpP);
				tP2 = opCode(mul_pd, mPy, mPy);
				tM2 = opCode(mul_pd, mMy, mMy);
				acu = opCode(setzero_pd);
#ifdef	__MIC__
				vel = opCode(gmin_pd, opCode(gmin_pd, tP2, tM2), v2p);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, vel, tP2, _CMP_EQ_OQ), acu, mPy);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, vel, tM2, _CMP_EQ_OQ), acu, mMy);
				acu = opCode(mask_add_pd, acu, opCode(cmp_pd, vel, v2p, _CMP_EQ_OQ), acu, tpP);
#elif	defined(__AVX__)
				vel = opCode(min_pd, opCode(min_pd, tP2, tM2), v2p);
				acu = opCode(add_pd, acu, opCode(add_pd,
					opCode(add_pd,
						opCode(and_pd, opCode(cmp_pd, vel, tP2, _CMP_EQ_OQ), mPy),
						opCode(and_pd, opCode(cmp_pd, vel, tM2, _CMP_EQ_OQ), mMy)),
					opCode(and_pd, opCode(cmp_pd, vel, v2p, _CMP_EQ_OQ), tpP)));
#else
				vel = opCode(min_pd, opCode(min_pd, tP2, tM2), v2p);
				acu = opCode(add_pd, acu, opCode(add_pd,
					 opCode(add_pd,
						opCode(and_pd, opCode(cmpeq_pd, vel, tP2), mPy),
						opCode(and_pd, opCode(cmpeq_pd, vel, tM2), mMy)),
					opCode(and_pd, opCode(cmpeq_pd, vel, v2p), tpP)));
#endif

				opCode(store_pd, &v[idxMz], tmp);
				opCode(store_pd, &m2[idx],  acu);
			}
		}
#undef	_MData_
#undef	step
#ifdef	__MIC__
	#undef	_MInt_
#endif
	}
	else if (precision == FIELD_SINGLE)
	{
#ifdef	__MIC__
	#define	_MData_ __m512
	#define	step 16
#elif	defined(__AVX__)
	#define	_MData_ __m256
	#define	step 8
#else
	#define	_MData_ __m128
	#define	step 4
#endif

#ifdef	USE_XEON
		const float * __restrict__ m	= (const float * __restrict__) m_;
		float * __restrict__ v		= (float * __restrict__) v_;
		float * __restrict__ m2		= (float * __restrict__) m2_;

		__assume_aligned(m, Align);
		__assume_aligned(v, Align);
		__assume_aligned(m2, Align);
#else
		const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_, Align);
		float * __restrict__ v		= (float * __restrict__) __builtin_assume_aligned (v_, Align);
		float * __restrict__ m2		= (float * __restrict__) __builtin_assume_aligned (m2_, Align);
#endif

		const float dzc = dz*c;
		const float dzd = dz*d;
		const float zR = *z;
		const float iZ = 1./zR;
		const float zQ = 9.*powf(zR, nQcd+3.);

#ifdef	__MIC__
		const size_t XC = (Lx<<4);
		const size_t YC = (Lx>>4);

		const float __attribute__((aligned(Align))) z0Aux[16]  = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };
		const float __attribute__((aligned(Align))) tpAux[16]  = { tV, tV, tV, tV, tV, tV, tV, tV, tV, tV, tV, tV, tV, tV, tV, tV };
		const float __attribute__((aligned(Align))) zQAux[16]  = { zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ };
		const float __attribute__((aligned(Align))) izAux[16]  = { iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ };
		const float __attribute__((aligned(Align))) d2Aux[16]  = { ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2 };
		const float __attribute__((aligned(Align))) dzcAux[16] = { dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc };
		const float __attribute__((aligned(Align))) dzdAux[16] = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd };
#elif	defined(__AVX__)
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) tpAux[8]  = { tV, tV, tV, tV, tV, tV, tV, tV };
		const float __attribute__((aligned(Align))) zQAux[8]  = { zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ };
		const float __attribute__((aligned(Align))) izAux[8]  = { iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ };
		const float __attribute__((aligned(Align))) d2Aux[8]  = { ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2 };
		const float __attribute__((aligned(Align))) dzcAux[8] = { dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc };
		const float __attribute__((aligned(Align))) dzdAux[8] = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd };
#else
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) tpAux[4]  = { tV, tV, tV, tV };
		const float __attribute__((aligned(Align))) zQAux[4]  = { zQ, zQ, zQ, zQ };
		const float __attribute__((aligned(Align))) izAux[4]  = { iZ, iZ, iZ, iZ };
		const float __attribute__((aligned(Align))) d2Aux[4]  = { ood2, ood2, ood2, ood2 };
		const float __attribute__((aligned(Align))) dzcAux[4] = { dzc, dzc, dzc, dzc };
		const float __attribute__((aligned(Align))) dzdAux[4] = { dzd, dzd, dzd, dzd };
#endif


		const _MData_ tpVec  = opCode(load_ps, tpAux);
		const _MData_ zQVec  = opCode(load_ps, zQAux);
		const _MData_ izVec  = opCode(load_ps, izAux);
		const _MData_ d2Vec  = opCode(load_ps, d2Aux);
		const _MData_ dzcVec = opCode(load_ps, dzcAux);
		const _MData_ dzdVec = opCode(load_ps, dzdAux);
#ifdef	__MIC__
		const _MData_ z0Vec  = opCode(load_ps, z0Aux);
#endif

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, vel, tpP, mPy, tpM, mMy, v2p, acu, tP2, tM2;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz;

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
#ifdef	__MIC__
					tpM = opCode(swizzle_ps, opCode(load_ps, &m[idxMy]), _MM_SWIZ_REG_CBAD);
					tpP = opCode(permute4f128_ps, tpM, _MM_PERM_CBAD);
					mMy = opCode(mask_blend_ps, opCode(int2mask, 0b0001000100010001), tpM, tpP);
#elif	defined(__AVX2__)	//AVX2
					mMy = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 7,0,1,2,3,4,5,6));
#elif	defined(__AVX__)	//AVX
					tpM = opCode(permute_ps, opCode(load_ps, &m[idxMy]), 0b10010011);
					tpP = opCode(permute2f128_ps, tpM, tpM, 0b00000001);
					mMy = opCode(blend_ps, tpM, tpP, 0b00010001);
#else
					tpM = opCode(load_ps, &m[idxMy]);
					mMy = opCode(shuffle_ps, tpM, tpM, 0b10010011);
#endif
				}
				else
				{
					idxMy = idx - XC;
					mMy = opCode(load_ps, &m[idxMy]);

					if (X[1] == YC-1)
					{
						idxPy = idx - Sf + XC;
#ifdef	__MIC__
						tpM = opCode(swizzle_ps, opCode(load_ps, &m[idxPy]), _MM_SWIZ_REG_ADCB);
						tpP = opCode(permute4f128_ps, tpM, _MM_PERM_ADCB);
						mPy = opCode(mask_blend_ps, opCode(int2mask, 0b1110111011101110), tpM, tpP);
#elif	defined(__AVX2__)	//AVX2
						mPy = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 1,2,3,4,5,6,7,0));
#elif	defined(__AVX__)	//AVX
						tpM = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b00111001);
						tpP = opCode(permute2f128_ps, tpM, tpM, 0b00000001);
						mPy = opCode(blend_ps, tpM, tpP, 0b10001000);
#else
						tpP = opCode(load_ps, &m[idxPy]);
						mPy = opCode(shuffle_ps, tpP, tpP, 0b00111001);
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

				mel = opCode(load_ps, &m[idx]);

				/*	idxPx	*/

				vel = opCode(sub_ps, opCode(load_ps, &m[idxPx]), mel);
				tpM = opCode(sub_ps, vel, tpVec);
				tpP = opCode(add_ps, vel, tpVec);
				v2p = opCode(mul_ps, vel, vel);
				tP2 = opCode(mul_ps, tpP, tpP);
				tM2 = opCode(mul_ps, tpM, tpM);
#ifdef	__MIC__
				tmp = opCode(gmin_ps, opCode(gmin_ps, tP2, tM2), v2p);
				acu = opCode(mask_add_ps, z0Vec, opCode(cmp_ps, tmp, tP2, _CMP_EQ_OQ), z0Vec, tpP);
				acu = opCode(mask_add_ps, acu,   opCode(cmp_ps, tmp, tM2, _CMP_EQ_OQ), acu,   tpM);
				acu = opCode(mask_add_ps, acu,   opCode(cmp_ps, tmp, v2p, _CMP_EQ_OQ), acu,   vel);
#elif	defined(__AVX__)
				acu = opCode(setzero_ps);
				tmp = opCode(min_ps, opCode(min_ps, tP2, tM2), v2p);
				acu = opCode(add_ps,
					opCode(add_ps,
						opCode(and_ps, opCode(cmp_ps, tmp, tP2, _CMP_EQ_OQ), tpP),
						opCode(and_ps, opCode(cmp_ps, tmp, tM2, _CMP_EQ_OQ), tpM)),
					opCode(and_ps, opCode(cmp_ps, tmp, v2p, _CMP_EQ_OQ), vel));
#else
				acu = opCode(setzero_ps);
				tmp = opCode(min_ps, opCode(min_ps, tP2, tM2), v2p);
				acu = opCode(add_ps,
					opCode(add_ps,
						opCode(and_ps, opCode(cmpeq_ps, tmp, tP2), tpP),
						opCode(and_ps, opCode(cmpeq_ps, tmp, tM2), tpM)),
					opCode(and_ps, opCode(cmpeq_ps, tmp, v2p), vel));
#endif
				/*	idxMx	*/

				vel = opCode(sub_ps, opCode(load_ps, &m[idxMx]), mel);
				tpM = opCode(sub_ps, vel, tpVec);
				tpP = opCode(add_ps, vel, tpVec);
				v2p = opCode(mul_ps, vel, vel);
				tP2 = opCode(mul_ps, tpP, tpP);
				tM2 = opCode(mul_ps, tpM, tpM);
#ifdef	__MIC__
				tmp = opCode(gmin_ps, opCode(gmin_ps, tP2, tM2), v2p);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, tmp, tP2, _CMP_EQ_OQ), acu, tpP);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, tmp, tM2, _CMP_EQ_OQ), acu, tpM);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, tmp, v2p, _CMP_EQ_OQ), acu, vel);
#elif	defined(__AVX__)
				tmp = opCode(min_ps, opCode(min_ps, tP2, tM2), v2p);
				acu = opCode(add_ps, acu, opCode(add_ps,
					opCode(add_ps,
						opCode(and_ps, opCode(cmp_ps, tmp, tP2, _CMP_EQ_OQ), tpP),
						opCode(and_ps, opCode(cmp_ps, tmp, tM2, _CMP_EQ_OQ), tpM)),
					opCode(and_ps, opCode(cmp_ps, tmp, v2p, _CMP_EQ_OQ), vel)));
#else
				tmp = opCode(min_ps, opCode(min_ps, tP2, tM2), v2p);
				acu = opCode(add_ps, acu, opCode(add_ps,
					 opCode(add_ps,
						opCode(and_ps, opCode(cmpeq_ps, tmp, tP2), tpP),
						opCode(and_ps, opCode(cmpeq_ps, tmp, tM2), tpM)),
					opCode(and_ps, opCode(cmpeq_ps, tmp, v2p), vel)));
#endif
				/*	idxPz	*/

				vel = opCode(sub_ps, opCode(load_ps, &m[idxPz]), mel);
				tpM = opCode(sub_ps, vel, tpVec);
				tpP = opCode(add_ps, vel, tpVec);
				v2p = opCode(mul_ps, vel, vel);
				tP2 = opCode(mul_ps, tpP, tpP);
				tM2 = opCode(mul_ps, tpM, tpM);
#ifdef	__MIC__
				tmp = opCode(gmin_ps, opCode(gmin_ps, tP2, tM2), v2p);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, tmp, tP2, _CMP_EQ_OQ), acu, tpP);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, tmp, tM2, _CMP_EQ_OQ), acu, tpM);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, tmp, v2p, _CMP_EQ_OQ), acu, vel);
#elif	defined(__AVX__)
				tmp = opCode(min_ps, opCode(min_ps, tP2, tM2), v2p);
				acu = opCode(add_ps, acu, opCode(add_ps,
					opCode(add_ps,
						opCode(and_ps, opCode(cmp_ps, tmp, tP2, _CMP_EQ_OQ), tpP),
						opCode(and_ps, opCode(cmp_ps, tmp, tM2, _CMP_EQ_OQ), tpM)),
					opCode(and_ps, opCode(cmp_ps, tmp, v2p, _CMP_EQ_OQ), vel)));
#else
				tmp = opCode(min_ps, opCode(min_ps, tP2, tM2), v2p);
				acu = opCode(add_ps, acu, opCode(add_ps,
					 opCode(add_ps,
						opCode(and_ps, opCode(cmpeq_ps, tmp, tP2), tpP),
						opCode(and_ps, opCode(cmpeq_ps, tmp, tM2), tpM)),
					opCode(and_ps, opCode(cmpeq_ps, tmp, v2p), vel)));
#endif
				/*	idxMz	*/

				vel = opCode(sub_ps, opCode(load_ps, &m[idxMz]), mel);
				tpM = opCode(sub_ps, vel, tpVec);
				tpP = opCode(add_ps, vel, tpVec);
				v2p = opCode(mul_ps, vel, vel);
				tP2 = opCode(mul_ps, tpP, tpP);
				tM2 = opCode(mul_ps, tpM, tpM);
#ifdef	__MIC__
				tmp = opCode(gmin_ps, opCode(gmin_ps, tP2, tM2), v2p);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, tmp, tP2, _CMP_EQ_OQ), acu, tpP);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, tmp, tM2, _CMP_EQ_OQ), acu, tpM);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, tmp, v2p, _CMP_EQ_OQ), acu, vel);
#elif	defined(__AVX__)
				tmp = opCode(min_ps, opCode(min_ps, tP2, tM2), v2p);
				acu = opCode(add_ps, acu, opCode(add_ps,
					opCode(add_ps,
						opCode(and_ps, opCode(cmp_ps, tmp, tP2, _CMP_EQ_OQ), tpP),
						opCode(and_ps, opCode(cmp_ps, tmp, tM2, _CMP_EQ_OQ), tpM)),
					opCode(and_ps, opCode(cmp_ps, tmp, v2p, _CMP_EQ_OQ), vel)));
#else
				tmp = opCode(min_ps, opCode(min_ps, tP2, tM2), v2p);
				acu = opCode(add_ps, acu, opCode(add_ps,
					 opCode(add_ps,
						opCode(and_ps, opCode(cmpeq_ps, tmp, tP2), tpP),
						opCode(and_ps, opCode(cmpeq_ps, tmp, tM2), tpM)),
					opCode(and_ps, opCode(cmpeq_ps, tmp, v2p), vel)));
#endif
				/*	idxPy	*/

				vel = opCode(sub_ps, mPy, mel);
				tpM = opCode(sub_ps, vel, tpVec);
				tpP = opCode(add_ps, vel, tpVec);
				v2p = opCode(mul_ps, vel, vel);
				tP2 = opCode(mul_ps, tpP, tpP);
				tM2 = opCode(mul_ps, tpM, tpM);
#ifdef	__MIC__
				tmp = opCode(gmin_ps, opCode(gmin_ps, tP2, tM2), v2p);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, tmp, tP2, _CMP_EQ_OQ), acu, tpP);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, tmp, tM2, _CMP_EQ_OQ), acu, tpM);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, tmp, v2p, _CMP_EQ_OQ), acu, vel);
#elif	defined(__AVX__)
				tmp = opCode(min_ps, opCode(min_ps, tP2, tM2), v2p);
				acu = opCode(add_ps, acu, opCode(add_ps,
					opCode(add_ps,
						opCode(and_ps, opCode(cmp_ps, tmp, tP2, _CMP_EQ_OQ), tpP),
						opCode(and_ps, opCode(cmp_ps, tmp, tM2, _CMP_EQ_OQ), tpM)),
					opCode(and_ps, opCode(cmp_ps, tmp, v2p, _CMP_EQ_OQ), vel)));
#else
				tmp = opCode(min_ps, opCode(min_ps, tP2, tM2), v2p);
				acu = opCode(add_ps, acu, opCode(add_ps,
					 opCode(add_ps,
						opCode(and_ps, opCode(cmpeq_ps, tmp, tP2), tpP),
						opCode(and_ps, opCode(cmpeq_ps, tmp, tM2), tpM)),
					opCode(and_ps, opCode(cmpeq_ps, tmp, v2p), vel)));
#endif
				/*	idxMy	*/

				vel = opCode(sub_ps, mMy, mel);
				tpM = opCode(sub_ps, vel, tpVec);
				tpP = opCode(add_ps, vel, tpVec);
				v2p = opCode(mul_ps, vel, vel);
				tP2 = opCode(mul_ps, tpP, tpP);
				tM2 = opCode(mul_ps, tpM, tpM);
#ifdef	__MIC__
				tmp = opCode(gmin_ps, opCode(gmin_ps, tP2, tM2), v2p);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, tmp, tP2, _CMP_EQ_OQ), acu, tpP);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, tmp, tM2, _CMP_EQ_OQ), acu, tpM);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, tmp, v2p, _CMP_EQ_OQ), acu, vel);
#elif	defined(__AVX__)
				tmp = opCode(min_ps, opCode(min_ps, tP2, tM2), v2p);
				acu = opCode(add_ps, acu, opCode(add_ps,
					opCode(add_ps,
						opCode(and_ps, opCode(cmp_ps, tmp, tP2, _CMP_EQ_OQ), tpP),
						opCode(and_ps, opCode(cmp_ps, tmp, tM2, _CMP_EQ_OQ), tpM)),
					opCode(and_ps, opCode(cmp_ps, tmp, v2p, _CMP_EQ_OQ), vel)));
#else
				tmp = opCode(min_ps, opCode(min_ps, tP2, tM2), v2p);
				acu = opCode(add_ps, acu, opCode(add_ps,
					 opCode(add_ps,
						opCode(and_ps, opCode(cmpeq_ps, tmp, tP2), tpP),
						opCode(and_ps, opCode(cmpeq_ps, tmp, tM2), tpM)),
					opCode(and_ps, opCode(cmpeq_ps, tmp, v2p), vel)));
#endif

//#ifdef	__MIC__		NO SE SI VA A FUNCIONAR PORQUE CREO QUE EL SENO NO ESTA DEFINIDO EN KNC

//#else
				tpM = opCode(sub_ps,
					opCode(mul_ps, acu, d2Vec),
					opCode(mul_ps, zQVec, opCode(sin_ps, opCode(mul_ps, mel, izVec))));
//#endif
				mPy = opCode(load_ps, &v[idxMz]);

#if	defined(__MIC__) || defined(__FMA__)
				tmp = opCode(fmadd_ps, tpM, dzcVec, mPy);
				tpP = opCode(fmadd_ps, tmp, dzdVec, mel);
#else
				tmp = opCode(add_ps, mPy, opCode(mul_ps, tpM, dzcVec));
				tpP = opCode(add_ps, mel, opCode(mul_ps, tmp, dzdVec));
#endif

				/* Make sure the result is between -pi and pi	*/

				mMy = opCode(sub_ps, tpP, tpVec);
				mPy = opCode(add_ps, tpP, tpVec);
				v2p = opCode(mul_ps, tpP, tpP);
				tP2 = opCode(mul_ps, mPy, mPy);
				tM2 = opCode(mul_ps, mMy, mMy);
				acu = opCode(setzero_ps);
#ifdef	__MIC__
				vel = opCode(gmin_ps, opCode(gmin_ps, tP2, tM2), v2p);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, vel, tP2, _CMP_EQ_OQ), acu, mPy);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, vel, tM2, _CMP_EQ_OQ), acu, mMy);
				acu = opCode(mask_add_ps, acu, opCode(cmp_ps, vel, v2p, _CMP_EQ_OQ), acu, tpP);
#elif	defined(__AVX__)
				vel = opCode(min_ps, opCode(min_ps, tP2, tM2), v2p);
				acu = opCode(add_ps, acu, opCode(add_ps,
					opCode(add_ps,
						opCode(and_ps, opCode(cmp_ps, vel, tP2, _CMP_EQ_OQ), mPy),
						opCode(and_ps, opCode(cmp_ps, vel, tM2, _CMP_EQ_OQ), mMy)),
					opCode(and_ps, opCode(cmp_ps, vel, v2p, _CMP_EQ_OQ), tpP)));
#else
				vel = opCode(min_ps, opCode(min_ps, tP2, tM2), v2p);
				acu = opCode(add_ps, acu, opCode(add_ps,
					 opCode(add_ps,
						opCode(and_ps, opCode(cmpeq_ps, vel, tP2), mPy),
						opCode(and_ps, opCode(cmpeq_ps, vel, tM2), mMy)),
					opCode(and_ps, opCode(cmpeq_ps, vel, v2p), tpP)));
#endif

				opCode(store_ps, &v[idxMz], tmp);
				opCode(store_ps, &m2[idx],  acu);
			}
		}
#undef	_MData_
#undef	step
	}
}

void	propThetaXeon	(Scalar *axionField, const double dz, const double delta2, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision)
{
#ifdef USE_XEON
	const int  micIdx = commAcc();
	const size_t ext = V + S;
	const double ood2 = 1./delta2;
	double *z = axionField->zV();

	int bulk  = 32;

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		propThetaKernelXeon(mX, vX, m2X, z, dz, C1, D1, ood2, nQcd, Lx, 2*S, V, precision);
	}
	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		propThetaKernelXeon(mX, vX, m2X, z, dz, C1, D1, ood2, nQcd, Lx, S, 2*S, precision);
		propThetaKernelXeon(mX, vX, m2X, z, dz, C1, D1, ood2, nQcd, Lx, V, ext, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	*z += dz*D1;

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		propThetaKernelXeon(m2X, vX, mX, z, dz, C2, D2, ood2, nQcd, Lx, 2*S, V, precision);
	}
	axionField->exchangeGhosts(FIELD_M2);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		propThetaKernelXeon(m2X, vX, mX, z, dz, C2, D2, ood2, nQcd, Lx, S, 2*S, precision);
		propThetaKernelXeon(m2X, vX, mX, z, dz, C2, D2, ood2, nQcd, Lx, V, ext, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	*z += dz*D2;

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		propThetaKernelXeon(mX, vX, m2X, z, dz, C3, D3, ood2, nQcd, Lx, 2*S, V, precision);
	}
	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		propThetaKernelXeon(mX, vX, m2X, z, dz, C3, D3, ood2, nQcd, Lx, S, 2*S, precision);
		propThetaKernelXeon(mX, vX, m2X, z, dz, C3, D3, ood2, nQcd, Lx, V, ext, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	*z += dz*D3;

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		propThetaKernelXeon(m2X, vX, mX, z, dz, C4, D4, ood2, nQcd, Lx, 2*S, V, precision);
	}
	axionField->exchangeGhosts(FIELD_M2);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		propThetaKernelXeon(m2X, vX, mX, z, dz, C4, D4, ood2, nQcd, Lx, S, 2*S, precision);
		propThetaKernelXeon(m2X, vX, mX, z, dz, C4, D4, ood2, nQcd, Lx, V, ext, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	*z += dz*D4;
#endif
}

void	propThetaCpu	(Scalar *axionField, const double dz, const double delta2, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision)
{
	const double ood2 = 1./delta2;
	double *z = axionField->zV();

	axionField->sendGhosts(FIELD_M, COMM_SDRV);
	propThetaKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C1, D1, ood2, nQcd, Lx, 2*S, V, precision);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
	propThetaKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C1, D1, ood2, nQcd, Lx, S, 2*S, precision);
	propThetaKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C1, D1, ood2, nQcd, Lx, V, V+S, precision);
	*z += dz*D1;

	axionField->sendGhosts(FIELD_M2, COMM_SDRV);
	propThetaKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C2, D2, ood2, nQcd, Lx, 2*S, V, precision);
	axionField->sendGhosts(FIELD_M2, COMM_WAIT);
	propThetaKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C2, D2, ood2, nQcd, Lx, S, 2*S, precision);
	propThetaKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C2, D2, ood2, nQcd, Lx, V, V+S, precision);
	*z += dz*D2;

	axionField->sendGhosts(FIELD_M, COMM_SDRV);
	propThetaKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C3, D3, ood2, nQcd, Lx, 2*S, V, precision);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
	propThetaKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C3, D3, ood2, nQcd, Lx, S, 2*S, precision);
	propThetaKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C3, D3, ood2, nQcd, Lx, V, V+S, precision);
	*z += dz*D3;

	axionField->sendGhosts(FIELD_M2, COMM_SDRV);
	propThetaKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C4, D4, ood2, nQcd, Lx, 2*S, V, precision);
	axionField->sendGhosts(FIELD_M2, COMM_WAIT);
	propThetaKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C4, D4, ood2, nQcd, Lx, S, 2*S, precision);
	propThetaKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C4, D4, ood2, nQcd, Lx, V, V+S, precision);
	*z += dz*D4;
}
