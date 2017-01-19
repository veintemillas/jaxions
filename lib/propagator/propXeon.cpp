#include<cstdio>
#include<cmath>
#include"scalar/scalarField.h"
#include"enum-field.h"
#include"propagator/RKParms.h"
#include"scalar/varNQCD.h"

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


#ifndef	__MIC__
#ifdef	__AVX__
void printFloat(size_t idx, size_t con, __m256 dat)
{
	if (idx == con) {
		float caca[8];
		opCode(storeu_ps, caca, dat);
		printf ("%e %e | %e %e | %e %e | %e %e\n", caca[0], caca[1], caca[2], caca[3], caca[4], caca[5], caca[6], caca[7]);
	}
}

void printDouble(size_t idx, size_t con, __m256d dat)
{
	if (idx == con) {
		double caca[4];
		opCode(storeu_pd, caca, dat);
		printf ("%le %le | %le %le\n", caca[0], caca[1], caca[2], caca[3]);
	}
}

#else

void printFloat(size_t idx, size_t con, __m128 dat)
{
	if (idx == con) {
		float caca[4];
		opCode(storeu_ps, caca, dat);
		printf ("%e %e | %e %e\n", caca[0], caca[1], caca[2], caca[3]);
	}
}

void printDouble(size_t idx, size_t con, __m128d dat)
{
	if (idx == con) {
		double caca[2];
		opCode(storeu_pd, caca, dat);
		printf ("%le %le\n", caca[0], caca[1]);
	}
}
#endif
#else
__attribute__((target(mic)))
void printFloat(size_t idx, size_t con, __m512 dat)
{
	if (idx == con) {
		static float __attribute((aligned(64))) caca[16];
		opCode(store_ps, caca, dat);
		printf ("%e %e | %e %e | %e %e | %e %e | %e %e | %e %e | %e %e | %e %e\n", caca[0], caca[1], caca[2], caca[3], caca[4], caca[5], caca[6], caca[7],
											   caca[8], caca[9], caca[10], caca[11], caca[12], caca[13], caca[14], caca[15]);
	}
}

__attribute__((target(mic)))
void printDouble(size_t idx, size_t con, __m512d dat)
{
	if (idx == con) {
		static double  __attribute((aligned(64))) caca[8];
		opCode(store_pd, caca, dat);
		printf ("%le %le | %le %le | %le %le | %le %le\n", caca[0], caca[1], caca[2], caca[3], caca[4], caca[5], caca[6], caca[7]);
	}
}
#endif
#ifdef USE_XEON
__attribute__((target(mic)))
#endif
void	propagateKernelXeon(const void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_, double *z, const double dz, const double c, const double d,
			    const double ood2, const double LL, const double nQcd, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision)
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
		const double z2 = zR*zR;
		//const double zQ = 9.*pow(zR, nQcd+3.);
		const double zQ = axionmass2(zR, nQcd, 1.5, 3.)*zR*zR*zR;

#ifdef	__MIC__
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const double __attribute__((aligned(Align))) z2Aux[8]  = {-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2 };
		const double __attribute__((aligned(Align))) zQAux[8]  = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) c6Aux[8]  = {-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6. };
		const double __attribute__((aligned(Align))) lbAux[8]  = { LL, LL, LL, LL, LL, LL, LL, LL };
		const double __attribute__((aligned(Align))) d2Aux[8]  = { ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2 };
		const double __attribute__((aligned(Align))) dzcAux[8] = { dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc };
		const double __attribute__((aligned(Align))) dzdAux[8] = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd };
#elif	defined(__AVX__)
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) z2Aux[4]  = {-z2,-z2,-z2,-z2 };
		const double __attribute__((aligned(Align))) zQAux[4]  = { zQ, 0., zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) c6Aux[4]  = {-6.,-6.,-6.,-6. };
		const double __attribute__((aligned(Align))) lbAux[4]  = { LL, LL, LL, LL };
		const double __attribute__((aligned(Align))) d2Aux[4]  = { ood2, ood2, ood2, ood2 };
		const double __attribute__((aligned(Align))) dzcAux[4] = { dzc, dzc, dzc, dzc };
		const double __attribute__((aligned(Align))) dzdAux[4] = { dzd, dzd, dzd, dzd };
#else
		const size_t XC = Lx;
		const size_t YC = Lx;

		const double __attribute__((aligned(Align))) z2Aux[2]  = {-z2,-z2 };
		const double __attribute__((aligned(Align))) zQAux[2]  = { zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) c6Aux[2]  = {-6.,-6. };
		const double __attribute__((aligned(Align))) lbAux[2]  = { LL, LL };
		const double __attribute__((aligned(Align))) d2Aux[2]  = { ood2, ood2 };
		const double __attribute__((aligned(Align))) dzcAux[2] = { dzc, dzc };
		const double __attribute__((aligned(Align))) dzdAux[2] = { dzd, dzd };

#endif
		const _MData_ z2Vec  = opCode(load_pd, z2Aux);
		const _MData_ zQVec  = opCode(load_pd, zQAux);
		const _MData_ c6Vec  = opCode(load_pd, c6Aux);
		const _MData_ lbVec  = opCode(load_pd, lbAux);
		const _MData_ d2Vec  = opCode(load_pd, d2Aux);
		const _MData_ dzcVec = opCode(load_pd, dzcAux);
		const _MData_ dzdVec = opCode(load_pd, dzdAux);

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, mPx, mPy;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxPx, idxMx, idxPy, idxMy, idxPz, idxMz, idxP0;

				{
					size_t tmi = idx/XC, tpi;

					tpi = tmi/YC;
					X[1] = tmi - tpi*YC;
					X[0] = idx - tmi*XC;
				}

				if (X[0] == XC-step)
					idxPx = ((idx - XC + step) << 1);
				else
					idxPx = ((idx + step) << 1);

				if (X[0] == 0)
					idxMx = ((idx + XC - step) << 1);
				else
					idxMx = ((idx - step) << 1);

				if (X[1] == 0)
				{
					idxMy = ((idx + Sf - XC) << 1);
					idxPy = ((idx + XC) << 1);
//					mPx = opCode(load_pd, &m[idxMy]);
#ifdef	__MIC__
					tmp = opCode(add_pd, opCode(castps_pd, opCode(permute4f128_ps, opCode(castpd_ps, opCode(load_pd, &m[idxMy])), _MM_PERM_CBAD)), opCode(load_pd, &m[idxPy]));
#elif	defined(__AVX__)
					mPx = opCode(load_pd, &m[idxMy]);
					tmp = opCode(add_pd, opCode(permute2f128_pd, mPx, mPx, 0b00000001), opCode(load_pd, &m[idxPy]));
#else
					tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
#endif
				}
				else
				{
					idxMy = ((idx - XC) << 1);

					if (X[1] == YC-1)
					{
						idxPy = ((idx - Sf + XC) << 1);
#ifdef	__MIC__
						tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(castps_pd, opCode(permute4f128_ps, opCode(castpd_ps, opCode(load_pd, &m[idxPy])), _MM_PERM_ADCB)));
#elif	defined(__AVX__)
						mPx = opCode(load_pd, &m[idxPy]);
						tmp = opCode(add_pd, opCode(permute2f128_pd, mPx, mPx, 0b00000001), opCode(load_pd, &m[idxMy]));
#else
						tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
#endif
					}
					else
					{
						idxPy = ((idx + XC) << 1);
						tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
					}
				}

				idxPz = ((idx+Sf) << 1);
				idxMz = ((idx-Sf) << 1);
				idxP0 = (idx << 1);

				mel = opCode(load_pd, &m[idxP0]);
				mPy = opCode(mul_pd, mel, mel);

#ifdef	__MIC__
				mPx = opCode(add_pd, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, mPy), _MM_PERM_BADC)), mPy);
#elif defined(__AVX__)
				mPx = opCode(add_pd, opCode(permute_pd, mPy, 0b00000101), mPy);
#else
				mPx = opCode(add_pd, opCode(shuffle_pd, mPy, mPy, 0b00000001), mPy);
#endif

				mPx = opCode(sub_pd,
					opCode(add_pd,
						opCode(mul_pd,
							opCode(add_pd,
								opCode(add_pd,
									opCode(load_pd, &m[idxMz]),
									opCode(add_pd,
										opCode(add_pd,
											opCode(add_pd, tmp, opCode(load_pd, &m[idxPx])),
											opCode(load_pd, &m[idxMx])),
										opCode(load_pd, &m[idxPz]))),
								opCode(mul_pd, mel, c6Vec)),
							d2Vec),
						zQVec),
					opCode(mul_pd,
						opCode(mul_pd,
							opCode(add_pd, mPx, z2Vec),
							lbVec),
						mel));

				mPy = opCode(load_pd, &v[idxMz]);
#if	defined(__MIC__) || defined(__FMA__)
				tmp = opCode(fmadd_pd, mPx, dzcVec, mPy);
				mPx = opCode(fmadd_pd, tmp, dzdVec, mel);
#else
				tmp = opCode(add_pd, mPy, opCode(mul_pd, mPx, dzcVec));
				mPx = opCode(add_pd, mel, opCode(mul_pd, tmp, dzdVec));
#endif
				opCode(store_pd,  &v[idxMz], tmp);
				opCode(store_pd, &m2[idxP0], mPx);
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
		const float z2 = zR*zR;
		//const float zQ = 9.*powf(zR, nQcd+3.);
		const float zQ = axionmass2((double) zR, nQcd, 1.5 , 3.)*zR*zR*zR;

#ifdef	__MIC__
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) z2Aux[16]  = {-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2};
		const float __attribute__((aligned(Align))) zQAux[16]  = { zQ, 0., zQ, 0., zQ, 0., zQ, 0., zQ, 0., zQ, 0., zQ, 0., zQ, 0.};
		const float __attribute__((aligned(Align))) c6Aux[16]  = {-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.};
		const float __attribute__((aligned(Align))) lbAux[16]  = { LL, LL, LL, LL, LL, LL, LL, LL, LL, LL, LL, LL, LL, LL, LL, LL};
		const float __attribute__((aligned(Align))) d2Aux[16]  = { ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2 };
		const float __attribute__((aligned(Align))) dzcAux[16] = { dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc };
		const float __attribute__((aligned(Align))) dzdAux[16] = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd };
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) z2Aux[8]  = {-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2 };
		const float __attribute__((aligned(Align))) zQAux[8]  = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };
		const float __attribute__((aligned(Align))) c6Aux[8]  = {-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6. };
		const float __attribute__((aligned(Align))) lbAux[8]  = { LL, LL, LL, LL, LL, LL, LL, LL };
		const float __attribute__((aligned(Align))) d2Aux[8]  = { ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2 };
		const float __attribute__((aligned(Align))) dzcAux[8] = { dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc };
		const float __attribute__((aligned(Align))) dzdAux[8] = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd };
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const float __attribute__((aligned(Align))) z2Aux[4]  = {-z2,-z2,-z2,-z2 };
		const float __attribute__((aligned(Align))) zQAux[4]  = { zQ, 0., zQ, 0. };
		const float __attribute__((aligned(Align))) c6Aux[4]  = {-6.,-6.,-6.,-6. };
		const float __attribute__((aligned(Align))) lbAux[4]  = { LL, LL, LL, LL };
		const float __attribute__((aligned(Align))) d2Aux[4]  = { ood2, ood2, ood2, ood2 };
		const float __attribute__((aligned(Align))) dzcAux[4] = { dzc, dzc, dzc, dzc };
		const float __attribute__((aligned(Align))) dzdAux[4] = { dzd, dzd, dzd, dzd };
#endif

		const _MData_ z2Vec  = opCode(load_ps, z2Aux);
		const _MData_ zQVec  = opCode(load_ps, zQAux);
		const _MData_ c6Vec  = opCode(load_ps, c6Aux);
		const _MData_ lbVec  = opCode(load_ps, lbAux);
		const _MData_ d2Vec  = opCode(load_ps, d2Aux);
		const _MData_ dzcVec = opCode(load_ps, dzcAux);
		const _MData_ dzdVec = opCode(load_ps, dzdAux);

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, mPx, mPy, mMx;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0;

				{
					size_t tmi = idx/XC, itp;

					itp = tmi/YC;
					X[1] = tmi - itp*YC;
					X[0] = idx - tmi*XC;
				}

				if (X[0] == XC-step)
					idxPx = ((idx - XC + step) << 1);
				else
					idxPx = ((idx + step) << 1);

				if (X[0] == 0)
					idxMx = ((idx + XC - step) << 1);
				else
					idxMx = ((idx - step) << 1);

				if (X[1] == 0)
				{
					idxMy = ((idx + Sf - XC) << 1);
					idxPy = ((idx + XC) << 1);

#ifdef	__MIC__
					mMx = opCode(swizzle_ps, opCode(load_ps, &m[idxMy]), _MM_SWIZ_REG_BADC);
					mPx = opCode(permute4f128_ps, mMx, _MM_PERM_CBAD);
					tmp = opCode(add_ps, opCode(mask_blend_ps, opCode(int2mask, 0b0011001100110011), mMx, mPx), opCode(load_ps, &m[idxPy]));
#elif	defined(__AVX2__)	//AVX2
					tmp = opCode(add_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 6,7,0,1,2,3,4,5)),  opCode(load_ps, &m[idxPy]));
#elif	defined(__AVX__)	//AVX
					mMx = opCode(permute_ps, opCode(load_ps, &m[idxMy]), 0b01001110);
					mPx = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
					tmp = opCode(add_ps, opCode(blend_ps, mMx, mPx, 0b00110011), opCode(load_ps, &m[idxPy]));
#else
					mMx = opCode(load_ps, &m[idxMy]);
					tmp = opCode(add_ps, opCode(shuffle_ps, mMx, mMx, 0b01001110), opCode(load_ps, &m[idxPy]));
#endif
				}
				else
				{
					idxMy = ((idx - XC) << 1);

					if (X[1] == YC-1)
					{
						idxPy = ((idx - Sf + XC) << 1);
#ifdef	__MIC__
						mMx = opCode(swizzle_ps, opCode(load_ps, &m[idxPy]), _MM_SWIZ_REG_BADC);
						mPx = opCode(permute4f128_ps, mMx, _MM_PERM_ADCB);
						tmp = opCode(add_ps, opCode(mask_blend_ps, opCode(int2mask, 0b1100110011001100), mMx, mPx), opCode(load_ps, &m[idxMy]));
#elif	defined(__AVX2__)	//AVX2
						tmp = opCode(add_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 2,3,4,5,6,7,0,1)), opCode(load_ps, &m[idxMy]));
#elif	defined(__AVX__)	//AVX
						mMx = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b01001110);
						mPx = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
						tmp = opCode(add_ps, opCode(blend_ps, mMx, mPx, 0b11001100), opCode(load_ps, &m[idxMy]));
#else
						mPx = opCode(load_ps, &m[idxPy]);
						tmp = opCode(add_ps, opCode(shuffle_ps, mPx, mPx, 0b01001110), opCode(load_ps, &m[idxMy]));
#endif
					}
					else
					{
						idxPy = ((idx + XC) << 1);
						tmp = opCode(add_ps, opCode(load_ps, &m[idxPy]), opCode(load_ps, &m[idxMy]));
					}
				}

				idxPz = ((idx+Sf) << 1);
				idxMz = ((idx-Sf) << 1);
				idxP0 = (idx << 1);

				mel = opCode(load_ps, &m[idxP0]);
				mPy = opCode(mul_ps, mel, mel);

#ifdef	__MIC__
				mPx = opCode(add_ps, opCode(swizzle_ps, mPy, _MM_SWIZ_REG_CDAB), mPy);
#elif	defined(__AVX__)
				mPx = opCode(add_ps, opCode(permute_ps, mPy, 0b10110001), mPy);
#else
				mPx = opCode(add_ps, opCode(shuffle_ps, mPy, mPy, 0b10110001), mPy);
#endif
				mMx = opCode(sub_ps,
					opCode(add_ps,
						opCode(mul_ps,
							opCode(add_ps,
								opCode(add_ps,
									opCode(load_ps, &m[idxMz]),
									opCode(add_ps,
										opCode(add_ps,
											opCode(add_ps, tmp, opCode(load_ps, &m[idxPx])),
											opCode(load_ps, &m[idxMx])),
										opCode(load_ps, &m[idxPz]))),
								opCode(mul_ps, mel, c6Vec)),
							d2Vec),
						zQVec),
					opCode(mul_ps,
						opCode(mul_ps,
							opCode(add_ps, mPx, z2Vec),
							lbVec),
						mel));
				mPy = opCode(load_ps, &v[idxMz]);

#if	defined(__MIC__) || defined(__FMA__)
				tmp = opCode(fmadd_ps, mMx, dzcVec, mPy);
				mPx = opCode(fmadd_ps, tmp, dzdVec, mel);
#else
				tmp = opCode(add_ps, mPy, opCode(mul_ps, mMx, dzcVec));
				mPx = opCode(add_ps, mel, opCode(mul_ps, tmp, dzdVec));
#endif
				opCode(store_ps,  &v[idxMz], tmp);
				opCode(store_ps, &m2[idxP0], mPx);
			}
		}
#undef	_MData_
#undef	step
	}
}

void	propagateXeon	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision)
{
#ifdef USE_XEON
	const int  micIdx = commAcc();
	const size_t ext = V + S;
	const double ood2 = 1./delta2;
	double *z = axionField->zV();
	double lambda = LL;

	int bulk  = 32;

	if (axionField->Lambda() != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		propagateKernelXeon(mX, vX, m2X, z, dz, C1, D1, ood2, lambda, nQcd, Lx, 2*S, V, precision);
	}
	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		propagateKernelXeon(mX, vX, m2X, z, dz, C1, D1, ood2, lambda, nQcd, Lx, S, 2*S, precision);
		propagateKernelXeon(mX, vX, m2X, z, dz, C1, D1, ood2, lambda, nQcd, Lx, V, ext, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	*z += dz*D1;

	if (axionField->Lambda() != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		propagateKernelXeon(m2X, vX, mX, z, dz, C2, D2, ood2, lambda, nQcd, Lx, 2*S, V, precision);
	}
	axionField->exchangeGhosts(FIELD_M2);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		propagateKernelXeon(m2X, vX, mX, z, dz, C2, D2, ood2, lambda, nQcd, Lx, S, 2*S, precision);
		propagateKernelXeon(m2X, vX, mX, z, dz, C2, D2, ood2, lambda, nQcd, Lx, V, ext, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	*z += dz*D2;

	if (axionField->Lambda() != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		propagateKernelXeon(mX, vX, m2X, z, dz, C3, D3, ood2, lambda, nQcd, Lx, 2*S, V, precision);
	}
	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		propagateKernelXeon(mX, vX, m2X, z, dz, C3, D3, ood2, lambda, nQcd, Lx, S, 2*S, precision);
		propagateKernelXeon(mX, vX, m2X, z, dz, C3, D3, ood2, lambda, nQcd, Lx, V, ext, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	*z += dz*D3;

	if (axionField->Lambda() != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		propagateKernelXeon(m2X, vX, mX, z, dz, C4, D4, ood2, lambda, nQcd, Lx, 2*S, V, precision);
	}
	axionField->exchangeGhosts(FIELD_M2);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		propagateKernelXeon(m2X, vX, mX, z, dz, C4, D4, ood2, lambda, nQcd, Lx, S, 2*S, precision);
		propagateKernelXeon(m2X, vX, mX, z, dz, C4, D4, ood2, lambda, nQcd, Lx, V, ext, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	*z += dz*D4;
#endif
}

void	propagateCpu	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision)
{
	const double ood2 = 1./delta2;
	double *z = axionField->zV();
	double lambda = LL;

	axionField->sendGhosts(FIELD_M, COMM_SDRV);

	if (axionField->Lambda() != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C1, D1, ood2, lambda, nQcd, Lx, 2*S, V, precision);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
	propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C1, D1, ood2, lambda, nQcd, Lx, S, 2*S, precision);
	propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C1, D1, ood2, lambda, nQcd, Lx, V, V+S, precision);
	*z += dz*D1;

	axionField->sendGhosts(FIELD_M2, COMM_SDRV);

	if (axionField->Lambda() != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C2, D2, ood2, lambda, nQcd, Lx, 2*S, V, precision);
	axionField->sendGhosts(FIELD_M2, COMM_WAIT);
	propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C2, D2, ood2, lambda, nQcd, Lx, S, 2*S, precision);
	propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C2, D2, ood2, lambda, nQcd, Lx, V, V+S, precision);
	*z += dz*D2;

	axionField->sendGhosts(FIELD_M, COMM_SDRV);

	if (axionField->Lambda() != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C3, D3, ood2, lambda, nQcd, Lx, 2*S, V, precision);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
	propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C3, D3, ood2, lambda, nQcd, Lx, S, 2*S, precision);
	propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C3, D3, ood2, lambda, nQcd, Lx, V, V+S, precision);
	*z += dz*D3;

	axionField->sendGhosts(FIELD_M2, COMM_SDRV);

	if (axionField->Lambda() != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C4, D4, ood2, lambda, nQcd, Lx, 2*S, V, precision);
	axionField->sendGhosts(FIELD_M2, COMM_WAIT);
	propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C4, D4, ood2, lambda, nQcd, Lx, S, 2*S, precision);
	propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C4, D4, ood2, lambda, nQcd, Lx, V, V+S, precision);
	*z += dz*D4;
}

#ifdef USE_XEON
__attribute__((target(mic)))
#endif
void	updateMXeon(void * __restrict__ m_, const void * __restrict__ v_, const double dz, const double d, const size_t Vo, const size_t Vf, const size_t Sf, FieldPrecision precision)
{
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
		double * __restrict__ m		= (double * __restrict__) m_;
		const double * __restrict__ v	= (const double * __restrict__) v_;

		__assume_aligned(m, Align);
		__assume_aligned(v, Align);
#else
		double * __restrict__ m		= (double * __restrict__) __builtin_assume_aligned (m_, Align);
		const double * __restrict__ v	= (const double * __restrict__) __builtin_assume_aligned (v_, Align);
#endif
		const double dzd = dz*d;

#ifdef	__MIC__
		const double __attribute__((aligned(Align))) dzdAux[8] = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd };
#else
		const double __attribute__((aligned(Align))) dzdAux[4] = { dzd, dzd, dzd, dzd };
#endif

		const _MData_ dzdVec = opCode(load_pd, dzdAux);

		#pragma omp parallel default(shared)
		{
			register _MData_ mIn, vIn, tmp;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
#ifdef	__MIC__
				vIn = opCode(load_pd, &v[2*(idx-Sf)]);
				mIn = opCode(load_pd, &m[2*idx]);
				tmp = opCode(fmadd_pd, dzdVec, vIn, mIn);
				opCode(store_pd, &m[2*idx], tmp);
#else
				mIn = opCode(load_pd, &m[2*idx]);
				tmp = opCode(load_pd, &v[2*(idx-Sf)]);
				vIn = opCode(mul_pd, dzdVec, tmp);
				tmp = opCode(add_pd, mIn, vIn);
				opCode(store_pd, &m[2*idx], tmp);
#endif
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
		float * __restrict__ m		= (float * __restrict__) m_;
		const float * __restrict__ v	= (const float * __restrict__) v_;

		__assume_aligned(m, Align);
		__assume_aligned(v, Align);
#else
		float * __restrict__ m		= (float * __restrict__) __builtin_assume_aligned (m_, Align);
		const float * __restrict__ v	= (const float * __restrict__) __builtin_assume_aligned (v_, Align);
#endif
		const float dzd = dz*d;
#ifdef	__MIC__
		const float __attribute__((aligned(Align))) dzdAux[16] = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd  };
#elif	defined(__AVX__)
		const float __attribute__((aligned(Align))) dzdAux[8]  = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd };
#else
		const float __attribute__((aligned(Align))) dzdAux[4]  = { dzd, dzd, dzd, dzd };
#endif
		const _MData_ dzdVec = opCode(load_ps, dzdAux);

		#pragma omp parallel default(shared)
		{
			register _MData_ mIn, vIn, tmp;
			register size_t idxP0, idxMz;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				idxP0 = idx << 1;
				idxMz = (idx - Sf) << 1;
#ifdef	__MIC__
				vIn = opCode(load_ps, &v[idxMz]);
				mIn = opCode(load_ps, &m[idxP0]);
				tmp = opCode(fmadd_ps, dzdVec, vIn, mIn);
				opCode(store_ps, &m[idxP0], tmp);
#else
				vIn = opCode(load_ps, &v[idxMz]);
				mIn = opCode(load_ps, &m[idxP0]);
				tmp = opCode(add_ps, mIn, opCode(mul_ps, dzdVec, vIn));
				opCode(store_ps, &m[idxP0], tmp);
#endif
			}
		}
#undef	_MData_
#undef	step
	}
}

#ifdef USE_XEON
__attribute__((target(mic)))
#endif
void	updateVXeon(const void * __restrict__ m_, void * __restrict__ v_, double *z, const double dz, const double c, const double ood2,
		    const double LL, const double nQcd, const size_t Lx, const size_t Vo, const size_t Vf, const size_t Sf, FieldPrecision precision)
{
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
		const double * __restrict__ m = (const double * __restrict__) m_;
		double * __restrict__ v = (double * __restrict__) v_;

		__assume_aligned(m, Align);
		__assume_aligned(v, Align);
#else
		const double * __restrict__ m = (const double * __restrict__) __builtin_assume_aligned (m_, Align);
		double * __restrict__ v = (double * __restrict__) __builtin_assume_aligned (v_, Align);
#endif
		const double zR = *z;
		const double z2 = zR*zR;
		//const double zQ = 9.*pow(zR, nQcd+3.);
		const double zQ = axionmass2(zR, nQcd, 1.5, 3.)*zR*zR*zR;
		const double dzc = dz*c;
#ifdef	__MIC__
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const double __attribute__((aligned(Align))) z2Aux[8]  = {-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2 };
		const double __attribute__((aligned(Align))) zQAux[8]  = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) c6Aux[8]  = {-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6. };
		const double __attribute__((aligned(Align))) lbAux[8]  = { LL, LL, LL, LL, LL, LL, LL, LL };
		const double __attribute__((aligned(Align))) d2Aux[8]  = { ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2 };
		const double __attribute__((aligned(Align))) dzcAux[8] = { dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc };
#elif	defined(__AVX__)
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) z2Aux[4]  = {-z2,-z2,-z2,-z2 };
		const double __attribute__((aligned(Align))) zQAux[4]  = { zQ, 0., zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) c6Aux[4]  = {-6.,-6.,-6.,-6. };
		const double __attribute__((aligned(Align))) lbAux[4]  = { LL, LL, LL, LL };
		const double __attribute__((aligned(Align))) d2Aux[4]  = { ood2, ood2, ood2, ood2 };
		const double __attribute__((aligned(Align))) dzcAux[4] = { dzc, dzc, dzc, dzc };
#else
		const size_t XC = Lx;
		const size_t YC = Lx;

		const double __attribute__((aligned(Align))) z2Aux[2]  = {-z2,-z2 };
		const double __attribute__((aligned(Align))) zQAux[2]  = { zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) c6Aux[2]  = {-6.,-6. };
		const double __attribute__((aligned(Align))) lbAux[2]  = { LL, LL };
		const double __attribute__((aligned(Align))) d2Aux[2]  = { ood2, ood2 };
		const double __attribute__((aligned(Align))) dzcAux[2] = { dzc, dzc,  };
#endif
		const _MData_ z2Vec  = opCode(load_pd, z2Aux);
		const _MData_ zQVec  = opCode(load_pd, zQAux);
		const _MData_ c6Vec  = opCode(load_pd, c6Aux);
		const _MData_ lbVec  = opCode(load_pd, lbAux);
		const _MData_ d2Vec  = opCode(load_pd, d2Aux);
		const _MData_ dzcVec = opCode(load_pd, dzcAux);

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, mPx, mPy;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxPx, idxMx, idxPy, idxMy, idxPz, idxMz, idxP0;

				{
					size_t tmi = idx/XC, tpi;

					tpi = tmi/YC;
					X[1] = tmi - tpi*YC;
					X[0] = idx - tmi*XC;
				}

				if (X[0] == XC-step)
					idxPx = ((idx - XC + step) << 1);
				else
					idxPx = ((idx + step) << 1);

				if (X[0] == 0)
					idxMx = ((idx + XC - step) << 1);
				else
					idxMx = ((idx - step) << 1);

				if (X[1] == 0)
				{
					idxMy = ((idx + Sf - XC) << 1);
					idxPy = ((idx + XC) << 1);
					mPx = opCode(load_pd, &m[idxMy]);
#ifdef	__MIC__
					tmp = opCode(add_pd, opCode(castps_pd, opCode(permute4f128_ps, opCode(castpd_ps, opCode(load_pd, &m[idxMy])), _MM_PERM_CBAD)), opCode(load_pd, &m[idxPy]));
#elif	defined(__AVX__)
					mPx = opCode(load_pd, &m[idxMy]);
					tmp = opCode(add_pd, opCode(permute2f128_pd, mPx, mPx, 0b00000001), opCode(load_pd, &m[idxPy]));
#else
					tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
#endif
				}
				else
				{
					idxMy = ((idx - XC) << 1);

					if (X[1] == YC-1)
					{
						idxPy = ((idx - Sf + XC) << 1);
#ifdef	__MIC__
						tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(castps_pd, opCode(permute4f128_ps, opCode(castpd_ps, opCode(load_pd, &m[idxPy])), _MM_PERM_ADCB)));
#elif	defined(__AVX__)
						mPx = opCode(load_pd, &m[idxPy]);
						tmp = opCode(add_pd, opCode(permute2f128_pd, mPx, mPx, 0b00000001), opCode(load_pd, &m[idxMy]));
#else
						tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
#endif
					}
					else
					{
						idxPy = ((idx + XC) << 1);
						tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
					}
				}

				idxPz = ((idx+Sf) << 1);
				idxMz = ((idx-Sf) << 1);
				idxP0 = (idx << 1);

				mel = opCode(load_pd, &m[idxP0]);
				mPy = opCode(mul_pd, mel, mel);

#ifdef	__MIC__
				mPx = opCode(add_pd, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, mPy), _MM_PERM_BADC)), mPy);
#elif	defined(__AVX__)
				mPx = opCode(add_pd, opCode(permute_pd, mPy, 0b00000101), mPy);
#else
				mPx = opCode(add_pd, opCode(shuffle_pd, mPy, mPy, 0b00000001), mPy);
#endif

				mPx = opCode(sub_pd,
					opCode(add_pd,
						opCode(mul_pd,
							opCode(add_pd,
								opCode(add_pd,
									opCode(load_pd, &m[idxMz]),
									opCode(add_pd,
										opCode(add_pd,
											opCode(add_pd, tmp, opCode(load_pd, &m[idxPx])),
											opCode(load_pd, &m[idxMx])),
										opCode(load_pd, &m[idxPz]))),
								opCode(mul_pd, mel, c6Vec)),
							d2Vec),
						zQVec),
					opCode(mul_pd,
						opCode(mul_pd,
							opCode(add_pd, mPx, z2Vec),
							lbVec),
						mel));

				mPy = opCode(load_pd, &v[idxMz]);
#if	defined(__MIC__) || defined(__FMA__)
				tmp = opCode(fmadd_pd, mPx, dzcVec, mPy);
#else
				tmp = opCode(add_pd, mPy, opCode(mul_pd, mPx, dzcVec));
#endif
				opCode(store_pd,  &v[idxMz], tmp);
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
		float * __restrict__ v		= (float * __restrict__) v_;

		__assume_aligned(m, Align);
		__assume_aligned(v, Align);
#else
		const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_, Align);
		float * __restrict__ v		= (float * __restrict__) __builtin_assume_aligned (v_, Align);
#endif
		const float zR = *z;
		const float z2 = zR*zR;
		//const float zQ = 9.*powf(zR, nQcd+3.);
		const float zQ = (float) axionmass2( (double) zR, nQcd, 1.5, 3.)*zR*zR*zR;
		const float dzc = dz*c;
#ifdef	__MIC__
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) z2Aux[16]  = {-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2};
		const float __attribute__((aligned(Align))) zQAux[16]  = { zQ, 0., zQ, 0., zQ, 0., zQ, 0., zQ, 0., zQ, 0., zQ, 0., zQ, 0.};
		const float __attribute__((aligned(Align))) c6Aux[16]  = {-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.};
		const float __attribute__((aligned(Align))) lbAux[16]  = { LL, LL, LL, LL, LL, LL, LL, LL, LL, LL, LL, LL, LL, LL, LL, LL};
		const float __attribute__((aligned(Align))) d2Aux[16]  = { ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2 };
		const float __attribute__((aligned(Align))) dzcAux[16] = { dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc };
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) z2Aux[8]  = {-z2,-z2,-z2,-z2,-z2,-z2,-z2,-z2 };
		const float __attribute__((aligned(Align))) zQAux[8]  = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };
		const float __attribute__((aligned(Align))) c6Aux[8]  = {-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6. };
		const float __attribute__((aligned(Align))) lbAux[8]  = { LL, LL, LL, LL, LL, LL, LL, LL };
		const float __attribute__((aligned(Align))) d2Aux[8]  = { ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2 };
		const float __attribute__((aligned(Align))) dzcAux[8] = { dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc };
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const float __attribute__((aligned(Align))) z2Aux[4]  = {-z2,-z2,-z2,-z2 };
		const float __attribute__((aligned(Align))) zQAux[4]  = { zQ, 0., zQ, 0. };
		const float __attribute__((aligned(Align))) c6Aux[4]  = {-6.,-6.,-6.,-6. };
		const float __attribute__((aligned(Align))) lbAux[4]  = { LL, LL, LL, LL };
		const float __attribute__((aligned(Align))) d2Aux[4]  = { ood2, ood2, ood2, ood2 };
		const float __attribute__((aligned(Align))) dzcAux[4] = { dzc, dzc, dzc, dzc };
#endif
		const _MData_ z2Vec  = opCode(load_ps, z2Aux);
		const _MData_ zQVec  = opCode(load_ps, zQAux);
		const _MData_ c6Vec  = opCode(load_ps, c6Aux);
		const _MData_ lbVec  = opCode(load_ps, lbAux);
		const _MData_ d2Vec  = opCode(load_ps, d2Aux);
		const _MData_ dzcVec = opCode(load_ps, dzcAux);

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, mPx, mPy, mMx;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0;

				{
					size_t tmi = idx/XC, itp;

					itp = tmi/YC;
					X[1] = tmi - itp*YC;
					X[0] = idx - tmi*XC;
				}

				if (X[0] == XC-step)
					idxPx = ((idx - XC + step) << 1);
				else
					idxPx = ((idx + step) << 1);

				if (X[0] == 0)
					idxMx = ((idx + XC - step) << 1);
				else
					idxMx = ((idx - step) << 1);

				if (X[1] == 0)
				{
					idxMy = ((idx + Sf - XC) << 1);
					idxPy = ((idx + XC) << 1);

#ifdef	__MIC__
					mMx = opCode(swizzle_ps, opCode(load_ps, &m[idxMy]), _MM_SWIZ_REG_BADC);
					mPx = opCode(permute4f128_ps, mMx, _MM_PERM_CBAD);
					tmp = opCode(add_ps, opCode(mask_blend_ps, opCode(int2mask, 0b0011001100110011), mMx, mPx), opCode(load_ps, &m[idxPy]));
#elif	defined(__AVX2__)	//AVX2
					tmp = opCode(add_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 6,7,0,1,2,3,4,5)),  opCode(load_ps, &m[idxPy]));
#elif	defined(__AVX__)	//AVX
					mMx = opCode(permute_ps, opCode(load_ps, &m[idxMy]), 0b01001110);
					mPx = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
					tmp = opCode(add_ps, opCode(blend_ps, mMx, mPx, 0b00110011), opCode(load_ps, &m[idxPy]));
#else
					mMx = opCode(load_ps, &m[idxMy]);
					tmp = opCode(add_ps, opCode(shuffle_ps, mMx, mMx, 0b01001110), opCode(load_ps, &m[idxPy]));
#endif
				}
				else
				{
					idxMy = ((idx - XC) << 1);

					if (X[1] == YC-1)
					{
						idxPy = ((idx - Sf + XC) << 1);
#ifdef	__MIC__
						mMx = opCode(swizzle_ps, opCode(load_ps, &m[idxPy]), _MM_SWIZ_REG_BADC);
						mPx = opCode(permute4f128_ps, mMx, _MM_PERM_ADCB);
						tmp = opCode(add_ps, opCode(mask_blend_ps, opCode(int2mask, 0b1100110011001100), mMx, mPx), opCode(load_ps, &m[idxMy]));
#elif	defined(__AVX2__)	//AVX2
						tmp = opCode(add_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 2,3,4,5,6,7,0,1)), opCode(load_ps, &m[idxMy]));
#elif	defined(__AVX__)	//AVX
						mMx = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b01001110);
						mPx = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
						tmp = opCode(add_ps, opCode(blend_ps, mMx, mPx, 0b11001100), opCode(load_ps, &m[idxMy]));
#else
						mMx = opCode(load_ps, &m[idxPy]);
						tmp = opCode(add_ps, opCode(shuffle_ps, mMx, mMx, 0b01001110), opCode(load_ps, &m[idxMy]));
#endif
					}
					else
					{
						idxPy = ((idx + XC) << 1);
						tmp = opCode(add_ps, opCode(load_ps, &m[idxPy]), opCode(load_ps, &m[idxMy]));
					}
				}

				idxPz = ((idx+Sf) << 1);
				idxMz = ((idx-Sf) << 1);
				idxP0 = (idx << 1);

				mel = opCode(load_ps, &m[idxP0]);
				mPy = opCode(mul_ps, mel, mel);

#ifdef	__MIC__
				mPx = opCode(add_ps, opCode(swizzle_ps, mPy, _MM_SWIZ_REG_CDAB), mPy);
#elif	defined(__AVX__)
				mPx = opCode(add_ps, opCode(permute_ps, mPy, 0b10110001), mPy);
#else
				mPx = opCode(add_ps, opCode(shuffle_ps, mPy, mPy, 0b10110001), mPy);
#endif
				mMx = opCode(sub_ps,
					opCode(add_ps,
						opCode(mul_ps,
							opCode(add_ps,
								opCode(add_ps,
									opCode(load_ps, &m[idxMz]),
									opCode(add_ps,
										opCode(add_ps,
											opCode(add_ps, tmp, opCode(load_ps, &m[idxPx])),
											opCode(load_ps, &m[idxMx])),
										opCode(load_ps, &m[idxPz]))),
								opCode(mul_ps, mel, c6Vec)),
							d2Vec),
						zQVec),
					opCode(mul_ps,
						opCode(mul_ps,
							opCode(add_ps, mPx, z2Vec),
							lbVec),
						mel));
				mPy = opCode(load_ps, &v[idxMz]);

#if	defined(__MIC__) || defined(__FMA__)
				tmp = opCode(fmadd_ps, mMx, dzcVec, mPy);
#else
				tmp = opCode(add_ps, mPy, opCode(mul_ps, mMx, dzcVec));
#endif
				opCode(store_ps,  &v[idxMz], tmp);
			}
		}
#undef	_MData_
#undef	step
	}
}

void	propLowMemXeon	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision)
{
#ifdef USE_XEON
	const int  micIdx = commAcc();
	const size_t ext = V + S;
	const double ood2 = 1./delta2;
	double *z = (double *) __builtin_assume_aligned((void *) axionField->zV(), Align);
	double lambda = LL;

	int bulk  = 32;

	if (axionField->Lambda() != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX : ReUseX) signal(&bulk)
	{
		updateVXeon(mX, vX, z, dz, C1, ood2, LL, nQcd, Lx, 2*S, V, S, precision);
		updateMXeon(mX, vX, dz, D1, 3*S, V-S, S, precision);
	}
	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX : ReUseX)
	{
		updateVXeon(mX, vX, z, dz, C1, ood2, lambda, nQcd, Lx, S, 2*S, S, precision);
		updateVXeon(mX, vX, z, dz, C1, ood2, lambda, nQcd, Lx, V, ext, S, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	#pragma offload target(mic:micIdx) nocopy(mX, vX : ReUseX)
	{
		updateMXeon(mX, vX, dz, D1, S,   3*S, S, precision);
		updateMXeon(mX, vX, dz, D1, V-S, ext, S, precision);
	}

	*z += dz*D1;

	if (axionField->Lambda() != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX : ReUseX) signal(&bulk)
	{
		updateVXeon(mX, vX, z, dz, C2, ood2, lambda, nQcd, Lx, 2*S, V, S, precision);
		updateMXeon(mX, vX, dz, D2, 3*S, V-S, S, precision);
	}
	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX : ReUseX)
	{
		updateVXeon(mX, vX, z, dz, C2, ood2, lambda, nQcd, Lx, S, 2*S, S, precision);
		updateVXeon(mX, vX, z, dz, C2, ood2, lambda, nQcd, Lx, V, ext, S, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	#pragma offload target(mic:micIdx) nocopy(mX, vX : ReUseX)
	{
		updateMXeon(mX, vX, dz, D2, S,   3*S, S, precision);
		updateMXeon(mX, vX, dz, D2, V-S, ext, S, precision);
	}

	*z += dz*D2;

	if (axionField->Lambda() != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX : ReUseX) signal(&bulk)
	{
		updateVXeon(mX, vX, z, dz, C3, ood2, lambda, nQcd, Lx, 2*S, V, S, precision);
		updateMXeon(mX, vX, dz, D3, 3*S, V-S, S, precision);
	}
	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX : ReUseX)
	{
		updateVXeon(mX, vX, z, dz, C3, ood2, lambda, nQcd, Lx, S, 2*S, S, precision);
		updateVXeon(mX, vX, z, dz, C3, ood2, lambda, nQcd, Lx, V, ext, S, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	#pragma offload target(mic:micIdx) nocopy(mX, vX : ReUseX)
	{
		updateMXeon(mX, vX, dz, D3, S,   3*S, S, precision);
		updateMXeon(mX, vX, dz, D3, V-S, ext, S, precision);
	}

	*z += dz*D3;

	if (axionField->Lambda() != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX : ReUseX) signal(&bulk)
	{
		updateVXeon(mX, vX, z, dz, C4, ood2, lambda, nQcd, Lx, 2*S, V, S, precision);
		updateMXeon(mX, vX, dz, D4, 3*S, V-S, S, precision);
	}
	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX : ReUseX)
	{
		updateVXeon(mX, vX, z, dz, C4, ood2, lambda, nQcd, Lx, S, 2*S, S, precision);
		updateVXeon(mX, vX, z, dz, C4, ood2, lambda, nQcd, Lx, V, ext, S, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	#pragma offload target(mic:micIdx) nocopy(mX, vX : ReUseX)
	{
		updateMXeon(mX, vX, dz, D4, S,   3*S, S, precision);
		updateMXeon(mX, vX, dz, D4, V-S, ext, S, precision);
	}

	*z += dz*D4;

#endif
}
void	propLowMemCpu	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision)
{
	const double ood2 = 1./delta2;
	double *z = axionField->zV();
	double lambda = LL;

	axionField->sendGhosts(FIELD_M, COMM_SDRV);

	if (axionField->Lambda() != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	updateVXeon(axionField->mCpu(), axionField->vCpu(), z, dz, C1, ood2, lambda, nQcd, Lx, S, V + S, S, precision);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
	updateMXeon(axionField->mCpu(), axionField->vCpu(), dz, D1, S, V + S, S, precision);
	*z += dz*D1;

	axionField->sendGhosts(FIELD_M, COMM_SDRV);

	if (axionField->Lambda() != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	updateVXeon(axionField->mCpu(), axionField->vCpu(), z, dz, C2, ood2, lambda, nQcd, Lx, S, V + S, S, precision);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
	updateMXeon(axionField->mCpu(), axionField->vCpu(), dz, D2, S, V + S, S, precision);
	*z += dz*D2;

	axionField->sendGhosts(FIELD_M, COMM_SDRV);

	if (axionField->Lambda() != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	updateVXeon(axionField->mCpu(), axionField->vCpu(), z, dz, C3, ood2, lambda, nQcd, Lx, S, V + S, S, precision);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
	updateMXeon(axionField->mCpu(), axionField->vCpu(), dz, D3, S, V + S, S, precision);
	*z += dz*D3;

	axionField->sendGhosts(FIELD_M, COMM_SDRV);

	if (axionField->Lambda() != LAMBDA_FIXED)
		lambda = LL/((*z)*(*z));

	updateVXeon(axionField->mCpu(), axionField->vCpu(), z, dz, C4, ood2, lambda, nQcd, Lx, S, V + S, S, precision);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
	updateMXeon(axionField->mCpu(), axionField->vCpu(), dz, D4, S, V + S, S, precision);
	*z += dz*D4;
}
