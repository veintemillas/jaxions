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

//#if	defined(__AVX__) || defined(__AVX2__) || defined(__MIC__)
	#include <immintrin.h>
//#else
//	#include <xmmintrin.h>
//#endif


#ifdef	__MIC__
	#define	Align 64
	#define	_PREFIX_ _mm512
#else
	#if not defined(__AVX__) and not defined(__AVX2__)
//		#error("AVX instruction set required")
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
void	propThetaKernelXeon(const void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_, double *z, const double dz, const double c, const double d,
			    const double ood2, const double LL, const double nQcd, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision)
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

		const double __attribute__((aligned(Align))) zQAux[8]  = { zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ };
		const double __attribute__((aligned(Align))) izAux[8]  = { iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ };
		const double __attribute__((aligned(Align))) c6Aux[8]  = {-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6. };
		const double __attribute__((aligned(Align))) d2Aux[8]  = { ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2 };
		const double __attribute__((aligned(Align))) dzcAux[8] = { dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc };
		const double __attribute__((aligned(Align))) dzdAux[8] = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd };

		const int    __attribute__((aligned(Align))) shfRg[16] = {14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
		const int    __attribute__((aligned(Align))) shfLf[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1};
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const double __attribute__((aligned(Align))) zQAux[4]  = { zQ, zQ, zQ, zQ };
		const double __attribute__((aligned(Align))) izAux[4]  = { iZ, iZ, iZ, iZ };
		const double __attribute__((aligned(Align))) c6Aux[4]  = {-6.,-6.,-6.,-6. };
		const double __attribute__((aligned(Align))) d2Aux[4]  = { ood2, ood2, ood2, ood2 };
		const double __attribute__((aligned(Align))) dzcAux[4] = { dzc, dzc, dzc, dzc };
		const double __attribute__((aligned(Align))) dzdAux[4] = { dzd, dzd, dzd, dzd };
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) zQAux[2]  = { zQ, zQ };
		const double __attribute__((aligned(Align))) izAux[2]  = { iZ, iZ };
		const double __attribute__((aligned(Align))) c6Aux[2]  = {-6.,-6. };
		const double __attribute__((aligned(Align))) d2Aux[2]  = { ood2, ood2 };
		const double __attribute__((aligned(Align))) dzcAux[2] = { dzc, dzc };
		const double __attribute__((aligned(Align))) dzdAux[2] = { dzd, dzd };

#endif
		const _MData_ zQVec  = opCode(load_pd, zQAux);
		const _MData_ izVec  = opCode(load_pd, izAux);
		const _MData_ c6Vec  = opCode(load_pd, c6Aux);
		const _MData_ d2Vec  = opCode(load_pd, d2Aux);
		const _MData_ dzcVec = opCode(load_pd, dzcAux);
		const _MData_ dzdVec = opCode(load_pd, dzdAux);

#ifdef __MIC__
		const _MInt_  vShRg  = opCode(load_si512, shfRg);
		const _MInt_  vShLf  = opCode(load_si512, shfLf);
#endif

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
#ifdef	__MIC__
					tmp = opCode(add_pd,
						opCode(castsi512_pd, opCode(permutevar_epi32, vShRg, opCode(castpd_si512, opCode(load_pd, &m[idxMy])))),
						opCode(load_pd, &m[idxPy]));
#elif	defined(__AVX2__)	//AVX2
					tmp = opCode(add_pd, opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_ps, &m[idxMy])), opCode(setr_epi32, 6,7,0,1,2,3,4,5)),  opCode(load_ps, &m[idxPy])));
#elif	defined(__AVX__)
					mMx = opCode(permute_pd, opCode(load_pd, &m[idxMy]), 0b00000101);
					mPx = opCode(permute2f128_pd, mMx, mMx, 0b00000001);
					tmp = opCode(add_pd, opCode(blend_pd, mMx, mPx, 0b00000101), opCode(load_pd, &m[idxPy]));
#else
					tmp = opCode(add_pd, opCode(permute_pd, opCode(load_pd, &m[idxMy]), 0x00000001), opCode(load_pd, &m[idxPy]));
#endif
				}
				else
				{
					idxMy = ((idx - XC) << 1);

					if (X[1] == YC-1)
					{
						idxPy = ((idx - Sf + XC) << 1);
#ifdef	__MIC__
						tmp = opCode(add_pd,
							opCode(castsi512_pd, opCode(permutevar_epi32, vShLf, opCode(castpd_si512, opCode(load_pd, &m[idxPy])))),
							opCode(load_pd, &m[idxMy]));
#elif	defined(__AVX2__)	//AVX2
						tmp = opCode(add_pd, opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_pd, &m[idxPy])), opCode(setr_epi32, 2,3,4,5,6,7,0,1)),  opCode(load_pd, &m[idxMy])));
#elif	defined(__AVX__)
						mMx = opCode(permute_pd, opCode(load_pd, &m[idxPy]), 0b00000101);
						mPx = opCode(permute2f128_pd, mMx, mMx, 0b00000001);
						tmp = opCode(add_pd, opCode(blend_pd, mMx, mPx, 0b00001010), opCode(load_pd, &m[idxMy]));
#else
						tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(permute_pd, opCode(load_pd, &m[idxPy]), 0x00000001));
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
//#ifdef	__MIC__		NO SE SI VA A FUNCIONAR PORQUE CREO QUE EL SENO NO ESTA DEFINIDO EN KNC

//#else
				mMx = opCode(sub_pd, 
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
					opCode(mul_pd,
						zQVec,
						opCode(sin_pd, opCode(mul_pd, mel, izVec))));
//#endif
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

		const float __attribute__((aligned(Align))) zQAux[16]  = { zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ};
		const float __attribute__((aligned(Align))) izAux[16]  = { iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ };
		const float __attribute__((aligned(Align))) c6Aux[16]  = {-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6.};
		const float __attribute__((aligned(Align))) d2Aux[16]  = { ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2 };
		const float __attribute__((aligned(Align))) dzcAux[16] = { dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc };
		const float __attribute__((aligned(Align))) dzdAux[16] = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd };
#elif	defined(__AVX__)
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) zQAux[8]  = { zQ, zQ, zQ, zQ, zQ, zQ, zQ, zQ };
		const float __attribute__((aligned(Align))) izAux[8]  = { iZ, iZ, iZ, iZ, iZ, iZ, iZ, iZ };
		const float __attribute__((aligned(Align))) c6Aux[8]  = {-6.,-6.,-6.,-6.,-6.,-6.,-6.,-6. };
		const float __attribute__((aligned(Align))) d2Aux[8]  = { ood2, ood2, ood2, ood2, ood2, ood2, ood2, ood2 };
		const float __attribute__((aligned(Align))) dzcAux[8] = { dzc, dzc, dzc, dzc, dzc, dzc, dzc, dzc };
		const float __attribute__((aligned(Align))) dzdAux[8] = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd };
#else
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) zQAux[4]  = { zQ, zQ, zQ, zQ };
		const float __attribute__((aligned(Align))) izAux[4]  = { iZ, iZ, iZ, iZ };
		const float __attribute__((aligned(Align))) c6Aux[4]  = {-6.,-6.,-6.,-6. };
		const float __attribute__((aligned(Align))) lbAux[4]  = { LL, LL, LL, LL };
		const float __attribute__((aligned(Align))) d2Aux[4]  = { ood2, ood2, ood2, ood2 };
		const float __attribute__((aligned(Align))) dzcAux[4] = { dzc, dzc, dzc, dzc };
		const float __attribute__((aligned(Align))) dzdAux[4] = { dzd, dzd, dzd, dzd };
#endif


		const _MData_ zQVec  = opCode(load_ps, zQAux);
		const _MData_ izVec  = opCode(load_ps, izAux);
		const _MData_ c6Vec  = opCode(load_ps, c6Aux);
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
					mMx = opCode(swizzle_ps, opCode(load_ps, &m[idxMy]), _MM_SWIZ_REG_CBAD);
					mPx = opCode(permute4f128_ps, mMx, _MM_PERM_CBAD);
					tmp = opCode(add_ps, opCode(mask_blend_ps, opCode(int2mask, 0b0001000100010001), mMx, mPx), opCode(load_ps, &m[idxPy]));
#elif	defined(__AVX2__)	//AVX2
					tmp = opCode(add_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 7,0,1,2,3,4,5,6)),  opCode(load_ps, &m[idxPy]));
#elif	defined(__AVX__)	//AVX
					mMx = opCode(permute_ps, opCode(load_ps, &m[idxMy]), 0b10010011);
					mPx = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
					tmp = opCode(add_ps, opCode(blend_ps, mMx, mPx, 0b00010001), opCode(load_ps, &m[idxPy]));
#else
					mMx = opCode(load_ps, &m[idxMy]);
					tmp = opCode(add_ps, opCode(shuffle_ps, mMx, mMx, 0b10010011), opCode(load_ps, &m[idxPy]));
#endif
				}
				else
				{
					idxMy = ((idx - XC) << 1);

					if (X[1] == YC-1)
					{
						idxPy = ((idx - Sf + XC) << 1);
#ifdef	__MIC__
						mMx = opCode(swizzle_ps, opCode(load_ps, &m[idxPy]), _MM_SWIZ_REG_ADCB);
						mPx = opCode(permute4f128_ps, mMx, _MM_PERM_ADCB);
						tmp = opCode(add_ps, opCode(mask_blend_ps, opCode(int2mask, 0b1110111011101110), mMx, mPx), opCode(load_ps, &m[idxMy]));
#elif	defined(__AVX2__)	//AVX2
						tmp = opCode(add_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 1,2,3,4,5,6,7,0)), opCode(load_ps, &m[idxMy]));
#elif	defined(__AVX__)	//AVX
						mMx = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b00111001);
						mPx = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
						tmp = opCode(add_ps, opCode(blend_ps, mMx, mPx, 0b10001000), opCode(load_ps, &m[idxMy]));
#else
						mPx = opCode(load_ps, &m[idxPy]);
						tmp = opCode(add_ps, opCode(shuffle_ps, mPx, mPx, 0b00111001), opCode(load_ps, &m[idxMy]));
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
//#ifdef	__MIC__		NO SE SI VA A FUNCIONAR PORQUE CREO QUE EL SENO NO ESTA DEFINIDO EN KNC

//#else
				mMx = opCode(sub_ps, 
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
					opCode(mul_ps,
						zQVec,
						opCode(sin_ps, opCode(mul_ps, mel, izVec))));
//#endif
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
		propagateKernelXeon(mX, vX, m2X, z, dz, C1, D1, ood2, nQcd, Lx, 2*S, V, precision);
	}
	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		propagateKernelXeon(mX, vX, m2X, z, dz, C1, D1, ood2, nQcd, Lx, S, 2*S, precision);
		propagateKernelXeon(mX, vX, m2X, z, dz, C1, D1, ood2, nQcd, Lx, V, ext, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	*z += dz*D1;

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		propagateKernelXeon(m2X, vX, mX, z, dz, C2, D2, ood2, nQcd, Lx, 2*S, V, precision);
	}
	axionField->exchangeGhosts(FIELD_M2);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		propagateKernelXeon(m2X, vX, mX, z, dz, C2, D2, ood2, nQcd, Lx, S, 2*S, precision);
		propagateKernelXeon(m2X, vX, mX, z, dz, C2, D2, ood2, nQcd, Lx, V, ext, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	*z += dz*D2;

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		propagateKernelXeon(mX, vX, m2X, z, dz, C3, D3, ood2, nQcd, Lx, 2*S, V, precision);
	}
	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		propagateKernelXeon(mX, vX, m2X, z, dz, C3, D3, ood2, nQcd, Lx, S, 2*S, precision);
		propagateKernelXeon(mX, vX, m2X, z, dz, C3, D3, ood2, nQcd, Lx, V, ext, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	*z += dz*D3;

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		propagateKernelXeon(m2X, vX, mX, z, dz, C4, D4, ood2, nQcd, Lx, 2*S, V, precision);
	}
	axionField->exchangeGhosts(FIELD_M2);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		propagateKernelXeon(m2X, vX, mX, z, dz, C4, D4, ood2, nQcd, Lx, S, 2*S, precision);
		propagateKernelXeon(m2X, vX, mX, z, dz, C4, D4, ood2, nQcd, Lx, V, ext, precision);
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
        propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C1, D1, ood2, nQcd, Lx, 2*S, V, precision);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
        propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C1, D1, ood2, nQcd, Lx, S, 2*S, precision);
        propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C1, D1, ood2, nQcd, Lx, V, V+S, precision);
	*z += dz*D1;

	axionField->sendGhosts(FIELD_M2, COMM_SDRV);
        propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C2, D2, ood2, nQcd, Lx, 2*S, V, precision);
	axionField->sendGhosts(FIELD_M2, COMM_WAIT);
        propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C2, D2, ood2, nQcd, Lx, S, 2*S, precision);
        propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C2, D2, ood2, nQcd, Lx, V, V+S, precision);
	*z += dz*D2;

	axionField->sendGhosts(FIELD_M, COMM_SDRV);
        propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C3, D3, ood2, nQcd, Lx, 2*S, V, precision);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
        propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C3, D3, ood2, nQcd, Lx, S, 2*S, precision);
        propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C3, D3, ood2, nQcd, Lx, V, V+S, precision);
	*z += dz*D3;

	axionField->sendGhosts(FIELD_M2, COMM_SDRV);
        propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C4, D4, ood2, nQcd, Lx, 2*S, V, precision);
	axionField->sendGhosts(FIELD_M2, COMM_WAIT);
        propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C4, D4, ood2, nQcd, Lx, S, 2*S, precision);
        propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C4, D4, ood2, nQcd, Lx, V, V+S, precision);
	*z += dz*D4;
}

