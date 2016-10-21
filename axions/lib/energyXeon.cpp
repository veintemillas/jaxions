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


#include<cstdio>
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
void	energyKernelXeon(const void * __restrict__ m_, void * __restrict__ v_, double *z, const double dz, const double ood2, const double LL,
			 const double nQcd, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision)
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

		__assume_aligned(m, Align);
		__assume_aligned(v, Align);
#else
		const double * __restrict__ m	= (const double * __restrict__) __builtin_assume_aligned (m_, Align);
		double * __restrict__ v		= (double * __restrict__) __builtin_assume_aligned (v_, Align);
#endif
		const double zR  = *z;
		const double iz  = 1./zR;
		const double iz2 = iz*iz;
		const double zQ = 9.*pow(zR, nQcd+2.);
		const double lZ = 0.25*L*L*zR*zR;

#ifdef	__MIC__
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const double __attribute__((aligned(Align))) oneAux[8]  = { 1., 1., 1., 1., 1., 1., 1., 1. };
		const double __attribute__((aligned(Align))) cjgAux[8]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
		const double __attribute__((aligned(Align))) ivZAux[8]  = { iz, 0., iz, 0., iz, 0., iz, 0. };	// Only real part
		const double __attribute__((aligned(Align))) ivZ2Aux[8] = {iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2 };
#elif	defined(__AVX__)
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) oneAux[4]  = { 1., 1., 1., 1. };
		const double __attribute__((aligned(Align))) cjgAux[4]  = { 1.,-1., 1.,-1. };
		const double __attribute__((aligned(Align))) ivZAux[4]  = { iz, 0., iz, 0. };	// Only real part
		const double __attribute__((aligned(Align))) ivZ2Aux[4] = {iz2,iz2,iz2,iz2 };
#else
		const size_t XC = Lx;
		const size_t YC = Lx;

		const double __attribute__((aligned(Align))) oneAux[2]  = { 1., 1. };
		const double __attribute__((aligned(Align))) cjgAux[2]  = { 1.,-1. };
		const double __attribute__((aligned(Align))) ivZAux[2]  = { iz, 0. };	// Only real part
		const double __attribute__((aligned(Align))) ivZ2Aux[2] = {iz2,iz2 };

#endif
		const _MData_ one  = opCode(load_pd, oneAux);
		const _MData_ cjg  = opCode(load_pd, cjgAux);
		const _MData_ ivZ  = opCode(load_pd, ivZAux);
		const _MData_ ivZ2 = opCode(load_pd, ivZ2Aux);

		#pragma omp parallel default(shared) 
		{
			_MData_ tmp1, tmp2, mel, vel, mMx, mMy, mMz, mdv, mod, mTp;
			_MData_ Grx,  Gry,  Grz, tGp, tVp, tKp, mCg, mSg;

			#pragma omp for schedule(static) reduction(+:Vrho,Vth,Krho,Kth,Grho,Gth)
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
					mMy = opCode(add_pd, opCode(load_pd, &m[idxPy]), opCode(castps_pd, opCode(permute4f128_ps, opCode(castpd_ps, opCode(load_pd, &m[idxMy])), _MM_PERM_CBAD)));
#elif	defined(__AVX__)
					mMx = opCode(load_pd, &m[idxMy]);
					mMy = opCode(add_pd, opCode(load_pd, &m[idxPy]), opCode(permute2f128_pd, mMx, mMx, 0b00000001));
#else
					mMy = opCode(sub_pd, opCode(load_pd, &m[idxPy]), opCode(load_pd, &m[idxMy]));
#endif
				}
				else
				{
					idxMy = ((idx - XC) << 1);

					if (X[1] == YC-1)
					{
						idxPy = ((idx - Sf + XC) << 1);
#ifdef	__MIC__
						mMy = opCode(sub_pd, opCode(castps_pd, opCode(permute4f128_ps, opCode(castpd_ps, opCode(load_pd, &m[idxPy])), _MM_PERM_ADCB)), opCode(load_pd, &m[idxMy]));
#elif	defined(__AVX__)
						mMx = opCode(load_pd, &m[idxPy]);
						mMy = opCode(sub_pd, opCode(permute2f128_pd, mMx, mMx, 0b00000001), opCode(load_pd, &m[idxMy]));
#else
						mMy = opCode(sub_pd, opCode(load_pd, &m[idxPy]), opCode(load_pd, &m[idxMy]));
#endif
					}
					else
					{
						idxPy = ((idx + XC) << 1);
						mMy = opCode(sub_pd, opCode(load_pd, &m[idxPy]), opCode(load_pd, &m[idxMy]));
					}
				}

				// Tienes mMy y los puntos para mMx y mMz. Calcula todo ya!!!

				idxPz = ((idx+Sf) << 1);
				idxMz = ((idx-Sf) << 1);
				idxP0 = (idx << 1);

				// Empiezo aqui
				mMx = opCode(sub_pd, opCode(load_pd, &m[idxPx]), opCode(load_pd, &m[idxMx]));
				// mMy ya esta cargado
				mMz = opCode(sub_pd, opCode(load_pd, &m[idzPx]), opCode(load_pd, &m[idzMx]));

				mel = opCode(load_pd, &m[idxP0]);//Carga m
				vel = opCode(load_pd, &v[idxMz]);//Carga v
				mod = opCode(mul_pd, mel, mel);

#ifdef	__MIC__
				mTp = opCode(add_pd, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, mod), _MM_PERM_BADC)), mod);
#elif defined(__AVX__)
				mTp = opCode(add_pd, opCode(permute_pd, mod, 0b00000101), mod);
#else
				mTp = opCode(add_pd, opCode(shuffle_pd, mod, mod, 0b00000001), mod);
#endif
				mod = opCode(mul_pd, mTp, ivZ2);	// Factor |mel|^2/z^2, util luego

				mCg = opCode(div_pd, opCode(mul_pd, mel, cjg), mTp);	// Ahora mCg tiene 1/mel

				// Meto en mSg = shuffled(mCg) (Intercambio parte real por parte imaginaria)
#ifdef	__MIC__
				mSg = opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, mCg), _MM_PERM_BADC));
#elif defined(__AVX__)
				mSg = opCode(permute_pd, mCg, 0b00000101);
#else
				mSg = opCode(shuffle_pd, mCg, mCg, 0b00000001);
#endif

				// Calculo los gradientes
#ifdef	__MIC__
				tmp1 = opCode(mul_pd, mMx, mCg);
				tmp2 = opCode(mul_pd, mMx, mSg);

				tmp1 = opCode(mask_add_pd,
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tmp2), _MM_PERM_BADC)),
					opCode(int2mask, 0b0000000001010101),
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tmp1), _MM_PERM_BADC)),
					tmp1);

				Grx = opCode(mask_add_pd, tmp1, opCode(int2mask, 0b0000000010101010), tmp1, tmp2);

				tmp1 = opCode(mul_pd, mMy, mCg);
				tmp2 = opCode(mul_pd, mMy, mSg);

				tmp1 = opCode(mask_add_pd,
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tmp2), _MM_PERM_BADC)),
					opCode(int2mask, 0b0000000001010101),
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tmp1), _MM_PERM_BADC)),
					tmp1);

				Gry = opCode(mask_add_pd, tmp1, opCode(int2mask, 0b0000000010101010), tmp1, tmp2);

				tmp1 = opCode(mul_pd, mMz, mCg);
				tmp2 = opCode(mul_pd, mMz, mSg);

				tmp1 = opCode(mask_add_pd,
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tmp2), _MM_PERM_BADC)),
					opCode(int2mask, 0b0000000001010101),
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tmp1), _MM_PERM_BADC)),
					tmp1);

				Grz = opCode(mask_add_pd, tmp1, opCode(int2mask, 0b0000000010101010), tmp1, tmp2);
#else				// Las instrucciones se llaman igual con AVX o con SSE3
				Grx = opCode(hadd_pd, opCode(mul_pd, mMx, mCg), opCode(mul_pd, mMx, mSg));
				Gry = opCode(hadd_pd, opCode(mul_pd, mMy, mCg), opCode(mul_pd, mMy, mSg));
				Grz = opCode(hadd_pd, opCode(mul_pd, mMz, mCg), opCode(mul_pd, mMz, mSg));
				mdv = opCode(sub_pd,  opCode(hadd_pd, opCode(mul_pd, vel, mCg), opCode(mul_pd, vel, mSg)), ivZ);
#endif
				tGp = opCode(add_pd,
					tGp,
					opCode(mul_pd,
						opcode(add_pd,
							opCode(add_pd,
								opCode(mul_pd, Grx, Grx),
								opCode(mul_pd, Gry, Gry)),
							opCode(mul_pd, Grz, Grz)),
						mod));
					
				tKp = opCode(add_pd, tKp, opCode(mul_pd, opCode(mul_pd, mdv, mdv), mod));

				mTp = opCode(sub_pd, mod, one);
				mod = opCode(mul_pd, mTp, mTp);
				mTp = opCode(sub_pd, one, opCode(mul_pd, mel, ivZ));
#ifdef	__MIC__
				vel = opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, mTp), _MM_PERM_BADC));
				tVp =opCode(add_pd, tVp, opCode(mask_blend_pd, opCode(int2mask, 0b0000000010101010), mod, vel));
#elif defined(__AVX__)
				tVp = opCode(add_pd, tVp, opCode(blend_pd, mod, opCode(permute_pd, mTp, 0b00000101), 0b10101010);
#else
				tVp = opCode(add_pd, tVp, opCode(shuffle_pd, mod, mTp, 0b00000001));
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

		const float zR  = *z;
		const float iz  = 1./zR;
		const float iz2 = iz*iz;
		const float zQ = 9.f*powf(zR, nQcd+2.);
		const float lZ = 0.25f*L*L*zR*zR;

#ifdef	__MIC__
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) oneAux[16]  = { 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
		const float __attribute__((aligned(Align))) cjgAux[16]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
		const float __attribute__((aligned(Align))) ivZAux[16]  = { iz, 0., iz, 0., iz, 0., iz, 0., iz, 0., iz, 0., iz, 0., iz, 0. };
		const float __attribute__((aligned(Align))) ivZ2Aux[16] = {iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2 };
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) oneAux[8]  = { 1., 1., 1., 1., 1., 1., 1., 1. };
		const float __attribute__((aligned(Align))) cjgAux[8]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
		const float __attribute__((aligned(Align))) ivZAux[8]  = { iz, 0., iz, 0., iz, 0., iz, 0. };	// Only real part
		const float __attribute__((aligned(Align))) ivZ2Aux[8] = {iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2 };
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const float __attribute__((aligned(Align))) oneAux[4]  = { 1., 1., 1., 1. };
		const float __attribute__((aligned(Align))) cjgAux[4]  = { 1.,-1., 1.,-1. };
		const float __attribute__((aligned(Align))) ivZAux[4]  = { iz, 0., iz, 0. };	// Only real part
		const float __attribute__((aligned(Align))) ivZ2Aux[4] = {iz2,iz2,iz2,iz2 };
#endif

		const _MData_ one  = opCode(load_ps, oneAux);
		const _MData_ cjg  = opCode(load_ps, cjgAux);
		const _MData_ ivZ  = opCode(load_ps, ivZAux);
		const _MData_ ivZ2 = opCode(load_ps, ivZ2Aux);

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
					mMz = opCode(permute4f128_ps, mMx, _MM_PERM_CBAD);
					mMy = opCode(sub_ps, opCode(load_ps, &m[idxPy]), opCode(mask_blend_ps, opCode(int2mask, 0b0011001100110011), mMx, mMz));
#elif	defined(__AVX2__)
					mMy = opCode(sub_ps,  opCode(load_ps, &m[idxPy]), opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 6,7,0,1,2,3,4,5)));
#elif	defined(__AVX__)
					mMx = opCode(permute_ps, opCode(load_ps, &m[idxMy]), 0b01001110);
					mMz = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
					mMy = opCode(sub_ps, opCode(load_ps, &m[idxPy]), opCode(blend_ps, mMx, mMz, 0b00110011));
#else
					mMx = opCode(load_ps, &m[idxMy]);
					mMy = opCode(sub_ps, opCode(load_ps, &m[idxPy]), opCode(shuffle_ps, mMx, mMx, 0b01001110));
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
						mMz = opCode(permute4f128_ps, mMx, _MM_PERM_ADCB);
						mMy = opCode(sub_ps, opCode(mask_blend_ps, opCode(int2mask, 0b1100110011001100), mMx, mMz), opCode(load_ps, &m[idxMy]));
#elif	defined(__AVX2__)
						mMy = opCode(sub_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 2,3,4,5,6,7,0,1)), opCode(load_ps, &m[idxMy]));
#elif	defined(__AVX__)
						mMx = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b01001110);
						mMz = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
						mMy = opCode(sub_ps, opCode(blend_ps, mMx, mMz, 0b11001100), opCode(load_ps, &m[idxMy]));
#else
						mMx = opCode(load_ps, &m[idxPy]);
						mMy = opCode(sub_ps, opCode(shuffle_ps, mMx, mMx, 0b01001110), opCode(load_ps, &m[idxMy]));
#endif
					}
					else
					{
						idxPy = ((idx + XC) << 1);
						mMy = opCode(sub_ps, opCode(load_ps, &m[idxPy]), opCode(load_ps, &m[idxMy]));
					}
				}

				idxPz = ((idx+Sf) << 1);
				idxMz = ((idx-Sf) << 1);
				idxP0 = (idx << 1);

				// Empiezo aqui
				mMx = opCode(sub_ps, opCode(load_ps, &m[idxPx]), opCode(load_ps, &m[idxMx]));
				// mMy ya esta cargado
				mMz = opCode(sub_ps, opCode(load_ps, &m[idzPx]), opCode(load_ps, &m[idzMx]));

				mel = opCode(load_ps, &m[idxP0]);//Carga m
				vel = opCode(load_ps, &v[idxMz]);//Carga v
				mod = opCode(mul_ps, mel, mel);

#ifdef	__MIC__
				mTp = opCode(add_ps, opCode(swizzle_ps, mod, _MM_SWIZ_REG_CDAB), mod);
#elif defined(__AVX__)
				mTp = opCode(add_ps, opCode(permute_ps, mod, 0b10110001), mod);
#else
				mTp = opCode(add_ps, opCode(shuffle_ps, mod, mod, 0b10110001), mod);
#endif
				mod = opCode(mul_ps, mTp, ivZ2);	// Factor |mel|^2/z^2, util luego

				mCg = opCode(div_ps, opCode(mul_ps, mel, cjg), mTp);	// Ahora mCg tiene 1/mel

				// Meto en mSg = shuffled(mCg) (Intercambio parte real por parte imaginaria)
#ifdef	__MIC__
				mSg = opCode(swizzle_ps, mCg, _MM_SWIZ_REG_CDAB);
#elif defined(__AVX__)
				mSg = opCode(permute_ps, mCg, 0b10110001);
#else
				mSg = opCode(shuffle_ps, mCg, mCg, 0b10110001);
#endif

				// Calculo los gradientes
#ifdef	__MIC__
				tmp1 = opCode(mul_ps, mMx, mCg);
				tmp2 = opCode(mul_ps, mMx, mSg);

				tmp1 = opCode(mask_add_ps,
					opCode(swizzle_ps, tmp2, _MM_SWIZ_REG_CDAB),
					opCode(int2mask, 0b0101010101010101),
					opCode(swizzle_ps, tmp1, _MM_SWIZ_REG_CDAB),
					tmp1);

				Grx = opCode(mask_add_ps, tmp1, opCode(int2mask, 0b1010101010101010), tmp1, tmp2);

				tmp1 = opCode(mul_ps, mMy, mCg);
				tmp2 = opCode(mul_ps, mMy, mSg);

				tmp1 = opCode(mask_add_ps,
					opCode(swizzle_ps, tmp2, _MM_SWIZ_REG_CDAB),
					opCode(int2mask, 0b0101010101010101),
					opCode(swizzle_ps, tmp1, _MM_SWIZ_REG_CDAB),
					tmp1);

				Gry = opCode(mask_add_ps, tmp1, opCode(int2mask, 0b1010101010101010), tmp1, tmp2);

				tmp1 = opCode(mul_ps, mMz, mCg);
				tmp2 = opCode(mul_ps, mMz, mSg);

				tmp1 = opCode(mask_add_ps,
					opCode(swizzle_ps, tmp2, _MM_SWIZ_REG_CDAB),
					opCode(int2mask, 0b0101010101010101),
					opCode(swizzle_ps, tmp1, _MM_SWIZ_REG_CDAB),
					tmp1);

				Grz = opCode(mask_add_ps, tmp1, opCode(int2mask, 0b1010101010101010), tmp1, tmp2);
#else				// Las instrucciones se llaman igual con AVX o con SSE3
				TODO ESTA PARTE + mdv CON XEON PHI, QUE SE TE HA OLVIDADO

				Grx = opCode(hadd_pd, opCode(mul_pd, mMx, mCg), opCode(mul_pd, mMx, mSg));
				Gry = opCode(hadd_pd, opCode(mul_pd, mMy, mCg), opCode(mul_pd, mMy, mSg));
				Grz = opCode(hadd_pd, opCode(mul_pd, mMz, mCg), opCode(mul_pd, mMz, mSg));
				mdv = opCode(sub_pd,  opCode(hadd_pd, opCode(mul_pd, vel, mCg), opCode(mul_pd, vel, mSg)), ivZ);
#endif
				tGp = opCode(add_ps,
					tGp,
					opCode(mul_ps,
						opcode(add_ps,
							opCode(add_ps,
								opCode(mul_ps, Grx, Grx),
								opCode(mul_ps, Gry, Gry)),
							opCode(mul_ps, Grz, Grz)),
						mod));
					
				tKp = opCode(add_ps, tKp, opCode(mul_ps, opCode(mul_ps, mdv, mdv), mod));

				mTp = opCode(sub_ps, mod, one);
				mod = opCode(mul_ps, mTp, mTp);
				mTp = opCode(sub_ps, one, opCode(mul_ps, mel, ivZ));
#ifdef	__MIC__
				vel = opCode(swizzle_ps, mTp, _MM_SWIZ_REG_CDAB);
				tVp =opCode(add_ps, tVp, opCode(mask_blend_ps, opCode(int2mask, 0b1010101010101010), mod, vel));
#elif defined(__AVX__)
				tVp = opCode(add_ps, tVp, opCode(blend_ps, mod, opCode(permute_pd, mTp, 0b10110001), 0b10101010);
#else
				tVp = opCode(add_ps, tVp, opCode(shuffle_pd, mod, mTp, 0b10110001));
#endif
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

	int bulk  = 32;

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		propagateKernelXeon(mX, vX, m2X, z, dz, C1, D1, ood2, LL, nQcd, Lx, 2*S, V, precision);
	}
	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		propagateKernelXeon(mX, vX, m2X, z, dz, C1, D1, ood2, LL, nQcd, Lx, S, 2*S, precision);
		propagateKernelXeon(mX, vX, m2X, z, dz, C1, D1, ood2, LL, nQcd, Lx, V, ext, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	*z += dz*D1;

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		propagateKernelXeon(m2X, vX, mX, z, dz, C2, D2, ood2, LL, nQcd, Lx, 2*S, V, precision);
	}
	axionField->exchangeGhosts(FIELD_M2);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		propagateKernelXeon(m2X, vX, mX, z, dz, C2, D2, ood2, LL, nQcd, Lx, S, 2*S, precision);
		propagateKernelXeon(m2X, vX, mX, z, dz, C2, D2, ood2, LL, nQcd, Lx, V, ext, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	*z += dz*D2;

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		propagateKernelXeon(mX, vX, m2X, z, dz, C3, D3, ood2, LL, nQcd, Lx, 2*S, V, precision);
	}
	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		propagateKernelXeon(mX, vX, m2X, z, dz, C3, D3, ood2, LL, nQcd, Lx, S, 2*S, precision);
		propagateKernelXeon(mX, vX, m2X, z, dz, C3, D3, ood2, LL, nQcd, Lx, V, ext, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	*z += dz*D3;

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		propagateKernelXeon(m2X, vX, mX, z, dz, C4, D4, ood2, LL, nQcd, Lx, 2*S, V, precision);
	}
	axionField->exchangeGhosts(FIELD_M2);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		propagateKernelXeon(m2X, vX, mX, z, dz, C4, D4, ood2, LL, nQcd, Lx, S, 2*S, precision);
		propagateKernelXeon(m2X, vX, mX, z, dz, C4, D4, ood2, LL, nQcd, Lx, V, ext, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	*z += dz*D4;
#endif
}

void	propagateCpu	(Scalar *axionField, const double dz, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision)
{
	const double ood2 = 1./delta2;
	double *z = axionField->zV();

	axionField->sendGhosts(FIELD_M, COMM_SDRV);
        propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C1, D1, ood2, LL, nQcd, Lx, 2*S, V, precision);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
        propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C1, D1, ood2, LL, nQcd, Lx, S, 2*S, precision);
        propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C1, D1, ood2, LL, nQcd, Lx, V, V+S, precision);
	*z += dz*D1;

	axionField->sendGhosts(FIELD_M2, COMM_SDRV);
        propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C2, D2, ood2, LL, nQcd, Lx, 2*S, V, precision);
	axionField->sendGhosts(FIELD_M2, COMM_WAIT);
        propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C2, D2, ood2, LL, nQcd, Lx, S, 2*S, precision);
        propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C2, D2, ood2, LL, nQcd, Lx, V, V+S, precision);
	*z += dz*D2;

	axionField->sendGhosts(FIELD_M, COMM_SDRV);
        propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C3, D3, ood2, LL, nQcd, Lx, 2*S, V, precision);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
        propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C3, D3, ood2, LL, nQcd, Lx, S, 2*S, precision);
        propagateKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, dz, C3, D3, ood2, LL, nQcd, Lx, V, V+S, precision);
	*z += dz*D3;

	axionField->sendGhosts(FIELD_M2, COMM_SDRV);
        propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C4, D4, ood2, LL, nQcd, Lx, 2*S, V, precision);
	axionField->sendGhosts(FIELD_M2, COMM_WAIT);
        propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C4, D4, ood2, LL, nQcd, Lx, S, 2*S, precision);
        propagateKernelXeon(axionField->m2Cpu(), axionField->vCpu(), axionField->mCpu(), z, dz, C4, D4, ood2, LL, nQcd, Lx, V, V+S, precision);
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
		const double zQ = 9.*pow(zR, nQcd+3.);
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
		const float zQ = 9.*powf(zR, nQcd+3.);
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

	int bulk  = 32;

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX : ReUseX) signal(&bulk)
	{
		updateVXeon(mX, vX, z, dz, C1, ood2, LL, nQcd, Lx, 2*S, V, S, precision);
		updateMXeon(mX, vX, dz, D1, 3*S, V-S, S, precision);
	}
	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX : ReUseX)
	{
		updateVXeon(mX, vX, z, dz, C1, ood2, LL, nQcd, Lx, S, 2*S, S, precision);
		updateVXeon(mX, vX, z, dz, C1, ood2, LL, nQcd, Lx, V, ext, S, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	#pragma offload target(mic:micIdx) nocopy(mX, vX : ReUseX)
	{
		updateMXeon(mX, vX, dz, D1, S,   3*S, S, precision);
		updateMXeon(mX, vX, dz, D1, V-S, ext, S, precision);
	}

	*z += dz*D1;

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX : ReUseX) signal(&bulk)
	{
		updateVXeon(mX, vX, z, dz, C2, ood2, LL, nQcd, Lx, 2*S, V, S, precision);
		updateMXeon(mX, vX, dz, D2, 3*S, V-S, S, precision);
	}
	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX : ReUseX)
	{
		updateVXeon(mX, vX, z, dz, C2, ood2, LL, nQcd, Lx, S, 2*S, S, precision);
		updateVXeon(mX, vX, z, dz, C2, ood2, LL, nQcd, Lx, V, ext, S, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	#pragma offload target(mic:micIdx) nocopy(mX, vX : ReUseX)
	{
		updateMXeon(mX, vX, dz, D2, S,   3*S, S, precision);
		updateMXeon(mX, vX, dz, D2, V-S, ext, S, precision);
	}
	*z += dz*D2;

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX : ReUseX) signal(&bulk)
	{
		updateVXeon(mX, vX, z, dz, C3, ood2, LL, nQcd, Lx, 2*S, V, S, precision);
		updateMXeon(mX, vX, dz, D3, 3*S, V-S, S, precision);
	}
	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX : ReUseX)
	{
		updateVXeon(mX, vX, z, dz, C3, ood2, LL, nQcd, Lx, S, 2*S, S, precision);
		updateVXeon(mX, vX, z, dz, C3, ood2, LL, nQcd, Lx, V, ext, S, precision);
	}
	#pragma offload_wait target(mic:micIdx) wait(&bulk)

	#pragma offload target(mic:micIdx) nocopy(mX, vX : ReUseX)
	{
		updateMXeon(mX, vX, dz, D3, S,   3*S, S, precision);
		updateMXeon(mX, vX, dz, D3, V-S, ext, S, precision);
	}
	*z += dz*D3;

	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX : ReUseX) signal(&bulk)
	{
		updateVXeon(mX, vX, z, dz, C4, ood2, LL, nQcd, Lx, 2*S, V, S, precision);
		updateMXeon(mX, vX, dz, D4, 3*S, V-S, S, precision);
	}
	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX : ReUseX)
	{
		updateVXeon(mX, vX, z, dz, C4, ood2, LL, nQcd, Lx, S, 2*S, S, precision);
		updateVXeon(mX, vX, z, dz, C4, ood2, LL, nQcd, Lx, V, ext, S, precision);
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

	axionField->sendGhosts(FIELD_M, COMM_SDRV);
	updateVXeon(axionField->mCpu(), axionField->vCpu(), z, dz, C1, ood2, LL, nQcd, Lx, S, V + S, S, precision);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
	updateMXeon(axionField->mCpu(), axionField->vCpu(), dz, D1, S, V + S, S, precision);
	*z += dz*D1;

	axionField->sendGhosts(FIELD_M, COMM_SDRV);
	updateVXeon(axionField->mCpu(), axionField->vCpu(), z, dz, C2, ood2, LL, nQcd, Lx, S, V + S, S, precision);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
	updateMXeon(axionField->mCpu(), axionField->vCpu(), dz, D2, S, V + S, S, precision);
	*z += dz*D2;

	axionField->sendGhosts(FIELD_M, COMM_SDRV);
	updateVXeon(axionField->mCpu(), axionField->vCpu(), z, dz, C3, ood2, LL, nQcd, Lx, S, V + S, S, precision);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
	updateMXeon(axionField->mCpu(), axionField->vCpu(), dz, D3, S, V + S, S, precision);
	*z += dz*D3;

	axionField->sendGhosts(FIELD_M, COMM_SDRV);
	updateVXeon(axionField->mCpu(), axionField->vCpu(), z, dz, C4, ood2, LL, nQcd, Lx, S, V + S, S, precision);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
	updateMXeon(axionField->mCpu(), axionField->vCpu(), dz, D4, S, V + S, S, precision);
	*z += dz*D4;
}
