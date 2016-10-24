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
void	energyKernelXeon(const void * __restrict__ m_, const void * __restrict__ v_, double *z, const double ood2, const double LL,
			 const double nQcd, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision, void *eRes)
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

		double	Vrho, Vth, Krho, Kth, Grho, Gth;

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
		const double lZ = 0.25*LL*zR*zR;

		double tmpS[2*step] __attribute__((aligned(Align)));
#ifdef	__MIC__
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const double __attribute__((aligned(Align))) zeroAux[8] = { 0., 0., 0., 0., 0., 0., 0., 0. };
		const double __attribute__((aligned(Align))) oneAux[8]  = { 1., 1., 1., 1., 1., 1., 1., 1. };
		const double __attribute__((aligned(Align))) cjgAux[8]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
		const double __attribute__((aligned(Align))) ivZAux[8]  = { iz, 0., iz, 0., iz, 0., iz, 0. };	// Only real part
		const double __attribute__((aligned(Align))) ivZ2Aux[8] = {iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2 };
#elif	defined(__AVX__)
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) zeroAux[4] = { 0., 0., 0., 0. };
		const double __attribute__((aligned(Align))) oneAux[2]  = { 1., 1. };
		const double __attribute__((aligned(Align))) cjgAux[4]  = { 1.,-1., 1.,-1. };
		const double __attribute__((aligned(Align))) ivZAux[4]  = { iz, 0., iz, 0. };	// Only real part
		const double __attribute__((aligned(Align))) ivZ2Aux[4] = {iz2,iz2,iz2,iz2 };
#else
		const size_t XC = Lx;
		const size_t YC = Lx;

		const double __attribute__((aligned(Align))) zeroAux[2] = { 0., 0. };
		const double __attribute__((aligned(Align))) oneAux[2]  = { 1., 1. };
		const double __attribute__((aligned(Align))) cjgAux[2]  = { 1.,-1. };
		const double __attribute__((aligned(Align))) ivZAux[2]  = { iz, 0. };	// Only real part
		const double __attribute__((aligned(Align))) ivZ2Aux[2] = {iz2,iz2 };

#endif
		const _MData_ zero = opCode(load_pd, zeroAux);
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
				mMz = opCode(sub_pd, opCode(load_pd, &m[idxPz]), opCode(load_pd, &m[idxMz]));

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

				tmp1 = opCode(mul_pd, vel, mCg);
				tmp2 = opCode(mul_pd, vel, mSg);

				tmp1 = opCode(mask_add_pd,
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tmp2), _MM_PERM_BADC)),
					opCode(int2mask, 0b0000000001010101),
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tmp1), _MM_PERM_BADC)),
					tmp1);

				mdv = opCode(sub_pd, opCode(mask_add_pd, tmp1, opCode(int2mask, 0b0000000010101010), tmp1, tmp2), ivZ);
#else				// Las instrucciones se llaman igual con AVX o con SSE3
				Grx = opCode(hadd_pd, opCode(mul_pd, mMx, mCg), opCode(mul_pd, mMx, mSg));
				Gry = opCode(hadd_pd, opCode(mul_pd, mMy, mCg), opCode(mul_pd, mMy, mSg));
				Grz = opCode(hadd_pd, opCode(mul_pd, mMz, mCg), opCode(mul_pd, mMz, mSg));
				mdv = opCode(sub_pd,  opCode(hadd_pd, opCode(mul_pd, vel, mCg), opCode(mul_pd, vel, mSg)), ivZ);
#endif
				tGp = opCode(add_pd,
					tGp,
					opCode(mul_pd,
						opCode(add_pd,
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
				tVp = opCode(add_pd, tVp, opCode(mask_blend_pd, opCode(int2mask, 0b0000000010101010), mod, vel));
#elif defined(__AVX__)
				tVp = opCode(add_pd, tVp, opCode(blend_pd, mod, opCode(permute_pd, mTp, 0b00000101), 0b00001010));
#else
				tVp = opCode(add_pd, tVp, opCode(shuffle_pd, mod, mTp, 0b00000001));
#endif

#ifdef	__MIC__
				opCode(store_pd, tmpS, tGp);
				Grho = tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
				Gth  = tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];

				opCode(store_pd, tmpS, tVp);
				Vrho = tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
				Vth  = tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];

				opCode(store_pd, tmpS, tKp);
				Krho = tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
				Kth  = tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];
#elif defined(__AVX__)
				opCode(store_pd, tmpS, tGp);
				Grho = tmpS[0] + tmpS[2];
				Gth  = tmpS[1] + tmpS[3];

				opCode(store_pd, tmpS, tVp);
				Vrho = tmpS[0] + tmpS[2];
				Vth  = tmpS[1] + tmpS[3];

				opCode(store_pd, tmpS,tKp);
				Krho = tmpS[0] + tmpS[2];
				Kth  = tmpS[1] + tmpS[3];
#elif defined(__AVX__)
#else
				opCode(store_pd, tmpS, tGp);
				Grho = tmpS[0];
				Gth  = tmpS[1];
                                                           
				opCode(store_pd, tmpS, tVp);
				Vrho = tmpS[0];
				Vth  = tmpS[1];
                                                           
				opCode(store_pd, tmpS, tKp);
				Krho = tmpS[0];
				Kth  = tmpS[1];
#endif
			}
		}

		static_cast<double *> (eRes)[0] = Grho*0.375*ood2/((double) (Vf-Vo));
		static_cast<double *> (eRes)[1] = Gth *0.375*ood2/((double) (Vf-Vo));
		static_cast<double *> (eRes)[2] = Vrho*lZ/((double) (Vf-Vo));
		static_cast<double *> (eRes)[3] = Vth *zQ/((double) (Vf-Vo));
		static_cast<double *> (eRes)[4] = Krho*0.5/((double) (Vf-Vo));
		static_cast<double *> (eRes)[5] = Kth *0.5/((double) (Vf-Vo));
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

		float	Vrho, Vth, Krho, Kth, Grho, Gth;

#ifdef	USE_XEON
		const float * __restrict__ m	= (const float * __restrict__) m_;
		float * __restrict__ v		= (float * __restrict__) v_;

		__assume_aligned(m, Align);
		__assume_aligned(v, Align);
#else
		const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_, Align);
		float * __restrict__ v		= (float * __restrict__) __builtin_assume_aligned (v_, Align);
#endif

		const float zR  = *z;
		const float iz  = 1./zR;
		const float iz2 = iz*iz;
		const float zQ = 9.f*powf(zR, nQcd+2.);
		const float lZ = 0.25f*LL*zR*zR;

		float tmpS[2*step] __attribute__((aligned(Align)));
#ifdef	__MIC__
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) zeroAux[16] = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };
		const float __attribute__((aligned(Align))) oneAux[16]  = { 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
		const float __attribute__((aligned(Align))) cjgAux[16]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
		const float __attribute__((aligned(Align))) ivZAux[16]  = { iz, 0., iz, 0., iz, 0., iz, 0., iz, 0., iz, 0., iz, 0., iz, 0. };
		const float __attribute__((aligned(Align))) ivZ2Aux[16] = {iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2 };
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) zeroAux[8] = { 0., 0., 0., 0., 0., 0., 0., 0. };
		const float __attribute__((aligned(Align))) oneAux[8]  = { 1., 1., 1., 1., 1., 1., 1., 1. };
		const float __attribute__((aligned(Align))) cjgAux[8]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
		const float __attribute__((aligned(Align))) ivZAux[8]  = { iz, 0., iz, 0., iz, 0., iz, 0. };	// Only real part
		const float __attribute__((aligned(Align))) ivZ2Aux[8] = {iz2,iz2,iz2,iz2,iz2,iz2,iz2,iz2 };
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const float __attribute__((aligned(Align))) zeroAux[4] = { 0., 0., 0., 0. };
		const float __attribute__((aligned(Align))) oneAux[4]  = { 1., 1., 1., 1. };
		const float __attribute__((aligned(Align))) cjgAux[4]  = { 1.,-1., 1.,-1. };
		const float __attribute__((aligned(Align))) ivZAux[4]  = { iz, 0., iz, 0. };	// Only real part
		const float __attribute__((aligned(Align))) ivZ2Aux[4] = {iz2,iz2,iz2,iz2 };
#endif

		const _MData_ zero = opCode(load_ps, zeroAux);
		const _MData_ one  = opCode(load_ps, oneAux);
		const _MData_ cjg  = opCode(load_ps, cjgAux);
		const _MData_ ivZ  = opCode(load_ps, ivZAux);
		const _MData_ ivZ2 = opCode(load_ps, ivZ2Aux);

		#pragma omp parallel default(shared) 
		{
			_MData_ tmp1, tmp2, mel, vel, mMx, mMy, mMz, mdv, mod, mTp;
			_MData_ Grx,  Gry,  Grz, tGp, tVp, tKp, mCg, mSg;

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
				mMz = opCode(sub_ps, opCode(load_ps, &m[idxPz]), opCode(load_ps, &m[idxMz]));

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

				tmp1 = opCode(mul_ps, vel, mCg);
				tmp2 = opCode(mul_ps, vel, mSg);

				tmp1 = opCode(mask_add_ps,
					opCode(swizzle_ps, tmp2, _MM_SWIZ_REG_CDAB),
					opCode(int2mask, 0b0101010101010101),
					opCode(swizzle_ps, tmp1, _MM_SWIZ_REG_CDAB),
					tmp1);

				mdv = opCode(sub_ps, opCode(mask_add_ps, tmp1, opCode(int2mask, 0b1010101010101010), tmp1, tmp2), ivZ);
#elif defined(__AVX__)
				Grx = opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, mMx, mCg), opCode(mul_ps, mMx, mSg)), 0b11011000);
				Gry = opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, mMy, mCg), opCode(mul_ps, mMy, mSg)), 0b11011000);
				Grz = opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, mMz, mCg), opCode(mul_ps, mMz, mSg)), 0b11011000);
				mdv = opCode(sub_ps, opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, vel, mCg), opCode(mul_ps, vel, mSg)), 0b11011000), ivZ);
#else
				Grx = opCode(shuffle_ps, opCode(hadd_ps, opCode(mul_ps, mMx, mCg), opCode(mul_ps, mMx, mSg)), 0b11011000);
				Gry = opCode(shuffle_ps, opCode(hadd_ps, opCode(mul_ps, mMy, mCg), opCode(mul_ps, mMy, mSg)), 0b11011000);
				Grz = opCode(shuffle_ps, opCode(hadd_ps, opCode(mul_ps, mMz, mCg), opCode(mul_ps, mMz, mSg)), 0b11011000);
				mdv = opCode(sub_ps, opCode(shuffle_ps, opCode(hadd_ps, opCode(mul_ps, vel, mCg), opCode(mul_ps, vel, mSg)), 0b11011000), ivZ);
#endif
				tGp = opCode(add_ps,
					tGp,
					opCode(mul_ps,
						opCode(add_ps,
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
				tVp = opCode(add_ps, tVp, opCode(mask_blend_ps, opCode(int2mask, 0b1010101010101010), mod, vel));
#elif defined(__AVX__)
				tVp = opCode(add_ps, tVp, opCode(blend_ps, mod, opCode(permute_ps, mTp, 0b10110001), 0b10101010)); //REVISAR
#else
				tVp = opCode(add_ps, tVp, opCode(shuffle_ps, mod, mTp, 0b10110001)); // REVISAR
#endif

#ifdef	__MIC__
				opCode(store_ps, tmpS, tGp);
				Grho = tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14];
				Gth  = tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15];

				opCode(store_ps, tmpS, tVp);
				Vrho = tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14];
				Vth  = tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15];

				opCode(store_ps, tmpS, tKp);
				Krho = tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14];
				Kth  = tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15];
#elif defined(__AVX__)
				opCode(store_ps, tmpS, tGp);
				Grho = tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
				Gth  = tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];

				opCode(store_ps, tmpS, tVp);
				Vrho = tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
				Vth  = tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];

				opCode(store_ps, tmpS, tKp);
				Krho = tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
				Kth  = tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];
#elif defined(__AVX__)
#else
				opCode(store_ps, tmpS, tGp);
				Grho = tmpS[0] + tmpS[2];
				Gth  = tmpS[1] + tmpS[3];

				opCode(store_ps, tmpS, tVp);
				Vrho = tmpS[0] + tmpS[2];
				Vth  = tmpS[1] + tmpS[3];

				opCode(store_ps, tmpS, tKp);
				Krho = tmpS[0] + tmpS[2];
				Kth  = tmpS[1] + tmpS[3];
#endif
			}
		}

		static_cast<float *> (eRes)[0] = Grho*0.375*ood2/((float) (Vf-Vo));
		static_cast<float *> (eRes)[1] = Gth *0.375*ood2/((float) (Vf-Vo));
		static_cast<float *> (eRes)[2] = Vrho*lZ/((float) (Vf-Vo));
		static_cast<float *> (eRes)[3] = Vth *zQ/((float) (Vf-Vo));
		static_cast<float *> (eRes)[4] = Krho*0.5/((float) (Vf-Vo));
		static_cast<float *> (eRes)[5] = Kth *0.5/((float) (Vf-Vo));
#undef	_MData_
#undef	step
	}
}

void	energyXeon	(Scalar *axionField, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, void *eRes)
{
#ifdef USE_XEON
	const int  micIdx = commAcc(); 
	const double ood2 = 1./delta2;
	double *z = axionField->zV();

	int bulk  = 32;

	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX) signal(&bulk)
	{
		energyKernelXeon(mX, vX, z, ood2, LL, nQcd, Lx, S, V+S, precision, eRes);
	}
#endif
}

void	energyCpu	(Scalar *axionField, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, void *eRes)
{
	const double ood2 = 1./delta2;
	double *z = axionField->zV();

	axionField->sendGhosts(FIELD_M, COMM_SDRV);
	axionField->sendGhosts(FIELD_M, COMM_WAIT);
	energyKernelXeon(axionField->mCpu(), axionField->vCpu(), z, ood2, LL, nQcd, Lx, S, V+S, precision, eRes);
}