#include<cstdio>
#include<cmath>
#include"scalar/scalarField.h"
#include"enum-field.h"
#include"scalar/varNQCD.h"

#ifdef USE_XEON
	#include"comms/comms.h"
	#include"utils/xeonDefs.h"
#endif

#include "utils/triSimd.h"
#include "utils/parse.h"

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

#ifdef USE_XEON
__attribute__((target(mic)))
#endif
template<const VqcdType VQcd>
void	energyKernelXeon(const void * __restrict__ m_, const void * __restrict__ v_, double *z, const double ood2, const double LL, const double nQcd,
			 const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision, void * __restrict__ eRes_, const double shift)
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

		double	Vrho = 0., Vth = 0., Krho = 0., Kth = 0., Gxrho = 0., Gxth = 0., Gyrho = 0., Gyth = 0., Gzrho = 0., Gzth = 0.;

#ifdef	USE_XEON
		const double * __restrict__ m	= (const double * __restrict__) m_;
		const double * __restrict__ v	= (const double * __restrict__) v_;

		__assume_aligned(m, Align);
		__assume_aligned(v, Align);
#else
		const double * __restrict__ m	= (const double * __restrict__) __builtin_assume_aligned (m_, Align);
		const double * __restrict__ v	= (const double * __restrict__) __builtin_assume_aligned (v_, Align);
#endif
		double * __restrict__ eRes	= (double * __restrict__) eRes_;

		const double zR  = *z;
		const double iz  = 1./zR;
		const double iz2 = iz*iz;
		//const double zQ = 9.*pow(zR, nQcd+2.);
		const double zQ = axionmass2((double) zR, nQcd, zthres, zrestore)*zR*zR;
		const double lZ = 0.25*LL*zR*zR;
#ifdef	__MIC__
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const double __attribute__((aligned(Align))) cjgAux[8]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
		const double __attribute__((aligned(Align))) ivZAux[8]  = { iz, 0., iz, 0., iz, 0., iz, 0. };	// Only real part
		const double __attribute__((aligned(Align))) shfAux[8]  = {shift, 0., shift, 0., shift, 0., shift, 0. };
#elif	defined(__AVX__)
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) cjgAux[4]  = { 1.,-1., 1.,-1. };
		const double __attribute__((aligned(Align))) ivZAux[4]  = { iz, 0., iz, 0. };	// Only real part
		const double __attribute__((aligned(Align))) shfAux[4]  = {shift, 0., shift, 0. };
#else
		const size_t XC = Lx;
		const size_t YC = Lx;

		const double __attribute__((aligned(Align))) cjgAux[2]  = { 1.,-1. };
		const double __attribute__((aligned(Align))) ivZAux[2]  = { iz, 0. };	// Only real part
		const double __attribute__((aligned(Align))) shfAux[2]  = {shift, 0.};

#endif
		const _MData_ cjg  = opCode(load_pd, cjgAux);
		const _MData_ ivZ  = opCode(load_pd, ivZAux);
		const _MData_ shVc = opCode(load_pd, shfAux);

		#pragma omp parallel default(shared)
		{
			_MData_ mel, vel, mPx, mMx, mPy, mMy, mPz, mMz, mdv, mod, mTp;
			_MData_ Grx, Gry, Grz, tGx, tGy, tGz, tVp, tKp, mCg, mSg;

			double tmpS[2*step] __attribute__((aligned(Align)));

			#pragma omp for schedule(static) reduction(+:Vrho,Vth,Krho,Kth,Gxrho,Gxth,Gyrho,Gyth,Gzrho,Gzth)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxPx, idxMx, idxPy, idxMy, idxPz, idxMz, idxP0;

				{
					size_t tmi = idx/XC, tpi;

					tpi = tmi/YC;
					X[1] = tmi - tpi*YC;
					X[0] = idx - tmi*XC;
				}

				idxPz = ((idx+Sf) << 1);
				idxMz = ((idx-Sf) << 1);
				idxP0 = (idx << 1);

				mel = opCode(load_pd, &m[idxP0]); //Carga m con shift

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

					mPy = opCode(sub_pd, opCode(load_pd, &m[idxPy]), mel);
#ifdef	__MIC__
					mMy = opCode(sub_pd, mel, opCode(castps_pd, opCode(permute4f128_ps, opCode(castpd_ps, opCode(load_pd, &m[idxMy])), _MM_PERM_CBAD)));
#elif	defined(__AVX__)
					mMx = opCode(load_pd, &m[idxMy]);
					mMy = opCode(sub_pd, mel, opCode(permute2f128_pd, mMx, mMx, 0b00000001));
#else
					mMy = opCode(sub_pd, mel, opCode(load_pd, &m[idxMy]));
#endif
				}
				else
				{
					idxMy = ((idx - XC) << 1);

					mMy = opCode(sub_pd, mel, opCode(load_pd, &m[idxMy]));

					if (X[1] == YC-1)
					{
						idxPy = ((idx - Sf + XC) << 1);
#ifdef	__MIC__
						mPy = opCode(sub_pd, opCode(castps_pd, opCode(permute4f128_ps, opCode(castpd_ps, opCode(load_pd, &m[idxPy])), _MM_PERM_ADCB)), mel);
#elif	defined(__AVX__)
						mMx = opCode(load_pd, &m[idxPy]);
						mPy = opCode(sub_pd, opCode(permute2f128_pd, mMx, mMx, 0b00000001), mel);
#else
						mPy = opCode(sub_pd, opCode(load_pd, &m[idxPy]), mel);
#endif
					}
					else
					{
						idxPy = ((idx + XC) << 1);
						mPy = opCode(sub_pd, opCode(load_pd, &m[idxPy]), mel);
					}
				}

				mPx = opCode(sub_pd, opCode(load_pd, &m[idxPx]), mel);
				mMx = opCode(sub_pd, mel, opCode(load_pd, &m[idxMx]));
				// mPy y mMy ya están cargado
				mPz = opCode(sub_pd, opCode(load_pd, &m[idxPz]), mel);
				mMz = opCode(sub_pd, mel, opCode(load_pd, &m[idxMz]));

				vel = opCode(load_pd, &v[idxMz]);//Carga v
				mod = opCode(mul_pd, mel, mel);
/*
#ifdef	__MIC__
				mTp = opCode(add_pd, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, mod), _MM_PERM_BADC)), mod);
#elif defined(__AVX__)
				mTp = opCode(add_pd, opCode(permute_pd, mod, 0b00000101), mod);
#else
				mTp = opCode(add_pd, opCode(shuffle_pd, mod, mod, 0b00000001), mod);
#endif
*/
				mTp = opCode(md2_pd, mod);

				mod = opCode(mul_pd, mTp, opCode(set1_pd, iz2));	// Factor |mel|^2/z^2, util luego

				mCg = opCode(div_pd, mel, mTp);	// Ahora mCg tiene 1/mel

				// Meto en mSg = shuffled(mCg) (Intercambio parte real por parte imaginaria)
#ifdef	__MIC__
				mSg = opCode(mul_pd, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, mCg), _MM_PERM_BADC)), cjg);
#elif defined(__AVX__)
				mSg = opCode(mul_pd, opCode(permute_pd, mCg, 0b00000101), cjg);
#else
				mSg = opCode(mul_pd, opCode(shuffle_pd, mCg, mCg, 0b00000001), cjg);
#endif

				// Calculo los gradientes
#ifdef	__MIC__
				tGx = opCode(mul_pd, mPx, mCg);
				tGy = opCode(mul_pd, mPx, mSg);

				tGz = opCode(mask_add_pd,
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGy), _MM_PERM_BADC)),
					opCode(int2mask, 0b0000000001010101),
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGx), _MM_PERM_BADC)),
					tGx);

				mdv = opCode(mask_add_pd, tGz, opCode(int2mask, 0b0000000010101010), tGz, tGy);

				tGx = opCode(mul_pd, mMx, mCg);
				tGy = opCode(mul_pd, mMx, mSg);

				tGz = opCode(mask_add_pd,
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGy), _MM_PERM_BADC)),
					opCode(int2mask, 0b0000000001010101),
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGx), _MM_PERM_BADC)),
					tGx);

				Grx = opCode(mask_add_pd, tGz, opCode(int2mask, 0b0000000010101010), tGz, tGy);
				Grx = opCode(add_ps,
					opCode(mul_ps, mdv, mdv),
					opCode(mul_ps, Grx, Grx));

				tGx = opCode(mul_pd, mPy, mCg);
				tGy = opCode(mul_pd, mPy, mSg);

				tGz = opCode(mask_add_pd,
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGy), _MM_PERM_BADC)),
					opCode(int2mask, 0b0000000001010101),
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGx), _MM_PERM_BADC)),
					tGx);

				mdv = opCode(mask_add_pd, tGz, opCode(int2mask, 0b0000000010101010), tGz, tGy);

				tGx = opCode(mul_pd, mMy, mCg);
				tGy = opCode(mul_pd, mMy, mSg);

				tGz = opCode(mask_add_pd,
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGy), _MM_PERM_BADC)),
					opCode(int2mask, 0b0000000001010101),
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGx), _MM_PERM_BADC)),
					tGx);

				Gry = opCode(mask_add_pd, tGz, opCode(int2mask, 0b0000000010101010), tGz, tGy);
				Gry = opCode(add_ps,
					opCode(mul_ps, mdv, mdv),
					opCode(mul_ps, Gry, Gry));

				tGx = opCode(mul_pd, mPz, mCg);
				tGy = opCode(mul_pd, mPz, mSg);

				tGz = opCode(mask_add_pd,
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGy), _MM_PERM_BADC)),
					opCode(int2mask, 0b0000000001010101),
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGx), _MM_PERM_BADC)),
					tGx);

				mdv = opCode(mask_add_pd, tGz, opCode(int2mask, 0b0000000010101010), tGz, tGy);

				tGx = opCode(mul_pd, mMz, mCg);
				tGy = opCode(mul_pd, mMz, mSg);

				tGz = opCode(mask_add_pd,
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGy), _MM_PERM_BADC)),
					opCode(int2mask, 0b0000000001010101),
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGx), _MM_PERM_BADC)),
					tGx);

				Grz = opCode(mask_add_pd, tGz, opCode(int2mask, 0b0000000010101010), tGz, tGy);
				Grz = opCode(add_ps,
					opCode(mul_ps, mdv, mdv),
					opCode(mul_ps, Grz, Grz));

				tGx = opCode(mul_pd, vel, mCg);
				tGy = opCode(mul_pd, vel, mSg);

				tGz = opCode(mask_add_pd,
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGy), _MM_PERM_BADC)),
					opCode(int2mask, 0b0000000001010101),
					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGx), _MM_PERM_BADC)),
					tGx);

				mdv = opCode(sub_pd, opCode(mask_add_pd, tGz, opCode(int2mask, 0b0000000010101010), tGz, tGy), ivZ);
#else				// Las instrucciones se llaman igual con AVX o con SSE3
				Grx = opCode(hadd_pd, opCode(mul_pd, mPx, mCg), opCode(mul_pd, mPx, mSg));
				mdv = opCode(hadd_pd, opCode(mul_pd, mMx, mCg), opCode(mul_pd, mMx, mSg));
				Grx = opCode(add_pd,
					opCode(mul_pd, mdv, mdv),
					opCode(mul_pd, Grx, Grx));

				Gry = opCode(hadd_pd, opCode(mul_pd, mPy, mCg), opCode(mul_pd, mPy, mSg));
				mdv = opCode(hadd_pd, opCode(mul_pd, mMy, mCg), opCode(mul_pd, mMy, mSg));
				Gry = opCode(add_pd,
					opCode(mul_pd, mdv, mdv),
					opCode(mul_pd, Gry, Gry));

				Grz = opCode(hadd_pd, opCode(mul_pd, mPz, mCg), opCode(mul_pd, mPz, mSg));
				mdv = opCode(hadd_pd, opCode(mul_pd, mMz, mCg), opCode(mul_pd, mMz, mSg));
				Grz = opCode(add_pd,
					opCode(mul_pd, mdv, mdv),
					opCode(mul_pd, Grz, Grz));

				mdv = opCode(sub_pd, opCode(hadd_pd, opCode(mul_pd, vel, mCg), opCode(mul_pd, vel, mSg)), ivZ);
#endif
				tGx = opCode(mul_pd, mod, Grx);
				tGy = opCode(mul_pd, mod, Gry);
				tGz = opCode(mul_pd, mod, Grz);

				tKp = opCode(mul_pd, mod, opCode(mul_pd, mdv, mdv));

				Grx = opCode(sub_pd, mel, shVc);
				Gry = opCode(mul_pd, Grx, Grx);
				Grz = opCode(md2_pd, Gry);
				Gry = opCode(mul_pd, Grz, opCode(set1_pd, iz2));

				mSg = opCode(sub_pd, Gry, opCode(set1_pd, 1.0));
				mod = opCode(mul_pd, mSg, mSg);

				switch	(VQcd) {
					case	VQCD_1:
						mCg = opCode(sub_pd, opCode(set1_pd, 1.0), opCode(div_pd, Grx, opCode(sqrt_pd, Grz)));  // 1-m/|m|
						break;

					case	VQCD_2:
						mTp = opCode(sub_pd, opCode(set1_pd, 1.0), opCode(mul_pd, Grx, ivZ));
						mCg = opCode(mul_pd, mTp, mTp);
						break;
				}
#ifdef	__MIC__
				vel = opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, mCg), _MM_PERM_BADC));
				tVp = opCode(mask_blend_pd, opCode(int2mask, 0b0000000010101010), mod, vel);
#elif defined(__AVX__)
				tVp = opCode(blend_pd, mod, opCode(permute_pd, mCg, 0b00000101), 0b00001010);
#else
				tVp = opCode(shuffle_pd, mod, mCg, 0b00000001);
#endif

#ifdef	__MIC__
				opCode(store_pd, tmpS, tGx);
				Gxrho += tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
				Gxth  += tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];

				opCode(store_pd, tmpS, tGy);
				Gyrho += tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
				Gyth  += tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];

				opCode(store_pd, tmpS, tGz);
				Gzrho += tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
				Gzth  += tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];

				opCode(store_pd, tmpS, tVp);
				Vrho += tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
				Vth  += tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];

				opCode(store_pd, tmpS, tKp);
				Krho += tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
				Kth  += tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];
#elif defined(__AVX__)
				opCode(store_pd, tmpS, tGx);
				Gxrho += tmpS[0] + tmpS[2];
				Gxth  += tmpS[1] + tmpS[3];

				opCode(store_pd, tmpS, tGy);
				Gyrho += tmpS[0] + tmpS[2];
				Gyth  += tmpS[1] + tmpS[3];

				opCode(store_pd, tmpS, tGz);
				Gzrho += tmpS[0] + tmpS[2];
				Gzth  += tmpS[1] + tmpS[3];

				opCode(store_pd, tmpS, tVp);
				Vrho += tmpS[0] + tmpS[2];
				Vth  += tmpS[1] + tmpS[3];

				opCode(store_pd, tmpS, tKp);
				Krho += tmpS[0] + tmpS[2];
				Kth  += tmpS[1] + tmpS[3];
#else
				opCode(store_pd, tmpS, tGx);
				Gxrho += tmpS[0];
				Gxth  += tmpS[1];

				opCode(store_pd, tmpS, tGy);
				Gyrho += tmpS[0];
				Gyth  += tmpS[1];

				opCode(store_pd, tmpS, tGz);
				Gzrho += tmpS[0];
				Gzth  += tmpS[1];

				opCode(store_pd, tmpS, tVp);
				Vrho += tmpS[0];
				Vth  += tmpS[1];

				opCode(store_pd, tmpS, tKp);
				Krho += tmpS[0];
				Kth  += tmpS[1];
#endif
			}
		}

		const double o2 = ood2*0.25;

		eRes[RH_GRX] = Gxrho*o2;
		eRes[TH_GRX] = Gxth *o2;
		eRes[RH_GRY] = Gyrho*o2;
		eRes[TH_GRY] = Gyth *o2;
		eRes[RH_GRZ] = Gzrho*o2;
		eRes[TH_GRZ] = Gzth *o2;
		eRes[RH_POT] = Vrho *lZ;
		eRes[TH_POT] = Vth  *zQ;
		eRes[RH_KIN] = Krho *.5;
		eRes[TH_KIN] = Kth  *.5;
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
		double	Vrho = 0., Vth = 0., Krho = 0., Kth = 0., Gxrho = 0., Gxth = 0., Gyrho = 0., Gyth = 0., Gzrho = 0., Gzth = 0.;

#ifdef	USE_XEON
		const float * __restrict__ m	= (const float * __restrict__) m_;
		const float * __restrict__ v	= (const float * __restrict__) v_;

		__assume_aligned(m, Align);
		__assume_aligned(v, Align);
#else
		const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_, Align);
		const float * __restrict__ v	= (const float * __restrict__) __builtin_assume_aligned (v_, Align);
#endif
		double * __restrict__ eRes	= (double * __restrict__) eRes_;

		const float zR  = *z;
		const float iz  = 1./zR;
		const float iz2 = iz*iz;
		//const float zQ = 9.f*powf(zR, nQcd+2.);
		const float zQ = axionmass2((float) zR, nQcd, zthres, zrestore)*zR*zR;
		const float lZ = 0.25f*LL*zR*zR;
#ifdef	__MIC__
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) cjgAux[16]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
		const float __attribute__((aligned(Align))) ivZAux[16]  = { iz, 0., iz, 0., iz, 0., iz, 0., iz, 0., iz, 0., iz, 0., iz, 0. };
		const float __attribute__((aligned(Align))) shfAux[16]  = {shift, 0., shift, 0., shift, 0., shift, 0., shift, 0., shift, 0., shift, 0., shift, 0. };
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) cjgAux[8]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
		const float __attribute__((aligned(Align))) ivZAux[8]  = { iz, 0., iz, 0., iz, 0., iz, 0. };	// Only real part
		const float __attribute__((aligned(Align))) shfAux[8]  = {shift, 0., shift, 0., shift, 0., shift, 0.};
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const float __attribute__((aligned(Align))) cjgAux[4]  = { 1.,-1., 1.,-1. };
		const float __attribute__((aligned(Align))) ivZAux[4]  = { iz, 0., iz, 0. };	// Only real part
		const float __attribute__((aligned(Align))) shfAux[4]  = {shift, 0., shift, 0. };
#endif

		const _MData_ cjg  = opCode(load_ps, cjgAux);
		const _MData_ ivZ  = opCode(load_ps, ivZAux);
		const _MData_ shVc = opCode(load_ps, shfAux);

		#pragma omp parallel default(shared)
		{
			_MData_ mel, vel, mPx, mMx, mPy, mMy, mPz, mMz, mdv, mod, mTp;
			_MData_ Grx, Gry, Grz, tGx, tGy, tGz, tVp, tKp, mCg, mSg;

			float tmpS[2*step] __attribute__((aligned(Align)));

			#pragma omp for schedule(static) reduction(+:Vrho,Vth,Krho,Kth,Gxrho,Gxth,Gyrho,Gyth,Gzrho,Gzth)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0;

				idxPz = ((idx+Sf) << 1);
				idxMz = ((idx-Sf) << 1);
				idxP0 =  (idx     << 1);

				mel = opCode(load_ps, &m[idxP0]);

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

					mPy = opCode(sub_ps, opCode(load_ps, &m[idxPy]), mel);
#ifdef	__MIC__
					mMx = opCode(swizzle_ps, opCode(load_ps, &m[idxMy]), _MM_SWIZ_REG_BADC);
					mMz = opCode(permute4f128_ps, mMx, _MM_PERM_CBAD);
					mMy = opCode(sub_ps, mel, opCode(mask_blend_ps, opCode(int2mask, 0b0011001100110011), mMx, mMz));
#elif	defined(__AVX2__)
					mMy = opCode(sub_ps, mel, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 6,7,0,1,2,3,4,5)));
#elif	defined(__AVX__)
					mMx = opCode(permute_ps, opCode(load_ps, &m[idxMy]), 0b01001110);
					mMz = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
					mMy = opCode(sub_ps, mel, opCode(blend_ps, mMx, mMz, 0b00110011));
#else
					mMx = opCode(load_ps, &m[idxMy]);
					mMy = opCode(sub_ps, mel, opCode(shuffle_ps, mMx, mMx, 0b01001110));
#endif
				}
				else
				{
					idxMy = ((idx - XC) << 1);

					if (X[1] == YC-1)
					{
						idxPy = ((idx - Sf + XC) << 1);

						mMy = opCode(sub_ps, mel, opCode(load_ps, &m[idxMy]));
#ifdef	__MIC__
						mMx = opCode(swizzle_ps, opCode(load_ps, &m[idxPy]), _MM_SWIZ_REG_BADC);
						mMz = opCode(permute4f128_ps, mMx, _MM_PERM_ADCB);
						mPy = opCode(sub_ps, opCode(mask_blend_ps, opCode(int2mask, 0b1100110011001100), mMx, mMz), mel);
#elif	defined(__AVX2__)
						mPy = opCode(sub_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 2,3,4,5,6,7,0,1)), mel);
#elif	defined(__AVX__)
						mMx = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b01001110);
						mMz = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
						mPy = opCode(sub_ps, opCode(blend_ps, mMx, mMz, 0b11001100), mel);
#else
						mMx = opCode(load_ps, &m[idxPy]);
						mPy = opCode(sub_ps, opCode(shuffle_ps, mMx, mMx, 0b01001110), mel);
#endif
					}
					else
					{
						idxPy = ((idx + XC) << 1);
						mPy = opCode(sub_ps, opCode(load_ps, &m[idxPy]), mel);
						mMy = opCode(sub_ps, mel, opCode(load_ps, &m[idxMy]));
					}
				}

				// Empiezo aqui
				mPx = opCode(sub_ps, opCode(load_ps, &m[idxPx]), mel);
				mMx = opCode(sub_ps, mel, opCode(load_ps, &m[idxMx]));
				// mPy, mMy ya están cargados
				mPz = opCode(sub_ps, opCode(load_ps, &m[idxPz]), mel);
				mMz = opCode(sub_ps, mel, opCode(load_ps, &m[idxMz]));

				vel = opCode(load_ps, &v[idxMz]); // Carga v
				mod = opCode(mul_ps, mel, mel);
				mTp = opCode(md2_ps, mod);
				mod = opCode(mul_ps, mTp, opCode(set1_ps, iz2));	// Factor |mel|^2/z^2, util luego

				mCg = opCode(div_ps, mel, mTp);	// Ahora mCg tiene 1/mel

				// Meto en mSg = shuffled(mCg) (Intercambio parte real por parte imaginaria)
#ifdef	__MIC__
				mSg = opCode(mul_ps, cjg, opCode(swizzle_ps, mCg, _MM_SWIZ_REG_CDAB));
#elif defined(__AVX__)
				mSg = opCode(mul_ps, cjg, opCode(permute_ps, mCg, 0b10110001));
#else
				mSg = opCode(mul_ps, cjg, opCode(shuffle_ps, mCg, mCg, 0b10110001));
#endif

				// Calculo los gradientes
#ifdef	__MIC__
				tGx = opCode(mul_ps, mPx, mCg);
				tGy = opCode(mul_ps, mPx, mSg);

				tGz = opCode(mask_add_ps,
					opCode(swizzle_ps, tGy, _MM_SWIZ_REG_CDAB),
					opCode(int2mask, 0b0101010101010101),
					opCode(swizzle_ps, tGx, _MM_SWIZ_REG_CDAB),
					tGx);

				mdv = opCode(mask_add_ps, tGz, opCode(int2mask, 0b1010101010101010), tGz, tGy);

				tGx = opCode(mul_ps, mMx, mCg);
				tGy = opCode(mul_ps, mMx, mSg);

				tGz = opCode(mask_add_ps,
					opCode(swizzle_ps, tGy, _MM_SWIZ_REG_CDAB),
					opCode(int2mask, 0b0101010101010101),
					opCode(swizzle_ps, tGx, _MM_SWIZ_REG_CDAB),
					tGx);

				Grx = opCode(mask_add_ps, tGz, opCode(int2mask, 0b1010101010101010), tGz, tGy);
				Grx = opCode(add_ps,
					opCode(mul_ps, mdv, mdv),
					opCode(mul_ps, Grx, Grx));

				tGx = opCode(mul_ps, mPy, mCg);
				tGy = opCode(mul_ps, mPy, mSg);

				tGz = opCode(mask_add_ps,
					opCode(swizzle_ps, tGy, _MM_SWIZ_REG_CDAB),
					opCode(int2mask, 0b0101010101010101),
					opCode(swizzle_ps, tGx, _MM_SWIZ_REG_CDAB),
					tGx);

				mdv = opCode(mask_add_ps, tGz, opCode(int2mask, 0b1010101010101010), tGz, tGy);

				tGx = opCode(mul_ps, mMy, mCg);
				tGy = opCode(mul_ps, mMy, mSg);

				tGz = opCode(mask_add_ps,
					opCode(swizzle_ps, tGy, _MM_SWIZ_REG_CDAB),
					opCode(int2mask, 0b0101010101010101),
					opCode(swizzle_ps, tGx, _MM_SWIZ_REG_CDAB),
					tGx);

				Gry = opCode(mask_add_ps, tGz, opCode(int2mask, 0b1010101010101010), tGz, tGy);
				Gry = opCode(add_ps,
					opCode(mul_ps, mdv, mdv),
					opCode(mul_ps, Gry, Gry));

				tGx = opCode(mul_ps, mPz, mCg);
				tGy = opCode(mul_ps, mPz, mSg);

				tGz = opCode(mask_add_ps,
					opCode(swizzle_ps, tGy, _MM_SWIZ_REG_CDAB),
					opCode(int2mask, 0b0101010101010101),
					opCode(swizzle_ps, tGx, _MM_SWIZ_REG_CDAB),
					tGx);

				mdv = opCode(mask_add_ps, tGz, opCode(int2mask, 0b1010101010101010), tGz, tGy);

				tGx = opCode(mul_ps, mMz, mCg);
				tGy = opCode(mul_ps, mMz, mSg);

				tGz = opCode(mask_add_ps,
					opCode(swizzle_ps, tGy, _MM_SWIZ_REG_CDAB),
					opCode(int2mask, 0b0101010101010101),
					opCode(swizzle_ps, tGx, _MM_SWIZ_REG_CDAB),
					tGx);

				Grz = opCode(mask_add_ps, tGz, opCode(int2mask, 0b1010101010101010), tGz, tGy);
				Grz = opCode(add_ps,
					opCode(mul_ps, mdv, mdv),
					opCode(mul_ps, Grz, Grz));

				tGx = opCode(mul_ps, vel, mCg);
				tGy = opCode(mul_ps, vel, mSg);

				tGz = opCode(mask_add_ps,
					opCode(swizzle_ps, tGy, _MM_SWIZ_REG_CDAB),
					opCode(int2mask, 0b0101010101010101),
					opCode(swizzle_ps, tGx, _MM_SWIZ_REG_CDAB),
					tGx);

				mdv = opCode(sub_ps, opCode(mask_add_ps, tGz, opCode(int2mask, 0b1010101010101010), tGz, tGy), ivZ);
#elif defined(__AVX__)
				Grx = opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, mPx, mCg), opCode(mul_ps, mPx, mSg)), 0b11011000);
				mdv = opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, mMx, mCg), opCode(mul_ps, mMx, mSg)), 0b11011000);
				Grx = opCode(add_ps,
					opCode(mul_ps, mdv, mdv),
					opCode(mul_ps, Grx, Grx));

				Gry = opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, mPy, mCg), opCode(mul_ps, mPy, mSg)), 0b11011000);
				mdv = opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, mMy, mCg), opCode(mul_ps, mMy, mSg)), 0b11011000);
				Gry = opCode(add_ps,
					opCode(mul_ps, mdv, mdv),
					opCode(mul_ps, Gry, Gry));

				Grz = opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, mPz, mCg), opCode(mul_ps, mPz, mSg)), 0b11011000);
				mdv = opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, mMz, mCg), opCode(mul_ps, mMz, mSg)), 0b11011000);
				Grz = opCode(add_ps,
					opCode(mul_ps, mdv, mdv),
					opCode(mul_ps, Grz, Grz));

				mdv = opCode(sub_ps, opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, vel, mCg), opCode(mul_ps, vel, mSg)), 0b11011000), ivZ);
#else
				tKp = opCode(hadd_ps, opCode(mul_ps, vel, mCg), opCode(mul_ps, vel, mSg));

				tGx = opCode(hadd_ps, opCode(mul_ps, mPx, mCg), opCode(mul_ps, mPx, mSg));
				mdv = opCode(hadd_ps, opCode(mul_ps, mMx, mCg), opCode(mul_ps, mMx, mSg));
				tVp = opCode(shuffle_ps, tGx, tGx, 0b11011000);
				vel = opCode(shuffle_ps, mdv, mdv, 0b11011000);
				Grx = opCode(add_ps,
					opCode(mul_ps, vel, vel),
					opCode(mul_ps, tVp, tVp));

				tGy = opCode(hadd_ps, opCode(mul_ps, mPy, mCg), opCode(mul_ps, mPy, mSg));
				mdv = opCode(hadd_ps, opCode(mul_ps, mMy, mCg), opCode(mul_ps, mMy, mSg));
				tVp = opCode(shuffle_ps, tGy, tGy, 0b11011000);
				vel = opCode(shuffle_ps, mdv, mdv, 0b11011000);
				Gry = opCode(add_ps,
					opCode(mul_ps, vel, vel),
					opCode(mul_ps, tVp, tVp));

				tGz = opCode(hadd_ps, opCode(mul_ps, mPz, mCg), opCode(mul_ps, mPz, mSg));
				mdv = opCode(hadd_ps, opCode(mul_ps, mMz, mCg), opCode(mul_ps, mMz, mSg));
				tVp = opCode(shuffle_ps, tGz, tGz, 0b11011000);
				vel = opCode(shuffle_ps, mdv, mdv, 0b11011000);
				Grz = opCode(add_ps,
					opCode(mul_ps, vel, vel),
					opCode(mul_ps, tVp, tVp));

				mdv = opCode(sub_ps, opCode(shuffle_ps, tKp, tKp, 0b11011000), ivZ);
#endif
				tGx = opCode(mul_ps, mod, Grx);
				tGy = opCode(mul_ps, mod, Gry);
				tGz = opCode(mul_ps, mod, Grz);

				tKp = opCode(mul_ps, opCode(mul_ps, mdv, mdv), mod);

				Grx = opCode(sub_ps, mel, shVc);  // Aplica shift
				Gry = opCode(mul_ps, Grx, Grx);
				Grz = opCode(md2_ps, Gry);
				Gry = opCode(mul_ps, Grz, opCode(set1_ps, iz2));

				mSg = opCode(sub_ps, Gry, opCode(set1_ps, 1.0f));
				mod = opCode(mul_ps, mSg, mSg);

				switch	(VQcd) {
					case	VQCD_1:
						mCg = opCode(sub_ps, opCode(set1_ps, 1.0f), opCode(div_ps, Grx, opCode(sqrt_ps, Grz)));  // 1-m/|m|
						break;

					case	VQCD_2:
						mTp = opCode(sub_ps, opCode(set1_ps, 1.0f), opCode(mul_ps, Grx, ivZ));
						mCg = opCode(mul_ps, mTp, mTp);
						break;
				}
#ifdef	__MIC__
				tVp = opCode(mask_blend_ps, opCode(int2mask, 0b1010101010101010), mod, opCode(swizzle_ps, mCg, _MM_SWIZ_REG_CDAB));
#elif defined(__AVX__)
				tVp = opCode(blend_ps, mod, opCode(permute_ps, mCg, 0b10110001), 0b10101010);
#else
				mdv = opCode(shuffle_ps, mod, mCg, 0b10001000); //Era 11011000
				tVp = opCode(shuffle_ps, mdv, mdv, 0b11011000);
#endif

#ifdef	__MIC__
				opCode(store_ps, tmpS, tGx);
				Gxrho += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
				Gxth  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15]);

				opCode(store_ps, tmpS, tGy);
				Gyrho += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
				Gyth  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15]);

				opCode(store_ps, tmpS, tGz);
				Gzrho += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
				Gzth  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15]);

				opCode(store_ps, tmpS, tVp);
				Vrho  += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
				Vth   += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15]);

				opCode(store_ps, tmpS, tKp);
				Krho  += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
				Kth   += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15]);
#elif defined(__AVX__)
				opCode(store_ps, tmpS, tGx);
				Gxrho += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
				Gxth  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7]);

				opCode(store_ps, tmpS, tGy);
				Gyrho += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
				Gyth  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7]);

				opCode(store_ps, tmpS, tGz);
				Gzrho += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
				Gzth  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7]);

				opCode(store_ps, tmpS, tVp);
				Vrho  += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
				Vth   += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7]);

				opCode(store_ps, tmpS, tKp);
				Krho  += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
				Kth   += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7]);
#else
/*
				opCode(store_ps, tmpS, tGp);
				Grho += tmpS[0] + tmpS[2];
				Gth  += tmpS[1] + tmpS[3];
*/
				opCode(store_ps, tmpS, tGx);
				Gxrho += (double) (tmpS[0] + tmpS[2]);
				Gxth  += (double) (tmpS[1] + tmpS[3]);

				opCode(store_ps, tmpS, tGy);
				Gyrho += (double) (tmpS[0] + tmpS[2]);
				Gyth  += (double) (tmpS[1] + tmpS[3]);

				opCode(store_ps, tmpS, tGz);
				Gzrho += (double) (tmpS[0] + tmpS[2]);
				Gzth  += (double) (tmpS[1] + tmpS[3]);

				opCode(store_ps, tmpS, tVp);
				Vrho += (double) (tmpS[0] + tmpS[2]);
				Vth  += (double) (tmpS[1] + tmpS[3]);

				opCode(store_ps, tmpS, tKp);
				Krho += (double) (tmpS[0] + tmpS[2]);
				Kth  += (double) (tmpS[1] + tmpS[3]);

#endif
			}
		}

		const double o2 = ood2*0.25;

		eRes[RH_GRX] = Gxrho*o2;
		eRes[TH_GRX] = Gxth *o2;
		eRes[RH_GRY] = Gyrho*o2;
		eRes[TH_GRY] = Gyth *o2;
		eRes[RH_GRZ] = Gzrho*o2;
		eRes[TH_GRZ] = Gzth *o2;
		eRes[RH_POT] = Vrho *lZ;
		eRes[TH_POT] = Vth  *zQ;
		eRes[RH_KIN] = Krho *.5;
		eRes[TH_KIN] = Kth  *.5;
#undef	_MData_
#undef	step
	}
}

void	energyXeon	(Scalar *axionField, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, void *eRes, const double shift, const VqcdType VQcd)
{
#ifdef USE_XEON
	const int  micIdx = commAcc();
	const double ood2 = 1./delta2;
	double *z  = axionField->zV();
	double *eR = static_cast<double*>(eRes);

	axionField->exchangeGhosts(FIELD_M);

	switch (VQcd) {
		case	VQCD_1:
			#pragma offload target(mic:micIdx) in(z:length(8),shift UseX) out(eR:length(16) UseX) nocopy(mX, vX, m2X : ReUseX)
			{
				energyKernelXeon<VQCD_1>(mX, vX, z, ood2, LL, nQcd, Lx, S, V+S, precision, (void*) eR, shift);
			}
			break;

		case	VQCD_2:
			#pragma offload target(mic:micIdx) in(z:length(8),shift UseX) out(eR:length(16) UseX) nocopy(mX, vX, m2X : ReUseX)
			{
				energyKernelXeon<VQCD_2>(mX, vX, z, ood2, LL, nQcd, Lx, S, V+S, precision, (void*) eR, shift);
			}
			break;
	}
#endif
}

void	energyCpu	(Scalar *axionField, const double delta2, const double LL, const double nQcd, const size_t Lx, const size_t V, const size_t S, FieldPrecision precision, void *eRes, const double shift, const VqcdType VQcd)
{
	const double ood2 = 1./delta2;
	double *z = axionField->zV();

	axionField->exchangeGhosts(FIELD_M);

	switch (VQcd) {
		case	VQCD_1:
			energyKernelXeon<VQCD_1>(axionField->mCpu(), axionField->vCpu(), z, ood2, LL, nQcd, Lx, S, V+S, precision, eRes, shift);
			break;

		case	VQCD_2:
			energyKernelXeon<VQCD_2>(axionField->mCpu(), axionField->vCpu(), z, ood2, LL, nQcd, Lx, S, V+S, precision, eRes, shift);
			break;
	}
}
