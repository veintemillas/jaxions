#include<cstdio>
#include<cmath>
#include "scalar/scalarField.h"
#include "enum-field.h"
//#include "scalar/varNQCD.h"

#include"utils/triSimd.h"
#include"utils/parse.h"

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

template<const bool map, const bool wMod>
void	energyThetaKernelXeon(const void * __restrict__ m_, const void * __restrict__ v_, void * __restrict__ m2_, double *R, const double ood2, const double aMass2,
			 const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision, void * __restrict__ eRes_)
{
	const size_t Sf = Lx*Lx;

	double * __restrict__ eRes = (double * __restrict__) eRes_;
	double gxC = 0., gyC = 0., gzC = 0., ktC = 0., ptC = 0.;

	if (precision == FIELD_DOUBLE)
	{
#if	defined(__AVX512F__)
	#define	_MData_ __m512d
	#define	step 8
#elif	defined(__AVX__)
	#define	_MData_ __m256d
	#define	step 4
#else
	#define	_MData_ __m128d
	#define	step 2
#endif

		const double * __restrict__ m	= (const double * __restrict__) __builtin_assume_aligned (m_, Align);
		const double * __restrict__ v	= (const double * __restrict__) __builtin_assume_aligned (v_, Align);
		double * __restrict__ m2	= (double * __restrict__) __builtin_assume_aligned (m2_,Align);

		const double zR  = *R;
		const double iz  = 1./zR;
		const double iz2 = iz*iz;
		const double zQ  = aMass2*zR*zR;
		const double o2  = ood2*iz2;
		const double tV  = 2.*M_PI*zR;
#if	defined(__AVX512F__)
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

#if	defined(__AVX512F__)
		const auto vShRg  = opCode(load_si512, shfRg);
		const auto vShLf  = opCode(load_si512, shfLf);
#endif
		const _MData_ hlf   = opCode(set1_pd, 0.5);
//		const _MData_ one   = opCode(set1_pd, 1.0);
		const _MData_ two   = opCode(set1_pd, 2.0);
		const _MData_ tpVec = opCode(set1_pd, tV);
		const _MData_ izVec = opCode(set1_pd, iz);

		#pragma omp parallel default(shared) reduction(+:gxC,gyC,gzC,ktC,ptC)
		{
			_MData_ mel, vel, mMx, mMy, mMz, mPx, mPy, mPz, tmp, grd;

			double tmpGx[step] __attribute__((aligned(Align)));
			double tmpGy[step] __attribute__((aligned(Align)));
			double tmpGz[step] __attribute__((aligned(Align)));
			double tmpV [step] __attribute__((aligned(Align)));
			double tmpK [step] __attribute__((aligned(Align)));

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[3], idxPx, idxMx, idxPy, idxMy, idxPz, idxMz, idxP0;

				{
					size_t tmi = idx/XC;

					X[2] = tmi/YC;
					X[1] = tmi - X[2]*YC;
					X[0] = idx - tmi*XC;
					X[2]--;	// Removes ghosts
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
#if	defined(__AVX512F__)
					mMy = opCode(add_pd, opCode(permutexvar_pd, vShRg, opCode(load_pd, &m[idxMy])), mPy);
#elif	defined(__AVX2__)       //AVX2
					mMy = opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_pd, &m[idxMy])), opCode(setr_epi32, 6,7,0,1,2,3,4,5)));
#elif	defined(__AVX__)
					mel = opCode(permute_pd, opCode(load_pd, &m[idxMy]), 0b00000101);
					vel = opCode(permute2f128_pd, mel, mel, 0b00000001);
					mMy = opCode(blend_pd, mel, vel, 0b00000101);
#else
					mel = opCode(load_pd, &m[idxMy]);
					mMy = opCode(shuffle_pd, mel, mel, 0x00000001);
#endif
				}
				else
				{
					idxMy = idx - XC;
					mMy = opCode(load_pd, &m[idxMy]);

					if (X[1] == YC-1)
					{
						idxPy = idx - Sf + XC;
#if	defined(__AVX512F__)
						mPy = opCode(add_pd, opCode(permutexvar_pd, vShLf, opCode(load_pd, &m[idxPy])), mMy);
#elif	defined(__AVX2__)       //AVX2
						mPy = opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_pd, &m[idxPy])), opCode(setr_epi32, 2,3,4,5,6,7,0,1)));
#elif	defined(__AVX__)
						mel = opCode(permute_pd, opCode(load_pd, &m[idxPy]), 0b00000101);
						vel = opCode(permute2f128_pd, mel, mel, 0b00000001);
						mPy = opCode(blend_pd, mel, vel, 0b00001010);
#else
						vel = opCode(load_pd, &m[idxPy]);
						mPy = opCode(shuffle_pd, vel, vel, 0b00000001);
#endif
					}
					else
					{
						idxPy = idx + XC;
						mPy = opCode(load_pd, &m[idxPy]);
					}
				}

				// Tienes mMy y los puntos para mMx y mMz. Calcula todo ya!!!

				idxPz = idx+Sf;
				idxMz = idx-Sf;
				idxP0 = idx;

				mel = opCode(load_pd, &m[idxP0]); // Carga m
				vel = opCode(load_pd, &v[idxMz]); // Carga v

				// Calculo los gradientes con módulo

				grd = opCode(sub_pd, opCode(load_pd, &m[idxPx]), mel);
				if (wMod) {
					tmp = opCode(mod_pd, grd, tpVec);
					mPx = opCode(mul_pd, tmp, tmp);
				} else
					mPx = opCode(mul_pd, grd, grd);

				grd = opCode(sub_pd, opCode(load_pd, &m[idxMx]), mel);
				if (wMod) {
					tmp = opCode(mod_pd, grd, tpVec);
					mMx = opCode(mul_pd, tmp, tmp);
				} else
				mMx = opCode(mul_pd, grd, grd);

				grd = opCode(sub_pd, mPy, mel);
				if (wMod) {
					tmp = opCode(mod_pd, grd, tpVec);
					mPy = opCode(mul_pd, tmp, tmp);
				} else
				mPy = opCode(mul_pd, grd, grd);

				grd = opCode(sub_pd, mMy, mel);
				if (wMod) {
					tmp = opCode(mod_pd, grd, tpVec);
					mMy = opCode(mul_pd, tmp, tmp);
				} else
				mMy = opCode(mul_pd, grd, grd);

				grd = opCode(sub_pd, opCode(load_pd, &m[idxPz]), mel);
				if (wMod) {
					tmp = opCode(mod_pd, grd, tpVec);
					mPz = opCode(mul_pd, tmp, tmp);
				} else
				mPz = opCode(mul_pd, grd, grd);

				grd = opCode(sub_pd, opCode(load_pd, &m[idxMz]), mel);
				if (wMod) {
					tmp = opCode(mod_pd, grd, tpVec);
					mMz = opCode(mul_pd, tmp, tmp);
				} else
				mMz = opCode(mul_pd, grd, grd);

				grd = opCode(add_pd, mPx, mMx);
				mMx = opCode(add_pd, mPy, mMy);
				mMy = opCode(add_pd, mPz, mMz);

				mPz = opCode(sub_pd, vel, opCode(mul_pd, mel, izVec));
				mPx = opCode(mul_pd, mPz, mPz);

				tmp = opCode(sin_pd, opCode(mul_pd, hlf, opCode(mul_pd, mel, izVec)));
				mPy = opCode(mul_pd, opCode(mul_pd, tmp, tmp), two);

				opCode(store_pd, tmpGx, grd);
				opCode(store_pd, tmpGy, mMx);
				opCode(store_pd, tmpGz, mMy);
				opCode(store_pd, tmpK,  mPx);
				opCode(store_pd, tmpV,  mPy);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					ptC += tmpV[ih];
					ktC += tmpK[ih];
					gxC += tmpGx[ih];
					gyC += tmpGy[ih];
					gzC += tmpGz[ih];

					if	(map == true) {
						unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
						m2[iNx] = (tmpGx[ih] + tmpGy[ih] + tmpGz[ih])*o2 + tmpK[ih]*iz2*0.5 + tmpV[ih]*zQ;
					}
				}
			}
		}

		gxC *= o2; gyC *= o2; gzC *= o2; ktC *= iz2*0.5; ptC *= zQ;

#undef	_MData_
#undef	step
	}
	else if (precision == FIELD_SINGLE)
	{
#if	defined(__AVX512F__)
	#define	_MData_ __m512
	#define	step 16
#elif	defined(__AVX__)
	#define	_MData_ __m256
	#define	step 8
#else
	#define	_MData_ __m128
	#define	step 4
#endif

		const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_, Align);
		const float * __restrict__ v	= (const float * __restrict__) __builtin_assume_aligned (v_, Align);
		float * __restrict__ m2		= (float * __restrict__) __builtin_assume_aligned (m2_,Align);

		const float zR  = *R;
		const float iz  = 1./zR;
		const float iz2 = iz*iz;
		const float zQ = aMass2*zR*zR;
		const float o2 = ood2*iz2;
		const float tV = 2.f*M_PI*zR;
#if	defined(__AVX512F__)
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

		const _MData_ hlf   = opCode(set1_ps, .5f);
//		const _MData_ one   = opCode(set1_ps, 1.f);
		const _MData_ two   = opCode(set1_ps, 2.f);
		const _MData_ izVec = opCode(set1_ps, iz);
		const _MData_ tpVec = opCode(set1_ps, tV);

		#pragma omp parallel default(shared) reduction(+:gxC,gyC,gzC,ktC,ptC)
		{
			_MData_ mel, vel, grd, tmp, mMx, mMy, mMz, mPx, mPy, mPz;

			float tmpGx[step] __attribute__((aligned(Align)));
			float tmpGy[step] __attribute__((aligned(Align)));
			float tmpGz[step] __attribute__((aligned(Align)));
			float tmpK [step] __attribute__((aligned(Align)));
			float tmpV [step] __attribute__((aligned(Align)));

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[3], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0;

				{
					size_t tmi = idx/XC;

					X[2] = tmi/YC;
					X[1] = tmi - X[2]*YC;
					X[0] = idx - tmi*XC;
					X[2]--;	// Removes ghosts
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
#if	defined(__AVX512F__)
					mMy = opCode(permutexvar_ps, vShRg, opCode(load_ps, &m[idxMy]));
#elif	defined(__AVX2__)
					mMy = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 7,0,1,2,3,4,5,6));
#elif	defined(__AVX__)
					mel = opCode(permute_ps, opCode(load_ps, &m[idxMy]), 0b10010011);
					vel = opCode(permute2f128_ps, mel, mel, 0b00000001);
					mMy = opCode(blend_ps, mel, vel, 0b00010001);
#else
					mel = opCode(load_ps, &m[idxMy]);
					mMy = opCode(shuffle_ps, mel, mel, 0b10010011);
#endif
				}
				else
				{
					idxMy = idx - XC;
					mMy = opCode(load_ps, &m[idxMy]);

					if (X[1] == YC-1)
					{
						idxPy = idx - Sf + XC;
#if	defined(__AVX512F__)
						mPy = opCode(permutexvar_ps, vShLf, opCode(load_ps, &m[idxPy]));
#elif	defined(__AVX2__)
						mPy = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 1,2,3,4,5,6,7,0));
#elif	defined(__AVX__)
						mel = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b00111001);
						vel = opCode(permute2f128_ps, mel, mel, 0b00000001);
						mPy = opCode(blend_ps, mel, vel, 0b10001000);
#else
						vel = opCode(load_ps, &m[idxPy]);
						mPy = opCode(shuffle_ps, vel, vel, 0b00111001);
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
				idxP0 = idx;

				mel = opCode(load_ps, &m[idxP0]); // Carga m
				vel = opCode(load_ps, &v[idxMz]); // Carga v

				// Calculo los gradientes con módulo
				// Version sin módulo ; descomentar si modulo necesario ; assumes continuous field
				grd = opCode(sub_ps, opCode(load_ps, &m[idxPx]), mel);
				if (wMod) {
					tmp = opCode(mod_ps, grd, tpVec);
					mPx = opCode(mul_ps, tmp, tmp);
				} else
					mPx = opCode(mul_ps, grd, grd);

				grd = opCode(sub_ps, opCode(load_ps, &m[idxMx]), mel);
				if (wMod) {
					tmp = opCode(mod_ps, grd, tpVec);
					mMx = opCode(mul_ps, tmp, tmp);
				} else
					mMx = opCode(mul_ps, grd, grd);

				grd = opCode(sub_ps, mPy, mel);
				if (wMod) {
					tmp = opCode(mod_ps, grd, tpVec);
					mPy = opCode(mul_ps, tmp, tmp);
				} else
					mPy = opCode(mul_ps, grd, grd);

				grd = opCode(sub_ps, mMy, mel);
				if (wMod) {
					tmp = opCode(mod_ps, grd, tpVec);
					mMy = opCode(mul_ps, tmp, tmp);
				} else
					mMy = opCode(mul_ps, grd, grd);

				grd = opCode(sub_ps, opCode(load_ps, &m[idxPz]), mel);
				if (wMod) {
					tmp = opCode(mod_ps, grd, tpVec);
					mPz = opCode(mul_ps, tmp, tmp);
				} else
					mPz = opCode(mul_ps, grd, grd);

				grd = opCode(sub_ps, opCode(load_ps, &m[idxMz]), mel);
				if (wMod) {
					tmp = opCode(mod_ps, grd, tpVec);
					mMz = opCode(mul_ps, tmp, tmp);
				} else
					mMz = opCode(mul_ps, grd, grd);

				grd = opCode(add_ps, mPx, mMx);
				mMx = opCode(add_ps, mPy, mMy);
				mMy = opCode(add_ps, mPz, mMz);

				// KINETIC
				// Added full contribution, cancels outside the horizon
				tmp = opCode(sub_ps, vel , opCode(mul_ps, mel, izVec));
				mPx = opCode(mul_ps, tmp, tmp);

				// POTENTIAL
				tmp = opCode(sin_ps, opCode(mul_ps, hlf, opCode(mul_ps, mel, izVec)));
				mPy = opCode(mul_ps, opCode(mul_ps, tmp, tmp), two);

				opCode(store_ps, tmpGx, grd);
				opCode(store_ps, tmpGy, mMx);
				opCode(store_ps, tmpGz, mMy);
				opCode(store_ps, tmpK,  mPx);
				opCode(store_ps, tmpV,  mPy);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					ptC += (double) (tmpV[ih]);
					ktC += (double) (tmpK[ih]);
					gxC += (double) (tmpGx[ih]);
					gyC += (double) (tmpGy[ih]);
					gzC += (double) (tmpGz[ih]);

					// Saves map
					if	(map == true) {
						unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
						m2[iNx] = (tmpGx[ih] + tmpGy[ih] + tmpGz[ih])*o2 + tmpK[ih]*iz2*0.5 + tmpV[ih]*zQ;
					}
				}
			}
		}

		gxC *= o2; gyC *= o2; gzC *= o2; ktC *= 0.5*iz2; ptC *= zQ;
#undef	_MData_
#undef	step
	}

	eRes[TH_GRX] = gxC;
	eRes[TH_GRY] = gyC;
	eRes[TH_GRZ] = gzC;
	eRes[TH_KIN] = ktC;
	eRes[TH_POT] = ptC;
}

template<const bool mod>
void	energyThetaCpu	(Scalar *axionField, const double delta2, const double aMass2, void *eRes, const bool map)
{
	const double ood2 = 0.25/delta2;
	double *R = axionField->RV();
	const FieldPrecision precision = axionField->Precision();
	const size_t Lx = axionField->Length();
	const size_t Vo = axionField->Surf();
	const size_t Vf = Vo + axionField->Size();

	axionField->exchangeGhosts(FIELD_M);

	switch	(map) {
		case	true:
			energyThetaKernelXeon<true, mod>(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), R, ood2, aMass2, Lx, Vo, Vf, precision, eRes);
			break;

		case	false:
			energyThetaKernelXeon<false,mod>(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), R, ood2, aMass2, Lx, Vo, Vf, precision, eRes);
			break;
	}
}

void	energyThetaCpu	(Scalar *axionField, const double delta2, const double aMass2, void *eRes, const bool map, const bool mod)
{
	switch	(mod) {
		case	true:
			energyThetaCpu<true> (axionField, delta2, aMass2, eRes, map);
			break;

		case	false:
			energyThetaCpu<false>(axionField, delta2, aMass2, eRes, map);
			break;
	}
}
