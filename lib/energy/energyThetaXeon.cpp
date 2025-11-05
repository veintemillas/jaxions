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

template<const EnType emask, const bool wMod>
void	energyThetaKernelXeon(const void * __restrict__ m_, const void * __restrict__ v_, void * __restrict__ m2_,
	double *R, double *z, double frw, const double ood2, const double aMass2,
			 const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision, void * __restrict__ eRes_, Scalar *fieldo)
{

	char *strdaa = static_cast<char *>(static_cast<void *>(fieldo->sData()));

	const size_t Sf = Lx*Lx;
	const size_t Ng = Vo/Sf;

	double * __restrict__ eRes = (double * __restrict__) eRes_;
	double gxC = 0., gyC = 0., gzC = 0., ktC = 0., ptC = 0.;
	double gxCM = 0., gyCM = 0., gzCM = 0., ktCM = 0., ptCM = 0.;
	double nummask = 0.;

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

		/* computes physical, not comoving enegy density */
		const double zR  = *R;
		const double zz  = *z;
		const double iz  = 1.0/zR;
		const double d1  = frw*zR/zz;
		const double iz2 = iz*iz;
		const double iz4 = iz2*iz2;
		const double zQ  = aMass2;
		const double o2  = ood2*iz2*iz2;
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

		#pragma omp parallel default(shared) reduction(+:gxC,gyC,gzC,ktC,ptC,gxCM,gyCM,gzCM,ktCM,ptCM,nummask)
		{
			_MData_ mel, vel, mMx, mMy, mMz, mPx, mPy, mPz, tmp, grd;
			_MData_ Mask;

			double tmpGx[step] __attribute__((aligned(Align)));
			double tmpGy[step] __attribute__((aligned(Align)));
			double tmpGz[step] __attribute__((aligned(Align)));
			double tmpV [step] __attribute__((aligned(Align)));
			double tmpK [step] __attribute__((aligned(Align)));
			double tmpM [step] __attribute__((aligned(Align)));

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[3], idxPx, idxMx, idxPy, idxMy, idxPz, idxMz, idxP0;

				{
					size_t tmi = idx/XC;

					X[2] = tmi/YC;
					X[1] = tmi - X[2]*YC;
					X[0] = idx - tmi*XC;
					X[2] -= Ng;	// Removes ghosts
				}

				// Prepare mask in m2 (only if m2 contains the mask)
				// counter to check if some of the entries of the vector is in the mask
				size_t ups = 0;
				if (emask & EN_MASK){

					#pragma unroll
					for (int ih=0; ih<step; ih++){
						unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
						if (strdaa[iNx] & STRING_MASK)
						{
							ups += 1;}
					}
					// If required prepare mask
					if (ups > 0)
					{
						#pragma unroll
							for (int ih=0; ih<step; ih++){
							unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
							if (strdaa[iNx] & STRING_MASK){
								tmpM[ih] = 1.0;
							} else {
								tmpM[ih] = 0.0;
							}
						}
						Mask = opCode(load_pd, tmpM);
						nummask += (double) ups;
					}

					// If only masked energy required there is nothing else to do
					if ( (ups == 0) && (emask == EN_MASK) )
						continue;
					// If map of masked energy ; setm2 to 0 and leave;
					if ( (ups == 0) && (emask == EN_MAPMASK) ) {
						#pragma unroll
						for (int ih=0; ih<step; ih++) {
							unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
							m2[iNx]    = 0;
						}
						continue;
					}
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
					mMy = opCode(permutexvar_pd, vShRg, opCode(load_pd, &m[idxMy]));
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
						mPy = opCode(permutexvar_pd, vShLf, opCode(load_pd, &m[idxPy]));
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
				vel = opCode(load_pd, &v[idxP0-Ng*Sf]); // Carga v

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

				mPz = opCode(sub_pd, vel, opCode(mul_pd, opCode(set1_pd,d1), opCode(mul_pd, mel, izVec)));
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

					if	(emask & EN_MAP) {
						unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
						if (emask == EN_MAPMASK)
							m2[iNx] = tmpM[ih]*((tmpGx[ih] + tmpGy[ih] + tmpGz[ih])*o2 + tmpK[ih]*iz4*0.5 + tmpV[ih]*zQ); //masked map
						else
							m2[iNx] = (tmpGx[ih] + tmpGy[ih] + tmpGz[ih])*o2 + tmpK[ih]*iz4*0.5 + tmpV[ih]*zQ;
					}
				}
				// masked energy
				if ((emask & EN_MASK) && (ups > 0) ) {
					opCode(store_pd, tmpGx, opCode(mul_pd,grd,Mask));
					opCode(store_pd, tmpGy, opCode(mul_pd,mMx,Mask));
					opCode(store_pd, tmpGz, opCode(mul_pd,mMy,Mask));
					opCode(store_pd, tmpK,  opCode(mul_pd,mPx,Mask));
					opCode(store_pd, tmpV,  opCode(mul_pd,mPy,Mask));
					#pragma unroll
					for (int ih=0; ih<step; ih++)
					{
						ptCM += tmpV[ih];
						ktCM += tmpK[ih];
						gxCM += tmpGx[ih];
						gyCM += tmpGy[ih];
						gzCM += tmpGz[ih];
					}
				}
			}
		}

		gxC *= o2; gyC *= o2; gzC *= o2; ktC *= iz4*0.5; ptC *= zQ;
		gxCM *= o2; gyCM *= o2; gzCM *= o2; ktCM *= iz4*0.5; ptCM *= zQ;

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

		/* computes physical, not comoving enegy density */

		const float zR  = *R;
		const float zz  = *z;
		const float iz  = 1.f/zR;
		const float d1  = frw*zR/zz;
		const float iz2 = iz*iz;
		const float iz4 = iz2*iz2;
		const float zQ = aMass2;
		const float o2 = ood2*iz2*iz2;
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

		#pragma omp parallel default(shared) reduction(+:gxC,gyC,gzC,ktC,ptC,gxCM,gyCM,gzCM,ktCM,ptCM,nummask)
		{
			_MData_ mel, vel, grd, tmp, mMx, mMy, mMz, mPx, mPy, mPz;
			_MData_ Mask;

			float tmpGx[step] __attribute__((aligned(Align)));
			float tmpGy[step] __attribute__((aligned(Align)));
			float tmpGz[step] __attribute__((aligned(Align)));
			float tmpK [step] __attribute__((aligned(Align)));
			float tmpV [step] __attribute__((aligned(Align)));
			float tmpM [step] __attribute__((aligned(Align)));

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[3], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0;

				{
					size_t tmi = idx/XC;

					X[2] = tmi/YC;
					X[1] = tmi - X[2]*YC;
					X[0] = idx - tmi*XC;
					X[2] -= Ng;	// Removes ghosts
				}

				// Prepare mask in m2 (only if m2 contains the mask)
				// counter to check if some of the entries of the vector is in the mask
				size_t ups = 0;
				if (emask & EN_MASK){

					#pragma unroll
					for (int ih=0; ih<step; ih++){
						unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
						if (strdaa[iNx] & STRING_MASK)
						{
							ups += 1;}
					}
					// If required prepare mask
					if (ups > 0)
					{
						#pragma unroll
							for (int ih=0; ih<step; ih++){
							unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
							if (strdaa[iNx] & STRING_MASK){
								tmpM[ih] = 1.0;
							} else {
								tmpM[ih] = 0.0;
							}
						}
						Mask = opCode(load_ps, tmpM);
						nummask += (double) ups;
					}

					// If only masked energy required there is nothing else to do
					if ( (ups == 0) && (emask == EN_MASK) )
						continue;
					// If map of masked energy ; setm2 to 0 and leave;
					if ( (ups == 0) && (emask == EN_MAPMASK) ) {
						#pragma unroll
						for (int ih=0; ih<step; ih++) {
							unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
							m2[iNx]    = 0;
						}
						continue;
					}
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
				vel = opCode(load_ps, &v[idxP0-Ng*Sf]); // Carga v

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
				tmp = opCode(sub_ps, vel , opCode(mul_ps, opCode(set1_ps, d1), opCode(mul_ps, mel, izVec)));
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

					if	(emask & EN_MAP) {
						unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
						if (emask == EN_MAPMASK)
							m2[iNx] = tmpM[ih]*((tmpGx[ih] + tmpGy[ih] + tmpGz[ih])*o2 + tmpK[ih]*iz4*0.5 + tmpV[ih]*zQ); //masked map
						else
							m2[iNx] = (tmpGx[ih] + tmpGy[ih] + tmpGz[ih])*o2 + tmpK[ih]*iz4*0.5 + tmpV[ih]*zQ;
					}
				}
				// masked energy
				if ((emask & EN_MASK) && (ups > 0) ) {
					opCode(store_ps, tmpGx, opCode(mul_ps,grd,Mask));
					opCode(store_ps, tmpGy, opCode(mul_ps,mMx,Mask));
					opCode(store_ps, tmpGz, opCode(mul_ps,mMy,Mask));
					opCode(store_ps, tmpK,  opCode(mul_ps,mPx,Mask));
					opCode(store_ps, tmpV,  opCode(mul_ps,mPy,Mask));
					#pragma unroll
					for (int ih=0; ih<step; ih++)
					{
						ptCM += (double) (tmpV[ih]);
						ktCM += (double) (tmpK[ih]);
						gxCM += (double) (tmpGx[ih]);
						gyCM += (double) (tmpGy[ih]);
						gzCM += (double) (tmpGz[ih]);
					}
				}
			}
		}

		gxC *= o2; gyC *= o2; gzC *= o2; ktC *= 0.5*iz4; ptC *= zQ;
		gxCM *= o2; gyCM *= o2; gzCM *= o2; ktCM *= 0.5*iz4; ptCM *= zQ;
#undef	_MData_
#undef	step
	}

	eRes[TH_GRX] = gxC;
	eRes[TH_GRY] = gyC;
	eRes[TH_GRZ] = gzC;
	eRes[TH_KIN] = ktC;
	eRes[TH_POT] = ptC;

	eRes[TH_GRXM] = gxCM;
	eRes[TH_GRYM] = gyCM;
	eRes[TH_GRZM] = gzCM;
	eRes[TH_KINM] = ktCM;
	eRes[TH_POTM] = ptCM;

	eRes[MM_NUMM] = nummask;

	//LogOut("Energy %f %f %f %f %f\n",  eRes[TH_GRX],  eRes[TH_GRY],  eRes[TH_GRZ],  eRes[TH_KIN],  eRes[TH_POT]);
	//LogOut("Energy %f %f %f %f %f\n",  eRes[TH_GRXM],  eRes[TH_GRYM],  eRes[TH_GRZM],  eRes[TH_KINM],  eRes[TH_POTM]);
	//LogOut("nmp %f %f\n", eRes[MM_NUMM], nummask);
}

void	energyThetaCpu	(Scalar *axionField, const double delta2, const double aMass2, void *eRes, const EnType mapmask, const bool mod)
{
	const double ood2 = 0.25/delta2;
	double *R  = axionField->RV();
	double *z  = axionField->zV();
	double frw = axionField->BckGnd()->Frw();

	const FieldPrecision precision = axionField->Precision();
	const size_t Lx = axionField->Length();
	const size_t Vo = axionField->getNg()*axionField->Surf();
	const size_t Vf = Vo + axionField->Size();

	axionField->exchangeGhosts(FIELD_M);

	#define CASE_(mo,ene) \
	case ene: \
		energyThetaKernelXeon<ene,mo>(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), R, z, frw, ood2, aMass2, Lx, Vo, Vf, precision, eRes, axionField); \
	break;

	#define CASE_MULTIE_(mo2) \
	switch (mapmask){ \
	CASE_(mo2,EN_ENE) \
	CASE_(mo2,EN_MAP) \
	CASE_(mo2,EN_MASK) \
	CASE_(mo2,EN_ENEMASK) \
	CASE_(mo2,EN_MAPMASK) \
	CASE_(mo2,EN_ENEMAPMASK) \
	}

	switch	(mod) {
		case	true:
			CASE_MULTIE_(true)
			break;
		case	false:
			CASE_MULTIE_(false)
			break;
	}
}
