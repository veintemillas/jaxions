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

// #if	defined(__AVX512F__)
// 	#define	Align 64
// 	#define	_PREFIX_ _mm512
// #else
// 	#if not defined(__AVX__) and not defined(__AVX2__)
// 		#define	Align 16
// 		#define	_PREFIX_ _mm
// 	#else
// 		#define	Align 32
// 		#define	_PREFIX_ _mm256
// 	#endif
// #endif
//
// template<const bool map>
// void	energyNaxionKernelXeon(const void * __restrict__ m_, const void * __restrict__ v_, void * __restrict__ m2_, PropParms ppar, FieldPrecision precision, void * __restrict__ eRes_)
// {
// 	const double R = ppar.R, ood2a = ppar.ood2a, ct = ppar.ct, massA = ppar.massA, frw = ppar.frw, beta = ppar.beta;
// 	const size_t Lx = ppar.Lx, Sf = Lx*Lx, Vo = ppar.Vo, Vf = ppar.Vf, Ng = ppar.Ng, Vh = Vf+2*Vo;
//
// 	double * __restrict__ eRes = (double * __restrict__) eRes_;
// 	double gxC = 0., gyC = 0., gzC = 0., ktC = 0., ptC = 0.;
//
// 	if (precision == FIELD_DOUBLE)
// 	{
// #if	defined(__AVX512F__)
// 	#define	_MData_ __m512d
// 	#define	step 4
// #elif	defined(__AVX__)
// 	#define	_MData_ __m256d
// 	#define	step 2
// #else
// 	#define	_MData_ __m128d
// 	#define	step 1
// #endif
// 		double	Vrho = 0., Vth = 0., Krho = 0., Kth = 0., Gxrho = 0., Gxth = 0., Gyrho = 0., Gyth = 0., Gzrho = 0., Gzth = 0.;
// 		double	VrhoM = 0., VthM = 0., KrhoM = 0., KthM = 0., GxrhoM = 0., GxthM = 0., GyrhoM = 0., GythM = 0., GzrhoM = 0., GzthM = 0.;
// 		double	Rrho  = 0., RrhoM=0., nummask = 0.;
//
// 		const double * __restrict__ m	= (const double * __restrict__) __builtin_assume_aligned (m_,  Align);
// 		const double * __restrict__ v	= (const double * __restrict__) __builtin_assume_aligned (v_,  Align);
// 					double * __restrict__ m2	= (      double * __restrict__) __builtin_assume_aligned (m2_, Align);
//
// 		double * __restrict__ eRes	= (double * __restrict__) eRes_;
//
// 		const double zR  = *R;
// 		const double iz  = 1./zR;
// 		const double iz2 = iz*iz;
// 		//const double zQ = 9.*pow(zR, nQcd+2.);
// 		const double zQ = aMass2*zR*zR;
// 		const double lZ = 0.25*LL*zR*zR;
// #if	defined(__AVX512F__)
// 		const size_t XC = (Lx<<2);
// 		const size_t YC = (Lx>>2);
//
// 		const double    __attribute__((aligned(Align))) cjgAux[8] = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
// 		const double    __attribute__((aligned(Align))) QCD2Aux[8]= { 1., 0., 1., 0., 1., 0., 1., 0. };
// 		const double    __attribute__((aligned(Align))) ivZAux[8] = { iz, 0., iz, 0., iz, 0., iz, 0. };	// Only real part
// 		const double    __attribute__((aligned(Align))) shfAux[8] = {shift, 0., shift, 0., shift, 0., shift, 0. };
// 		const double    __attribute__((aligned(Align))) lzQAux[8] = { lZ, zQ, lZ, zQ, lZ, zQ, lZ, zQ };
// 		const long long __attribute__((aligned(Align))) shfRg[8]  = {6, 7, 0, 1, 2, 3, 4, 5 };
// 		const long long __attribute__((aligned(Align))) shfLf[8]  = {2, 3, 4, 5, 6, 7, 0, 1 };
//
// 		const auto vShRg = opCode(load_si512, shfRg);
// 		const auto vShLf = opCode(load_si512, shfLf);
// #elif	defined(__AVX__)
// 		const size_t XC = (Lx<<1);
// 		const size_t YC = (Lx>>1);
//
// 		const double __attribute__((aligned(Align))) cjgAux[4] = { 1.,-1., 1.,-1. };
// 		const double __attribute__((aligned(Align))) QCD2Aux[4]= { 1., 0., 1., 0. };
// 		const double __attribute__((aligned(Align))) ivZAux[4] = { iz, 0., iz, 0. };	// Only real part
// 		const double __attribute__((aligned(Align))) shfAux[4] = {shift, 0., shift, 0. };
// 		const double __attribute__((aligned(Align))) lzQAux[4] = { lZ, zQ, lZ, zQ };
// #else
// 		const size_t XC = Lx;
// 		const size_t YC = Lx;
//
// 		const double __attribute__((aligned(Align))) cjgAux[2] = { 1.,-1. };
// 		const double __attribute__((aligned(Align))) QCD2Aux[2]= { 1., 0. };
// 		const double __attribute__((aligned(Align))) ivZAux[2] = { iz, 0. };	// Only real part
// 		const double __attribute__((aligned(Align))) shfAux[2] = {shift, 0.};
// 		const double __attribute__((aligned(Align))) lzQAux[2] = { lZ, zQ };
//
// #endif
// 		const _MData_ cjg  = opCode(load_pd, cjgAux);
// 		const _MData_ ivZ  = opCode(load_pd, ivZAux);
// 		const _MData_ shVc = opCode(load_pd, shfAux);
// 		const _MData_ pVec = opCode(load_pd, lzQAux);
// 		const _MData_ one  = opCode(set1_pd, 1.0);
// 		const _MData_ hVec = opCode(set1_pd, 0.5);
// 		const _MData_ ivZ2 = opCode(set1_pd, iz2);
// 		const _MData_ oVec = opCode(set1_pd, o2);
// 		//for VQCD2
// 		const _MData_ qcd2 = opCode(load_pd, QCD2Aux);
// 		const _MData_ iiiZ = opCode(set1_pd, iz);
//
// 		#pragma omp parallel default(shared)
// 		{
// 			_MData_ mel, vel, mPx, mMx, mPy, mMy, mPz, mMz, mdv, mod, mTp;
// 			_MData_ Grx, Gry, Grz, tGx, tGy, tGz, tVp, tKp, mCg, mSg;
// 			_MData_ Mask, Frho;
//
// 			double tmpS[2*step] __attribute__((aligned(Align)));
//
// 			#pragma omp for schedule(static) reduction(+:Vrho,Vth,Krho,Kth,Gxrho,Gxth,Gyrho,Gyth,Gzrho,Gzth,VrhoM,VthM,KrhoM,KthM,GxrhoM,GxthM,GyrhoM,GythM,GzrhoM,GzthM,Rrho)
// 			for (size_t idx = Vo; idx < Vf; idx += step)
// 			{
// 				size_t X[3], idxPx, idxMx, idxPy, idxMy, idxPz, idxMz, idxP0, idxV0;
//
// 				{
// 					size_t tmi = idx/XC;
//
// 					X[2] = tmi/YC;
// 					X[1] = tmi - X[2]*YC;
// 					X[0] = idx - tmi*XC;
// 					X[2] -= NN; // remove ghosts
// 				}
//
// 				idxPz = ((idx+Sf) << 1);
// 				idxMz = ((idx-Sf) << 1);
// 				idxP0 = (idx << 1);
// 				idxV0 = (idx - NN*Sf) << 1;
//
// 				mel = opCode(load_pd, &m[idxP0]); //Carga m con shift
//
// // Prepare mask in m2 (only if m2 contains the mask in complex notation)
// // counter to check if some of the entries of the vector is in the mask
// 				size_t ups = 0;
// 				if (emask & EN_MASK){
//
// 					#pragma unroll
// 					for (int ih=0; ih<step; ih++){
// 						unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
// 						if (strdaa[iNx] & STRING_MASK)
// 						{
// 							ups += 1;}
// 					}
// 					// If required prepare mask
// 					if (ups > 0)
// 					{
// 						#pragma unroll
// 							for (int ih=0; ih<step; ih++){
// 							unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
// 							if (strdaa[iNx] & STRING_MASK){
// 								tmpS[(ih<<1)+1] = 1.f; // imag part
// 								tmpS[(ih<<1)]   = 1.f; // real part
// 							} else {
// 								tmpS[(ih<<1)+1] = 0.f; // imag part
// 								tmpS[(ih<<1)]   = 0.f; // real part
// 							}
// 						}
// 						Mask = opCode(load_pd, tmpS);
// 						nummask += (double) ups;
// 					}
//
// 					// If only masked energy required there is nothing else to do
// 					if ( (ups == 0) && (emask == EN_MASK) )
// 						continue;
// 					// If map of masked energy ; setm2 to 0 and leave;
// 					if ( (ups == 0) && (emask == EN_MAPMASK) ) {
// 						#pragma unroll
// 						for (int ih=0; ih<step; ih++) {
// 							unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
// 							m2[iNx]    = 0;  m2[iNx+Vt] = 0;   // theta and Rho field
// 						}
// 						continue;
// 					}
// 				}
//
// 				if (X[0] == XC-step)
// 					idxPx = ((idx - XC + step) << 1);
// 				else
// 					idxPx = ((idx + step) << 1);
//
// 				if (X[0] == 0)
// 					idxMx = ((idx + XC - step) << 1);
// 				else
// 					idxMx = ((idx - step) << 1);
//
// 				if (X[1] == 0)
// 				{
// 					idxMy = ((idx + Sf - XC) << 1);
// 					idxPy = ((idx + XC) << 1);
//
// 					mPy = opCode(sub_pd, opCode(load_pd, &m[idxPy]), mel);
// #if	defined(__AVX512F__)
// 					mMy = opCode(sub_pd, mel, opCode(permutexvar_pd, vShRg, opCode(load_pd, &m[idxMy])));
// #elif	defined(__AVX__)
// 					mMx = opCode(load_pd, &m[idxMy]);
// 					mMy = opCode(sub_pd, mel, opCode(permute2f128_pd, mMx, mMx, 0b00000001));
// #else
// 					mMy = opCode(sub_pd, mel, opCode(load_pd, &m[idxMy]));
// #endif
// 				}
// 				else
// 				{
// 					idxMy = ((idx - XC) << 1);
//
// 					mMy = opCode(sub_pd, mel, opCode(load_pd, &m[idxMy]));
//
// 					if (X[1] == YC-1)
// 					{
// 						idxPy = ((idx - Sf + XC) << 1);
// #if	defined(__AVX512F__)
// 						mPy = opCode(sub_pd, opCode(permutexvar_pd, vShLf, opCode(load_pd, &m[idxPy])), mel);
// #elif	defined(__AVX__)
// 						mMx = opCode(load_pd, &m[idxPy]);
// 						mPy = opCode(sub_pd, opCode(permute2f128_pd, mMx, mMx, 0b00000001), mel);
// #else
// 						mPy = opCode(sub_pd, opCode(load_pd, &m[idxPy]), mel);
// #endif
// 					}
// 					else
// 					{
// 						idxPy = ((idx + XC) << 1);
// 						mPy = opCode(sub_pd, opCode(load_pd, &m[idxPy]), mel);
// 					}
// 				}
//
// 				mPx = opCode(sub_pd, opCode(load_pd, &m[idxPx]), mel);
// 				mMx = opCode(sub_pd, mel, opCode(load_pd, &m[idxMx]));
// 				// mPy y mMy ya están cargado
// 				mPz = opCode(sub_pd, opCode(load_pd, &m[idxPz]), mel);
// 				mMz = opCode(sub_pd, mel, opCode(load_pd, &m[idxMz]));
//
// 				vel = opCode(load_pd, &v[idxV0]);//Carga v
// 				mod = opCode(mul_pd, mel, mel);
//
// 				mTp = opCode(md2_pd, mod);
//
// 				mod = opCode(mul_pd, mTp, ivZ2);	// Factor |mel|^2/z^2, util luego
//
// 				mCg = opCode(div_pd, mel, mTp);	// Ahora mCg tiene 1/mel
//
// 				// Meto en mSg = shuffled(mCg) (Intercambio parte real por parte imaginaria)
// #if	defined(__AVX__)
// 				mSg = opCode(mul_pd, opCode(permute_pd, mCg, 0b01010101), cjg);
// #else
// 				mSg = opCode(mul_pd, opCode(shuffle_pd, mCg, mCg, 0b00000001), cjg);
// #endif
//
// 				// Calculo los gradientes
// #if	defined(__AVX512F__)
// 				tGx = opCode(mul_pd, mPx, mCg);
// 				tGy = opCode(mul_pd, mPx, mSg);
//
// 				tGz = opCode(mask_add_pd,
// 					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGy), _MM_PERM_BADC)),
// 					opCode(kmov, 0b0000000001010101),
// 					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGx), _MM_PERM_BADC)),
// 					tGx);
//
// 				mdv = opCode(mask_add_pd, tGz, opCode(kmov, 0b0000000010101010), tGz, tGy);
//
// 				tGx = opCode(mul_pd, mMx, mCg);
// 				tGy = opCode(mul_pd, mMx, mSg);
//
// 				tGz = opCode(mask_add_pd,
// 					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGy), _MM_PERM_BADC)),
// 					opCode(kmov, 0b0000000001010101),
// 					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGx), _MM_PERM_BADC)),
// 					tGx);
//
// 				Grx = opCode(mask_add_pd, tGz, opCode(kmov, 0b0000000010101010), tGz, tGy);
// 				Grx = opCode(add_pd,
// 					opCode(mul_pd, mdv, mdv),
// 					opCode(mul_pd, Grx, Grx));
//
// 				tGx = opCode(mul_pd, mPy, mCg);
// 				tGy = opCode(mul_pd, mPy, mSg);
//
// 				tGz = opCode(mask_add_pd,
// 					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGy), _MM_PERM_BADC)),
// 					opCode(kmov, 0b0000000001010101),
// 					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGx), _MM_PERM_BADC)),
// 					tGx);
//
// 				mdv = opCode(mask_add_pd, tGz, opCode(kmov, 0b0000000010101010), tGz, tGy);
//
// 				tGx = opCode(mul_pd, mMy, mCg);
// 				tGy = opCode(mul_pd, mMy, mSg);
//
// 				tGz = opCode(mask_add_pd,
// 					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGy), _MM_PERM_BADC)),
// 					opCode(kmov, 0b0000000001010101),
// 					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGx), _MM_PERM_BADC)),
// 					tGx);
//
// 				Gry = opCode(mask_add_pd, tGz, opCode(kmov, 0b0000000010101010), tGz, tGy);
// 				Gry = opCode(add_pd,
// 					opCode(mul_pd, mdv, mdv),
// 					opCode(mul_pd, Gry, Gry));
//
// 				tGx = opCode(mul_pd, mPz, mCg);
// 				tGy = opCode(mul_pd, mPz, mSg);
//
// 				tGz = opCode(mask_add_pd,
// 					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGy), _MM_PERM_BADC)),
// 					opCode(kmov, 0b0000000001010101),
// 					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGx), _MM_PERM_BADC)),
// 					tGx);
//
// 				mdv = opCode(mask_add_pd, tGz, opCode(kmov, 0b0000000010101010), tGz, tGy);
//
// 				tGx = opCode(mul_pd, mMz, mCg);
// 				tGy = opCode(mul_pd, mMz, mSg);
//
// 				tGz = opCode(mask_add_pd,
// 					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGy), _MM_PERM_BADC)),
// 					opCode(kmov, 0b0000000001010101),
// 					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGx), _MM_PERM_BADC)),
// 					tGx);
//
// 				Grz = opCode(mask_add_pd, tGz, opCode(kmov, 0b0000000010101010), tGz, tGy);
// 				Grz = opCode(add_pd,
// 					opCode(mul_pd, mdv, mdv),
// 					opCode(mul_pd, Grz, Grz));
//
// 				tGx = opCode(mul_pd, vel, mCg);
// 				tGy = opCode(mul_pd, vel, mSg);
//
// 				tGz = opCode(mask_add_pd,
// 					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGy), _MM_PERM_BADC)),
// 					opCode(kmov, 0b0000000001010101),
// 					opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tGx), _MM_PERM_BADC)),
// 					tGx);
//
// 				mdv = opCode(sub_pd, opCode(mask_add_pd, tGz, opCode(kmov, 0b0000000010101010), tGz, tGy), ivZ);
// #else				// Las instrucciones se llaman igual con AVX o con SSE3
// 				Grx = opCode(hadd_pd, opCode(mul_pd, mPx, mCg), opCode(mul_pd, mPx, mSg));
// 				mdv = opCode(hadd_pd, opCode(mul_pd, mMx, mCg), opCode(mul_pd, mMx, mSg));
// 				Grx = opCode(add_pd,
// 					opCode(mul_pd, mdv, mdv),
// 					opCode(mul_pd, Grx, Grx));
//
// 				Gry = opCode(hadd_pd, opCode(mul_pd, mPy, mCg), opCode(mul_pd, mPy, mSg));
// 				mdv = opCode(hadd_pd, opCode(mul_pd, mMy, mCg), opCode(mul_pd, mMy, mSg));
// 				Gry = opCode(add_pd,
// 					opCode(mul_pd, mdv, mdv),
// 					opCode(mul_pd, Gry, Gry));
//
// 				Grz = opCode(hadd_pd, opCode(mul_pd, mPz, mCg), opCode(mul_pd, mPz, mSg));
// 				mdv = opCode(hadd_pd, opCode(mul_pd, mMz, mCg), opCode(mul_pd, mMz, mSg));
// 				Grz = opCode(add_pd,
// 					opCode(mul_pd, mdv, mdv),
// 					opCode(mul_pd, Grz, Grz));
//
// 				mdv = opCode(sub_pd, opCode(hadd_pd, opCode(mul_pd, vel, mCg), opCode(mul_pd, vel, mSg)), ivZ);
// #endif
// 				tGx = opCode(mul_pd, mod, Grx);
// 				tGy = opCode(mul_pd, mod, Gry);
// 				tGz = opCode(mul_pd, mod, Grz);
//
// 				tKp = opCode(mul_pd, mod, opCode(mul_pd, mdv, mdv));
//
// 				Grx = opCode(sub_pd, mel, shVc);
// 				Gry = opCode(mul_pd, Grx, Grx);
// 				Grz = opCode(md2_pd, Gry);
// 				Gry = opCode(mul_pd, Grz, ivZ2);
// 				Frho = opCode(sqrt_pd, Gry);
//
// 				switch	(VQcd & VQCD_TYPE) {
//
// 					default:
// 					case	VQCD_1:
// 					case	VQCD_0:
// 						mSg = opCode(sub_pd, Gry, one);
// 						mod = opCode(mul_pd, mSg, mSg);
// 						mCg = opCode(sub_pd, one, opCode(div_pd, Grx, opCode(sqrt_pd, Grz)));  // 1-m/|m|
// 						break;
//
// 					case	VQCD_PQ_ONLY:
// 						mSg = opCode(sub_pd, Gry, one);
// 						mod = opCode(mul_pd, mSg, mSg);
// 						mCg = one;  // will be multiplied by 0
// 						break;
//
// 					case	VQCD_1_PQ_2:
// 						mSg = opCode(sub_pd, opCode(mul_pd, Gry, Gry), one);   // |rho|^4-1
// 						mod = opCode(mul_pd, mSg, mSg);   											// (|rho|^4-1)^2
// 						mCg = opCode(sub_pd, one, opCode(div_pd, Grx, opCode(sqrt_pd, Grz)));  // 1-m/|m|
// 						break;
//
// 					case	VQCD_2:
// 						// mSg = opCode(sub_pd, Gry, one);
// 						// mod = opCode(mul_pd, mSg, mSg);
// 						// mTp = opCode(sub_pd, one, opCode(mul_pd, Grx, ivZ));
// 						// mCg = opCode(mul_pd, mTp, mTp);
//
// 						mSg = opCode(sub_pd, Gry, one);   // |rho|^2-1
// 						mod = opCode(mul_pd, mSg, mSg);   // (|rho|^2-1)^2
// 						mTp = opCode(sub_pd, qcd2, opCode(mul_pd, Grx, iiiZ));								// (1-Re'/Z), -Im/Z
// 						mCg = opCode(mul_pd, hVec, opCode(md2_pd, opCode(mul_pd,mTp,mTp) ));	// 0.5*((1-Re'/Z)^2+(Im/Z)^2), ...2
//
// 						break;
//
// 					case	VQCD_1N2:		//to be checked
// 						mSg = opCode(sub_pd, Gry, one);   // |rho|^2-1
// 						mod = opCode(mul_pd, mSg, mSg);   // (|rho|^2-1)^2
// 						mCg = opCode(mul_pd, hVec, opCode(sub_pd, one, opCode(div_pd, Gry, Grz)));
// 					break;
// 				}
// #if	defined(__AVX512F__)
// 				tVp = opCode(mask_blend_pd, opCode(kmov, 0b10101010), mod, opCode(permute_pd, mCg, 0b01010101));
// #elif	defined(__AVX__)
// 				tVp = opCode(blend_pd, mod, opCode(permute_pd, mCg, 0b00000101), 0b00001010);
// #else
// 				tVp = opCode(shuffle_pd, mod, mCg, 0b00000001);
// #endif
// 				if	(emask & EN_MAP) {
// 					mdv = opCode(add_pd,
// 						opCode(add_pd,
// 							opCode(mul_pd, tKp, hVec),
// 							opCode(mul_pd, opCode(add_pd, tGx, opCode(add_pd, tGy, tGz)), oVec)),
// 						opCode(mul_pd, pVec, tVp));
//
// 						//masked map
// 						if (emask == EN_MAPMASK)
// 							mdv = opCode(mul_pd,Mask,mdv);
//
// 						opCode(store_pd, tmpS, mdv);
//
// 					#pragma unroll
// 					for (int ih=0; ih<step; ih++)
// 					{
// 						unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
// 						m2[iNx]    = tmpS[(ih<<1)+1]; // Theta field
// 						m2[iNx+Vt] = tmpS[(ih<<1)];   // Rho field
// 					}
// 				}
//
// //-----------------------------------------------------------------------------
// // TOTAL ENERGY
// //-----------------------------------------------------------------------------
//
// if (emask & EN_ENE){
// #if	defined(__AVX512F__)
// 				opCode(store_pd, tmpS, tGx);
// 				Gxrho += tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
// 				Gxth  += tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];
//
// 				opCode(store_pd, tmpS, tGy);
// 				Gyrho += tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
// 				Gyth  += tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];
//
// 				opCode(store_pd, tmpS, tGz);
// 				Gzrho += tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
// 				Gzth  += tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];
//
// 				opCode(store_pd, tmpS, tVp);
// 				Vrho += tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
// 				Vth  += tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];
//
// 				opCode(store_pd, tmpS, tKp);
// 				Krho += tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
// 				Kth  += tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];
//
// 				opCode(store_pd, tmpS, Frho);
// 				Rrho += tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
//
// #elif defined(__AVX__)
// 				opCode(store_pd, tmpS, tGx);
// 				Gxrho += tmpS[0] + tmpS[2];
// 				Gxth  += tmpS[1] + tmpS[3];
//
// 				opCode(store_pd, tmpS, tGy);
// 				Gyrho += tmpS[0] + tmpS[2];
// 				Gyth  += tmpS[1] + tmpS[3];
//
// 				opCode(store_pd, tmpS, tGz);
// 				Gzrho += tmpS[0] + tmpS[2];
// 				Gzth  += tmpS[1] + tmpS[3];
//
// 				opCode(store_pd, tmpS, tVp);
// 				Vrho += tmpS[0] + tmpS[2];
// 				Vth  += tmpS[1] + tmpS[3];
//
// 				opCode(store_pd, tmpS, tKp);
// 				Krho += tmpS[0] + tmpS[2];
// 				Kth  += tmpS[1] + tmpS[3];
//
// 				opCode(store_pd, tmpS, Frho);
// 				Rrho += tmpS[0] + tmpS[2];
// #else
// 				opCode(store_pd, tmpS, tGx);
// 				Gxrho += tmpS[0];
// 				Gxth  += tmpS[1];
//
// 				opCode(store_pd, tmpS, tGy);
// 				Gyrho += tmpS[0];
// 				Gyth  += tmpS[1];
//
// 				opCode(store_pd, tmpS, tGz);
// 				Gzrho += tmpS[0];
// 				Gzth  += tmpS[1];
//
// 				opCode(store_pd, tmpS, tVp);
// 				Vrho += tmpS[0];
// 				Vth  += tmpS[1];
//
// 				opCode(store_pd, tmpS, tKp);
// 				Krho += tmpS[0];
// 				Kth  += tmpS[1];
//
// 				opCode(store_pd, tmpS, Frho);
// 				Rrho += tmpS[0];
// #endif
// }
//
// //-----------------------------------------------------------------------------
// // MASKED ENERGY
// //-----------------------------------------------------------------------------
//
// // if requested and required by having some element inside the mask!
// 				if ((emask & EN_MASK) && (ups > 0) )
// 				{
// #if	defined(__AVX512F__)
// 								opCode(store_pd, tmpS, opCode(mul_pd,tGx,Mask));
// 								GxrhoM +=  tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
// 								GxthM  +=  tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];
//
// 								opCode(store_pd, tmpS, opCode(mul_pd,tGy,Mask));
// 								GyrhoM +=  tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
// 								GythM  +=  tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];
//
// 								opCode(store_pd, tmpS, opCode(mul_pd,tGz,Mask));
// 								GzrhoM +=  tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
// 								GzthM  +=  tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];
//
// 								opCode(store_pd, tmpS, opCode(mul_pd,tVp,Mask));
// 								VrhoM  +=  tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
// 								VthM   +=  tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];
//
// 								opCode(store_pd, tmpS, opCode(mul_pd,tKp,Mask));
// 								KrhoM  +=  tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
// 								KthM   +=  tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7];
//
// 								opCode(store_pd, tmpS, opCode(mul_pd,Frho,Mask));
// 								RrhoM  +=  tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
// #elif defined(__AVX__)
// 								opCode(store_pd, tmpS, opCode(mul_pd,tGx,Mask));
// 								GxrhoM += tmpS[0] + tmpS[2];
// 								GxthM  += tmpS[1] + tmpS[3];
//
// 								opCode(store_pd, tmpS, opCode(mul_pd,tGy,Mask));
// 								GyrhoM += tmpS[0] + tmpS[2];
// 								GythM  += tmpS[1] + tmpS[3];
//
// 								opCode(store_pd, tmpS, opCode(mul_pd,tGz,Mask));
// 								GzrhoM += tmpS[0] + tmpS[2];
// 								GzthM  += tmpS[1] + tmpS[3];
//
// 								opCode(store_pd, tmpS, opCode(mul_pd,tVp,Mask));
// 								VrhoM  += tmpS[0] + tmpS[2];
// 								VthM   += tmpS[1] + tmpS[3];
//
// 								opCode(store_pd, tmpS, opCode(mul_pd,tKp,Mask));
// 								KrhoM  += tmpS[0] + tmpS[2];
// 								KthM   += tmpS[1] + tmpS[3];
//
// 								opCode(store_pd, tmpS, opCode(mul_pd,Frho,Mask));
// 								RrhoM  +=  tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6];
// #else
// 								opCode(store_pd, tmpS, opCode(mul_pd,tGx,Mask));
// 								GxrhoM +=  tmpS[0];
// 								GxthM  +=  tmpS[1];
//
// 								opCode(store_pd, tmpS, opCode(mul_pd,tGy,Mask));
// 								GyrhoM +=  tmpS[0];
// 								GythM  +=  tmpS[1];
//
// 								opCode(store_pd, tmpS, opCode(mul_pd,tGz,Mask));
// 								GzrhoM +=  tmpS[0];
// 								GzthM  +=  tmpS[1];
//
// 								opCode(store_pd, tmpS, opCode(mul_pd,tVp,Mask));
// 								VrhoM  +=  tmpS[0] ;
// 								VthM   +=  tmpS[1] ;
//
// 								opCode(store_pd, tmpS, opCode(mul_pd,tKp,Mask));
// 								KrhoM  +=  tmpS[0] ;
// 								KthM   +=  tmpS[1] ;
//
// 								opCode(store_pd, tmpS, opCode(mul_pd,Frho,Mask));
// 								RrhoM  +=  tmpS[0];
// #endif
// 				} //end masked energy
// 			} //end for loop
// 		} //end parallel
//
// 		eRes[RH_GRX] = Gxrho*o2;
// 		eRes[TH_GRX] = Gxth *o2;
// 		eRes[RH_GRY] = Gyrho*o2;
// 		eRes[TH_GRY] = Gyth *o2;
// 		eRes[RH_GRZ] = Gzrho*o2;
// 		eRes[TH_GRZ] = Gzth *o2;
// 		eRes[RH_POT] = Vrho *lZ;
// 		eRes[TH_POT] = Vth  *zQ;
// 		eRes[RH_KIN] = Krho *.5;
// 		eRes[TH_KIN] = Kth  *.5;
//
// 		eRes[RH_GRXM] = GxrhoM*o2;
// 		eRes[TH_GRXM] = GxthM *o2;
// 		eRes[RH_GRYM] = GyrhoM*o2;
// 		eRes[TH_GRYM] = GythM *o2;
// 		eRes[RH_GRZM] = GzrhoM*o2;
// 		eRes[TH_GRZM] = GzthM *o2;
// 		eRes[RH_POTM] = VrhoM *lZ;
// 		eRes[TH_POTM] = VthM  *zQ;
// 		eRes[RH_KINM] = KrhoM *.5;
// 		eRes[TH_KINM] = KthM  *.5;
// 		eRes[RH_RHOM] = RrhoM;
//
// 		eRes[MM_NUMM] = nummask;
//
// #undef	_MData_
// #undef	step
// } //end loop DOUBLE precision
// 	else if (precision == FIELD_SINGLE)
// 	{
// #if	defined(__AVX512F__)
// 	#define	_MData_ __m512
// 	#define	step 8
// #elif	defined(__AVX__)
// 	#define	_MData_ __m256
// 	#define	step 4
// #else
// 	#define	_MData_ __m128
// 	#define	step 2
// #endif
// 		double	Vrho  = 0., Vth  = 0., Krho  = 0., Kth  = 0., Gxrho  = 0., Gxth  = 0., Gyrho  = 0., Gyth  = 0., Gzrho  = 0., Gzth  = 0.;
// 		double	VrhoM = 0., VthM = 0., KrhoM = 0., KthM = 0., GxrhoM = 0., GxthM = 0., GyrhoM = 0., GythM = 0., GzrhoM = 0., GzthM = 0.;
// 		double	Rrho  = 0., RrhoM=0., nummask = 0.;
//
// 		const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_,  Align);
// 		const float * __restrict__ v	= (const float * __restrict__) __builtin_assume_aligned (v_,  Align);
// 					float * __restrict__ m2	= (      float * __restrict__) __builtin_assume_aligned (m2_, Align);
//
// 		double * __restrict__ eRes	= (double * __restrict__) eRes_;
//
// 		const float zR  = *R;
// 		const float iz  = (float) (1.f/zR);
// 		const float iz2 = iz*iz;
// 		//const float zQ = 9.f*powf(zR, nQcd+2.);
// 		const float zQ = aMass2*zR*zR;
// 		const float lZ = 0.25f*LL*zR*zR;
// 		const float sh = shift;				// Makes clang happy
//
// #if	defined(__AVX512F__)
// 		const size_t XC = (Lx<<3);
// 		const size_t YC = (Lx>>3);
// // Usa constexpr y setr_epi64
// 		const float __attribute__((aligned(Align))) cjgAux[16]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
// 		const float __attribute__((aligned(Align))) QCD2Aux[16] = { 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0. };
// 		const float __attribute__((aligned(Align))) ivZAux[16]  = { iz, 0., iz, 0., iz, 0., iz, 0., iz, 0., iz, 0., iz, 0., iz, 0. };
// 		const float __attribute__((aligned(Align))) shfAux[16]  = { sh, 0., sh, 0., sh, 0., sh, 0., sh, 0., sh, 0., sh, 0., sh, 0. };
// 		const float __attribute__((aligned(Align))) lzQAux[16]  = { lZ, zQ, lZ, zQ, lZ, zQ, lZ, zQ, lZ, zQ, lZ, zQ, lZ, zQ, lZ, zQ };
// 		const int   __attribute__((aligned(Align))) shfRg[16]   = {14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
// 		const int   __attribute__((aligned(Align))) shfLf[16]   = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1};
//
// 		const auto vShRg = opCode(load_si512, shfRg);
// 		const auto vShLf = opCode(load_si512, shfLf);
// #elif	defined(__AVX__)
// 		const size_t XC = (Lx<<2);
// 		const size_t YC = (Lx>>2);
//
// 		const float __attribute__((aligned(Align))) cjgAux[8]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
// 		const float __attribute__((aligned(Align))) QCD2Aux[8] = { 1., 0., 1., 0., 1., 0., 1., 0. };
// 		const float __attribute__((aligned(Align))) ivZAux[8]  = { iz, 0., iz, 0., iz, 0., iz, 0. };	// Only real part
// 		const float __attribute__((aligned(Align))) shfAux[8]  = { sh, 0., sh, 0., sh, 0., sh, 0.};
// 		const float __attribute__((aligned(Align))) lzQAux[8]  = { lZ, zQ, lZ, zQ, lZ, zQ, lZ, zQ };
// #else
// 		const size_t XC = (Lx<<1);
// 		const size_t YC = (Lx>>1);
//
// 		const float __attribute__((aligned(Align))) cjgAux[4]  = { 1.,-1., 1.,-1. };
// 		const float __attribute__((aligned(Align))) QCD2Aux[4] = { 1., 0., 1., 0. };
// 		const float __attribute__((aligned(Align))) ivZAux[4]  = { iz, 0., iz, 0. };	// Only real part
// 		const float __attribute__((aligned(Align))) shfAux[4]  = { sh, 0., sh, 0. };
// 		const float __attribute__((aligned(Align))) lzQAux[4]  = { lZ, zQ, lZ, zQ };
// #endif
//
// 		const _MData_ cjg  = opCode(load_ps, cjgAux);
// 		const _MData_ ivZ  = opCode(load_ps, ivZAux);
// 		const _MData_ shVc = opCode(load_ps, shfAux);
// 		const _MData_ pVec = opCode(load_ps, lzQAux);
// 		const _MData_ one  = opCode(set1_ps, 1.f);
// 		const _MData_ hVec = opCode(set1_ps, 0.5f);
// 		const _MData_ ivZ2 = opCode(set1_ps, iz2);
// 		const _MData_ oVec = opCode(set1_ps, o2);
// 		//for VQCD2
// 		const _MData_ qcd2 = opCode(load_ps, QCD2Aux);
// 		const _MData_ iiiZ = opCode(set1_ps, iz);
//
// 		#pragma omp parallel default(shared)
// 		{
// 			_MData_ mel, vel, mPx, mMx, mPy, mMy, mPz, mMz, mdv, mod, mTp;
// 			_MData_ Grx, Gry, Grz, tGx, tGy, tGz, tVp, tKp, mCg, mSg;
// 			_MData_ Mask, Frho;
//
// 			float tmpS[2*step] __attribute__((aligned(Align)));
//
// 			#pragma omp for schedule(static) reduction(+:Vrho,Vth,Krho,Kth,Gxrho,Gxth,Gyrho,Gyth,Gzrho,Gzth,VrhoM,VthM,KrhoM,KthM,GxrhoM,GxthM,GyrhoM,GythM,GzrhoM,GzthM,Rrho,RrhoM,nummask)
// 			for (size_t idx = Vo; idx < Vf; idx += step)
// 			{
//
// 				size_t X[3], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0, idxV0;
//
// 				idxPz = ((idx+Sf) << 1);
// 				idxMz = ((idx-Sf) << 1);
// 				idxP0 =  (idx     << 1);
// 				idxV0 = (idx - NN*Sf) << 1;
//
// 				//Paxion
// 				mel = opCode(load_ps, &m[idxP0]);
//
// 				{
// 					size_t tmi = idx/XC;
//
// 					X[2] = tmi/YC;
// 					X[1] = tmi - X[2]*YC;
// 					X[0] = idx - tmi*XC;
// 					X[2] -= NN; // Removes ghost zone // not used?
// 				}
//
//
// 				// calculate idx of neighbours and gradients
//
// 				if (X[0] == XC-step)
// 					idxPx = ((idx - XC + step) << 1);
// 				else
// 					idxPx = ((idx + step) << 1);
//
// 				if (X[0] == 0)
// 					idxMx = ((idx + XC - step) << 1);
// 				else
// 					idxMx = ((idx - step) << 1);
//
// 				if (X[1] == 0)
// 				{
// 					idxMy = ((idx + Sf - XC) << 1);
// 					idxPy = ((idx + XC) << 1);
//
// 					mPy = opCode(sub_ps, opCode(load_ps, &m[idxPy]), mel);
// #if	defined(__AVX512F__)
// 					mMy = opCode(sub_ps, mel, opCode(permutexvar_ps, vShRg, opCode(load_ps, &m[idxMy])));
// #elif	defined(__AVX2__)
// 					mMy = opCode(sub_ps, mel, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 6,7,0,1,2,3,4,5)));
// #elif	defined(__AVX__)
// 					mMx = opCode(permute_ps, opCode(load_ps, &m[idxMy]), 0b01001110);
// 					mMz = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
// 					mMy = opCode(sub_ps, mel, opCode(blend_ps, mMx, mMz, 0b00110011));
// #else
// 					mMx = opCode(load_ps, &m[idxMy]);
// 					mMy = opCode(sub_ps, mel, opCode(shuffle_ps, mMx, mMx, 0b01001110));
// #endif
// 				}
// 				else
// 				{
// 					idxMy = ((idx - XC) << 1);
//
// 					mMy = opCode(sub_ps, mel, opCode(load_ps, &m[idxMy]));
//
// 					if (X[1] == YC-1)
// 					{
// 						idxPy = ((idx - Sf + XC) << 1);
// #if	defined(__AVX512F__)
// 						mPy = opCode(sub_ps, opCode(permutexvar_ps, vShLf, opCode(load_ps, &m[idxPy])), mel);
// #elif	defined(__AVX2__)
// 						mPy = opCode(sub_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 2,3,4,5,6,7,0,1)), mel);
// #elif	defined(__AVX__)
// 						mMx = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b01001110);
// 						mMz = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
// 						mPy = opCode(sub_ps, opCode(blend_ps, mMx, mMz, 0b11001100), mel);
// #else
// 						mMx = opCode(load_ps, &m[idxPy]);
// 						mPy = opCode(sub_ps, opCode(shuffle_ps, mMx, mMx, 0b01001110), mel);
// #endif
// 					}
// 					else
// 					{
// 						idxPy = ((idx + XC) << 1);
// 						mPy = opCode(sub_ps, opCode(load_ps, &m[idxPy]), mel);
// 					}
// 				}
//
// // Start gradient e computation
// 				mPx = opCode(sub_ps, opCode(load_ps, &m[idxPx]), mel);
// 				mMx = opCode(sub_ps, mel, opCode(load_ps, &m[idxMx]));
// // mPy, mMy already loaded
// 				mPz = opCode(sub_ps, opCode(load_ps, &m[idxPz]), mel);
// 				mMz = opCode(sub_ps, mel, opCode(load_ps, &m[idxMz]));
// // load field velocity
// 				vel = opCode(load_ps, &v[idxV0]);
// // r^2, i^2...
// 				mod = opCode(mul_ps, mel, mel);
// // r^2+i^2, r^2+i^2  (m^2)
// 				mTp = opCode(md2_ps, mod);
// // Factor |mel|^2/z^2, util luego
// 				mod = opCode(mul_ps, mTp, ivZ2);
// // r/m^2, i/m^2
// 				mCg = opCode(div_ps, mel, mTp);	// Ahora mCg tiene 1/mel
//
// // Meto en mSg = shuffled(mCg) (Intercambio parte real por parte imaginaria)
// // i/m^2, r/m^2
// #if	defined(__AVX__)
// 				mSg = opCode(mul_ps, cjg, opCode(permute_ps, mCg, 0b10110001));
// #else
// 				mSg = opCode(mul_ps, cjg, opCode(shuffle_ps, mCg, mCg, 0b10110001));
// #endif
//
// // Gradients
// #if	defined(__AVX512F__)
// 				tGx = opCode(mul_ps, mPx, mCg);
// 				tGy = opCode(mul_ps, mPx, mSg);
//
// 				tGz = opCode(mask_add_ps,
// 					opCode(permute_ps, tGy, 0b10110001),
// 					opCode(kmov, 0b0101010101010101),
// 					opCode(permute_ps, tGx, 0b10110001),
// 					tGx);
//
// 				mdv = opCode(mask_add_ps, tGz, opCode(kmov, 0b1010101010101010), tGz, tGy);
//
// 				tGx = opCode(mul_ps, mMx, mCg);
// 				tGy = opCode(mul_ps, mMx, mSg);
//
// 				tGz = opCode(mask_add_ps,
// 					opCode(permute_ps, tGy, 0b10110001),
// 					opCode(kmov, 0b0101010101010101),
// 					opCode(permute_ps, tGx, 0b10110001),
// 					tGx);
//
// 				Grx = opCode(mask_add_ps, tGz, opCode(kmov, 0b1010101010101010), tGz, tGy);
// 				Grx = opCode(add_ps,
// 					opCode(mul_ps, mdv, mdv),
// 					opCode(mul_ps, Grx, Grx));
//
// 				tGx = opCode(mul_ps, mPy, mCg);
// 				tGy = opCode(mul_ps, mPy, mSg);
//
// 				tGz = opCode(mask_add_ps,
// 					opCode(permute_ps, tGy, 0b10110001),
// 					opCode(kmov, 0b0101010101010101),
// 					opCode(permute_ps, tGx, 0b10110001),
// 					tGx);
//
// 				mdv = opCode(mask_add_ps, tGz, opCode(kmov, 0b1010101010101010), tGz, tGy);
//
// 				tGx = opCode(mul_ps, mMy, mCg);
// 				tGy = opCode(mul_ps, mMy, mSg);
//
// 				tGz = opCode(mask_add_ps,
// 					opCode(permute_ps, tGy, 0b10110001),
// 					opCode(kmov, 0b0101010101010101),
// 					opCode(permute_ps, tGx, 0b10110001),
// 					tGx);
//
// 				Gry = opCode(mask_add_ps, tGz, opCode(kmov, 0b1010101010101010), tGz, tGy);
// 				Gry = opCode(add_ps,
// 					opCode(mul_ps, mdv, mdv),
// 					opCode(mul_ps, Gry, Gry));
//
// 				tGx = opCode(mul_ps, mPz, mCg);
// 				tGy = opCode(mul_ps, mPz, mSg);
//
// 				tGz = opCode(mask_add_ps,
// 					opCode(permute_ps, tGy, 0b10110001),
// 					opCode(kmov, 0b0101010101010101),
// 					opCode(permute_ps, tGx, 0b10110001),
// 					tGx);
//
// 				mdv = opCode(mask_add_ps, tGz, opCode(kmov, 0b1010101010101010), tGz, tGy);
//
// 				tGx = opCode(mul_ps, mMz, mCg);
// 				tGy = opCode(mul_ps, mMz, mSg);
//
// 				tGz = opCode(mask_add_ps,
// 					opCode(permute_ps, tGy, 0b10110001),
// 					opCode(kmov, 0b0101010101010101),
// 					opCode(permute_ps, tGx, 0b10110001),
// 					tGx);
//
// 				Grz = opCode(mask_add_ps, tGz, opCode(kmov, 0b1010101010101010), tGz, tGy);
// 				Grz = opCode(add_ps,
// 					opCode(mul_ps, mdv, mdv),
// 					opCode(mul_ps, Grz, Grz));
//
// 				tGx = opCode(mul_ps, vel, mCg);
// 				tGy = opCode(mul_ps, vel, mSg);
//
// 				tGz = opCode(mask_add_ps,
// 					opCode(permute_ps, tGy, 0b10110001),
// 					opCode(kmov, 0b0101010101010101),
// 					opCode(permute_ps, tGx, 0b10110001),
// 					tGx);
//
// 				mdv = opCode(sub_ps, opCode(mask_add_ps, tGz, opCode(kmov, 0b1010101010101010), tGz, tGy), ivZ);
// #elif	defined(__AVX__)
// 				Grx = opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, mPx, mCg), opCode(mul_ps, mPx, mSg)), 0b11011000);
// 				mdv = opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, mMx, mCg), opCode(mul_ps, mMx, mSg)), 0b11011000);
// 				Grx = opCode(add_ps,
// 					opCode(mul_ps, mdv, mdv),
// 					opCode(mul_ps, Grx, Grx));
//
// 				Gry = opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, mPy, mCg), opCode(mul_ps, mPy, mSg)), 0b11011000);
// 				mdv = opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, mMy, mCg), opCode(mul_ps, mMy, mSg)), 0b11011000);
// 				Gry = opCode(add_ps,
// 					opCode(mul_ps, mdv, mdv),
// 					opCode(mul_ps, Gry, Gry));
//
// 				Grz = opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, mPz, mCg), opCode(mul_ps, mPz, mSg)), 0b11011000);
// 				mdv = opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, mMz, mCg), opCode(mul_ps, mMz, mSg)), 0b11011000);
// 				Grz = opCode(add_ps,
// 					opCode(mul_ps, mdv, mdv),
// 					opCode(mul_ps, Grz, Grz));
//
// 				mdv = opCode(sub_ps, opCode(permute_ps, opCode(hadd_ps, opCode(mul_ps, vel, mCg), opCode(mul_ps, vel, mSg)), 0b11011000), ivZ);
// #else
// 				tKp = opCode(hadd_ps, opCode(mul_ps, vel, mCg), opCode(mul_ps, vel, mSg));
//
// 				tGx = opCode(hadd_ps, opCode(mul_ps, mPx, mCg), opCode(mul_ps, mPx, mSg));
// 				mdv = opCode(hadd_ps, opCode(mul_ps, mMx, mCg), opCode(mul_ps, mMx, mSg));
// 				tVp = opCode(shuffle_ps, tGx, tGx, 0b11011000);
// 				vel = opCode(shuffle_ps, mdv, mdv, 0b11011000);
// 				Grx = opCode(add_ps,
// 					opCode(mul_ps, vel, vel),
// 					opCode(mul_ps, tVp, tVp));
//
// 				tGy = opCode(hadd_ps, opCode(mul_ps, mPy, mCg), opCode(mul_ps, mPy, mSg));
// 				mdv = opCode(hadd_ps, opCode(mul_ps, mMy, mCg), opCode(mul_ps, mMy, mSg));
// 				tVp = opCode(shuffle_ps, tGy, tGy, 0b11011000);
// 				vel = opCode(shuffle_ps, mdv, mdv, 0b11011000);
// 				Gry = opCode(add_ps,
// 					opCode(mul_ps, vel, vel),
// 					opCode(mul_ps, tVp, tVp));
//
// 				tGz = opCode(hadd_ps, opCode(mul_ps, mPz, mCg), opCode(mul_ps, mPz, mSg));
// 				mdv = opCode(hadd_ps, opCode(mul_ps, mMz, mCg), opCode(mul_ps, mMz, mSg));
// 				tVp = opCode(shuffle_ps, tGz, tGz, 0b11011000);
// 				vel = opCode(shuffle_ps, mdv, mdv, 0b11011000);
// 				Grz = opCode(add_ps,
// 					opCode(mul_ps, vel, vel),
// 					opCode(mul_ps, tVp, tVp));
//
// 				mdv = opCode(sub_ps, opCode(shuffle_ps, tKp, tKp, 0b11011000), ivZ);
// #endif
//
// // Theta gradient energy times mod^2/z^2
// 				tGx = opCode(mul_ps, mod, Grx);
// 				tGy = opCode(mul_ps, mod, Gry);
// 				tGz = opCode(mul_ps, mod, Grz);
// // Theta kinetic energy times mod^2/z^2
// 				tKp = opCode(mul_ps, opCode(mul_ps, mdv, mdv), mod);
//
// // RE-S=Re', Im, ...
// 				Grx = opCode(sub_ps, mel, shVc);
// // (Re-s)^2,Im^2
// 				Gry = opCode(mul_ps, Grx, Grx);
// // (Re-s)^2+Im^2,(Re-s)^2+Im^2
// 				Grz = opCode(md2_ps, Gry);
// // (Re-s)^2+Im^2/z2,(Re-s)^2+Im^2/z2
// 				Gry = opCode(mul_ps, Grz, ivZ2);
// // Rho, Rho (duplicated)
// 				Frho = opCode(sqrt_ps, Gry);
//
// 				switch	(VQcd & VQCD_TYPE) {
//
// 					default:
// 					case	VQCD_1:
// 					case	VQCD_0:
// 						mSg = opCode(sub_ps, Gry, one);   // |rho|^2-1
// 						mod = opCode(mul_ps, mSg, mSg);   // (|rho|^2-1)^2
// 						mCg = opCode(sub_ps, one, opCode(div_ps, Grx, opCode(sqrt_ps, Grz)));  // 1-m/|m|  // 1-Re'/M , 1-Im/M
// 						break;
//
// 					case	VQCD_PQ_ONLY:
// 						mSg = opCode(sub_ps, Gry, one);   // |rho|^2-1
// 						mod = opCode(mul_ps, mSg, mSg);   // (|rho|^2-1)^2
// 						mCg = one ; // will be multiplied by 0
// 						break;
//
// 					case	VQCD_1_PQ_2:
// 						mSg = opCode(sub_ps, opCode(mul_ps, Gry, Gry), one);   	// |rho|^4-1
// 						mod = opCode(mul_ps, mSg, mSg);   											// (|rho|^4-1)^2
// 						mCg = opCode(sub_ps, one, opCode(div_ps, Grx, opCode(sqrt_ps, Grz)));  // 1-m/|m|
// 						break;
//
// 					case	VQCD_2:
// 						mSg = opCode(sub_ps, Gry, one);   // |rho|^2-1
// 						mod = opCode(mul_ps, mSg, mSg);   // (|rho|^2-1)^2
// 						// mTp = opCode(sub_ps, one, opCode(mul_ps, Grx, ivZ));  // 1-Re'/Z,     0
// 						// mCg = opCode(mul_ps, mTp, mTp);												// (1-Re'/Z)^2, 0
// 						// opCode(store_ps, tmpS, Grx);
// 						// printf("m %f %f %f %f\n",tmpS[0],tmpS[1],tmpS[2],tmpS[3]);
// 						// mTp = opCode(sub_ps, opCode(mul_ps,ivZ,opCode(set1_ps,zR)), opCode(div_ps, Grx, opCode(set1_ps,zR)));
// 						// opCode(store_ps, tmpS, mTp);
// 						// printf("a %f %f %f %f\n",tmpS[0],tmpS[1],tmpS[2],tmpS[3]);
// 						// mCg = opCode(mul_ps, hVec, opCode(md2_ps, opCode(mul_ps,mTp,mTp) ));
// 						// opCode(store_ps, tmpS, mCg);
// 						// printf("a %f %f %f %f\n\n",tmpS[0],tmpS[1],tmpS[2],tmpS[3]);
// 						mTp = opCode(sub_ps, qcd2, opCode(mul_ps, Grx, iiiZ));								// (1-Re'/Z), -Im/Z
// 						mCg = opCode(mul_ps, hVec, opCode(md2_ps, opCode(mul_ps,mTp,mTp) ));	// 0.5*((1-Re'/Z)^2+(Im/Z)^2), ...2
// 						break;
//
// 					case	VQCD_1N2:		//to be checked
// 						mSg = opCode(sub_ps, Gry, one);   // |rho|^2-1
// 						mod = opCode(mul_ps, mSg, mSg);   // (|rho|^2-1)^2
// 						// Gry = [Re^2, Im^2]
// 						// Grz = Mod^2, Mod^2...
// 						// ...   Im^2/Mod^22
// 						// ... = (1 - Gry/Grz)/2
// 						mCg = opCode(mul_ps, hVec,opCode(sub_ps, one, opCode(div_ps,Gry,Grz)));
// 					break;
//
// 				}
//
// 				// now combine axion and saxion V in one vector
// 				// mod = Sgood_1 shit Sgood_2 shit
// 				// mCg = Agood_1 shit Agood_2 shit
// 				// tVp = Sgood_1
// #if	defined(__AVX512F__)
// 				tVp = opCode(mask_blend_ps, opCode(kmov, 0b1010101010101010), mod, opCode(permute_ps, mCg, 0b10110001));
// #elif	defined(__AVX__)
// 				// permute > 1-Im/M , 1-Re'/M
// 				// blend   > (|rho|^2-1)^2, 1-Re'/M, ...
// 				tVp = opCode(blend_ps, mod, opCode(permute_ps, mCg, 0b10110001), 0b10101010);
// #else
// 				mdv = opCode(shuffle_ps, mod, mCg, 0b10001000); //Era 11011000  //  1-Im/M , 1-Re'/M
// 				tVp = opCode(shuffle_ps, mdv, mdv, 0b11011000); 								//  (|rho|^2-1)^2, 1-Re'/M, ...
// #endif
//
// // Copy energy map to m2
// 				if (emask & EN_MAP) {
// // Total energy
// 					mdv = opCode(add_ps,
// 						opCode(add_ps,
// 							opCode(mul_ps, tKp, hVec),
// 							opCode(mul_ps, opCode(add_ps, tGx, opCode(add_ps, tGy, tGz)), oVec)),
// 						opCode(mul_ps, tVp, pVec));
//
// 					//masked map?
// 					if (emask == EN_MAPMASK)
// 						mdv = opCode(mul_ps,Mask,mdv);
//
// 					opCode(store_ps, tmpS, mdv);
//
// 					#pragma unroll
// 					for (int ih=0; ih<step; ih++)
// 					{
// 						unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
// 						m2[iNx]    = tmpS[(ih<<1)+1]; // Theta field
// 						m2[iNx+Vt] = tmpS[(ih<<1)];   // Rho field
// 					}
// 				}
//
// //-----------------------------------------------------------------------------
// // TOTAL ENERGY
// //-----------------------------------------------------------------------------
//
// 			if (emask & EN_ENE){
// #if	defined(__AVX512F__)
// 				opCode(store_ps, tmpS, tGx);
// 				Gxrho += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
// 				Gxth  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15]);
//
// 				opCode(store_ps, tmpS, tGy);
// 				Gyrho += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
// 				Gyth  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15]);
//
// 				opCode(store_ps, tmpS, tGz);
// 				Gzrho += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
// 				Gzth  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15]);
//
// 				opCode(store_ps, tmpS, tVp);
// 				Vrho  += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
// 				Vth   += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15]);
//
// 				opCode(store_ps, tmpS, tKp);
// 				Krho  += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
// 				Kth   += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15]);
//
// 				opCode(store_ps, tmpS, Frho);
// 				Rrho  += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
// #elif defined(__AVX__)
// 				opCode(store_ps, tmpS, tGx);
// 				Gxrho += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
// 				Gxth  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7]);
//
// 				opCode(store_ps, tmpS, tGy);
// 				Gyrho += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
// 				Gyth  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7]);
//
// 				opCode(store_ps, tmpS, tGz);
// 				Gzrho += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
// 				Gzth  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7]);
//
// 				opCode(store_ps, tmpS, tVp);
// 				Vrho  += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
// 				Vth   += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7]);
//
// 				opCode(store_ps, tmpS, tKp);
// 				Krho  += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
// 				Kth   += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7]);
//
// 				opCode(store_ps, tmpS, Frho);
// 				Rrho  += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
// #else
// 				opCode(store_ps, tmpS, tGx);
// 				Gxrho += (double) (tmpS[0] + tmpS[2]);
// 				Gxth  += (double) (tmpS[1] + tmpS[3]);
//
// 				opCode(store_ps, tmpS, tGy);
// 				Gyrho += (double) (tmpS[0] + tmpS[2]);
// 				Gyth  += (double) (tmpS[1] + tmpS[3]);
//
// 				opCode(store_ps, tmpS, tGz);
// 				Gzrho += (double) (tmpS[0] + tmpS[2]);
// 				Gzth  += (double) (tmpS[1] + tmpS[3]);
//
// 				opCode(store_ps, tmpS, tVp);
// 				Vrho += (double) (tmpS[0] + tmpS[2]);
// 				Vth  += (double) (tmpS[1] + tmpS[3]);
//
// 				opCode(store_ps, tmpS, tKp);
// 				Krho += (double) (tmpS[0] + tmpS[2]);
// 				Kth  += (double) (tmpS[1] + tmpS[3]);
//
// 				opCode(store_ps, tmpS, Frho);
// 				Rrho += (double) (tmpS[0] + tmpS[2]);
// #endif
// }
//
// //-----------------------------------------------------------------------------
// // MASKED ENERGY
// //-----------------------------------------------------------------------------
//
// // if requested and required by having some element inside the mask!
// 			if ( (emask & EN_MASK) && (ups > 0) )
// 				{
// #if	defined(__AVX512F__)
// 								opCode(store_ps, tmpS, opCode(mul_ps,tGx,Mask));
// 								GxrhoM += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
// 								GxthM  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15]);
//
// 								opCode(store_ps, tmpS, opCode(mul_ps,tGy,Mask));
// 								GyrhoM += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
// 								GythM  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15]);
//
// 								opCode(store_ps, tmpS, opCode(mul_ps,tGz,Mask));
// 								GzrhoM += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
// 								GzthM  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15]);
//
// 								opCode(store_ps, tmpS, opCode(mul_ps,tVp,Mask));
// 								VrhoM  += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
// 								VthM   += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15]);
//
// 								opCode(store_ps, tmpS, opCode(mul_ps,tKp,Mask));
// 								KrhoM  += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
// 								KthM   += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7] + tmpS[9] + tmpS[11] + tmpS[13] + tmpS[15]);
//
// 								opCode(store_ps, tmpS, opCode(mul_ps,Frho,Mask));
// 								RrhoM  += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6] + tmpS[8] + tmpS[10] + tmpS[12] + tmpS[14]);
//
// #elif defined(__AVX__)
// 								opCode(store_ps, tmpS, opCode(mul_ps,tGx,Mask));
// 								GxrhoM += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
// 								GxthM  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7]);
//
// 								opCode(store_ps, tmpS, opCode(mul_ps,tGy,Mask));
// 								GyrhoM += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
// 								GythM  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7]);
//
// 								opCode(store_ps, tmpS, opCode(mul_ps,tGz,Mask));
// 								GzrhoM += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
// 								GzthM  += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7]);
//
// 								opCode(store_ps, tmpS, opCode(mul_ps,tVp,Mask));
// 								VrhoM  += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
// 								VthM   += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7]);
//
// 								opCode(store_ps, tmpS, opCode(mul_ps,tKp,Mask));
// 								KrhoM  += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
// 								KthM   += (double) (tmpS[1] + tmpS[3] + tmpS[5] + tmpS[7]);
//
// 								opCode(store_ps, tmpS, opCode(mul_ps,Frho,Mask));
// 								RrhoM  += (double) (tmpS[0] + tmpS[2] + tmpS[4] + tmpS[6]);
// #else
// 								opCode(store_ps, tmpS, opCode(mul_ps,tGx,Mask));
// 								GxrhoM += (double) (tmpS[0] + tmpS[2]);
// 								GxthM  += (double) (tmpS[1] + tmpS[3]);
//
// 								opCode(store_ps, tmpS, opCode(mul_ps,tGy,Mask));
// 								GyrhoM += (double) (tmpS[0] + tmpS[2]);
// 								GythM  += (double) (tmpS[1] + tmpS[3]);
//
// 								opCode(store_ps, tmpS, opCode(mul_ps,tGz,Mask));
// 								GzrhoM += (double) (tmpS[0] + tmpS[2]);
// 								GzthM  += (double) (tmpS[1] + tmpS[3]);
//
// 								opCode(store_ps, tmpS, opCode(mul_ps,tVp,Mask));
// 								VrhoM += (double) (tmpS[0] + tmpS[2]);
// 								VthM  += (double) (tmpS[1] + tmpS[3]);
//
// 								opCode(store_ps, tmpS, opCode(mul_ps,tKp,Mask));
// 								KrhoM += (double) (tmpS[0] + tmpS[2]);
// 								KthM  += (double) (tmpS[1] + tmpS[3]);
//
// 								opCode(store_ps, tmpS, opCode(mul_ps,Frho,Mask));
// 								RrhoM  += (double) (tmpS[0] + tmpS[2]);
// #endif
// 				} //end masked energy
// 			} //end for loop
// 		} //end parallel
//
// 		eRes[RH_GRX]  = Gxrho*o2;
// 		eRes[TH_GRX]  = Gxth *o2;
// 		eRes[RH_GRY]  = Gyrho*o2;
// 		eRes[TH_GRY]  = Gyth *o2;
// 		eRes[RH_GRZ]  = Gzrho*o2;
// 		eRes[TH_GRZ]  = Gzth *o2;
// 		eRes[RH_POT]  = Vrho *lZ;
// 		eRes[TH_POT]  = Vth  *zQ;
// 		eRes[RH_KIN]  = Krho *.5;
// 		eRes[TH_KIN]  = Kth  *.5;
// 		eRes[TH_KIN]  = Kth  *.5;
// 		eRes[RH_RHO]  = Rrho;
//
// 		eRes[RH_GRXM] = GxrhoM*o2;
// 		eRes[TH_GRXM] = GxthM *o2;
// 		eRes[RH_GRYM] = GyrhoM*o2;
// 		eRes[TH_GRYM] = GythM *o2;
// 		eRes[RH_GRZM] = GzrhoM*o2;
// 		eRes[TH_GRZM] = GzthM *o2;
// 		eRes[RH_POTM] = VrhoM *lZ;
// 		eRes[TH_POTM] = VthM  *zQ;
// 		eRes[RH_KINM] = KrhoM *.5;
// 		eRes[TH_KINM] = KthM  *.5;
// 		eRes[RH_RHOM] = RrhoM;
//
// 		eRes[MM_NUMM] = nummask;
//
//
// //LogOut("Energy %f %f %f %f %f\n",  eRes[RH_GRX],  eRes[RH_GRY],  eRes[RH_GRZ],  eRes[RH_KIN],  eRes[RH_POT]);
// //LogOut("Energy %f %f %f %f %f\n",  eRes[RH_GRXM],  eRes[RH_GRYM],  eRes[RH_GRZM],  eRes[RH_KINM],  eRes[RH_POTM]);
// //LogOut("Energy %f %f %f %f %f\n",  eRes[TH_GRX],  eRes[TH_GRY],  eRes[TH_GRZ],  eRes[TH_KIN],  eRes[TH_POT]);
// //LogOut("Energy %f %f %f %f %f\n",  eRes[TH_GRXM],  eRes[TH_GRYM],  eRes[TH_GRZM],  eRes[TH_KINM],  eRes[TH_POTM]);
//
// } //end loop single precision
// #undef	_MData_
// #undef	step
// }

void	energyNaxionCpu	(Scalar *axionField, const double delta2, const double aMass2, void *eRes, const bool map)
{
	PropParms ppar ;
	/* Energy computed with 1 neighbours even if Ng propagation. Some non-conservation expected! */
	ppar.Ng     = axionField->getNg();
	ppar.Lx     = axionField->Length();
	ppar.Vo     = ppar.Ng*axionField->Surf();
	ppar.Vf     = ppar.Vo + axionField->Size();
	ppar.ct     = *axionField->zV();
	ppar.R      = *axionField->RV();
	ppar.massA  = axionField->AxionMass();
	ppar.frw    = axionField->BckGnd()->Frw();
	/* Still requires to sum the grad p contribution
	includes the NR factor 1/2mc
	and the standard 1/(R^2 delta^2) of conformal ADM coordinates*/
	ppar.ood2a  = 0.25/ppar.massA/ppar.R/pow(ppar.R * axionField->BckGnd()->PhysSize()/axionField->Length(),2.);
	ppar.beta   = -1.0/(16.0*ppar.R*ppar.R);
	const FieldPrecision precision = axionField->Precision();

	// axionField->exchangeGhosts(FIELD_M);
	//
	// switch	(map) {
	// 	case	true:
	// 		energyNaxionKernelXeon<true>(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), ppar, precision, eRes);
	// 		break;
	//
	// 	case	false:
	// 		energyNaxionKernelXeon<false>(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), ppar, precision, eRes);
	// 		break;
	// }
}
