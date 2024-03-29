#include<cstdio>
#include<cmath>
#include"scalar/scalarField.h"
#include"enum-field.h"
//#include"scalar/varNQCD.h"

#include "utils/triSimd.h"
#include "utils/parse.h"

#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

#include <immintrin.h>

#if	defined(__AVX512F__)
	#define	Align 64
	#define	_PREFIX_ _mm512
	#define	_MInt_  __m512i
#else
	#if not defined(__AVX__) and not defined(__AVX2__)
		#define	Align 16
		#define	_PREFIX_ _mm
	#else
		#define	Align 32
		#define	_PREFIX_ _mm256
	#endif
#endif
/*
#define	bSizeX	1024
#define	bSizeY	64
#define	bSizeZ	2
*/
template<const VqcdType VQcd>
// inline	void	propagateKernelXeon(const void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_, size_t NN, double *R, const double dz, const double c, const double d,
// 				    const double ood2, const double LL, const double aMass2, const double gamma, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision,
// 				    const unsigned int bSizeX, const unsigned int bSizeY, const unsigned int bSizeZ)
inline	void	propagateKernelXeon(const void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_, PropParms ppar, const double dz, const double c, const double d,
				    const size_t Vo, const size_t Vf, FieldPrecision precision, const unsigned int bSizeX, const unsigned int bSizeY, const unsigned int bSizeZ)
{

	const size_t NN    = ppar.Ng;
	const size_t Lx    = ppar.Lx;
	const size_t Sf    = Lx*Lx;
	const size_t NSf   = Sf*NN;
	const double *PC   = ppar.PC;

	const double R     = ppar.R;
	const double ood2  = ppar.ood2a;
	const double mA2   = ppar.massA2;
	const double gamma = ppar.gamma;
	const double LL    = ppar.lambda;
	const double Rpp   = ppar.Rpp;
	const double Rp    = ppar.Rp ;
	const double deti  = ppar.dectime ;

	if (Vo>Vf)
		return ;

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

		const double * __restrict__ m	= (const double * __restrict__) __builtin_assume_aligned (m_, Align);
		double * __restrict__ v		= (double * __restrict__) __builtin_assume_aligned (v_, Align);
		double * __restrict__ m2	= (double * __restrict__) __builtin_assume_aligned (m2_, Align);

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
		double GGGG = gamma/R;
		if (deti > 0)
			GGGG *= R2/deti;
		const double mola = GGGG*dzc/2.;
		const double damp1 = 1./(1.+mola);
		const double damp2 = (1.-mola)*damp1;
		const double epsi = mola/(1.+mola);

		_MData_ COV[5];
		for (size_t nv = 0; nv < NN ; nv++)
			COV[nv]  = opCode(set1_pd, PC[nv]*ood2);

#if	defined(__AVX512F__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const double    __attribute__((aligned(Align))) zQAux[8] = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };	// Only real part
		const double    __attribute__((aligned(Align))) zNAux[8] = { zN,-zN, zN,-zN, zN,-zN, zN,-zN };	// to complex congugate
		const double    __attribute__((aligned(Align))) zRAux[8] = { R , 0., R , 0., R , 0., R , 0. };	// Only real part
		const double    __attribute__((aligned(Align))) cjgAux[8] = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
		const long long __attribute__((aligned(Align))) shfRg[8] = {6, 7, 0, 1, 2, 3, 4, 5 };
		const long long __attribute__((aligned(Align))) shfLf[8] = {2, 3, 4, 5, 6, 7, 0, 1 };

		const _MInt_  vShRg = opCode(load_si512, shfRg);
		const _MInt_  vShLf = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) zQAux[4] = { zQ, 0., zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zNAux[4] = { zN,-zN, zN,-zN };	// to complex congugate
		const double __attribute__((aligned(Align))) zRAux[4] = { R , 0., R , 0. };	// Only real part
		const double __attribute__((aligned(Align))) cjgAux[4] = { 1.,-1., 1.,-1. };
#else
		const size_t XC = Lx;
		const size_t YC = Lx;

		const double __attribute__((aligned(Align))) zQAux[2] = { zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zNAux[2] = { zN,-zN };	// to complex congugate
		const double __attribute__((aligned(Align))) zRAux[2] = { R , 0. };	// Only real part
		const double __attribute__((aligned(Align))) cjgAux[2] = { 1.,-1. };
#endif
		const _MData_ zQVec  = opCode(load_pd, zQAux);
		const _MData_ zNVec  = opCode(load_pd, zNAux);
		const _MData_ zRVec  = opCode(load_pd, zRAux);
		const _MData_ cjg    = opCode(load_pd, cjgAux);

		const uint z0 = Vo/(Lx*Lx);
		const uint zF = Vf/(Lx*Lx);
		const uint zM = (zF-z0+bSizeZ-1)/bSizeZ;
		const uint bY = (YC + bSizeY - 1)/bSizeY;

		for (uint zT = 0; zT < zM; zT++)
		 for (uint yT = 0; yT < bY; yT++)
		  #pragma omp parallel default(shared)
		  {
		    _MData_ tmp, mel, mPx, mPy, mMx, lap, tmp2;
		    #pragma omp for collapse(3) schedule(static)
		    for (uint zz = 0; zz < bSizeZ; zz++) {
		     for (uint yy = 0; yy < bSizeY; yy++) {
		      for (uint xC = 0; xC < XC; xC += step) {
			uint zC = zz + bSizeZ*zT + z0;
			uint yC = yy + bSizeY*yT;

			size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0, idxV0;
			size_t idx = zC*(YC*XC) + yC*XC + xC;

			// If YC (or zF-z0) is not divisible by bSizeY (bSizeZ), there is a possibility of exceeding the assumed domain in the last block.
			// This may be avoided by adjusting bSizeY (bSizeZ) in tunePropagator.
			if ((yC >= YC) || (zC >= zF)) continue;
			{
				X[0] = xC;
				X[1] = yC;
			}

			idxP0 =  (idx << 1);
			idxV0 =  (idx-NSf) << 1;
			mel = opCode(load_pd, &m[idxP0]);
			lap = opCode(set1_pd, 0.0);


			for (size_t nv=1; nv < NN+1; nv++)
			{

					if (X[0] < nv*step)
						idxMx = ((idx + XC - nv*step) << 1);
					else
						idxMx = ((idx - nv*step) << 1);
					//x+
					if (X[0] + nv*step >= XC)
						idxPx = ((idx + nv*step - XC) << 1);
					else
						idxPx = ((idx + nv*step) << 1);

					if (X[1] < nv)
					{
						idxMy = ((idx + Sf - nv*XC) << 1);
						idxPy = ((idx + nv*XC) << 1);
#if	defined(__AVX512F__)
						tmp = opCode(add_pd, opCode(permutexvar_pd, vShRg, opCode(load_pd, &m[idxMy])), opCode(load_pd, &m[idxPy]));
#elif	defined(__AVX__)
						mPx = opCode(load_pd, &m[idxMy]);
						tmp = opCode(add_pd, opCode(permute2f128_pd, mPx, mPx, 0b00000001), opCode(load_pd, &m[idxPy]));
#else
						tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
#endif
					}
					else
					{
						idxMy = ((idx - nv*XC) << 1);

						if (X[1] + nv >= YC)
						{
							idxPy = ((idx + nv*XC - Sf) << 1);
#if	defined(__AVX512F__)
							tmp = opCode(add_pd, opCode(permutexvar_pd, vShLf, opCode(load_pd, &m[idxPy])), opCode(load_pd, &m[idxMy]));
#elif	defined(__AVX__)
							mPx = opCode(load_pd, &m[idxPy]);
							tmp = opCode(add_pd, opCode(permute2f128_pd, mPx, mPx, 0b00000001), opCode(load_pd, &m[idxMy]));
#else
							tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
#endif
						}
						else
						{
							idxPy = ((idx + nv*XC) << 1);
							tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
						}
					}// end Y+Y-

					// add X+ X-
					tmp = opCode(add_pd,tmp,opCode(add_pd, opCode(load_pd, &m[idxPx]), opCode(load_pd, &m[idxMx])));
					// add Z+ Z-
					idxPz = ((idx+nv*Sf) << 1);
					idxMz = ((idx-nv*Sf) << 1);
					tmp = opCode(add_pd,tmp,opCode(add_pd,opCode(load_pd, &m[idxMz]),opCode(load_pd, &m[idxPz])));

					tmp = opCode(add_pd,tmp,opCode(mul_pd, mel,opCode(set1_pd, -6.0)));

					tmp = opCode(mul_pd,tmp, COV[nv-1]);
					lap = opCode(add_pd,lap,tmp);

				} //end neighbour loop nv

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
			if (VQcd & V_EVOL_THETA)
				mMx = lap;
			else
				switch	(VQcd & V_PQ) {
					case V_PQ1:
						mMx = opCode(sub_pd, lap,
										opCode(mul_pd, mel,
											opCode(sub_pd,
												opCode(mul_pd, opCode(sub_pd, mPx, opCode(set1_pd, R2)),
													opCode(set1_pd, LL)),
												opCode(set1_pd, Rpp))));
					break;
					case V_PQ3:
						// a = p^2 - R^2
						tmp2 = opCode(sub_pd, mPx, opCode(set1_pd, R2));
						// b = 2R^2(1-R/p)
						tmp  = opCode(mul_pd, opCode(set1_pd, 2*R2),
										opCode(sub_pd,opCode(set1_pd, 1.0),
											opCode(div_pd,opCode(set1_pd, R),
												opCode(sqrt_pd, mPx))));
						// tmp(cuadratic) if p > R and tmp2(quartic) if p < R
						tmp = opCode(kkk_pd,tmp,tmp2,mPx,opCode(set1_pd, R2));

						mMx = opCode(sub_pd, lap,
										opCode(mul_pd, mel,
											opCode(sub_pd,
												opCode(mul_pd, tmp,
													opCode(set1_pd, LL)),
												opCode(set1_pd, Rpp))));
					break;
					case V_PQ2:
						mMx = opCode(sub_pd, lap,
										opCode(mul_pd, mel,
											opCode(sub_pd,
												opCode(mul_pd,
													opCode(sub_pd, opCode(mul_pd, mPx, mPx), opCode(set1_pd, R4)),
														opCode(mul_pd, mPx, opCode(set1_pd, LaLa))),
											opCode(set1_pd, Rpp))));
					break;
				}
			/* mMx = mMx + VQCD part */
			if ( !(VQcd & V_EVOL_RHO) )
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
					case V_QCDL:
					/* Compute explicitly each arctan */
					break;

					default:
					case V_QCD0:
					break;
				}

			mPy = opCode(load_pd, &v[idxV0]);

			/* Proyect accelerations (mMx) and velocities (mPy) if needed */

			if (VQcd & V_EVOL_THETA)
			{
#if	defined(__AVX__)// || defined(__AVX512F__)
				//0.-(-mi mr)
				lap = opCode(permute_pd, opCode(mul_pd, mel, cjg), 0b01010101);
				//1.- (ar ai)*(-mi mr) = (-ar*mi ai*mr)
				auto vecmv = opCode(mul_pd, mMx, lap);
				//2.- (-ar*mi ai*mr, -ar*mi ai*mr)
				auto vecma = opCode(add_pd, opCode(permute_pd, vecmv, 0b01010101), vecmv);
				//3.- (vr vi)*(-mi mr) = (-vr*mi vi*mr)
				vecmv = opCode(mul_pd, mPy, lap);
				//4.- (-vr*mi vi*mr, -vr*mi vi*mr)
				vecmv = opCode(add_pd, opCode(permute_pd, vecmv, 0b01010101), vecmv);
#else
				lap = opCode(mul_pd, mel, cjg);
				lap = opCode(shuffle_pd, lap, lap, 0b00000001);
				auto vecmv = opCode(mul_pd, mMx, lap);
				auto vecma = opCode(add_pd, opCode(shuffle_pd, vecmv, vecmv, 0b00000001), vecmv);
				vecmv = opCode(mul_pd, mPy, lap);
				vecmv = opCode(add_pd, opCode(shuffle_pd, vecmv, vecmv, 0b00000001), vecmv);
#endif
				//5.- (-ar*mi ai*mr, -ar*mi ai*mr)*(-mi mr)/|m|^2 + R''/R (mr mi)
				mMx   = opCode(add_pd,
									opCode(div_pd, opCode(mul_pd, lap, vecma), mPx),
										opCode(mul_pd, mel, opCode(set1_pd, Rpp)));
				//6.- (-vr*mi vi*mr, -vr*mi vi*mr)*(-mi mr)/|m|^2 + R/R (mr mi)
				mPy   = opCode(add_pd,
									opCode(div_pd, opCode(mul_pd, lap, vecmv), mPx),
										opCode(mul_pd, mel, opCode(set1_pd, Rp)));
			}

			switch	(VQcd & V_DAMP) {

				default:
				case	V_NONE:
#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_pd, mMx, opCode(set1_pd, dzc), mPy);
#else
				tmp = opCode(add_pd, mPy, opCode(mul_pd, mMx, opCode(set1_pd, dzc)));
#endif
				break;

				case	V_DAMP_RHO:
				{
					//New implementation
					tmp = opCode(mul_pd, mel, mPy);
#if	defined(__AVX__)// || defined(__AVX512F__)
					auto vecmv = opCode(add_pd, opCode(permute_pd, tmp, 0b00000101), tmp);
#else
					auto vecmv = opCode(add_pd, opCode(shuffle_pd, tmp, tmp, 0b00000001), tmp);
#endif

					// vecma = MA
					// mel = M, mMx = A
					tmp = opCode(mul_pd, mel, mMx);
#if	defined(__AVX__)// || defined(__AVX512F__)
					auto vecma = opCode(add_pd, opCode(permute_pd, tmp, 0b00000001), tmp);
#else
					auto vecma = opCode(add_pd, opCode(shuffle_pd, tmp, tmp, 0b00000001), tmp);
#endif

#if	defined(__AVX512F__) || defined(__FMA__)
// A*dzc + mPy - epsi (M/|M|^2)(2*(vecmv-|M|^2 R'/R) +vecma dzc)
tmp = opCode(sub_pd,
	opCode(fmadd_pd, mMx, opCode(set1_pd, dzc), mPy),
	opCode(mul_pd, opCode(mul_pd, opCode(set1_pd, epsi), opCode(div_pd, mel, mPx)),
		opCode(fmadd_pd, opCode(sub_pd,vecmv,opCode(mul_pd,mPx,opCode(set1_pd, Rp))), opCode(set1_pd, 2.0), opCode(mul_pd, vecma, opCode(set1_pd, dzc)))));
#else
tmp = opCode(sub_pd,
	opCode(add_pd, mPy, opCode(mul_pd, mMx, opCode(set1_pd, dzc))),
	opCode(mul_pd, opCode(mul_pd, opCode(set1_pd, epsi), opCode(div_pd, mel, mPx)),
		opCode(add_pd,
			opCode(mul_pd, opCode(sub_pd,vecmv,opCode(mul_pd,mPx,opCode(set1_pd, Rp))), opCode(set1_pd, 2.0)),
			opCode(mul_pd, vecma, opCode(set1_pd, dzc)))));
#endif
				}
				break;

				case	V_DAMP_ALL:
#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_pd, mPy, opCode(set1_pd, damp2), opCode(mul_pd, mMx, opCode(set1_pd, damp1*dzc)));
#else
				tmp = opCode(add_pd, opCode(mul_pd, mPy, opCode(set1_pd, damp2)), opCode(mul_pd, mMx, opCode(set1_pd, damp1*dzc)));
#endif
				break;
			}

			if (VQcd & V_EVOL_RHO)
			{
				auto vecmv = opCode(mul_pd, mel, tmp);
#if	defined(__AVX__)// || defined(__AVX512F__)
				auto vecma = opCode(add_pd, opCode(permute_pd, vecmv, 0b00000101), vecmv);
#else
				auto vecma = opCode(add_pd, opCode(shuffle_pd, vecmv, vecmv, 0b00000001), vecmv);
#endif
				tmp   = opCode(div_pd, opCode(mul_pd, mel, vecma), mPx);
			}

#if	defined(__AVX512F__) || defined(__FMA__)
			mPx = opCode(fmadd_pd, tmp, opCode(set1_pd, dzd), mel);
#else
			mPx = opCode(add_pd, mel, opCode(mul_pd, tmp, opCode(set1_pd, dzd)));
#endif
			opCode(store_pd,  &v[idxV0], tmp);
			opCode(stream_pd, &m2[idxP0], mPx);
		      }
		    }
		  }
		}
#undef	_MData_
#undef	step
	} else if (precision == FIELD_SINGLE) {
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

		const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_, Align);
		float * __restrict__ v		= (float * __restrict__) __builtin_assume_aligned (v_, Align);
		float * __restrict__ m2		= (float * __restrict__) __builtin_assume_aligned (m2_, Align);

		const float dzc = dz*c;
		const float dzd = dz*d;
		const float Rf  = (float) R;
		const float R2  = Rf*Rf;
		const float zQ = (float) (mA2*R2*Rf);

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

		const float R4 = R2*R2;
		const float LaLa = LL*2.f/R4;
		float GGGG = gamma/Rf;
		if (deti > 0)
			GGGG *= R2/deti;
		const float mola = GGGG*dzc/2.f;
		const float damp1 = 1.f/(1.f+mola);
		const float damp2 = (1.f-mola)*damp1;
		const float epsi = mola/(1.f+mola);

		_MData_ COV[5];
		for (size_t nv = 0; nv < NN ; nv++)
			COV[nv]  = opCode(set1_ps, PC[nv]*ood2);

#if	defined(__AVX512F__)
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) zQAux[16] = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zNAux[16] = { zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN };
		const float __attribute__((aligned(Align))) zRAux[16] = { Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f };
		// TEMPO
		const float __attribute__((aligned(Align))) cjgAux[16]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1. };

		const int   __attribute__((aligned(Align))) shfRg[16] = {14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
		const int   __attribute__((aligned(Align))) shfLf[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1};

		const _MInt_  vShRg  = opCode(load_si512, shfRg);
		const _MInt_  vShLf  = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) zQAux[8]  = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zNAux[8]  = { zN, -zN, zN, -zN, zN, -zN, zN, -zN };
		const float __attribute__((aligned(Align))) zRAux[8]  = { Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f };

		const float __attribute__((aligned(Align))) cjgAux[8]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const float __attribute__((aligned(Align))) zQAux[4]  = { zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zNAux[4]  = { zN, -zN, zN, -zN };
		const float __attribute__((aligned(Align))) zRAux[4]  = { Rf, 0.f, Rf, 0.f };

		const float __attribute__((aligned(Align))) cjgAux[4]  = { 1.,-1., 1.,-1. };
#endif
		const _MData_ zQVec  = opCode(load_ps, zQAux);
		const _MData_ zNVec  = opCode(load_ps, zNAux);
		const _MData_ zRVec  = opCode(load_ps, zRAux);
		const _MData_ cjg  = opCode(load_ps, cjgAux);

		const uint z0 = Vo/(Lx*Lx);
		const uint zF = Vf/(Lx*Lx);
		const uint zM = (zF-z0+bSizeZ-1)/bSizeZ;
		const uint bY = (YC + bSizeY - 1)/bSizeY;

LogMsg(VERB_PARANOID,"[pX] z0 %d zF %d zM %d bY %d bSizeZ %d bSizeY %d [NN %d]",z0, zF, zM, bY, bSizeZ, bSizeY, NN);LogFlush();

		for (uint zT = 0; zT < zM; zT++)
		 for (uint yT = 0; yT < bY; yT++)
		  #pragma omp parallel default(shared)
		  {
		    _MData_ tmp, mel, mPx, mPy, mMx, tmp2, lap;
		    #pragma omp for collapse(3) schedule(static)
		    for (uint zz = 0; zz < bSizeZ; zz++) {
		     for (uint yy = 0; yy < bSizeY; yy++) {
		      for (uint xC = 0; xC < XC; xC += step) {
			uint zC = zz + bSizeZ*zT + z0;
			uint yC = yy + bSizeY*yT;

			size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0, idxV0;
			size_t idx = zC*(YC*XC) + yC*XC + xC;

			if ((yC >= YC) || (zC >= zF)) continue;

			X[0] = xC;
			X[1] = yC;

			idxP0 =  (idx << 1);
			idxV0 =  (idx-NSf) << 1;
			mel = opCode(load_ps, &m[idxP0]);
			lap = opCode(set1_ps, 0.f);

			for (size_t nv=1; nv < NN+1; nv++)
			{
				if (X[0] < nv*step)
					idxMx = ((idx + XC - nv*step) << 1);
				else
					idxMx = ((idx - nv*step) << 1);
				//x+
				if (X[0] + nv*step >= XC)
					idxPx = ((idx - XC + nv*step) << 1);
				else
					idxPx = ((idx + nv*step) << 1);

				if (X[1] < nv )
				{
					idxMy = ((idx + Sf - nv*XC) << 1);
					idxPy = ((idx + nv*XC) << 1);

	#if	defined(__AVX512F__)
					tmp = opCode(add_ps, opCode(permutexvar_ps, vShRg, opCode(load_ps, &m[idxMy])), opCode(load_ps, &m[idxPy]));
	#elif	defined(__AVX2__)
					tmp = opCode(add_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 6,7,0,1,2,3,4,5)),  opCode(load_ps, &m[idxPy]));
	#elif	defined(__AVX__)
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
					idxMy = ((idx - nv*XC) << 1);

					if (X[1] + nv >= YC)
					{
						idxPy = ((idx + nv*XC - Sf) << 1);
	#if	defined(__AVX512F__)
						tmp = opCode(add_ps, opCode(permutexvar_ps, vShLf, opCode(load_ps, &m[idxPy])), opCode(load_ps, &m[idxMy]));
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
						idxPy = ((idx + nv*XC) << 1);
						tmp = opCode(add_ps, opCode(load_ps, &m[idxPy]), opCode(load_ps, &m[idxMy]));
					}
				} // end Y+Y-

				// add X+ X-
				tmp = opCode(add_ps,tmp,opCode(add_ps, opCode(load_ps, &m[idxPx]), opCode(load_ps, &m[idxMx])));

				// add Z+ Z-
				idxPz = ((idx+nv*Sf) << 1);
				idxMz = ((idx-nv*Sf) << 1);
				tmp = opCode(add_ps,tmp,opCode(add_ps,opCode(load_ps, &m[idxMz]),opCode(load_ps, &m[idxPz])));

				tmp = opCode(add_ps,tmp,opCode(mul_ps, mel,opCode(set1_ps, -6.f)));
				tmp = opCode(mul_ps,tmp, COV[nv-1]);
				lap = opCode(add_ps,lap,tmp);
			} //end neightbour loop


			mPy = opCode(mul_ps, mel, mel);		// M1^2 M2^2

			// mPx is M1^2+M2^2 M1^2+M2^2
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
			if (VQcd & V_EVOL_THETA)
				mMx = lap;
			else
				switch	(VQcd & V_PQ) {
					case V_PQ1:
						mMx = opCode(sub_ps, lap,
										opCode(mul_ps, mel,
											opCode(sub_ps,
												opCode(mul_ps, opCode(sub_ps, mPx, opCode(set1_ps, R2)),
													opCode(set1_ps, LL)),
												opCode(set1_ps, Rpp))));
					break;
					case V_PQ3:
						// a = p^2 - R^2
						tmp2 = opCode(sub_ps, mPx, opCode(set1_ps, R2));
						// b = 2R^2(1-R/p)
						tmp  = opCode(mul_ps, opCode(set1_ps, 2*R2),
										opCode(sub_ps,opCode(set1_ps, 1.f),
											opCode(div_ps,opCode(set1_ps, Rf),
												opCode(sqrt_ps, mPx))));
						// tmp(cuadratic) if p > R and tmp2(quartic) if p < R
						tmp = opCode(kkk_ps,tmp,tmp2,mPx,opCode(set1_ps, R2));

						mMx = opCode(sub_ps, lap,
										opCode(mul_ps, mel,
											opCode(sub_ps,
												opCode(mul_ps, tmp,
													opCode(set1_ps, LL)),
												opCode(set1_ps, Rpp))));
					break;
					case V_PQ2:
						mMx = opCode(sub_ps, lap,
										opCode(mul_ps, mel,
											opCode(sub_ps,
												opCode(mul_ps,
													opCode(sub_ps, opCode(mul_ps, mPx, mPx), opCode(set1_ps, R4)),
														opCode(mul_ps, mPx, opCode(set1_ps, LaLa))),
											opCode(set1_ps, Rpp))));
					break;
				}
			/* mMx = mMx + VQCD part */
			if ( !(VQcd & V_EVOL_RHO) )
				switch	(VQcd & V_QCD) {
					case V_QCD1:
						mMx = opCode(add_ps, mMx, zQVec);
					break;
					case V_QCDV:
						mMx = opCode(add_ps, mMx, opCode(mul_ps, opCode(set1_ps, zQ), opCode(sub_ps, zRVec, mel)));
					break;
					case V_QCD2:
						mMx = opCode(add_ps, mMx, opCode(mul_ps,zNVec,mel));
					break;
					case V_QCDC:
						tmp2 = opCode(div_ps,
										opCode(vqcd0_ps,mel),
											opCode(sqrt_ps, opCode(mul_ps, mPx, opCode(mul_ps, mPx, mPx) ) ) ); //
						mMx = opCode(add_ps, mMx, opCode(mul_ps, zNVec, tmp2));
					break;
					default:
					case V_QCDL:
						tmp2 = opCode(div_ps,
										opCode(vqcd0_ps,mel),
											opCode(sqrt_ps, opCode(mul_ps, mPx, opCode(mul_ps, mPx, mPx) ) ) ); //
						mMx = opCode(add_ps, mMx, opCode(mul_ps, zNVec, tmp2));
					break;
					case V_QCD0:
					break;
				}


			mPy = opCode(load_ps, &v[idxV0]);

			/* Proyect accelerations (mMx) and velocities (mPy) if needed */

			if (VQcd & V_EVOL_THETA)
			{
#if	defined(__AVX__)// || defined(__AVX512F__)
				//0.-(-mi mr)
				lap = opCode(permute_ps, opCode(mul_ps, mel, cjg), 0b10110001);
				//1.- (ar ai)*(-mi mr) = (-ar*mi ai*mr)
				auto vecmv = opCode(mul_ps, mMx, lap);
				//2.- (-ar*mi ai*mr, -ar*mi ai*mr)
				auto vecma = opCode(add_ps, opCode(permute_ps, vecmv, 0b10110001), vecmv);
				//3.- (vr vi)*(-mi mr) = (-vr*mi vi*mr)
				vecmv = opCode(mul_ps, mPy, lap);
				//4.- (-vr*mi vi*mr, -vr*mi vi*mr)
				vecmv = opCode(add_ps, opCode(permute_ps, vecmv, 0b10110001), vecmv);
#else
				lap = opCode(mul_ps, mel, cjg);
				lap = opCode(shuffle_ps, lap, lap, 0b10110001);
				auto vecmv = opCode(mul_ps, mMx, lap);
				auto vecma = opCode(add_ps, opCode(shuffle_ps, vecmv, vecmv, 0b10110001), vecmv);
				vecmv = opCode(mul_ps, mPy, lap);
				vecmv = opCode(add_ps, opCode(shuffle_ps, vecmv, vecmv, 0b10110001), vecmv);
#endif
				//5.- (-ar*mi ai*mr, -ar*mi ai*mr)*(-mi mr)/|m|^2 + R''/R (mr mi)
				mMx   = opCode(add_ps,
									opCode(div_ps, opCode(mul_ps, lap, vecma), mPx),
										opCode(mul_ps, mel, opCode(set1_ps, Rpp)));
				//6.- (-vr*mi vi*mr, -vr*mi vi*mr)*(-mi mr)/|m|^2 + R/R (mr mi)
				mPy   = opCode(add_ps,
									opCode(div_ps, opCode(mul_ps, lap, vecmv), mPx),
										opCode(mul_ps, mel, opCode(set1_ps, Rp)));
			}

			/* update velocities with/without damping */
			switch (VQcd & V_DAMP) {

				default:
				case	V_NONE:
#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_ps, mMx, opCode(set1_ps, dzc), mPy);
#else
			 	tmp = opCode(add_ps, mPy, opCode(mul_ps, mMx, opCode(set1_ps, dzc)));
#endif
				break;

				case	V_DAMP_RHO:
				{
// NEW implementation
// V = (V+Adt) - (epsi M/|M|^2)(2 MV+ MA*dt - 2 |M|^2/t)
// recall
// V=mPy     A=mMx    |M|^2=mPx      M=mel
// vecmv = MV
					tmp = opCode(mul_ps, mel, mPy);
#if	defined(__AVX__)// || defined(__AVX512F__)
					auto vecmv = opCode(add_ps, opCode(permute_ps, tmp, 0b10110001), tmp);
#else
					auto vecmv = opCode(add_ps, opCode(shuffle_ps, tmp, tmp, 0b10110001), tmp);
#endif

					// vecma = MA
					// mel = M, mMx = A
					tmp = opCode(mul_ps, mel, mMx);
#if	defined(__AVX__)// || defined(__AVX512F__)
					auto vecma = opCode(add_ps, opCode(permute_ps, tmp, 0b10110001), tmp);
#else
					auto vecma = opCode(add_ps, opCode(shuffle_ps, tmp, tmp, 0b10110001), tmp);
#endif

#if	defined(__AVX512F__) || defined(__FMA__)
// A*dzc + mPy - epsi (M/|M|^2)(2*(vecmv-|M|^2 R'/R) +vecma dzc)
tmp = opCode(sub_ps,
	opCode(fmadd_ps, mMx, opCode(set1_ps, dzc), mPy),
	opCode(mul_ps, opCode(mul_ps, opCode(set1_ps, epsi), opCode(div_ps, mel, mPx)),
		opCode(fmadd_ps, opCode(sub_ps,vecmv,opCode(mul_ps,mPx,opCode(set1_ps, Rp))), opCode(set1_ps, 2.f), opCode(mul_ps, vecma, opCode(set1_ps, dzc)))));
#else

tmp = opCode(sub_ps,
	opCode(add_ps, mPy, opCode(mul_ps, mMx, opCode(set1_ps, dzc))),
	opCode(mul_ps, opCode(mul_ps, opCode(set1_ps, epsi), opCode(div_ps, mel, mPx)),
		opCode(add_ps,
			opCode(mul_ps, opCode(sub_ps,vecmv,opCode(mul_ps,mPx,opCode(set1_ps, Rp))), opCode(set1_ps, 2.f)),
			opCode(mul_ps, vecma, opCode(set1_ps, dzc)))));
#endif
				}
				break;

				case	V_DAMP_ALL:
				// damping all directions implementation
#if	defined(__AVX512F__) || defined(__FMA__)
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
				//1.- (mr mi)*(vr vi) = (mr*vr vi*mi)
				auto vecmv = opCode(mul_ps, mel, tmp);
				//2.- (mr*vr+vi*mi, mr*vr+vi*mi)
#if	defined(__AVX__)// || defined(__AVX512F__)
				auto vecma = opCode(add_ps, opCode(permute_ps, vecmv, 0b10110001), vecmv);
#else
				auto vecma = opCode(add_ps, opCode(shuffle_ps, vecmv, vecmv, 0b10110001), vecmv);
#endif
				//3.- (mr*vr+vi*mi, mr*vr+vi*mi)*(mr,mi)/|m|^2 = [mr*vr+vi*mi]/|m|^2 * (mr,mi)
				tmp   = opCode(div_ps, opCode(mul_ps, mel, vecma), mPx);
			}

// 			if (VQcd & V_EVOL_THETA)
// 			{
// #if	defined(__AVX__)// || defined(__AVX512F__)
// 				//0.-(-mi mr)
// 				lap = opCode(permute_ps, opCode(mul_ps, mel, cjg), 0b10110001);
// 				//1.- (vr vi)*(-mi mr) = (-vr*mi vi*mr)
// 				auto vecmv = opCode(mul_ps, tmp, lap);
// 				//2.- (-vr*mi vi*mr, -vr*mi vi*mr)
// 				auto vecma = opCode(add_ps, opCode(permute_ps, vecmv, 0b10110001), vecmv);
// #else
// 				lap = opCode(mul_ps, mel, cjg);
// 				lap = opCode(shuffle_ps, lap, lap, 0b10110001);
// 				auto vecmv = opCode(mul_ps, tmp, lap);
// 				auto vecma = opCode(add_ps, opCode(shuffle_ps, vecmv, vecmv, 0b10110001), vecmv);
// #endif
// 				//3.- (-vr*mi vi*mr, -vr*mi vi*mr)*(-mi mr)/|m|^2 + R''/R (mr mi)
// 				tmp   = opCode(add_ps,
// 									opCode(div_ps, opCode(mul_ps, lap, vecma), mPx),
// 										opCode(mul_ps, mel, opCode(set1_ps, Rpp)));
// 				//4.- (mr mi) -> (1+R'/R')(mr mi) for the Phi-update
// 				mel  = opCode(mul_ps, mel, opCode(set1_ps, Rp1));
// 			}


#if	defined(__AVX512F__) || defined(__FMA__)
			mPx = opCode(fmadd_ps, tmp, opCode(set1_ps, dzd), mel);
#else
			mPx = opCode(add_ps, mel, opCode(mul_ps, tmp, opCode(set1_ps, dzd)));
#endif


			opCode(store_ps,  &v[idxV0], tmp);
			opCode(stream_ps, &m2[idxP0], mPx);	// Avoids cache thrashing
			}
		}
		}
		}
#undef	_MData_
#undef	step
	}
}





inline	void	updateMXeon(void * __restrict__ m_, const void * __restrict__ v_, const double dz, const double d, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision,
			    const unsigned int bSizeX, const unsigned int bSizeY, const unsigned int bSizeZ)
{
	const uint z0 = Vo/(Lx*Lx);
	const uint zF = Vf/(Lx*Lx);
	const uint zM = (zF-z0+bSizeZ-1)/bSizeZ;

	if (precision == FIELD_DOUBLE)
	{
	#if	defined(__AVX512F__)
		#define	_MData_ __m512d
		#define	step 4
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);
	#elif	defined(__AVX__)
		#define	_MData_ __m256d
		#define	step 2
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);
	#else
		#define	_MData_ __m128d
		#define	step 1
		const size_t XC = Lx;
		const size_t YC = Lx;
	#endif

		double * __restrict__ m		= (double * __restrict__) __builtin_assume_aligned (m_, Align);
		const double * __restrict__ v	= (const double * __restrict__) __builtin_assume_aligned (v_, Align);

		const double dzd = dz*d;

		const uint bY = (YC + bSizeY - 1)/bSizeY;

		for (uint zT = 0; zT < zM; zT++)
		 for (uint yT = 0; yT < bY; yT++)
		  #pragma omp parallel default(shared)
		  {
			register _MData_ mIn, vIn, tmp;
			register size_t idxM0, idxV0;

			#pragma omp for collapse(3) schedule(static)
			for (uint zz = 0; zz < bSizeZ; zz++) {
		 	  for (uint yy = 0; yy < bSizeY; yy++) {
			    for (uint xC = 0; xC < XC; xC += step) {
			      uint zC = zz + bSizeZ*zT + z0;
			      uint yC = yy + bSizeY*yT;

			      auto idx = zC*(YC*XC) + yC*XC + xC;

			      if ((yC >= YC) || (zC >= zF)) continue;

			      idxM0 =  idx       << 1;
			      idxV0 = (idx - Vo) << 1;

#if	defined(__AVX512F__) || defined(__FMA__)
			      vIn = opCode(load_pd, &v[idxV0]);
			      mIn = opCode(load_pd, &m[idxM0]);
			      tmp = opCode(fmadd_pd, opCode(set1_pd, dzd), vIn, mIn);
			      opCode(store_pd, &m[idxM0], tmp);
#else
			      mIn = opCode(load_pd, &m[idxM0]);
			      tmp = opCode(load_pd, &v[idxV0]);
			      vIn = opCode(mul_pd, opCode(set1_pd, dzd), tmp);
			      tmp = opCode(add_pd, mIn, vIn);
			      opCode(store_pd, &m[idxM0], tmp);
#endif
			    }
			  }
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
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);
	#elif	defined(__AVX__)
		#define	_MData_ __m256
		#define	step 4
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);
	#else
		#define	_MData_ __m128
		#define	step 2
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);
	#endif

		float * __restrict__ m		= (float * __restrict__) __builtin_assume_aligned (m_, Align);
		const float * __restrict__ v	= (const float * __restrict__) __builtin_assume_aligned (v_, Align);

		const float dzd = dz*d;
#if	defined(__AVX512F__)
//		const float __attribute__((aligned(Align))) dzdAux[16] = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd  };
#elif	defined(__AVX__)
//		const float __attribute__((aligned(Align))) dzdAux[8]  = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd };
#else
//		const float __attribute__((aligned(Align))) dzdAux[4]  = { dzd, dzd, dzd, dzd };
#endif
//		const _MData_ dzdVec = opCode(load_ps, dzdAux);

		const uint bY = (YC + bSizeY - 1)/bSizeY;

		for (uint zT = 0; zT < zM; zT++)
		 for (uint yT = 0; yT < bY; yT++)
		    #pragma omp parallel default(shared)
		    {
			register _MData_ mIn, vIn, tmp;
			register size_t idxM0, idxV0;

			#pragma omp for collapse(3) schedule(static)
			for (uint zz = 0; zz < bSizeZ; zz++) {
		 	  for (uint yy = 0; yy < bSizeY; yy++) {
			    for (uint xC = 0; xC < XC; xC += step) {
			      uint zC = zz + bSizeZ*zT + z0;
			      uint yC = yy + bSizeY*yT;

			      auto idx = zC*(YC*XC) + yC*XC + xC;

			      if ((yC >= YC) || (zC >= zF)) continue;

			      idxM0 =  idx       << 1;
			      idxV0 = (idx - Vo) << 1;

#if	defined(__AVX512F__) || defined(__FMA__)
			      vIn = opCode(load_ps, &v[idxV0]);
			      mIn = opCode(load_ps, &m[idxM0]);
			      tmp = opCode(fmadd_ps, opCode(set1_ps, dzd), vIn, mIn);
			      opCode(store_ps, &m[idxM0], tmp);
#else
			      vIn = opCode(load_ps, &v[idxV0]);
			      mIn = opCode(load_ps, &m[idxM0]);
			      tmp = opCode(add_ps, mIn, opCode(mul_ps, opCode(set1_ps, dzd), vIn));
			      opCode(store_ps, &m[idxM0], tmp);
#endif
			    }
			  }
			}
		    }
#undef	_MData_
#undef	step
	}
}


template<const VqcdType VQcd>
inline	void	updateVXeon(const void * __restrict__ m_, void * __restrict__ v_, PropParms ppar, const double dz, const double c,
				    const size_t Vo, const size_t Vf, FieldPrecision precision, const unsigned int bSizeX, const unsigned int bSizeY, const unsigned int bSizeZ)
{

	const size_t NN    = ppar.Ng;
	const size_t Lx    = ppar.Lx;
	const size_t Sf    = Lx*Lx;
	const size_t NSf   = Sf*NN;
	const double *PC   = ppar.PC;

	const double R     = ppar.R;
	const double ood2  = ppar.ood2a;
	const double mA2   = ppar.massA2;
	const double gamma = ppar.gamma;
	const double LL    = ppar.lambda;
	const double Rpp   = ppar.Rpp;
	const double Rp    = ppar.Rp ;
	const double deti  = ppar.dectime ;

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

		const double * __restrict__ m	= (const double * __restrict__) __builtin_assume_aligned (m_, Align);
		double * __restrict__ v		= (double * __restrict__) __builtin_assume_aligned (v_, Align);

		const double dzc = dz*c;
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
		double GGGG = gamma/R;
		if (deti > 0)
			GGGG *= R2/deti;
		const double mola = GGGG*dzc/2.;
		const double damp1 = 1./(1.+mola);
		const double damp2 = (1.-mola)*damp1;
		const double epsi = mola/(1.+mola);

		_MData_ COV[5];
		for (size_t nv = 0; nv < NN ; nv++)
			COV[nv]  = opCode(set1_pd, PC[nv]*ood2);

#if	defined(__AVX512F__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const double    __attribute__((aligned(Align))) zQAux[8] = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };	// Only real part
		const double    __attribute__((aligned(Align))) zNAux[8] = { zN,-zN, zN,-zN, zN,-zN, zN,-zN };	// to complex congugate
		const double    __attribute__((aligned(Align))) zRAux[8] = { R , 0., R , 0., R , 0., R , 0. };	// Only real part
		const long long __attribute__((aligned(Align))) shfRg[8] = {6, 7, 0, 1, 2, 3, 4, 5 };
		const long long __attribute__((aligned(Align))) shfLf[8] = {2, 3, 4, 5, 6, 7, 0, 1 };

		const _MInt_  vShRg = opCode(load_si512, shfRg);
		const _MInt_  vShLf = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) zQAux[4] = { zQ, 0., zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zNAux[4] = { zN,-zN, zN,-zN };	// to complex congugate
		const double __attribute__((aligned(Align))) zRAux[4] = { R , 0., R , 0. };	// Only real part
#else
		const size_t XC = Lx;
		const size_t YC = Lx;

		const double __attribute__((aligned(Align))) zQAux[2] = { zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zNAux[2] = { zN,-zN };	// to complex congugate
		const double __attribute__((aligned(Align))) zRAux[2] = { R , 0. };	// Only real part
#endif
		const _MData_ zQVec  = opCode(load_pd, zQAux);
		const _MData_ zNVec  = opCode(load_pd, zNAux);
		const _MData_ zRVec  = opCode(load_pd, zRAux);

		const uint z0 = Vo/(Lx*Lx);
		const uint zF = Vf/(Lx*Lx);
		const uint zM = (zF-z0+bSizeZ-1)/bSizeZ;
		const uint bY = (YC + bSizeY - 1)/bSizeY;

		for (uint zT = 0; zT < zM; zT++)
		 for (uint yT = 0; yT < bY; yT++)
		  #pragma omp parallel default(shared)
		  {
		    _MData_ tmp, mel, mPx, mPy, mMx, lap, tmp2;
		    #pragma omp for collapse(3) schedule(static)
		    for (uint zz = 0; zz < bSizeZ; zz++) {
		     for (uint yy = 0; yy < bSizeY; yy++) {
		      for (uint xC = 0; xC < XC; xC += step) {
			uint zC = zz + bSizeZ*zT + z0;
			uint yC = yy + bSizeY*yT;

			size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0, idxV0;
			size_t idx = zC*(YC*XC) + yC*XC + xC;

			if ((yC >= YC) || (zC >= zF)) continue;
			{
				X[0] = xC;
				X[1] = yC;
			}

			idxP0 =  (idx << 1);
			idxV0 =  (idx-NSf) << 1;
			mel = opCode(load_pd, &m[idxP0]);
			lap = opCode(set1_pd, 0.0);


			for (size_t nv=1; nv < NN+1; nv++)
			{

					if (X[0] < nv*step)
						idxMx = ((idx + XC - nv*step) << 1);
					else
						idxMx = ((idx - nv*step) << 1);
					//x+
					if (X[0] + nv*step >= XC)
						idxPx = ((idx - XC + nv*step) << 1);
					else
						idxPx = ((idx + nv*step) << 1);

					if (X[1] < nv)
					{
						idxMy = ((idx + Sf - nv*XC) << 1);
						idxPy = ((idx + nv*XC) << 1);
#if	defined(__AVX512F__)
						tmp = opCode(add_pd, opCode(permutexvar_pd, vShRg, opCode(load_pd, &m[idxMy])), opCode(load_pd, &m[idxPy]));
#elif	defined(__AVX__)
						mPx = opCode(load_pd, &m[idxMy]);
						tmp = opCode(add_pd, opCode(permute2f128_pd, mPx, mPx, 0b00000001), opCode(load_pd, &m[idxPy]));
#else
						tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
#endif
					}
					else
					{
						idxMy = ((idx - nv*XC) << 1);

						if (X[1] + nv >= YC)
						{
							idxPy = ((idx + nv*XC - Sf) << 1);
#if	defined(__AVX512F__)
							tmp = opCode(add_pd, opCode(permutexvar_pd, vShLf, opCode(load_pd, &m[idxPy])), opCode(load_pd, &m[idxMy]));
#elif	defined(__AVX__)
							mPx = opCode(load_pd, &m[idxPy]);
							tmp = opCode(add_pd, opCode(permute2f128_pd, mPx, mPx, 0b00000001), opCode(load_pd, &m[idxMy]));
#else
							tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
#endif
						}
						else
						{
							idxPy = ((idx + nv*XC) << 1);
							tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
						}
					}// end Y+Y-

					// add X+ X-
					tmp = opCode(add_pd,tmp,opCode(add_pd, opCode(load_pd, &m[idxPx]), opCode(load_pd, &m[idxMx])));
					// add Z+ Z-
					idxPz = ((idx+nv*Sf) << 1);
					idxMz = ((idx-nv*Sf) << 1);
					tmp = opCode(add_pd,tmp,opCode(add_pd,opCode(load_pd, &m[idxMz]),opCode(load_pd, &m[idxPz])));

					tmp = opCode(add_pd,tmp,opCode(mul_pd, mel,opCode(set1_pd, -6.0)));

					tmp = opCode(mul_pd,tmp, COV[nv-1]);
					lap = opCode(add_pd,lap,tmp);

				} //end neighbour loop nv

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
					mMx = opCode(sub_pd, lap,
									opCode(mul_pd, mel,
										opCode(sub_pd,
											opCode(mul_pd, opCode(sub_pd, mPx, opCode(set1_pd, R2)),
												opCode(set1_pd, LL)),
											opCode(set1_pd, Rpp))));
				break;
				case V_PQ3:
					// a = p^2 - R^2
					tmp2 = opCode(sub_pd, mPx, opCode(set1_pd, R2));
					// b = 2R^2(1-R/p)
					tmp  = opCode(mul_pd, opCode(set1_pd, 2*R2),
									opCode(sub_pd,opCode(set1_pd, 1.0),
										opCode(div_pd,opCode(set1_pd, R),
											opCode(sqrt_pd, mPx))));
					// tmp(cuadratic) if p > R and tmp2(quartic) if p < R
					tmp = opCode(kkk_pd,tmp,tmp2,mPx,opCode(set1_pd, R2));

					mMx = opCode(sub_pd, lap,
									opCode(mul_pd, mel,
										opCode(sub_pd,
											opCode(mul_pd, tmp,
												opCode(set1_pd, LL)),
											opCode(set1_pd, Rpp))));
				break;
				case V_PQ2:
					mMx = opCode(sub_pd, lap,
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
					mMx = opCode(add_pd, mMx, opCode(mul_pd,zNVec,mel));
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

			mPy = opCode(load_pd, &v[idxV0]);

			switch	(VQcd & V_DAMP) {

				default:
				case	V_NONE:
#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_pd, mMx, opCode(set1_pd, dzc), mPy);
#else
				tmp = opCode(add_pd, mPy, opCode(mul_pd, mMx, opCode(set1_pd, dzc)));
#endif
				break;

				case	V_DAMP_RHO:
				{
					//New implementation
					tmp = opCode(mul_pd, mel, mPy);
#if	defined(__AVX__)// || defined(__AVX512F__)
					auto vecmv = opCode(add_pd, opCode(permute_pd, tmp, 0b00000101), tmp);
#else
					auto vecmv = opCode(add_pd, opCode(shuffle_pd, tmp, tmp, 0b00000001), tmp);
#endif

					// vecma = MA
					// mel = M, mMx = A
					tmp = opCode(mul_pd, mel, mMx);
#if	defined(__AVX__)// || defined(__AVX512F__)
					auto vecma = opCode(add_pd, opCode(permute_pd, tmp, 0b00000001), tmp);
#else
					auto vecma = opCode(add_pd, opCode(shuffle_pd, tmp, tmp, 0b00000001), tmp);
#endif

#if	defined(__AVX512F__) || defined(__FMA__)
// A*dzc + mPy - epsi (M/|M|^2)(2*(vecmv-|M|^2 R'/R) +vecma dzc)
tmp = opCode(sub_pd,
	opCode(fmadd_pd, mMx, opCode(set1_pd, dzc), mPy),
	opCode(mul_pd, opCode(mul_pd, opCode(set1_pd, epsi), opCode(div_pd, mel, mPx)),
		opCode(fmadd_pd, opCode(sub_pd,vecmv,opCode(mul_pd,mPx,opCode(set1_pd, Rp))), opCode(set1_pd, 2.0), opCode(mul_pd, vecma, opCode(set1_pd, dzc)))));
#else
tmp = opCode(sub_pd,
	opCode(add_pd, mPy, opCode(mul_pd, mMx, opCode(set1_pd, dzc))),
	opCode(mul_pd, opCode(mul_pd, opCode(set1_pd, epsi), opCode(div_pd, mel, mPx)),
		opCode(add_pd,
			opCode(mul_pd, opCode(sub_pd,vecmv,opCode(mul_pd,mPx,opCode(set1_pd, Rp))), opCode(set1_pd, 2.0)),
			opCode(mul_pd, vecma, opCode(set1_pd, dzc)))));
#endif
				}
				break;

				case	V_DAMP_ALL:
#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_pd, mPy, opCode(set1_pd, damp2), opCode(mul_pd, mMx, opCode(set1_pd, damp1*dzc)));
#else
				tmp = opCode(add_pd, opCode(mul_pd, mPy, opCode(set1_pd, damp2)), opCode(mul_pd, mMx, opCode(set1_pd, damp1*dzc)));
#endif
				break;
			}

			if (VQcd & V_EVOL_RHO)
			{
				auto vecmv = opCode(mul_pd, mel, tmp);
#if	defined(__AVX__)// || defined(__AVX512F__)
				auto vecma = opCode(add_pd, opCode(permute_pd, vecmv, 0b00000101), vecmv);
#else
				auto vecma = opCode(add_pd, opCode(shuffle_pd, vecmv, vecmv, 0b00000001), vecmv);
#endif
				tmp   = opCode(div_pd, opCode(mul_pd, mel, vecma), mPx);
			}

			opCode(store_pd,  &v[idxV0], tmp);

		      }
		    }
		  }
		}
#undef	_MData_
#undef	step
	} else if (precision == FIELD_SINGLE) {
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

		const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_, Align);
		float * __restrict__ v		= (float * __restrict__) __builtin_assume_aligned (v_, Align);

		const float dzc = dz*c;
		const float Rf  = (float) R;
		const float R2  = Rf*Rf;
		const float zQ = (float) (mA2*R2*Rf);

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

		const float R4 = R2*R2;
		const float LaLa = LL*2.f/R4;
		float GGGG = gamma/Rf;
		if (deti > 0)
			GGGG *= R2/deti;
		const float mola = GGGG*dzc/2.f;
		const float damp1 = 1.f/(1.f+mola);
		const float damp2 = (1.f-mola)*damp1;
		const float epsi = mola/(1.f+mola);

		_MData_ COV[5];
		for (size_t nv = 0; nv < NN ; nv++)
			COV[nv]  = opCode(set1_ps, PC[nv]*ood2);

#if	defined(__AVX512F__)
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) zQAux[16] = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zNAux[16] = { zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN };
		const float __attribute__((aligned(Align))) zRAux[16] = { Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f };

		const int   __attribute__((aligned(Align))) shfRg[16] = {14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
		const int   __attribute__((aligned(Align))) shfLf[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1};

		const _MInt_  vShRg  = opCode(load_si512, shfRg);
		const _MInt_  vShLf  = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) zQAux[8]  = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zNAux[8]  = { zN, -zN, zN, -zN, zN, -zN, zN, -zN };
		const float __attribute__((aligned(Align))) zRAux[8]  = { Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f };
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const float __attribute__((aligned(Align))) zQAux[4]  = { zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zNAux[4]  = { zN, -zN, zN, -zN };
		const float __attribute__((aligned(Align))) zRAux[4]  = { Rf, 0.f, Rf, 0.f };
#endif
		const _MData_ zQVec  = opCode(load_ps, zQAux);
		const _MData_ zNVec  = opCode(load_ps, zNAux);
		const _MData_ zRVec  = opCode(load_ps, zRAux);

		const uint z0 = Vo/(Lx*Lx);
		const uint zF = Vf/(Lx*Lx);
		const uint zM = (zF-z0+bSizeZ-1)/bSizeZ;
		const uint bY = (YC + bSizeY - 1)/bSizeY;

LogMsg(VERB_PARANOID,"[pX] z0 %d zF %d zM %d bY %d bSizeZ %d bSizeY %d [NN %d]",z0, zF, zM, bY, bSizeZ, bSizeY, NN);LogFlush();

		for (uint zT = 0; zT < zM; zT++)
		 for (uint yT = 0; yT < bY; yT++)
		  #pragma omp parallel default(shared)
		  {
		    _MData_ tmp, mel, mPx, mPy, mMx, tmp2, lap;
		    #pragma omp for collapse(3) schedule(static)
		    for (uint zz = 0; zz < bSizeZ; zz++) {
		     for (uint yy = 0; yy < bSizeY; yy++) {
		      for (uint xC = 0; xC < XC; xC += step) {
			uint zC = zz + bSizeZ*zT + z0;
			uint yC = yy + bSizeY*yT;

			size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0, idxV0;
			size_t idx = zC*(YC*XC) + yC*XC + xC;

			if ((yC >= YC) || (zC >= zF)) continue;

			X[0] = xC;
			X[1] = yC;

			idxP0 =  (idx << 1);
			idxV0 =  (idx-NSf) << 1;
			mel = opCode(load_ps, &m[idxP0]);
			lap = opCode(set1_ps, 0.f);

			for (size_t nv=1; nv < NN+1; nv++)
			{
				if (X[0] < nv*step)
					idxMx = ((idx + XC - nv*step) << 1);
				else
					idxMx = ((idx - nv*step) << 1);
				//x+
				if (X[0] + nv*step >= XC)
					idxPx = ((idx - XC + nv*step) << 1);
				else
					idxPx = ((idx + nv*step) << 1);

				if (X[1] < nv )
				{
					idxMy = ((idx + Sf - nv*XC) << 1);
					idxPy = ((idx + nv*XC) << 1);

	#if	defined(__AVX512F__)
					tmp = opCode(add_ps, opCode(permutexvar_ps, vShRg, opCode(load_ps, &m[idxMy])), opCode(load_ps, &m[idxPy]));
	#elif	defined(__AVX2__)
					tmp = opCode(add_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 6,7,0,1,2,3,4,5)),  opCode(load_ps, &m[idxPy]));
	#elif	defined(__AVX__)
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
					idxMy = ((idx - nv*XC) << 1);

					if (X[1] + nv >= YC)
					{
						idxPy = ((idx + nv*XC - Sf) << 1);
	#if	defined(__AVX512F__)
						tmp = opCode(add_ps, opCode(permutexvar_ps, vShLf, opCode(load_ps, &m[idxPy])), opCode(load_ps, &m[idxMy]));
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
						idxPy = ((idx + nv*XC) << 1);
						tmp = opCode(add_ps, opCode(load_ps, &m[idxPy]), opCode(load_ps, &m[idxMy]));
					}
				} // end Y+Y-

				// add X+ X-
				tmp = opCode(add_ps,tmp,opCode(add_ps, opCode(load_ps, &m[idxPx]), opCode(load_ps, &m[idxMx])));

				// add Z+ Z-
				idxPz = ((idx+nv*Sf) << 1);
				idxMz = ((idx-nv*Sf) << 1);
				tmp = opCode(add_ps,tmp,opCode(add_ps,opCode(load_ps, &m[idxMz]),opCode(load_ps, &m[idxPz])));

				tmp = opCode(add_ps,tmp,opCode(mul_ps, mel,opCode(set1_ps, -6.f)));
				tmp = opCode(mul_ps,tmp, COV[nv-1]);
				lap = opCode(add_ps,lap,tmp);
			} //end neightbour loop


			mPy = opCode(mul_ps, mel, mel);		// M1^2 M2^2

			// mPx is M1^2+M2^2 M1^2+M2^2
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
					mMx = opCode(sub_ps, lap,
									opCode(mul_ps, mel,
										opCode(sub_ps,
											opCode(mul_ps, opCode(sub_ps, mPx, opCode(set1_ps, R2)),
												opCode(set1_ps, LL)),
											opCode(set1_ps, Rpp))));
				break;
				case V_PQ3:
					// a = p^2 - R^2
					tmp2 = opCode(sub_ps, mPx, opCode(set1_ps, R2));
					// b = 2R^2(1-R/p)
					tmp  = opCode(mul_ps, opCode(set1_ps, 2*R2),
									opCode(sub_ps,opCode(set1_ps, 1.f),
										opCode(div_ps,opCode(set1_ps, Rf),
											opCode(sqrt_ps, mPx))));
					// tmp(cuadratic) if p > R and tmp2(quartic) if p < R
					tmp = opCode(kkk_ps,tmp,tmp2,mPx,opCode(set1_ps, R2));

					mMx = opCode(sub_ps, lap,
									opCode(mul_ps, mel,
										opCode(sub_ps,
											opCode(mul_ps, tmp,
												opCode(set1_ps, LL)),
											opCode(set1_ps, Rpp))));
				break;
				case V_PQ2:
					mMx = opCode(sub_ps, lap,
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
				case V_QCDL:
				case V_QCDC:
					tmp2 = opCode(div_ps,
									opCode(vqcd0_ps,mel),
										opCode(sqrt_ps, opCode(mul_ps, mPx, opCode(mul_ps, mPx, mPx) ) ) ); //
					mMx = opCode(add_ps, mMx, opCode(mul_ps, zNVec, tmp2));
				break;
				default:
				case V_QCD0:
				break;
			}


			mPy = opCode(load_ps, &v[idxV0]);

			switch (VQcd & V_DAMP) {

				default:
				case	V_NONE:
#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_ps, mMx, opCode(set1_ps, dzc), mPy);
#else
			 	tmp = opCode(add_ps, mPy, opCode(mul_ps, mMx, opCode(set1_ps, dzc)));
#endif
				break;

				case	V_DAMP_RHO:
				{
// NEW implementation
// V = (V+Adt) - (epsi M/|M|^2)(2 MV+ MA*dt - 2 |M|^2/t)
// recall
// V=mPy     A=mMx    |M|^2=mPx      M=mel
// vecmv = MV
					tmp = opCode(mul_ps, mel, mPy);
#if	defined(__AVX__)// || defined(__AVX512F__)
					auto vecmv = opCode(add_ps, opCode(permute_ps, tmp, 0b10110001), tmp);
#else
					auto vecmv = opCode(add_ps, opCode(shuffle_ps, tmp, tmp, 0b10110001), tmp);
#endif

					// vecma = MA
					// mel = M, mMx = A
					tmp = opCode(mul_ps, mel, mMx);
#if	defined(__AVX__)// || defined(__AVX512F__)
					auto vecma = opCode(add_ps, opCode(permute_ps, tmp, 0b10110001), tmp);
#else
					auto vecma = opCode(add_ps, opCode(shuffle_ps, tmp, tmp, 0b10110001), tmp);
#endif

#if	defined(__AVX512F__) || defined(__FMA__)
// A*dzc + mPy - epsi (M/|M|^2)(2*(vecmv-|M|^2 R'/R) +vecma dzc)
tmp = opCode(sub_ps,
	opCode(fmadd_ps, mMx, opCode(set1_ps, dzc), mPy),
	opCode(mul_ps, opCode(mul_ps, opCode(set1_ps, epsi), opCode(div_ps, mel, mPx)),
		opCode(fmadd_ps, opCode(sub_ps,vecmv,opCode(mul_ps,mPx,opCode(set1_ps, Rp))), opCode(set1_ps, 2.f), opCode(mul_ps, vecma, opCode(set1_ps, dzc)))));
#else

tmp = opCode(sub_ps,
	opCode(add_ps, mPy, opCode(mul_ps, mMx, opCode(set1_ps, dzc))),
	opCode(mul_ps, opCode(mul_ps, opCode(set1_ps, epsi), opCode(div_ps, mel, mPx)),
		opCode(add_ps,
			opCode(mul_ps, opCode(sub_ps,vecmv,opCode(mul_ps,mPx,opCode(set1_ps, Rp))), opCode(set1_ps, 2.f)),
			opCode(mul_ps, vecma, opCode(set1_ps, dzc)))));
#endif
				}
				break;

				case	V_DAMP_ALL:
				// damping all directions implementation
#if	defined(__AVX512F__) || defined(__FMA__)
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
#if	defined(__AVX__)// || defined(__AVX512F__)
				auto vecma = opCode(add_ps, opCode(permute_ps, vecmv, 0b10110001), vecmv);
#else
				auto vecma = opCode(add_ps, opCode(shuffle_ps, vecmv, vecmv, 0b10110001), vecmv);
#endif
				tmp   = opCode(div_ps, opCode(mul_ps, mel, vecma), mPx);
			}

			opCode(store_ps,  &v[idxV0], tmp);

		      }
		    }
		  }
		}
#undef	_MData_
#undef	step
	}
}



#undef	opCode
#undef	opCode_N
#undef	opCode_P
#undef	Align
#undef	_PREFIX_
