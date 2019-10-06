#include<cstdio>
#include<cmath>
#include"scalar/scalarField.h"
#include"enum-field.h"
//#include"scalar/varNQCD.h"
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

//----------------------------------------------------------------------------// FUN
//----------------------------------------------------------------------------//



//----------------------------------------------------------------------------//

// FIX ME FOR DOUBLE PRECISION!
template<const VqcdType VQcd>
inline	void	propagateNNKernelXeon(const void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_, double *R, const double dz, const double c, const double d,
				    const double ood2, const double LL, const double aMass2, const double gamma, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision,
				    const unsigned int bSizeX, const unsigned int bSizeY, const unsigned int bSizeZ)
{
	const size_t Sf = Lx*Lx;


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
		const double zR = *R;
		const double z2 = zR*zR;
		//const double zQ = 9.*pow(zR, nQcd+3.);
		const double zQ = aMass2*z2*zR;
		const double zN = aMass2*z2/2;

		const double z4 = z2*z2;
		const double LaLa = LL*2./z4;
		const double GGGG = pow(ood2,0.5)*gamma;
//		const double GGiZ = GGGG/zR;
		const double mola = GGGG*dzc/2.;
		const double damp1 = 1./(1.+mola);
		const double damp2 = (1.-mola)*damp1;
		const double epsi = mola/(1.+mola);

		double CO[5] = {0, 0, 0, 0, 0} ;
		if (Ng == 0) {
			return;
		}	else if (Ng == 1) {
			CO[0] = 1.  ;
		}	else if (Ng == 2) {
			CO[0] = 4./3.; CO[1] = -1./12.;
		} else if (Ng == 3) {
			CO[0] = 1.5    ; CO[1] = -3./20.0; CO[2] = 1./90. ;
		} else if (Ng == 4) {
			CO[0] = 1.6    ; CO[1] = -0.2    ; CO[2] = 8./315. ; CO[3] = -1./560. ;
		} else if (Ng == 5) {
			CO[0] = 5./3.  ; CO[1] = -5./21. ; CO[2] = 5./126. ; CO[3] = -5./1008. ; CO[4] = 1./3150. ;
	 	}
		_MData_ COV[5];
		for (size_t nv = 0; nv < Ng ; nv++)
			COV[nv]  = opCode(set1_pd, CO[nv]*ood2);

#if	defined(__AVX512F__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const double    __attribute__((aligned(Align))) zQAux[8] = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };	// Only real part
		const double    __attribute__((aligned(Align))) zNAux[8] = { zN,-zN, zN,-zN, zN,-zN, zN,-zN };	// to complex congugate
		const double    __attribute__((aligned(Align))) zRAux[8] = { zR, 0., zR, 0., zR, 0., zR, 0. };	// Only real part
		const long long __attribute__((aligned(Align))) shfRg[8] = {6, 7, 0, 1, 2, 3, 4, 5 };
		const long long __attribute__((aligned(Align))) shfLf[8] = {2, 3, 4, 5, 6, 7, 0, 1 };

		const _MInt_  vShRg = opCode(load_si512, shfRg);
		const _MInt_  vShLf = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) zQAux[4] = { zQ, 0., zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zNAux[4] = { zN,-zN, zN,-zN };	// to complex congugate
		const double __attribute__((aligned(Align))) zRAux[4] = { zR, 0., zR, 0. };	// Only real part
#else
		const size_t XC = Lx;
		const size_t YC = Lx;

		const double __attribute__((aligned(Align))) zQAux[2] = { zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zNAux[2] = { zN,-zN };	// to complex congugate
		const double __attribute__((aligned(Align))) zRAux[2] = { zR, 0. };	// Only real part
#endif
		const _MData_ zQVec  = opCode(load_pd, zQAux);
		const _MData_ zNVec  = opCode(load_pd, zNAux);
		const _MData_ zRVec  = opCode(load_pd, zRAux);

		const uint z0 = Vo/(Lx*Lx);
		const uint zF = Vf/(Lx*Lx);
		const uint zM = (zF-z0+bSizeZ-1)/bSizeZ;
		const uint bY = (YC + bSizeY - 1)/bSizeY;

LogMsg(VERB_PARANOID,"[propNN] Ng %d zM %d bY %d bSizeZ %d bSizeY %d XC %d",Ng,zM,bY,bSizeZ,bSizeY,XC ); // tuned chunks
		for (uint zT = 0; zT < zM; zT++)
		 for (uint yT = 0; yT < bY; yT++)
		  #pragma omp parallel default(shared)
		  {
		    _MData_ tmp, mel, mPx, mPy, mMx, lap;
		    #pragma omp for collapse(3) schedule(static)
		    for (uint zz = 0; zz < bSizeZ; zz++) {
		     for (uint yy = 0; yy < bSizeY; yy++) {
		      for (uint xC = 0; xC < XC; xC += step) {
			uint zC = zz + bSizeZ*zT + z0;
			uint yC = yy + bSizeY*yT;

			size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0, idxV0;
			size_t idxb = yC*XC + xC;
			size_t idx = zC*(YC*XC) + yC*XC + xC;

			if (idx >= Vf)
				continue;

			X[0] = xC;
			X[1] = yC;

			idxP0 =  (idx << 1);
			idxV0 = ((idx-Sf) << 1);
			mel = opCode(load_pd, &m[idxP0]);
			lap = opCode(set1_pd, 0.0);

			for (size_t nv=1; nv < Ng+1; nv++)
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
				tmp = opCode(add_pd,tmp,opCode(mul_pd, mel,opCode(set1_pd, -6.0)));

				if (zC < nv)
				{
					idxPz = ((idx+nv*Sf) << 1);
					tmp = opCode(add_pd,tmp,opCode(load_pd, &m[idxPz]));
				}
				else if (zC + nv > sizeZ+1)
				{
					idxMz = ((idx-nv*Sf) << 1);
					tmp = opCode(add_pd,tmp,opCode(load_pd, &m[idxMz]));
				}
				else
				{
					idxPz = ((idx+nv*Sf) << 1);
					idxMz = ((idx-nv*Sf) << 1);
					tmp = opCode(add_pd,tmp,opCode(add_pd,opCode(load_pd, &m[idxMz]),opCode(load_pd, &m[idxPz])));
				}

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

			switch	(VQcd & VQCD_TYPE) {
				default:
				case	VQCD_1:
				mMx = opCode(sub_pd,
					opCode(add_pd, lap, zQVec),
					opCode(mul_pd,
						opCode(mul_pd,
							opCode(sub_pd, mPx, opCode(set1_pd, z2)),
							opCode(set1_pd, LL)),
											 mel));
				break;

				case	VQCD_1_PQ_2:
				mMx = opCode(sub_pd,
					opCode(add_pd, lap, zQVec),
						opCode(mul_pd,
							opCode(mul_pd,
								opCode(sub_pd, opCode(mul_pd, mPx, mPx), opCode(set1_pd, z4)),
								opCode(mul_pd, mPx, opCode(set1_pd, LaLa))),
							mel));
				 break;

				//FIX ME ADDITIONAL POTENTIALS!
				case	VQCD_2:
				mMx = opCode(sub_pd,
					opCode(sub_pd,
						opCode(mul_pd,
							opCode(add_pd,
								opCode(add_pd,
									opCode(load_pd, &m[idxMz]),
									opCode(add_pd,
										opCode(add_pd,
											opCode(add_pd, tmp, opCode(load_pd, &m[idxPx])),
											opCode(load_pd, &m[idxMx])),
										opCode(load_pd, &m[idxPz]))),
								opCode(mul_pd, mel, opCode(set1_pd, -6.0))),
							opCode(set1_pd, ood2)),
						opCode(mul_pd, opCode(set1_pd, zQ), opCode(sub_pd, mel, zRVec))),
					opCode(mul_pd,
						opCode(mul_pd,
							opCode(sub_pd, mPx, opCode(set1_pd, z2)),
							opCode(set1_pd, LL)),
						mel));
				break;

				case	VQCD_1N2:
				mMx = opCode(sub_pd,
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
								opCode(mul_pd, mel, opCode(set1_pd, -6.0))),
							opCode(set1_pd, ood2)),
							// 1N2 part
						opCode(mul_pd,zNVec,mel)),
					opCode(mul_pd,
						opCode(mul_pd,
							opCode(sub_pd, mPx, opCode(set1_pd, z2)),
							opCode(set1_pd, LL)),
						mel));
				break;

			}

			mPy = opCode(load_pd, &v[idxMz]);

			switch	(VQcd & VQCD_DAMP) {

				default:
				case	VQCD_NONE:
#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_pd, mMx, opCode(set1_pd, dzc), mPy);
#else
				tmp = opCode(add_pd, mPy, opCode(mul_pd, mMx, opCode(set1_pd, dzc)));
#endif
				break;

				case	VQCD_DAMP_RHO:
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
// A*dzc + mPy - epsi (M/|M|^2)(2*(vecmv-|M|^2/t) +vecma dzc)
tmp = opCode(sub_pd,
	opCode(fmadd_pd, mMx, opCode(set1_pd, dzc), mPy),
	opCode(mul_pd, opCode(mul_pd, opCode(set1_pd, epsi), opCode(div_pd, mel, mPx)),
		opCode(fmadd_pd, opCode(sub_pd,vecmv,opCode(div_pd,mPx,opCode(set1_pd, zR))), opCode(set1_pd, 2.0), opCode(mul_pd, vecma, opCode(set1_pd, dzc)))));
#else
tmp = opCode(sub_pd,
	opCode(add_pd, mPy, opCode(mul_pd, mMx, opCode(set1_pd, dzc))),
	opCode(mul_pd, opCode(mul_pd, opCode(set1_pd, epsi), opCode(div_pd, mel, mPx)),
		opCode(add_pd,
			opCode(mul_pd, opCode(sub_pd,vecmv,opCode(div_pd,mPx,opCode(set1_pd, zR))), opCode(set1_pd, 2.0)),
			opCode(mul_pd, vecma, opCode(set1_pd, dzc)))));
#endif
				}
				break;

				case	VQCD_DAMP_ALL:
#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_pd, mPy, opCode(set1_pd, damp2), opCode(mul_pd, mMx, opCode(set1_pd, damp1*dzc)));
#else
				tmp = opCode(add_pd, opCode(mul_pd, mPy, opCode(set1_pd, damp2)), opCode(mul_pd, mMx, opCode(set1_pd, damp1*dzc)));
#endif
				break;
			}

			if (VQcd & VQCD_EVOL_RHO)
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
			opCode(store_pd,  &v[idxMz], tmp);
			opCode(stream_pd, &m2[idxP0], mPx);
		      }
		    }
		  }
		}
#undef	_MData_
#undef	step
	}
	// ----------------------------------------------------------------------//
	else if (precision == FIELD_SINGLE) {
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
		const float zR = *R;
		const float z2 = zR*zR;
		//const float zQ = 9.*powf(zR, nQcd+3.);
		const float zQ = (float) (aMass2*z2*zR);
		//For VQCD_1N2
		const float zN = (float) (aMass2*z2)/2;

		const float z4 = z2*z2;
		const float LaLa = LL*2.f/z4;
		const float GGGG = pow(ood2,0.5)*gamma;
//		const float GGiZ = GGGG/zR;
		const float mola = GGGG*dzc/2.f;
		const float damp1 = 1.f/(1.f+mola);
		const float damp2 = (1.f-mola)*damp1;
		const float epsi = mola/(1.f+mola);

		float CO[5] = {0, 0, 0, 0, 0} ;
		if (Ng == 0) {
			return;
		}	else if (Ng == 1) {
			CO[0] = 1.  ;
		}	else if (Ng == 2) {
			CO[0] = 4./3.; CO[1] = -1./12.;
		} else if (Ng == 3) {
			CO[0] = 1.5    ; CO[1] = -3./20.0; CO[2] = 1./90. ;
		} else if (Ng == 4) {
			CO[0] = 1.6    ; CO[1] = -0.2    ; CO[2] = 8./315. ; CO[3] = -1./560. ;
		} else if (Ng == 5) {
			CO[0] = 5./3.  ; CO[1] = -5./21. ; CO[2] = 5./126. ; CO[3] = -5./1008. ; CO[4] = 1./3150. ;
	 	}
		_MData_ COV[5];
		for (size_t nv = 0; nv < Ng ; nv++)
			COV[nv]  = opCode(set1_ps, CO[nv]*ood2);

#if	defined(__AVX512F__)
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) zQAux[16] = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zNAux[16] = { zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN };
		const float __attribute__((aligned(Align))) zRAux[16] = { zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f };
		const int   __attribute__((aligned(Align))) shfRg[16] = {14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
		const int   __attribute__((aligned(Align))) shfLf[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1};

		const _MInt_  vShRg  = opCode(load_si512, shfRg);
		const _MInt_  vShLf  = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) zQAux[8]  = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zNAux[8]  = { zN, -zN, zN, -zN, zN, -zN, zN, -zN };
		const float __attribute__((aligned(Align))) zRAux[8]  = { zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f };
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const float __attribute__((aligned(Align))) zQAux[4]  = { zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zNAux[4]  = { zN, -zN, zN, -zN };
		const float __attribute__((aligned(Align))) zRAux[4]  = { zR, 0.f, zR, 0.f };
#endif
		const _MData_ zQVec  = opCode(load_ps, zQAux);
		const _MData_ zNVec  = opCode(load_ps, zNAux);
		const _MData_ zRVec  = opCode(load_ps, zRAux);

		const uint z0 = Vo/(Lx*Lx);
		const uint zF = Vf/(Lx*Lx);
		const uint zM = (zF-z0+bSizeZ-1)/bSizeZ;
		const uint bY = (YC + bSizeY - 1)/bSizeY;

		// LogOut("[prop] Ng %d Lz %d\n",Ng,sizeZ); // The Ng defition reaches here from parse.h
LogMsg(VERB_PARANOID,"[propNN] Ng %d zM %d bY %d bSizeZ %d bSizeY %d XC %d",Ng,zM,bY,bSizeZ,bSizeY,XC ); // tuned chunks
		for (uint zT = 0; zT < zM; zT++)
		 for (uint yT = 0; yT < bY; yT++)
		  #pragma omp parallel default(shared)
		  {
		    _MData_ tmp, mel, mPx, mPy, mMx, lap;
		    #pragma omp for collapse(3) schedule(static)
		    for (uint zz = 0; zz < bSizeZ; zz++) {
		     for (uint yy = 0; yy < bSizeY; yy++) {
		      for (uint xC = 0; xC < XC; xC += step) {
			uint zC = zz + bSizeZ*zT + z0;
			uint yC = yy + bSizeY*yT;
			// LogOut("ca");
			size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0, idxV0;
			size_t idxb = yC*XC + xC;
			size_t idx = zC*(YC*XC) + yC*XC + xC;

			if (idx >= Vf)
				continue;

			// Central points of the stencil (folded coordinates)
			X[0] = xC;
			X[1] = yC;

			idxP0 =  (idx << 1);
			idxV0 = ((idx-Sf) << 1);
			mel = opCode(load_ps, &m[idxP0]);
			lap = opCode(set1_ps, 0.f);

// if (idxb==0) LogOut("r%d ghost %f %f \n",commRank(),m[idxP0-2*Sf],m[idxP0-2*Sf+1]);
			//Laplacian
			// Neighbour loop
			for (size_t nv=1; nv < Ng+1; nv++)
			{
				// LogOut("ca nv %d\n",nv);
				//x-
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
					//sum Y+ + Y-
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

// opCode(store_ps, &m2[idxP0], tmp);
// if (idxb==0) LogOut("r%d nv%d Ylap %f %f \n",commRank(),nv,m2[idxP0],m2[idxP0+1]);

					// add X+ X-
					tmp = opCode(add_ps,tmp,opCode(add_ps, opCode(load_ps, &m[idxPx]), opCode(load_ps, &m[idxMx])));
					// add -6 X0
// opCode(store_ps, &m2[idxP0], tmp);
// if (idxb==0) LogOut("r%d nv%d XYlap %f %f \n",commRank(),nv,m2[idxP0],m2[idxP0+1]);

					tmp = opCode(add_ps,tmp,opCode(mul_ps, mel,opCode(set1_ps, -6.f)));
// opCode(store_ps, &m2[idxP0], tmp);
// if (idxb==0) LogOut("r%d nv%d -6lap %f %f \n",commRank(),nv,m2[idxP0],m2[idxP0+1]);

					//z is defined from m[0] ghosts have to be taken into account
					// we will only start with zC = 1, never compute 0
					// zC = 1 and nv = 1 will automatically read 0ghost
					// zC = sizeZ will automatically read final-ghost
					if (zC < nv)
					{
						idxPz = ((idx+nv*Sf) << 1);
						//idxMz included in nv=ZC
						tmp = opCode(add_ps,tmp,opCode(load_ps, &m[idxPz]));
// opCode(store_ps, &m2[idxP0], tmp);
// if (idxb==0) LogOut("shaved %f %f \n",m2[idxP0],m2[idxP0+1]);
					}
					else if (zC + nv > sizeZ+1)
					{
						idxMz = ((idx-nv*Sf) << 1);
						//idxPz included in nv=ZC
						tmp = opCode(add_ps,tmp,opCode(load_ps, &m[idxMz]));
// opCode(store_ps, &m2[idxP0], tmp);
// if (idxb==0) LogOut("shamed %f %f \n",m2[idxP0],m2[idxP0+1]);
					}
					else
					{
						idxPz = ((idx+nv*Sf) << 1);
						idxMz = ((idx-nv*Sf) << 1);
						tmp = opCode(add_ps,tmp,opCode(add_ps,opCode(load_ps, &m[idxMz]),opCode(load_ps, &m[idxPz])));
// opCode(store_ps, &m2[idxP0], tmp);
// if (idxb==0) LogOut("cute %f %f \n",m2[idxP0],m2[idxP0+1]);
					}
					// final laplacian assembly
// opCode(store_ps, &m2[idxP0], tmp);
// if (idxb==0) LogOut("nv%d YZXlap %f %f \n",nv,m2[idxP0],m2[idxP0+1]);
					tmp = opCode(mul_ps,tmp, COV[nv-1]);
					// if (idx==Sf) LogOut("r%d fufu 00b nv %d %f %f %f %f\n",commRank(),CO[nvbase+0],0,m[idxP0],m[idxP0+1],m[idxP1],m[idxP1+1]);
// opCode(store_ps, &m2[idxP0], tmp);
// if (idxb==0) LogOut("nv%d CYZXap %f %f (ood2 %f)\n",nv,m2[idxP0],m2[idxP0+1],ood2);
					lap = opCode(add_ps,lap,tmp);
// opCode(store_ps, &m2[idxP0], lap);
// if (idxb==0) LogOut("nv%d LAPLAZ %f %f (ood2 %f)\n\n",nv,m2[idxP0],m2[idxP0+1],ood2);

			} //end neighbour loop
			// LogOut("jamon\n");
			mPy = opCode(mul_ps, mel, mel);

#if	defined(__AVX__)// || defined(__AVX512F__)
			mPx = opCode(add_ps, opCode(permute_ps, mPy, 0b10110001), mPy);
#else
			mPx = opCode(add_ps, opCode(shuffle_ps, mPy, mPy, 0b10110001), mPy);
#endif

			switch	(VQcd & VQCD_TYPE) {
					default:
					case	VQCD_1:
					mMx = opCode(sub_ps,
						opCode(add_ps, lap, zQVec),
						opCode(mul_ps,
							opCode(mul_ps,
								opCode(sub_ps, mPx, opCode(set1_ps, z2)),
								opCode(set1_ps, LL)),
								         mel));
					break;

					case	VQCD_1_PQ_2:
					mMx = opCode(sub_ps,
						opCode(add_ps, lap, zQVec),
							opCode(mul_ps,
								opCode(mul_ps,
									opCode(sub_ps, opCode(mul_ps, mPx, mPx), opCode(set1_ps, z4)),
									opCode(mul_ps, mPx, opCode(set1_ps, LaLa))),
								mel));
					 break;

					//FIX ME ADDITIONAL POTENTIALS!
	 				case	VQCD_2:
	 				mMx = opCode(sub_ps,
	 					opCode(sub_ps,
	 						opCode(mul_ps,
	 							opCode(add_ps,
	 								opCode(add_ps,
	 									opCode(load_ps, &m[idxMz]),
	 									opCode(add_ps,
	 										opCode(add_ps,
	 											opCode(add_ps, tmp, opCode(load_ps, &m[idxPx])),
	 											opCode(load_ps, &m[idxMx])),
	 										opCode(load_ps, &m[idxPz]))),
	 								opCode(mul_ps, mel, opCode(set1_ps, -6.f))),
	 							opCode(set1_ps, ood2)),
	 						opCode(mul_ps, opCode(set1_ps, zQ), opCode(sub_ps, mel, zRVec))),
	 					opCode(mul_ps,
	 						opCode(mul_ps,
	 							opCode(sub_ps, mPx, opCode(set1_ps, z2)),
	 							opCode(set1_ps, LL)),
	 						mel));
	 				break;

					case	VQCD_1N2:
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
									opCode(mul_ps, mel, opCode(set1_ps, -6.f))),
								opCode(set1_ps, ood2)),
								// 1N2 part
							opCode(mul_ps,zNVec,mel)),
						opCode(mul_ps,
							opCode(mul_ps,
								opCode(sub_ps, mPx, opCode(set1_ps, z2)),
								opCode(set1_ps, LL)),
							mel));
					break;
			}




			mPy = opCode(load_ps, &v[idxV0]);

			switch (VQcd & VQCD_DAMP) {

				default:
				case	VQCD_NONE:
#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_ps, mMx, opCode(set1_ps, dzc), mPy);
#else
			 	tmp = opCode(add_ps, mPy, opCode(mul_ps, mMx, opCode(set1_ps, dzc)));
#endif
				break;

				case	VQCD_DAMP_RHO:
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
// A*dzc + mPy - epsi (M/|M|^2)(2*(vecmv-|M|^2/t) +vecma dzc)
tmp = opCode(sub_ps,
	opCode(fmadd_ps, mMx, opCode(set1_ps, dzc), mPy),
	opCode(mul_ps, opCode(mul_ps, opCode(set1_ps, epsi), opCode(div_ps, mel, mPx)),
		opCode(fmadd_ps, opCode(sub_ps,vecmv,opCode(div_ps,mPx,opCode(set1_ps, zR))), opCode(set1_ps, 2.f), opCode(mul_ps, vecma, opCode(set1_ps, dzc)))));
#else

tmp = opCode(sub_ps,
	opCode(add_ps, mPy, opCode(mul_ps, mMx, opCode(set1_ps, dzc))),
	opCode(mul_ps, opCode(mul_ps, opCode(set1_ps, epsi), opCode(div_ps, mel, mPx)),
		opCode(add_ps,
			opCode(mul_ps, opCode(sub_ps,vecmv,opCode(div_ps,mPx,opCode(set1_ps, zR))), opCode(set1_ps, 2.f)),
			opCode(mul_ps, vecma, opCode(set1_ps, dzc)))));
#endif
				}
				break;

				case	VQCD_DAMP_ALL:
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
			if (VQcd & VQCD_EVOL_RHO)
			{
				auto vecmv = opCode(mul_ps, mel, tmp);
#if	defined(__AVX__)// || defined(__AVX512F__)
				auto vecma = opCode(add_ps, opCode(permute_ps, vecmv, 0b10110001), vecmv);
#else
				auto vecma = opCode(add_ps, opCode(shuffle_ps, vecmv, vecmv, 0b10110001), vecmv);
#endif
				tmp   = opCode(div_ps, opCode(mul_ps, mel, vecma), mPx);
			}

#if	defined(__AVX512F__) || defined(__FMA__)
			mPx = opCode(fmadd_ps, tmp, opCode(set1_ps, dzd), mel);
#else
			mPx = opCode(add_ps, mel, opCode(mul_ps, tmp, opCode(set1_ps, dzd)));
#endif
			opCode(store_ps,  &v[idxV0], tmp);
			opCode(stream_ps, &m2[idxP0], mPx);	// Avoids cache thrashing
// if (idxb==0) LogOut("final mevolto %f %f\n\n",m2[idxP0],m2[idxP0+1]);
		      }
		    }
		  }
		}
#undef	_MData_
#undef	step
	}
}


//----------------------------------------------------------------------------// FUN
//----------------------------------------------------------------------------//



//----------------------------------------------------------------------------//
// FIXME DO WE NEED THIS FUNCTION?
// inline	void	updateMXeon(void * __restrict__ m_, const void * __restrict__ v_, const double dz, const double d, const size_t Vo, const size_t Vf, const size_t Sf, FieldPrecision precision)
// {
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
//
// 		double * __restrict__ m		= (double * __restrict__) __builtin_assume_aligned (m_, Align);
// 		const double * __restrict__ v	= (const double * __restrict__) __builtin_assume_aligned (v_, Align);
//
// 		const double dzd = dz*d;
//
// 		#pragma omp parallel default(shared)
// 		{
// 			register _MData_ mIn, vIn, tmp;
// 			register size_t idxP0, idxMz;
//
// 			#pragma omp for schedule(static)
// 			for (size_t idx = Vo; idx < Vf; idx += step)
// 			{
// #if	defined(__AVX512F__) || defined(__FMA__)
// 				vIn = opCode(load_pd, &v[idxMz]);
// 				mIn = opCode(load_pd, &m[idxP0]);
// 				tmp = opCode(fmadd_pd, opCode(set1_pd, dzd), vIn, mIn);
// 				opCode(store_pd, &m[idxP0], tmp);
// #else
// 				mIn = opCode(load_pd, &m[idxP0]);
// 				tmp = opCode(load_pd, &v[idxMz]);
// 				vIn = opCode(mul_pd, opCode(set1_pd, dzd), tmp);
// 				tmp = opCode(add_pd, mIn, vIn);
// 				opCode(store_pd, &m[idxP0], tmp);
// #endif
// 			}
// 		}
// #undef	_MData_
// #undef	step
// 	}
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
//
// 		float * __restrict__ m		= (float * __restrict__) __builtin_assume_aligned (m_, Align);
// 		const float * __restrict__ v	= (const float * __restrict__) __builtin_assume_aligned (v_, Align);
//
// 		const float dzd = dz*d;
// #if	defined(__AVX512F__)
// //		const float __attribute__((aligned(Align))) dzdAux[16] = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd  };
// #elif	defined(__AVX__)
// //		const float __attribute__((aligned(Align))) dzdAux[8]  = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd };
// #else
// //		const float __attribute__((aligned(Align))) dzdAux[4]  = { dzd, dzd, dzd, dzd };
// #endif
// //		const _MData_ dzdVec = opCode(load_ps, dzdAux);
//
// 		#pragma omp parallel default(shared)
// 		{
// 			register _MData_ mIn, vIn, tmp;
// 			register size_t idxP0, idxMz;
//
// 			#pragma omp for schedule(static)
// 			for (size_t idx = Vo; idx < Vf; idx += step)
// 			{
// 				idxP0 = idx << 1;
// 				idxMz = (idx - Sf) << 1;
// #if	defined(__AVX512F__) || defined(__FMA__)
// 				vIn = opCode(load_ps, &v[idxMz]);
// 				mIn = opCode(load_ps, &m[idxP0]);
// 				tmp = opCode(fmadd_ps, opCode(set1_ps, dzd), vIn, mIn);
// 				opCode(store_ps, &m[idxP0], tmp);
// #else
// 				vIn = opCode(load_ps, &v[idxMz]);
// 				mIn = opCode(load_ps, &m[idxP0]);
// 				tmp = opCode(add_ps, mIn, opCode(mul_ps, opCode(set1_ps, dzd), vIn));
// 				opCode(store_ps, &m[idxP0], tmp);
// #endif
// 			}
// 		}
// #undef	_MData_
// #undef	step
// 	}
// }


//----------------------------------------------------------------------------// FUN
//----------------------------------------------------------------------------//



//----------------------------------------------------------------------------//
// FIX ME ADAPT the v part of propagateNNKernelXeon here;
// not needed for even step propagators!
// template<const VqcdType VQcd>
// inline	void	updateVNNXeon(const void * __restrict__ m_, void * __restrict__ v_, double *R, const double dz, const double c, const double ood2,
// 			    const double LL, const double aMass2, const double gamma, const size_t Lx, const size_t Vo, const size_t Vf, const size_t Sf, FieldPrecision precision)
// {
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
//
// 		const double * __restrict__ m = (const double * __restrict__) __builtin_assume_aligned (m_, Align);
// 		double * __restrict__ v = (double * __restrict__) __builtin_assume_aligned (v_, Align);
//
// 		const double zR = *R;
// 		const double z2 = zR*zR;
// 		//const double zQ = 9.*pow(zR, nQcd+3.);
// 		const double zQ = aMass2*z2*zR;
// 		const double zN = aMass2*z2/2;
// 		const double dzc = dz*c;
//
// 		const double z4 = z2*z2;
// 		const double LaLa = LL*2./z4;
// 		const double GGGG = pow(ood2,0.5)*gamma;
// //		const double GGiZ = GGGG/zR;
// 		const double mola = GGGG*dzc/2.;
// 		const double damp1 = 1./(1.+mola);
// 		const double damp2 = (1.-mola)*damp1;
// 		const double epsi = mola/(1.+mola);
//
// #if	defined(__AVX512F__)
// 		const size_t XC = (Lx<<2);
// 		const size_t YC = (Lx>>2);
//
// 		const double __attribute__((aligned(Align))) zQAux[8] = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };	// Only real part
// 		const double __attribute__((aligned(Align))) zNAux[8] = { zN,-zN, zN,-zN, zN,-zN, zN,-zN };	// to complex congugate
// 		const double __attribute__((aligned(Align))) zRAux[8] = { zR, 0., zR, 0., zR, 0., zR, 0. };	// Only real part
// 		const int    __attribute__((aligned(Align))) shfRg[8] = {6, 7, 0, 1, 2, 3, 4, 5 };
// 		const int    __attribute__((aligned(Align))) shfLf[8] = {2, 3, 4, 5, 6, 7, 0, 1 };
//
// 		const _MInt_  vShRg  = opCode(load_si512, shfRg);
// 		const _MInt_  vShLf  = opCode(load_si512, shfLf);
// #elif	defined(__AVX__)
// 		const size_t XC = (Lx<<1);
// 		const size_t YC = (Lx>>1);
//
// 		const double __attribute__((aligned(Align))) zQAux[4] = { zQ, 0., zQ, 0. };	// Only real part
// 		const double __attribute__((aligned(Align))) zNAux[4] = { zN,-zN, zN,-zN };	// to complex congugate
// 		const double __attribute__((aligned(Align))) zRAux[4] = { zR, 0., zR, 0. };	// Only real part
// #else
// 		const size_t XC = Lx;
// 		const size_t YC = Lx;
//
// 		const double __attribute__((aligned(Align))) zQAux[2] = { zQ, 0. };	// Only real part
// 		const double __attribute__((aligned(Align))) zNAux[2] = { zN,-zN };	// to complex congugate
// 		const double __attribute__((aligned(Align))) zRAux[2] = { zR, 0. };	// Only real part
// #endif
// 		const _MData_ zQVec  = opCode(load_pd, zQAux);
// 		const _MData_ zNVec  = opCode(load_pd, zNAux);
// 		const _MData_ zRVec  = opCode(load_pd, zRAux);
//
// 		#pragma omp parallel default(shared)
// 		{
// 			_MData_ tmp, mel, mPx, mPy, mMx;
//
// 			#pragma omp for schedule(static)
// 			for (size_t idx = Vo; idx < Vf; idx += step)
// 			{
// 				size_t X[2], idxPx, idxMx, idxPy, idxMy, idxPz, idxMz, idxP0;
//
// 				{
// 					size_t tmi = idx/XC, tpi;
//
// 					tpi = tmi/YC;
// 					X[1] = tmi - tpi*YC;
// 					X[0] = idx - tmi*XC;
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
// #if	defined(__AVX512F__)
// 					tmp = opCode(add_pd, opCode(permutexvar_pd, vShRg, opCode(load_pd, &m[idxMy])), opCode(load_pd, &m[idxPy]));
// #elif	defined(__AVX__)
// 					mPx = opCode(load_pd, &m[idxMy]);
// 					tmp = opCode(add_pd, opCode(permute2f128_pd, mPx, mPx, 0b00000001), opCode(load_pd, &m[idxPy]));
// #else
// 					tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
// #endif
// 				}
// 				else
// 				{
// 					idxMy = ((idx - XC) << 1);
//
// 					if (X[1] == YC-1)
// 					{
// 						idxPy = ((idx - Sf + XC) << 1);
// #if	defined(__AVX512F__)
// 						tmp = opCode(add_pd, opCode(permutexvar_pd, vShLf, opCode(load_pd, &m[idxPy])), opCode(load_pd, &m[idxMy]));
// #elif	defined(__AVX__)
// 						mPx = opCode(load_pd, &m[idxPy]);
// 						tmp = opCode(add_pd, opCode(permute2f128_pd, mPx, mPx, 0b00000001), opCode(load_pd, &m[idxMy]));
// #else
// 						tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
// #endif
// 					}
// 					else
// 					{
// 						idxPy = ((idx + XC) << 1);
// 						tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
// 					}
// 				}
//
// 				idxPz = ((idx+Sf) << 1);
// 				idxMz = ((idx-Sf) << 1);
// 				idxP0 = (idx << 1);
//
// 				mel = opCode(load_pd, &m[idxP0]);
// 				mPy = opCode(mul_pd, mel, mel);
//
// #if	defined(__AVX512F__)
// 				mPx = opCode(add_pd, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, mPy), _MM_PERM_BADC)), mPy);
// #elif	defined(__AVX__)
// 				mPx = opCode(add_pd, opCode(permute_pd, mPy, 0b00000101), mPy);
// #else
// 				mPx = opCode(add_pd, opCode(shuffle_pd, mPy, mPy, 0b00000001), mPy);
// #endif
// 				switch	(VQcd & VQCD_TYPE) {
//
// 					default:
// 					case	VQCD_1:
// 						mMx = opCode(sub_pd,
// 							opCode(add_pd,
// 								opCode(mul_pd,
// 									opCode(add_pd,
// 										opCode(add_pd,
// 											opCode(load_pd, &m[idxMz]),
// 											opCode(add_pd,
// 												opCode(add_pd,
// 													opCode(add_pd, tmp, opCode(load_pd, &m[idxPx])),
// 													opCode(load_pd, &m[idxMx])),
// 												opCode(load_pd, &m[idxPz]))),
// 										opCode(mul_pd, mel, opCode(set1_pd, -6.0))),
// 									opCode(set1_pd, ood2)),
// 								zQVec),
// 							opCode(mul_pd,
// 								opCode(mul_pd,
// 									opCode(sub_pd, mPx, opCode(set1_pd, z2)),
// 									opCode(set1_pd, LL)),
// 								mel));
// 						break;
//
// 					case	VQCD_1_PQ_2:
// 						mMx = opCode(sub_pd,
// 							opCode(add_pd,
// 								opCode(mul_pd,
// 									opCode(add_pd,
// 										opCode(add_pd,
// 											opCode(load_pd, &m[idxMz]),
// 											opCode(add_pd,
// 												opCode(add_pd,
// 													opCode(add_pd, tmp, opCode(load_pd, &m[idxPx])),
// 													opCode(load_pd, &m[idxMx])),
// 												opCode(load_pd, &m[idxPz]))),
// 										opCode(mul_pd, mel, opCode(set1_pd, -6.0))),
// 									opCode(set1_pd, ood2)),
// 								zQVec),
// 							opCode(mul_pd,
// 								opCode(mul_pd,
// 									opCode(sub_pd, opCode(mul_pd, mPx, mPx), opCode(set1_pd, z4)),
// 									opCode(mul_pd, mPx, opCode(set1_pd, LaLa))),
// 								mel));
// 						 break;
//
// 					case	VQCD_2:
// 						mMx = opCode(sub_pd,
// 							opCode(sub_pd,
// 								opCode(mul_pd,
// 									opCode(add_pd,
// 										opCode(add_pd,
// 											opCode(load_pd, &m[idxMz]),
// 											opCode(add_pd,
// 												opCode(add_pd,
// 													opCode(add_pd, tmp, opCode(load_pd, &m[idxPx])),
// 													opCode(load_pd, &m[idxMx])),
// 												opCode(load_pd, &m[idxPz]))),
// 										opCode(mul_pd, mel, opCode(set1_pd, -6.0))),
// 									opCode(set1_pd, (float) ood2)),
// 								opCode(mul_pd, opCode(set1_pd, zQ), opCode(sub_pd, mel, zRVec))),
// 							opCode(mul_pd,
// 								opCode(mul_pd,
// 									opCode(sub_pd, mPx, opCode(set1_pd, z2)),
// 									opCode(set1_pd, LL)),
// 								mel));
// 						break;
//
// 					case	VQCD_1N2:
// 					mMx = opCode(sub_pd,
// 						opCode(add_pd,
// 							opCode(mul_pd,
// 								opCode(add_pd,
// 									opCode(add_pd,
// 										opCode(load_pd, &m[idxMz]),
// 										opCode(add_pd,
// 											opCode(add_pd,
// 												opCode(add_pd, tmp, opCode(load_pd, &m[idxPx])),
// 												opCode(load_pd, &m[idxMx])),
// 											opCode(load_pd, &m[idxPz]))),
// 									opCode(mul_pd, mel, opCode(set1_pd, -6.0))),
// 								opCode(set1_pd, ood2)),
// 								// 1N2 part
// 							opCode(mul_pd,zNVec,mel)),
// 						opCode(mul_pd,
// 							opCode(mul_pd,
// 								opCode(sub_pd, mPx, opCode(set1_pd, z2)),
// 								opCode(set1_pd, LL)),
// 							mel));
// 					break;
//
// 				}
//
// 				mPy = opCode(load_pd, &v[idxMz]);
//
// 				switch	(VQcd & VQCD_DAMP) {
//
// 					default:
// 					case	VQCD_NONE:
// #if	defined(__AVX512F__) || defined(__FMA__)
// 						tmp = opCode(fmadd_pd, mMx, opCode(set1_pd, dzc), mPy);
// #else
// 						tmp = opCode(add_pd, mPy, opCode(mul_pd, mMx, opCode(set1_pd, dzc)));
// #endif
// 						break;
//
// 					case	VQCD_DAMP_RHO:
// 					{
// 						//New implementation
// 						tmp = opCode(mul_pd, mel, mPy);
// 	#if	defined(__AVX__)// || defined(__AVX512F__)
// 						auto vecmv = opCode(add_pd, opCode(permute_pd, tmp, 0b00000101), tmp);
// 	#else
// 						auto vecmv = opCode(add_pd, opCode(shuffle_pd, tmp, tmp, 0b00000001), tmp);
// 	#endif
//
// 						// vecma = MA
// 						// mel = M, mMx = A
// 						tmp = opCode(mul_pd, mel, mMx);
// 	#if	defined(__AVX__)// || defined(__AVX512F__)
// 						auto vecma = opCode(add_pd, opCode(permute_pd, tmp, 0b00000001), tmp);
// 	#else
// 						auto vecma = opCode(add_pd, opCode(shuffle_pd, tmp, tmp, 0b00000001), tmp);
// 	#endif
//
// 	#if	defined(__AVX512F__) || defined(__FMA__)
// 	// A*dzc + mPy - epsi (M/|M|^2)(2*(vecmv-|M|^2/t) +vecma dzc)
// 	tmp = opCode(sub_pd,
// 		opCode(fmadd_pd, mMx, opCode(set1_pd, dzc), mPy),
// 		opCode(mul_pd, opCode(mul_pd, opCode(set1_pd, epsi), opCode(div_pd, mel, mPx)),
// 			opCode(fmadd_pd, opCode(sub_pd,vecmv,opCode(div_pd,mPx,opCode(set1_pd, zR))), opCode(set1_pd, 2.0), opCode(mul_pd, vecma, opCode(set1_pd, dzc)))));
// 	#else
// 	tmp = opCode(sub_pd,
// 		opCode(add_pd, mPy, opCode(mul_pd, mMx, opCode(set1_pd, dzc))),
// 		opCode(mul_pd, opCode(mul_pd, opCode(set1_pd, epsi), opCode(div_pd, mel, mPx)),
// 			opCode(add_pd,
// 				opCode(mul_pd, opCode(sub_pd,vecmv,opCode(div_pd,mPx,opCode(set1_pd, zR))), opCode(set1_pd, 2.0)),
// 				opCode(mul_pd, vecma, opCode(set1_pd, dzc)))));
// 	#endif
// 					}
// 					break;
//
// 					case	VQCD_DAMP_ALL:
// #if	defined(__AVX512F__) || defined(__FMA__)
// 						tmp = opCode(fmadd_pd, mPy, opCode(set1_pd, damp2), opCode(mul_pd, mMx, opCode(set1_pd, damp1*dzc)));
// #else
// 						tmp = opCode(add_pd, opCode(mul_pd, mPy, opCode(set1_pd,damp2)), opCode(mul_pd, mMx, opCode(set1_pd, damp1*dzc)));
// #endif
// 						break;
// 				}
//
// 				if (VQcd & VQCD_EVOL_RHO)
// 				{
// 					auto vecmv = opCode(mul_pd, mel, tmp);
// #if	defined(__AVX__)// || defined(__AVX512F__)
// 					auto vecma = opCode(add_pd, opCode(permute_pd, vecmv, 0b00000101), vecmv);
// #else
// 					auto vecma = opCode(add_pd, opCode(shuffle_pd, vecmv, vecmv, 0b00000001), vecmv);
// #endif
// 					tmp   = opCode(div_pd, opCode(mul_pd, mel, vecma), mPx);
// 				}
//
// 				opCode(store_pd,  &v[idxMz], tmp);
// 			}
// 		}
// #undef	_MData_
// #undef	step
// 	}
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
//
// 		const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_, Align);
// 		float * __restrict__ v		= (float * __restrict__) __builtin_assume_aligned (v_, Align);
//
// 		const float zR = *R;
// 		const float z2 = zR*zR;
// 		//const float zQ = 9.*powf(zR, nQcd+3.);
// 		const float zQ = (float) (aMass2*z2*zR);
// 		//For VQCD_1N2
// 		const float zN = (float) (aMass2*z2)/2;
// 		const float dzc = dz*c;
//
// 		const float z4 = z2*z2;
// 		const float LaLa = LL*2./z4;
// 		const float GGGG = pow(ood2, 0.5)*gamma;
// //		const float GGiZ = GGGG/zR;
// 		const float mola = GGGG*dzc/2.;
// 		const float damp1 = 1.f/(1.f+mola);
// 		const float damp2 = (1.f-mola)*damp1;
// 		const float epsi = mola/(1.f+mola);
//
// #if	defined(__AVX512F__)
// 		const size_t XC = (Lx<<3);
// 		const size_t YC = (Lx>>3);
//
// 		const float __attribute__((aligned(Align))) zQAux[16] = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
// 		const float __attribute__((aligned(Align))) zNAux[16] = { zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN };
// 		const float __attribute__((aligned(Align))) zRAux[16] = { zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f };
// 		const int   __attribute__((aligned(Align))) shfRg[16] = {14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
// 		const int   __attribute__((aligned(Align))) shfLf[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1 };
//
// 		const _MInt_  vShRg  = opCode(load_si512, shfRg);
// 		const _MInt_  vShLf  = opCode(load_si512, shfLf);
// #elif	defined(__AVX__)
// 		const size_t XC = (Lx<<2);
// 		const size_t YC = (Lx>>2);
//
// 		const float __attribute__((aligned(Align))) zQAux[8]  = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
// 		const float __attribute__((aligned(Align))) zNAux[8]  = { zN, -zN, zN, -zN, zN, -zN, zN, -zN };
// 		const float __attribute__((aligned(Align))) zRAux[8]  = { zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f };
// #else
// 		const size_t XC = (Lx<<1);
// 		const size_t YC = (Lx>>1);
//
// 		const float __attribute__((aligned(Align))) zQAux[4]  = { zQ, 0.f, zQ, 0.f };
// 		const float __attribute__((aligned(Align))) zQAux[4]  = { zN, -zN, zN, -zN };
// 		const float __attribute__((aligned(Align))) zRAux[4]  = { zR, 0.f, zR, 0.f };
// #endif
// 		const _MData_ zQVec  = opCode(load_ps, zQAux);
// 		const _MData_ zNVec  = opCode(load_ps, zNAux);
// 		const _MData_ zRVec  = opCode(load_ps, zRAux);
//
// 		#pragma omp parallel default(shared)
// 		{
// 			_MData_ tmp, mel, mPx, mPy, mMx;
//
// 			#pragma omp for schedule(static)
// 			for (size_t idx = Vo; idx < Vf; idx += step)
// 			{
// 				size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0;
//
// 				{
// 					size_t tmi = idx/XC, itp;
//
// 					itp = tmi/YC;
// 					X[1] = tmi - itp*YC;
// 					X[0] = idx - tmi*XC;
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
// #if	defined(__AVX512F__)
// 					tmp = opCode(add_ps, opCode(permutexvar_ps, vShRg, opCode(load_ps, &m[idxMy])), opCode(load_ps, &m[idxPy]));
// #elif	defined(__AVX2__)	//AVX2
// 					tmp = opCode(add_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 6,7,0,1,2,3,4,5)),  opCode(load_ps, &m[idxPy]));
// #elif	defined(__AVX__)	//AVX
// 					mMx = opCode(permute_ps, opCode(load_ps, &m[idxMy]), 0b01001110);
// 					mPx = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
// 					tmp = opCode(add_ps, opCode(blend_ps, mMx, mPx, 0b00110011), opCode(load_ps, &m[idxPy]));
// #else
// 					mMx = opCode(load_ps, &m[idxMy]);
// 					tmp = opCode(add_ps, opCode(shuffle_ps, mMx, mMx, 0b01001110), opCode(load_ps, &m[idxPy]));
// #endif
// 				}
// 				else
// 				{
// 					idxMy = ((idx - XC) << 1);
//
// 					if (X[1] == YC-1)
// 					{
// 						idxPy = ((idx - Sf + XC) << 1);
// #if	defined(__AVX512F__)
// 						tmp = opCode(add_ps, opCode(permutexvar_ps, vShLf, opCode(load_ps, &m[idxPy])), opCode(load_ps, &m[idxMy]));
// #elif	defined(__AVX2__)	//AVX2
// 						tmp = opCode(add_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 2,3,4,5,6,7,0,1)), opCode(load_ps, &m[idxMy]));
// #elif	defined(__AVX__)	//AVX
// 						mMx = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b01001110);
// 						mPx = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
// 						tmp = opCode(add_ps, opCode(blend_ps, mMx, mPx, 0b11001100), opCode(load_ps, &m[idxMy]));
// #else
// 						mMx = opCode(load_ps, &m[idxPy]);
// 						tmp = opCode(add_ps, opCode(shuffle_ps, mMx, mMx, 0b01001110), opCode(load_ps, &m[idxMy]));
// #endif
// 					}
// 					else
// 					{
// 						idxPy = ((idx + XC) << 1);
// 						tmp = opCode(add_ps, opCode(load_ps, &m[idxPy]), opCode(load_ps, &m[idxMy]));
// 					}
// 				}
//
// 				idxPz = ((idx+Sf) << 1);
// 				idxMz = ((idx-Sf) << 1);
// 				idxP0 = (idx << 1);
//
// 				mel = opCode(load_ps, &m[idxP0]);
// 				mPy = opCode(mul_ps, mel, mel);
//
// #if	defined(__AVX__) || defined(__AVX512F__)
// 				mPx = opCode(add_ps, opCode(permute_ps, mPy, 0b10110001), mPy);
// #else
// 				mPx = opCode(add_ps, opCode(shuffle_ps, mPy, mPy, 0b10110001), mPy);
// #endif
// 				switch	(VQcd & VQCD_TYPE) {
// 					default:
// 					case	VQCD_1:
// 					mMx = opCode(sub_ps,
// 						opCode(add_ps,
// 							opCode(mul_ps,
// 								opCode(add_ps,
// 									opCode(add_ps,
// 										opCode(load_ps, &m[idxMz]),
// 										opCode(add_ps,
// 											opCode(add_ps,
// 												opCode(add_ps, tmp, opCode(load_ps, &m[idxPx])),
// 												opCode(load_ps, &m[idxMx])),
// 											opCode(load_ps, &m[idxPz]))),
// 									opCode(mul_ps, mel, opCode(set1_ps, -6.f))),
// 								opCode(set1_ps, ood2)),
// 							zQVec),
// 						opCode(mul_ps,
// 							opCode(mul_ps,
// 								opCode(sub_ps, mPx, opCode(set1_ps, z2)),
// 								opCode(set1_ps, (float) LL)),
// 							mel));
// 					break;
//
// 					case	VQCD_1_PQ_2:
// 					mMx = opCode(sub_ps,
// 						opCode(add_ps,
// 							opCode(mul_ps,
// 								opCode(add_ps,
// 									opCode(add_ps,
// 										opCode(load_ps, &m[idxMz]),
// 									 	opCode(add_ps,
// 											opCode(add_ps,
// 												opCode(add_ps, tmp, opCode(load_ps, &m[idxPx])),
// 												opCode(load_ps, &m[idxMx])),
// 											opCode(load_ps, &m[idxPz]))),
// 									opCode(mul_ps, mel, opCode(set1_ps, -6.f))),
// 								opCode(set1_ps, ood2)),
// 							zQVec),
// 						opCode(mul_ps,
// 							opCode(mul_ps,
// 								opCode(sub_ps, opCode(mul_ps, mPx, mPx), opCode(set1_ps, z4)),
// 								opCode(mul_ps, mPx, opCode(set1_ps, LaLa))),
// 							mel));
// 					break;
//
// 					case	VQCD_2:
// 					mMx = opCode(sub_ps,
// 						opCode(sub_ps,
// 							opCode(mul_ps,
// 								opCode(add_ps,
// 									opCode(add_ps,
// 										opCode(load_ps, &m[idxMz]),
// 										opCode(add_ps,
// 											opCode(add_ps,
// 												opCode(add_ps, tmp, opCode(load_ps, &m[idxPx])),
// 												opCode(load_ps, &m[idxMx])),
// 											opCode(load_ps, &m[idxPz]))),
// 									opCode(mul_ps, mel, opCode(set1_ps, -6.f))),
// 								opCode(set1_ps, (float) ood2)),
// 							opCode(mul_ps, opCode(set1_ps, zQ), opCode(sub_ps, mel, zRVec))),
// 						opCode(mul_ps,
// 							opCode(mul_ps,
// 								opCode(sub_ps, mPx, opCode(set1_ps, z2)),
// 								opCode(set1_ps, (float) LL)),
// 							mel));
// 					break;
//
// 					case	VQCD_1N2:
// 					mMx = opCode(sub_ps,
// 						opCode(add_ps,
// 							opCode(mul_ps,
// 								opCode(add_ps,
// 									opCode(add_ps,
// 										opCode(load_ps, &m[idxMz]),
// 										opCode(add_ps,
// 											opCode(add_ps,
// 												opCode(add_ps, tmp, opCode(load_ps, &m[idxPx])),
// 												opCode(load_ps, &m[idxMx])),
// 											opCode(load_ps, &m[idxPz]))),
// 									opCode(mul_ps, mel, opCode(set1_ps, -6.f))),
// 								opCode(set1_ps, ood2)),
// 								// 1N2 part
// 							opCode(mul_ps,zNVec,mel)),
// 						opCode(mul_ps,
// 							opCode(mul_ps,
// 								opCode(sub_ps, mPx, opCode(set1_ps, z2)),
// 								opCode(set1_ps, LL)),
// 							mel));
// 					break;
//
// 					case	VQCD_QUAD:
// 					mMx = opCode(sub_ps,
// 						opCode(add_ps,
// 							opCode(mul_ps,
// 								opCode(add_ps,
// 									opCode(add_ps,
// 										opCode(load_ps, &m[idxMz]),
// 										opCode(add_ps,
// 											opCode(add_ps,
// 												opCode(add_ps, tmp, opCode(load_ps, &m[idxPx])),
// 												opCode(load_ps, &m[idxMx])),
// 											opCode(load_ps, &m[idxPz]))),
// 									opCode(mul_ps, mel, opCode(set1_ps, -6.f))),
// 								opCode(set1_ps, ood2)),
// 								// 1N2 part
// 							opCode(mul_ps,zNVec,mel)),
// 						opCode(mul_ps,
// 							opCode(mul_ps,
// 								opCode(sub_ps, mPx, opCode(set1_ps, z2)),
// 								opCode(set1_ps, LL)),
// 							mel));
// 					break;
//
//
// 				}
//
// 				mPy = opCode(load_ps, &v[idxMz]);
//
// 				switch (VQcd & VQCD_DAMP) {
//
// 					default:
// 					case	VQCD_NONE:
// #if	defined(__AVX512F__) || defined(__FMA__)
// 					tmp = opCode(fmadd_ps, mMx, opCode(set1_ps, dzc), mPy);
// #else
// 			 		tmp = opCode(add_ps, mPy, opCode(mul_ps, mMx, opCode(set1_ps, dzc)));
// #endif
// 					break;
//
// 					case	VQCD_DAMP_RHO:
// 					{
// // NEW implementation
// // V = (V+Adt) - (epsi M/|M|^2)(2 MV+ MA*dt - 2 |M|^2/t)
// // recall
// // V=mPy     A=mMx    |M|^2=mPx      M=mel
// // vecmv = MV
// 					tmp = opCode(mul_ps, mel, mPy);
// #if	defined(__AVX__)// || defined(__AVX512F__)
// 					auto vecmv = opCode(add_ps, opCode(permute_ps, tmp, 0b10110001), tmp);
// #else
// 					auto vecmv = opCode(add_ps, opCode(shuffle_ps, tmp, tmp, 0b10110001), tmp);
// #endif
//
// 					// vecma = MA
// 					// mel = M, mMx = A
// 					tmp = opCode(mul_ps, mel, mMx);
// #if	defined(__AVX__)// || defined(__AVX512F__)
// 					auto vecma = opCode(add_ps, opCode(permute_ps, tmp, 0b10110001), tmp);
// #else
// 					auto vecma = opCode(add_ps, opCode(shuffle_ps, tmp, tmp, 0b10110001), tmp);
// #endif
//
// #if	defined(__AVX512F__) || defined(__FMA__)
// // A*dzc + mPy - epsi (M/|M|^2)(2*(vecmv-|M|^2/t) +vecma dzc)
// tmp = opCode(sub_ps,
// 	opCode(fmadd_ps, mMx, opCode(set1_ps, dzc), mPy),
// 	opCode(mul_ps, opCode(mul_ps, opCode(set1_ps, epsi), opCode(div_ps, mel, mPx)),
// 		opCode(fmadd_ps, opCode(sub_ps,vecmv,opCode(div_ps,mPx,opCode(set1_ps, zR))), opCode(set1_ps, 2.f), opCode(mul_ps, vecma, opCode(set1_ps, dzc)))));
// #else
//
// tmp = opCode(sub_ps,
// 	opCode(add_ps, mPy, opCode(mul_ps, mMx, opCode(set1_ps, dzc))),
// 	opCode(mul_ps, opCode(mul_ps, opCode(set1_ps, epsi), opCode(div_ps, mel, mPx)),
// 		opCode(add_ps,
// 			opCode(mul_ps, opCode(sub_ps,vecmv,opCode(div_ps,mPx,opCode(set1_ps, zR))), opCode(set1_ps, 2.f)),
// 			opCode(mul_ps, vecma, opCode(set1_ps, dzc)))));
// #endif
// 					}
// 					break;
//
// 					case	VQCD_DAMP_ALL:
// 					tmp = opCode(add_ps, opCode(mul_ps, mPy, opCode(set1_ps, damp2)), opCode(mul_ps, mMx, opCode(set1_ps, damp1*dzc)));
// 					break;
// 				}
//
// 				if (VQcd & VQCD_EVOL_RHO)
// 				{
// 					auto vecmv = opCode(mul_ps, mel, tmp);
// #if	defined(__AVX__)// || defined(__AVX512F__)
// 					auto vecma = opCode(add_ps, opCode(permute_ps, vecmv, 0b10110001), vecmv);
// #else
// 					auto vecma = opCode(add_ps, opCode(shuffle_ps, vecmv, vecmv, 0b10110001), vecmv);
// #endif
// 					tmp   = opCode(div_ps, opCode(mul_ps, mel, vecma), mPx);
// 				}
// 				opCode(store_ps,  &v[idxMz], tmp);
// 			}
// 		}
// #undef	_MData_
// #undef	step
// 	}
// }

//----------------------------------------------------------------------------//
// PREPARES GHOST REGIONS
//----------------------------------------------------------------------------//
// uses same indices than kernel
// Vo and Vf must be the same: begginig of the slice which we will have to prepare
// if Vo = Sf (slice 1 of m or 1st slice of data and Ng = 2
// we take the last (physical) slices of the previous rank Lz-2 Lz-1 with weigths
// normalised to the last i.e.
// Lz-2*C[2]/C[1] + Lz-1
// because the


	//FIX ME DOUBLE PRECISION?
template<const VqcdType VQcd>
inline	void	prepareGhostKernelXeon(const void * __restrict__ m_, void * __restrict__ vg_, const double ood2, const size_t Lx, const size_t z0, FieldPrecision precision)
{
	// Computes the parts of the laplacian of slices z0 and Lz-1-z0 that reside in other ranks anc copies them into vGhost
	// Sum m[idx+z0*Sf*nv]C[nv + nvbase] ; idx =0...Sf-1
	// Sum m[idx+(Lz-1-z0)*Sf*nv]C[nv + nvbase] ; idx =0...Sf-1
	// z0 plays the role of bsl in nlaplacian.cpp
	// preamble

	//Volume of data and surface slice
	const size_t Sf = Lx*Lx;
	const size_t Vo = Lx*Lx*sizeZ;
	if (z0 > Ng)
	{
		LogError("No preparation needed! why did you call this function?");
		return; //
	}


	// number of slices to sum
	const size_t nvMax= Ng-z0;
	// z0 = 0 and Ng =1 > 1 slice
	// z0 = 0 and Ng =7 > 7 slices

	// shift for the Coefficients
	const size_t nvbase = z0;

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
		double * __restrict__ vg		= (double * __restrict__) __builtin_assume_aligned (vg_, Align);

#if	defined(__AVX512F__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);
#else
		const size_t XC = Lx;
		const size_t YC = Lx;
#endif

double CO[5] = {0, 0, 0, 0, 0} ;
if (Ng == 0) {
	return;
}	else if (Ng == 1) {
	CO[0] = 1.  ;
}	else if (Ng == 2) {
	CO[0] = 4./3.; CO[1] = -1./12.;
} else if (Ng == 3) {
	CO[0] = 1.5    ; CO[1] = -3./20.0; CO[2] = 1./90. ;
} else if (Ng == 4) {
	CO[0] = 1.6    ; CO[1] = -0.2    ; CO[2] = 8./315. ; CO[3] = -1./560. ;
} else if (Ng == 5) {
	CO[0] = 5./3.  ; CO[1] = -5./21. ; CO[2] = 5./126. ; CO[3] = -5./1008. ; CO[4] = 1./3150. ;
}

_MData_ COV[5];
// we normalise the coefficients to the first one, which will get its notmalisation later
for (size_t nv = 0; nv < Ng ; nv++)
	COV[nv]  = opCode(set1_pd, CO[nv]/CO[nvbase]);

		  #pragma omp parallel default(shared)
		  {
		    _MData_ tmp0, tmp1;
		    #pragma omp for schedule(static)
				 for (uint idx = 0; idx < Sf; idx += step) {
						size_t idx0 = Sf + idx; // accounts for Ghost region +1
						size_t idxP0 = (idx0 << 1);
						size_t idx1 = Vo + idx; // accounts for Ghost region +1
						size_t idxP1 = (idx1 << 1);
						tmp0 = 	opCode(load_pd,&m[idxP0]);
						tmp1 = 	opCode(load_pd,&m[idxP1]);

						for (uint nv = 1; nv < nvMax; nv++) { // sum over neighbours is sum over zC
							idx0 += Sf;
							idxP0 = (idx0 << 1);
							idx1 -= Sf;
							idxP1 = (idx1 << 1);
							tmp0 = opCode(add_pd,tmp0,opCode(mul_pd,opCode(load_pd,&m[idxP0]), COV[nvbase+nv] ));
							tmp1 = opCode(add_pd,tmp1,opCode(mul_pd,opCode(load_pd,&m[idxP1]), COV[nvbase+nv] ));
						}
						// store it in vGhost // sendGhost will send it to the right place
						idxP0 = (idx << 1);
						opCode(store_pd,  &vg[idxP0], tmp0);
						idx1 = Sf + idx; // accounts for Ghost region +1
						idxP1 = (idx1 << 1);
						opCode(store_pd,  &vg[idxP1], tmp1);
		 		}
		  }
#undef	_MData_
#undef	step
	} // ----------------------------------------------------------------------//
	else if (precision == FIELD_SINGLE) {
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
		float * __restrict__ vg		= (float * __restrict__) __builtin_assume_aligned (vg_, Align);

#if	defined(__AVX512F__)
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);
#endif

float CO[5] = {0, 0, 0, 0, 0} ;
if (Ng == 0) {
	return;
}	else if (Ng == 1) {
	CO[0] = 1.  ;
}	else if (Ng == 2) {
	CO[0] = 4./3.; CO[1] = -1./12.;
} else if (Ng == 3) {
	CO[0] = 1.5    ; CO[1] = -3./20.0; CO[2] = 1./90. ;
} else if (Ng == 4) {
	CO[0] = 1.6    ; CO[1] = -0.2    ; CO[2] = 8./315. ; CO[3] = -1./560. ;
} else if (Ng == 5) {
	CO[0] = 5./3.  ; CO[1] = -5./21. ; CO[2] = 5./126. ; CO[3] = -5./1008. ; CO[4] = 1./3150. ;
}
_MData_ COV[5];
// we normalise the coefficients to the first one, which will get its notmalisation later
for (size_t nv = 0; nv < Ng ; nv++)
	COV[nv]  = opCode(set1_ps, CO[nv]/CO[nvbase]);

		  #pragma omp parallel default(shared)
		  {
		    _MData_ tmp0, tmp1;
		    #pragma omp for schedule(static)
				 for (uint idx = 0; idx < Sf; idx += step) {
						size_t idx0 = Sf + idx; // accounts for Ghost region +1
						size_t idxP0 = (idx0 << 1);
						size_t idx1 = Vo + idx; // accounts for Ghost region +1
						size_t idxP1 = (idx1 << 1);
						tmp0 = 	opCode(load_ps,&m[idxP0]);
						tmp1 = 	opCode(load_ps,&m[idxP1]);
// if (idx==0) LogOut("r%d check00p CO %f nv %d %f %f %f %f\n",commRank(),CO[nvbase+0],0,m[idxP0],m[idxP0+1],m[idxP1],m[idxP1+1]);

						for (uint nv = 1; nv < nvMax; nv++) { // sum over neighbours is sum over zC // first neighbour is already loaded
							idx0 += Sf;
							idxP0 = (idx0 << 1);
							idx1 -= Sf;
							idxP1 = (idx1 << 1);
							tmp0 = opCode(add_ps, tmp0, opCode(mul_ps,opCode(load_ps,&m[idxP0]),COV[nvbase+nv]));
							tmp1 = opCode(add_ps, tmp1, opCode(mul_ps,opCode(load_ps,&m[idxP1]),COV[nvbase+nv]));
// if (idx==0) LogOut("r%d check00p CO %f nv %d %f %f %f %f\n",commRank(),CO[nvbase+nv],nv,m[idxP0],m[idxP0+1],m[idxP1],m[idxP1+1]);
						}
						// store it in vGhost // sendGhost will send it to the right place
						idxP0 = (idx << 1);
						opCode(store_ps,  &vg[idxP0], tmp0);
						idx1 = Sf + idx; // accounts for Ghost region +1
						idxP1 = (idx1 << 1);
						opCode(store_ps,  &vg[idxP1], tmp1);
// if (idx==0) LogOut("r%d                   %d %f %f %f %f\n",commRank(),vg[idxP0],vg[idxP0+1],vg[idxP1],vg[idxP1+1]);

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
