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
inline	void	propagateNaxKernelXeon(const void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_, PropParms ppar, const double dz, const double c, const double d,
				const size_t Vo, const size_t Vf, FieldPrecision precision, const unsigned int bSizeX, const unsigned int bSizeY, const unsigned int bSizeZ)
{
	const size_t ct  = ppar.ct;
	const size_t NN  = ppar.Ng;
	const size_t Lx  = ppar.Lx;
	const size_t Sf = Lx*Lx;
	const size_t NSf = Sf*NN;
	const double *PC = ppar.PC;
	const double R    = ppar.R;
	const double beta = ppar.beta;
	const double ood2 = dz*d*ppar.ood2a/(2.0*ppar.massA*R);

	/* In Time-spliting, the NL part can be integrated exactly
		|P|^2 is constant
		beta(t) can be integrated numerically
		but if beta is power law of scale factor, like -1/4 R^2 = -1/4 1/ct^2frw
		using 2frw-1 = u
		then K = -|P|^2/4 * (1/u) [ (t0+dt)^u - (t0)^u  ]/(t0+dt)^u(t0)^u
		for frw = 0, it works (linear case)
		for frw = 1, RD is also smooth
		beware of 1/2 case!

		We kick with laplacian with c
		we drift" with selfinteractions with d
		*/

	/*We define Kt positive for beta negative (the sign is applied below mpVec)*/

	// double u, Kt, sKt, cKt;
	// if (ppar.frw != 0.5) {
	// 	u  = 2*ppar.frw - 1.0;
	// 	Kt = ct*(pow(ct+dz*d,u)-pow(ct,u))/(4.0*R*R*u*pow(ct+dz*d,u));
	// } else {
	// 	Kt = ct/(4.0*R*R)*log(1.0+dz*d/ct);
	// }
	const double u   = 2*ppar.frw - 1.0;
	const double KKt = ct*beta*(pow(ct+dz*d,u)-pow(ct,u))/(4.0*R*R*u*pow(ct+dz*d,u));

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

		_MData_ COV[5];
		for (size_t nv = 0; nv < NN ; nv++)
			COV[nv]  = opCode(set1_pd, PC[nv]*ood2);

#if	defined(__AVX512F__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const double    __attribute__((aligned(Align))) mpAux[8] = {-1., 1.,-1., 1.,-1., 1.,-1., 1.};	// to complex congugate
		const long long __attribute__((aligned(Align))) shfRg[8] = {6, 7, 0, 1, 2, 3, 4, 5 };
		const long long __attribute__((aligned(Align))) shfLf[8] = {2, 3, 4, 5, 6, 7, 0, 1 };

		const _MInt_  vShRg = opCode(load_si512, shfRg);
		const _MInt_  vShLf = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) mpAux[4] = {-1.,1.,-1.,1.};	// to complex congugate
#else
		const size_t XC = Lx;
		const size_t YC = Lx;

		const double __attribute__((aligned(Align))) mpAux[2] = { -1.,1. };	// to complex congugate
#endif
		const _MData_ mpVec  = opCode(load_pd, mpAux);
		const _MData_ KKtVec = opCode(set1_pd, KKt);

		const uint z0 = Vo/(Lx*Lx);
		const uint zF = Vf/(Lx*Lx);
		const uint zM = (zF-z0+bSizeZ-1)/bSizeZ;
		const uint bY = (YC + bSizeY - 1)/bSizeY;

LogMsg(VERB_DEBUG,"[pX] z0 %d zF %d zM %d bY %d bSizeZ %d bSizeY %d [NN %d]",z0, zF, zM, bY, bSizeZ, bSizeY, NN);LogFlush();

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
			size_t idx = zC*(YC*XC) + yC*XC + xC;

			if (idx >= Vf)
				continue;
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

			/* Laplacian stored in lap including 1/2cmA */

				 /* Velocity P_ct = I LapP/2cmA  + I |P|^2/4R^2 P

				1 - kick with laplacian */

 #if	defined(__AVX512F__)
 			tmp = opCode(mul_pd, mpVec, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, lap), _MM_PERM_BADC)));
 #elif	defined(__AVX__)
 			tmp = opCode(mul_pd, mpVec, opCode(permute_pd, lap, 0b00000101));
 #else
 			tmp = opCode(mul_pd, mpVec, opCode(shuffle_pd, lap, lap, 0b00000001));
 #endif
			// update
			mPx = opCode(add_pd, mel, tmp);

		/* mPx is intermediate value of P and is used to compute the next kick
			 2 - kick with potential */

			mPy = opCode(mul_pd, mPx, mPx);

#if	defined(__AVX512F__)
			tmp = opCode(add_pd, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, mPy), _MM_PERM_BADC)), mPy);
#elif	defined(__AVX__)
			tmp = opCode(add_pd, opCode(permute_pd, mPy, 0b00000101), mPy);
#else
			tmp = opCode(add_pd, opCode(shuffle_pd, mPy, mPy, 0b00000001), mPy);
#endif

			/* |P|^2 stored in tmp
			all orders update
			q = q0 cos + p0 sin
			p = p0 cos - q0 sin

			first build the cos (trivial
			and sin (requiring a permutation)
			vectors from tmp,
			then add them
			*/

			mPy = opCode(mul_pd, mPx, opCode(cos_pd, opCode(mul_pd, KKtVec, tmp)));
			tmp = opCode(mul_pd, mPx, opCode(sin_pd, opCode(mul_pd, KKtVec, tmp)));    // (q0,p0)sinKt permute and sign change!

#if	defined(__AVX512F__)
			lap = opCode(mul_pd, mpVec, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, tmp), _MM_PERM_BADC)));
#elif	defined(__AVX__)
			lap = opCode(mul_pd, mpVec, opCode(permute_pd, tmp, 0b00000101));
#else
			lap = opCode(mul_pd, mpVec, opCode(shuffle_pd, tmp, tmp, 0b00000001));
#endif

			tmp = opCode(add_pd, lap, mPy);

			opCode(stream_pd, &m2[idxP0], tmp);
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

		// Factors for the drift with self-interactions
		const float KKtf = KKt;

		// Factor for the "kick" with laplacian including dz
		// dz * d * inverse lattice spacing^2/ (2 cmA)
		const float ood2f = ood2;

		_MData_ COV[5];
		for (size_t nv = 0; nv < NN ; nv++)
			COV[nv]  = opCode(set1_ps, PC[nv]*ood2f);

#if	defined(__AVX512F__)
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) mpAux[16] = { -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f };

		const int   __attribute__((aligned(Align))) shfRg[16] = {14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
		const int   __attribute__((aligned(Align))) shfLf[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1};

		const _MInt_  vShRg  = opCode(load_si512, shfRg);
		const _MInt_  vShLf  = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) mpAux[8]  = { -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f };
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const float __attribute__((aligned(Align))) mpAux[4]  = { -1.f, 1.f, -1.f, 1.f };
#endif
		const _MData_ mpVec  = opCode(load_ps, mpAux);
		const _MData_ KKtVec = opCode(set1_ps, KKtf);

		const uint z0 = Vo/(Lx*Lx);
		const uint zF = Vf/(Lx*Lx);
		const uint zM = (zF-z0+bSizeZ-1)/bSizeZ;
		const uint bY = (YC + bSizeY - 1)/bSizeY;

LogMsg(VERB_DEBUG,"[pX] z0 %d zF %d zM %d bY %d bSizeZ %d bSizeY %d [NN %d]",z0, zF, zM, bY, bSizeZ, bSizeY, NN);LogFlush();

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

			if (idx >= Vf)
				continue;

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

		/* Laplacian stored in lap including dzd /2cmA delta^2 */

			 /* Velocity P_ct = I LapP/2cmA  + I |P|^2/4R^2 P

			1 - kick with laplacian */

#if	defined(__AVX__)// || defined(__AVX512F__)
			tmp = opCode(mul_ps, mpVec, opCode(permute_ps, lap, 0b10110001));
#else
			tmp = opCode(mul_ps, mpVec, opCode(shuffle_ps, lap, lap, 0b10110001));
#endif
			// update
			mPx = opCode(add_ps, mel, tmp);

			/* mPx is intermediate value of P and is used to compute the next kick
				 2 - kick with potential */

			mPy = opCode(mul_ps, mPx, mPx);		// M1^2 M2^2

#if	defined(__AVX__)// || defined(__AVX512F__)
			tmp = opCode(add_ps, opCode(permute_ps, mPy, 0b10110001), mPy);
#else
			tmp = opCode(add_ps, opCode(shuffle_ps, mPy, mPy, 0b10110001), mPy);
#endif

			/* |P|^2 stored in tmp
			all orders update
			q = q0 cos + p0 sin
			p = p0 cos - q0 sin

			first build the cos (trivial
			and sin (requiring a permutation)
			vectors from tmp,
			then add them
			*/

			mPy = opCode(mul_ps, mPx, opCode(cos_ps, opCode(mul_ps, KKtVec, tmp)));
			tmp = opCode(mul_ps, mPx, opCode(sin_ps, opCode(mul_ps, KKtVec, tmp)));    // (q0,p0)sinKt permute and sign change!

#if	defined(__AVX__)// || defined(__AVX512F__)
			lap = opCode(mul_ps, mpVec, opCode(permute_ps, tmp, 0b10110001));
#else
			lap = opCode(mul_ps, mpVec, opCode(shuffle_ps, tmp, tmp, 0b10110001));
#endif

			tmp = opCode(add_ps, lap, mPy);

			opCode(stream_ps, &m2[idxP0], tmp);	// Avoids cache thrashing
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
