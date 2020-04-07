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
template<const KickDriftType KDtype>
inline	void	propagatePaxKernelXeon(const void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_, PropParms ppar, const double dz,
				const size_t Vo, const size_t Vf, FieldPrecision precision, const unsigned int bSizeX, const unsigned int bSizeY, const unsigned int bSizeZ)
{
	const double ct   = ppar.ct;
	const size_t NN   = ppar.Ng;
	const size_t Lx   = ppar.Lx;
	const size_t Sf   = Lx*Lx;
	const size_t NSf  = Sf*NN;
	const double *PC  = ppar.PC;
	const double R    = ppar.R;

	const double beta = ppar.beta;
	const double u    = 2.0*ppar.frw - 1.0;
	const double KKt  = ppar.sign*ppar.beta*ct*(pow(ct+dz,u)-pow(ct,u))/(8.0*R*R*u*pow(ct+dz,u));
	LogMsg(VERB_DEBUG,"PPX ct  %e dz  %e FRW %f R %e u %f sign %d beta %f",ct,dz,ppar.frw,R,u,ppar.sign,ppar.beta);
	LogMsg(VERB_DEBUG,"PPX KKt %e ");
	/* integrate in time assuming powerlaw int d z/ m_A R */
	const double grav = -ppar.massA*ppar.grav*dz;
	double alpho = (ppar.n - 1);
	double dzp = (std::abs(alpho) < 1.e-2) ? ct*std::log(1.0 + dz/ct) : ct/alpho*(1. - pow(1.+dz/ct,-alpho));
	const double ood2 = ppar.sign*dzp*ppar.ood2a/(2.0*ppar.massA*R);
	LogMsg(VERB_DEBUG,"PPX od2 %e dct %e dcp %e n %.2f alpho %.e",ood2, dz, dzp, ppar.n,alpho);
	LogMsg(VERB_DEBUG,"mA grav %e",grav);

	if (precision == FIELD_DOUBLE)
	{
#ifdef	__AVX512F__
	#define	_MData_ __m512d
	#define	step 8
#elif	defined(__AVX__)
	#define	_MData_ __m256d
	#define	step 4
#else
	#define	_MData_ __m128d
	#define	step 2
#endif

		double * __restrict__ m  = (double * __restrict__) __builtin_assume_aligned (m_, Align);
		double * __restrict__ v  = (double * __restrict__) __builtin_assume_aligned (v_, Align);
		double * __restrict__ m2 = (double * __restrict__) __builtin_assume_aligned (m2_, Align);

		_MData_ COV[5];
		for (size_t nv = 0; nv < NN ; nv++)
			COV[nv]  = opCode(set1_pd, PC[nv]*ood2);

#ifdef	__AVX512F__
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
		const _MData_ m6Vec  = opCode(set1_pd, -6.0);
		const _MData_ KKtVec = opCode(set1_pd, KKt);
		const _MData_ graVec = opCode(set1_pd, grav);


#ifdef	__AVX512F__
		const auto vShRg  = opCode(load_si512, shfRg);
		const auto vShLf  = opCode(load_si512, shfLf);
#endif

		const uint z0 = Vo/(Lx*Lx);
		const uint zF = Vf/(Lx*Lx);
		const uint zM = (zF - z0 + bSizeZ - 1)/bSizeZ;
		const uint bY = (YC      + bSizeY - 1)/bSizeY;

		for (uint zT = 0; zT < zM; zT++)
		 for (uint yT = 0; yT < bY; yT++)
		  #pragma omp parallel default(shared)
		  {
		    _MData_ tmp, mel, vel, mPy, mMy, acu, lap;

		    #pragma omp for collapse(3) schedule(static)
		    for (uint zz = 0; zz < bSizeZ; zz++) {
		     for (uint yy = 0; yy < bSizeY; yy++) {
		      for (uint xC = 0; xC < XC; xC += step) {
			uint zC = zz + bSizeZ*zT + z0;
			uint yC = yy + bSizeY*yT;

			size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz;
			size_t idx = zC*(YC*XC) + yC*XC + xC;

			if (idx >= Vf)
				continue;
			{
				X[0] = xC;
				X[1] = yC;
			}

			mel = opCode(load_pd, &m[idx]);


			switch(KDtype)
			{
				case KIDI_LAP:
				{
							lap = opCode(set1_pd, 0.0); // for the laplacian

							for (size_t nv=1; nv < NN+1; nv++)
							{
								if (X[0] < nv*step)
									idxMx = ( idx + XC - nv*step );
								else
									idxMx = ( idx - nv*step );
								//x+
								if (X[0] + nv*step >= XC)
									idxPx = ( idx - XC + nv*step );
								else
									idxPx = ( idx + nv*step );

								if (X[1] < nv )
								{
									idxMy = ( idx + Sf - nv*XC );
									idxPy = ( idx + nv*XC );

									mPy = opCode(load_pd, &m[idxPy]);
#ifdef	__AVX512F__
									mMy = opCode(permutexvar_pd, vShRg, opCode(load_pd, &m[idxMy]));
#elif	defined(__AVX2__)
									mMy = opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_pd, &m[idxMy])), opCode(setr_epi32, 6,7,0,1,2,3,4,5)));
#elif	defined(__AVX__)
									acu = opCode(permute_pd, opCode(load_pd, &m[idxMy]), 0b00000101);
									vel = opCode(permute2f128_pd, acu, acu, 0b00000001);
									mMy = opCode(blend_pd, acu, vel, 0b00000101);
#else
									acu = opCode(load_pd, &m[idxMy]);
									mMy = opCode(shuffle_pd, acu, acu, 0x00000001);
#endif
								}
else
								{
									idxMy = ( idx - nv*XC );
									mMy = opCode(load_pd, &m[idxMy]);

									if (X[1] + nv >= YC)
									{
										idxPy = ( idx + nv*XC - Sf );
#ifdef	__AVX512F__
										mPy = opCode(permutexvar_pd, vShLf, opCode(load_pd, &m[idxPy]));
#elif	defined(__AVX2__)	//AVX2
										mPy = opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_pd, &m[idxPy])), opCode(setr_epi32, 2,3,4,5,6,7,0,1)));
#elif	defined(__AVX__)
										acu = opCode(permute_pd, opCode(load_pd, &m[idxPy]), 0b00000101);
										vel = opCode(permute2f128_pd, acu, acu, 0b00000001);
										mPy = opCode(blend_pd, acu, vel, 0b00001010);
#else
										vel = opCode(load_pd, &m[idxPy]);
										mPy = opCode(shuffle_pd, vel, vel, 0x00000001);
#endif
									}
									else
									{
										idxPy = ( idx + nv*XC );
										mPy = opCode(load_pd, &m[idxPy]);
									}
								}

								// sum Y+Y-
								// sum X+X-
								// sum Z+Z-
								idxPz = idx+nv*Sf;
								idxMz = idx-nv*Sf;

								acu = 	opCode(add_pd,
													opCode(add_pd,
														opCode(add_pd, mPy, mMy),
															opCode(add_pd,
																opCode(add_pd, opCode(load_pd, &m[idxPx]), opCode(load_pd, &m[idxMx])),
																	opCode(add_pd, opCode(load_pd, &m[idxPz]), opCode(load_pd, &m[idxMz]))
															)
													),
													opCode(mul_pd, mel, m6Vec));

								lap = opCode(add_pd, lap, opCode(mul_pd,acu,COV[nv-1]));

							} // End neighbour loop

							tmp = opCode(add_pd, opCode(load_pd, &v[idx]), lap);
							opCode(store_pd, &v[idx], tmp);

				} // end LAP case
				break;

				case KIDI_POT:
				{
						vel = opCode(load_pd, &v[idx]);
						acu = opCode(mul_pd, KKtVec, opCode(add_pd, opCode(mul_pd,vel,vel), opCode(mul_pd,mel,mel)));
						mMy = opCode(sin_pd, acu);
						mPy = opCode(cos_pd, acu);
						tmp = opCode(sub_pd, opCode(mul_pd, mPy, mel), opCode(mul_pd, mMy, vel));
						opCode(store_pd, &m[idx], tmp);
						tmp = opCode(add_pd, opCode(mul_pd, mPy, vel), opCode(mul_pd, mMy, mel));
						opCode(store_pd, &v[idx], tmp);
				}
				break;
				case KIDI_POT_GRAV:
				{
						vel = opCode(load_pd, &v[idx]);
						acu = opCode(mul_pd, KKtVec, opCode(add_pd, opCode(mul_pd,vel,vel), opCode(mul_pd,mel,mel)));
						acu = opCode(add_pd, acu, opCode(mul_pd, graVec, opCode(load_pd, &m2[idx])));
						mMy = opCode(sin_pd, acu);
						mPy = opCode(cos_pd, acu);
						tmp = opCode(sub_pd, opCode(mul_pd, mPy, mel), opCode(mul_pd, mMy, vel));
						opCode(store_pd, &m[idx], tmp);
						tmp = opCode(add_pd, opCode(mul_pd, mPy, vel), opCode(mul_pd, mMy, mel));
						opCode(store_pd, &v[idx], tmp);
				}
				break;
			} // End prepare cases
		   }
		  }
		 }
		}
#undef	_MData_
#undef	step
	}
	else if (precision == FIELD_SINGLE)
	{
#ifdef	__AVX512F__
	#define	_MData_ __m512
	#define	step 16
#elif	defined(__AVX__)
	#define	_MData_ __m256
	#define	step 8
#else
	#define	_MData_ __m128
	#define	step 4
#endif

		float * __restrict__ m	= (float * __restrict__) __builtin_assume_aligned (m_, Align);
		float * __restrict__ v	= (float * __restrict__) __builtin_assume_aligned (v_, Align);
		float * __restrict__ m2	= (float * __restrict__) __builtin_assume_aligned (m2_, Align);

		// // conformal axion mass
		// const float cmA   = ppar.massA*ppar.R;
		// // inverse lattice spacing^2/2 cmA
		// const float ood2 = ppar.sign*ppar.ood2a/(2.0*cmA);
		// const float i4R2 = ppar.sign*ppar.beta*1.0/(4.0*ppar.R*ppar.R);

		// Factors for the drift with self-interactions
		const float KKtf = KKt;
		const float gravf = grav;

		// Factor for the "kick" with laplacian including dz
		// dz * d * inverse lattice spacing^2/ (2 cmA)
		const float ood2f = ood2;

		_MData_ COV[5];
		for (size_t nv = 0; nv < NN ; nv++)
			COV[nv]  = opCode(set1_ps, PC[nv]*ood2f);

#ifdef	__AVX512F__
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

		const _MData_ m6Vec  = opCode(set1_ps, -6.f);
		const _MData_ KKtVec = opCode(set1_ps, KKtf);
		const _MData_ graVec = opCode(set1_ps, gravf);//i4R2);

		const uint z0 = Vo/(Lx*Lx);
		const uint zF = Vf/(Lx*Lx);
		const uint zM = (zF - z0 + bSizeZ - 1)/bSizeZ;
		const uint bY = (YC      + bSizeY - 1)/bSizeY;

		for (uint zT = 0; zT < zM; zT++)
		 for (uint yT = 0; yT < bY; yT++)
		  #pragma omp parallel default(shared)
		  {
		    _MData_ tmp, mel, vel, mPy, mMy, acu, lap;

		    #pragma omp for collapse(3) schedule(static)
		    for (uint zz = 0; zz < bSizeZ; zz++) {
		     for (uint yy = 0; yy < bSizeY; yy++) {
		      for (uint xC = 0; xC < XC; xC += step) {
			uint zC = zz + bSizeZ*zT + z0;
			uint yC = yy + bSizeY*yT;

			size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz;
			size_t idx = zC*(YC*XC) + yC*XC + xC;

			if (idx >= Vf)
				continue;
			{
				X[0] = xC;
				X[1] = yC;
			}

			mel = opCode(load_ps, &m[idx]);

			switch(KDtype)
			{
				case KIDI_LAP:
				{
							lap = opCode(set1_ps, 0.f); // for the laplacian


							for (size_t nv=1; nv < NN+1; nv++)
							{
								if (X[0] < nv*step)
									idxMx = ( idx + XC - nv*step );
								else
									idxMx = ( idx - nv*step );
								//x+
								if (X[0] + nv*step >= XC)
									idxPx = ( idx - XC + nv*step );
								else
									idxPx = ( idx + nv*step );

								if (X[1] < nv )
								{
									idxMy = ( idx + Sf - nv*XC );
									idxPy = ( idx + nv*XC );

									mPy = opCode(load_ps, &m[idxPy]);
#ifdef	__AVX512F__
									mMy = opCode(permutexvar_ps, vShRg, opCode(load_ps, &m[idxMy]));
#elif	defined(__AVX2__)
									mMy = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 7,0,1,2,3,4,5,6));
#elif	defined(__AVX__)
									tmp = opCode(permute_ps, opCode(load_ps, &m[idxMy]), 0b10010011);
									vel = opCode(permute2f128_ps, tmp, tmp, 0b00000001);
									mMy = opCode(blend_ps, tmp, vel, 0b00010001);
#else
									tmp = opCode(load_ps, &m[idxMy]);
									mMy = opCode(shuffle_ps, tmp, tmp, 0b10010011);
#endif
								}
								else
								{
									idxMy = ( idx - nv*XC );
									mMy = opCode(load_ps, &m[idxMy]);

									if (X[1] + nv >= YC)
									{
										idxPy = ( idx + nv*XC - Sf );
#ifdef	__AVX512F__
										mPy = opCode(permutexvar_ps, vShLf, opCode(load_ps, &m[idxPy]));
#elif	defined(__AVX2__)
										mPy = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 1,2,3,4,5,6,7,0));
#elif	defined(__AVX__)
										tmp = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b00111001);
										vel = opCode(permute2f128_ps, tmp, tmp, 0b00000001);
										mPy = opCode(blend_ps, tmp, vel, 0b10001000);
#else
										vel = opCode(load_ps, &m[idxPy]);
										mPy = opCode(shuffle_ps, vel, vel, 0b00111001);
#endif
									}
									else
									{
										idxPy = ( idx + nv*XC );
										mPy = opCode(load_ps, &m[idxPy]);
									}
								}
								// sum Y+Y-
								// sum X+X-
								// sum Z+Z-
								idxPz = idx+nv*Sf;
								idxMz = idx-nv*Sf;

								acu = 	opCode(add_ps,
													opCode(add_ps,
														opCode(add_ps, mPy, mMy),
															opCode(add_ps,
																opCode(add_ps, opCode(load_ps, &m[idxPx]), opCode(load_ps, &m[idxMx])),
																	opCode(add_ps, opCode(load_ps, &m[idxPz]), opCode(load_ps, &m[idxMz]))
															)
													),
													opCode(mul_ps, mel, m6Vec));

								lap = opCode(add_ps, lap, opCode(mul_ps, acu, COV[nv-1]));

							} // End neighbour loop
							vel = opCode(load_ps, &v[idx]);
							tmp = opCode(add_ps, vel, lap);
							opCode(store_ps, &v[idx], tmp);

							// if (idx == Vo){
							// 	printf("idx Vo z y x %u %u %u\n ",zC,yC,xC);
							// 	printsVar(mel, "mel");
							// 	printsVar(lap, "lap");
							// 	printsVar(vel, "vel");
							// 	printsVar(tmp, "+++");
							// 	}

				} //end LAP cases
				break;

				case KIDI_POT:
				{
						vel = opCode(load_ps, &v[idx]);
						acu = opCode(mul_ps, KKtVec, opCode(add_ps, opCode(mul_ps,vel,vel), opCode(mul_ps,mel,mel)));
						mMy = opCode(sin_ps, acu);
						mPy = opCode(cos_ps, acu);
						tmp = opCode(sub_ps, opCode(mul_ps, mPy, mel), opCode(mul_ps, mMy, vel));
						opCode(store_ps, &m[idx], tmp);
						tmp = opCode(add_ps, opCode(mul_ps, mPy, vel), opCode(mul_ps, mMy, mel));
						opCode(store_ps, &v[idx], tmp);
				}
				break;

				case KIDI_POT_GRAV:
				{
						vel = opCode(load_ps, &v[idx]);
						acu = opCode(mul_ps, KKtVec, opCode(add_ps, opCode(mul_ps,vel,vel), opCode(mul_ps,mel,mel)));
						acu = opCode(add_ps, acu, opCode(mul_ps, graVec, opCode(load_ps, &m2[idx])));
						mMy = opCode(sin_ps, acu);
						mPy = opCode(cos_ps, acu);
						tmp = opCode(sub_ps, opCode(mul_ps, mPy, mel), opCode(mul_ps, mMy, vel));
						opCode(store_ps, &m[idx], tmp);
						tmp = opCode(add_ps, opCode(mul_ps, mPy, vel), opCode(mul_ps, mMy, mel));
						opCode(store_ps, &v[idx], tmp);
				}
				break;
			} // End switch prepare cases


		      }
		    }
		  }
		} //end loop
#undef	_MData_
#undef	step
} // end single
}







#undef	opCode
#undef	opCode_N
#undef	opCode_P
#undef	Align
#undef	_PREFIX_
