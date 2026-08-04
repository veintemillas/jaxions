#include <cstdio>
#include <cmath>
#include "scalar/scalarField.h"
#include "enum-field.h"
//#include "scalar/varNQCD.h"
#include "utils/parse.h"

#include "utils/triSimd.h"

#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

#include <immintrin.h>

#ifdef	__AVX512F__
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

template<const bool wMod, const VqcdType VQcd>
inline	void	propThetaKernelXeon(const void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_, const PropParms ppar, const double dz, const double c, const double d,
						const size_t Vo, const size_t Vf, FieldPrecision precision, const unsigned int bSizeX, const unsigned int bSizeY, const unsigned int bSizeZ)
{
	const size_t NN   = ppar.Ng;
	const size_t Lx   = ppar.Lx;
	const size_t Sf   = Lx*Lx;
	const size_t NSf  = Sf*NN;
	const double *PC  = ppar.PC;
	const double R    = ppar.R;
	const double ood2 = ppar.ood2a;
	const double Rpp  = ppar.Rpp;
	const double mA2  = ppar.massA2;

	const double beta  = ppar.beta;


	if (Vo>Vf)
		return ;

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

		const double * __restrict__ m	= (const double * __restrict__) __builtin_assume_aligned (m_, Align);
		double * __restrict__ v		= (double * __restrict__) __builtin_assume_aligned (v_, Align);
		double * __restrict__ m2	= (double * __restrict__) __builtin_assume_aligned (m2_, Align);

		const double dzc = dz*c;
		const double dzd = dz*d;
		const double zQ = mA2*R*R*R;
		const double iz = 1.0/R;
		const double tV	= 2.*M_PI*R;

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
		const _MData_ tpVec  = opCode(set1_pd, tV);
		const _MData_ zQVec  = opCode(set1_pd, zQ);
		const _MData_ d2Vec  = opCode(set1_pd, ood2);
		const _MData_ dzcVec = opCode(set1_pd, dzc);
		const _MData_ dzdVec = opCode(set1_pd, dzd);
		const _MData_ izVec  = opCode(set1_pd, iz);

		const _MData_ PiVec  = opCode(set1_pd,  M_PI);
		const _MData_ iPVec  = opCode(set1_pd, -M_PI);

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

			// If YC (or zF-z0) is not divisible by bSizeY (bSizeZ), there is a possibility of exceeding the assumed domain in the last block.
			// This may be avoided by adjusting bSizeY (bSizeZ) in tunePropagator.
			if ((yC >= YC) || (zC >= zF)) continue;
			{
				//size_t tmi = idx/XC, itp;

				//itp = tmi/YC;
				//X[1] = tmi - itp*YC;
				//X[0] = idx - tmi*XC;
				X[0] = xC;
				X[1] = yC;
			}

			mel = opCode(load_pd, &m[idx]);
			lap = opCode(set1_pd, 0.0); // for the laplacian
/*
		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, vel, mPy, mMy, acu;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxPx, idxMx, idxPy, idxMy, idxPz, idxMz;

				mel = opCode(load_pd, &m[idx]);

				{
					size_t tmi = idx/XC, tpi;

					tpi = tmi/YC;
					X[1] = tmi - tpi*YC;
					X[0] = idx - tmi*XC;
				}
*/
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

				if (wMod) {
					/*	idxPx	*/

					vel = opCode(sub_pd, opCode(load_pd, &m[idxPx]), mel);
					acu = opCode(mod_pd, vel, tpVec);

					/*	idxMx	*/

					vel = opCode(sub_pd, opCode(load_pd, &m[idxMx]), mel);
					acu = opCode(add_pd, opCode(mod_pd, vel, tpVec), acu);

					/*	idxPz	*/

					vel = opCode(sub_pd, opCode(load_pd, &m[idxPz]), mel);
					acu = opCode(add_pd, opCode(mod_pd, vel, tpVec), acu);

					/*	idxMz	*/

					vel = opCode(sub_pd, opCode(load_pd, &m[idxMz]), mel);
					acu = opCode(add_pd, opCode(mod_pd, vel, tpVec), acu);

					/*	idxPy	*/

					vel = opCode(sub_pd, mPy, mel);
					acu = opCode(add_pd, opCode(mod_pd, vel, tpVec), acu);

					/*	idxMy	*/

					vel = opCode(sub_pd, mMy, mel);
					acu = opCode(add_pd, opCode(mod_pd, vel, tpVec), acu);

				}
				else {
				acu = 	opCode(add_pd,
									opCode(add_pd,
										opCode(add_pd, mPy, mMy),
											opCode(add_pd,
												opCode(add_pd, opCode(load_pd, &m[idxPx]), opCode(load_pd, &m[idxMx])),
													opCode(add_pd, opCode(load_pd, &m[idxPz]), opCode(load_pd, &m[idxMz]))
											)
									),
									opCode(mul_pd, mel, opCode(set1_pd, -6.0)));
				}

				lap = opCode(add_pd, lap, opCode(mul_pd,acu,COV[nv-1]));

			} // End neighbour loop

			/* Acceleration
					lap - mA2R3 sin(psi/R) + Rpp psi*/
			switch(VQcd){
				default:
				case V_QCDC:
					acu = opCode(sub_pd, lap,
									opCode(sub_pd, opCode(mul_pd, zQVec, opCode(sin_pd, opCode(mul_pd, mel, izVec))),
										opCode(mul_pd, opCode(set1_pd, Rpp), mel)));
				break;

				case V_QCD0:
				acu = opCode(add_pd, lap, opCode(mul_pd, opCode(set1_pd, Rpp), mel));
				break;

				case V_QCDL:
					acu = opCode(sub_pd, lap,
									opCode(sub_pd, opCode(mul_pd, zQVec, opCode(mul_pd, mel, izVec)),
										opCode(mul_pd, opCode(set1_pd, Rpp), mel)));
				break;

				case V_QCDS:
				vel = opCode(max_pd, iPVec, opCode(min_pd, PiVec, opCode(mul_pd, mel, izVec)));
				acu = opCode(sub_pd, lap,
								opCode(sub_pd, opCode(mul_pd, zQVec, opCode(sin_pd, vel )),
									opCode(mul_pd, opCode(set1_pd, Rpp), mel)));
				break;
			}
			/* Update  */
			vel = opCode(load_pd, &v[idx-NSf]);

#if	defined(__AVX512F__) || defined(__FMA__)
			tmp = opCode(fmadd_pd, acu, dzcVec, vel);
			mMy = opCode(fmadd_pd, tmp, dzdVec, mel);
#else
			tmp = opCode(add_pd, vel, opCode(mul_pd, acu, dzcVec));
			mMy = opCode(add_pd, mel, opCode(mul_pd, tmp, dzdVec));
#endif

			/*	Store	*/

			if (wMod)
				mMy = opCode(mod_pd, mMy, tpVec);

			opCode(store_pd, &v[idx-NSf], tmp);
			opCode(store_pd, &m2[idx],  mMy);
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

		const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_, Align);
		float * __restrict__ v		= (float * __restrict__) __builtin_assume_aligned (v_, Align);
		float * __restrict__ m2		= (float * __restrict__) __builtin_assume_aligned (m2_, Align);

		const float dzc = dz*c;
		const float dzd = dz*d;
		const float Rf = R;
		const float zQ = (float) (mA2*Rf*Rf*Rf);
		const float iz = 1.f/Rf;
		const float tV = 2.*M_PI*Rf;

		const float betaf = (float) beta;

		_MData_ COV[5];
		for (size_t nv = 0; nv < NN ; nv++)
			COV[nv]  = opCode(set1_ps, PC[nv]*ood2);

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
		const _MData_ tpVec  = opCode(set1_ps, tV);
		const _MData_ zQVec  = opCode(set1_ps, zQ);
		const _MData_ d2Vec  = opCode(set1_ps, ood2);
		const _MData_ dzcVec = opCode(set1_ps, dzc);
		const _MData_ dzdVec = opCode(set1_ps, dzd);
		const _MData_ izVec  = opCode(set1_ps, iz);

		const _MData_ PiVec  = opCode(set1_ps, (float)  M_PI);
		const _MData_ iPVec  = opCode(set1_ps, (float) -M_PI);

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

			if ((yC >= YC) || (zC >= zF)) continue;
			{
				//size_t tmi = idx/XC, itp;

				//itp = tmi/YC;
				//X[1] = tmi - itp*YC;
				//X[0] = idx - tmi*XC;
				X[0] = xC;
				X[1] = yC;
			}

			mel = opCode(load_ps, &m[idx]);
			lap = opCode(set1_ps, 0.f); // for the laplacian
/*
		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, vel, mPy, mMy, acu;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz;

				mel = opCode(load_ps, &m[idx]);

				{
					size_t tmi = idx/XC, itp;

					itp = tmi/YC;
					X[1] = tmi - itp*YC;
					X[0] = idx - tmi*XC;
				}
*/
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

				if (wMod) {
					/*	idxPx	*/

					vel = opCode(sub_ps, opCode(load_ps, &m[idxPx]), mel);
					acu = opCode(mod_ps, vel, tpVec);

					/*	idxMx	*/

					vel = opCode(sub_ps, opCode(load_ps, &m[idxMx]), mel);
					acu = opCode(add_ps, opCode(mod_ps, vel, tpVec), acu);

					/*	idxPz	*/

					vel = opCode(sub_ps, opCode(load_ps, &m[idxPz]), mel);
					acu = opCode(add_ps, opCode(mod_ps, vel, tpVec), acu);

					/*	idxMz	*/

					vel = opCode(sub_ps, opCode(load_ps, &m[idxMz]), mel);
					acu = opCode(add_ps, opCode(mod_ps, vel, tpVec), acu);

					/*	idxPy	*/

					vel = opCode(sub_ps, mPy, mel);
					acu = opCode(add_ps, opCode(mod_ps, vel, tpVec), acu);

					/*	idxMy	*/

					vel = opCode(sub_ps, mMy, mel);
					acu = opCode(add_ps, opCode(mod_ps, vel, tpVec), acu);

				} else {
					acu = 	opCode(add_ps,
										opCode(add_ps,
											opCode(add_ps, mPy, mMy),
												opCode(add_ps,
													opCode(add_ps, opCode(load_ps, &m[idxPx]), opCode(load_ps, &m[idxMx])),
														opCode(add_ps, opCode(load_ps, &m[idxPz]), opCode(load_ps, &m[idxMz]))
												)
										),
										opCode(mul_ps, mel, opCode(set1_ps, -6.f)));
				}

				lap = opCode(add_ps, lap, opCode(mul_ps, acu, COV[nv-1]));

			} // End neighbour loop


			switch(VQcd){
				default:
				case V_QCDC:
				acu = opCode(sub_ps, lap,
								opCode(sub_ps, opCode(mul_ps, zQVec, opCode(sin_ps, opCode(mul_ps, mel, izVec))),
									opCode(mul_ps, opCode(set1_ps, Rpp), mel)));
				break;

				case V_QCD0:
				acu = opCode(add_ps, lap, opCode(mul_ps, opCode(set1_ps, Rpp), mel));
				break;

				case V_QCDL:
				tmp = opCode(mul_ps, mel, izVec);
				vel = opCode(mul_ps, opCode(mul_ps, tmp, tmp), opCode(set1_ps, 0.1666667*betaf));
				tmp = opCode(mul_ps, tmp, opCode(sub_ps, opCode(set1_ps, 1), opCode(sub_ps, vel, opCode(mul_ps, opCode(mul_ps, vel, vel), opCode(set1_ps, 0.3)))));
				acu = opCode(sub_ps, lap,
								opCode(sub_ps, opCode(mul_ps, zQVec, tmp),
									opCode(mul_ps, opCode(set1_ps, Rpp), mel)));
				break;

				case V_QCDS:
				vel = opCode(max_ps, iPVec, opCode(min_ps, PiVec, opCode(mul_ps, mel, izVec)));
				acu = opCode(sub_ps, lap,
								opCode(sub_ps, opCode(mul_ps, zQVec, opCode(sin_ps, vel )),
									opCode(mul_ps, opCode(set1_ps, Rpp), mel)));
				break;
			}


			// this line kills axion self-interactions STERILE MODE!!
			//opCode(mul_ps, zQVec, opCode(mul_ps, mel, izVec)));

			vel = opCode(load_ps, &v[idx-NSf]);

#if	defined(__MIC__) || defined(__AVX512F__) || defined(__FMA__)
			tmp = opCode(fmadd_ps, acu, dzcVec, vel);
			mMy = opCode(fmadd_ps, tmp, dzdVec, mel);
#else
			tmp = opCode(add_ps, vel, opCode(mul_ps, acu, dzcVec));
			mMy = opCode(add_ps, mel, opCode(mul_ps, tmp, dzdVec));
#endif
			/*	Store	*/

			if (wMod)
				mMy = opCode(mod_ps, mMy, tpVec);

			opCode(store_ps, &v[idx-NSf], tmp);
			opCode(store_ps, &m2[idx],  mMy);
		      }
		    }
		  }
		}
#undef	_MData_
#undef	step
	}
}

inline	void	updateMThetaXeon(void * __restrict__ m_, const void * __restrict__ v_, const double dz, const double d, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision,
				 const unsigned int bSizeX, const unsigned int bSizeY, const unsigned int bSizeZ)
{
	const uint z0 = Vo/(Lx*Lx);
	const uint zF = Vf/(Lx*Lx);
	const uint zM = (zF-z0+bSizeZ-1)/bSizeZ;

	if (precision == FIELD_DOUBLE)
	{
	#if	defined(__AVX512F__)
		#define	_MData_ __m512d
		#define	step 8
		const size_t XC = (Lx<<4);
		const size_t YC = (Lx>>4);
	#elif	defined(__AVX__)
		#define	_MData_ __m256d
		#define	step 4
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);
	#else
		#define	_MData_ __m128d
		#define	step 2
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);
	#endif

		double * __restrict__ m		= (double * __restrict__) __builtin_assume_aligned (m_, Align);
		const double * __restrict__ v	= (const double * __restrict__) __builtin_assume_aligned (v_, Align);

		const double dzd = dz*d;

		const uint bY = (YC + bSizeY - 1)/bSizeY;

		for (uint zT = 0; zT < zM; zT++)
		 for (uint yT = 0; yT < bY; yT++)
		  #pragma omp parallel default(shared)
		  {
			_MData_ mIn, vIn, tmp;
			size_t idxV0;

			#pragma omp for collapse(3) schedule(static)
			for (uint zz = 0; zz < bSizeZ; zz++) {
		 	  for (uint yy = 0; yy < bSizeY; yy++) {
			    for (uint xC = 0; xC < XC; xC += step) {
			      uint zC = zz + bSizeZ*zT + z0;
			      uint yC = yy + bSizeY*yT;

			      auto idx = zC*(YC*XC) + yC*XC + xC;

			      if ((yC >= YC) || (zC >= zF)) continue;

			      idxV0 = idx - Vo;

#if	defined(__AVX512F__) || defined(__FMA__)
			      vIn = opCode(load_pd, &v[idxV0]);
			      mIn = opCode(load_pd, &m[idx]);
			      tmp = opCode(fmadd_pd, opCode(set1_pd, dzd), vIn, mIn);
			      opCode(store_pd, &m[idx], tmp);
#else
			      mIn = opCode(load_pd, &m[idx]);
			      tmp = opCode(load_pd, &v[idxV0]);
			      vIn = opCode(mul_pd, opCode(set1_pd, dzd), tmp);
			      tmp = opCode(add_pd, mIn, vIn);
			      opCode(store_pd, &m[idx], tmp);
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
		#define	step 16
		const size_t XC = (Lx<<4);
		const size_t YC = (Lx>>4);
	#elif	defined(__AVX__)
		#define	_MData_ __m256
		#define	step 8
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);
	#else
		#define	_MData_ __m128
		#define	step 4
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);
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
			_MData_ mIn, vIn, tmp;
			size_t idxV0;

			#pragma omp for collapse(3) schedule(static)
			for (uint zz = 0; zz < bSizeZ; zz++) {
		 	  for (uint yy = 0; yy < bSizeY; yy++) {
			    for (uint xC = 0; xC < XC; xC += step) {
			      uint zC = zz + bSizeZ*zT + z0;
			      uint yC = yy + bSizeY*yT;

			      auto idx = zC*(YC*XC) + yC*XC + xC;

			      if ((yC >= YC) || (zC >= zF)) continue;

			      idxV0 = idx - Vo;

#if	defined(__AVX512F__) || defined(__FMA__)
			      vIn = opCode(load_ps, &v[idxV0]);
			      mIn = opCode(load_ps, &m[idx]);
			      tmp = opCode(fmadd_ps, opCode(set1_ps, dzd), vIn, mIn);
			      opCode(store_ps, &m[idx], tmp);
#else
			      vIn = opCode(load_ps, &v[idxV0]);
			      mIn = opCode(load_ps, &m[idx]);
			      tmp = opCode(add_ps, mIn, opCode(mul_ps, opCode(set1_ps, dzd), vIn));
			      opCode(store_ps, &m[idx], tmp);
#endif
			    }
			  }
			}
		    }
#undef	_MData_
#undef	step
	}
}




inline	void	propThetaKernelXeon(const void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_, const PropParms ppar, const double dz, const double c, const double d,
				    const size_t Vo, const size_t Vf, FieldPrecision precision, const unsigned int bSizeX, const unsigned int bSizeY, const unsigned int bSizeZ, const bool wMod, const VqcdType VQcd)
{
	/* Warning! to avoid the false vacuum locking at large mA, we switch off V(theta) for theta>pi */
	// bool sat = (ppar.massA2*ppar.R*ppar.R > 27.62 * ppar.ood2a);
	bool sat =false;

	switch (VQcd & V_QCD)
	{
		case V_QCD0:
		case V_NONE:
		{
LogMsg(VERB_PARANOID,"[PT] propTheta QCD0");
				switch (wMod) {
					case	true:
						propThetaKernelXeon<true,V_QCD0> (m_, v_, m2_, ppar, dz, c, d, Vo, Vf, precision, bSizeX, bSizeY, bSizeZ);
						break;

					case	false:
						propThetaKernelXeon<false,V_QCD0>(m_, v_, m2_, ppar, dz, c, d, Vo, Vf, precision, bSizeX, bSizeY, bSizeZ);
						break;
				}
		} break;
		case V_QCD1:
		case V_QCDV:
		case V_QCDC:
		{
LogMsg(VERB_PARANOID,"[PT] propTheta QCDC (saturated %d = 1/0 true/false)",sat);
				switch (wMod) {
					case	true:
						propThetaKernelXeon<true,V_QCDC> (m_, v_, m2_, ppar, dz, c, d, Vo, Vf, precision, bSizeX, bSizeY, bSizeZ);
						break;

					case	false:
						if (!sat)
							propThetaKernelXeon<false,V_QCDC>(m_, v_, m2_, ppar, dz, c, d, Vo, Vf, precision, bSizeX, bSizeY, bSizeZ);
						else
							propThetaKernelXeon<false,V_QCDS>(m_, v_, m2_, ppar, dz, c, d, Vo, Vf, precision, bSizeX, bSizeY, bSizeZ);
						break;
				}
		} break;
		case V_QCDL:
		{
LogMsg(VERB_PARANOID,"[PT] propTheta QCDL %d",V_QCDL);
				switch (wMod) {
					case	true:
						propThetaKernelXeon<true,V_QCDL> (m_, v_, m2_, ppar, dz, c, d, Vo, Vf, precision, bSizeX, bSizeY, bSizeZ);
						break;

					case	false:
						propThetaKernelXeon<false,V_QCDL>(m_, v_, m2_, ppar, dz, c, d, Vo, Vf, precision, bSizeX, bSizeY, bSizeZ);
						break;
				}
		} break;
		default:
			LogError("PropXeon Case not recognised: exit!");
		break;
	}
}

/* This is the propagator for zero mode and linear fluctuations as decoupled fields
linear regime, includes gravity in the RHS ... woow
we always use it in double precision */

inline void propLinearModeKernelXeon(
	const void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_,
	const void * __restrict__ g_,
	const void * __restrict__ k_, const void * __restrict__ k2_,
	double *epsilon0_, double *epsilon0p_, const bool useEpsilonZeroMode,
	const PropParms ppar, const double dz, const double c, const double d)
{
#ifdef __AVX512F__
	#define _MData_ __m512d
	#define step 8
#elif defined(__AVX__)
	#define _MData_ __m256d
	#define step 4
#else
	#define _MData_ __m128d
	#define step 2
#endif

	const double * __restrict__ m  = (const double *) __builtin_assume_aligned(m_,  Align);
	double * __restrict__ v        = (double *) __builtin_assume_aligned(v_,  Align);
	double * __restrict__ m2       = (double *) __builtin_assume_aligned(m2_, Align);

	const double * __restrict__ g  = (const double *) __builtin_assume_aligned(g_,  Align);   // Phi_k(0)
	const double * __restrict__ kk = (const double *) __builtin_assume_aligned(k_,  Align);   // k
	const double * __restrict__ k2 = (const double *) __builtin_assume_aligned(k2_, Align);   // k^2

	const double dzc = dz*c;
	const double dzd = dz*d;

	const double R    = ppar.R;
	const double Rpp  = ppar.Rpp;   // R''/R
	const double Rp   = ppar.Rp;    // R'/R
	const double mA2  = ppar.massA2;
	const double eta  = ppar.ct;
	const double beta = ppar.n;  // Xi =  d log chi / d log T
	const bool   rhs  = !ppar.rhsoff;  // use the rhs?

	/* Save the old homogeneous background used by every fluctuation mode. */
	const double ctheta0  = m[0];
	const double ctheta0p = v[0];
	const double theta0   = useEpsilonZeroMode
		? std::acos(-1.0) - *epsilon0_
		: ctheta0/R;
	const double thpR     = useEpsilonZeroMode
		? -R*(*epsilon0p_)
		: ctheta0p - ctheta0*Rp;   // Theta' R

	const double mR2c = mA2*R*R*cos(theta0);
	const double mR2s = mA2*R*R*R*sin(theta0);

	const _MData_ dzcVec  = opCode(set1_pd, dzc);
	const _MData_ dzdVec  = opCode(set1_pd, dzd);
	const _MData_ RppVec  = opCode(set1_pd, Rpp);
	const _MData_ mR2cVec = opCode(set1_pd, mR2c);
	const _MData_ mR2sVec = opCode(set1_pd, mR2s);
	const _MData_ etaVec  = opCode(set1_pd, eta);
	const _MData_ thpRVec = opCode(set1_pd, thpR);
	const _MData_ b4Vec   = opCode(set1_pd, beta/4.0);
	const _MData_ t23Vec  = opCode(set1_pd, 2.0/3.0);
	const _MData_ twoVec  = opCode(set1_pd, 2.0);
	const _MData_ thrVec  = opCode(set1_pd, 3.0);
	const _MData_ m4Vec   = opCode(set1_pd, -4.0);
	const _MData_ iS3Vec  = opCode(set1_pd, 1.0/sqrt(3.0));


	#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < ppar.nmodes; i += step) {

		_MData_ mk   = opCode(load_pd, &m[i]);    // psi_k
		_MData_ vk   = opCode(load_pd, &v[i]);    // psi_k'
		_MData_ kv   = opCode(load_pd, &kk[i]);   // k
		_MData_ k2v  = opCode(load_pd, &k2[i]);   // k^2
		_MData_ src  = opCode(set1_pd, 0.0);

		if (rhs)
		{
			_MData_ phi0 = opCode(load_pd, &g[i]);  // Phi_k(0)

			// Useful common factors
			_MData_ phi03 = opCode(mul_pd, phi0, thrVec);     // 3*Phi_k(0)
			_MData_ kis3  = opCode(mul_pd, kv, iS3Vec);       // k/sqrt(3)

			// x = k eta / sqrt(3)
			_MData_ x  = opCode(mul_pd, kis3, etaVec);
			_MData_ x2 = opCode(mul_pd, x, x);
			_MData_ x3 = opCode(mul_pd, x2, x);
			_MData_ x4 = opCode(mul_pd, x2, x2);

			_MData_ sx = opCode(sin_pd, x);
			_MData_ cx = opCode(cos_pd, x);

			// numPhi = sin(x) - x cos(x)
			_MData_ numPhi = opCode(sub_pd, sx, opCode(mul_pd, x, cx));

			// Phi_k = 3 Phi_k(0) [sin(x) - x cos(x)] / x^3
			_MData_ phik = opCode(mul_pd, phi03, opCode(div_pd, numPhi, x3));

			// numGp = (x^2 - 3) sin(x) + 3 x cos(x)
			_MData_ numGp = opCode(add_pd,
				opCode(mul_pd, opCode(sub_pd, x2, thrVec), sx),
				opCode(mul_pd, opCode(mul_pd, thrVec, x), cx));

			// Phi'_k = 3 Phi_k(0) (k/sqrt3) * numGp / x^4
			_MData_ gp = opCode(mul_pd,
				opCode(mul_pd, phi03, kis3),
				opCode(div_pd, numGp, x4));

			// k*eta = sqrt(3)*x  (reuse x instead of recomputing kv*eta)
			_MData_ keta  = opCode(mul_pd, x, opCode(sqrt_pd, thrVec)); // or precompute sqrt3Vec outside loop
			_MData_ keta2 = opCode(mul_pd, keta, keta);

			// Reused pieces
			_MData_ twoPhik  = opCode(mul_pd, twoVec, phik);                  // 2 Phi_k
			_MData_ twoEtaGp = opCode(mul_pd, opCode(mul_pd, twoVec, etaVec), gp); // 2 eta Phi'_k

			// delta_rad = (2/3)(k eta)^2 Phi_k + 2 eta Phi'_k + 2 Phi_k
			_MData_ bracket = opCode(add_pd,
				opCode(mul_pd, t23Vec, opCode(mul_pd, keta2, phik)),
				opCode(add_pd, twoEtaGp, twoPhik));

			// src = -4 Phi'_k theta0'R + mR2s [ 2 Phi_k + (beta/4) * bracket ]
			_MData_ metricSrc = opCode(mul_pd, m4Vec, opCode(mul_pd, gp, thpRVec));
			_MData_ tempSrc   = opCode(add_pd, twoPhik, opCode(mul_pd, b4Vec, bracket));
			src = opCode(add_pd, metricSrc, opCode(mul_pd, mR2sVec, tempSrc));
		}

		// acc = [R''/R - (mA^2 R^2 cos(theta0) + k^2)] psi_k + src
		_MData_ coeff = opCode(sub_pd, RppVec, opCode(add_pd, mR2cVec, k2v));
		_MData_ acc   = opCode(add_pd, opCode(mul_pd, coeff, mk), src);

	#if defined(__AVX512F__) || defined(__FMA__)
		_MData_ vnew = opCode(fmadd_pd, acc, dzcVec, vk);
		_MData_ mnew = opCode(fmadd_pd, vnew, dzdVec, mk);
	#else
		_MData_ vnew = opCode(add_pd, vk, opCode(mul_pd, acc, dzcVec));
		_MData_ mnew = opCode(add_pd, mk, opCode(mul_pd, vnew, dzdVec));
	#endif

		opCode(store_pd, &v[i],  vnew);
		opCode(store_pd, &m2[i], mnew);
	}

	/*
	 * Evolve epsilon0 = pi-theta0 separately.  Centring the Hubble-friction
	 * term on the velocity kick gives
	 *
	 *   v+ = [(1-H h)/(1+H h)] v- + h a/(1+H h),
	 *
	 * with a = m_a^2 R^2 sin(epsilon0).  The caller synchronizes the legacy
	 * psi0 and psi0' slots after advancing the stage time and scale factor.
	 */
	if (useEpsilonZeroMode)
	{
		const double kick = dzc;
		const double denominator = 1.0 + Rp*kick;
		const double acceleration = mA2*R*R*std::sin(*epsilon0_);
		const double epsilonVelocity =
			((1.0 - Rp*kick)*(*epsilon0p_) + kick*acceleration)
			/ denominator;
		*epsilon0p_ = epsilonVelocity;
		*epsilon0_ += dzd*epsilonVelocity;
	}

// 	#pragma omp parallel for schedule(static)
// 	for (size_t i = 0; i < ppar.nmodes; i += step) {
//
// 		// Load a SIMD packet of modes starting at i
// 		// mk   = current mode amplitude psi_k
// 		// vk   = current mode velocity psi_k'
// 		// phi0 = primordial / initial gravitational potential Phi_k(0)
// 		// kv   = comoving wavenumber k
// 		// k2v  = k^2
// 		_MData_ mk   = opCode(load_pd, &m[i]);
// 		_MData_ vk   = opCode(load_pd, &v[i]);
// 		_MData_ kv   = opCode(load_pd, &kk[i]);
// 		_MData_ k2v  = opCode(load_pd, &k2[i]);
// 		_MData_ src  = opCode(set1_pd, 0.0);
//
// 		if (rhs)
// 		{
// 			_MData_ phi0 = opCode(load_pd, &g[i]);
//
// 			// x = k eta / sqrt(3)
// 			// This is the usual radiation-era sound-horizon variable
// 			// appearing in the analytic evolution of the metric potential.
// 			_MData_ x  = opCode(mul_pd, opCode(mul_pd, kv, etaVec), iS3Vec);
// 			_MData_ x2 = opCode(mul_pd, x, x);    // x^2
// 			_MData_ x3 = opCode(mul_pd, x2, x);   // x^3
// 			_MData_ x4 = opCode(mul_pd, x2, x2);  // x^4
//
// 			// sin(x), cos(x): needed for the analytic radiation-era Phi_k solution
// 			_MData_ sx = opCode(sin_pd, x);
// 			_MData_ cx = opCode(cos_pd, x);
//
// 			// Gravitational potential at conformal time eta:
// 			//
// 			// Phi_k(eta) = 3 Phi_k(0) [sin(x) - x cos(x)] / x^3
// 			//
// 			// This is the standard radiation-dominated transfer function for Phi_k.
// 			_MData_ phik = opCode(mul_pd, thrVec,
// 				opCode(mul_pd, phi0,
// 					opCode(div_pd,
// 						opCode(sub_pd, sx, opCode(mul_pd, x, cx)),
// 						x3)));
//
// 		// Conformal-time derivative of the gravitational potential:
// 		//
// 		// Phi'_k(eta) = Phi_k(0) (k/sqrt(3)) * 3 * [ (x^2 - 3) sin(x) + 3 x cos(x) ] / x^4
// 		//
// 		// This enters the source term that drives the mode equation.
// 			_MData_ gp = opCode(mul_pd, phi0,
// 				opCode(mul_pd, opCode(mul_pd, kv, iS3Vec),
// 					opCode(mul_pd, thrVec,
// 						opCode(div_pd,
// 							opCode(add_pd,
// 								opCode(mul_pd, opCode(sub_pd, x2, thrVec), sx),
// 								opCode(mul_pd, opCode(mul_pd, thrVec, x), cx)),
// 						x4))));
//
// 			// k*eta and (k*eta)^2
// 			// These combinations appear repeatedly in the forcing/source structure.
// 			_MData_ keta  = opCode(mul_pd, kv, etaVec);
// 			_MData_ keta2 = opCode(mul_pd, keta, keta);
//
// 			// Auxiliary combination entering the external source:
// 			//
// 			// bracket = (2/3) (k eta)^2 Phi_k + 2 eta Phi'_k + Phi_k
// 			//
// 			// This looks like the metric combination induced by the scalar perturbation
// 			// that couples into the axion/fluctuation mode equation.
// 			_MData_ bracket = opCode(add_pd,
// 				opCode(mul_pd, t23Vec, opCode(mul_pd, keta2, phik)),
// 				opCode(add_pd,
// 					opCode(mul_pd, opCode(mul_pd, twoVec, etaVec), gp),
// 						opCode(mul_pd, twoVec, phik)));
//
// 			// Source term driving psi_k:
// 			//
// 			// src = -4 Phi'_k theta0' + (-R^2 sin(theta0)) [ 2 Phi_k + b4 * bracket ]
// 			//
// 			// Interpreting your variable names:
// 			// - thpRVec probably stores theta0' (background field derivative)
// 			// - m4Vec  likely corresponds to -4
// 			// - mR2sVec likely corresponds to -R^2 sin(theta0)
// 			// - b4Vec is some model-dependent coefficient multiplying 'bracket'
// 			src = opCode(add_pd,
// 				opCode(mul_pd, m4Vec, opCode(mul_pd, gp, thpRVec)),
// 				opCode(mul_pd, mR2sVec,
// 					opCode(add_pd,
// 						opCode(mul_pd, twoVec, phik),
// 						opCode(mul_pd, b4Vec, bracket))));
// 		} // end calculating source
//
// 		// Mode acceleration psi_k'' from the linearized equation of motion:
// 		//
// 		// psi_k'' = [ -mA^2 R^2 cos(theta0) - k^2 + R''/R ] psi_k + src
// 		//
// 		// where:
// 		// - mR2cVec = mA^2 R^2 cos(theta0)
// 		// - k2v     = k^2
// 		// - RppVec  = R''/R
// 		//
// 		// So the first term is the homogeneous evolution operator,
// 		// and src is the inhomogeneous forcing.
// 		_MData_ acc = opCode(add_pd,
// 			opCode(mul_pd,
// 				opCode(sub_pd, RppVec, opCode(add_pd, mR2cVec, k2v)),
// 				mk),
// 			src);
//
// #if defined(__AVX512F__) || defined(__FMA__)
// 		// Time update:
// 		// 1) update velocity/momentum: psi_k' <- psi_k' + psi_k'' * dzc
// 		// 2) update field amplitude:    psi_k  <- psi_k  + psi_k'  * dzd
// 		//
// 		// Using fused multiply-add when available for better performance/precision.
// 		_MData_ vnew = opCode(fmadd_pd, acc, dzcVec, vk);
// 		_MData_ mnew = opCode(fmadd_pd, vnew, dzdVec, mk);
// #else
// 		// Same update without FMA support.
// 		_MData_ vnew = opCode(add_pd, vk, opCode(mul_pd, acc, dzcVec));
// 		_MData_ mnew = opCode(add_pd, mk, opCode(mul_pd, vnew, dzdVec));
// #endif
//
// 		// Store updated psi_k' back into v
// 		// Store updated psi_k into m2
// 		// (m2 may be the output buffer for the new field state)
// 		opCode(store_pd, &v[i],  vnew);
// 		opCode(store_pd, &m2[i], mnew);
// 	}

	/* overwrite zero mode with correct scalar equation */

	const double acc0  = Rpp*ctheta0 - mR2s;
	const double v0new = ctheta0p + dzc*acc0;
	const double m0new = ctheta0  + dzd*v0new;

	v[0]  = v0new;
	m2[0] = m0new;

	LogMsg(VERB_PARANOID,
		"[LNK] zero mode th %.3e cthp %.3e acc %.3e -> cth %.3e cthp %.3e",
		theta0, ctheta0p, acc0, m0new, v0new);

#undef _MData_
#undef step

}

#undef	opCode
#undef	opCode_N
#undef	opCode_P
#undef	Align
#undef	_PREFIX_
