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
				case VQCD_0:
					acu = opCode(sub_pd, lap,
									opCode(sub_pd, opCode(mul_pd, zQVec, opCode(sin_pd, opCode(mul_pd, mel, izVec))),
										opCode(mul_pd, opCode(set1_pd, Rpp), mel)));
				break;
				case VQCD_QUAD:
					acu = opCode(sub_pd, lap,
									opCode(sub_pd, opCode(mul_pd, zQVec, opCode(mul_pd, mel, izVec)),
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

			acu = opCode(sub_ps, lap,
							opCode(sub_ps, opCode(mul_ps, zQVec, opCode(sin_ps, opCode(mul_ps, mel, izVec))),
								opCode(mul_ps, opCode(set1_ps, Rpp), mel)));

			switch(VQcd){
				case VQCD_0:
				acu = opCode(sub_ps, lap,
								opCode(sub_ps, opCode(mul_ps, zQVec, opCode(sin_ps, opCode(mul_ps, mel, izVec))),
									opCode(mul_ps, opCode(set1_ps, Rpp), mel)));
				break;
				case VQCD_QUAD:
				acu = opCode(sub_ps, lap,
								opCode(sub_ps, opCode(mul_ps, zQVec, opCode(mul_ps, mel, izVec)),
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
			register _MData_ mIn, vIn, tmp;
			register size_t idxV0;

			#pragma omp for collapse(3) schedule(static)
			for (uint zz = 0; zz < bSizeZ; zz++) {
		 	  for (uint yy = 0; yy < bSizeY; yy++) {
			    for (uint xC = 0; xC < XC; xC += step) {
			      uint zC = zz + bSizeZ*zT + z0;
			      uint yC = yy + bSizeY*yT;

			      auto idx = zC*(YC*XC) + yC*XC + xC;

			      if (idx >= Vf)
				continue;

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
			register _MData_ mIn, vIn, tmp;
			register size_t idxV0;

			#pragma omp for collapse(3) schedule(static)
			for (uint zz = 0; zz < bSizeZ; zz++) {
		 	  for (uint yy = 0; yy < bSizeY; yy++) {
			    for (uint xC = 0; xC < XC; xC += step) {
			      uint zC = zz + bSizeZ*zT + z0;
			      uint yC = yy + bSizeY*yT;

			      auto idx = zC*(YC*XC) + yC*XC + xC;

			      if (idx >= Vf)
				continue;

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

#undef	opCode
#undef	opCode_N
#undef	opCode_P
#undef	Align
#undef	_PREFIX_


inline	void	propThetaKernelXeon(const void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_, const PropParms ppar, const double dz, const double c, const double d,
				    const size_t Vo, const size_t Vf, FieldPrecision precision, const unsigned int bSizeX, const unsigned int bSizeY, const unsigned int bSizeZ, const bool wMod, const VqcdType VQcd)
{
	switch (VQcd)
	{
		case VQCD_0:
		case VQCD_1:
		case VQCD_2:
		{
				switch (wMod) {
					case	true:
						propThetaKernelXeon<true,VQCD_0> (m_, v_, m2_, ppar, dz, c, d, Vo, Vf, precision, bSizeX, bSizeY, bSizeZ);
						break;

					case	false:
						propThetaKernelXeon<false,VQCD_0>(m_, v_, m2_, ppar, dz, c, d, Vo, Vf, precision, bSizeX, bSizeY, bSizeZ);
						break;
				}
		} return;
		case VQCD_QUAD:
		{
				switch (wMod) {
					case	true:
						propThetaKernelXeon<true,VQCD_QUAD> (m_, v_, m2_, ppar, dz, c, d, Vo, Vf, precision, bSizeX, bSizeY, bSizeZ);
						break;

					case	false:
						propThetaKernelXeon<false,VQCD_QUAD>(m_, v_, m2_, ppar, dz, c, d, Vo, Vf, precision, bSizeX, bSizeY, bSizeZ);
						break;
				}
		} return;
	}
}
