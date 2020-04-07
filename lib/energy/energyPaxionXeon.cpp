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

template<const bool map>
void	energyPaxionKernelXeon(const void * __restrict__ m_, const void * __restrict__ v_, void * __restrict__ m2_, const PropParms ppar, FieldPrecision precision, void * __restrict__ eRes_)
{

	const double R     = ppar.R;
	const double ood2a = ppar.ood2a/R/R;
	const double beta  = ppar.beta;
	const size_t Lx = ppar.Lx, Sf = Lx*Lx, Vo = ppar.Vo, Vf = ppar.Vf, Ng = ppar.Ng, Vh = Vf+Ng*Vo;

	double * __restrict__ eRes = (double * __restrict__) eRes_;
	double gxC = 0., gyC = 0., gzC = 0., ntC = 0., ptC = 0.;

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

		#pragma omp parallel default(shared) reduction(+:gxC,gyC,gzC,ntC,ptC)
		{
			_MData_ mel, vel, mMx, mMy, mMyp, mMz, mPx, mPy, mPyp, mPz, tmp, grd;

			double tmpGx[step] __attribute__((aligned(Align)));
			double tmpGy[step] __attribute__((aligned(Align)));
			double tmpGz[step] __attribute__((aligned(Align)));
			double tmpV [step] __attribute__((aligned(Align)));
			double tmpN [step] __attribute__((aligned(Align)));

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
					mPy  = opCode(load_pd, &m[idxPy]);
					mPyp = opCode(load_pd, &v[idxPy]);
#if	defined(__AVX512F__)
					mMy  = opCode(add_pd, opCode(permutexvar_pd, vShRg, opCode(load_pd, &m[idxMy])), mPy);
					mMyp = opCode(add_pd, opCode(permutexvar_pd, vShRg, opCode(load_pd, &v[idxMy])), mPy);
#elif	defined(__AVX2__)       //AVX2
					mMy  = opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_pd, &m[idxMy])), opCode(setr_epi32, 6,7,0,1,2,3,4,5)));
					mMyp = opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_pd, &v[idxMy])), opCode(setr_epi32, 6,7,0,1,2,3,4,5)));
#elif	defined(__AVX__)
					mel = opCode(permute_pd, opCode(load_pd, &m[idxMy]), 0b00000101);
					vel = opCode(permute2f128_pd, mel, mel, 0b00000001);
					mMy = opCode(blend_pd, mel, vel, 0b00000101);
					mel = opCode(permute_pd, opCode(load_pd, &v[idxMy]), 0b00000101);
					vel = opCode(permute2f128_pd, mel, mel, 0b00000001);
					mMyp = opCode(blend_pd, mel, vel, 0b00000101);
#else
					mel = opCode(load_pd, &m[idxMy]);
					mMy = opCode(shuffle_pd, mel, mel, 0x00000001);
					mel = opCode(load_pd, &v[idxMy]);
					mMyp = opCode(shuffle_pd, mel, mel, 0x00000001);
#endif
				}
				else
				{
					idxMy = idx - XC;
					mMy  = opCode(load_pd, &m[idxMy]);
					mMyp = opCode(load_pd, &v[idxMy]);

					if (X[1] == YC-1)
					{
						idxPy = idx - Sf + XC;
#if	defined(__AVX512F__)
						mPy  = opCode(add_pd, opCode(permutexvar_pd, vShLf, opCode(load_pd, &m[idxPy])), mMy);
						mPyp = opCode(add_pd, opCode(permutexvar_pd, vShLf, opCode(load_pd, &v[idxPy])), mMy);
#elif	defined(__AVX2__)       //AVX2
						mPy  = opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_pd, &m[idxPy])), opCode(setr_epi32, 2,3,4,5,6,7,0,1)));
						mPyp = opCode(castsi256_pd, opCode(permutevar8x32_epi32, opCode(castpd_si256, opCode(load_pd, &v[idxPy])), opCode(setr_epi32, 2,3,4,5,6,7,0,1)));
#elif	defined(__AVX__)
						mel = opCode(permute_pd, opCode(load_pd, &m[idxPy]), 0b00000101);
						vel = opCode(permute2f128_pd, mel, mel, 0b00000001);
						mPy = opCode(blend_pd, mel, vel, 0b00001010);
						mel = opCode(permute_pd, opCode(load_pd, &v[idxPy]), 0b00000101);
						vel = opCode(permute2f128_pd, mel, mel, 0b00000001);
						mPyp = opCode(blend_pd, mel, vel, 0b00001010);
#else
						vel = opCode(load_pd, &m[idxPy]);
						mPy = opCode(shuffle_pd, vel, vel, 0b00000001);
						vel = opCode(load_pd, &v[idxPy]);
						mPyp = opCode(shuffle_pd, vel, vel, 0b00000001);
#endif
					}
					else
					{
						idxPy = idx + XC;
						mPy  = opCode(load_pd, &m[idxPy]);
						mPyp = opCode(load_pd, &v[idxPy]);
					}
				}

				// Tienes mMy y los puntos para mMx y mMz. Calcula todo ya!!!

				idxPz = idx+Sf;
				idxMz = idx-Sf;
				idxP0 = idx;

				mel = opCode(load_pd, &m[idxP0]);    // Carga m
				vel = opCode(load_pd, &v[idxP0]); // Carga v

				/* Gradients (without mod)
				   both for q and p */

				grd = opCode(sub_pd, opCode(load_pd, &m[idxPx]), mel);
					mPx = opCode(mul_pd, grd, grd);
				grd = opCode(sub_pd, opCode(load_pd, &v[idxPx]), vel);
					mPx = opCode(add_pd, mPx, opCode(mul_pd, grd, grd));

				grd = opCode(sub_pd, opCode(load_pd, &m[idxMx]), mel);
					mMx = opCode(mul_pd, grd, grd);
				grd = opCode(sub_pd, opCode(load_pd, &v[idxMx]), vel);
					mMx = opCode(add_pd, mMx, opCode(mul_pd, grd, grd));

				grd = opCode(sub_pd, mPy, mel);
					mPy = opCode(mul_pd, grd, grd);
				grd = opCode(sub_pd, mPyp, vel); // check!
					mPy = opCode(add_pd, mPy, opCode(mul_pd, grd, grd));

				grd = opCode(sub_pd, mMy, mel);
					mMy = opCode(mul_pd, grd, grd);
				grd = opCode(sub_pd, mMyp, vel); // check!
					mMy = opCode(add_pd, mMy, opCode(mul_pd, grd, grd));

				grd = opCode(sub_pd, opCode(load_pd, &m[idxPz]), mel);
					mPz = opCode(mul_pd, grd, grd);
				grd = opCode(sub_pd, opCode(load_pd, &v[idxPz]), vel); // check!
					mPz = opCode(add_pd, mPz, opCode(mul_pd, grd, grd));

				grd = opCode(sub_pd, opCode(load_pd, &m[idxMz]), mel);
					mMz = opCode(mul_pd, grd, grd);
				grd = opCode(sub_pd, opCode(load_pd, &v[idxMz]), vel); // check!
					mMz = opCode(add_pd, mMz, opCode(mul_pd, grd, grd));

				grd = opCode(add_pd, mPx, mMx);
				mMx = opCode(add_pd, mPy, mMy);
				mMy = opCode(add_pd, mPz, mMz);

				/* Kinetic energy is zero by definition */
				/* Number density */

				mPx =  opCode(add_pd, opCode(mul_pd, mel, mel),opCode(mul_pd, vel, vel));

				/* Potential energy is beta (p^2+q^2)
				 		For axion beta = - 1/8R^2 */

				mPy = opCode(mul_pd, mPx, mPx);

				opCode(store_pd, tmpGx, grd);
				opCode(store_pd, tmpGy, mMx);
				opCode(store_pd, tmpGz, mMy);
				opCode(store_pd, tmpN,  mPx);
				opCode(store_pd, tmpV,  mPy);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					ptC += tmpV[ih];
					ntC += tmpN[ih];
					gxC += tmpGx[ih];
					gyC += tmpGy[ih];
					gzC += tmpGz[ih];

					if	(map == true) {
						unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
						m2[iNx]    =  tmpN[ih];
						m2[iNx+Vh] = (tmpGx[ih] + tmpGy[ih] + tmpGz[ih])*ood2a + tmpV[ih]*beta;
					}
				}
			}
		}

		gxC *= ood2a; gyC *= ood2a; gzC *= ood2a; ptC *= beta;

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

		/* compute, reduce over energies and create map in m2 */
		#pragma omp parallel default(shared) reduction(+:gxC,gyC,gzC,ntC,ptC)
		{
			_MData_ mel, vel, mMx, mMy, mMyp, mMz, mPx, mPy, mPyp, mPz, tmp, grd;

			float tmpGx[step] __attribute__((aligned(Align)));
			float tmpGy[step] __attribute__((aligned(Align)));
			float tmpGz[step] __attribute__((aligned(Align)));
			float tmpN [step] __attribute__((aligned(Align)));
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
					X[2] -= Ng;	// Removes ghosts
				}

				/* Gradient energies */

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
					mPyp = opCode(load_ps, &v[idxPy]);
#if	defined(__AVX512F__)
					mMy = opCode(permutexvar_ps, vShRg, opCode(load_ps, &m[idxMy]));
					mPyp = opCode(permutexvar_ps, vShRg, opCode(load_ps, &v[idxMy]));;
#elif	defined(__AVX2__)
					mMy = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 7,0,1,2,3,4,5,6));
					mMyp = opCode(permutevar8x32_ps, opCode(load_ps, &v[idxMy]), opCode(setr_epi32, 7,0,1,2,3,4,5,6));
#elif	defined(__AVX__)
					mel  = opCode(permute_ps, opCode(load_ps, &m[idxMy]), 0b10010011);
					vel  = opCode(permute2f128_ps, mel, mel, 0b00000001);
					mMy  = opCode(blend_ps, mel, vel, 0b00010001);
					mel  = opCode(permute_ps, opCode(load_ps, &v[idxMy]), 0b00000101);
					vel  = opCode(permute2f128_ps, mel, mel, 0b00000001);
					mMyp = opCode(blend_ps, mel, vel, 0b00010001);
#else
					mel  = opCode(load_ps, &m[idxMy]);
					mMy  = opCode(shuffle_ps, mel, mel, 0b10010011);
					mel  = opCode(load_ps, &v[idxMy]);
					mMyp = opCode(shuffle_ps, mel, mel, 0b10010011);
#endif
				}
				else
				{
					idxMy = idx - XC;
					mMy  = opCode(load_ps, &m[idxMy]);
					mMyp = opCode(load_ps, &v[idxMy]);

					if (X[1] == YC-1)
					{
						idxPy = idx - Sf + XC;
#if	defined(__AVX512F__)
						mPy  = opCode(permutexvar_ps, vShLf, opCode(load_ps, &m[idxPy]));
						mPyp = opCode(permutexvar_ps, vShLf, opCode(load_ps, &v[idxPy]));
#elif	defined(__AVX2__)
						mPy  = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 1,2,3,4,5,6,7,0));
						mPyp = opCode(permutevar8x32_ps, opCode(load_ps, &v[idxPy]), opCode(setr_epi32, 1,2,3,4,5,6,7,0));
#elif	defined(__AVX__)
						mel  = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b00111001);
						vel  = opCode(permute2f128_ps, mel, mel, 0b00000001);
						mPy  = opCode(blend_ps, mel, vel, 0b10001000);
						mel  = opCode(permute_ps, opCode(load_ps, &v[idxPy]), 0b00111001);
						vel  = opCode(permute2f128_ps, mel, mel, 0b00000001);
						mPyp = opCode(blend_ps, mel, vel, 0b10001000);
#else
						vel = opCode(load_ps, &m[idxPy]);
						mPy = opCode(shuffle_ps, vel, vel, 0b00111001);
						vel = opCode(load_ps, &v[idxPy]);
						mPyp = opCode(shuffle_ps, vel, vel, 0b00111001);
#endif
					}
					else
					{
						idxPy = idx + XC;
						mPy  = opCode(load_ps, &m[idxPy]);
						mPyp = opCode(load_ps, &v[idxPy]);
					}
				}

				idxPz = idx+Sf;
				idxMz = idx-Sf;
				idxP0 = idx;

				mel = opCode(load_ps, &m[idxP0]);    // Carga m
				vel = opCode(load_ps, &v[idxP0]); // Carga v

				/* Gradients (without mod) */
				grd = opCode(sub_ps, opCode(load_ps, &m[idxPx]), mel);
					mPx = opCode(mul_ps, grd, grd);
				grd = opCode(sub_ps, opCode(load_ps, &v[idxPx]), vel);
					mPx = opCode(add_ps, mPx, opCode(mul_ps, grd, grd));

				grd = opCode(sub_ps, opCode(load_ps, &m[idxMx]), mel);
					mMx = opCode(mul_ps, grd, grd);
				grd = opCode(sub_ps, opCode(load_ps, &v[idxMx]), vel);
					mMx = opCode(add_ps, mMx, opCode(mul_ps, grd, grd));

				grd = opCode(sub_ps, mPy, mel);
					mPy = opCode(mul_ps, grd, grd);
				grd = opCode(sub_ps, mPyp, vel);
					mPy = opCode(add_ps, mPy, opCode(mul_ps, grd, grd));

				grd = opCode(sub_ps, mMy, mel);
					mMy = opCode(mul_ps, grd, grd);
				grd = opCode(sub_ps, mMyp, vel);
					mMy = opCode(add_ps, mMy, opCode(mul_ps, grd, grd));

				grd = opCode(sub_ps, opCode(load_ps, &m[idxPz]), mel);
					mPz = opCode(mul_ps, grd, grd);
				grd = opCode(sub_ps, opCode(load_ps, &v[idxPz]), vel);
					mPz = opCode(add_ps, mPz, opCode(mul_ps, grd, grd));

				grd = opCode(sub_ps, opCode(load_ps, &m[idxMz]), mel);
					mMz = opCode(mul_ps, grd, grd);
				grd = opCode(sub_ps, opCode(load_ps, &v[idxMz]), vel);
					mMz = opCode(add_ps, mMz, opCode(mul_ps, grd, grd));

					// if (idx==Vf-step || idx==Vo)
					// {
					// 	printf("%zu\n",idx);
					// 	printsVar(mMz, "Mz");
					// 	printsVar(mPz, "Pz");
					// }


				grd = opCode(add_ps, mPx, mMx);
				mMx = opCode(add_ps, mPy, mMy);
				mMy = opCode(add_ps, mPz, mMz);

				/* Kinetic energy is zero by definition */
				/* Number density */

				mPx =  opCode(add_ps, opCode(mul_ps, mel, mel),opCode(mul_ps, vel, vel));

				/* Potential energy is beta (p^2+q^2)
				 		For axion beta = - 1/8R^2 */

				mPy = opCode(mul_ps, mPx, mPx);

				opCode(store_ps, tmpGx, grd);
				opCode(store_ps, tmpGy, mMx);
				opCode(store_ps, tmpGz, mMy);
				opCode(store_ps, tmpN,  mPx);
				opCode(store_ps, tmpV,  mPy);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					ptC += (double) (tmpV[ih]);
					ntC += (double) (tmpN[ih]);
					gxC += (double) (tmpGx[ih]);
					gyC += (double) (tmpGy[ih]);
					gzC += (double) (tmpGz[ih]);

					// Saves map
					if	(map == true) {
						unsigned long long iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + X[2]*Sf);
						m2[iNx]    =  tmpN[ih];
						m2[iNx+Vh] = (tmpGx[ih] + tmpGy[ih] + tmpGz[ih])*ood2a + tmpV[ih]*beta;
					}
				}
			}
		}

		gxC *= ood2a; gyC *= ood2a; gzC *= ood2a; ptC *= beta;
#undef	_MData_
#undef	step
	}

	eRes[TH_GRX] = gxC;
	eRes[TH_GRY] = gyC;
	eRes[TH_GRZ] = gzC;
	eRes[TH_KIN] = ntC;
	eRes[TH_POT] = ptC;
}

void	energyPaxionCpu	(Scalar *axionField, void *eRes, const bool map)
{
	PropParms ppar ;
	/* Energy computed with 1 neighbours even if Ng propagation. Some non-conservation expected! */
	/* Energy densities in ADM units */
	ppar.Ng     = axionField->getNg();
	ppar.Lx     = axionField->Length();
	ppar.Vo     = ppar.Ng*axionField->Surf();
	ppar.Vf     = ppar.Vo + axionField->Size();
	ppar.ct     = *axionField->zV();
	ppar.R      = *axionField->RV();
	ppar.massA  = axionField->AxionMass();
	ppar.frw    = axionField->BckGnd()->Frw();

	/*energy density is computed in physical coordinates, not comoving
	  rho_Grad = 1/2m_A |grad cpax|^2 /R^5   units: [H1fA]^2
		rho_SI   = -1/16  |cpax|^4 / R^6       units: [H1fA]^2
		the number density is computed in comoving coordinates for plotting purposes
		this is stored in eRes[TH_KIN] and corresponds to
		n        = |cpax|^2                    units: [H1fA^2*(R/R1)^3]
		energy density is just
		rho      = |cpax|^2 x mA/(R1/R^3) */

	ppar.ood2a  = 0.25/ppar.massA/pow(axionField->BckGnd()->PhysSize()/axionField->Length(),2.)/pow(ppar.R,5);
	ppar.beta   = -1.0/(16.0)/pow(ppar.R,6);
	const FieldPrecision precision = axionField->Precision();

	axionField->exchangeGhosts(FIELD_M);
	axionField->exchangeGhosts(FIELD_V);

	switch	(map) {
		case	true:
			energyPaxionKernelXeon<true>(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), ppar, precision, eRes);
			break;

		case	false:
			energyPaxionKernelXeon<false>(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), ppar, precision, eRes);
			break;
	}
}
