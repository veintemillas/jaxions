#include<cstdio>
#include<cmath>
#include "scalar/scalarField.h"
#include "enum-field.h"

#ifdef USE_XEON
	#include "comms/comms.h"
	#include "utils/xeonDefs.h"
#endif


#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

#include <immintrin.h>


#ifdef	__MIC__
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

#ifdef	__MIC__
	#define	_MData_ __m512d
#elif	defined(__AVX__)
	#define	_MData_ __m256d
#else
	#define	_MData_ __m128d
#endif 
inline _MData_	opCode(cos_pd, _MData_ x)
{
	_MData_ tmp2, tmp4, tmp6, a, b, c;
	static const double a_s = -0.0415758., b_s = 0.00134813., c_s = -(1+4.*M_PI2*a_s+6.*M_PI4*b_s)/(8.*M_PI6);

	a = opCode(set1_pd, a_s);
	b = opCode(set1_pd, b_s);
	c = opCode(set1_pd, c_s);

	tmp2 = opCode(mul_pd, x, x);
	tmp4 = opCode(mul_pd, tmp2, tmp2);
	tmp6 = opCode(mul_pd, tmp2, tmp4);
	return opCode(sub_pd, opCode(set1_pd, 1.),
		opCode(add_pd, opCode(mul_pd, opCode(set1_pd, 0.5), tmp2),
			opCode(add_pd,
				opCode(add_pd,
					opCode(mul_pd, tmp4, a),
					opCode(mul_pd, tmp6, b)),
				opCode(mul_pd, c, opCode(mul_pd, tmp4, tmp4))));
}

#undef	_MData_

#ifdef	__MIC__
	#define	_MData_ __m512
#elif	defined(__AVX__)
	#define	_MData_ __m256
#else
	#define	_MData_ __m128
#endif 
inline _MData_	opCode(cos_ps, _MData_ x)
{
	_MData_ tmp2, tmp4, tmp6, a, b, c;
	static const double a_s = -0.0415758., b_s = 0.00134813., c_s = -(1+4.*M_PI2*a_s+6.*M_PI4*b_s)/(8.*M_PI6);

	a = opCode(set1_ps, a_s);
	b = opCode(set1_ps, b_s);
	c = opCode(set1_ps, c_s);

	tmp2 = opCode(mul_ps, x, x);
	tmp4 = opCode(mul_ps, tmp2, tmp2);
	tmp6 = opCode(mul_ps, tmp2, tmp4);
	return opCode(sub_ps, opCode(set1_ps, 1.),
		opCode(add_ps, opCode(mul_ps, opCode(set1_ps, 0.5), tmp2),
			opCode(add_ps,
				opCode(add_ps,
					opCode(mul_ps, tmp4, a),
					opCode(mul_ps, tmp6, b)),
				opCode(mul_ps, c, opCode(mul_ps, tmp4, tmp4))));
}

#undef	_MData_


#ifdef USE_XEON
__attribute__((target(mic)))
#endif
void	energyMapThetaKernelXeon(const void * __restrict__ m_, const void * __restrict__ v_, void * __restrict__ m2_, double *z, const double ood2, const double nQcd,
			 const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision)
{
	const size_t Sf = Lx*Lx;

	if (precision == FIELD_DOUBLE)
	{
#ifdef	__MIC__
	#define	_MData_ __m512d
	#define	step 8
#elif	defined(__AVX__)
	#define	_MData_ __m256d
	#define	step 4
#else
	#define	_MData_ __m128d
	#define	step 2
#endif

#ifdef	USE_XEON
		const double * __restrict__ m	= (const double * __restrict__) m_;
		const double * __restrict__ v	= (const double * __restrict__) v_;
		double * __restrict__ m2	= (double * __restrict__) m2_;

		__assume_aligned(m, Align);
		__assume_aligned(v, Align);
		__assume_aligned(m2,Align);
#else
		const double * __restrict__ m	= (const double * __restrict__) __builtin_assume_aligned (m_, Align);
		const double * __restrict__ v	= (const double * __restrict__) __builtin_assume_aligned (v_, Align);
		double * __restrict__ m2	= (double * __restrict__) __builtin_assume_aligned (m2_,Align);
#endif
		const double zR  = *z;
		const double iz  = 1./zR;
		const double zQ = 9.*pow(zR, nQcd+2.);
		const double o2 = ood2*0.375;
#ifdef	__MIC__
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);
#endif

		#pragma omp parallel default(shared) 
		{
			_MData_ mel, vel, mMx, mMy, mMz, mdv, mod, mTp;
			_MData_ Grx,  Gry,  Grz, tGx, tGy, tGz, tVp, tKp, mCg, mSg;

			double tmpS[step] __attribute__((aligned(Align)));
			double tmpV[step] __attribute__((aligned(Align)));

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[3], idxPx, idxMx, idxPy, idxMy, idxPz, idxMz, idxP0;

				{
					size_t tmi = idx/XC;

					X[2] = tmi/YC;
					X[1] = tmi - X[2]*YC;
					X[0] = idx - tmi*XC;
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
#ifdef	__MIC__
					mMy = opCode(sub_pd, opCode(load_pd, &m[idxPy]), opCode(castps_pd, opCode(permute4f128_ps, opCode(castpd_ps, opCode(load_pd, &m[idxMy])), _MM_PERM_CBAD)));
#elif	defined(__AVX__)
					mMx = opCode(load_pd, &m[idxMy]);
					mMy = opCode(sub_pd, opCode(load_pd, &m[idxPy]), opCode(permute2f128_pd, mMx, mMx, 0b00000001));
#else
					mMy = opCode(sub_pd, opCode(load_pd, &m[idxPy]), opCode(load_pd, &m[idxMy]));
#endif
				}
				else
				{
					idxMy = idx - XC;

					if (X[1] == YC-1)
					{
						idxPy = idx - Sf + XC;
#ifdef	__MIC__
						mMy = opCode(sub_pd, opCode(castps_pd, opCode(permute4f128_ps, opCode(castpd_ps, opCode(load_pd, &m[idxPy])), _MM_PERM_ADCB)), opCode(load_pd, &m[idxMy]));
#elif	defined(__AVX__)
						mMx = opCode(load_pd, &m[idxPy]);
						mMy = opCode(sub_pd, opCode(permute2f128_pd, mMx, mMx, 0b00000001), opCode(load_pd, &m[idxMy]));
#else
						mMy = opCode(sub_pd, opCode(load_pd, &m[idxPy]), opCode(load_pd, &m[idxMy]));
#endif
					}
					else
					{
						idxPy = ((idx + XC) << 1);
						mMy = opCode(sub_pd, opCode(load_pd, &m[idxPy]), opCode(load_pd, &m[idxMy]));
					}
				}

				// Tienes mMy y los puntos para mMx y mMz. Calcula todo ya!!!

				idxPz = idx+Sf;
				idxMz = idx-Sf;
				idxP0 = idx;

				mel = opCode(load_pd, &m[idxP0]);//Carga m
				vel = opCode(load_pd, &v[idxMz]);//Carga v

				// Calculo los gradientes
				mPx = opCode(sub_pd, opCode(load_pd, &m[idxPx]), mel);
				mMx = opCode(sub_pd, opCode(load_pd, &m[idxMx]), mel);
				mPz = opCode(sub_pd, opCode(load_pd, &m[idxPz]), mel);
				mMz = opCode(sub_pd, opCode(load_pd, &m[idxMz]), mel);
				mPy = opCode(sub_pd, mPy, mel);
				mMy = opCode(sub_pd, mMy, mel);

				grd = opCode(mul_pd,
					opCode(add_pd,
						opCode(add_pd,
							opCode(add_pd,
								opCode(mul_pd, mPx, mPx),
								opCode(mul_pd, mMx, mMx)),
							opCode(add_pd,
								opCode(mul_pd, mPy, mPy),
								opCode(mul_pd, mMy, mMy))),
						opCode(mul_pd, mPz, mPz),
						opCode(mul_pd, mMz, mMz)),
					opCode(set1_pd, ood2));

				mPx = opCode(add_pd,
					opCode(mul_pd,
						opCode(set1_pd, 0.5),
						opCode(mul_pd, vel, vel)),
					opCode(mul_pd,
						opCode(set1_pd, zQ),
						opCode(sub_pd,
							opCode(set1_pd, 1.),
							opCode(mul_pd,
								opCode(set1_pd, iZ),
								opCode(cos_pd, mel)))));

				// STORAGE, CHECK BOUNDARIES (Y DIRECTION MOSTLY)

				opCode(store_pd, tmp, opCode(add_pd, grd, mPx);
				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					int iNx   = (X[0]/step + (X[1]+ih*YC)*Lx + (X[2]-1)*Sf);
					m2[iNx]   = tmp[ih];
				}
			}
		}

#undef	_MData_
#undef	step
	}
	else if (precision == FIELD_SINGLE)
	{
#ifdef	__MIC__
	#define	_MData_ __m512
	#define	step 16
#elif	defined(__AVX__)
	#define	_MData_ __m256
	#define	step 8
#else
	#define	_MData_ __m128
	#define	step 4
#endif

#ifdef	USE_XEON
		const float * __restrict__ m	= (const float * __restrict__) m_;
		const float * __restrict__ v	= (const float * __restrict__) v_;
		float * __restrict__ m2		= (float * __restrict__) m2_;

		__assume_aligned(m, Align);
		__assume_aligned(v, Align);
		__assume_aligned(m2,Align);
#else
		const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_, Align);
		const float * __restrict__ v	= (const float * __restrict__) __builtin_assume_aligned (v_, Align);
		float * __restrict__ m2		= (float * __restrict__) __builtin_assume_aligned (m2_,Align);
#endif
		const float zR  = *z;
		const float iz  = 1./zR;
		const float zQ = 9.f*powf(zR, nQcd+2.);
		const float o2 = ood2*0.375;
#ifdef	__MIC__
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);
#endif

		#pragma omp parallel default(shared) 
		{
			_MData_ mel, vel, mMx, mMy, mMz, mdv, mod, mTp;
			_MData_ Grx,  Gry,  Grz, tGx, tGy, tGz, tVp, tKp, mCg, mSg;

			float tmpS[2*step] __attribute__((aligned(Align)));
			float tmpV[2*step] __attribute__((aligned(Align)));

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[3], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0;

				{
					size_t tmi = idx/XC;

					X[2] = tmi/YC;
					X[1] = tmi - X[2]*YC;
					X[0] = idx - tmi*XC;
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
#ifdef  __MIC__
					tpM = opCode(swizzle_ps, opCode(load_ps, &m[idxMy]), _MM_SWIZ_REG_CBAD);
					tpP = opCode(permute4f128_ps, tpM, _MM_PERM_CBAD);
					mMy = opCode(mask_blend_ps, opCode(int2mask, 0b0001000100010001), tpM, tpP);
#elif	defined(__AVX2__)	//AVX2
					mMy = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 7,0,1,2,3,4,5,6));
#elif	defined(__AVX__)	//AVX
					tpM = opCode(permute_ps, opCode(load_ps, &m[idxMy]), 0b10010011);
					tpP = opCode(permute2f128_ps, tpM, tpM, 0b00000001);
					mMy = opCode(blend_ps, tpM, tpP, 0b00010001);
#else
					tpM = opCode(load_ps, &m[idxMy]);
					mMy = opCode(shuffle_ps, tpM, tpM, 0b10010011);
#endif
				}
				else
				{
					idxMy = idx - XC;
					mMy = opCode(load_ps, &m[idxMy]);

					if (X[1] == YC-1)
					{
						idxPy = idx - Sf + XC;
#ifdef  __MIC__
						tpM = opCode(swizzle_ps, opCode(load_ps, &m[idxPy]), _MM_SWIZ_REG_ADCB);
						tpP = opCode(permute4f128_ps, tpM, _MM_PERM_ADCB);
						mPy = opCode(mask_blend_ps, opCode(int2mask, 0b1110111011101110), tpM, tpP);
#elif	defined(__AVX2__)	//AVX2
						mPy = opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 1,2,3,4,5,6,7,0));
#elif	defined(__AVX__)	//AVX
						tpM = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b00111001);
						tpP = opCode(permute2f128_ps, tpM, tpM, 0b00000001);
						mPy = opCode(blend_ps, tpM, tpP, 0b10001000);
#else
						tpP = opCode(load_ps, &m[idxPy]);
						mPy = opCode(shuffle_ps, tpP, tpP, 0b00111001);
#endif
					}
					else
					{
						idxPy = idx + XC;
						mPy = opCode(load_ps, &m[idxPy]);
					}
				}

				// Tienes mMy y los puntos para mMx y mMz. Calcula todo ya!!!

				idxPz = idx+Sf;
				idxMz = idx-Sf;
				idxP0 = idx;

				mel = opCode(load_ps, &m[idxP0]);//Carga m
				vel = opCode(load_ps, &v[idxMz]);//Carga v

				// Calculo los gradientes
				mPx = opCode(sub_ps, opCode(load_ps, &m[idxPx]), mel);
				mMx = opCode(sub_ps, opCode(load_ps, &m[idxMx]), mel);
				mPz = opCode(sub_ps, opCode(load_ps, &m[idxPz]), mel);
				mMz = opCode(sub_ps, opCode(load_ps, &m[idxMz]), mel);
				mPy = opCode(sub_ps, mPy, mel);
				mMy = opCode(sub_ps, mMy, mel);

				grd = opCode(mul_ps,
					opCode(add_ps,
						opCode(add_ps,
							opCode(add_ps,
								opCode(mul_ps, mPx, mPx),
								opCode(mul_ps, mMx, mMx)),
							opCode(add_ps,
								opCode(mul_ps, mPy, mPy),
								opCode(mul_ps, mMy, mMy))),
						opCode(mul_ps, mPz, mPz),
						opCode(mul_ps, mMz, mMz)),
					opCode(set1_ps, ood2));

				mPx = opCode(add_ps,
					opCode(mul_ps,
						opCode(set1_ps, 0.5),
						opCode(mul_ps, vel, vel)),
					opCode(mul_ps,
						opCode(set1_ps, zQ),
						opCode(sub_ps,
							opCode(set1_ps, 1.),
							opCode(mul_ps,
								opCode(set1_ps, iZ),
								opCode(cos_ps, mel)))));

				// CHECK BOUNDARIES (Y DIRECTION MOSTLY)

				opCode(store_ps, tmp, opCode(add_ps, grd, mPx);

				#pragma unroll
				for (int ih=0; ih<step; ih++)
				{
					int iNx   = (X[0]/step + (X[1]+(ih>>1)*YC)*Lx + (X[2]-1)*Sf);
					m2[iNx]   = tmp[ih];
				}
			}
		}
#undef	_MData_
#undef	step
	}
}

void	energyMapThetaXeon	(Scalar *axionField, const double delta2, const double nQcd, const size_t Lx, const size_t V, const size_t S)
{
#ifdef USE_XEON
	const int  micIdx = commAcc(); 
	const double ood2 = 1./delta2;
	double *z  = axionField->zV();
	const FieldPrecision precision = axionField->Precision();

	axionField->exchangeGhosts(FIELD_M);
	#pragma offload target(mic:micIdx) in(z:length(8) UseX) nocopy(mX, vX, m2X : ReUseX)
	{
		energyMapThetaKernelXeon(mX, vX, m2X, z, ood2, nQcd, Lx, S, V+S,precision);
	}
#endif
}

void	energyMapThetaCpu	(Scalar *axionField, const double delta2, const double nQcd, const size_t Lx, const size_t V, const size_t S)
{
	const double ood2 = 1./delta2;
	double *z = axionField->zV();
	const FieldPrecision precision = axionField->Precision();

	axionField->exchangeGhosts(FIELD_M);
	energyMapThetaKernelXeon(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), z, ood2, nQcd, Lx, S, V+S, precision);
}
