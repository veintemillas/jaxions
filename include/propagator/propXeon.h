#include<cstdio>
#include<cmath>
#include"scalar/scalarField.h"
#include"enum-field.h"
#include"scalar/varNQCD.h"
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

template<const VqcdType VQcd>
inline	void	propagateKernelXeon(const void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_, double *z, const double dz, const double c, const double d,
				    const double ood2, const double LL, const double nQcd, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision)
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
		const double zR = *z;
		const double z2 = zR*zR;
		//const double zQ = 9.*pow(zR, nQcd+3.);
		const double zQ = axionmass2(zR, nQcd, zthres, zrestore)*zR*zR*zR;

#if	defined(__AVX512F__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const double    __attribute__((aligned(Align))) zQAux[8] = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };	// Only real part
		const double    __attribute__((aligned(Align))) zRAux[8] = { zR, 0., zR, 0., zR, 0., zR, 0. };	// Only real part
		const long long __attribute__((aligned(Align))) shfRg[8] = {6, 7, 0, 1, 2, 3, 4, 5 };
		const long long __attribute__((aligned(Align))) shfLf[8] = {2, 3, 4, 5, 6, 7, 0, 1 };

		const _MInt_  vShRg = opCode(load_si512, shfRg);
		const _MInt_  vShLf = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) zQAux[4] = { zQ, 0., zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zRAux[4] = { zR, 0., zR, 0. };	// Only real part
#else
		const size_t XC = Lx;
		const size_t YC = Lx;

		const double __attribute__((aligned(Align))) zQAux[2] = { zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zRAux[2] = { zR, 0. };	// Only real part
#endif
		const _MData_ zQVec  = opCode(load_pd, zQAux);
		const _MData_ zRVec  = opCode(load_pd, zRAux);

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, mPx, mPy, mMx;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxPx, idxMx, idxPy, idxMy, idxPz, idxMz, idxP0;

				{
					size_t tmi = idx/XC, tpi;

					tpi = tmi/YC;
					X[1] = tmi - tpi*YC;
					X[0] = idx - tmi*XC;
				}

				if (X[0] == XC-step)
					idxPx = ((idx - XC + step) << 1);
				else
					idxPx = ((idx + step) << 1);

				if (X[0] == 0)
					idxMx = ((idx + XC - step) << 1);
				else
					idxMx = ((idx - step) << 1);

				if (X[1] == 0)
				{
					idxMy = ((idx + Sf - XC) << 1);
					idxPy = ((idx + XC) << 1);
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
					idxMy = ((idx - XC) << 1);

					if (X[1] == YC-1)
					{
						idxPy = ((idx - Sf + XC) << 1);
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
						idxPy = ((idx + XC) << 1);
						tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
					}
				}

				idxPz = ((idx+Sf) << 1);
				idxMz = ((idx-Sf) << 1);
				idxP0 = (idx << 1);

				mel = opCode(load_pd, &m[idxP0]);
				mPy = opCode(mul_pd, mel, mel);

#if	defined(__AVX512F__)
				mPx = opCode(add_pd, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, mPy), _MM_PERM_BADC)), mPy);
#elif	defined(__AVX__)
				mPx = opCode(add_pd, opCode(permute_pd, mPy, 0b00000101), mPy);
#else
				mPx = opCode(add_pd, opCode(shuffle_pd, mPy, mPy, 0b00000001), mPy);
#endif
				switch	(VQcd) {
					case	VQCD_1:
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
								zQVec),
							opCode(mul_pd,
								opCode(mul_pd,
									opCode(sub_pd, mPx, opCode(set1_pd, z2)),
									opCode(set1_pd, LL)),
								mel));
						break;

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
				}

				mPy = opCode(load_pd, &v[idxMz]);

#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_pd, mMx, opCode(set1_pd, dzc), mPy);
				mPx = opCode(fmadd_pd, tmp, opCode(set1_pd, dzd), mel);
#else
				tmp = opCode(add_pd, mPy, opCode(mul_pd, mMx, opCode(set1_pd, dzc)));
				mPx = opCode(add_pd, mel, opCode(mul_pd, tmp, opCode(set1_pd, dzd)));
#endif
				opCode(store_pd,  &v[idxMz], tmp);
				opCode(store_pd, &m2[idxP0], mPx);
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
		const float zR = *z;
		const float z2 = zR*zR;
		//const float zQ = 9.*powf(zR, nQcd+3.);
		const float zQ = axionmass2((double) zR, nQcd, zthres, zrestore)*zR*zR*zR;

#if	defined(__AVX512F__)
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) zQAux[16] = { zQ, 0., zQ, 0., zQ, 0., zQ, 0., zQ, 0., zQ, 0., zQ, 0., zQ, 0.};
		const float __attribute__((aligned(Align))) zRAux[16] = { zR, 0., zR, 0., zR, 0., zR, 0., zR, 0., zR, 0., zR, 0., zR, 0.};
		const int   __attribute__((aligned(Align))) shfRg[16] = {14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
		const int   __attribute__((aligned(Align))) shfLf[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1};

		const _MInt_  vShRg  = opCode(load_si512, shfRg);
		const _MInt_  vShLf  = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) zQAux[8]  = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };
		const float __attribute__((aligned(Align))) zRAux[8]  = { zR, 0., zR, 0., zR, 0., zR, 0. };
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const float __attribute__((aligned(Align))) zQAux[4]  = { zQ, 0., zQ, 0. };
		const float __attribute__((aligned(Align))) zRAux[4]  = { zR, 0., zR, 0. };
#endif
		const _MData_ zQVec  = opCode(load_ps, zQAux);
		const _MData_ zRVec  = opCode(load_ps, zRAux);

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, mPx, mPy, mMx;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0;

				{
					size_t tmi = idx/XC, itp;

					itp = tmi/YC;
					X[1] = tmi - itp*YC;
					X[0] = idx - tmi*XC;
				}

				if (X[0] == XC-step)
					idxPx = ((idx - XC + step) << 1);
				else
					idxPx = ((idx + step) << 1);

				if (X[0] == 0)
					idxMx = ((idx + XC - step) << 1);
				else
					idxMx = ((idx - step) << 1);

				if (X[1] == 0)
				{
					idxMy = ((idx + Sf - XC) << 1);
					idxPy = ((idx + XC) << 1);

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
					idxMy = ((idx - XC) << 1);

					if (X[1] == YC-1)
					{
						idxPy = ((idx - Sf + XC) << 1);
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
						idxPy = ((idx + XC) << 1);
						tmp = opCode(add_ps, opCode(load_ps, &m[idxPy]), opCode(load_ps, &m[idxMy]));
					}
				}

				idxPz = ((idx+Sf) << 1);
				idxMz = ((idx-Sf) << 1);
				idxP0 = (idx << 1);

				mel = opCode(load_ps, &m[idxP0]);
				mPy = opCode(mul_ps, mel, mel);

#if	defined(__AVX__)// || defined(__AVX512F__)
				mPx = opCode(add_ps, opCode(permute_ps, mPy, 0b10110001), mPy);
#else
				mPx = opCode(add_ps, opCode(shuffle_ps, mPy, mPy, 0b10110001), mPy);
#endif
				switch	(VQcd) {
					case	VQCD_1:
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
										opCode(mul_ps, mel, opCode(set1_ps, -6.0))),
									opCode(set1_ps, ood2)),
								zQVec),
							opCode(mul_ps,
								opCode(mul_ps,
									opCode(sub_ps, mPx, opCode(set1_ps, z2)),
									opCode(set1_ps, LL)),
								mel));
						break;

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
										opCode(mul_ps, mel, opCode(set1_ps, -6.0))),
									opCode(set1_ps, ood2)),
								opCode(mul_ps, opCode(set1_ps, zQ), opCode(sub_ps, mel, zRVec))),
							opCode(mul_ps,
								opCode(mul_ps,
									opCode(sub_ps, mPx, opCode(set1_ps, z2)),
									opCode(set1_ps, LL)),
								mel));
						break;
				}

				mPy = opCode(load_ps, &v[idxMz]);

#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_ps, mMx, opCode(set1_ps, dzc), mPy);
				mPx = opCode(fmadd_ps, tmp, opCode(set1_ps, dzd), mel);
#else
				tmp = opCode(add_ps, mPy, opCode(mul_ps, mMx, opCode(set1_ps, dzc)));
				mPx = opCode(add_ps, mel, opCode(mul_ps, tmp, opCode(set1_ps, dzd)));
#endif
				opCode(store_ps,  &v[idxMz], tmp);
				opCode(store_ps, &m2[idxP0], mPx);
			}
		}
#undef	_MData_
#undef	step
	}
}

inline	void	updateMXeon(void * __restrict__ m_, const void * __restrict__ v_, const double dz, const double d, const size_t Vo, const size_t Vf, const size_t Sf, FieldPrecision precision)
{
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

		double * __restrict__ m		= (double * __restrict__) __builtin_assume_aligned (m_, Align);
		const double * __restrict__ v	= (const double * __restrict__) __builtin_assume_aligned (v_, Align);

		const double dzd = dz*d;

		#pragma omp parallel default(shared)
		{
			register _MData_ mIn, vIn, tmp;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
#if	defined(__AVX512F__) || defined(__FMA__)
				vIn = opCode(load_pd, &v[2*(idx-Sf)]);
				mIn = opCode(load_pd, &m[2*idx]);
				tmp = opCode(fmadd_pd, opCode(set1_pd, dzd), vIn, mIn);
				opCode(store_pd, &m[2*idx], tmp);
#else
				mIn = opCode(load_pd, &m[2*idx]);
				tmp = opCode(load_pd, &v[2*(idx-Sf)]);
				vIn = opCode(mul_pd, opCode(set1_pd, dzd), tmp);
				tmp = opCode(add_pd, mIn, vIn);
				opCode(store_pd, &m[2*idx], tmp);
#endif
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
#elif	defined(__AVX__)
	#define	_MData_ __m256
	#define	step 4
#else
	#define	_MData_ __m128
	#define	step 2
#endif

		float * __restrict__ m		= (float * __restrict__) __builtin_assume_aligned (m_, Align);
		const float * __restrict__ v	= (const float * __restrict__) __builtin_assume_aligned (v_, Align);

		const float dzd = dz*d;
#if	defined(__AVX512F__)
		const float __attribute__((aligned(Align))) dzdAux[16] = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd  };
#elif	defined(__AVX__)
		const float __attribute__((aligned(Align))) dzdAux[8]  = { dzd, dzd, dzd, dzd, dzd, dzd, dzd, dzd };
#else
		const float __attribute__((aligned(Align))) dzdAux[4]  = { dzd, dzd, dzd, dzd };
#endif
		const _MData_ dzdVec = opCode(load_ps, dzdAux);

		#pragma omp parallel default(shared)
		{
			register _MData_ mIn, vIn, tmp;
			register size_t idxP0, idxMz;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				idxP0 = idx << 1;
				idxMz = (idx - Sf) << 1;
#if	defined(__AVX512F__) || defined(__FMA__)
				vIn = opCode(load_ps, &v[idxMz]);
				mIn = opCode(load_ps, &m[idxP0]);
				tmp = opCode(fmadd_ps, opCode(set1_ps, dzd), vIn, mIn);
				opCode(store_ps, &m[idxP0], tmp);
#else
				vIn = opCode(load_ps, &v[idxMz]);
				mIn = opCode(load_ps, &m[idxP0]);
				tmp = opCode(add_ps, mIn, opCode(mul_ps, opCode(set1_ps, dzd), vIn));
				opCode(store_ps, &m[idxP0], tmp);
#endif
			}
		}
#undef	_MData_
#undef	step
	}
}

template<const VqcdType VQcd>
inline	void	updateVXeon(const void * __restrict__ m_, void * __restrict__ v_, double *z, const double dz, const double c, const double ood2,
			    const double LL, const double nQcd, const size_t Lx, const size_t Vo, const size_t Vf, const size_t Sf, FieldPrecision precision)
{
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

		const double * __restrict__ m = (const double * __restrict__) __builtin_assume_aligned (m_, Align);
		double * __restrict__ v = (double * __restrict__) __builtin_assume_aligned (v_, Align);

		const double zR = *z;
		const double z2 = zR*zR;
		//const double zQ = 9.*pow(zR, nQcd+3.);
		const double zQ = axionmass2(zR, nQcd, zthres, zrestore)*zR*zR*zR;
		const double dzc = dz*c;
#if	defined(__AVX512F__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const double __attribute__((aligned(Align))) zQAux[8] = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zRAux[8] = { zR, 0., zR, 0., zR, 0., zR, 0. };	// Only real part
		const int    __attribute__((aligned(Align))) shfRg[8] = {6, 7, 0, 1, 2, 3, 4, 5 };
		const int    __attribute__((aligned(Align))) shfLf[8] = {2, 3, 4, 5, 6, 7, 0, 1 };

		const _MInt_  vShRg  = opCode(load_si512, shfRg);
		const _MInt_  vShLf  = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) zQAux[4] = { zQ, 0., zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zRAux[4] = { zR, 0., zR, 0. };	// Only real part
#else
		const size_t XC = Lx;
		const size_t YC = Lx;

		const double __attribute__((aligned(Align))) zQAux[2] = { zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zRAux[2] = { zR, 0. };	// Only real part
#endif
		const _MData_ zQVec  = opCode(load_pd, zQAux);
		const _MData_ zRVec  = opCode(load_pd, zRAux);

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, mPx, mPy, mMx;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxPx, idxMx, idxPy, idxMy, idxPz, idxMz, idxP0;

				{
					size_t tmi = idx/XC, tpi;

					tpi = tmi/YC;
					X[1] = tmi - tpi*YC;
					X[0] = idx - tmi*XC;
				}

				if (X[0] == XC-step)
					idxPx = ((idx - XC + step) << 1);
				else
					idxPx = ((idx + step) << 1);

				if (X[0] == 0)
					idxMx = ((idx + XC - step) << 1);
				else
					idxMx = ((idx - step) << 1);

				if (X[1] == 0)
				{
					idxMy = ((idx + Sf - XC) << 1);
					idxPy = ((idx + XC) << 1);
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
					idxMy = ((idx - XC) << 1);

					if (X[1] == YC-1)
					{
						idxPy = ((idx - Sf + XC) << 1);
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
						idxPy = ((idx + XC) << 1);
						tmp = opCode(add_pd, opCode(load_pd, &m[idxMy]), opCode(load_pd, &m[idxPy]));
					}
				}

				idxPz = ((idx+Sf) << 1);
				idxMz = ((idx-Sf) << 1);
				idxP0 = (idx << 1);

				mel = opCode(load_pd, &m[idxP0]);
				mPy = opCode(mul_pd, mel, mel);

#if	defined(__AVX512F__)
				mPx = opCode(add_pd, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, mPy), _MM_PERM_BADC)), mPy);
#elif	defined(__AVX__)
				mPx = opCode(add_pd, opCode(permute_pd, mPy, 0b00000101), mPy);
#else
				mPx = opCode(add_pd, opCode(shuffle_pd, mPy, mPy, 0b00000001), mPy);
#endif
				switch	(VQcd) {
					case	VQCD_1:
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
								zQVec),
							opCode(mul_pd,
								opCode(mul_pd,
									opCode(sub_pd, mPx, opCode(set1_pd, z2)),
									opCode(set1_pd, LL)),
								mel));
						break;

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
									opCode(set1_pd, (float) ood2)),
								opCode(mul_pd, opCode(set1_pd, zQ), opCode(sub_pd, mel, zRVec))),
							opCode(mul_pd,
								opCode(mul_pd,
									opCode(sub_pd, mPx, opCode(set1_pd, z2)),
									opCode(set1_pd, LL)),
								mel));
						break;
				}

				mPy = opCode(load_pd, &v[idxMz]);
#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_pd, mMx, opCode(set1_pd, dzc), mPy);
#else
				tmp = opCode(add_pd, mPy, opCode(mul_pd, mMx, opCode(set1_pd, dzc)));
#endif
				opCode(store_pd,  &v[idxMz], tmp);
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
#elif	defined(__AVX__)
	#define	_MData_ __m256
	#define	step 4
#else
	#define	_MData_ __m128
	#define	step 2
#endif

		const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_, Align);
		float * __restrict__ v		= (float * __restrict__) __builtin_assume_aligned (v_, Align);

		const float zR = *z;
		const float z2 = zR*zR;
		//const float zQ = 9.*powf(zR, nQcd+3.);
		const float zQ = (float) axionmass2( (double) zR, nQcd, zthres, zrestore)*zR*zR*zR;
		const float dzc = dz*c;
#if	defined(__AVX512F__)
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) zQAux[16]  = { zQ, 0., zQ, 0., zQ, 0., zQ, 0., zQ, 0., zQ, 0., zQ, 0., zQ, 0.};
		const float __attribute__((aligned(Align))) zRAux[16]  = { zR, 0., zR, 0., zR, 0., zR, 0., zR, 0., zR, 0., zR, 0., zR, 0.};
		const int   __attribute__((aligned(Align))) shfRg[16] = {14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
		const int   __attribute__((aligned(Align))) shfLf[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1};

		const _MInt_  vShRg  = opCode(load_si512, shfRg);
		const _MInt_  vShLf  = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) zQAux[8]  = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };
		const float __attribute__((aligned(Align))) zRAux[8]  = { zR, 0., zR, 0., zR, 0., zR, 0. };
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);

		const float __attribute__((aligned(Align))) zQAux[4]  = { zQ, 0., zQ, 0. };
		const float __attribute__((aligned(Align))) zRAux[4]  = { zR, 0., zR, 0. };
#endif
		const _MData_ zQVec  = opCode(load_ps, zQAux);
		const _MData_ zRVec  = opCode(load_ps, zRAux);

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, mPx, mPy, mMx;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0;

				{
					size_t tmi = idx/XC, itp;

					itp = tmi/YC;
					X[1] = tmi - itp*YC;
					X[0] = idx - tmi*XC;
				}

				if (X[0] == XC-step)
					idxPx = ((idx - XC + step) << 1);
				else
					idxPx = ((idx + step) << 1);

				if (X[0] == 0)
					idxMx = ((idx + XC - step) << 1);
				else
					idxMx = ((idx - step) << 1);

				if (X[1] == 0)
				{
					idxMy = ((idx + Sf - XC) << 1);
					idxPy = ((idx + XC) << 1);

#if	defined(__AVX512F__)
					tmp = opCode(add_ps, opCode(permutexvar_ps, vShRg, opCode(load_ps, &m[idxMy])), opCode(load_ps, &m[idxPy]));
#elif	defined(__AVX2__)	//AVX2
					tmp = opCode(add_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxMy]), opCode(setr_epi32, 6,7,0,1,2,3,4,5)),  opCode(load_ps, &m[idxPy]));
#elif	defined(__AVX__)	//AVX
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
					idxMy = ((idx - XC) << 1);

					if (X[1] == YC-1)
					{
						idxPy = ((idx - Sf + XC) << 1);
#if	defined(__AVX512F__)
						tmp = opCode(add_ps, opCode(permutexvar_ps, vShLf, opCode(load_ps, &m[idxPy])), opCode(load_ps, &m[idxMy]));
#elif	defined(__AVX2__)	//AVX2
						tmp = opCode(add_ps, opCode(permutevar8x32_ps, opCode(load_ps, &m[idxPy]), opCode(setr_epi32, 2,3,4,5,6,7,0,1)), opCode(load_ps, &m[idxMy]));
#elif	defined(__AVX__)	//AVX
						mMx = opCode(permute_ps, opCode(load_ps, &m[idxPy]), 0b01001110);
						mPx = opCode(permute2f128_ps, mMx, mMx, 0b00000001);
						tmp = opCode(add_ps, opCode(blend_ps, mMx, mPx, 0b11001100), opCode(load_ps, &m[idxMy]));
#else
						mMx = opCode(load_ps, &m[idxPy]);
						tmp = opCode(add_ps, opCode(shuffle_ps, mMx, mMx, 0b01001110), opCode(load_ps, &m[idxMy]));
#endif
					}
					else
					{
						idxPy = ((idx + XC) << 1);
						tmp = opCode(add_ps, opCode(load_ps, &m[idxPy]), opCode(load_ps, &m[idxMy]));
					}
				}

				idxPz = ((idx+Sf) << 1);
				idxMz = ((idx-Sf) << 1);
				idxP0 = (idx << 1);

				mel = opCode(load_ps, &m[idxP0]);
				mPy = opCode(mul_ps, mel, mel);

#if	defined(__AVX__) || defined(__AVX512F__)
				mPx = opCode(add_ps, opCode(permute_ps, mPy, 0b10110001), mPy);
#else
				mPx = opCode(add_ps, opCode(shuffle_ps, mPy, mPy, 0b10110001), mPy);
#endif
				switch	(VQcd) {
					case	VQCD_1:
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
										opCode(mul_ps, mel, opCode(set1_ps, -6.0f))),
									opCode(set1_ps, ood2)),
								zQVec),
							opCode(mul_ps,
								opCode(mul_ps,
									opCode(sub_ps, mPx, opCode(set1_ps, z2)),
									opCode(set1_ps, (float) LL)),
								mel));
						break;

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
										opCode(mul_ps, mel, opCode(set1_ps, -6.0f))),
									opCode(set1_ps, (float) ood2)),
								opCode(mul_ps, opCode(set1_ps, zQ), opCode(sub_ps, mel, zRVec))),
							opCode(mul_ps,
								opCode(mul_ps,
									opCode(sub_ps, mPx, opCode(set1_ps, z2)),
									opCode(set1_ps, (float) LL)),
								mel));
						break;
				}

				mPy = opCode(load_ps, &v[idxMz]);

#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_ps, mMx, opCode(set1_ps, dzc), mPy);
#else
				tmp = opCode(add_ps, mPy, opCode(mul_ps, mMx, opCode(set1_ps, dzc)));
#endif
				opCode(store_ps,  &v[idxMz], tmp);
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
