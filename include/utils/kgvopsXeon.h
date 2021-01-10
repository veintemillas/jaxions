#include <cmath>
#include <complex>

#include <omp.h>

#include "enum-field.h"
#include "scalar/scalarField.h"
#include "utils/index.h"

#include "utils/triSimd.h"
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

/* Define template functions that build the velocity vectorised
vr0 vi0 vr1 vi1 ... + vr4 vi4 vr5 vi 5 ...
mr0 mi0 mr1 mi1 ... + mr4 mi4 mr5 mi5 ...
[vi0*mr[0]-vr[0]mi[0]/(mr0mr0+mi0mi0)] [1] [2] [3] ... [8]

requires straightforward ops + fusing two vectors before writting


vr0 vi0 vr1 vi1 ... + vr4 vi4 vr5 vi 5 ...
-vi0 vr0 -vi1 vr1 ... + -vi4 vr4 -vi5 vr5 ...   permute 01 23 and multiply -1,1
-vi0mr0 vi0mi0 -vi1mr1 vr1mi1 ... + ...         multiply times m
build
-vi0mr0+vi0mi0/M2 ??? -vi1mr1+vr1mi1/M2 ???
merge
 */
 using namespace indexXeon;

void	buildc_k_KernelXeon(const void * __restrict__ m_, const void * __restrict__ v_, void * __restrict__ m2_, Scalar *fieldo)
{

	FieldPrecision precision = fieldo->Precision();

	/* computes physical energies, not comoving*/

	const size_t NN     = fieldo->getNg();
 	const size_t Lx     = fieldo->Length();
	const size_t Lz     = fieldo->Depth();
 	const double R      = *(fieldo->RV());
	const size_t Sf     = fieldo->Surf();
	const size_t Vo     = fieldo->Surf()*NN;
	const size_t Vf     = fieldo->Size()+Vo;

// LogMsg(VERB_DEBUG,"Sf %d Vt %d NN %d", Sf, Vt, NN);LogFlush();

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

		const double * __restrict__ m	= (const double * __restrict__) __builtin_assume_aligned (m_,  Align);
		const double * __restrict__ v	= (const double * __restrict__) __builtin_assume_aligned (v_,  Align);
		      double * __restrict__ m2	= (      double * __restrict__) __builtin_assume_aligned (m2_, Align);

#if	defined(__AVX512F__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);
		const double    __attribute__((aligned(Align))) cjgAux[8] = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
#elif	defined(__AVX__)
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);
		const double __attribute__((aligned(Align))) cjgAux[4] = { 1.,-1., 1.,-1. };
#else
		const size_t XC = Lx;
		const size_t YC = Lx;
		const double __attribute__((aligned(Align))) cjgAux[2] = { 1.,-1. };
#endif
		const _MData_ RVec = opCode(set1_pd, R);
		const _MData_ cjg  = opCode(load_pd, cjgAux);

		#pragma omp parallel default(shared)
		{
			_MData_ mel1, vel1, mod1, lap1, mel2, vel2, mod2, lap2;
			size_t X[3], idxP1, idxV1, idxP2, idxV2,idx0;

			double tmpS[2*step] __attribute__((aligned(Align)));

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += 2*step)
			{
				idx0  =  idx - Vo;
				idxP1 =  (idx          << 1);
				idxV1 =  (idx0         << 1);
				idxP2 =  (idx  + step  << 1);
				idxV2 =  (idx0 + step  << 1);

// conformal field value
				mel1 = opCode(load_pd, &m[idxP1]);
				mel2 = opCode(load_pd, &m[idxP2]);

// load field velocity
				vel1 = opCode(load_pd, &v[idxV1]);
				vel2 = opCode(load_pd, &v[idxV2]);
// r^2, i^2...
				mod1 = opCode(mul_pd, mel1, mel1);
				mod2 = opCode(mul_pd, mel2, mel2);
// r^2+i^2, r^2+i^2  (m^2)
				mod1 = opCode(md2_pd, mod1);
				mod2 = opCode(md2_pd, mod2);
// r/m^2, i/m^2
				mod1 = opCode(div_pd, mel1, mod1);	// Ahora mod tiene 1/mel
				mod2 = opCode(div_pd, mel2, mod2);

#if	defined(__AVX__)// || defined(__AVX512F__)
				lap1 = opCode(permute_pd, opCode(mul_pd, mod1, cjg), 0b01010101);
				lap2 = opCode(permute_pd, opCode(mul_pd, mod2, cjg), 0b01010101);
				lap1 = opCode(mul_pd, vel1, lap1);
				lap2 = opCode(mul_pd, vel2, lap2);
				lap1 = opCode(add_pd, opCode(permute_pd, lap1, 0b01010101), lap1);
				lap2 = opCode(add_pd, opCode(permute_pd, lap2, 0b01010101), lap2);
#else
				lap1 = opCode(mul_pd, mod1, cjg);
				lap1 = opCode(shuffle_pd, lap1, lap1, 0b00000001);
				lap1 = opCode(mul_pd, vel1, lap1);
				lap1 = opCode(add_pd, opCode(shuffle_pd, lap1, lap1, 0b00000001), lap1);
				lap2 = opCode(mul_pd, mod1, cjg);
				lap2 = opCode(shuffle_pd, lap2, lap2, 0b00000001);
				lap2 = opCode(mul_pd, vel1, lap2);
				lap2 = opCode(add_pd, opCode(shuffle_pd, lap2, lap2, 0b00000001), lap2);
#endif

				/* v0-v1-v2-v3- v4-v5-v6-v7- >> v0v1v2v3v4v5v6v7 */
#if	defined(__AVX512F__)
								// mod1 = opCode(blend_pd, lap1, lap2, 0b10101010);
#elif	defined(__AVX__)
								// mod1 = opCode(blend_pd, lap1, lap2, 0b10101010);
#else
								mod1 = opCode(blend_pd, lap1, lap2, 0b10101010);
#endif
				/* scale factor for conformal field velocity */
				mod1 = opCode(mul_pd, mod1, RVec);

				/* store field unpadded */
				// opCode(store_pd,  &m2[idx0], mod1);

				/* store field padded.- does not work */
				// size_t tmp = idx0/Lx;
				// unsigned long long iNx = idx0 + 2*tmp;
				// opCode(store_pd, &m2[iNx], mod1);

				/* store field padded it could be done directly
				assumes Lx is multiple of vector size*/
				size_t tmp = idx0/Lx;
				unsigned long long iNx = idx0 + 2*tmp;
				opCode(store_pd, tmpS, mod1);
				#pragma unroll
				for (int ih=0; ih<2*step; ih++)
					m2[iNx+ih]    = tmpS[ih];

			} //end for loop
		} //end parallel

#undef	_MData_
#undef	step
} //end loop DOUBLE precision
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

		const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_,  Align);
		const float * __restrict__ v	= (const float * __restrict__) __builtin_assume_aligned (v_,  Align);
		      float * __restrict__ m2	= (      float * __restrict__) __builtin_assume_aligned (m2_, Align);


#if	defined(__AVX512F__)
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);
		const float __attribute__((aligned(Align))) cjgAux[16]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
#elif	defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);
		const float __attribute__((aligned(Align))) cjgAux[8]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);
		const float __attribute__((aligned(Align))) cjgAux[4]  = { 1.,-1., 1.,-1. };
#endif

		const _MData_ RVec = opCode(set1_ps, (float) R);
		const _MData_ cjg  = opCode(load_ps, cjgAux);

		#pragma omp parallel default(shared)
		{
			_MData_ mel1, vel1, mod1, lap1, mel2, vel2, mod2, lap2;
			size_t X[3], idxP1, idxV1, idxP2, idxV2, idx0;
			float tmpS[2*step] __attribute__((aligned(Align)));

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += 2*step)
			{
				idx0  =  idx - Vo;
				idxP1 =  (idx          << 1);
				idxV1 =  (idx0         << 1);
				idxP2 =  (idx  + step  << 1);
				idxV2 =  (idx0 + step  << 1);

// conformal field value
				mel1 = opCode(load_ps, &m[idxP1]);
				mel2 = opCode(load_ps, &m[idxP2]);

// load field velocity
				vel1 = opCode(load_ps, &v[idxV1]);
				vel2 = opCode(load_ps, &v[idxV2]);
// r^2, i^2...
				mod1 = opCode(mul_ps, mel1, mel1);
				mod2 = opCode(mul_ps, mel2, mel2);
// r^2+i^2, r^2+i^2  (m^2)
				mod1 = opCode(md2_ps, mod1);
				mod2 = opCode(md2_ps, mod2);
// r/m^2, i/m^2
				mod1 = opCode(div_ps, mel1, mod1);	// Ahora mod tiene 1/mel
				mod2 = opCode(div_ps, mel2, mod2);

#if	defined(__AVX__)// || defined(__AVX512F__)
				//0.-(-mi mr)/m^2
				lap1 = opCode(permute_ps, opCode(mul_ps, mod1, cjg), 0b10110001);
				lap2 = opCode(permute_ps, opCode(mul_ps, mod2, cjg), 0b10110001);
				//1.- (vr vi)*(-mi mr)/m^2 = (-vr*mi vi*mr)/m^2
				lap1 = opCode(mul_ps, vel1, lap1);
				lap2 = opCode(mul_ps, vel2, lap2);
				//2.- (-ar*mi+ai*mr, -ar*mi+ai*mr)
				lap1 = opCode(add_ps, opCode(permute_ps, lap1, 0b10110001), lap1);
				lap2 = opCode(add_ps, opCode(permute_ps, lap2, 0b10110001), lap2);
#else
				lap1 = opCode(mul_ps, mod1, cjg);
				lap2 = opCode(mul_ps, mod2, cjg);
				lap1 = opCode(shuffle_ps, lap1, lap1, 0b10110001);
				lap2 = opCode(shuffle_ps, lap2, lap2, 0b10110001);
				lap1 = opCode(mul_ps, vel1, lap1);
				lap2 = opCode(mul_ps, vel2, lap2);
				lap1 = opCode(add_ps, opCode(shuffle_ps, lap1, lap1, 0b10110001), lap1);
				lap2 = opCode(add_ps, opCode(shuffle_ps, lap2, lap2, 0b10110001), lap2);
#endif

// if((idx % 256*(256+1)) && (commRank()==0)){
// 	printsVar(lap1,"lap1\n");
// 	printsVar(lap2,"lap2\n");
// }

				/* v0-v1-v2-v3- v4-v5-v6-v7- >> v0v1v2v3v4v5v6v7 */
#if	defined(__AVX512F__)
				// mod1 = opCode(blend_ps, lap1, lap2, 0b10101010);
#elif	defined(__AVX2__)
				lap1 = opCode(permutevar8x32_ps, lap1, opCode(setr_epi32, 0,2,4,6,1,3,5,7));
				lap2 = opCode(permutevar8x32_ps, lap2, opCode(setr_epi32, 0,2,4,6,1,3,5,7));
				mod1 = opCode(blend_ps, lap1, lap2, 0b11110000);
#elif	defined(__AVX__)
				// lap1 = opCode(permute_ps, lap1, opCode(setr_epi32, 0,2,4,6,1,3,5,7));
				// lap2 = opCode(permute_ps, lap2, opCode(setr_epi32, 0,2,4,6,1,3,5,7));
				mod1 = opCode(blend_ps, lap1, lap2, 0b11110000);
#else
				mod1 = opCode(blend_ps, lap1, lap2, 0b10101010);
#endif
				/* scale factor for conformal field velocity */
				mod1 = opCode(mul_ps, mod1, RVec);

				/* store field unpadded */
				// opCode(store_ps,  &m2[idx0], mod1);

				/* store field padded.- does not work */
				// size_t tmp = idx0/Lx;
				// unsigned long long iNx = idx0 + 2*tmp;
				// opCode(store_ps, &m2[iNx], mod1);

				/* store field padded it could be done directly
				assumes Lx is multiple of vector size*/
				size_t tmp = idx0/Lx;
				unsigned long long iNx = idx0 + 2*tmp;
				opCode(store_ps, tmpS, mod1);
				#pragma unroll
				for (int ih=0; ih<2*step; ih++)
					m2[iNx+ih]    = tmpS[ih];


				// if((idx % 256*(256+1)) && (commRank()==0)){
				// 	printsVar(lap1,"lap1-\n");
				// 	printsVar(lap2,"lap2-\n");
				// 	printsVar(mod1,"mod1\n");
				// }

			} //end for loop
		} //end parallel

} //end loop single precision

#undef	_MData_
#undef	step
}
