#ifndef	__TRIGINTRINSICS
#define	__TRIGINTRINSICS

#include<cmath>
#include<simd/simd-table.h>

#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

#include <immintrin.h>

#ifdef	__AVX512F__
	#define _MData_ __m512d
	#define	_MInt_  __m512i
	#define	_MHnt_  __m256i
#elif   defined(__AVX__)
	#define _MData_ __m256d
	#define	_MInt_  __m256i
	#define	_MHnt_  __m128i
#else
	#define _MData_ __m128d
	#define	_MInt_  __m128i
#endif

#if	defined(__AVX512F__)
	#define	_PREFIX_ _mm512
	#define	_PREFXL_ _mm256
	#define opCodl(x,...) opCode_N(_PREFXL_, x, __VA_ARGS__)
#else
	#if not defined(__AVX__) and not defined(__AVX2__)
		#define	_PREFIX_ _mm
	#else
		#define	_PREFIX_ _mm256
		#define	_PREFXL_ _mm
		#define opCodl(x,...) opCode_N(_PREFXL_, x, __VA_ARGS__)
	#endif
#endif

#ifndef	__INTEL_COMPILER

/*	Sleef	*/

inline _MData_	opCode(sin_pd, _MData_ x)
{
	_MData_ u, d, s, uh, ul;
#ifndef	__AVX__
	_MInt_  qi;
#else
	_MHnt_  qi;
#endif

#if	defined(__AVX512F__)
	uh = opCode(mul_pd, opCode(roundscale_pd, opCode(mul_pd, x, rPid), 0x03), rCte);
	ul = opCode(roundscale_round_pd, opCode(fmsub_pd, x, oPid, uh), 0x04, _MM_FROUND_NO_EXC | _MM_FROUND_TO_NEAREST_INT);
#elif	defined(__FMA__)
	uh = opCode(mul_pd, opCode(round_pd, opCode(mul_pd, x, rPid), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), rCte);
	ul = opCode(round_pd, opCode(fmsub_pd, x, oPid, uh), _MM_FROUND_NO_EXC | _MM_FROUND_TO_NEAREST_INT);
#else
	uh = opCode(mul_pd, opCode(round_pd, opCode(mul_pd, x, rPid), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), rCte);
	ul = opCode(round_pd, opCode(sub_pd, opCode(mul_pd, x, oPid), uh), _MM_FROUND_NO_EXC | _MM_FROUND_TO_NEAREST_INT);
#endif 
#if	defined(__AVX__)
	qi = opCode(cvtpd_epi32, ul);
#else
	auto tmp = opCode(cvtpd_epi32, ul);
	qi = opCode(shuffle_epi32, tmp, 0b01010000);
#endif

#if	defined(__AVX512F__) || defined(__FMA__)
	d = opCode(fmadd_pd, uh, PiDd, x);
	d = opCode(fmadd_pd, ul, PiDd, d);
	d = opCode(fmadd_pd, uh, PiCd, d);
	d = opCode(fmadd_pd, ul, PiCd, d);
	d = opCode(fmadd_pd, uh, PiBd, d);
	d = opCode(fmadd_pd, ul, PiBd, d);
	d = opCode(fmadd_pd, opCode(add_pd, uh, ul), PiAd, d);
#else
	d = opCode(add_pd, opCode(mul_pd, ul, PiDd), x);
	d = opCode(add_pd, opCode(mul_pd, uh, PiDd), d);
	d = opCode(add_pd, opCode(mul_pd, ul, PiCd), d);
	d = opCode(add_pd, opCode(mul_pd, uh, PiCd), d);
	d = opCode(add_pd, opCode(mul_pd, ul, PiBd), d);
	d = opCode(add_pd, opCode(mul_pd, uh, PiBd), d);
	d = opCode(add_pd, opCode(mul_pd, opCode(add_pd, uh, ul), PiAd), d);
#endif

	s = opCode(mul_pd, d, d);
#ifdef	__AVX512F__
	d = opCode(castsi512_pd, opCode(mask_xor_epi64, opCode(castpd_si512, d),
		opCode(cmpeq_epi64_mask,
			opCode(cvtepi32_epi64, opCodl(and_si256, hOne, opCodl(and_si256, qi, hOne))),
			opCode(set1_epi64, 1)),
		opCode(castpd_si512, d), opCode(castpd_si512, zeroNegd)));
#elif	defined(__AVX2__)
	d = opCode(xor_pd, d,
		opCode(and_pd, zeroNegd,
			opCode(castsi256_pd,
				opCode(permutevar8x32_epi32,
					opCode(castsi128_si256, opCodl(cmpeq_epi32, hOne, opCodl(and_si128, qi, hOne))),
					opCode(set_epi32, 3, 3, 2, 2, 1, 1, 0, 0)))));
#elif	defined(__AVX__)
	d = opCode(xor_pd, d,
		opCode(and_pd, zeroNegd,
			opCode(cmp_pd,
				opCode(cvtepi32_pd, opCodl(cmpeq_epi32, hOne, opCodl(and_si128, qi, hOne))),
				opCode(set1_pd, -1.0),
				_CMP_EQ_UQ)));
#else
	d = opCode(xor_pd, d,
		opCode(and_pd, zeroNegd,
			opCode(castsi128_pd, opCode(cmpeq_epi32, one, opCode(and_si128, qi, one)))));
#endif

#if	defined(__AVX512F__) || defined(__FMA__)
	u = opCode(fmadd_pd, s0d, s, s1d); 
	u = opCode(fmadd_pd, u,   s, s2d); 
	u = opCode(fmadd_pd, u,   s, s3d); 
	u = opCode(fmadd_pd, u,   s, s4d); 
	u = opCode(fmadd_pd, u,   s, s5d); 
	u = opCode(fmadd_pd, u,   s, s6d); 
	u = opCode(fmadd_pd, u,   s, s7d); 
	u = opCode(fmadd_pd, u,   s, s8d); 
	u = opCode(fmadd_pd, s, opCode(mul_pd, u, d), d);
#else
	u = opCode(add_pd, opCode(mul_pd, s0d, s), s1d); 
	u = opCode(add_pd, opCode(mul_pd, u,   s), s2d); 
	u = opCode(add_pd, opCode(mul_pd, u,   s), s3d); 
	u = opCode(add_pd, opCode(mul_pd, u,   s), s4d); 
	u = opCode(add_pd, opCode(mul_pd, u,   s), s5d); 
	u = opCode(add_pd, opCode(mul_pd, u,   s), s6d); 
	u = opCode(add_pd, opCode(mul_pd, u,   s), s7d); 
	u = opCode(add_pd, opCode(mul_pd, u,   s), s8d); 
	u = opCode(add_pd, d, opCode(mul_pd, s, opCode(mul_pd, u, d)));
#endif

#ifdef	__AVX512F__
	u = opCode(mask_blend_pd,
		opCode(kandn,
			opCode(cmp_pd_mask, x, dInf, _CMP_EQ_UQ),
			opCode(kor,
				opCode(cmp_pd_mask, x, zeroNegd, _CMP_EQ_UQ),
				opCode(cmp_pd_mask,
					opCode(castsi512_pd, opCode(andnot_si512, opCode(castpd_si512, x), opCode(castpd_si512, zeroNegd))),
					TriMaxd, _CMP_GT_OQ))),
		u, zeroNegd);
#elif	defined(__AVX__)
	u = opCode(blendv_pd, u,
		zeroNegd,
		opCode(andnot_pd, opCode(cmp_pd, x, dInf, _CMP_EQ_UQ),
			opCode(or_pd,
				opCode(cmp_pd, x, zeroNegd, _CMP_EQ_UQ),
				opCode(cmp_pd,
					opCode(andnot_pd, x, zeroNegd),
					TriMaxd,
					_CMP_GT_OQ))));
#else
	u = opCode(blendv_pd, u,
		zeroNegd,
		opCode(andnot_pd, opCode(cmpeq_pd, x, dInf),
			opCode(or_pd,
				opCode(cmpeq_pd, x, zeroNegd),
				opCode(cmpgt_pd,
					opCode(andnot_pd, x, zeroNegd),
					TriMaxd))));
#endif
	return	u;
}

inline _MData_	opCode(cos_pd, _MData_ x)
{
	_MData_ u, d, s, uh, ul;
#ifndef	__AVX__
	_MInt_  qi;
#else
	_MHnt_  qi;
#endif

#if	defined(__AVX512F__)
	uh = opCode(roundscale_pd, opCode(fmsub_pd, x, dPid, rPid), 0x03);
	qi = opCode(cvtpd_epi32, opCode(fmadd_pd, x, oPid, opCode(fnmsub_pd, uh, dPid, dHlf)));
#elif	defined(__FMA__)
	uh = opCode(round_pd, opCode(fmsub_pd, x, dPid, rPid), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
	qi = opCode(cvtpd_epi32, opCode(fmadd_pd, x, oPid, opCode(fnmsub_pd, uh, dPid, dHlf)));
#else
	uh = opCode(round_pd, opCode(sub_pd, opCode(mul_pd, x, dPid), rPid), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
	qi = opCode(cvtpd_epi32, opCode(sub_pd, opCode(mul_pd, x, oPid), opCode(add_pd, opCode(mul_pd, uh, dPid), dHlf)));
#endif
	uh = opCode(mul_pd, uh, rCte);
#if	defined(__AVX__)
	qi = opCodl(add_epi32, opCodl(add_epi32, qi, qi), hOne);
	ul = opCode(cvtepi32_pd, qi);
#else
	qi = opCode(add_epi32, opCode(add_epi32, qi, qi), one);
	ul = opCode(cvtepi32_pd, qi);
	qi = opCode(shuffle_epi32, qi, 0b01010000);
#endif

#if	defined(__AVX512F__) || defined(__FMA__)
	d = opCode(fmadd_pd, uh, hPiDd, x);
	d = opCode(fmadd_pd, ul, hPiDd, d);
	d = opCode(fmadd_pd, uh, hPiCd, d);
	d = opCode(fmadd_pd, ul, hPiCd, d);
	d = opCode(fmadd_pd, uh, hPiBd, d);
	d = opCode(fmadd_pd, ul, hPiBd, d);
	d = opCode(fmadd_pd, opCode(add_pd, uh, ul), hPiAd, d);
#else
	d = opCode(add_pd, opCode(mul_pd, ul, hPiDd), x);
	d = opCode(add_pd, opCode(mul_pd, uh, hPiDd), d);
	d = opCode(add_pd, opCode(mul_pd, ul, hPiCd), d);
	d = opCode(add_pd, opCode(mul_pd, uh, hPiCd), d);
	d = opCode(add_pd, opCode(mul_pd, ul, hPiBd), d);
	d = opCode(add_pd, opCode(mul_pd, uh, hPiBd), d);
	d = opCode(add_pd, opCode(mul_pd, opCode(add_pd, uh, ul), hPiAd), d);
#endif

	s = opCode(mul_pd, d, d);

#ifdef	__AVX512F__
	d = opCode(castsi512_pd, opCode(mask_xor_epi64, opCode(castpd_si512, d),
		opCode(cmpneq_epi64_mask,
			opCode(cvtepi32_epi64, opCodl(and_si256, qi, hTwo)),
			opCode(set1_epi64, 2)),
		opCode(castpd_si512, d), opCode(castpd_si512, zeroNegd)));
#elif	defined(__AVX2__)
	d = opCode(xor_pd, d,
		opCode(and_pd, zeroNegd,
			opCode(castsi256_pd,
				opCode(permutevar8x32_epi32,
					opCode(castsi128_si256, opCodl(cmpeq_epi32, iZerh, opCodl(and_si128, qi, hTwo))),
					opCode(set_epi32, 3, 3, 2, 2, 1, 1, 0, 0)))));
#elif	defined(__AVX__)
	d = opCode(xor_pd, d,
		opCode(and_pd, zeroNegd,
			opCode(cmp_pd,
				opCode(cvtepi32_pd, opCodl(cmpeq_epi32, iZerh, opCodl(and_si128, qi, hTwo))),
				opCode(set1_pd, -1.0),
				_CMP_EQ_UQ)));
#else
	d = opCode(xor_pd, d,
		opCode(and_pd, zeroNegd,
			opCode(castsi128_pd, opCode(cmpeq_epi32, iZero, opCode(and_si128, qi, two)))));
#endif

#if	defined(__AVX512F__) || defined(__FMA__)
	u = opCode(fmadd_pd, s0d, s, s1d); 
	u = opCode(fmadd_pd, u,   s, s2d); 
	u = opCode(fmadd_pd, u,   s, s3d); 
	u = opCode(fmadd_pd, u,   s, s4d); 
	u = opCode(fmadd_pd, u,   s, s5d); 
	u = opCode(fmadd_pd, u,   s, s6d); 
	u = opCode(fmadd_pd, u,   s, s7d); 
	u = opCode(fmadd_pd, u,   s, s8d); 
	u = opCode(fmadd_pd, s, opCode(mul_pd, u, d), d);
#else
	u = opCode(add_pd, opCode(mul_pd, s0d, s), s1d); 
	u = opCode(add_pd, opCode(mul_pd, u,   s), s2d); 
	u = opCode(add_pd, opCode(mul_pd, u,   s), s3d); 
	u = opCode(add_pd, opCode(mul_pd, u,   s), s4d); 
	u = opCode(add_pd, opCode(mul_pd, u,   s), s5d); 
	u = opCode(add_pd, opCode(mul_pd, u,   s), s6d); 
	u = opCode(add_pd, opCode(mul_pd, u,   s), s7d); 
	u = opCode(add_pd, opCode(mul_pd, u,   s), s8d); 
	u = opCode(add_pd, d, opCode(mul_pd, s, opCode(mul_pd, u, d)));
#endif

#ifdef	__AVX512F__
	u = opCode(mask_blend_pd,
		opCode(kandn,
			opCode(cmp_pd_mask, x, dInf, _CMP_EQ_UQ),
			opCode(kor,
				opCode(cmp_pd_mask, x, zeroNegd, _CMP_EQ_UQ),
				opCode(cmp_pd_mask, opCode(castsi512_pd, opCode(andnot_si512, opCode(castpd_si512, x), opCode(castpd_si512, zeroNegd))), TriMaxd, _CMP_GT_OQ))),
		u, dOne);
#elif	defined(__AVX__)
	u = opCode(blendv_pd, u,
		dOne,
		opCode(andnot_pd, opCode(cmp_pd, x, dInf, _CMP_EQ_UQ),
			opCode(or_pd,
				opCode(cmp_pd, x, zeroNegd, _CMP_EQ_UQ),
				opCode(cmp_pd,
					opCode(andnot_pd, x, zeroNegd),
					TriMaxd,
					_CMP_GT_OQ))));
#else
	u = opCode(blendv_pd, u,
		dOne,
		opCode(andnot_pd, opCode(cmpeq_pd, x, dInf),
			opCode(or_pd,
				opCode(cmpeq_pd, x, zeroNegd),
				opCode(cmpgt_pd,
					opCode(andnot_pd, x, zeroNegd),
					TriMaxd))));
#endif
	return	u;
}

#undef	_MData_

#if	defined(__AVX512F__)
	#define	_MData_ __m512
#elif	defined(__AVX__)
	#define	_MData_ __m256
#else
	#define	_MData_ __m128
#endif 
#endif

#ifndef	__INTEL_COMPILER

inline _MData_	opCode(sin_ps, _MData_ x)
{
	_MData_ u, d, s;
	_MInt_  q;

	q = opCode(cvtps_epi32, opCode(mul_ps, x, oPif));
	u = opCode(cvtepi32_ps, q);

#if	defined(__AVX512F__) || defined(__FMA__)
	d = opCode(fmadd_ps, u, PiAf, x);
	d = opCode(fmadd_ps, u, PiBf, d);
	d = opCode(fmadd_ps, u, PiCf, d);
	d = opCode(fmadd_ps, u, PiDf, d);
#else
	d = opCode(add_ps, opCode(mul_ps, u, PiAf), x);
	d = opCode(add_ps, opCode(mul_ps, u, PiBf), d);
	d = opCode(add_ps, opCode(mul_ps, u, PiCf), d);
	d = opCode(add_ps, opCode(mul_ps, u, PiDf), d);
#endif

	s = opCode(mul_ps, d, d);
#ifdef	__AVX512F__
	d = opCode(castsi512_ps, opCode(mask_xor_epi32, opCode(castps_si512, d), opCode(cmpeq_epi32_mask, one, opCode(and_epi32, q, one)), opCode(castps_si512, d), opCode(castps_si512, zeroNegf)));
#elif	defined(__AVX2__)
	d = opCode(xor_ps, d,
		opCode(and_ps, zeroNegf,
			opCode(castsi256_ps, opCode(cmpeq_epi32, one,
				opCode(castps_si256, opCode(and_ps, opCode(castsi256_ps, q), opCode(castsi256_ps, one)))))));
#elif	defined(__AVX__)
	d = opCode(xor_ps, d,
		opCode(and_ps, zeroNegf,
			opCode(cmp_ps,
				opCode(castsi256_ps, one),
				opCode(and_ps, opCode(castsi256_ps, q), opCode(castsi256_ps, one)),
				_CMP_EQ_UQ)));
#else
	d = opCode(xor_ps, d,
		opCode(and_ps, zeroNegf,
			opCode(castsi128_ps, opCode(cmpeq_epi32, one, opCode(and_si128, q, one)))));
#endif

#if	defined(__AVX512F__) || defined(__FMA__)
	u = opCode(fmadd_ps, s0f, s, s1f); 
	u = opCode(fmadd_ps, u,   s, s2f); 
	u = opCode(fmadd_ps, u,   s, s3f); 
	u = opCode(fmadd_ps, s, opCode(mul_ps, u, d), d);
#else
	u = opCode(add_ps, opCode(mul_ps, s0f, s), s1f); 
	u = opCode(add_ps, opCode(mul_ps, u,   s), s2f); 
	u = opCode(add_ps, opCode(mul_ps, u,   s), s3f); 
	u = opCode(add_ps, d, opCode(mul_ps, s, opCode(mul_ps, u, d)));
#endif

#ifdef	__AVX512F__
	u = opCode(mask_blend_ps,
		opCode(kor,
			opCode(cmp_ps_mask, x, zeroNegf, _CMP_EQ_UQ),
			opCode(cmp_ps_mask, opCode(castsi512_ps, opCode(andnot_si512, opCode(castps_si512, x), opCode(castps_si512, zeroNegf))), TriMaxf, _CMP_GT_OQ)),
		u, zeroNegf);
#elif	defined(__AVX__)
	u = opCode(blendv_ps, u,
		zeroNegf,
		opCode(or_ps,
			opCode(cmp_ps, x, zeroNegf, _CMP_EQ_UQ),
			opCode(cmp_ps,
				opCode(andnot_ps, x, zeroNegf),
				TriMaxf,
				_CMP_GT_OQ)));

	u = opCode(or_ps, u, opCode(cmp_ps, d, fInf, _CMP_EQ_UQ));
#else
	u = opCode(blendv_ps, u,
		zeroNegf,
		opCode(or_ps,
			opCode(cmpeq_ps, x, zeroNegf),
			opCode(cmpgt_ps,
				opCode(andnot_ps, x, zeroNegf),
				TriMaxf)));
	u = opCode(or_ps, u, opCode(cmpeq_ps, d, fInf));
#endif

	return	u;
}

inline _MData_	opCode(cos_ps, _MData_ x)
{
	_MData_ u, d, s;
	_MInt_  q;

	q = opCode(cvtps_epi32, opCode(sub_ps, opCode(mul_ps, x, oPif), fHlf));
#if	defined(__AVX__) && (not defined(__AVX2__) && not defined(__AVX512F__))
	auto uq = opCode(extractf128_si256, q, 1);

	uq = opCodl(add_epi32, opCodl(add_epi32, uq, uq), opCode(castsi256_si128, one));

	q = opCode(castsi128_si256, opCodl(add_epi32, opCodl(add_epi32, opCode(castsi256_si128, q), opCode(castsi256_si128, q)), opCode(castsi256_si128, one)));
	q = opCode(insertf128_si256, q, uq, 1);
#else
	q = opCode(add_epi32, opCode(add_epi32, q, q), one);
#endif
	u = opCode(cvtepi32_ps, q);

#if	defined(__AVX512F__) || defined(__FMA__)
	d = opCode(fmadd_ps, u, hPiAf, x);
	d = opCode(fmadd_ps, u, hPiBf, d);
	d = opCode(fmadd_ps, u, hPiCf, d);
	d = opCode(fmadd_ps, u, hPiDf, d);
#else
	d = opCode(add_ps, opCode(mul_ps, u, hPiAf), x);
	d = opCode(add_ps, opCode(mul_ps, u, hPiBf), d);
	d = opCode(add_ps, opCode(mul_ps, u, hPiCf), d);
	d = opCode(add_ps, opCode(mul_ps, u, hPiDf), d);
#endif

	s = opCode(mul_ps, d, d);
#ifdef	__AVX512F__
	d = opCode(castsi512_ps, 
		opCode(mask_xor_epi32, opCode(castps_si512, d), opCode(cmpeq_epi32_mask, iZero, opCode(and_epi32, q, two)), opCode(castps_si512, d), opCode(castps_si512, zeroNegf)));
#elif	defined(__AVX2__)
	d = opCode(xor_ps, d,
		opCode(and_ps, zeroNegf,
			opCode(castsi256_ps, opCode(cmpeq_epi32, iZero,
				opCode(castps_si256, opCode(and_ps, opCode(castsi256_ps, q), opCode(castsi256_ps, two)))))));
#elif	defined(__AVX__)
	d = opCode(xor_ps, d,
		opCode(and_ps, zeroNegf,
			opCode(cmp_ps,
				opCode(and_ps, opCode(castsi256_ps, q), opCode(castsi256_ps, two)),
				opCode(castsi256_ps, iZero),
				_CMP_EQ_UQ)));
#else
	d = opCode(xor_ps, d,
		opCode(and_ps, zeroNegf,
			opCode(castsi128_ps, opCode(cmpeq_epi32, iZero, opCode(and_si128, q, two)))));
#endif

#if	defined(__AVX512F__) || defined(__FMA__)
	u = opCode(fmadd_ps, s0f, s, s1f); 
	u = opCode(fmadd_ps, u,   s, s2f); 
	u = opCode(fmadd_ps, u,   s, s3f); 
	u = opCode(fmadd_ps, s, opCode(mul_ps, u, d), d);
#else
	u = opCode(add_ps, opCode(mul_ps, s0f, s), s1f); 
	u = opCode(add_ps, opCode(mul_ps, u,   s), s2f); 
	u = opCode(add_ps, opCode(mul_ps, u,   s), s3f); 
	u = opCode(add_ps, d, opCode(mul_ps, s, opCode(mul_ps, u, d)));
#endif

#ifdef	__AVX512F__
	u = opCode(mask_blend_ps, opCode(cmp_ps_mask, d, fInf, _CMP_EQ_UQ),
		opCode(mask_blend_ps, opCode(knot, opCode(cmp_ps_mask, opCode(castsi512_ps, opCode(andnot_si512, opCode(castps_si512, x), opCode(castps_si512, zeroNegf))), TriMaxf, _CMP_GT_OQ)),
			zeroNegf, u),
		fInf);
#elif	defined(__AVX__)
	u = opCode(andnot_ps,
		opCode(cmp_ps,
			opCode(andnot_ps, x, zeroNegf),
			TriMaxf,
			_CMP_GT_OQ),
		u);

	u = opCode(or_ps, u, opCode(cmp_ps, d, fInf, _CMP_EQ_UQ));
#else
	u = opCode(andnot_ps,
		opCode(cmpgt_ps,
			opCode(andnot_ps, x, zeroNegf),
			TriMaxf),
		u);
	u = opCode(or_ps, u, opCode(cmpeq_ps, d, fInf));
#endif

	return	u;
}

#endif

#undef	_MData_
#undef	_PREFIX
#undef opCode_P
#undef opCode_N
#undef opCode

#endif
