#include<cmath>

#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

#include <immintrin.h>

#if	defined(__MIC__) || defined(__AVX512F__)
	#define _MData_ __m512d
#elif   defined(__AVX__)
	#define _MData_ __m256d
#else
	#define _MData_ __m128d
#endif

#if	defined(__MIC__) || defined(__AVX512F__)
	#define	_PREFIX_ _mm512
#else
	#if not defined(__AVX__) and not defined(__AVX2__)
		#define	_PREFIX_ _mm
	#else
		#define	_PREFIX_ _mm256
	#endif
#endif

#define	M_PI2	(M_PI *M_PI)
#define	M_PI4	(M_PI2*M_PI2)
#define	M_PI6	(M_PI4*M_PI2)

inline _MData_	opCode(sin_pd, _MData_ x)
{
	_MData_ tmp2, tmp3, tmp5, a, b, c;
	static const double a_s = -0.0415758*4., b_s = 0.00134813*6., c_s = -(1+M_PI2*a_s+M_PI4*b_s)/(M_PI6);

	a = opCode(set1_pd, a_s);
	b = opCode(set1_pd, b_s);
	c = opCode(set1_pd, c_s);

	tmp2 = opCode(mul_pd, x, x);
	tmp3 = opCode(mul_pd, tmp2, x);
	tmp5 = opCode(mul_pd, tmp3, tmp2);
	return opCode(add_pd, x, opCode(add_pd,
		opCode(add_pd,
			opCode(mul_pd, tmp3, a),
			opCode(mul_pd, tmp5, b)),
		opCode(mul_pd, c, opCode(mul_pd, tmp2, tmp5))));
}



inline _MData_	opCode(cos_pd, _MData_ x)
{
	_MData_ tmp2, tmp4, tmp6, a, b, c;
	static const double a_s = -0.0415758, b_s = 0.00134813, c_s = -(1+4.*M_PI2*a_s+6.*M_PI4*b_s)/(8.*M_PI6);

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
				opCode(mul_pd, c, opCode(mul_pd, tmp4, tmp4)))));
}

inline _MData_	opCode(mod_pd, _MData_ &x, const _MData_ &md)
{
	_MData_	min, ret;

	_MData_ xP  = opCode(add_pd, x,  md);
	_MData_ xM  = opCode(sub_pd, x,  md);
	_MData_ x2  = opCode(mul_pd, x,   x);
	_MData_ xP2 = opCode(mul_pd, xP, xP);
	_MData_ xM2 = opCode(mul_pd, xM, xM);
#if	defined(__MIC__) || defined(__AVX512F__)
#ifdef	__MIC__
	min = opCode(gmin_pd, opCode(gmin_pd, xM2, xP2), x2);
#elif	defined(__AVX512F__)
	min = opCode(min_pd,  opCode(min_pd,  xM2, xP2), x2);
#endif
	ret = opCode(mask_add_pd, opCode(setzero_pd), opCode(cmp_pd_mask, min, xP2, _CMP_EQ_OQ), opCode(setzero_pd), xP);
	ret = opCode(mask_add_pd, ret,                opCode(cmp_pd_mask, min, xM2, _CMP_EQ_OQ), ret,   xM);
	ret = opCode(mask_add_pd, ret,                opCode(cmp_pd_mask, min, x2,  _CMP_EQ_OQ), ret,   x);
#elif   defined(__AVX__)
//	ret = opCode(setzero_pd);
	min = opCode(min_pd, opCode(min_pd, xP2, xM2), x2);
	ret = opCode(add_pd,
		opCode(add_pd,
			opCode(and_pd, opCode(cmp_pd, min, xP2, _CMP_EQ_OS), xP),
			opCode(and_pd, opCode(cmp_pd, min, xM2, _CMP_EQ_OS), xM)),
		opCode(and_pd, opCode(cmp_pd, min, x2, _CMP_EQ_OS), x));
#else
//	ret = opCode(setzero_pd);
	min = opCode(min_pd, opCode(min_pd, xP2, xM2), x2);
	ret = opCode(add_pd,
		opCode(add_pd,
			opCode(and_pd, opCode(cmpeq_pd, min, xP2), xP),
			opCode(and_pd, opCode(cmpeq_pd, min, xM2), xM)),
		opCode(and_pd, opCode(cmpeq_pd, min, x2), x));
#endif
	return	ret;
}

inline _MData_	opCode(md2_pd, const _MData_ &x)
{
#ifdef	__MIC__
	return	opCode(add_pd, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, x), _MM_PERM_BADC)), x);
#elif	defined(__AVX512F__)
	return	opCode(add_pd, opCode(permute_pd, x, 0b01010101), x);
#elif defined(__AVX__)
	return	opCode(add_pd, opCode(permute_pd, x, 0b00000101), x);
#else
	return	opCode(add_pd, opCode(shuffle_pd, x, x, 0b00000001), x);
#endif
}

#undef	_MData_

#if	defined(__MIC__) || defined(__AVX512F__)
	#define	_MData_ __m512
#elif	defined(__AVX__)
	#define	_MData_ __m256
#else
	#define	_MData_ __m128
#endif 

inline _MData_	opCode(sin_ps, _MData_ x)
{
	_MData_ tmp2, tmp3, tmp5, a, b, c;
	static const float a_s = -0.0415758f*4.f, b_s = 0.00134813f*6.f, c_s = -(1+M_PI2*a_s+M_PI4*b_s)/(M_PI6);

	a = opCode(set1_ps, a_s);
	b = opCode(set1_ps, b_s);
	c = opCode(set1_ps, c_s);

	tmp2 = opCode(mul_ps, x, x);
	tmp3 = opCode(mul_ps, tmp2, x);
	tmp5 = opCode(mul_ps, tmp3, tmp2);
	return opCode(add_ps, x, opCode(add_ps,
		opCode(add_ps,
			opCode(mul_ps, tmp3, a),
			opCode(mul_ps, tmp5, b)),
		opCode(mul_ps, c, opCode(mul_ps, tmp2, tmp5))));
}

inline _MData_	opCode(cos_ps, _MData_ x)
{
	_MData_ tmp2, tmp4, tmp6;
	static constexpr double a_s = -0.0415758, b_s = 0.00134813, c_s = -(1+4.*M_PI2*a_s+6.*M_PI4*b_s)/(8.*M_PI6);

	const _MData_ a = opCode(set1_ps, a_s);
	const _MData_ b = opCode(set1_ps, b_s);
	const _MData_ c = opCode(set1_ps, c_s);
	const _MData_ o = opCode(set1_ps, 1.0);
	const _MData_ h = opCode(set1_ps, 0.5);

	tmp2 = opCode(mul_ps, x, x);
	tmp4 = opCode(mul_ps, tmp2, tmp2);
	tmp6 = opCode(mul_ps, tmp2, tmp4);
	return opCode(sub_ps, o,
		opCode(add_ps, opCode(mul_ps, h, tmp2),
			opCode(add_ps,
				opCode(add_ps,
					opCode(mul_ps, tmp4, a),
					opCode(mul_ps, tmp6, b)),
				opCode(mul_ps, c, opCode(mul_ps, tmp4, tmp4)))));
}

inline _MData_	opCode(mod_ps, _MData_ &x, const _MData_ &md)
{
	_MData_	min, ret;
/*
	x2 = opCode(div_ps, x, md);
	auto di = opCode(cvtps_epi32, x2);
	auto dv = opCode(cvtepi32_ps, di);
#ifdef	__FMA__
	ret = opCode(fnmadd_ps, dv, md, x);
#else
	ret = opCode(sub_ps, x, opCode(mul_ps, dv, md));
#endif
*/
	_MData_ xP  = opCode(add_ps, x,  md);
	_MData_ xM  = opCode(sub_ps, x,  md);
	_MData_ x2  = opCode(mul_ps, x,   x);
	_MData_ xP2 = opCode(mul_ps, xP, xP);
	_MData_ xM2 = opCode(mul_ps, xM, xM);
#if	defined(__MIC__) || defined(__AVX512F__)
#ifdef	__MIC__
	min = opCode(gmin_ps, opCode(gmin_ps, xM2, xP2), x2);
#elif	defined(__AVX512F__)
	min = opCode(min_ps,  opCode(min_ps,  xM2, xP2), x2);
#endif
	ret = opCode(mask_add_ps, opCode(setzero_ps), opCode(cmp_ps_mask, min, xP2, _CMP_EQ_OQ), opCode(setzero_ps), xP);
	ret = opCode(mask_add_ps, ret,                opCode(cmp_ps_mask, min, xM2, _CMP_EQ_OQ), ret,                xM);
	ret = opCode(mask_add_ps, ret,                opCode(cmp_ps_mask, min, x2,  _CMP_EQ_OQ), ret,                x);
#elif   defined(__AVX__)
//	ret = opCode(setzero_ps);
	min = opCode(min_ps, opCode(min_ps, xP2, xM2), x2);
	ret = opCode(add_ps,
		opCode(add_ps,
			opCode(and_ps, opCode(cmp_ps, min, xP2, _CMP_EQ_OQ), xP),
			opCode(and_ps, opCode(cmp_ps, min, xM2, _CMP_EQ_OQ), xM)),
		opCode(and_ps, opCode(cmp_ps, min, x2, _CMP_EQ_OQ), x));
#else
//	ret = opCode(setzero_ps);
	min = opCode(min_ps, opCode(min_ps, xP2, xM2), x2);
	ret = opCode(add_ps,
		opCode(add_ps,
			opCode(and_ps, opCode(cmpeq_ps, min, xP2), xP),
			opCode(and_ps, opCode(cmpeq_ps, min, xM2), xM)),
		opCode(and_ps, opCode(cmpeq_ps, min, x2), x));
#endif

	return	ret;
}

inline _MData_	opCode(md2_ps, const _MData_ &x)
{
#ifdef	__MIC__
	return	opCode(add_ps, opCode(swizzle_ps, x, _MM_SWIZ_REG_CDAB), x);
#elif	defined(__AVX512F__)
	return	opCode(add_ps, opCode(permute_ps, x, 0b11100001), x);
#elif defined(__AVX__)
	return	opCode(add_ps, opCode(permute_ps, x, 0b10110001), x);
#else
	return	opCode(add_ps, opCode(shuffle_ps, x, x, 0b10110001), x);
#endif
}

#undef	_MData_
#undef	_PREFIX
#undef opCode_P
#undef opCode_N
#undef opCode
