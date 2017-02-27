#include<cmath>

#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

#include <immintrin.h>

#ifdef  __MIC__
	#define _MData_ __m512d
#elif   defined(__AVX__)
	#define _MData_ __m256d
#else
	#define _MData_ __m128d
#endif

#ifdef	__MIC__
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

#undef	_MData_

#ifdef	__MIC__
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
	_MData_ tmp2, tmp4, tmp6, a, b, c;
	static const double a_s = -0.0415758, b_s = 0.00134813, c_s = -(1+4.*M_PI2*a_s+6.*M_PI4*b_s)/(8.*M_PI6);

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
				opCode(mul_ps, c, opCode(mul_ps, tmp4, tmp4)))));
}

#undef	_MData_
#undef	_PREFIX
#undef opCode_P
#undef opCode_N
#undef opCode
