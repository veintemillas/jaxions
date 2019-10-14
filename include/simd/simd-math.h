#ifndef	__LOGEXPINTRINSICS
#define	__LOGEXPINTRINSICS

#include<cmath>

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

#define	M_PI2	(M_PI *M_PI)
#define	M_PI4	(M_PI2*M_PI2)
#define	M_PI6	(M_PI4*M_PI2)

/*	TODO	Optimiza con FMA cuando esté disponible	*/

#ifndef	__INTEL_COMPILER

inline _MData_	opCode(exp_pd, _MData_ x) {
	_MInt_	xM;
	_MHnt_	N, M, n1, n2;
	_MData_ nf, R1, R2, R, Q, Sl, St, S;

/*
	1.  Filtra Nan
	2.  Filtra +inf -> +inf
	3.  Filtra -inf -> 0
        4.  Filtra Threshold_1 -> +inf / 0
	5.  Filtra Threshold_2 -> 1+x	<-- No hace falta, porque vamos a calcular la chunga igual...
*/
#if	defined(__AVX512F__)
	N   = opCode(cvtpd_epi32, opCode(roundscale_round_pd, opCode(mul_pd, x, vIvLd), 0x04, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
	auto mMsk = opCodl(cmpeq_epi32, opCodl(and_si256, hSignMsk, N), opCodl(setzero_si256));
	n2  = opCodl(and_si256, N, h32Mask);
#else
	N   = opCode(cvtpd_epi32, opCode(round_pd, opCode(mul_pd, x, vIvLd), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
	auto mMsk = opCodl(cmpeq_epi32, opCodl(and_si128, hSignMsk, N), opCodl(setzero_si128));
	n2  = opCodl(and_si128, N, h32Mask);
#endif

	n1  = opCodl(sub_epi32, N, n2);
	nf  = opCode(cvtepi32_pd,  N);
	R1  = opCode(sub_pd, x, opCode(mul_pd, nf, vL1d));
	R2  = opCode(mul_pd, nf, vL2d);
	N   = opCodl(srli_epi32, n1, 5);

#if	defined(__AVX512F__)
	M   = opCodl(blendv_epi8, opCodl(or_si256, N, hFillMsk), N, mMsk);
#else
	M   = opCodl(blendv_epi8, opCodl(or_si128, N, hFillMsk), N, mMsk);
#endif
	R   = opCode(add_pd, R1, R2);
	Q   = opCode(mul_pd,
		opCode(mul_pd, R, R),
		opCode(add_pd, vA1d, opCode(mul_pd, R, 
			opCode(add_pd, vA2d, opCode(mul_pd, R,
				opCode(add_pd, vA3d, opCode(mul_pd, R,
					opCode(add_pd, vA4d, opCode(mul_pd, R, vA5d)))))))));
	Q   = opCode(add_pd, R1, opCode(add_pd, R2, Q));
//#if	defined(__AVX512F__)
//	int vals[8];
//	opCodl(store_si256, static_cast<__m256i*>(static_cast<void*>(vals)), n2);
//	Sl  = opCode(set_pd, sLead_d[vals[7]], sLead_d[vals[6]], sLead_d[vals[5]], sLead_d[vals[4]], sLead_d[vals[3]], sLead_d[vals[2]], sLead_d[vals[1]], sLead_d[vals[0]]);
//	St  = opCode(set_pd, sTrail_d[vals[7]], sTrail_d[vals[6]], sTrail_d[vals[5]], sTrail_d[vals[4]], sTrail_d[vals[3]], sTrail_d[vals[2]], sTrail_d[vals[1]], sTrail_d[vals[0]]);
//#else
//	int vals[4];
//	opCodl(store_si128, static_cast<__m128i*>(static_cast<void*>(vals)), n2);
//	Sl  = opCode(set_pd, sLead_d[vals[3]], sLead_d[vals[2]], sLead_d[vals[1]], sLead_d[vals[0]]);
//	St  = opCode(set_pd, sTrail_d[vals[3]], sTrail_d[vals[2]], sTrail_d[vals[1]], sTrail_d[vals[0]]);
//#endif
#if	defined(__AVX512F__)
	Sl  = opCode(i32gather_pd, n2, sLead_d.data(),  8);
	St  = opCode(i32gather_pd, n2, sTrail_d.data(), 8);
#elif	defined(__AVX2__)
	Sl  = opCode(i32gather_pd, sLead_d.data(),  n2, 8);
	St  = opCode(i32gather_pd, sTrail_d.data(), n2, 8);
#else
	int vals[4];
	opCodl(store_si128, static_cast<__m128i*>(static_cast<void*>(vals)), n2);
	Sl  = opCode(set_pd, sLead_d[vals[3]], sLead_d[vals[2]], sLead_d[vals[1]], sLead_d[vals[0]]);
	St  = opCode(set_pd, sTrail_d[vals[3]], sTrail_d[vals[2]], sTrail_d[vals[1]], sTrail_d[vals[0]]);
#endif
	S   = opCode(add_pd, Sl, St);

#if	defined(__AVX512F__)
	R2 = opCode(castsi512_pd, opCode(shuffle_epi32, opCode(cvtepi32_epi64,
					opCodl(slli_epi32, opCodl(and_si256, opCodl(add_epi16, M, opCodl(set1_epi32, 1023)), opCodl(set1_epi32, 4095)), 20)),
					_MM_PERM_CDAB));
#elif	defined(__AVX2__)
	R2 = opCode(castsi256_pd, opCode(shuffle_epi32, opCode(cvtepi32_epi64,
					opCodl(slli_epi32, opCodl(and_si128, opCodl(add_epi16, M, opCodl(set1_epi32, 1023)), opCodl(set1_epi32, 4095)), 20)),
					0b10110001));
#else
	M  = opCodl(slli_epi32, opCodl(and_si128, opCodl(add_epi16, M, opCodl(set1_epi32, 1023)), opCodl(set1_epi32, 4095)), 20);
	xM = opCode(insertf128_si256, opCode(castsi128_si256, opCodl(shuffle_epi32, M, 0b01110010)), opCodl(shuffle_epi32, M, 0b11011000), 0b01); 
	R2 = opCode(castps_pd, opCode(blend_ps, opCode(castsi256_ps, xM), opCode(setzero_ps), 0b01010101));
#endif
	R1 = opCode(add_pd, Sl, opCode(add_pd, St, opCode(mul_pd, S, Q)));
	//return	opCode(mul_pd, R2, R1);
	return	opCode(and_pd, opCode(mul_pd, R2, R1), opCode(cmp_pd, x, opCode(set1_pd, -708.5), _CMP_NLT_UQ));

}

inline _MData_	opCode(log_pd, _MData_ x) {

	/*	1. Filtra negativos	*
	 *	2. Filtra Nan		*/
	_MData_ F, f, fc, i2, Y, Cl, Ct, m;
	_MHnt_	j;

#if	defined(__AVX512F__)
	_MInt_	mi;

	auto msk = opCode(cmp_pd_mask, opCode(castsi512_pd, opCode(and_si512, opCode(castpd_si512, x), opCode(castpd_si512, dInf))), opCode(setzero_pd), _CMP_EQ_UQ);
	f  = opCode(mask_blend_pd, msk, x, opCode(mul_pd, x, nmDnd));
	fc = opCode(castsi512_pd, opCode(and_si512, opCode(castpd_si512, f), opCode(castpd_si512, dInf)));
	mi = opCode(sub_epi64, opCode(srli_epi64, opCode(castpd_si512, fc), 52), opCode(set1_epi64, 1023));
	i2 = opCode(castsi512_pd, opCode(slli_epi64, opCode(sub_epi64, opCode(set1_epi64, 1023), mi), 52));
	m  = opCode(cvtepi32_pd, opCode(cvtepi64_epi32, mi));
#else
	_MHnt_	mi, mh, ml;
	auto msk = opCode(cmp_pd, opCode(and_pd, x, dInf), opCode(setzero_pd), _CMP_EQ_UQ);
	f  = opCode(blendv_pd, x, opCode(mul_pd, x, nmDnd), msk);
	fc = opCode(and_pd, f, dInf);
	ml = opCodl(sub_epi32, opCodl(srli_epi32, opCode(extractf128_si256, opCode(castpd_si256, fc), 0b00), 20), opCodl(set1_epi32, 1023));
	mh = opCodl(sub_epi32, opCodl(srli_epi32, opCode(extractf128_si256, opCode(castpd_si256, fc), 0b01), 20), opCodl(set1_epi32, 1023));
	mi = opCodl(castps_si128, opCodl(shuffle_ps, opCodl(castsi128_ps, ml), opCodl(castsi128_ps, mh), 0b11011101));
	mh = opCodl(slli_epi32, opCodl(sub_epi32, opCodl(set1_epi32, 1023), mi), 20);
	i2 = opCode(and_pd, opCode(castsi256_pd, opCode(insertf128_si256, opCode(castsi128_si256,
						opCodl(shuffle_epi32, mh, 0b01010000)), opCodl(shuffle_epi32, mh, 0b11111010), 0b01)), dInf);
	m  = opCode(cvtepi32_pd, mi);
#endif
	Y  = opCode(mul_pd, i2, f);

#if	defined(__AVX512F__)
	i2 = opCode(roundscale_round_pd, opCode(mul_pd, Y, opCode(set1_pd, 128.)), 0x04, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
	i2 = opCode(round_pd, opCode(mul_pd, Y, opCode(set1_pd, 128.f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#endif
	j  = opCodl(sub_epi32, opCode(cvtpd_epi32, i2), opCodl(set1_epi32, 128));

	F  = opCode(mul_pd, vi2t7d, i2);
	f  = opCode(sub_pd, Y, F);

//#if	defined(__AVX512F__)
//	int vals[8];
//	opCodl(store_si256, static_cast<__m256i*>(static_cast<void*>(vals)), j);
//	fc  = opCode(set_pd, cLead_d[vals[7]],  cLead_d[vals[6]],  cLead_d[vals[5]],  cLead_d[vals[4]],  cLead_d[vals[3]],  cLead_d[vals[2]],  cLead_d[vals[1]],  cLead_d[vals[0]]);
//	i2  = opCode(set_pd, cTrail_d[vals[7]], cTrail_d[vals[6]], cTrail_d[vals[5]], cTrail_d[vals[4]], cTrail_d[vals[3]], cTrail_d[vals[2]], cTrail_d[vals[1]], cTrail_d[vals[0]]);
//#else
//	int vals[4];
//	opCodl(store_si128, static_cast<__m128i*>(static_cast<void*>(vals)), j);
//	fc  = opCode(set_pd, cLead_d[vals[3]],  cLead_d[vals[2]],  cLead_d[vals[1]],  cLead_d[vals[0]]);
//	i2  = opCode(set_pd, cTrail_d[vals[3]], cTrail_d[vals[2]], cTrail_d[vals[1]], cTrail_d[vals[0]]);
//#endif
#if	defined(__AVX512F__)
	fc  = opCode(i32gather_pd, j, cLead_d.data(),  8);
	i2  = opCode(i32gather_pd, j, cTrail_d.data(), 8);
#elif	defined(__AVX2__)
	fc  = opCode(i32gather_pd, cLead_d.data(),  j, 8);
	i2  = opCode(i32gather_pd, cTrail_d.data(), j, 8);
#else
	int vals[4];
	opCodl(store_si128, static_cast<__m128i*>(static_cast<void*>(vals)), j);
	fc  = opCode(set_pd, cLead_d[vals[3]],  cLead_d[vals[2]],  cLead_d[vals[1]],  cLead_d[vals[0]]);
	i2  = opCode(set_pd, cTrail_d[vals[3]], cTrail_d[vals[2]], cTrail_d[vals[1]], cTrail_d[vals[0]]);
#endif

	Cl  = opCode(add_pd, opCode(mul_pd, m, vLg2ld), fc);
	Ct  = opCode(add_pd, opCode(mul_pd, m, vLg2td), i2);

	fc  = opCode(div_pd, f, F);	// Ver si podemos quitarnos esto de encima
	Y   = opCode(mul_pd, fc, 
		opCode(mul_pd, fc,
			opCode(add_pd, vB1d, opCode(mul_pd, fc,
				opCode(add_pd, vB2d, opCode(mul_pd, fc,
					opCode(add_pd, vB3d, opCode(mul_pd, fc,
						opCode(add_pd, vB4d, opCode(mul_pd, fc, vB5d))))))))));

#if	defined(__AVX512F__)
	return	opCode(add_pd, opCode(sub_pd, Cl, opCode(mask_blend_pd, msk, opCode(setzero_pd), lgDld)),
			opCode(add_pd, fc, opCode(add_pd, Y, opCode(sub_pd, Ct, opCode(mask_blend_pd, msk, opCode(setzero_pd), lgDtd)))));
#else
	return	opCode(add_pd, opCode(sub_pd, Cl, opCode(blendv_pd, opCode(setzero_pd), lgDld, msk)),
			opCode(add_pd, fc, opCode(add_pd, Y, opCode(sub_pd, Ct, opCode(blendv_pd, opCode(setzero_pd), lgDtd, msk)))));
#endif
}

#endif

#undef	_MData_

#if	defined(__AVX512F__)
	#define	_MData_ __m512
#elif	defined(__AVX__)
	#define	_MData_ __m256
#else
	#define	_MData_ __m128
#endif 

#ifndef	__INTEL_COMPILER

inline _MData_	opCode(exp_ps, _MData_ x) {
	_MInt_	N, M, n1, n2;
#if	defined(__AVX512F__)
	_MHnt_	Mh, Ml;
#else
	_MHnt_	n1h, n1l, Mh, Ml;
#endif
	_MData_ n1f, n2f, R1, R2, R, Q, Sl, St, S;

/*
	1.  Filtra Nan
	2.  Filtra +inf -> +inf
	3.  Filtra -inf -> 0
        4.  Filtra Threshold_1 -> +inf / 0 --> En realidad es -175.0 por abajo, no lo que pone ahí, por el exponente M
	5.  Filtra Threshold_2 -> 1+x	<-- No hace falta, porque vamos a calcular la chunga igual...
*/
#if	defined(__AVX512F__)
	N   = opCode(cvtps_epi32, opCode(roundscale_round_ps, opCode(mul_ps, x, vIvLf), 0x04, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
	auto mMsk = opCode(cmp_epi32_mask, opCode(and_si512, iSignMsk, N), opCode(setzero_si512), _MM_CMPINT_EQ);
	n2  = opCode(and_si512, N, m32Mask);
#else
	N   = opCode(cvtps_epi32, opCode(round_ps, opCode(mul_ps, x, vIvLf), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
	auto mMsk = opCode(cmp_ps, opCode(and_ps, opCode(castsi256_ps, iSignMsk), opCode(castsi256_ps, N)), opCode(setzero_ps), _CMP_EQ_UQ);
	n2  = opCode(castps_si256, opCode(and_ps, opCode(castsi256_ps, N), opCode(castsi256_ps, m32Mask)));
#endif

#if	defined(__AVX512F__) || defined(__AVX2__)
	n1  = opCode(sub_epi32, N, n2);
	n1f = opCode(cvtepi32_ps,  n1);
#elif	defined(__AVX__)
	Ml  = opCode(extractf128_si256, N,  0b0000);
	Mh  = opCode(extractf128_si256, N,  0b0001);
	n1l = opCode(extractf128_si256, n2, 0b0000);
	n1h = opCode(extractf128_si256, n2, 0b0001);
	n1l = opCodl(sub_epi32, Ml, n1l);
	n1h = opCodl(sub_epi32, Mh, n1h);
	n1f = opCode(cvtepi32_ps, opCode(insertf128_si256, opCode(insertf128_si256, opCode(setzero_si256), n1h, 0b0001), n1l, 0b0000));
#endif
        n2f = opCode(cvtepi32_ps, n2);
	R1  = opCode(sub_ps,
		opCode(sub_ps, x, opCode(mul_ps, n1f, vL1f)),
		opCode(mul_ps, n2f, vL1f));
	R2  = opCode(mul_ps, opCode(cvtepi32_ps, N), vL2f);	// - Sign in vL2f
#if	defined(__AVX512F__) || defined(__AVX2__)
	N = opCode(srli_epi32, n1, 5);
#elif	defined(__AVX__)
	N = opCode(insertf128_si256, opCode(insertf128_si256, opCode(setzero_si256), opCodl(srli_epi32, n1h, 5), 0b0001), opCodl(srli_epi32, n1l, 5), 0b0000);
#endif

#if	defined(__AVX512F__)
	M   = opCode(mask_blend_epi32, mMsk, opCode(or_si512, N, iFillMsk), N);
#else
	M   = opCode(castps_si256,
		opCode(blendv_ps,
			opCode(or_ps,
				opCode(castsi256_ps, N),
				opCode(castsi256_ps, iFillMsk)),
			opCode(castsi256_ps, N),
			mMsk));
#endif
	R   = opCode(add_ps, R1, R2);
	Q   = opCode(mul_ps,
		opCode(mul_ps, R, R),
		opCode(add_ps, vA1f, opCode(mul_ps, R, vA2f)));
	Q   = opCode(add_ps, R1, opCode(add_ps, R2, Q));
//#if	defined(__AVX512F__)
//	int vals[16];
//	opCode(store_si512, static_cast<__m512i*>(static_cast<void*>(vals)), n2);
//	Sl  = opCode(set_ps, sLead_f[vals[15]], sLead_f[vals[14]], sLead_f[vals[13]], sLead_f[vals[12]], sLead_f[vals[11]], sLead_f[vals[10]], sLead_f[vals[ 9]], sLead_f[vals[8]],
//			     sLead_f[vals[ 7]], sLead_f[vals[ 6]], sLead_f[vals[ 5]], sLead_f[vals[ 4]], sLead_f[vals[ 3]], sLead_f[vals[ 2]], sLead_f[vals[ 1]], sLead_f[vals[0]]);
//	St  = opCode(set_ps, sTrail_f[vals[15]], sTrail_f[vals[14]], sTrail_f[vals[13]], sTrail_f[vals[12]], sTrail_f[vals[11]], sTrail_f[vals[10]], sTrail_f[vals[ 9]], sTrail_f[vals[ 8]],
//			     sTrail_f[vals[ 7]], sTrail_f[vals[ 6]], sTrail_f[vals[ 5]], sTrail_f[vals[ 4]], sTrail_f[vals[ 3]], sTrail_f[vals[ 2]], sTrail_f[vals[ 1]], sTrail_f[vals[ 0]]);
//#else
//	int vals[8];
//	opCode(store_si256, static_cast<__m256i*>(static_cast<void*>(vals)), n2);
//	Sl  = opCode(set_ps, sLead_f[vals[7]], sLead_f[vals[6]], sLead_f[vals[5]], sLead_f[vals[4]], sLead_f[vals[3]], sLead_f[vals[2]], sLead_f[vals[1]], sLead_f[vals[0]]);
//	St  = opCode(set_ps, sTrail_f[vals[7]], sTrail_f[vals[6]], sTrail_f[vals[5]], sTrail_f[vals[4]], sTrail_f[vals[3]], sTrail_f[vals[2]], sTrail_f[vals[1]], sTrail_f[vals[0]]);
//#endif
#if	defined(__AVX512F__)
	Sl  = opCode(i32gather_ps, n2, sLead_f.data(),  4);
	St  = opCode(i32gather_ps, n2, sTrail_f.data(), 4);
#elif	defined(__AVX2__)
	Sl  = opCode(i32gather_ps, sLead_f.data(),  n2, 4);
	St  = opCode(i32gather_ps, sTrail_f.data(), n2, 4);
#else
	int vals[8];
	opCode(store_si256, static_cast<__m256i*>(static_cast<void*>(vals)), n2);
	Sl  = opCode(set_ps, sLead_f[vals[7]], sLead_f[vals[6]], sLead_f[vals[5]], sLead_f[vals[4]], sLead_f[vals[3]], sLead_f[vals[2]], sLead_f[vals[1]], sLead_f[vals[0]]);
	St  = opCode(set_ps, sTrail_f[vals[7]], sTrail_f[vals[6]], sTrail_f[vals[5]], sTrail_f[vals[4]], sTrail_f[vals[3]], sTrail_f[vals[2]], sTrail_f[vals[1]], sTrail_f[vals[0]]);
#endif
	S   = opCode(add_ps, Sl, St);
#if	defined(__AVX512F__)
	Ml = opCode(extracti64x4_epi64, M,  0b00);
	Mh = opCode(extracti64x4_epi64, M,  0b01);

	auto msk = opCode(cmp_epi32_mask, opCode(set1_epi32, -126), M, _MM_CMPINT_NLE);

	M  = opCode(mask_add_epi32, M, msk, M, opCode(set1_epi32, 126));
	R2 = opCode(castsi512_ps, opCode(slli_epi32, opCode(and_si512, opCode(set1_epi32, 255),
			opCode(inserti64x4, opCode(inserti64x4, opCode(setzero_si512),
				opCodl(add_epi8, Mh, opCodl(set1_epi32, 127)), 0b01),
				opCodl(add_epi8, Ml, opCodl(set1_epi32, 127)), 0b00)), 23));
#elif	defined(__AVX2__)
	N  = opCode(cmpgt_epi32, opCode(set1_epi32, -126), M);
	M  = opCode(add_epi32, opCode(and_si256, N, opCode(set1_epi32, 126)), M);
	R2 = opCode(castsi256_ps, opCode(slli_epi32, opCode(and_si256, opCode(add_epi8, M, opCode(set1_epi32, 127)), opCode(set1_epi32, 255)), 23));
#elif	defined(__AVX__)
	Ml = opCode(extractf128_si256, M,  0b0000);
	Mh = opCode(extractf128_si256, M,  0b0001);

	n1l = opCodl(cmpgt_epi32, opCodl(set1_epi32, -126), Ml); 
	n1h = opCodl(cmpgt_epi32, opCodl(set1_epi32, -126), Mh);
	N   = opCode(insertf128_si256, opCode(insertf128_si256, opCode(setzero_si256), n1l, 0b00), n1h, 0b01);	

	Ml  = opCodl(add_epi32, opCodl(and_si128, n1l, opCodl(set1_epi32, 126)), Ml);
	Mh  = opCodl(add_epi32, opCodl(and_si128, n1h, opCodl(set1_epi32, 126)), Mh);

	R2 = opCode(castsi256_ps, opCode(insertf128_si256, opCode(insertf128_si256, opCode(setzero_si256),
			opCodl(slli_epi32, opCodl(and_si128, opCodl(add_epi8, Mh, opCodl(set1_epi32, 127)), opCodl(set1_epi32, 255)), 23), 0b0001),
			opCodl(slli_epi32, opCodl(and_si128, opCodl(add_epi8, Ml, opCodl(set1_epi32, 127)), opCodl(set1_epi32, 255)), 23), 0b0000));
#endif
	R1 = opCode(add_ps, Sl, opCode(add_ps, St, opCode(mul_ps, S, Q)));
#if	defined(__AVX512F__)
	R  = opCode(mask_blend_ps, msk, opCode(set1_ps, 1.0f), vExpf);
	return	opCode(mask_blend_ps, opCode(cmp_ps_mask, x, opCode(set1_ps, -175.0f), _CMP_GE_OQ),
			opCode(mul_ps, R, opCode(mul_ps, R2, R1)),
			opCode(setzero_ps));
#else
	R  = opCode(blendv_ps, opCode(set1_ps, 1.0f), vExpf, opCode(castsi256_ps, N));
	return	opCode(and_ps, opCode(mul_ps, R, opCode(mul_ps, R2, R1)), opCode(cmp_ps, x, opCode(set1_ps, -175.0f), _CMP_NLT_UQ));
#endif
}

inline _MData_	opCode(log_ps, _MData_ x) {

	/*	1. Filtra negativos	*
	 *	2. Filtra Nan		*/
	_MData_ F, f, fc, i2, Y, Cl, Ct, m;
	_MInt_	j, mi;
#if	!defined(__AVX512F__) && !defined(__AVX2__)
	_MHnt_	mh, ml;
#endif

#if	defined(__AVX512F__)
	auto msk = opCode(cmp_ps_mask, opCode(castsi512_ps, opCode(and_si512, opCode(castps_si512, x), opCode(castps_si512, fInf))), opCode(setzero_ps), _CMP_EQ_UQ);
	f  = opCode(mask_blend_ps, msk, x, opCode(mul_ps, x, nmDnf));
	fc = opCode(castsi512_ps, opCode(and_si512, opCode(castps_si512, f), opCode(castps_si512, fInf)));
	mi = opCode(sub_epi32, opCode(srli_epi32, opCode(castps_si512, fc), 23), opCode(set1_epi32, 127));
	i2 = opCode(castsi512_ps, opCode(slli_epi32, opCode(sub_epi32, opCode(set1_epi32, 127), mi), 23));
	m  = opCode(cvtepi32_ps, mi);
#else
	auto msk = opCode(cmp_ps, opCode(and_ps, x, fInf), opCode(setzero_ps), _CMP_EQ_UQ);
	f  = opCode(blendv_ps, x, opCode(mul_ps, x, nmDnf), msk);
	fc = opCode(and_ps, f, fInf);
	#if	defined(__AVX2__)
	mi = opCode(sub_epi32, opCode(srli_epi32, opCode(castps_si256, fc), 23), opCode(set1_epi32, 127));
	i2 = opCode(castsi256_ps, opCode(slli_epi32, opCode(sub_epi32, opCode(set1_epi32, 127), mi), 23));
	m  = opCode(cvtepi32_ps, mi);
	#else
	ml = opCodl(sub_epi32, opCodl(srli_epi32, opCode(extractf128_si256, opCode(castps_si256, fc), 0b00), 23), opCodl(set1_epi32, 127));
	mh = opCodl(sub_epi32, opCodl(srli_epi32, opCode(extractf128_si256, opCode(castps_si256, fc), 0b01), 23), opCodl(set1_epi32, 127));
	i2 = opCode(castsi256_ps, opCode(insertf128_si256, opCode(castsi128_si256,
					opCodl(slli_epi32, opCodl(sub_epi32, opCodl(set1_epi32, 127), ml), 23)),
					opCodl(slli_epi32, opCodl(sub_epi32, opCodl(set1_epi32, 127), mh), 23), 0b01));
	m  = opCode(cvtepi32_ps, opCode(insertf128_si256, opCode(castsi128_si256, ml), mh, 0b01));
	#endif
#endif
	Y  = opCode(mul_ps, i2, f);

#if	defined(__AVX512F__)
	i2 = opCode(roundscale_round_ps, opCode(mul_ps, Y, opCode(set1_ps, 128.f)), 0x04, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	j  = opCode(sub_epi32, opCode(cvtps_epi32, i2), opCode(set1_epi32, 128));
#elif	defined(__AVX2__)
	i2 = opCode(round_ps, opCode(mul_ps, Y, opCode(set1_ps, 128.f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	j  = opCode(sub_epi32, opCode(cvtps_epi32, i2), opCode(set1_epi32, 128));
#else
	i2 = opCode(round_ps, opCode(mul_ps, Y, opCode(set1_ps, 128.f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	mi = opCode(cvtps_epi32, i2);
	ml = opCode(extractf128_si256, mi, 0b00);
	mh = opCode(extractf128_si256, mi, 0b01);
	j  = opCode(insertf128_si256, opCode(castsi128_si256, opCodl(sub_epi32, ml, opCodl(set1_epi32, 128))), opCodl(sub_epi32, mh, opCodl(set1_epi32, 128)), 0b01);
#endif

	F  = opCode(mul_ps, vi2t7f, i2);
	f  = opCode(sub_ps, Y, F);

#if	defined(__AVX512F__)
	fc  = opCode(i32gather_ps, j, cLead_f.data(),  4);
	i2  = opCode(i32gather_ps, j, cTrail_f.data(), 4);
#elif	defined(__AVX2__)
	fc  = opCode(i32gather_ps, cLead_f.data(),  j, 4);
	i2  = opCode(i32gather_ps, cTrail_f.data(), j, 4);
#else
	int vals[8];
	opCode(store_si256, static_cast<__m256i*>(static_cast<void*>(vals)), j);
	fc  = opCode(set_ps, cLead_f[vals[7]],  cLead_f[vals[6]],  cLead_f[vals[5]],  cLead_f[vals[4]],  cLead_f[vals[3]],  cLead_f[vals[2]],  cLead_f[vals[1]],  cLead_f[vals[0]]);
	i2  = opCode(set_ps, cTrail_f[vals[7]], cTrail_f[vals[6]], cTrail_f[vals[5]], cTrail_f[vals[4]], cTrail_f[vals[3]], cTrail_f[vals[2]], cTrail_f[vals[1]], cTrail_f[vals[0]]);
#endif
	Cl  = opCode(add_ps, opCode(mul_ps, m, vLg2lf), fc);
	Ct  = opCode(add_ps, opCode(mul_ps, m, vLg2tf), i2);

	fc  = opCode(div_ps, f, F);	// Ver si podemos quitarnos esto de encima
	Y   = opCode(mul_ps, fc, 
		opCode(mul_ps, fc,
			opCode(add_ps, vB1f, opCode(mul_ps, fc, vB2f))));

#if	defined(__AVX512F__)
	return	opCode(add_ps, opCode(sub_ps, Cl, opCode(mask_blend_ps, msk, opCode(setzero_ps), lgDlf)),
			opCode(add_ps, fc, opCode(add_ps, Y, opCode(sub_ps, Ct, opCode(mask_blend_ps, msk, opCode(setzero_ps), lgDtf)))));
#else
	return	opCode(add_ps, opCode(sub_ps, Cl, opCode(blendv_ps, opCode(setzero_ps), lgDlf, msk)),
			opCode(add_ps, fc, opCode(add_ps, Y, opCode(sub_ps, Ct, opCode(blendv_ps, opCode(setzero_ps), lgDtf, msk)))));
#endif
}

#endif

#undef	_MData_
#undef	_PREFIX
#undef opCode_P
#undef opCode_N
#undef opCode

#endif
