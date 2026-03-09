#ifndef	_INTRINSICS_
#define	_INTRINSICS_

#include<cmath>

#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

#include <immintrin.h>


// --- ISA-dependent vector types ---
#ifdef __AVX512F__
    #define _MData_ __m512d
    #define _MInt_  __m512i
    #define _MHnt_  __m256i
#elif defined(__AVX__)
    #define _MData_ __m256d
    #define _MInt_  __m256i
    #define _MHnt_  __m128i
#else
    #define _MData_ __m128d
    #define _MInt_  __m128i
#endif

// --- Opcode prefix selection + "low" prefix helper ---
#ifdef __AVX512F__
    #define _PREFIX_  _mm512
    #define _PREFXL_  _mm256
    #define opCodl(x, ...) opCode_N(_PREFXL_, x, __VA_ARGS__)

    /* AVX-512 does NOT provide _mm512_rsqrt_ps/_mm512_rcp_ps.
       Map the generic names used by opCode(...) to the correct AVX-512 intrinsics. */
    #ifndef _mm512_rsqrt_ps
      #if defined(__AVX512ER__)
        #define _mm512_rsqrt_ps _mm512_rsqrt28_ps   /* Knights/ER: higher-precision seed */
      #else
        #define _mm512_rsqrt_ps _mm512_rsqrt14_ps   /* Baseline AVX-512F */
      #endif
    #endif

    #ifndef _mm512_rcp_ps
      #define _mm512_rcp_ps _mm512_rcp14_ps         /* Baseline AVX-512F reciprocal */
    #endif

#else
    #if !defined(__AVX__) && !defined(__AVX2__)
        #define _PREFIX_  _mm
    #else
        #define _PREFIX_  _mm256
        #define _PREFXL_  _mm
        #define opCodl(x, ...) opCode_N(_PREFXL_, x, __VA_ARGS__)
    #endif
#endif

#define	M_PI2	(M_PI *M_PI)
#define	M_PI4	(M_PI2*M_PI2)
#define	M_PI6	(M_PI4*M_PI2)

#ifndef	__INTEL_COMPILER

constexpr double Inf_d = __builtin_inf();
constexpr double Nan_d = __builtin_nan("");//0xFFFFF");

constexpr double PiA_d = -3.1415926218032836914;
constexpr double PiB_d = -3.1786509424591713469e-08;
constexpr double PiC_d = -1.2246467864107188502e-16;
constexpr double PiD_d = -1.2736634327021899816e-24;

constexpr double s0_d  = -7.97255955009037868891952e-18;
constexpr double s1_d  =  2.81009972710863200091251e-15;
constexpr double s2_d  = -7.64712219118158833288484e-13;
constexpr double s3_d  =  1.60590430605664501629054e-10;
constexpr double s4_d  = -2.50521083763502045810755e-08;
constexpr double s5_d  =  2.75573192239198747630416e-06;
constexpr double s6_d  = -0.000198412698412696162806809;
constexpr double s7_d  =  0.00833333333333332974823815;
constexpr double s8_d  = -0.166666666666666657414808;
#ifdef	__AVX512F__
constexpr _MData_ rPid	    = { M_1_PI/(1<<24), M_1_PI/(1<<24), M_1_PI/(1<<24), M_1_PI/(1<<24), M_1_PI/(1<<24), M_1_PI/(1<<24), M_1_PI/(1<<24), M_1_PI/(1<<24) };
constexpr _MData_ dPid	    = { M_1_PI/(1<<23), M_1_PI/(1<<23), M_1_PI/(1<<23), M_1_PI/(1<<23), M_1_PI/(1<<23), M_1_PI/(1<<23), M_1_PI/(1<<23), M_1_PI/(1<<23) };
constexpr _MData_ rCte      = {   16777216.,   16777216.,   16777216.,   16777216.,   16777216.,   16777216.,   16777216.,   16777216. };
constexpr _MData_ oPid      = {     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI };
constexpr _MData_ zeroNegd  = {        -0.0,        -0.0,        -0.0,        -0.0,        -0.0,        -0.0,        -0.0,        -0.0 };
constexpr _MData_ dHlf      = {         0.5,         0.5,         0.5,         0.5,         0.5,         0.5,         0.5,         0.5 };
constexpr _MData_ dOne      = {         1.0,         1.0,         1.0,         1.0,         1.0,         1.0,         1.0,         1.0 };
constexpr _MData_ TriMaxd   = {        1e15,        1e15,        1e15,        1e15,        1e15,        1e15,        1e15,        1e15 };
constexpr _MData_ dInf      = {       Inf_d,       Inf_d,       Inf_d,       Inf_d,       Inf_d,       Inf_d,       Inf_d,       Inf_d };
constexpr _MData_ dNan      = {       Nan_d,       Nan_d,       Nan_d,       Nan_d,       Nan_d,       Nan_d,       Nan_d,       Nan_d };
constexpr _MData_ PiAd      = {       PiA_d,       PiA_d,       PiA_d,       PiA_d,       PiA_d,       PiA_d,       PiA_d,       PiA_d };
constexpr _MData_ PiBd      = {       PiB_d,       PiB_d,       PiB_d,       PiB_d,       PiB_d,       PiB_d,       PiB_d,       PiB_d };
constexpr _MData_ PiCd      = {       PiC_d,       PiC_d,       PiC_d,       PiC_d,       PiC_d,       PiC_d,       PiC_d,       PiC_d };
constexpr _MData_ PiDd      = {       PiD_d,       PiD_d,       PiD_d,       PiD_d,       PiD_d,       PiD_d,       PiD_d,       PiD_d };
constexpr _MData_ hPiAd     = {   0.5*PiA_d,   0.5*PiA_d,   0.5*PiA_d,   0.5*PiA_d,   0.5*PiA_d,   0.5*PiA_d,   0.5*PiA_d,   0.5*PiA_d };
constexpr _MData_ hPiBd     = {   0.5*PiB_d,   0.5*PiB_d,   0.5*PiB_d,   0.5*PiB_d,   0.5*PiB_d,   0.5*PiB_d,   0.5*PiB_d,   0.5*PiB_d };
constexpr _MData_ hPiCd     = {   0.5*PiC_d,   0.5*PiC_d,   0.5*PiC_d,   0.5*PiC_d,   0.5*PiC_d,   0.5*PiC_d,   0.5*PiC_d,   0.5*PiC_d };
constexpr _MData_ hPiDd     = {   0.5*PiD_d,   0.5*PiD_d,   0.5*PiD_d,   0.5*PiD_d,   0.5*PiD_d,   0.5*PiD_d,   0.5*PiD_d,   0.5*PiD_d };
constexpr _MData_ s0d       = {        s0_d,        s0_d,        s0_d,        s0_d,        s0_d,        s0_d,        s0_d,        s0_d };
constexpr _MData_ s1d       = {        s1_d,        s1_d,        s1_d,        s1_d,        s1_d,        s1_d,        s1_d,        s1_d };
constexpr _MData_ s2d       = {        s2_d,        s2_d,        s2_d,        s2_d,        s2_d,        s2_d,        s2_d,        s2_d };
constexpr _MData_ s3d       = {        s3_d,        s3_d,        s3_d,        s3_d,        s3_d,        s3_d,        s3_d,        s3_d };
constexpr _MData_ s4d       = {        s4_d,        s4_d,        s4_d,        s4_d,        s4_d,        s4_d,        s4_d,        s4_d };
constexpr _MData_ s5d       = {        s5_d,        s5_d,        s5_d,        s5_d,        s5_d,        s5_d,        s5_d,        s5_d };
constexpr _MData_ s6d       = {        s6_d,        s6_d,        s6_d,        s6_d,        s6_d,        s6_d,        s6_d,        s6_d };
constexpr _MData_ s7d       = {        s7_d,        s7_d,        s7_d,        s7_d,        s7_d,        s7_d,        s7_d,        s7_d };
constexpr _MData_ s8d       = {        s8_d,        s8_d,        s8_d,        s8_d,        s8_d,        s8_d,        s8_d,        s8_d };
constexpr _MInt_  iZero     = {           0,           0,           0,           0,           0,           0,           0,           0 };
constexpr _MInt_  one       = {  4294967297,  4294967297,  4294967297,  4294967297,  4294967297,  4294967297,  4294967297,  4294967297 };
constexpr _MInt_  two       = {  8589934594,  8589934594,  8589934594,  8589934594,  8589934594,  8589934594,  8589934594,  8589934594 };
constexpr _MHnt_  hOne      = {  4294967297,  4294967297,  4294967297,  4294967297 };
constexpr _MHnt_  hTwo      = {  8589934594,  8589934594,  8589934594,  8589934594 };
constexpr _MHnt_  iZerh     = {           0,           0,           0,           0 };
constexpr _MData_ cMulSgn   = {          1.,         -1.,          1.,          -1.,         1.,          -1.,         1.,          -1.};

#elif   defined(__AVX__)
constexpr _MData_ rPid	    = { M_1_PI/(1<<24), M_1_PI/(1<<24), M_1_PI/(1<<24), M_1_PI/(1<<24) };
constexpr _MData_ dPid	    = { M_1_PI/(1<<23), M_1_PI/(1<<23), M_1_PI/(1<<23), M_1_PI/(1<<23) };
constexpr _MData_ rCte      = {     (1<<24),     (1<<24),     (1<<24),     (1<<24) };
constexpr _MData_ dCte      = {     (1<<23),     (1<<23),     (1<<23),     (1<<23) };
constexpr _MData_ oPid      = {      M_1_PI,      M_1_PI,      M_1_PI,      M_1_PI };
constexpr _MData_ zeroNegd  = {        -0.0,        -0.0,        -0.0,        -0.0 };
constexpr _MData_ dHlf      = {         0.5,         0.5,         0.5,         0.5 };
constexpr _MData_ dOne      = {         1.0,         1.0,         1.0,         1.0 };
#ifdef	__FMA__
constexpr _MData_ TriMaxd   = {        1e15,        1e15,        1e15,        1e15 };
#else
constexpr _MData_ TriMaxd   = {        1e12,        1e12,        1e12,        1e12 };
#endif
constexpr _MData_ dInf      = {       Inf_d,       Inf_d,       Inf_d,       Inf_d };
constexpr _MData_ dNan      = {       Nan_d,       Nan_d,       Nan_d,       Nan_d };
constexpr _MData_ PiAd      = {       PiA_d,       PiA_d,       PiA_d,       PiA_d };
constexpr _MData_ PiBd      = {       PiB_d,       PiB_d,       PiB_d,       PiB_d };
constexpr _MData_ PiCd      = {       PiC_d,       PiC_d,       PiC_d,       PiC_d };
constexpr _MData_ PiDd      = {       PiD_d,       PiD_d,       PiD_d,       PiD_d };
constexpr _MData_ hPiAd     = {   0.5*PiA_d,   0.5*PiA_d,   0.5*PiA_d,   0.5*PiA_d };
constexpr _MData_ hPiBd     = {   0.5*PiB_d,   0.5*PiB_d,   0.5*PiB_d,   0.5*PiB_d };
constexpr _MData_ hPiCd     = {   0.5*PiC_d,   0.5*PiC_d,   0.5*PiC_d,   0.5*PiC_d };
constexpr _MData_ hPiDd     = {   0.5*PiD_d,   0.5*PiD_d,   0.5*PiD_d,   0.5*PiD_d };
constexpr _MData_ s0d       = {        s0_d,        s0_d,        s0_d,        s0_d };
constexpr _MData_ s1d       = {        s1_d,        s1_d,        s1_d,        s1_d };
constexpr _MData_ s2d       = {        s2_d,        s2_d,        s2_d,        s2_d };
constexpr _MData_ s3d       = {        s3_d,        s3_d,        s3_d,        s3_d };
constexpr _MData_ s4d       = {        s4_d,        s4_d,        s4_d,        s4_d };
constexpr _MData_ s5d       = {        s5_d,        s5_d,        s5_d,        s5_d };
constexpr _MData_ s6d       = {        s6_d,        s6_d,        s6_d,        s6_d };
constexpr _MData_ s7d       = {        s7_d,        s7_d,        s7_d,        s7_d };
constexpr _MData_ s8d       = {        s8_d,        s8_d,        s8_d,        s8_d };
constexpr _MInt_  iZero     = {           0,           0,           0,           0 };
constexpr _MData_ cMulSgn   = {          1.,         -1.,          1.,          -1.};
constexpr _MInt_  one       = {  4294967297,  4294967297,  4294967297,  4294967297 };
constexpr _MInt_  two       = {  8589934594,  8589934594,  8589934594,  8589934594 };
constexpr _MHnt_  iZerh     = {           0,           0 };
constexpr _MHnt_  hOne      = {  4294967297,  4294967297 };
constexpr _MHnt_  hTwo      = {  8589934594,  8589934594 };
#else
constexpr _MData_ rPid	    = { M_1_PI/(1<<24), M_1_PI/(1<<24) };
constexpr _MData_ dPid	    = { M_1_PI/(1<<23), M_1_PI/(1<<23) };
constexpr _MData_ rCte      = {   (1 << 24),   (1 << 24) };
constexpr _MData_ dCte      = {   (1 << 23),   (1 << 23) };
constexpr _MData_ oPid      = {      M_1_PI,      M_1_PI };
constexpr _MData_ zeroNegd  = {        -0.0,        -0.0 };
constexpr _MData_ dHlf      = {         0.5,         0.5 };
constexpr _MData_ dOne      = {         1.0,         1.0 };
constexpr _MData_ TriMaxd   = {        1e15,        1e15 };
constexpr _MData_ dInf      = {       Inf_d,       Inf_d };
constexpr _MData_ dNan      = {       Nan_d,       Nan_d };
constexpr _MData_ PiAd      = {       PiA_d,       PiA_d };
constexpr _MData_ PiBd      = {       PiB_d,       PiB_d };
constexpr _MData_ PiCd      = {       PiC_d,       PiC_d };
constexpr _MData_ PiDd      = {       PiD_d,       PiD_d };
constexpr _MData_ hPiAd     = {   0.5*PiA_d,   0.5*PiA_d };
constexpr _MData_ hPiBd     = {   0.5*PiB_d,   0.5*PiB_d };
constexpr _MData_ hPiCd     = {   0.5*PiC_d,   0.5*PiC_d };
constexpr _MData_ hPiDd     = {   0.5*PiD_d,   0.5*PiD_d };
constexpr _MData_ s0d       = {        s0_d,        s0_d };
constexpr _MData_ s1d       = {        s1_d,        s1_d };
constexpr _MData_ s2d       = {        s2_d,        s2_d };
constexpr _MData_ s3d       = {        s3_d,        s3_d };
constexpr _MData_ s4d       = {        s4_d,        s4_d };
constexpr _MData_ s5d       = {        s5_d,        s5_d };
constexpr _MData_ s6d       = {        s6_d,        s6_d };
constexpr _MData_ s7d       = {        s7_d,        s7_d };
constexpr _MData_ s8d       = {        s8_d,        s8_d };
constexpr _MInt_  iZero     = {           0,           0 };
constexpr _MData_ cMulSgn   = {          1.,         -1. };
constexpr _MInt_  one       = {  4294967297,  4294967297 };
constexpr _MInt_  two       = {  8589934594,  8589934594 };
#endif

#ifdef	__AVX__
inline void printhVar(_MHnt_ d, const char *name)
#else
inline void printhVar(_MInt_ d, const char *name)
#endif
{
	printf ("%s", name);
#if	defined(__AVX512F__)
	int r[8] __attribute((aligned(32)));
	opCodl(store_si256, ((_MHnt_ *)r), d);
	for (int i=0; i<8; i++)
#elif	defined(__AVX__)
	int r[4] __attribute((aligned(16)));
	opCodl(store_si128, ((_MHnt_ *)r), d);
	for (int i=0; i<4; i++)
#else
	int r[4] __attribute((aligned(16)));
	opCode(store_si128, ((_MInt_ *)r), d);
	for (int i=0; i<4; i++)
#endif
		printf(" %d", r[i]);
	printf("\n");
}

inline void printdVar(_MData_ d, const char *name) {

	printf ("%s", name);
#if	defined(__AVX512F__)
	for (int i=0; i<8; i++)
#elif	defined(__AVX__)
	for (int i=0; i<4; i++)
#else
	for (int i=0; i<2; i++)
#endif
		printf(" %+lf", d[i]);
	printf("\n");
}

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

#endif

// from a, b, c, d, returns a if c>d and b if c<d
inline _MData_	opCode(kkk_pd, const _MData_ &a, const _MData_ &b, const _MData_ &c, const _MData_ &d)
{
_MData_	ret;

#if	defined(__AVX512F__)
	ret = opCode(mask_add_pd, opCode(setzero_pd), opCode(cmp_pd_mask, c, d, _CMP_GE_OQ), opCode(setzero_pd), a);
	ret = opCode(mask_add_pd, ret,                opCode(cmp_pd_mask, c, d, _CMP_LT_OQ), ret,                b);
#elif   defined(__AVX__)
	ret = opCode(add_pd,
					opCode(and_pd, opCode(cmp_pd, c, d, _CMP_GE_OQ), a),
					opCode(and_pd, opCode(cmp_pd, c, d, _CMP_LT_OQ), b));
#else
	ret = opCode(add_pd,
			opCode(and_pd, opCode(cmpge_pd, c, d), a),
			opCode(and_pd, opCode(cmplt_pd, c, d), b));
#endif

	return	ret;
}

inline _MData_	opCode(mod_pd, _MData_ &x, const _MData_ &md)
{
	_MData_	min, ret;

	_MData_ xP  = opCode(add_pd, x,  md);
	_MData_ xM  = opCode(sub_pd, x,  md);
	_MData_ x2  = opCode(mul_pd, x,   x);
	_MData_ xP2 = opCode(mul_pd, xP, xP);
	_MData_ xM2 = opCode(mul_pd, xM, xM);

	min = opCode(min_pd, opCode(min_pd, xP2, xM2), x2);
#ifdef	__AVX512F__
	ret = opCode(mask_add_pd, opCode(setzero_pd), opCode(cmp_pd_mask, min, xP2, _CMP_EQ_OQ), opCode(setzero_pd), xP);
	ret = opCode(mask_add_pd, ret,                opCode(cmp_pd_mask, min, xM2, _CMP_EQ_OQ), ret,   xM);
	ret = opCode(mask_add_pd, ret,                opCode(cmp_pd_mask, min, x2,  _CMP_EQ_OQ), ret,   x);
#elif   defined(__AVX__)
	ret = opCode(add_pd,
		opCode(add_pd,
			opCode(and_pd, opCode(cmp_pd, min, xP2, _CMP_EQ_OS), xP),
			opCode(and_pd, opCode(cmp_pd, min, xM2, _CMP_EQ_OS), xM)),
		opCode(and_pd, opCode(cmp_pd, min, x2, _CMP_EQ_OS), x));
#else
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
#ifdef	__AVX512F__
	return	opCode(add_pd, opCode(permute_pd, x, 0b01010101), x);
#elif defined(__AVX__)
	return	opCode(add_pd, opCode(permute_pd, x, 0b00000101), x);
#else
	return	opCode(add_pd, opCode(shuffle_pd, x, x, 0b00000001), x);
#endif
}

//returns y^2, xy
inline _MData_	opCode(vqcd0_pd, const _MData_ &x)
{
#if defined(__AVX__)
	return	opCode(mul_pd, opCode(permute_pd, x, 0b00010001), opCode(permute_pd, x, 0b01010101));  //(y,x)*(y,y)
#else
	return	opCode(mul_pd, opCode(shuffle_pd, x, x, 0b0001), opCode(shuffle_pd, x, x, 0b0101));
#endif
}

/* -----------------------------------------------------------------------
   double-precision rsqrt fallback for triSimd:
   Usage:  opCode(rsqrt_pd, x)   // works for SSE2/AVX/AVX-512 widths
   Default = precise 1/sqrt.  Optionally enable fast AVX-512 seed + NR:
   #define TRISIMD_USE_AVX512_RSQRT14 1
   ----------------------------------------------------------------------- */
static inline _MData_ opCode(rsqrt_pd, _MData_ x) {
#if defined(__AVX512F__) && defined(TRISIMD_USE_AVX512_RSQRT14)
    /* Fast path (AVX-512): VRSQRT14PD seed + one Newton step:
       y <- y * (1.5 - 0.5*x*y*y) */
    _MData_ y = opCode(rsqrt14_pd, x);                       // VRSQRT14PD
    _MData_ halfx = opCode(mul_pd, x, opCode(set1_pd, 0.5));
  #if defined(__FMA__) || defined(__AVX512F__)
    _MData_ term = opCode(fnmsub_pd, halfx, opCode(mul_pd, y, y), opCode(set1_pd, 1.5));
    return opCode(mul_pd, y, term);
  #else
    _MData_ y2   = opCode(mul_pd, y, y);
    _MData_ term = opCode(sub_pd, opCode(set1_pd, 1.5), opCode(mul_pd, halfx, y2));
    return opCode(mul_pd, y, term);
  #endif
#else
    /* Precise & portable: 1 / sqrt(x) */
    return opCode(div_pd, opCode(set1_pd, 1.0), opCode(sqrt_pd, x));
#endif
}

#undef	_MData_

#if	defined(__AVX512F__)
	#define	_MData_ __m512
#elif	defined(__AVX__)
	#define	_MData_ __m256
#else
	#define	_MData_ __m128
#endif

#ifndef	__INTEL_COMPILER

constexpr float Inf_f = __builtin_inff();
constexpr float Nan_f = __builtin_nanf("0x3FFFFF");

constexpr float PiA_f = -3.140625f;
constexpr float PiB_f = -0.0009670257568359375f;
constexpr float PiC_f = -6.2771141529083251953e-07f;
constexpr float PiD_f = -1.2154201256553420762e-10f;

constexpr float s0_f  =  2.6083159809786593541503e-06f;
constexpr float s1_f  = -0.0001981069071916863322258f;
constexpr float s2_f  =  0.00833307858556509017944336f;
constexpr float s3_f  = -0.166666597127914428710938f;
#ifdef	__AVX512F__
constexpr _MData_ oPif      = {     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,
				    1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI };
constexpr _MData_ zeroNegf  = {       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,
				      -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f };
constexpr _MData_ fHlf      = {        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,
				       0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f };
constexpr _MData_ TriMaxf   = {         1e7,         1e7,         1e7,         1e7,         1e7,         1e7,         1e7,         1e7,
					1e7,         1e7,         1e7,         1e7,         1e7,         1e7,         1e7,         1e7 };
constexpr _MData_ fInf      = {       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,
				      Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f };
constexpr _MData_ fNan      = {       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,
				      Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f };
constexpr _MData_ PiAf      = {       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,
				      PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f };
constexpr _MData_ PiBf      = {       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,
				      PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f };
constexpr _MData_ PiCf      = {       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,
				      PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f };
constexpr _MData_ PiDf      = {       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,
				      PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f };
constexpr _MData_ hPiAf     = {  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,
				 0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f };
constexpr _MData_ hPiBf     = {  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,
				 0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f };
constexpr _MData_ hPiCf     = {  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,
				 0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f };
constexpr _MData_ hPiDf     = {  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,
				 0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f };
constexpr _MData_ s0f       = {        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,
				       s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f };
constexpr _MData_ s1f       = {        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,
				       s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f };
constexpr _MData_ s2f       = {        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,
				       s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f };
constexpr _MData_ s3f       = {        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,
				       s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f };
constexpr _MData_ cMulSgnf  = { 1.f,-1.f, 1.f,-1.f, 1.f,-1.f, 1.f,-1.f, 1.f,-1.f, 1.f,-1.f, 1.f,-1.f, 1.f,-1.f };
#elif   defined(__AVX__)
constexpr _MData_ oPif      = {     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI };
constexpr _MData_ zeroNegf  = {       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f,       -0.0f };
constexpr _MData_ fHlf       = {        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f,        0.5f };
#ifdef	__FMA__
constexpr _MData_ TriMaxf   = {         1e7,         1e7,         1e7,         1e7,         1e7,         1e7,         1e7,         1e7 };
#else
constexpr _MData_ TriMaxf   = {         1e5,         1e5,         1e5,         1e5,         1e5,         1e5,         1e5,         1e5 };
#endif
constexpr _MData_ fInf      = {       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f,       Inf_f };
constexpr _MData_ fNan      = {       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f,       Nan_f };
constexpr _MData_ PiAf      = {       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f,       PiA_f };
constexpr _MData_ PiBf      = {       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f,       PiB_f };
constexpr _MData_ PiCf      = {       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f,       PiC_f };
constexpr _MData_ PiDf      = {       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f,       PiD_f };
constexpr _MData_ hPiAf     = {  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f };
constexpr _MData_ hPiBf     = {  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f };
constexpr _MData_ hPiCf     = {  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f };
constexpr _MData_ hPiDf     = {  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f };
constexpr _MData_ s0f       = {        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f,        s0_f };
constexpr _MData_ s1f       = {        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f,        s1_f };
constexpr _MData_ s2f       = {        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f,        s2_f };
constexpr _MData_ s3f       = {        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f,        s3_f };
constexpr _MData_ cMulSgnf  = {         1.f,        -1.f,         1.f,        -1.f,         1.f,        -1.f,         1.f,        -1.f};
#else
constexpr _MData_ oPif      = {     1./M_PI,     1./M_PI,     1./M_PI,     1./M_PI };
constexpr _MData_ zeroNegf  = {       -0.0f,       -0.0f,       -0.0f,       -0.0f };
constexpr _MData_ fHlf       = {        0.5f,        0.5f,        0.5f,        0.5f };
constexpr _MData_ TriMaxf   = {         1e5,         1e5,         1e5,         1e5 };
constexpr _MData_ fInf      = {       Inf_f,       Inf_f,       Inf_f,       Inf_f };
constexpr _MData_ PiAf      = {       PiA_f,       PiA_f,       PiA_f,       PiA_f };
constexpr _MData_ PiBf      = {       PiB_f,       PiB_f,       PiB_f,       PiB_f };
constexpr _MData_ PiCf      = {       PiC_f,       PiC_f,       PiC_f,       PiC_f };
constexpr _MData_ PiDf      = {       PiD_f,       PiD_f,       PiD_f,       PiD_f };
constexpr _MData_ hPiAf     = {  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f,  0.5f*PiA_f };
constexpr _MData_ hPiBf     = {  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f,  0.5f*PiB_f };
constexpr _MData_ hPiCf     = {  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f,  0.5f*PiC_f };
constexpr _MData_ hPiDf     = {  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f,  0.5f*PiD_f };
constexpr _MData_ s0f       = {        s0_f,        s0_f,        s0_f,        s0_f };
constexpr _MData_ s1f       = {        s1_f,        s1_f,        s1_f,        s1_f };
constexpr _MData_ s2f       = {        s2_f,        s2_f,        s2_f,        s2_f };
constexpr _MData_ s3f       = {        s3_f,        s3_f,        s3_f,        s3_f };
constexpr _MData_ cMulSgnf  = {         1.f,        -1.f,         1.f,        -1.f};
#endif

/*	Sleef	*/
inline void printiVar(_MInt_ d, const char *name) {

	printf ("%s", name);
#if	defined(__AVX512F__)
	int r[16] __attribute((aligned(64)));
	opCode(store_si512, r, d);
	for (int i=0; i<16; i++)
#elif	defined(__AVX__)
	int r[8] __attribute((aligned(32)));
	opCode(store_si256, ((_MInt_ *)r), d);
	for (int i=0; i<8; i++)
#else
	int r[4] __attribute((aligned(16)));
	opCode(store_si128, ((_MInt_ *)r), d);
	for (int i=0; i<4; i++)
#endif
		printf(" %d", r[i]);
	printf("\n");
}

inline void printsVar(_MData_ d, const char *name) {

	printf ("%s", name);
#if	defined(__AVX512F__)
	for (int i=0; i<16; i++)
#elif	defined(__AVX__)
	for (int i=0; i<8; i++)
#else
	for (int i=0; i<4; i++)
#endif
		printf(" %f", d[i]);
	printf("\n");
}

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

// from a, b, c, returns a if c>d and b if c<d
inline _MData_	opCode(kkk_ps, const _MData_ &a, const _MData_ &b, const _MData_ &c, const _MData_ &d)
{
_MData_	ret;

#if	defined(__AVX512F__)
	ret = opCode(mask_add_ps, opCode(setzero_ps), opCode(cmp_ps_mask, c, d, _CMP_GE_OQ), opCode(setzero_ps), a);
	ret = opCode(mask_add_ps, ret,                opCode(cmp_ps_mask, c, d, _CMP_LT_OQ), ret,                b);
#elif   defined(__AVX__)
	ret = opCode(add_ps,
					opCode(and_ps, opCode(cmp_ps, c, d, _CMP_GE_OQ), a),
					opCode(and_ps, opCode(cmp_ps, c, d, _CMP_LT_OQ), b));
#else
	ret = opCode(add_ps,
			opCode(and_ps, opCode(cmpge_ps, c, d), a),
			opCode(and_ps, opCode(cmplt_ps, c, d), b));
#endif

	return	ret;
}

inline _MData_	opCode(mod_ps, _MData_ &x, const _MData_ &md)
{
	_MData_	min, ret;

	_MData_ xP  = opCode(add_ps, x,  md);
	_MData_ xM  = opCode(sub_ps, x,  md);
	_MData_ x2  = opCode(mul_ps, x,   x);
	_MData_ xP2 = opCode(mul_ps, xP, xP);
	_MData_ xM2 = opCode(mul_ps, xM, xM);

	min = opCode(min_ps, opCode(min_ps, xP2, xM2), x2);
#if	defined(__AVX512F__)
	ret = opCode(mask_add_ps, opCode(setzero_ps), opCode(cmp_ps_mask, min, xP2, _CMP_EQ_OQ), opCode(setzero_ps), xP);
	ret = opCode(mask_add_ps, ret,                opCode(cmp_ps_mask, min, xM2, _CMP_EQ_OQ), ret,                xM);
	ret = opCode(mask_add_ps, ret,                opCode(cmp_ps_mask, min, x2,  _CMP_EQ_OQ), ret,                x);
#elif   defined(__AVX__)
	ret = opCode(add_ps,
		opCode(add_ps,
			opCode(and_ps, opCode(cmp_ps, min, xP2, _CMP_EQ_OQ), xP),
			opCode(and_ps, opCode(cmp_ps, min, xM2, _CMP_EQ_OQ), xM)),
		opCode(and_ps, opCode(cmp_ps, min, x2, _CMP_EQ_OQ), x));
#else
	ret = opCode(add_ps,
		opCode(add_ps,
			opCode(and_ps, opCode(cmpeq_ps, min, xP2), xP),
			opCode(and_ps, opCode(cmpeq_ps, min, xM2), xM)),
		opCode(and_ps, opCode(cmpeq_ps, min, x2), x));
#endif

	return	ret;
}

/* x^2,y^2 returns x^2+y^2, x^2+y^2 */
inline _MData_	opCode(md2_ps, const _MData_ &x)
{
#if defined(__AVX__)
	return	opCode(add_ps, opCode(permute_ps, x, 0b10110001), x);
#else
	return	opCode(add_ps, opCode(shuffle_ps, x, x, 0b10110001), x);
#endif
}

/* x,y returns y^2, xy */
inline _MData_	opCode(vqcd0_ps, const _MData_ &x)
{
#if defined(__AVX__)
	return	opCode(mul_ps, opCode(permute_ps, x, 0b10110001), opCode(permute_ps, x, 0b11110101));  //(y,x)*(y,y)
#else
	return	opCode(mul_ps, opCode(shuffle_ps, x, x, 0b10110001), opCode(shuffle_ps, x, x, 0b11110101));
#endif
}

/* swap re<->im inside each complex pair: [re,im,re,im,...] -> [im,re,im,re,...]*/
inline _MData_ opCode(cswap_ps, const _MData_ &x)
{
#if defined(__AVX__)
    return opCode(permute_ps, x, 0b10110001);
#else
    return opCode(shuffle_ps, x, x, 0b10110001);
#endif
}

inline _MData_ opCode(cmul_ps, const _MData_ &a, const _MData_ &b)
{
#if defined(__AVX__)
    _MData_ brbr = opCode(permute_ps, b, 0b10100000);
    _MData_ bibi = opCode(permute_ps, b, 0b11110101);
#else
    _MData_ brbr = opCode(shuffle_ps, b, b, 0b10100000);
    _MData_ bibi = opCode(shuffle_ps, b, b, 0b11110101);
#endif

    _MData_ t1 = opCode(mul_ps, a, brbr);
    _MData_ t2 = opCode(mul_ps, opCode(cswap_ps, a), bibi);

#if defined(__AVX512F__) || defined(__FMA__)
    return opCode(fmadd_ps, cMulSgnf, t2, t1);
#else
    return opCode(add_ps, t1, opCode(mul_ps, cMulSgnf, t2));
#endif
}

/* [re,im,re,im,...] -> [re,re,re,re,...] (pairwise) */
inline _MData_ opCode(dupRe_ps, const _MData_ &x)
{
#if defined(__AVX__)
    return opCode(permute_ps, x, 0b10100000);   // (0,0,2,2,4,4,...)
#else
    return opCode(shuffle_ps, x, x, 0b10100000);
#endif
}

/* [re,im,re,im,...] -> [im,im,im,im,...] (pairwise) */
inline _MData_ opCode(dupIm_ps, const _MData_ &x)
{
#if defined(__AVX__)
    return opCode(permute_ps, x, 0b11110101);   // (1,1,3,3,5,5,...)
#else
    return opCode(shuffle_ps, x, x, 0b11110101);
#endif
}

/* returns Im(f1^2 * f2) replicated in both lanes of each complex pair */
inline _MData_ opCode(im_f1sq_f2_ps, const _MData_ &f1, const _MData_ &f2)
{
    // f1 = [a,b,...], f2 = [c,d,...]
    // a2,b2 in lanes
    _MData_ a2b2 = opCode(mul_ps, f1, f1);               // [a^2, b^2, ...]
    _MData_ diff = opCode(sub_ps, a2b2, opCode(cswap_ps, a2b2)); // [a^2-b^2, b^2-a^2,...]
    _MData_ a2mb2 = opCode(dupRe_ps, diff);              // [a^2-b^2, a^2-b^2, ...]

    // ab from your vqcd0_ps: [b^2, a*b] per pair
    _MData_ ab   = opCode(dupIm_ps, opCode(vqcd0_ps, f1));  // [ab,ab,...]
    _MData_ twoab = opCode(mul_ps, ab, opCode(set1_ps, 2.f));

    _MData_ c = opCode(dupRe_ps, f2);                    // [c,c,...]
    _MData_ d = opCode(dupIm_ps, f2);                    // [d,d,...]

#if defined(__AVX512F__) || defined(__FMA__)
    // S = (2ab)*c + (a^2-b^2)*d
    return opCode(fmadd_ps, a2mb2, d, opCode(mul_ps, twoab, c));
#else
    return opCode(add_ps, opCode(mul_ps, twoab, c), opCode(mul_ps, a2mb2, d));
#endif
}

// /* X = x_r,x_i
//    Y = y_r,y_i
//    returns x_ry_r-x_iy_i, x_r y_i + y_r x_i ... */
// inline _MData_ opCode(cmul_ps, const _MData_ &a, const _MData_ &b)
// {
//   // brbr = [br0, br0, br1, br1, ...]
// #if defined(__AVX__)
//   _MData_ brbr = opCode(permute_ps, b, 0b10100000);
//   _MData_ bibi = opCode(permute_ps, b, 0b11110101);
// #else
//   _MData_ brbr = opCode(shuffle_ps, b, b, 0b10100000);
//   _MData_ bibi = opCode(shuffle_ps, b, b, 0b11110101);
// #endif
//
//   // a * br  gives [ar*br, ai*br, ...]
//   _MData_ p1 = opCode(mul_ps, a, brbr);
//
//   // swap(a) * bi gives [ai*bi, ar*bi, ...]
//   _MData_ asw = opCode(cswap_ps, a);
//   _MData_ p2  = opCode(mul_ps, asw, bibi);
//
// // addsub: even lanes subtract, odd lanes add
// // => [ar*br - ai*bi, ai*br + ar*bi, ...]
//   return opCode(addsub_ps, p1, p2);
// }

// One-step Newton–Raphson refinement of rsqrt
// returns ~22–23 bits accuracy, good for your normalization pass.
static inline _MData_ rsqrt1_ps(const _MData_ s2)
{
    const _MData_ half         = opCode(set1_ps, 0.5f);
    const _MData_ threehalves  = opCode(set1_ps, 1.5f);

    _MData_ y  = opCode(rsqrt_ps, s2);           // ~12-bit seed
    _MData_ yy = opCode(mul_ps, y, y);

#if defined(__FMA__)
    // corr = 1.5 - 0.5*s2*yy   (use FMA if available)
  #if defined(__AVX2__) || defined(__AVX512F__)
    _MData_ corr = opCode(fnmadd_ps, opCode(mul_ps, half, s2), yy, threehalves);
  #else
    _MData_ corr = opCode(sub_ps, threehalves,
                          opCode(mul_ps, opCode(mul_ps, half, s2), yy));
  #endif
#else
    _MData_ corr = opCode(sub_ps, threehalves,
                          opCode(mul_ps, opCode(mul_ps, half, s2), yy));
#endif

    return opCode(mul_ps, y, corr);
}

#undef	_MData_
#undef	_PREFIX
#undef opCode_P
#undef opCode_N
#undef opCode

#endif
