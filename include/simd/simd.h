#ifndef	__SIMD_GLOBAL
#define	__SIMD_GLOBAL

#include <simd/simd-tri.h>
#include <simd/simd-math.h>

#ifdef  __AVX512F__
	#include <simd/simd-Avx512.h>
#elif   defined(__AVX2__ ) || defined(__AVX__)
	#include <simd/simd-Avx2.h>
#else
	#include <simd/simd-Sse4.h>
#endif

#endif
