#ifndef	_SIMD_
	#define	_SIMD_

	#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
	#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
	#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

	#include <immintrin.h>

	#ifdef  __AVX512F__
	        #define _MData_ __m512
	        #define _MInt_  __m512i
	        #define _MHnt_  __m256i
	#elif   defined(__AVX__)
	        #define _MData_ __m256
	        #define _MInt_  __m256i
	        #define _MHnt_  __m128i
	#else
	        #define _MData_ __m128d
	        #define _MInt_  __m128i
	#endif

	#if     defined(__AVX512F__)
	        #define _PREFIX_ _mm512
	        #define _PREFXL_ _mm256
	        #define opCodl(x,...) opCode_N(_PREFXL_, x, __VA_ARGS__)
	#else
	        #if not defined(__AVX__) and not defined(__AVX2__)
	                #define _PREFIX_ _mm
	        #else
	                #define _PREFIX_ _mm256
	                #define _PREFXL_ _mm
	                #define opCodl(x,...) opCode_N(_PREFXL_, x, __VA_ARGS__)
	        #endif
	#endif

	class	Simd_f	{
		private:

		_MData_	data;

		public:

		#ifdef	_AVX512F_
			Simd_f(float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7, float x8, float x9, float x10, float x11, float x12, float x13, float x14, float x15) {
				data = opCode(set_ps, x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1, x0);
			}
		#elif defined(_AVX_)
			Simd_f(float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7) {
				data = opCode(set_ps, x7, x6, x5, x4, x3, x2, x1, x0);
			}
		#else
			Simd_f(float x0, float x1, float x2, float x3) {
			data = opCode(set_ps, x3, x2, x1, x0);
		}
		#endif

			Simd_f(float x) {
			data = opCode(set1_ps, x0);
		}

			Simd_f(float *memAddress) {
			data = opCode(load_ps, memAddress);
		}

		void	save  (float *memAddress) {
			opCode(store_ps, memAddress);
		}

		void	stream (float *memAddress) {
			data = opCode(load_ps, memAddress);
		}

		Simd_f	operator+(Simd_f &x) {
			return	opCode(add_ps, this->data, x->data);
		}

		Simd_f	operator-(Simd_f &x) {
			return	opCode(sub_ps, this->data, x->data);
		}

		Simd_f	operator*(Simd_f &x) {
			return	opCode(mul_ps, this->data, x->data);
		}

		Simd_f	operator/(Simd_f &x) {
			return	opCode(div_ps, this->data, x->data);
		}

		Simd_f	operator!() {
		#if defined(__AVX__)
		        return  opCode(add_ps, opCode(permute_ps, this->data, 0b10110001), this->data);
		#else
		        return  opCode(add_ps, opCode(shuffle_ps, this->data, this->data, 0b10110001), this->data);
		#endif
		}

		Simd_f	operator~() {
		#if   defined(__AVX512F__)
			return	opCode(mul_ps, this->data, opCode(set_ps, -1., 0., -1., 0., -1., 0., -1., 0., -1., 0., -1., 0., -1., 0., -1., 0.));
		#elif defined(__AVX__)
			return	opCode(mul_ps, this->data, opCode(set_ps, -1., 0., -1., 0., -1., 0., -1., 0.));
		#else
			return	opCode(mul_ps, this->data, opCode(set_ps, -1., 0., -1., 0.));
		#endif
		}
		


