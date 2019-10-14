#ifndef	__SSE2__
	#error("Building settings won't allow compilation for Sse2")
#endif

#ifndef __SIMD
        #define __SIMD

	#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
	#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
	#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

	#include <immintrin.h>

	#define _MData_ __m128
	#define _MInt_  __m128i
	#define _MHnt_  __m28i
	#define _PREFIX_ _mm128
	#define _PREFXL_ _mm28
	#define opCodl(x,...) opCode_N(_PREFXL_, x, __VA_ARGS__)

        namespace Simd {

                constexpr size_t sAlign = 16;

		class	Simd_f	{
			private:

			_MData_	data;

			public:

				Simd_f(float x0, float x1, float x2, float x3) {
				data = opCode(set_ps, x3, x2, x1, x0);
			}

				Simd_f(float x) {
				data = opCode(set1_ps, x0);
			}

				Simd_f(float *memAddress) {
				data = opCode(load_ps, memAddress);
			}

				Simd_f(const _MData_ &in) : data(in) {};
				Simd_f(_MData_ &&in) : data(std::move(in)) {};

			inline	Simd_f&	operator=(const _MData_ &in) {
				data = in;
			}

			inline	Simd_f&	operator=(_MData_ &&in) {
				data = std::move(in);
			}

			void	save  (float *memAddress) {
				opCode(store_ps, static_cast<void*>(memAddress));
			}

			void	stream (float *memAddress) {
				opCode(stream_ps, static_cast<void*>(memAddress), this->data);
			}

			inline	Simd_f&	operator+(Simd_f &x) {
				return	opCode(add_ps, this->data, x->data);
			}

			inline	Simd_f&	operator-(Simd_f &x) {
				return	opCode(sub_ps, this->data, x->data);
			}

			inline	Simd_f&	operator*(Simd_f &x) {
				return	opCode(mul_ps, this->data, x->data);
			}

			inline	Simd_f&	operator/(Simd_f &x) {
				return	opCode(div_ps, this->data, x->data);
			}

			inline	Simd_f&	operator+=(Simd_f &x) {
				return	(*this)+x;
			}

			inline	Simd_f&	operator-(Simd_f &x) {
				return	(*this)-x;
			}

			inline	Simd_f&	operator*(Simd_f &x) {
				return	(*this)*x;
			}

			inline	Simd_f&	operator/(Simd_f &x) {
				return	(*this)/x;
			}

			inline	Simd_f&	operator!() {
			        return  opCode(add_ps, opCode(shuffle_ps, this->data, this->data, 0b10110001), this->data);
			}

			inline	Simd_f&	operator~() {
				return	opCode(mul_ps, this->data, opCode(set_ps, -1., 0., -1., 0.));
			}

			inline	Simd_f&	fma(Simd_f &a, Simd_f &b) {
				return	opCode(fmadd_ps, this->data, a, b);
			}

			inline	Simd_f&	fms(Simd_f &a, Simd_f &b) {
				return	opCode(fmsub_ps, this->data, a, b);
			}

			inline	Simd_f&	zPermute () {
			        return  opCode(shuffle_ps, this->data, this->data, 0b01001110);
			}

			inline	Simd_f&	tPermute () {
			        return  opCode(shuffle_ps, this->data, this->data, 0b10110001);
			}

			inline	float	operator[](int lane) {
				return	data[lane];
			}
		}
	}
#endif
