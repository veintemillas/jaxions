#if	!defined(__AVX2__) && !defined(__AVX__)
	#error("Building settings won't allow compilation for Avx/Avx-2")
#endif

#ifndef __SIMD
        #define __SIMD

	#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
	#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
	#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

	#include <immintrin.h>
	#include "enumFields.h"

	#define _MData_ __m256
	#define _MDatd_ __m256d
	#define _MInt_  __m256i
	#define _MHnt_  __m128i
	#define _PREFIX_ _mm256
	#define _PREFXL_ _mm128
	#define opCodl(x,...) opCode_N(_PREFXL_, x, __VA_ARGS__)

        namespace Simd {

                constexpr size_t sAlign = 32;

		class	Simd_f;
		class	Simd_d;

		class	Mask_f	{
                	static constexpr size_t sWide = sAlign/sizeof(int);

			private:

			_MInt_	data;

			public:

				Mask_f()			: data(opCode(setzero_si256)) {}
				Mask_f(const _MInt_ &in) 	: data(in) 		      {}


			inline	Mask_f	 operator& (const Mask_f &b) {
				return	opCode(castps_si256, opCode(and_ps, opCode(castsi256_ps, this->data), opCode(castsi256_ps, b.data)));
			}

			inline	Mask_f	&operator&=(const Mask_f &b) {
				(*this) = (*this)&b;
				return	(*this);
			}

			inline	Mask_f	 operator| (const Mask_f &b) {
				return	opCode(castps_si256, opCode(or_ps,  opCode(castsi256_ps, this->data), opCode(castsi256_ps, b.data)));
			}

			inline	Mask_f	&operator|=(const Mask_f &b) {
				(*this) = (*this)|b;
				return	(*this);
			}

			inline	Mask_f	 operator! () {
				return	opCode(castps_si256, opCode(andnot_ps, opCode(castsi256_ps, this->data), opCode(castsi256_ps, opCode(set1_epi32, 0xffffffff))));
			}

			inline	int	Count() {
				int count = 0;

				for (int i=0; i<4; i++) {
					auto &x = data[i];

					if (x != 0) {
						if (x == 0xffffffffffffffff)
							count+=2;
						else
							count++;
					}
				}

				return	count;
			}

			void	Print(const char *str)	{
				printiVar(this->data, str);
			}

			friend	class	Simd_f;
		};

		class	Simd_f	{
			private:

			_MData_	data;

			public:

                	static constexpr size_t sWide = sAlign/sizeof(float);
                	static constexpr size_t xWide = 2;
                	static constexpr size_t yWide = 2;
                	static constexpr size_t zWide = 2;

                	static constexpr size_t sWdCx = sAlign(2*sizeof(float));
                	static constexpr size_t xWdCx = 1;
                	static constexpr size_t yWdCx = 2;
                	static constexpr size_t zWdCx = 2;

			typedef float sData;
			typedef Mask_f Mask;

				Simd_f() {
				data = opCode(setzero_ps);
			}

				Simd_f(float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7) {
				data = opCode(set_ps, x7, x6, x5, x4, x3, x2, x1, x0);
			}

				Simd_f(uint x0, uint x1, uint x2, uint x3) {
				data = opCode(castsi256_ps, opCode(set_epi32, x3, x2, x1, x0, x3, x2, x1, x0));
			}

				Simd_f(float x0, float x1) {
				data = opCode(set_ps, x1, x0, x1, x0, x1, x0, x1, x0);
			}

				Simd_f(float x0) {
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

			void	Load  (float * __restrict__ memAddress) {
				float * __restrict__ mAddr = (float * __restrict__) __builtin_assume_aligned (memAddress, Simd::sAlign);
				data = opCode(load_ps, mAddr);
			}

			void	Save  (float * __restrict__ memAddress) {
				float * __restrict__ mAddr = (float * __restrict__) __builtin_assume_aligned (memAddress, Simd::sAlign);
				opCode(store_ps,  mAddr, this->data);
			}

			void	SaveMask(Mask_f msk, float * __restrict__ memAddress) {
				float * __restrict__ mAddr = (float * __restrict__) __builtin_assume_aligned (memAddress, Simd::sAlign);
				opCode(maskstore_ps, mAddr, msk.data, this->data);
			}

			void	Stream (float *memAddress) {
				float * __restrict__ mAddr = (float * __restrict__) __builtin_assume_aligned (memAddress, Simd::sAlign);
				opCode(stream_ps, mAddr, this->data);
			}

			void	StreamMask(Mask_f msk, const Simd_f &x, float * __restrict__ memAddress) {
				float * __restrict__ mAddr = (float * __restrict__) __builtin_assume_aligned (memAddress, Simd::sAlign);
				opCode(stream_ps, mAddr, opCode(blendv_ps, x.data, this->data, opCode(castsi256_ps, msk.data)));
			}

			inline	Simd_f	operator+(const Simd_f &x) {
				return	opCode(add_ps, this->data, x.data);
			}

			inline	Simd_f	operator-(const Simd_f &x) {
				return	opCode(sub_ps, this->data, x.data);
			}

			inline	Simd_f	operator*(const Simd_f &x) {
				return	opCode(mul_ps, this->data, x.data);
			}

			inline	Simd_f	operator/(const Simd_f &x) {
				return	opCode(div_ps, this->data, x.data);
			}

			inline	Simd_f	&operator+=(const Simd_f &x) {
				(*this) = (*this)+x;
				return	(*this);
			}

			inline	Simd_f	&operator-=(const Simd_f &x) {
				(*this) = (*this)-x;
				return	(*this);
			}

			inline	Simd_f	&operator*=(const Simd_f &x) {
				(*this) = (*this)*x;
				return	(*this);
			}

			inline	Simd_f	&operator/=(const Simd_f &x) {
				(*this) = (*this)/x;
				return	(*this);
			}

			inline	Simd_f	operator-() {
				return	opCode(sub_ps, opCode(setzero_ps), this->data);
			}

			inline	Simd_f	operator!() {
				return	opCode(add_ps, opCode(permute_ps, this->data, 0b10110001), this->data);
			}

			inline	Simd_f	operator~() {
				return	opCode(mul_ps, this->data, opCode(set_ps, -1., 0., -1., 0., -1., 0., -1., 0.));
			}

			inline	Simd_f	operator^(const Mask_f &msk) {
				return	opCode(and_ps, this->data, opCode(castsi256_ps, msk.data));
			}

			inline	Simd_f	operator&(const Simd_f &b) {
				return	opCode(and_ps, this->data, b.data);
			}

			inline	Simd_f	operator|(const Simd_f &b) {
				return	opCode(or_ps,  this->data, b.data);
			}

			inline	Simd_f	operator^(const Simd_f &b) {
				return	opCode(xor_ps, this->data, b.data);
			}

			inline	Simd_f	operator>>(uint i) {
				#ifndef	__AVX2__
				_MHnt_	high = opCode(extractf128_si256, opCode(castps_si256, this->data), 1);
				_MHnt_	low  = opCode(extractf128_si256, opCode(castps_si256, this->data), 0);
				return	opCode(insertf128_ps, opCode(castps128_ps256, opCodl(srli_epi32, low, i)), opCodl(srli_epi32, high, i), 1);
				#else
				return	opCode(castsi256_ps, opCode(srli_epi32, opCode(castps_si256, this->data), i));
				#endif
			}

			inline	Simd_f	operator<<(uint i) {
				#ifndef	__AVX2__
				_MHnt_	high = opCode(extractf128_si256, opCode(castps_si256, this->data), 1);
				_MHnt_	low  = opCode(extractf128_si256, opCode(castps_si256, this->data), 0);
				return	opCode(insertf128_ps, opCode(castps128_ps256, opCodl(srli_epi32, low, i)), opCodl(srli_epi32, high, i), 1);
				#else
				return	opCode(castsi256_ps, opCode(slli_epi32, opCode(castps_si256, this->data), i));
				#endif
			}

			/*	Global shift, only required for the vectorized RNG for single precision, these functions don't exist in double precision	*/
			inline	Simd_f	grShift(uint i) {
				#ifndef	__AVX2__
				_MHnt_	high = opCode(extractf128_si256, opCode(castps_si256, this->data), 1);
				_MHnt_	low  = opCode(extractf128_si256, opCode(castps_si256, this->data), 0);
				return	opCode(insertf128_ps, opCode(castps128_ps256, opCodl(srli_si128, low, i)), opCodl(srli_si128, high, i), 1);
				#else
				return	opCode(castsi256_ps, opCode(bsrli_epi128, opCode(castps_si256, this->data), i));
				#endif
			}

			inline	Simd_f	glShift(uint i) {
				#ifndef	__AVX2__
				_MHnt_	high = opCode(extractf128_si256, opCode(castps_si256, this->data), 1);
				_MHnt_	low  = opCode(extractf128_si256, opCode(castps_si256, this->data), 0);
				return	opCode(insertf128_ps, opCode(castps128_ps256, opCodl(slli_si128, low, i)), opCodl(slli_si128, high, i), 1);
				#else
				return	opCode(castsi256_ps, opCode(bslli_epi128, opCode(castps_si256, this->data), i));
				#endif
			}

			inline	Simd_f	&operator&=(const Simd_f &x) {
				(*this) = (*this)&x;
				return	(*this);
			}

			inline	Simd_f	&operator|=(const Simd_f &x) {
				(*this) = (*this)|x;
				return	(*this);
			}

			inline	Simd_f	&operator^=(const Simd_f &x) {
				(*this) = (*this)^x;
				return	(*this);
			}

			inline	Simd_f	&operator>>=(uint i) {
				(*this) = (*this)>>i;
				return	(*this);
			}

			inline	Simd_f	&operator<<=(uint i) {
				(*this) = (*this)<<i;
				return	(*this);
			}

			inline	Mask_f	operator>(const Simd_f &b) {
				return	opCode(castps_si256, opCode(cmp_ps, this->data, b.data, _CMP_GT_OQ));
			}

			inline	Mask_f	operator>=(const Simd_f &b) {
				return	opCode(castps_si256, opCode(cmp_ps, this->data, b.data, _CMP_GE_OQ));
			}

			inline	Mask_f	operator<(const Simd_f &b) {
				return	opCode(castps_si256, opCode(cmp_ps, this->data, b.data, _CMP_LT_OQ));
			}

			inline	Mask_f	operator<=(const Simd_f &b) {
				return	opCode(castps_si256, opCode(cmp_ps, this->data, b.data, _CMP_LE_OQ));
			}

			inline	Mask_f	operator==(const Simd_f &b) {
				return	opCode(castps_si256, opCode(cmp_ps, this->data, b.data, _CMP_EQ_UQ));
			}

			inline	Simd_f	fma(const Simd_f &a, const Simd_f &b) {
				return	opCode(fmadd_ps, this->data, a.data, b.data);
			}

			inline	Simd_f	fms(const Simd_f &a, const Simd_f &b) {
				return	opCode(fmsub_ps, this->data, a.data, b.data);
			}

			inline	Simd_f	xPermute () {
				return	opCode(permute_ps, this->data, 0b10110001);
			}

			inline	Simd_f	yPermute () {
				return	opCode(permute2f128_ps, this->data, this->data, 0b00000001);
			}

			inline	Simd_f	zPermute () {
				return	opCode(permute_ps, this->data, 0b01001110);
			}

			inline	Simd_f	rPermute () {
				return	opCode(permute_ps, this->data, 0b00011011);
			}

			inline	Simd_f	xPermCpx () {
				return	(*this);
			}

			inline	Simd_f	yPermCpx () {
				return	opCode(permute2f128_ps, this->data, this->data, 0b00000001);
			}

			inline	Simd_f	zPermCpx () {
				return	opCode(permute_ps, this->data, 0b01001110);
			}

			inline	void	SetRandom ();

			inline	float	Sum () {
				return	opCode(hadd_ps,
						opCode(hadd_ps,
							opCode(hadd_ps,
								(*this).data,
								opCode(permute2f128_ps, (*this).data, (*this).data, 0b00000001)),
					       		(*this).data),
					       	(*this).data)[0];
			}
// TODO rSum iSum
			inline	float&	operator[](int lane) {
				return	data[lane];
			}

			inline	_MData_&	raw() {
				return	data;
			}

			void	iPrint(const char *str)	{
				printuVar(opCode(castps_si256, this->data), str);
			}

			void	Print(const char *str)	{
				printsVar(this->data, str);
			}

			friend	class	Mask_f;

			friend  Simd_f  sqrt    (const Simd_f&);
			friend  Simd_f  cos     (const Simd_f&);
			friend  Simd_f  sin     (const Simd_f&);
			friend  Simd_f  log     (const Simd_f&);
			friend  Simd_f  exp     (const Simd_f&);
			friend  Simd_f  abs     (const Simd_f&);
		};

		Simd_f	sqrt	(const Simd_f&);
		Simd_f	cos	(const Simd_f&);
		Simd_f	sin	(const Simd_f&);
		Simd_f	log	(const Simd_f&);
		Simd_f	exp	(const Simd_f&);
		Simd_f	abs	(const Simd_f&);

		class	Mask_d	{
                	static constexpr size_t sWide = sAlign/sizeof(double);

			private:

			_MInt_	data;

			public:

				Mask_d()			: data(opCode(setzero_si256)) {}
				Mask_d(const _MInt_ &in) 	: data(in) 		      {}


			inline	Mask_d	 operator& (const Mask_d &b) {
				return	opCode(castpd_si256, opCode(and_pd, opCode(castsi256_pd, this->data), opCode(castsi256_pd, b.data)));
			}

			inline	Mask_d	&operator&=(const Mask_d &b) {
				(*this) = (*this)&b;
				return	(*this);
			}

			inline	Mask_d	 operator| (const Mask_d &b) {
				return	opCode(castpd_si256, opCode(or_pd,  opCode(castsi256_pd, this->data), opCode(castsi256_pd, b.data)));
			}

			inline	Mask_d	&operator|=(const Mask_d &b) {
				(*this) = (*this)|b;
				return	(*this);
			}

			inline	Mask_d	 operator! () {
				return	opCode(castpd_si256, opCode(andnot_pd, opCode(castsi256_pd, this->data), opCode(castsi256_pd, opCode(set1_epi32, 0xffffffff))));
			}

			inline	int	Count() {
				int count = 0;

				for (int i=0; i<4; i++) {
					auto &x = data[i];

					if (x != 0)
						count++;
				}

				return	count;
			}

			void	Print(const char *str)	{
				printlVar(this->data, str);
			}

			friend	class	Simd_d;
		};

		class	Simd_d	{
			private:

			_MDatd_	data;

			public:

                	static constexpr size_t sWide = sAlign/sizeof(double);
                	static constexpr size_t xWide = 1;
                	static constexpr size_t yWide = 2;
                	static constexpr size_t zWide = 2;

                	static constexpr size_t sWdCx = sAlign/(2*sizeof(double));
                	static constexpr size_t xWdCx = 1;
                	static constexpr size_t yWdCx = 1;
                	static constexpr size_t zWdCx = 2;

			typedef double sData;
			typedef Mask_d Mask;

				Simd_d() {
				data = opCode(setzero_pd);
			}

				Simd_d(double x0, double x1, double x2, double x3) {
				data = opCode(set_pd, x3, x2, x1, x0);
			}

				Simd_d(uint64 x0, uint64 x1) {
				data = opCode(castsi256_pd, opCode(set_epi64x, x1, x0, x1, x0));
			}

				Simd_d(double x0, double x1) {
				data = opCode(set_pd, x1, x0, x1, x0);
			}

				Simd_d(double x0) {
				data = opCode(set1_pd, x0);
			}

				Simd_d(double *memAddress) {
				data = opCode(load_pd, memAddress);
			}

				Simd_d(const _MDatd_ &in) : data(in) {};
				Simd_d(_MDatd_ &&in) : data(std::move(in)) {};

			inline	Simd_d&	operator=(const _MDatd_ &in) {
				data = in;
			}

			inline	Simd_d&	operator=(_MDatd_ &&in) {
				data = std::move(in);
			}

			void	Load  (double *memAddress) {
				data = opCode(load_pd, memAddress);
			}

			void	SaveMask(Mask_d msk, double *memAddress) {
				opCode(maskstore_pd, memAddress, msk.data, this->data);
			}

			void	Save  (double *memAddress) {
				opCode(store_pd,  memAddress, this->data);
			}

			void	Stream (double *memAddress) {
				opCode(stream_pd, memAddress, this->data);
			}

			void	StreamMask(Mask_d msk, const Simd_d &x, double * __restrict__ memAddress) {
				double * __restrict__ mAddr = (double * __restrict__) __builtin_assume_aligned (memAddress, Simd::sAlign);
				opCode(stream_pd, mAddr, opCode(blendv_pd, x.data, this->data, opCode(castsi256_pd, msk.data)));
			}

			inline	Simd_d	operator+(const Simd_d &x) {
				return	opCode(add_pd, this->data, x.data);
			}

			inline	Simd_d	operator-(const Simd_d &x) {
				return	opCode(sub_pd, this->data, x.data);
			}

			inline	Simd_d	operator*(const Simd_d &x) {
				return	opCode(mul_pd, this->data, x.data);
			}

			inline	Simd_d	operator/(const Simd_d &x) {
				return	opCode(div_pd, this->data, x.data);
			}

			inline	Simd_d	&operator+=(const Simd_d &x) {
				(*this) = (*this)+x;
				return	(*this);
			}

			inline	Simd_d	&operator-=(const Simd_d &x) {
				(*this) = (*this)-x;
				return	(*this);
			}

			inline	Simd_d	&operator*=(const Simd_d &x) {
				(*this) = (*this)*x;
				return	(*this);
			}

			inline	Simd_d	&operator/=(const Simd_d &x) {
				(*this) = (*this)/x;
				return	(*this);
			}

			inline	Simd_d	operator-() {
				return	opCode(sub_pd, opCode(setzero_pd), this->data);
			}

			inline	Simd_d	operator!() {
				return	opCode(add_pd, opCode(permute_pd, this->data, 0b00000101), this->data);
			}

			inline	Simd_d	operator~() {
				return	opCode(mul_pd, this->data, opCode(set_pd, -1., 0., -1., 0.));
			}

			inline	Simd_d	operator^(const Mask_d &msk) {
				return	opCode(and_pd, this->data, opCode(castsi256_pd, msk.data));
			}

			inline	Simd_d	operator&(const Simd_d &b) {
				return	opCode(and_pd, this->data, b.data);
			}

			inline	Simd_d	operator|(const Simd_d &b) {
				return	opCode(or_pd,  this->data, b.data);
			}

			inline	Simd_d	operator^(const Simd_d &b) {
				return	opCode(xor_pd, this->data, b.data);
			}

			inline	Simd_d	operator>>(uint i) {
				#ifndef	__AVX2__
				_MHnt_	high = opCode(extractf128_si256, opCode(castpd_si256, this->data), 1);
				_MHnt_	low  = opCode(extractf128_si256, opCode(castpd_si256, this->data), 0);
				return	opCode(insertf128_ps, opCode(castps128_ps256, opCodl(srli_epi64, low, i)), opCodl(srli_epi64, high, i), 1);
				#else
				return	opCode(castsi256_pd, opCode(srli_epi64, opCode(castpd_si256, this->data), i));
				#endif
			}

			inline	Simd_d	operator<<(uint i) {
				#ifndef	__AVX2__
				_MHnt_	high = opCode(extractf128_si256, opCode(castpd_si256, this->data), 1);
				_MHnt_	low  = opCode(extractf128_si256, opCode(castpd_si256, this->data), 0);
				return	opCode(insertf128_ps, opCode(castps128_ps256, opCodl(slli_epi64, low, i)), opCodl(slli_epi64, high, i), 1);
				#else
				return	opCode(castsi256_pd, opCode(slli_epi64, opCode(castpd_si256, this->data), i));
				#endif
			}

			inline	Simd_d	&operator&=(const Simd_d &x) {
				(*this) = (*this)&x;
				return	(*this);
			}

			inline	Simd_d	&operator|=(const Simd_d &x) {
				(*this) = (*this)|x;
				return	(*this);
			}

			inline	Simd_d	&operator^=(const Simd_d &x) {
				(*this) = (*this)^x;
				return	(*this);
			}

			inline	Simd_d	&operator>>=(uint i) {
				(*this) = (*this)>>i;
				return	(*this);
			}

			inline	Simd_d	&operator<<=(uint i) {
				(*this) = (*this)<<i;
				return	(*this);
			}

			inline	Mask_d	operator>(const Simd_d &b) {
				return	opCode(castpd_si256, opCode(cmp_pd, this->data, b.data, _CMP_GT_OQ));
			}

			inline	Mask_d	operator>=(const Simd_d &b) {
				return	opCode(castpd_si256, opCode(cmp_pd, this->data, b.data, _CMP_GE_OQ));
			}

			inline	Mask_d	operator<(const Simd_d &b) {
				return	opCode(castpd_si256, opCode(cmp_pd, this->data, b.data, _CMP_LT_OQ));
			}

			inline	Mask_d	operator<=(const Simd_d &b) {
				return	opCode(castpd_si256, opCode(cmp_pd, this->data, b.data, _CMP_LE_OQ));
			}

			inline	Mask_d	operator==(const Simd_d &b) {
				return	opCode(castpd_si256, opCode(cmp_pd, this->data, b.data, _CMP_EQ_UQ));
			}

			inline	Simd_d	fma(const Simd_d &a, const Simd_d &b) {
				return	opCode(fmadd_pd, this->data, a.data, b.data);
			}

			inline	Simd_d	fms(const Simd_d &a, const Simd_d &b) {
				return	opCode(fmsub_pd, this->data, a.data, b.data);
			}

			inline	Simd_d	xPermute () {
				return	(*this);
			}

			inline	Simd_d	yPermute () {
				return	opCode(permute2f128_pd, this->data, this->data, 0b00000001);
			}

			inline	Simd_d	zPermute () {
				return	opCode(permute_pd, this->data, 0b00000101);
			}

			inline	Simd_d	rPermute () {
				return	opCode(castps_pd, opCode(permute_ps, opCode(castpd_ps, this->data), 0b00011011));
			}

			inline	Simd_d	xPermCpx () {
				return	(*this);
			}

			inline	Simd_d	yPermCpx () {
				return	(*this);
			}

			inline	Simd_d	zPermCpx () {
				return	opCode(permute2f128_pd, this->data, this->data, 0b00000001);
			}

			inline	void	SetRandom ();
// {
//				(*this) = Simd_d(Su2Rand::genRand(), Su2Rand::genRand(), Su2Rand::genRand(), Su2Rand::genRand());
//			}

			inline	double	Sum () {
				return	opCode(hadd_pd,
						opCode(hadd_pd,
							(*this).data,
							opCode(permute2f128_pd, (*this).data, (*this).data, 0b00000001)),
					       	(*this).data)[0];
			}

			inline	double&	operator[](int lane) {
				return	data[lane];
			}

			inline	_MDatd_&	raw() {
				return	data;
			}

			void	iPrint(const char *str)	{
				printlVar(opCode(castpd_si256, this->data), str);
			}

			void	Print(const char *str)	{
				printdVar(this->data, str);
			}

			friend	class	Mask_f;

			friend  Simd_d  sqrt	(const Simd_d&);
			friend  Simd_d  cos	(const Simd_d&);
			friend  Simd_d  sin	(const Simd_d&);
			friend  Simd_d  log	(const Simd_d&);
			friend  Simd_d  exp	(const Simd_d&);
//			friend  Simd_d  log2	(const Simd_d&);
//			friend  Simd_d  exp2	(const Simd_d&);
			friend  Simd_d  abs	(const Simd_d&);
		};

		Simd_d	sqrt	(const Simd_d&);
		Simd_d	cos	(const Simd_d&);
		Simd_d	sin	(const Simd_d&);
		Simd_d	log	(const Simd_d&);
		Simd_d	exp	(const Simd_d&);
		Simd_d	abs	(const Simd_d&);
//		Simd_d  log2	(const Simd_d&);
//		Simd_d  exp2	(const Simd_d&);
	}
#endif
