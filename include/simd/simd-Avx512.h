#ifndef	__AVX512F__
	#error("Building settings won't allow compilation for Avx-512")
#endif

#ifndef __SIMD
        #define __SIMD

	#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
	#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
	#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

	#include <zmmintrin.h>
	#include "enumFields.h"

	#define _MData_ __m512
	#define _MDatd_ __m512d
	#define _MInt_  __m512i
	#define _MHnt_  __m256i
	#define _PREFIX_ _mm512
	#define _PREFXL_ _mm256
	#define opCodl(x,...) opCode_N(_PREFXL_, x, __VA_ARGS__)

        namespace Simd {

                constexpr size_t sAlign = 64;

		class	Simd_f;
		class	Simd_d;

		class	Mask_f	{
                	static constexpr size_t sWide = sAlign/sizeof(int);

			private:

			__mmask16	data;

			public:

				Mask_f()			: data(0) {}
				Mask_f(const __mmask16 &in) 	: data(in){}


			inline	Mask_f	 operator& (const Mask_f &b) {
				return	opCode(kand, this->data, b.data);
			}

			inline	Mask_f	&operator&=(const Mask_f &b) {
				(*this) = (*this)&b;
				return	(*this);
			}

			inline	Mask_f	 operator| (const Mask_f &b) {
				return	opCode(kor,  this->data, b.data);
			}

			inline	Mask_f	&operator|=(const Mask_f &b) {
				(*this) = (*this)|b;
				return	(*this);
			}

			inline	Mask_f	 operator! () {
				return	opCode(kxor, this->data, 0b1111111111111111);
			}

			inline	int	Count() {
				int count = 0;

				for (int i=0; i<16; i++)
					count += (data>>i)&1;

				return	count;
			}

			void	Print(const char *str)	{
				printf("%s %x", str, this->data);
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
                	static constexpr size_t tWide = 2;

			typedef float sData;
			typedef Mask_f Mask;

				Simd_f() {
				data = opCode(setzero_ps);
			}

				Simd_f(float x0, float x1, float x2,  float x3,  float x4,  float x5,  float x6,  float x7,
				       float x8, float x9, float x10, float x11, float x12, float x13, float x14, float x15) {
				data = opCode(set_ps, x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1, x0);
			}

				Simd_f(uint x0, uint x1, uint x2, uint x3) {
				data = opCode(castsi512_ps, opCode(set_epi32, x3, x2, x1, x0, x3, x2, x1, x0, x3, x2, x1, x0, x3, x2, x1, x0));
			}

				Simd_f(float x0, float x1) {
				data = opCode(set_ps, x1, x0, x1, x0, x1, x0, x1, x0, x1, x0, x1, x0, x1, x0, x1, x0);
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
				opCode(mask_store_ps, mAddr, msk.data, this->data);
			}

			void	Stream (float *memAddress) {
				float * __restrict__ mAddr = (float * __restrict__) __builtin_assume_aligned (memAddress, Simd::sAlign);
				opCode(stream_ps, mAddr, this->data);
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
				return	opCode(mul_ps, this->data, opCode(set_ps, -1., 0., -1., 0., -1., 0., -1., 0., -1., 0., -1., 0., -1., 0., -1., 0.));
			}

			inline	Simd_f	operator^(const Mask_f &msk) {
				return	opCode(maskz_mov_ps, msk.data, this->data);
			}

			inline	Simd_f	operator&(const Simd_f &b) {
				return	opCode(castsi512_ps, opCode(and_si512, opCode(castps_si512, this->data), opCode(castps_si512, b.data)));
			}

			inline	Simd_f	operator|(const Simd_f &b) {
				return	opCode(castsi512_ps, opCode(or_si512,  opCode(castps_si512, this->data), opCode(castps_si512, b.data)));
			}

			inline	Simd_f	operator^(const Simd_f &b) {
				return	opCode(castsi512_ps, opCode(xor_si512, opCode(castps_si512, this->data), opCode(castps_si512, b.data)));
			}

			inline	Simd_f	operator>>(uint i) {
				return	opCode(castsi512_ps, opCode(srli_epi32, opCode(castps_si512, this->data), i));
			}

			inline	Simd_f	operator<<(uint i) {
				return	opCode(castsi512_ps, opCode(slli_epi32, opCode(castps_si512, this->data), i));
			}

			inline	Simd_f	grShift(uint i) {
			#ifdef	__AVX512BW__
				return	opCode(castsi512_ps, opCode(bsrli_epi128, opCode(castps_si512, this->data), i));
			#else
				_MHnt_	low  = opCode(extracti64x4_epi64, this->data, 0);
				_MHnt_	high = opCode(extracti64x4_epi64, this->data, 1);
				return	opCode(castsi512_ps, opCode(inserti64x4, opCode(castsi256_si512, opCodl(bsrli_epi128, low, i)), opCodl(bsrli_epi128, high, i), 1));
			#endif
			}

			inline	Simd_f	glShift(uint i) {
			#ifdef	__AVX512BW__
				return	opCode(castsi512_ps, opCode(bslli_epi128, opCode(castps_si512, this->data), i));
			#else
				_MHnt_	low  = opCode(extracti64x4_epi64, this->data, 0);
				_MHnt_	high = opCode(extracti64x4_epi64, this->data, 1);
				return	opCode(castsi512_ps, opCode(inserti64x4, opCode(castsi256_si512, opCodl(bslli_epi128, low, i)), opCodl(bslli_epi128, high, i), 1));
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
				return	opCode(cmp_ps_mask, this->data, b.data, _CMP_GT_OQ);
			}

			inline	Mask_f	operator>=(const Simd_f &b) {
				return	opCode(cmp_ps_mask, this->data, b.data, _CMP_GE_OQ);
			}

			inline	Mask_f	operator<(const Simd_f &b) {
				return	opCode(cmp_ps_mask, this->data, b.data, _CMP_LT_OQ);
			}

			inline	Mask_f	operator<=(const Simd_f &b) {
				return	opCode(cmp_ps_mask, this->data, b.data, _CMP_LE_OQ);
			}

			inline	Mask_f	operator==(const Simd_f &b) {
				return	opCode(cmp_ps_mask, this->data, b.data, _CMP_EQ_UQ);
			}

			inline	Simd_f	fma(const Simd_f &a, const Simd_f &b) {
				return	opCode(fmadd_ps, this->data, a.data, b.data);
			}

			inline	Simd_f	fms(const Simd_f &a, const Simd_f &b) {
				return	opCode(fmsub_ps, this->data, a.data, b.data);
			}

		//	inline	Simd_f	xPermute () {	//?????
		//		return	opCode(shuffle_f32x4, this->data, this->data, 0b01001110);
		//	}

			inline	Simd_f	xPermute () {	//?????
				return	opCode(shuffle_f32x4, this->data, this->data, 0b10110001);
			}

			inline	Simd_f	yPermute () {
				return	opCode(permute_ps, this->data, 0b01001110);
			}

			inline	Simd_f	zPermute () {
				return	opCode(permute_ps, this->data, 0b10110001);
			}

			inline	Simd_f	rPermute () {
				return	opCode(permute_ps, this->data, 0b00011011);
			}

			inline	Simd_f	xPermCpx () {
				return	opCode(shuffle_f32x4, this->data, this->data, 0b10110001);
			}

			inline	Simd_f	yPermCpx () {
				return	opCode(permute_ps, this->data, 0b01001110);
			}

			inline	Simd_f	zPermCpx () {
				return	opCode(permute_ps, this->data, 0b10110001);
			}

			inline	void	SetRandom ();// {
//				(*this) = Simd_f(Su2Rand::genRand(), Su2Rand::genRand(), Su2Rand::genRand(), Su2Rand::genRand(),
//						 Su2Rand::genRand(), Su2Rand::genRand(), Su2Rand::genRand(), Su2Rand::genRand(),
////						 Su2Rand::genRand(), Su2Rand::genRand(), Su2Rand::genRand(), Su2Rand::genRand(),
//						 Su2Rand::genRand(), Su2Rand::genRand(), Su2Rand::genRand(), Su2Rand::genRand());
//			}

			inline	float	Sum () {
				return	opCode(reduce_add_ps, (*this).data);
			}

			inline	float&	operator[](int lane) {
				return	data[lane];
			}

			inline	_MData_&	raw() {
				return	data;
			}

			void	iPrint(const char *str)	{
				printuVar(opCode(castps_si512, this->data), str);
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

			__mmask8	data;

			public:

				Mask_d()			: data(0) {}
				Mask_d(const __mmask8 &in) 	: data(in){}


			inline	Mask_d	 operator& (const Mask_d &b) {
				return	(__mmask8) (opCode(kand, this->data, b.data) & 0b0000000011111111);
			}

			inline	Mask_d	&operator&=(const Mask_d &b) {
				(*this) = (*this)&b;
				return	(*this);
			}

			inline	Mask_d	 operator| (const Mask_d &b) {
				return	(__mmask8) (opCode(kor,  this->data, b.data) & 0b0000000011111111);
			}

			inline	Mask_d	&operator|=(const Mask_d &b) {
				(*this) = (*this)|b;
				return	(*this);
			}

			inline	Mask_d	 operator! () {
				return	(__mmask8) (opCode(kxor, this->data, 0b0000000011111111) & 0b0000000011111111);
			}

			inline	int	Count() {
				int count = 0;

				for (int i=0; i<8; i++)
					count += (data>>i)&1;

				return	count;
			}

			void	Print(const char *str)	{
				printf("%s %x", str, this->data);
			}

			friend	class	Simd_d;
		};

		class	Simd_d	{
			private:

			_MDatd_	data;

			public:

                	static constexpr size_t sWide = sAlign/sizeof(double);
                	static constexpr size_t xWide = 2;
                	static constexpr size_t yWide = 2;
                	static constexpr size_t zWide = 2;

                	static constexpr size_t sWdCx = sAlign/sizeof(double);
                	static constexpr size_t xWdCx = 1;
                	static constexpr size_t yWdCx = 2;
                	static constexpr size_t zWdCx = 2;

			typedef double sData;
			typedef Mask_d Mask;

				Simd_d() {
				data = opCode(setzero_pd);
			}

				Simd_d(double x0, double x1, double x2, double x3, double x4, double x5, double x6, double x7) {
				data = opCode(set_pd, x7, x6, x5, x4, x3, x2, x1, x0);
			}

				Simd_d(uint64 x0, uint64 x1) {
				data = opCode(castsi512_pd, opCode(set_epi64, x1, x0, x1, x0, x1, x0, x1, x0));
			}

				Simd_d(double x0, double x1) {
				data = opCode(set_pd, x1, x0, x1, x0, x1, x0, x1, x0);
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
				opCode(mask_store_pd, memAddress, msk.data, this->data);
			}

			void	Save  (double *memAddress) {
				opCode(store_pd,  memAddress, this->data);
			}

			void	Stream (double *memAddress) {
				opCode(stream_pd, memAddress, this->data);
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
				return	opCode(add_pd, opCode(permute_pd, this->data, 0b01010101), this->data);
			}

			inline	Simd_d	operator~() {
				return	opCode(mul_pd, this->data, opCode(set_pd, -1., 0., -1., 0., -1., 0., -1., 0.));
			}

			inline	Simd_d	operator^(const Mask_d &msk) {
				return	opCode(maskz_mov_pd, msk.data, this->data);
			}

			inline	Simd_d	operator&(const Simd_d &x) {
				return	opCode(castsi512_pd, opCode(and_si512, opCode(castpd_si512, this->data), opCode(castpd_si512, x.data)));
			}

			inline	Simd_d	operator|(const Simd_d &x) {
				return	opCode(castsi512_pd, opCode(or_si512,  opCode(castpd_si512, this->data), opCode(castpd_si512, x.data)));
			}

			inline	Simd_d	operator^(const Simd_d &x) {
				return	opCode(castsi512_pd, opCode(xor_si512, opCode(castpd_si512, this->data), opCode(castpd_si512, x.data)));
			}

			inline	Simd_d	operator>>(uint i) {
				return	opCode(castsi512_pd, opCode(srli_epi64, opCode(castpd_si512, this->data), i));
			}

			inline	Simd_d	operator<<(uint i) {
				return	opCode(castsi512_pd, opCode(slli_epi64, opCode(castpd_si512, this->data), i));
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
				return	opCode(cmp_pd_mask, this->data, b.data, _CMP_GT_OQ);
			}

			inline	Mask_d	operator>=(const Simd_d &b) {
				return	opCode(cmp_pd_mask, this->data, b.data, _CMP_GE_OQ);
			}

			inline	Mask_d	operator<(const Simd_d &b) {
				return	opCode(cmp_pd_mask, this->data, b.data, _CMP_LT_OQ);
			}

			inline	Mask_d	operator<=(const Simd_d &b) {
				return	opCode(cmp_pd_mask, this->data, b.data, _CMP_LE_OQ);
			}

			inline	Mask_d	operator==(const Simd_d &b) {
				return	opCode(cmp_pd_mask, this->data, b.data, _CMP_EQ_UQ);
			}

			inline	Simd_d	fma(const Simd_d &a, const Simd_d &b) {
				return	opCode(fmadd_pd, this->data, a.data, b.data);
			}

			inline	Simd_d	fms(const Simd_d &a, const Simd_d &b) {
				return	opCode(fmsub_pd, this->data, a.data, b.data);
			}

			inline	Simd_d	xPermute () {
				return	opCode(shuffle_f64x2, this->data, this->data, 0b01001110);
			}

			inline	Simd_d	yPermute () {
				return	opCode(shuffle_f64x2, this->data, this->data, 0b10110001);
			}

			inline	Simd_d	zPermute () {
				return	opCode(permute_pd, this->data, 0b01010101);
			}

			inline	Simd_d	rPermute () {
				return	opCode(castps_pd, opCode(permute_ps, opCode(castpd_ps, this->data), 0b00011011));
			}

			inline	Simd_d	xPermCpx () {
				return	(*this);
			}

			inline	Simd_d	yPermCpx () {
				return	opCode(shuffle_f64x2, this->data, this->data, 0b10110001);
			}

			inline	Simd_d	zPermCpx () {
				return	opCode(permute_pd, this->data, 0b01010101);
			}

			inline	void	SetRandom ();// {
//				(*this) = Simd_d(Su2Rand::genRand(), Su2Rand::genRand(), Su2Rand::genRand(), Su2Rand::genRand(),
//						 Su2Rand::genRand(), Su2Rand::genRand(), Su2Rand::genRand(), Su2Rand::genRand());
//			}

			inline	double	Sum () {
				return	opCode(reduce_add_pd, (*this).data);
			}

			inline	double&	operator[](int lane) {
				return	data[lane];
			}

			inline	_MDatd_&	raw() {
				return	data;
			}

			void	iPrint(const char *str)	{
				printlVar(opCode(castpd_si512, this->data), str);
			}

			void	Print(const char *str)	{
				printdVar(this->data, str);
			}

			friend	class	Mask_d;

			friend  Simd_d  sqrt	(const Simd_d&);
			friend  Simd_d  cos	(const Simd_d&);
			friend  Simd_d  sin	(const Simd_d&);
			friend  Simd_d  log	(const Simd_d&);
			friend  Simd_d  exp	(const Simd_d&);
			friend  Simd_d  abs	(const Simd_d&);
		};

		Simd_d	sqrt	(const Simd_d&);
		Simd_d	cos	(const Simd_d&);
		Simd_d	sin	(const Simd_d&);
		Simd_d	log	(const Simd_d&);
		Simd_d	exp	(const Simd_d&);
		Simd_d	abs	(const Simd_d&);
	}
#endif
