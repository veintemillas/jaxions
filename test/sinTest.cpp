#include<cstdio>
#include<cmath>
#include<chrono>
#include<string>
#include"utils/triSimd.h"

#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

#if	defined(__AVX512F__)
	#define	_PREFIX_ _mm512
#else
	#if not defined(__AVX__) and not defined(__AVX2__)
		#define	_PREFIX_ _mm
	#else
		#define	_PREFIX_ _mm256
	#endif
#endif


#if	defined(__AVX512F__)
	#define	_MDatd_ __m512d
	#define	_MData_ __m512
	#define step 16
	#define stpd 8
#elif	defined(__AVX__)
	#define	_MDatd_ __m256d
	#define	_MData_ __m256
	#define step 8
	#define stpd 4
#else
	#define	_MDatd_ __m128d
	#define	_MData_ __m128
	#define step 4
	#define stpd 2
#endif 

#define	KMAX 10000
#define	Tol  1.0e-6
#define	Tld  1.0e-4

int	main(int argc, char *argv[])
{
	std::chrono::high_resolution_clock::time_point	sTime;
	double	vTime = 0., cTime = 0.;
	float x, sX = 0.f*3.14159f, worse;

#if	defined(__AVX512F__)
	std::string name("Avx512f");
#elif	defined(__AVX2__)
	std::string name("Avx2");
#elif	defined(__AVX__)
	std::string name("Avx");
#else
	std::string name("Sse4.1");
#endif

#ifdef	__FMA__
	name.append(" + Fma");
#endif
	x = sX; worse = 0.;

	for (int k=0; k<KMAX; k++)
	{
#if	defined(__AVX512F__)
#elif	defined(__AVX__)
		_MData_	v = { x, x+0.01f, x+0.02f, x+0.03f, x+0.04f, x+0.05f, x+0.06f, x+0.07f };
#else
		_MData_	v = { x, x+0.01f, x+0.02f, x+0.03f };
#endif
	//	auto sCep = opCode(sin1_ps, v);
		sTime = std::chrono::high_resolution_clock::now();
		auto sSlf = opCode(sin_ps,  v);
		vTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - sTime).count()*1e-6;



		for (int i = 0; i < step; i++) {
			float y = x + ((float) i)*0.01f;
			sTime = std::chrono::high_resolution_clock::now();
			float res = sin(y);
			cTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - sTime).count()*1e-6;

			auto dif = fabsf(res - sSlf[i]);

			if (dif > Tol) {
				printf("Cagada %.4e Sleef %f vs Cmath %f (dif %e)\n", y, sSlf[i], res, dif);
				fflush(stdout);
			}

			if (dif > worse)
				worse = dif;
		}
		x += 0.01f * step;
	}

	printf ("Total sin time %d evaluations ==> %s %lf vs CMath %lf\nMax argument %lf\nWorse precision %e\nSpeedup %.2lfx\n", KMAX, name.c_str(), vTime, cTime, x, worse, cTime/vTime);

	vTime = 0., cTime = 0.;
	x = sX; worse = 0.;

	for (int k=0; k<KMAX; k++)
	{
#if	defined(__AVX512F__)
#elif	defined(__AVX__)
		_MData_	v = { x, x+0.01f, x+0.02f, x+0.03f, x+0.04f, x+0.05f, x+0.06f, x+0.07f };
#else
		_MData_	v = { x, x+0.01f, x+0.02f, x+0.03f };
#endif
	//	auto sCep = opCode(sin1_ps, v);
		sTime = std::chrono::high_resolution_clock::now();
		auto cSlf = opCode(cos_ps,  v);
		vTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - sTime).count()*1e-6;

		for (int i = 0; i < step; i++) {
			float y = x + ((float) i)*0.01f;
			sTime = std::chrono::high_resolution_clock::now();
			float res = cos(y);
			cTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - sTime).count()*1e-6;

			auto dif = fabsf(res - cSlf[i]);

			if (dif > Tol) {
				printf("Cagada %.4e Sleef %f vs Cmath %f (dif %e)\n", y, cSlf[i], res, dif);
				fflush(stdout);
			}

			if (dif > worse)
				worse = dif;
		}
		x += 0.01f * step;
	}

	printf ("Total cos time %d evaluations ==> %s %lf vs CMath %lf\nMax argument %lf\nWorse precision %e\nSpeedup %.2lfx\n", KMAX, name.c_str(), vTime, cTime, x, worse, cTime/vTime);

	double y, sY = 0.0*3.14159, dWorse;

	y = sY; dWorse = 0.;

	for (int k=0; k<KMAX; k++)
	{
#if	defined(__AVX512F__)
#elif	defined(__AVX__)
		_MDatd_	v = { y, y+0.01, y+0.02, y+0.03 };
#else
		_MDatd_	v = { y, y+0.01 };
#endif
		sTime = std::chrono::high_resolution_clock::now();
		auto sSlf = opCode(sin_pd,  v);
		vTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - sTime).count()*1e-6;

		for (int i = 0; i < stpd; i++) {
			double z = y + ((double) i)*0.01;
			sTime = std::chrono::high_resolution_clock::now();
			double res = sin(z);
			cTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - sTime).count()*1e-6;

			auto dif = fabs(res - sSlf[i]);

			if (dif > Tld) {
				printf("Cagada %.4e Sleef %f vs Cmath %f (dif %e)\n", z, sSlf[i], res, dif);
				fflush(stdout);
			}

			if (dif > dWorse)
				dWorse = dif;
		}
		y += 0.01 * stpd;
	}

	printf ("Total sin time %d evaluations ==> %s %lf vs CMath %lf\nMax argument %lf\nWorse precision %e\nSpeedup %.2lfx\n", KMAX, name.c_str(), vTime, cTime, y, dWorse, cTime/vTime);

	vTime = 0., cTime = 0.;
	y = sY; dWorse = 0.;

	for (int k=0; k<KMAX; k++)
	{
#if	defined(__AVX512F__)
#elif	defined(__AVX__)
		_MDatd_	v = { y, y+0.01, y+0.02, y+0.03 };
#else
		_MDatd_	v = { y, y+0.01 };
#endif
		sTime = std::chrono::high_resolution_clock::now();
		auto cSlf = opCode(cos_pd,  v);
		vTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - sTime).count()*1e-6;

		for (int i = 0; i < stpd; i++) {
			double z = y + ((double) i)*0.01;
			sTime = std::chrono::high_resolution_clock::now();
			double res = cos(z);
			cTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - sTime).count()*1e-6;

			auto dif = fabs(res - cSlf[i]);

			if (dif > 0.) {//Tld) {
				printf("Cagada %.4e Sleef %f vs Cmath %f (dif %e)\n", z, cSlf[i], res, dif);
				fflush(stdout);
			}
			if (dif > dWorse)
				dWorse = dif;
		}
		y += 0.01 * stpd;
	}

	printf ("Total cos time %d evaluations ==> %s %lf vs CMath %lf\nMax argument %lf\nWorse precision %e\nSpeedup %.2lfx\n", KMAX, name.c_str(), vTime, cTime, y, dWorse, cTime/vTime);

	return	0;
} 
		
