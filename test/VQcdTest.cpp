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

#define	KMAX 100000
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

	printf ("Single precision test\n");
	printf ("Sine\n");

	for (int k=0; k<KMAX; k++)
	{
#if	defined(__AVX512F__)
		_MData_	v = { x, x+0.01f, x+0.02f, x+0.03f, x+0.04f, x+0.05f, x+0.06f, x+0.07f, x+0.08f, x+0.09f, x+0.10f, x+0.11f, x+0.12f, x+0.13f, x+0.14f, x+0.15f };
#elif	defined(__AVX__)
		_MData_	v = { x, x+0.01f, x+0.02f, x+0.03f, x+0.04f, x+0.05f, x+0.06f, x+0.07f };
#else
		_MData_	v = { x, x+0.01f, x+0.02f, x+0.03f };
#endif
		sTime = std::chrono::high_resolution_clock::now();
		auto sSlf = opCode(vqcd0_ps,  v);
		vTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - sTime).count()*1e-6;

		for (int i = 0; i < step; i+=2) {
			float cx = x + ((float) (i+0))*0.01f;
			float cy = x + ((float) (i+1))*0.01f;
			sTime = std::chrono::high_resolution_clock::now();
			float rx = cy*cy;
			float ry = cx*cy;
			cTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - sTime).count()*1e-6;

			auto dx = fabsf(rx - sSlf[i+0]);
			auto dy = fabsf(ry - sSlf[i+1]);

	//		if (dif > Tol) {
	//			printf("Cagada %.4e Sleef %f vs Cmath %f (dif %e)\n", y, sSlf[i], res, dif);
	//			fflush(stdout);
	//		}

			auto dif = sqrt(dx*dx+dy*dy);
			if (dif > worse) {
				printf ("%e %e <==> %e %e | %e %e <==> %e %e\n", cx, cy, v[i], v[i+1], rx, ry, sSlf[i], sSlf[i+1]);
				worse = dif;
			}
		}
		x += 0.01f * step;
	}

	printf ("Total VQcd0 time %d evaluations ==> %s %lf vs CMath %lf\nMax argument %lf\nWorse precision %e\nSpeedup %.2lfx\n", KMAX, name.c_str(), vTime, cTime, x, worse, cTime/vTime);
	fflush (stdout);

	double y, sY = 0.0*3.14159, dWorse;

	y = sY; dWorse = 0.;

	printf ("Double precision test\n");
	printf ("Sine\n");

	for (int k=0; k<KMAX; k++)
	{
#if	defined(__AVX512F__)
		_MDatd_	v = { y, y+0.01, y+0.02, y+0.03, y+0.04, y+0.05, y+0.06, y+0.07 };
#elif	defined(__AVX__)
		_MDatd_	v = { y, y+0.01, y+0.02, y+0.03 };
#else
		_MDatd_	v = { y, y+0.01 };
#endif
		sTime = std::chrono::high_resolution_clock::now();
		auto sSlf = opCode(vqcd0_pd,  v);
		vTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - sTime).count()*1e-6;

		for (int i = 0; i < stpd; i++) {
			double cx = y + ((double) (i+0))*0.01;
			double cy = y + ((double) (i+1))*0.01;
			sTime = std::chrono::high_resolution_clock::now();
			double rx = cy*cy;
			double ry = cx*cy;
			cTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - sTime).count()*1e-6;

			auto dx = fabs(rx - sSlf[i+0]);
			auto dy = fabs(ry - sSlf[i+1]);

			//if (dif > Tld) {
			//	printf("Cagada %.4e Sleef %f vs Cmath %f (dif %e)\n", z, sSlf[i], res, dif);
			//	fflush(stdout);
			//}

			auto dif = sqrt(dx*dx+dy*dy);
			if (dif > dWorse) {
				printf ("%e %e <==> %e %e | %e %e <==> %e %e\n", cx, cy, v[i], v[i+1], rx, ry, sSlf[i], sSlf[i+1]);
				dWorse = dif;
			}
		}
		y += 0.01 * stpd;
	}

	printf ("Total VQcd time %d evaluations ==> %s %lf vs CMath %lf\nMax argument %lf\nWorse precision %e\nSpeedup %.2lfx\n", KMAX, name.c_str(), vTime, cTime, y, dWorse, cTime/vTime);

	return	0;
} 
		
