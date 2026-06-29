#ifndef	_ANYSTRING_SCALAR_
#define	_ANYSTRING_SCALAR_

	#include "scalar/scalarField.h"

	void	anystringConf (Scalar *field, IcData ic, double *x, double *y, double *z, size_t len, int *eps, size_t eps_len);

	#include<cstdio>
	#include<cmath>
	#include"scalar/scalarField.h"
	#include"enum-field.h"
	//#include"scalar/varNQCD.h"

	#include "utils/triSimd.h"
	#include "utils/parse.h"

	#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
	#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
	#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

	#include <immintrin.h>

	#if	defined(__AVX512F__)
		#define	Align 64
		#define	_PREFIX_ _mm512
		#define	_MInt_  __m512i
	#else
		#if not defined(__AVX__) and not defined(__AVX2__)
			#define	Align 16
			#define	_PREFIX_ _mm
		#else
			#define	Align 32
			#define	_PREFIX_ _mm256
		#endif
	#endif


	/*
		This function calculates complex theta in m2 calculated from the slid angle
		formula in axionnotes, anystring initial conditions

		xs, ys, zs are a list of coordinates of string segments

	*/

	inline	void	cthetaSolidAngleXeon(void * __restrict__ m2_, const size_t Lx, const size_t Lz, const size_t dLz, const size_t Tz, FieldPrecision precision,
												size_t len, double* xs, double* ys, double* zs, int *eps, int p_copies_dipole=0, int Bext=0, double Bamp=0.0)
	{

  /* com calculation */

	/* We calculate COM */

	double xc=0.0 ,yc=0.0 ,zc=0.0;
	for (uint i = 0; i < len; i++) {
		if (eps[i])
			continue;

		xc += xs[i];
		yc += ys[i];
		zc += zs[i];
	}
	xc /= (double)len;
	yc /= (double)len;
	zc /= (double)len;

	xc = floor(xc) + 0.5;
	yc = floor(yc) + 0.5;
	zc = floor(zc) + 0.5;

	LogMsg(VERB_NORMAL,"[theta3] COM (cube re-centered) %.f %.f %.f",xc,yc,zc);

	/* We calculate Vector Area */

	double Ax=0.0 ,Ay=0.0 ,Az=0.0;
	double length=0.0;
	for (uint i = 0; i < len; i++) {
		if (eps[i])
			continue;
		Ax += ys[i]*(zs[i+1]-zs[i])-zs[i]*(ys[i+1]-ys[i]);
		Ay += zs[i]*(xs[i+1]-xs[i])-xs[i]*(zs[i+1]-zs[i]);
		Az += xs[i]*(ys[i+1]-ys[i])-ys[i]*(xs[i+1]-xs[i]);
		length += std::sqrt( pow(zs[i+1]-zs[i],2)+pow(ys[i+1]-ys[i],2)+pow(xs[i+1]-xs[i],2));
	}
	Ax/=2;
	Ay/=2;
	Az/=2;
	LogMsg(VERB_NORMAL,"[theta3] Area %lf %lf %lf",Ax,Ay,Az);
	LogMsg(VERB_NORMAL,"[theta3] Length %lf",length);

	/* this factor of two is for the contribution to theta*/
	Ax /= 2.0;
	Ay /= 2.0;
	Az /= 2.0;


		if (precision == FIELD_DOUBLE)
		{
	#if	defined(__AVX512F__)
		#define	_MData_ __m512d
		#define	step 8
	#elif	defined(__AVX__)
		#define	_MData_ __m256d
		#define	step 4
	#else
		#define	_MData_ __m128d
		#define	step 2
	#endif

			double * __restrict__ m2	= (double * __restrict__) __builtin_assume_aligned (m2_, Align);

	#if	defined(__AVX512F__)
			const double __attribute__((aligned(Align))) Xinvec[8]  = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
	#elif	defined(__AVX__)
			const double __attribute__((aligned(Align))) Xinvec[4]  = { 0.0, 1.0, 2.0, 3.0 };
	#else
			const double __attribute__((aligned(Align))) Xinvec[2]  = { 0.0, 1.0};
	#endif

			const _MData_ XIN  = opCode(load_pd, Xinvec);

			const _MData_ XCOM  = opCode(set1_pd, xc);
			const _MData_ YCOM  = opCode(set1_pd, yc);
			const _MData_ ZCOM  = opCode(set1_pd, zc);

			const _MData_ AX  = opCode(set1_pd, Ax);
			const _MData_ AY  = opCode(set1_pd, Ay);
			const _MData_ AZ  = opCode(set1_pd, Az);

			auto cosi = [] (_MData_ x,_MData_ y1,_MData_ z1,_MData_ y2,_MData_ z2)
			{
				return opCode(mul_pd,x,opCode(sub_pd, opCode(mul_pd,y1,z2), opCode(mul_pd,y2,z1)));
			};

			auto scaprod = [](_MData_ x1,_MData_ y1,_MData_ z1,_MData_ x2,_MData_ y2,_MData_ z2)
			{
				return opCode(add_pd,opCode(add_pd,
					opCode(mul_pd,x1,x2),opCode(mul_pd,y1,y2)),opCode(mul_pd,z1,z2));
			};

			#pragma omp parallel default(shared)
			{
					_MData_ X,Y,Z,XS1,XS2,YS1,YS2,ZS1,ZS2,X3,Y3,Z3,N1,N2,N3;
					_MData_ AA,TH,TT,DD;

					const _MData_ ZERO = opCode(set1_pd, 0.0);
					const _MData_ ONE  = opCode(set1_pd, 1.0);

					#pragma omp for schedule(static)
			    for (size_t zz = 0; zz < Lz; zz++) {
						Z = opCode(set1_pd, (double) (dLz+zz));
			     for (size_t yy = 0; yy < Lx; yy++) {
						 Y = opCode(set1_pd, (double) yy);   // y positions
			      for (size_t xx = 0; xx < Lx; xx += step) {
							X = opCode(set1_pd, (double) xx) + XIN;           // x positions

					size_t idx = xx + Lx*(yy + Lx*zz);

					/* vector x */

					X3 = opCode(sub_pd,XCOM,X); // this is a true vector
					Y3 = opCode(sub_pd,YCOM,Y); // calculate once, load at end?
					Z3 = opCode(sub_pd,ZCOM,Z); // calculate once, load at end?
					N3 = opCode(sqrt_pd,scaprod(X3,Y3,Z3,X3,Y3,Z3));
					X3 = opCode(div_pd,X3,N3);
					Y3 = opCode(div_pd,Y3,N3);
					Z3 = opCode(div_pd,Z3,N3);
					/* loop over segments*/

					XS1 = opCode(sub_pd,opCode(set1_pd, xs[0]),X); // vec
					YS1 = opCode(sub_pd,opCode(set1_pd, ys[0]),Y) ; // copies
					ZS1 = opCode(sub_pd,opCode(set1_pd, zs[0]),Z); // copies
					N1  = opCode(sqrt_pd,scaprod(XS1,YS1,ZS1,XS1,YS1,ZS1));
					XS1 = opCode(div_pd,XS1,N1); // vec
					YS1 = opCode(div_pd,YS1,N1); // copies
					ZS1 = opCode(div_pd,ZS1,N1); // copies


					// complex accumulator for exp(iθ) ---
					_MData_ Xacc = ONE;
					_MData_ Yacc = ZERO;

					const int K = 4; // rescale cadence
					int kblk = 0;

					for(uint i=0; i<len-1; i++){         //1 segment less than points

						if (eps[i]>0)
							continue;

						XS2 = opCode(sub_pd,opCode(set1_pd, xs[i+1]),X); // vec
						YS2 = opCode(sub_pd,opCode(set1_pd, ys[i+1]),Y); // copies
						ZS2 = opCode(sub_pd,opCode(set1_pd, zs[i+1]),Z); // copies
						N2  = opCode(sqrt_pd,scaprod(XS2,YS2,ZS2,XS2,YS2,ZS2));
						XS2 = opCode(div_pd,XS2,N2); // vec
						YS2 = opCode(div_pd,YS2,N2); // copies
						ZS2 = opCode(div_pd,ZS2,N2); // copies

						// NUMERATOR R1.(R2xR3)
						AA = cosi(X3,YS1,ZS1,YS2,ZS2);
						AA = opCode(add_pd,AA,cosi(Y3,XS2,ZS2,XS1,ZS1));
						TT = opCode(add_pd,AA,cosi(Z3,XS1,YS1,XS2,YS2));

						// DENOMINATOR
						AA = ONE;
						AA = opCode(add_pd,AA,scaprod(XS2,YS2,ZS2,X3,Y3,Z3));
						AA = opCode(add_pd,AA,scaprod(XS1,YS1,ZS1,X3,Y3,Z3));
						AA = opCode(add_pd,AA,scaprod(XS2,YS2,ZS2,XS1,YS1,ZS1));

						// --- COMPLEX MULTIPLY: (Xacc + iYacc)*(D + iT) ---
						#if defined(__FMA__)
								_MData_ Xn = opCode(fmadd_pd, Xacc, AA, opCode(mul_pd, Yacc, opCode(sub_pd, ZERO, TT))); // X*D - Y*T
								_MData_ Yn = opCode(fmadd_pd, Xacc, TT, opCode(mul_pd, Yacc, AA));                        // X*T + Y*D
						#else
								_MData_ Xn = opCode(sub_pd, opCode(mul_pd,Xacc,AA), opCode(mul_pd,Yacc,TT));
								_MData_ Yn = opCode(add_pd, opCode(mul_pd,Xacc,TT), opCode(mul_pd,Yacc,AA));
						#endif
						Xacc = Xn; Yacc = Yn;

						// ---- simple rescale every K segments: make |acc|=1 using rsqrt ----
						if ((++kblk & (K-1)) == 0) {
							_MData_ mag2 = opCode(add_pd, opCode(mul_pd,Xacc,Xacc), opCode(mul_pd,Yacc,Yacc));
							_MData_ rinv = opCode(rsqrt_pd, mag2);
							Xacc = opCode(mul_pd, Xacc, rinv);
							Yacc = opCode(mul_pd, Yacc, rinv);
						}

						/* load next */
						XS1 = XS2;
						YS1 = YS2;
						ZS1 = ZS2;
						N1  = N2;
					}// end string segment sum

					if (p_copies_dipole>0)
					{
						TH = ZERO;
						for (int iz=-p_copies_dipole;iz<p_copies_dipole+1;iz++)
							for (int iy=-p_copies_dipole;iy<p_copies_dipole+1;iy++)
								for (int ix=-p_copies_dipole;ix<p_copies_dipole+1;ix++){
									if ((ix==0) && (iy==0) && (iz==0))
										continue;
									XS1 = opCode(add_pd,  X3, opCode(set1_pd, (float) (ix*Lx) ));
									YS1 = opCode(set1_pd, yc + ((double) iy*Lx - yy));
									ZS1 = opCode(set1_pd, zc + ((double) iz*Tz - zz - dLz));
									// A.R
									AA = opCode(add_pd,
												opCode(add_pd,opCode(mul_pd,AX,XS1),opCode(mul_pd,AY,YS1)),
													opCode(mul_pd,AZ,ZS1));
									// R^3
									TT = scaprod(XS1,YS1,ZS1,XS1,YS1,ZS1);
									DD = opCode(sqrt_pd,TT);
									DD = opCode(mul_pd,DD,TT); // R3
									TH = opCode(add_pd,TH,opCode(div_pd,AA,DD));
					}
					AA = opCode(cos_pd, TH);
					TT = opCode(sin_pd, TH);
					XS1 = opCode(sub_pd, opCode(mul_pd, Xacc, AA), opCode(mul_pd, Yacc, TT));
					Yacc = opCode(add_pd, opCode(mul_pd, Xacc, TT), opCode(mul_pd, Yacc, AA));
					Xacc = XS1;
					}

					if (Bext != 0)
					{
						double phase = Bamp*sin(Bext*6.2831853071795864*((double) yy)/((double)Lx));
						_MData_ TH = opCode(set1_pd, phase);
						AA = opCode(cos_pd, TH);
						TT = opCode(sin_pd, TH);
						XS1 = opCode(sub_pd, opCode(mul_pd, Xacc, AA), opCode(mul_pd, Yacc, TT));
						Yacc = opCode(add_pd, opCode(mul_pd, Xacc, TT), opCode(mul_pd, Yacc, AA));
						Xacc = XS1;
					}
					// Store Real and Imaginary
					{
			    _MData_ mag2 = opCode(add_pd, opCode(mul_pd,Xacc,Xacc), opCode(mul_pd,Yacc,Yacc));
			    _MData_ rinv = opCode(rsqrt_pd, mag2);
			    Xacc = opCode(mul_pd, Xacc, rinv);
			    Yacc = opCode(mul_pd, Yacc, rinv);

			    // If your complex field is interleaved float* (Re,Im,Re,Im,...):
			    // Store Re lanes then Im lanes interleaved per element:
			    // Option A (scalar scatter per lane is simplest to drop in)
			    double tmpRe[step], tmpIm[step];
			    opCode(store_pd, tmpRe, Xacc);
			    opCode(store_pd, tmpIm, Yacc);
			    for (int lane = 0; lane < (int)step; ++lane) {
						if (std::isnan(tmpRe[lane])){
							// nans ocurr when x coincides with string, where theta irrelevant
							m2[2*(idx + lane) + 0] = -1.0;
							m2[2*(idx + lane) + 1] = 0.0;
						}
						else {
			        m2[2*(idx + lane) + 0] = tmpRe[lane];
			        m2[2*(idx + lane) + 1] = tmpIm[lane];
						}
			    }
				} // end store


			      } //endloopx
			    } //endloopy
			  }//endloopz
			}//end parallel

	#undef	_MData_
	#undef	step

		} else if (precision == FIELD_SINGLE) {

	#if	defined(__AVX512F__)
		#define	_MData_ __m512
		#define	step 16
	#elif	defined(__AVX__)
		#define	_MData_ __m256
		#define	step 8
	#else
		#define	_MData_ __m128
		#define	step 4
	#endif


			std::vector<double> xsf,ysf,zsf;
			for (size_t i=0;i<len;i++){
				xsf.push_back(xs[i]);
				ysf.push_back(ys[i]);
				zsf.push_back(zs[i]);
			}

			float * __restrict__ m2		= (float * __restrict__) __builtin_assume_aligned (m2_, Align);

	#if	defined(__AVX512F__)
			const float __attribute__((aligned(Align))) Xinvec[16] = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f };
	#elif	defined(__AVX__)
			const float __attribute__((aligned(Align))) Xinvec[8]  = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f };
	#else
			const float __attribute__((aligned(Align))) Xinvec[4]  = { 0.f, 1.f, 2.f, 3.f };
	#endif

			const _MData_ XIN  = opCode(load_ps, Xinvec);

			float xcf = (float) xc;
			float ycf = (float) yc;
			float zcf = (float) zc;
			const _MData_ XCOM  = opCode(set1_ps, xcf);
			const _MData_ YCOM  = opCode(set1_ps, ycf);
			const _MData_ ZCOM  = opCode(set1_ps, zcf);

			const _MData_ AX  = opCode(set1_ps, (float) Ax);
			const _MData_ AY  = opCode(set1_ps, (float) Ay);
			const _MData_ AZ  = opCode(set1_ps, (float) Az);

			auto cosi = [](_MData_ x,_MData_ y1,_MData_ z1,_MData_ y2,_MData_ z2) {
#if defined(__FMA__)
				return opCode(mul_ps, x, opCode(fmsub_ps, y1, z2, opCode(mul_ps, y2, z1)));
#else
				return opCode(mul_ps,x, opCode(sub_ps, opCode(mul_ps,y1,z2), opCode(mul_ps,y2,z1)));
#endif
			};

			auto scaprod = [](_MData_ x1,_MData_ y1,_MData_ z1,_MData_ x2,_MData_ y2,_MData_ z2) {
#if defined(__FMA__)
				return opCode(fmadd_ps, z1, z2, opCode(fmadd_ps, x1, x2, opCode(mul_ps, y1, y2)));
#else
				return opCode(add_ps,opCode(add_ps,
					opCode(mul_ps,x1,x2),opCode(mul_ps,y1,y2)),opCode(mul_ps,z1,z2));
#endif
			};

			  #pragma omp parallel default(shared)
			  {
			    _MData_ X,Y,Z,XS1,XS2,YS1,YS2,ZS1,ZS2,X3,Y3,Z3,N1,N2,N3;
					_MData_ AA,TH,TT,DD;

					const _MData_ ZERO = opCode(set1_ps, 0.0f);
					const _MData_ ONE  = opCode(set1_ps, 1.0f);

			    #pragma omp for schedule(static)
			    for (size_t zz = 0; zz < Lz; zz++) {
						Z = opCode(set1_ps, (float) (dLz+zz));
			     for (size_t yy = 0; yy < Lx; yy++) {
						 Y = opCode(set1_ps, (float) yy);   // y positions
			      for (size_t xx = 0; xx < Lx; xx += step) {
							X = opCode(set1_ps, (float) xx) + XIN;           // x positions

				size_t idx = xx + Lx*(yy + Lx*zz);

				/* vector x */

				X3 = opCode(sub_ps,XCOM,X); // this is a true vector
				Y3 = opCode(sub_ps,YCOM,Y); // calculate once, load at end?
				Z3 = opCode(sub_ps,ZCOM,Z); // calculate once, load at end?
				N3 = rsqrt1_ps(scaprod(X3,Y3,Z3,X3,Y3,Z3));
				X3 = opCode(mul_ps,X3,N3);
				Y3 = opCode(mul_ps,Y3,N3);
				Z3 = opCode(mul_ps,Z3,N3);


				/* loop over segments*/

				XS1 = opCode(sub_ps,opCode(set1_ps, xsf[0]),X); // vec
				YS1 = opCode(sub_ps,opCode(set1_ps, ysf[0]),Y); // copies
				ZS1 = opCode(sub_ps,opCode(set1_ps, zsf[0]),Z); // copies
				N1  = rsqrt1_ps(scaprod(XS1,YS1,ZS1,XS1,YS1,ZS1));
				XS1 = opCode(mul_ps,XS1,N1); // vec
				YS1 = opCode(mul_ps,YS1,N1); // copies
				ZS1 = opCode(mul_ps,ZS1,N1); // copies

				// complex accumulator for exp(iθ) ---
				_MData_ Xacc = opCode(set1_ps, 1.0f);   // real
				_MData_ Yacc = opCode(set1_ps, 0.0f);   // imag

				const int K = 4; // rescale cadence
				int kblk = 0;

				for(uint i=0; i<len-1; i++){         //1 segment less than points

					if (eps[i]>0)
						continue;

					XS2 = opCode(sub_ps,opCode(set1_ps, xsf[i+1]), X); // vec
					YS2 = opCode(sub_ps,opCode(set1_ps, ysf[i+1]), Y); // copies
					ZS2 = opCode(sub_ps,opCode(set1_ps, zsf[i+1]), Z); // copies
					N2  = rsqrt1_ps(scaprod(XS2,YS2,ZS2,XS2,YS2,ZS2));
					XS2 = opCode(mul_ps,XS2,N2); // vec
					YS2 = opCode(mul_ps,YS2,N2); // copies
					ZS2 = opCode(mul_ps,ZS2,N2); // copies

					// NUMERATOR R1.(R2xR3)
					AA = cosi(X3,YS1,ZS1,YS2,ZS2);
					AA = opCode(add_ps,AA,cosi(Y3,XS2,ZS2,XS1,ZS1));
					TT = opCode(add_ps,AA,cosi(Z3,XS1,YS1,XS2,YS2));

					// DENOMINATOR
					AA = ONE;
					AA = opCode(add_ps,AA,scaprod(XS2,YS2,ZS2,X3,Y3,Z3));
					AA = opCode(add_ps,AA,scaprod(XS1,YS1,ZS1,X3,Y3,Z3));
					AA = opCode(add_ps,AA,scaprod(XS2,YS2,ZS2,XS1,YS1,ZS1));

					// --- COMPLEX MULTIPLY: (Xacc + iYacc)*(D + iT) ---

					#if defined(__FMA__)
					    _MData_ Xn = opCode(fmadd_ps, Xacc, AA, opCode(mul_ps, Yacc, opCode(sub_ps, ZERO, TT))); // X*D - Y*T
					    _MData_ Yn = opCode(fmadd_ps, Xacc, TT, opCode(mul_ps, Yacc, AA));                        // X*T + Y*D
					#else
					    _MData_ Xn = opCode(sub_ps, opCode(mul_ps,Xacc,AA), opCode(mul_ps,Yacc,TT));
					    _MData_ Yn = opCode(add_ps, opCode(mul_ps,Xacc,TT), opCode(mul_ps,Yacc,AA));
					#endif
					Xacc = Xn; Yacc = Yn;

					// ---- simple rescale every K segments: make |acc|=1 using rsqrt ----
					if ((++kblk & (K-1)) == 0) {
						_MData_ mag2 = opCode(add_ps, opCode(mul_ps,Xacc,Xacc), opCode(mul_ps,Yacc,Yacc));
						_MData_ rinv = opCode(rsqrt_ps, mag2);
						Xacc = opCode(mul_ps, Xacc, rinv);
						Yacc = opCode(mul_ps, Yacc, rinv);
					}

					/* load next */
					XS1 = XS2;
					YS1 = YS2;
					ZS1 = ZS2;
					N1  = N2;

				} // end string segment sum


				if (p_copies_dipole>0)
				{
					TH = ZERO;
					for (int iz=-p_copies_dipole;iz<p_copies_dipole+1;iz++)
						for (int iy=-p_copies_dipole;iy<p_copies_dipole+1;iy++)
							for (int ix=-p_copies_dipole;ix<p_copies_dipole+1;ix++){
								if ((ix==0) && (iy==0) && (iz==0))
									continue;
								XS1 = opCode(add_ps,  X3, opCode(set1_ps, (float) (ix*Lx) ));
								YS1 = opCode(set1_ps, ycf + ((float) iy*Lx - yy));
								ZS1 = opCode(set1_ps, zcf + ((float) iz*Tz - zz - dLz));
								// A.R
								AA = opCode(add_ps,
											opCode(add_ps,opCode(mul_ps,AX,XS1),opCode(mul_ps,AY,YS1)),
												opCode(mul_ps,AZ,ZS1));
								// R^2
								TT = scaprod(XS1,YS1,ZS1,XS1,YS1,ZS1);
								// 1/R
								DD = opCode(rsqrt_ps,TT);
								DD = opCode(mul_ps,DD,opCode(mul_ps,DD,DD)); // R3
								// TH = opCode(add_ps,TH,opCode(div_ps,AA,DD));
								#if defined(__FMA__)
								TH = opCode(fmadd_ps,AA,DD,TH);
								#else
								TH = opCode(add_ps, opCode(mul_ps,AA,DD), TH);
								#endif
				}
				AA = opCode(cos_ps, TH);
				TT = opCode(sin_ps, TH);
				XS1 = opCode(sub_ps, opCode(mul_ps, Xacc, AA), opCode(mul_ps, Yacc, TT));
				Yacc = opCode(add_ps, opCode(mul_ps, Xacc, TT), opCode(mul_ps, Yacc, AA));
				Xacc = XS1;
				}

				if (Bext != 0)
					{
						// topology, constant
						// float phase = (Bext*6.2831853071795864*((float) yy)/((float)Lx));
						// add also Bx
						// TH = opCode(add_ps,TH,
						// 	opCode(mul_ps,opCode(set1_ps,Bamp),
						// 		opCode(sin_ps,
						// 			opCode(mul_ps,opCode(set1_ps,Bext*6.2831853071795864/((float)Lx)),X))));
						// oscillatory
						float phase = Bamp*sin(Bext*6.2831853071795864*((float) yy)/((float)Lx));
						_MData_ TH = opCode(set1_ps, phase);
						AA = opCode(cos_ps, TH);
						TT = opCode(sin_ps, TH);
						XS1 = opCode(sub_ps, opCode(mul_ps, Xacc, AA), opCode(mul_ps, Yacc, TT));
						Yacc = opCode(add_ps, opCode(mul_ps, Xacc, TT), opCode(mul_ps, Yacc, AA));
						Xacc = XS1;
					}
				// Store theta
				// opCode(store_ps,  &m2[idx], TH);

				// Store Real and Imaginary
				{
		    _MData_ mag2 = opCode(add_ps, opCode(mul_ps,Xacc,Xacc), opCode(mul_ps,Yacc,Yacc));
		    _MData_ rinv = opCode(rsqrt_ps, mag2);
		    Xacc = opCode(mul_ps, Xacc, rinv);
		    Yacc = opCode(mul_ps, Yacc, rinv);

		    // If your complex field is interleaved float* (Re,Im,Re,Im,...):
		    // Store Re lanes then Im lanes interleaved per element:
		    // Option A (scalar scatter per lane is simplest to drop in)
		    float tmpRe[step], tmpIm[step];
		    opCode(store_ps, tmpRe, Xacc);
		    opCode(store_ps, tmpIm, Yacc);
		    for (int lane = 0; lane < (int)step; ++lane) {
					if (std::isnan(tmpRe[lane])){
						// nans ocurr when x coincides with string, where theta irrelevant
						m2[2*(idx + lane) + 0] = -1.f;
						m2[2*(idx + lane) + 1] = 0.f;
					}
					else {
		        m2[2*(idx + lane) + 0] = tmpRe[lane];
		        m2[2*(idx + lane) + 1] = tmpIm[lane];
					}
		    }
			} // end store

					} // end loopx
				}// end loopy
			}// end loopz
		} // end parallel
	#undef	_MData_
	#undef	step
		}
	}


	#undef	opCode
	#undef	opCode_N
	#undef	opCode_P
	#undef	Align
	#undef	_PREFIX_


#endif
