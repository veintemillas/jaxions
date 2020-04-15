#include<cstdio>
#include<cmath>
#include"scalar/scalarField.h"
#include"enum-field.h"

#include"utils/utils.h"
#include"fft/fftCode.h"

#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

#include <immintrin.h>

#if	defined(__AVX512F__)
	#define	Align 64
	#define	_PREFIX_ _mm512
	#define	_MInt_ __m512i
#else
	#if not defined(__AVX__) and not defined(__AVX2__)
		#define	Align 16
		#define	_PREFIX_ _mm
		#define	_MInt_ __m128i
	#else
		#define	Align 32
		#define	_PREFIX_ _mm256
		#define	_MInt_ __m256i
	#endif
#endif


/* computes the acceleration in m2, taking into account that m is already in m2, just that */
template<const VqcdType VQcd>
inline	void	fsAccKernelXeon(void * __restrict__ v_, void * __restrict__ m2_, double *R, const double dz, const double c, const double d,
				const double ood2, const double LL, const double aMass2, const double gamma, const double fMom, const size_t Lx, const size_t Vo, const size_t Vf, FieldPrecision precision)
{
	const size_t Sf = Lx*Lx;

	if (precision == FIELD_DOUBLE)
	{
#if	defined(__AVX512F__)
	#define	_MData_ __m512d
	#define	step 4
#elif	defined(__AVX__)
	#define	_MData_ __m256d
	#define	step 2
#else
	#define	_MData_ __m128d
	#define	step 1
#endif

		// double	     * __restrict__ m	= (double	* __restrict__)	__builtin_assume_aligned (m_,  Align);
		double	     * __restrict__ v	= (double	* __restrict__) __builtin_assume_aligned (v_,  Align);
		double       * __restrict__ m2	= (double * __restrict__) __builtin_assume_aligned (m2_, Align);

		const double dzc = dz*c;
		const double dzd = dz*d;
		const double zR = *R;
		const double z2 = zR*zR;
		const double z4 = z2*z2;
		const double zQ = aMass2*z2*zR;

		const double LaLa = LL*2./z4;
		const double GGGG = pow(ood2,0.5)*gamma;
		const double GGiZ = GGGG/zR;
		const double mola = GGGG*dzc/2.;
		const double damp1 = 1./(1.+mola);
		const double damp2 = (1.-mola)*damp1;
		const double epsi = mola/(1.+mola);

#if	defined(__AVX512F__)
//		const size_t XC = (Lx<<2);
//		const size_t YC = (Lx>>2);

		const double    __attribute__((aligned(Align))) zQAux[8] = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };	// Only real part
		const double    __attribute__((aligned(Align))) zRAux[8] = { zR, 0., zR, 0., zR, 0., zR, 0. };	// Only real part
		const long long __attribute__((aligned(Align))) shfRg[8] = {6, 7, 0, 1, 2, 3, 4, 5 };
		const long long __attribute__((aligned(Align))) shfLf[8] = {2, 3, 4, 5, 6, 7, 0, 1 };

		const auto  vShRg = opCode(load_si512, shfRg);
		const auto  vShLf = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
//		const size_t XC = (Lx<<1);
//		const size_t YC = (Lx>>1);

		const double __attribute__((aligned(Align))) zQAux[4] = { zQ, 0., zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zRAux[4] = { zR, 0., zR, 0. };	// Only real part
#else
//		const size_t XC = Lx;
//		const size_t YC = Lx;

		const double __attribute__((aligned(Align))) zQAux[2] = { zQ, 0. };	// Only real part
		const double __attribute__((aligned(Align))) zRAux[2] = { zR, 0. };	// Only real part

#endif
		const _MData_ zQVec  = opCode(load_pd, zQAux);
		const _MData_ zRVec  = opCode(load_pd, zRAux);
		const _MData_ fMVec  = opCode(set1_pd, fMom);

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, mPx, mMx, mPy;
			size_t idxMz, idxP0 ;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				idxMz = ((idx-Sf) << 1);
				idxP0 = (idx << 1);

				// mPx = opCode(load_pd, &m2[idxMz]);
				// tmp = opCode(mul_pd, mPx, fMVec);
				// mel = opCode(load_pd,  &m[idxP0]);
				// mPy = opCode(mul_pd, mel, mel);

				mel = opCode(load_pd,  &m2[idxMz]);
				mPy = opCode(mul_pd, mel, mel);

#if	defined(__AVX512F__)
				mPx = opCode(add_pd, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, mPy), _MM_PERM_BADC)), mPy);
#elif	defined(__AVX__)
				mPx = opCode(add_pd, opCode(permute_pd, mPy, 0b00000101), mPy);
#else
				mPx = opCode(add_pd, opCode(shuffle_pd, mPy, mPy, 0b00000001), mPy);
#endif

				switch	(VQcd & V_TYPE) {

					default:
					case	V_QCD1_PQ1:
						mMx = opCode(add_pd, tmp,
							opCode(sub_pd, zQVec,
							opCode(mul_pd, mel,
								opCode(mul_pd,
									opCode(sub_pd, mPx, opCode(set1_pd, z2)),
									opCode(set1_pd, LL)))));
						break;

					case	V_QCD1_PQ2:
						mMx = opCode(add_pd, tmp,
							opCode(sub_pd, zQVec,
							opCode(mul_pd, mel,
								opCode(mul_pd,
									opCode(sub_pd, opCode(mul_pd, mPx, mPx), opCode(set1_pd, z4)),
									opCode(set1_pd, LaLa)))));
						break;

					case	V_QCDV_PQ1:
						mMx = opCode(add_pd, tmp,
							opCode(sub_pd,
								opCode(mul_pd, opCode(set1_pd, zQ), opCode(sub_pd, zRVec, mel)),
								opCode(mul_pd, mel,
									opCode(mul_pd,
										opCode(sub_pd, mPx, opCode(set1_pd, z2)),
										opCode(set1_pd, LL)))));
						break;
				}

				opCode(store_pd, &m2[idxMz], mMx);
			}
		}
#undef	_MData_
#undef	step
	}
	else if (precision == FIELD_SINGLE)
	{
#if	defined(__AVX512F__)
	#define	_MData_ __m512
	#define	step 8
#elif	defined(__AVX__)
	#define	_MData_ __m256
	#define	step 4
#else
	#define	_MData_ __m128
	#define	step 2
#endif

		// float	    * __restrict__ m	= (      float * __restrict__) __builtin_assume_aligned (m_,  Align);
		float	    * __restrict__ v	= (      float * __restrict__) __builtin_assume_aligned (v_,  Align);
		float     * __restrict__ m2	= (      float * __restrict__) __builtin_assume_aligned (m2_, Align);

		const float dzc = dz*c;
		const float dzd = dz*d;
		const float zR = *R;
		const float z2 = zR*zR;
		const float z4 = z2*z2;
		const float zQ = (float) (aMass2*z2*zR);

		const float LaLa = LL*2.f/z4;
		const float GGGG = pow(ood2,0.5f)*gamma;
		const float GGiZ = GGGG/zR;
		const float mola = GGGG*dzc/2.f;
		const float damp1 = 1.f/(1.f+mola);
		const float damp2 = (1.f-mola)*damp1;
		const float epsi = mola/(1.f+mola);

#if	defined(__AVX512F__)
//		const size_t XC = (Lx<<3);
//		const size_t YC = (Lx>>3);

		const float __attribute__((aligned(Align))) zQAux[16] = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zRAux[16] = { zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f };
		const int   __attribute__((aligned(Align))) shfRg[16] = {14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
		const int   __attribute__((aligned(Align))) shfLf[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1};

		const auto  vShRg  = opCode(load_si512, shfRg);
		const auto  vShLf  = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
//		const size_t XC = (Lx<<2);
//		const size_t YC = (Lx>>2);

		const float __attribute__((aligned(Align))) zQAux[8]  = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zRAux[8]  = { zR, 0.f, zR, 0.f, zR, 0.f, zR, 0.f };
#else
//		const size_t XC = (Lx<<1);
//		const size_t YC = (Lx>>1);

		const float __attribute__((aligned(Align))) zQAux[4]  = { zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zRAux[4]  = { zR, 0.f, zR, 0.f };
#endif
		const _MData_ zQVec  = opCode(load_ps, zQAux);
		const _MData_ zRVec  = opCode(load_ps, zRAux);
		const _MData_ fMVec  = opCode(set1_ps, fMom);


		// LogOut("[aa] %f %f %f %f \n",m2[0],m2[1],m2[2],m2[3]);
		// LogOut("[aa] Sf, Vo, Vf, %lu %lu %lu \n",Sf, Vo,Vf);
		/* begin calculation */
		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, mPx, mPy, mMx;
			size_t idxMz, idxP0 ;

			/* load m2 */
			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				idxMz = ((idx-Sf) << 1);
				idxP0 = (idx << 1);

				// mPx = opCode(load_ps, &m2[idxMz]);
				// tmp = opCode(mul_ps, mPx, fMVec);
				// mel = opCode(load_ps, &m[idxP0]);
				// mPy = opCode(mul_ps, mel, mel);

				mel = opCode(load_ps, &m2[idxMz]);
				mPy = opCode(mul_ps, mel, mel);

				/* compute modulus in mPx */
#if	defined(__AVX__)// || defined(__AVX512F__)
				mPx = opCode(add_ps, opCode(permute_ps, mPy, 0b10110001), mPy);
#else
				mPx = opCode(add_ps, opCode(shuffle_ps, mPy, mPy, 0b10110001), mPy);
#endif
				/* compute acceleration without laplacian and without our friend axion potential */
				/* TODO ADD POTENTIALS */
				switch	(VQcd & V_TYPE) {

					default:
					case	V_QCD1_PQ1:
						mMx = opCode(mul_ps, mel,
									 opCode(mul_ps,
										opCode(sub_ps, opCode(set1_ps, z2), mPx),
										 opCode(set1_ps, LL))); // Phi(R^2 - |Phi|^2)LL
						break;

					case	V_QCD1_PQ2:
						mMx =
								opCode(mul_ps, mel,
									opCode(mul_ps,
										opCode(sub_ps, opCode(set1_ps, z4), opCode(mul_ps, mPx, mPx)),
										opCode(set1_ps, LaLa))); //
						break;

					case	V_QCDV_PQ1:
						/* finish */
						// mMx = opCode(sub_ps,
						// 	opCode(sub_ps, tmp, opCode(mul_ps, opCode(set1_ps, zQ), opCode(sub_ps, mel, zRVec))),
						// 	opCode(mul_ps,
						// 		opCode(mul_ps, mel,
						// 			opCode(sub_ps, mPx, opCode(set1_ps, z2))),
						// 			opCode(set1_ps, LL)));
						break;
				}
				/* we only store acceleration in m2 in configuration space
				because m and v are in Fourier space*/
				opCode(store_ps, &m2[idxMz], mMx);
				// LogOut("[aa] %lu, (%f,%f), (%f,%f) \n",idxMz, m2[idxMz],m2[idxMz+1],m2[idxMz+2],m2[idxMz+3]);
			}
		}
#undef	_MData_
#undef	step
	}
}









/* vector version*/

template<const VqcdType VQcd>
inline	void	fsPropKernelXeon(void * __restrict__ m_, void * __restrict__ v_, const void * __restrict__ m2_, double *R, const double dz, const double c, const double d,
				const double intemas3, const double iintemas3, const double shift, const double fMom1, const size_t Lx, const size_t Tz, FieldPrecision precision)
{
	const size_t Sf = Lx*Lx;

	if (precision == FIELD_SINGLE)
	{
	#if	defined(__AVX512F__)
	#define	_MData_ __m512
	#define	step 8
	#elif	defined(__AVX__)
	#define	_MData_ __m256
	#define	step 4
	#else
	#define	_MData_ __m128
	#define	step 2
	#endif

		float	    * __restrict__ m	= (      float * __restrict__) __builtin_assume_aligned (m_,  Align);
		float	    * __restrict__ v	= (      float * __restrict__) __builtin_assume_aligned (v_,  Align);
		const float * __restrict__ m2	= (const float * __restrict__) __builtin_assume_aligned (m2_, Align);

		// const float zR = *R;
		// const float z2 = zR*zR;
		// const float zQ = (float) (aMass2*z2*zR);

		const float dzc = dz*c;
		const float dzd = dz*d;
		const float twopioverL = (float) (fMom1);
		const float twopioverLdzd = (float) (fMom1*dz*d);

	#if	defined(__AVX512F__)
		// const float __attribute__((aligned(Align))) xxxx[8] = { 0 +0<<32 , 1+1<<32, 2+2<<32, 3 +3<<32, 4+4<<32, 5+5<<32, 6+6<<32, 7+7<<32};
		constexpr _MData_ xxxx = { 0.f , 0.f, 1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f, 5.f, 5.f, 6.f, 6.f, 7.f, 7.f };
	#elif	defined(__AVX__)
		constexpr _MData_ xxxx = { 0.f , 0.f, 1.f, 1.f, 2.f, 2.f, 3.f, 3.f };
	#else
		constexpr _MData_ xxxx = { 0.f , 0.f, 1.f, 1.f};
	#endif



		/* begin calculation */
		const size_t Ly = Lx/commSize();
		const size_t zBase = Ly*commRank();
		const size_t LyLz = Lx*Tz;
		const int hLx = Lx>>1;
		const int hTz = Tz>>1;
		const uint   maxLx = Lx;
		const size_t maxSf = maxLx*Tz;

		_MData_ TPL = opCode(set1_ps,twopioverL);
		_MData_ DZD = opCode(set1_ps,dzd);
		_MData_ DZC = opCode(set1_ps,dzc);
		// if (debug) LogOut("[fsin] Ly %lu Tz %lu zBase %lu \n",Ly,Tz,zBase);
		// if (debug) LogOut("[fsin] LyLz %lu hLx %d hTz %d maxLx %u maxSf %lu\n",LyLz,hLx,hTz,maxLx,maxSf);

		//zero mode issue
		float savem0, savem1, savev0, savev1;
		if (commRank() == 0)
		{
			savem0 = m[2*Sf];
			savem1 = m[2*Sf+1];
			savev0 = v[0];
			savev1 = v[1];
		}

		#pragma omp parallel default(shared)
		{
			_MData_ pV, aux, saux, caux, vV, mV, m2V;
			size_t idxm, idxv ;
			size_t sx, sy, sz;
			int    kx, ky, kz;
			size_t tmp;

			// number of local modes, I'd say Lx^2 Lz
			// start by splitting on z so that Lx^2 always fist an integer 32
			#pragma omp for schedule(static)
			for (size_t oy = 0; oy < Ly; oy++)	// As Javier pointed out, the transposition makes y the slowest coordinate
			{
				ky =(int) (oy + zBase);
				if (ky > hLx)
					ky -= static_cast<int>(Lx);

				for (size_t oz = 0; oz < Tz; oz++)
				{
					kz = (int) oz;
					if (kz > hTz) kz -= static_cast<int>(Tz);

					float pypz2 = (float) ky*ky+kz*kz;

					for (size_t ox = 0; ox < Lx; ox += step)
					{
						size_t idx = ox + oz*Lx + oy*LyLz;
						// complex positions ... jump by 2
						idxv = (idx << 1);
						idxm = ((idx+Sf) << 1);
						// kx
						kx = (int) ox;
						if (ox >= hLx)
						{ // 64; 0-15 16-31 32-47 48-63 -> 0-15 16-31 -32--17 -16--1
							// element 32 or -32 is the same;
							kx -= Lx;
						}
						float kkx = (float) kx;
						pV = opCode(add_ps,opCode(set1_ps,kkx),xxxx);
						//load stuff
						m2V = opCode(load_ps, &m2[idxv]);
						vV  = opCode(load_ps, &v[idxv]);
						mV  = opCode(load_ps, &m[idxm]);
						//calculate p
						pV  = opCode(mul_ps,TPL,
										opCode(sqrt_ps,opCode(add_ps,opCode(set1_ps,pypz2),
											opCode(mul_ps,pV, pV))));
						aux = opCode(mul_ps,pV,DZD);
						saux = opCode(sin_ps,aux);
						caux = opCode(cos_ps,aux);
						// iterate
						// v = v + m2 dct
						// v = v/p
						// v = v Cos[pdt] - m Sin[pdt]
						// m = m Cos[pdt] + v Sin[pdt]
						// v = v*p
						vV  = opCode(div_ps,opCode(add_ps,vV,opCode(mul_ps,m2V,DZC)),pV);// problem for pV = 0,... solved outside
						// vV  = opCode(mul_ps,vV,pV);
						aux = opCode(sub_ps,opCode(mul_ps,vV,caux),opCode(mul_ps,mV,saux));
						mV  = opCode(add_ps,opCode(mul_ps,mV,caux),opCode(mul_ps,vV,saux));
						vV  = opCode(mul_ps,aux,pV);
						// store
						opCode(store_ps, &v[idxv], vV);
						opCode(store_ps, &m[idxm], mV);
						// LogOut("[aa] (%d,%d,%d) %lu, (%f,%f),(%f,%f) v (%f,%f),(%f,%f) \n",ky,kz,kx,idxv, m[idxm],m[idxm+1],m[idxm+2],m[idxm+3],v[idxv],v[idxv+1],v[idxv+2],v[idxv+3]);
					}
				}
			}
		}
		// problem with 000 (division/pV) gives nAn,
		// it is redone here
		// V_QCD1 is included here in m2
		// it can be included exactly if a mass integral formula is given
		if (commRank() == 0)
		{
			// if chi included in m2
			// v[0] = savev0 + (m2[0]+zQ)*dzc;
			// v[1] = savev1 + (m2[0])*dzc;
			// m[2*Sf+0] = savem0 + V[0]*dzd;
			// m[2*Sf+1] = savem1 + V[1]*dzd;

			// if chi included in p - all orders
			savev0 = savev0 + m2[0]*dzc;
			savev1 = savev1 + m2[1]*dzc;
			// m = m0 + v0*dzd + iintemas3
			// v = v0 + intemas3
			m[2*Sf+0] = savem0 + savev0*dzd + iintemas3;
			m[2*Sf+1] = savem1 + savev1*dzd;
			v[0] = savev0 + intemas3;
			v[1] = savev1;

			// if crazy
			// m[2*Sf+0] = (1.f+shift)*(*R);
			// m[2*Sf+1] = 0.f;
			// v[0] = 0.f;
			// v[1] = 0.f;
		}
		// LogOut("[aa] (%d,%d,%d) %lu, (%f,%f),(%f,%f) v (%f,%f),(%f,%f) \n",0,0,0,0, m[2*Sf],m[2*Sf+1],m[2*Sf+2],m[2*Sf+3],v[0],v[1],v[2],v[3]);
	}
	else if (precision == FIELD_DOUBLE)
	{
#if	defined(__AVX512F__)
	#define	_MData_ __m512d
	#define	step 4
#elif	defined(__AVX__)
	#define	_MData_ __m256d
	#define	step 2
#else
	#define	_MData_ __m128d
	#define	step 1
#endif

		double	     * __restrict__ m	= (double	* __restrict__)	__builtin_assume_aligned (m_,  Align);
		double	     * __restrict__ v	= (double	* __restrict__) __builtin_assume_aligned (v_,  Align);
		const double * __restrict__ m2	= (const double * __restrict__) __builtin_assume_aligned (m2_, Align);

		const double dzc = dz*c;
		const double dzd = dz*d;

		}
#undef	_MData_
#undef	step
}
