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

#ifdef	__AVX512F__
	#define	Align 64
	#define	_PREFIX_ _mm512
#else
	#if not defined(__AVX__) and not defined(__AVX2__)
		#define	Align 16
		#define	_PREFIX_ _mm
	#else
		#define	Align 32
		#define	_PREFIX_ _mm256
	#endif
#endif


/* computes the acceleration in m2, taking into account that m is already in m2, just that */
template<const VqcdType VQcd>
inline	void	fsAccKernelThetaXeon(void * __restrict__ m2_, const PropParms ppar, const size_t Vo, const size_t Vf, FieldPrecision precision)
{
	const size_t Sf = ppar.Lx*ppar.Lx;
	const double R    = ppar.R;
	// const double ood2 = ppar.ood2a;
	const double Rpp  = ppar.Rpp;
	const double mA2  = ppar.massA2;

	if (precision == FIELD_DOUBLE)
	{
		#ifdef	__AVX512F__
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

		const double zQ = mA2*R*R*R;
		const double iz = 1.0/R;

		const _MData_ zQVec  = opCode(set1_pd, zQ);
		const _MData_ izVec  = opCode(set1_pd, iz);
		const _MData_ RppVec = opCode(set1_pd, Rpp);

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel;

			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{

				mel = opCode(load_pd,  &m2[idx]);
				/* - Acceleration
						= - mA2R3 sin(psi/R)
						missing Rpp psi to be included as a mass term */
				// tmp = opCode(mul_pd, zQVec, opCode(sin_pd, opCode(mul_pd, mel, izVec))));

				/* - Acceleration
						= - mA2R3 sin(psi/R) + Rpp psi
						 */
				tmp = opCode(sub_pd, opCode(mul_pd,RppVec,mel),
					opCode(mul_pd, zQVec, opCode(sin_pd, opCode(mul_pd, mel, izVec))));

				opCode(store_pd, &m2[idx], tmp);
			}
		}
#undef	_MData_
#undef	step
	}
	else if (precision == FIELD_SINGLE)
	{
#ifdef	__AVX512F__
	#define	_MData_ __m512
	#define	step 16
#elif	defined(__AVX__)
	#define	_MData_ __m256
	#define	step 8
#else
	#define	_MData_ __m128
	#define	step 4
#endif

		float     * __restrict__ m2	= (      float * __restrict__) __builtin_assume_aligned (m2_, Align);

		const float zQ = mA2*R*R*R;
		const float iz = 1.0/R;
		const float Rp2 = (float) Rpp;

		const _MData_ zQVec  = opCode(set1_ps, zQ);
		const _MData_ izVec  = opCode(set1_ps, iz);
		const _MData_ RppVec  = opCode(set1_ps, Rp2);

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel;

			/* load m2 */
			#pragma omp for schedule(static)
			for (size_t idx = Vo; idx < Vf; idx += step)
			{
				mel = opCode(load_ps,  &m2[idx]);

				// /* ojo! sign problem! */
				// /* - Acceleration
				// 		= mA2R3 sin(psi/R)
				// 		missing -Rpp psi */
				// tmp = opCode(mul_ps, zQVec, opCode(sin_ps, opCode(mul_ps, mel, izVec))));

				/* - Acceleration
						= - mA2R3 sin(psi/R) + Rpp psi
						 */
				tmp = opCode(sub_ps, opCode(mul_ps,RppVec,mel),
					opCode(mul_ps, zQVec, opCode(sin_ps, opCode(mul_ps, mel, izVec))));

				opCode(store_ps, &m2[idx], tmp);

			}
		}
#undef	_MData_
#undef	step
	}
}





template<const VqcdType VQcd>
inline	void	fsPropKernelThetaXeon(void * __restrict__ m_, void * __restrict__ v_, const void * __restrict__ m2_, const PropParms ppar,
				const double dz, const double c, const double d, FieldPrecision precision)
{
	const double fMom1 = ppar.fMom1;
	const size_t Lx = ppar.Lx;
	const size_t Tz = ppar.Tz;

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

		/* begin calculation */
		const size_t Ly = Lx/commSize();
		const size_t zBase = Ly*commRank();
		const size_t LyLz = Lx*Tz;
		const size_t hLx = Lx>>1;
		const size_t hTz = Tz>>1;
		const size_t maxLx = hLx+1;
		const size_t maxSf = maxLx*Tz;
		const size_t Vol   = maxLx*Tz*Ly;
		// printf("Lx %d Ly %d, zBase %d, LyLz %d, hLx %d hTz %d maxLx %d maxSf %d\n",Lx, Ly, zBase, LyLz, hLx, hTz, maxLx,maxSf);

		_MData_ TPL = opCode(set1_ps,twopioverL);
		_MData_ DZD = opCode(set1_ps,dzd);
		_MData_ DZC = opCode(set1_ps,dzc);
		// if (debug) LogOut("[fsin] Ly %lu Tz %lu zBase %lu \n",Ly,Tz,zBase);
		// if (debug) LogOut("[fsin] LyLz %lu hLx %d hTz %d maxLx %u maxSf %lu\n",LyLz,hLx,hTz,maxLx,maxSf);

		//zero mode issue
		float savem0, savem1, savev0, savev1;
		if (commRank() == 0)
		{
			savem0 = m[0];
			savem1 = m[1];
			savev0 = v[0];
			savev1 = v[1];
		}

		#pragma omp parallel default(shared)
		{
			_MData_ pV, aux, saux, caux, vV, mV, m2V;
			size_t idxc ;
			size_t sx, sy, sz;
			int    kx, ky, kz;
			size_t tmp;

			float pn[2*step];

			#pragma omp for schedule(static)
			for (size_t idx=0; idx < Vol; idx += step)
			{
if (debug) LogOut("[aa] ");
				/*First compute the momentum */
				for (int ii = 0; ii < step; ii++)
				{
					size_t idi = idx + ii;
					size_t tmp = idi/maxLx;
					int    kx  = idi - tmp*maxLx;
					int    ky  = tmp/Tz;
					int    kz  = tmp - ((size_t) ky)*Tz;
					ky += zBase;	// For MPI, transposition makes the Y-dimension smaller
					// if (kx > static_cast<int>(hLx)) kx -= static_cast<int>(Lx); // by definition hLx is always positive
					if (ky > static_cast<int>(hLx)) ky -= static_cast<int>(Lx);
					if (kz > static_cast<int>(hTz)) kz -= static_cast<int>(Tz);
					float k2   = (float) (kx*kx + ky*ky + kz*kz);
					pn[2*ii  ] = k2;
					pn[2*ii+1] = k2;
if (debug) LogOut("(%d, %d, %d) -> ",kx,ky,kz);
				}
if (debug) LogOut("\n");
				pV = opCode(load_ps,&pn[0]);

if (debug)  LogOut("[aa] idx %lu (%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f) \n",idx, pn[0], pn[1], pn[2], pn[3], pn[4], pn[5], pn[6], pn[7]);
				idxc = (idx << 1);

				//load stuff
				m2V = opCode(load_ps, &m2[idxc]);
				vV  = opCode(load_ps,  &v[idxc]);
				mV  = opCode(load_ps,  &m[idxc]);
				//calculate p
				pV  = opCode(mul_ps,TPL, opCode(sqrt_ps, pV));
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
				aux = opCode(sub_ps,opCode(mul_ps,vV,caux),opCode(mul_ps,mV,saux));
				mV  = opCode(add_ps,opCode(mul_ps,mV,caux),opCode(mul_ps,vV,saux));
				vV  = opCode(mul_ps,aux,pV);
						// store
				opCode(store_ps, &v[idxc], vV);
				opCode(store_ps, &m[idxc], mV);
if (debug) LogOut("[aa] m (%.2e, %.2e),(%.2e, %.2e) v (%.2e, %.2e),(%.2e, %.2e) \n",m[idx],m[idx+1],m[idx+2],m[idx+3],v[idx],v[idx+1],v[idx+2],v[idx+3]);
			}
		} //end parallel
		// problem with 000 (division/pV) gives nAn,
		// however it is completely trivial because p = 0, so the momentum integration is trivial and only
		// the non-linear acceleration is to be included (is in m2)
		if (commRank() == 0)
		{
			savev0 = savev0 + m2[0]*dzc;
			savev1 = savev1 + m2[1]*dzc;
			// m = m0 + v0*dzd + iintemas3
			// v = v0 + intemas3
			m[0] = savem0 + savev0*dzd;
			m[1] = savem1 + savev1*dzd;
			v[0] = savev0;
			v[1] = savev1;
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
