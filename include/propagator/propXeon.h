#include <cstdio>
#include <cmath>
#include "scalar/scalarField.h"
#include "enum-field.h"
//#include "scalar/varNQCD.h"

#include "utils/parse.h"
#include "simd/Simd.h"

template<class Simd_x, KernelType kType, const VqcdType VQcd, const int nNeigh=1, const bool UpdateM=true>
inline	void	propagateKernelXeon(const void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_,
				    const void * __restrict__ xGhost, const void * __restrict__ yGhost, const void * __restrict__ zGhost,
				    double *R, const double dz, const double c, const double d, const double ood2, const double LL,
				    const double aMass2, const double gamma, const size_t Lx, const size_t Ly, const size_t Lz,
				    const unsigned int bSizeX, const unsigned int bSizeY, const unsigned int bSizeZ)
{
	const size_t Sf = Lx*Lx;
	const size_t NSf = nNeigh*Sf;

	const Simd_x::sData * __restrict__ m	= (const Simd_x::sData * __restrict__) __builtin_assume_aligned (m_, Align);
	Simd_x::sData * __restrict__ v		= (Simd_x::sData * __restrict__) __builtin_assume_aligned (v_, Align);
	Simd_x::sData * __restrict__ m2		= (Simd_x::sData * __restrict__) __builtin_assume_aligned (m2_, Align);

	const Simd_x::sData dzc = dz*c;
	const Simd_x::sData dzd = dz*d;
	const Simd_x::sData zR = ((double) (*R));
	const Simd_x::sData z2 = zR*zR;
	const Simd_x::sData zQ = aMass2*z2*zR;
	const Simd_x::sData zN = (VQcd & VQCD_TYPE) == VQCD_1N2 ? (aMass2*z2/2.) : aMass2*z2*z2;

	const Simd_x::sData z4 = z2*z2;
	const Simd_x::sData LaLa = LL*2./z4;
	const Simd_x::sData GGGG = sqrt(ood2)*gamma;
	const Simd_x::sData mola = GGGG*dzc/2.;
	const Simd_x::sData damp1 = 1./(1.+mola);
	const Simd_x::sData damp2 = (1.-mola)*damp1;
	const Simd_x::sData epsi = mola/(1.+mola);

	// FIXME Fix this crap TEMPLATE
	Simd_x::sData CO[5] = {0, 0, 0, 0, 0} ;
	if (NN == 0) {
		return;
	}	else if (NN == 1) {
		CO[0] = 1.  ;
	}	else if (NN == 2) {
		CO[0] = 4./3.; CO[1] = -1./12.;
	} else if (NN == 3) {
		CO[0] = 1.5    ; CO[1] = -3./20.0; CO[2] = 1./90. ;
	} else if (NN == 4) {
		CO[0] = 1.6    ; CO[1] = -0.2    ; CO[2] = 8./315. ; CO[3] = -1./560. ;
	} else if (NN == 5) {
		CO[0] = 5./3.  ; CO[1] = -5./21. ; CO[2] = 5./126. ; CO[3] = -5./1008. ; CO[4] = 1./3150. ;
 	}
	_MData_ COV[5];
	for (size_t nv = 0; nv < NN ; nv++)
		COV[nv]  = opCode(set1_pd, CO[nv]*ood2);

	const size_t XC = Lx/Simd_x::xWdCx;
	const size_t YC = Ly/Simd_x::yWdCx;
	const size_t ZC = Lz/Simd_x::zWdCx;

	const Simd_x zQVec(zQ, 0.);
	const Simd_x zNVec(zN,-zN);
	const Simd_x zRVec(zR, 0.);

	uint x0, xF, y0, yF, z0, zF;

	// FIXME Movida con los ghosts y las zonas comunes a dos caras
	switch (GhostType) {

		case    xFace0:

		x0 = 0;                 xF = nNeigh;
		y0 = nNeigh;            yF = YC - nNeigh;
		z0 = nNeigh;            zF = ZC - nNeigh;
		break;

		case    xFaceF:

		x0 = XC - nNeigh;       xF = XC;
		y0 = nNeigh;            yF = YC - nNeigh;
		z0 = nNeigh;            zF = ZC - nNeigh;
		break;

		case    yFace0:

		x0 = nNeigh;            xF = XC - nNeigh;
		y0 = 0;                 yF = nNeigh;
		z0 = nNeigh;            zF = ZC - nNeigh;
		break;

		case    yFaceF:

		x0 = nNeigh;            xF = XC - nNeigh;
		y0 = YC - nNeigh;       yF = YC;
		z0 = nNeigh;            zF = ZC - nNeigh;
		break;

		case    zFace0:

		x0 = nNeigh;            xF = XC - nNeigh;
		y0 = nNeigh;            yF = YC - nNeigh;
		z0 = 0;                 zF = nNeigh;
		break;

		case    zFaceF:

		x0 = nNeigh;            xF = XC - nNeigh;
		y0 = nNeigh;            yF = YC - nNeigh;
		z0 = ZC - nNeigh;       zF = ZC;
		break;
	}

	const uint zM = (zF - z0 + bSizeZ - 1)/bSizeZ;
	const uint yM = (yF - y0 + bSizeY - 1)/bSizeY;
	const uint xM = (xF - x0 + bSizeX - 1)/bSizeX;

	for (uint zT = 0; zT < zM; zT++)
	 for (uint yT = 0; yT < yM; yT++)
	  for (uint xT = 0; xT < xM; xT++)
	  #pragma omp parallel default(shared)
	  {
	    _MData_ tmp, tm2, tm3, tm4, mel, lap;
	    #pragma omp for collapse(3) schedule(static)
	    for (uint zz = 0; zz < bSizeZ; zz++) {
	     for (uint yy = 0; yy < bSizeY; yy++) {
	      for (uint xx = 0; xx < bSizeX; xx++) {
		uint xC = xx + bSizeX*xT + x0;
		uint yC = yy + bSizeY*yT + y0;
		uint zC = zz + bSizeZ*zT + z0;

		size_t X[2], idxMx, idxPx, idxMy, idxPy, idxMz, idxPz, idxP0, idxV0;
		size_t idx = zC*(YC*XC) + yC*XC + xC;

		//if (idx >= Vf)
		//	continue;
		{
			X[0] = xC;
			X[1] = yC;
		}

		idxP0 = idx * Simd_x::sWide;
		mel.Load(&m[idxP0]);
		lap.Zero();

		// This loop performs many consecutive loads and stalls the processor
		// Try to distribute the loads and intercalate calculations
		// It describes border cases that are related to GhostType and the face exchange

		for (size_t nv=1; nv<=nNeigh; nv++)
		{

			if (xC < nv) {	// Mal, aquí leemos de ghosts
				idxMx = (idx + XC - nv) * Simd_x::sWide;
				tmp += tm2.Load(&m[idxMx]).xPermute();
			} else {
				idxMx = (idx - nv) * Simd_x::sWide;
				tmp += tm2.Load(&m[idxMx]);
			}

			if (xC >= XC - nv) {
				idxPx = (idx - XC + nv) * Simd_x::sWide;
				tmp += tm2.Load(&m[idxPx]).xPermute();
			} else {
				idxPx = ((idx + nv) * Simd_x::sWide;
				tmp += tm2.Load(&m[idxPx]);
			}

			if (yC < nv) {
				idxMy = (idx + XC - nv) * Simd_x::sWide;
				tmp += tm2.Load(&m[idxMy]).yPermute();
			} else {
				idxMy = (idx - nv) * Simd_x::sWide;
				tmp += tm2.Load(&m[idxMy]);
			}

			if (yC >= YC - nv) {
				idxPy = (idx - YC + nv) * Simd_x::sWide;
				tmp += tm2.Load(&m[idxPy]).yPermute();
			} else {
				idxPy = ((idx + nv) * Simd_x::sWide;
				tmp += tm2.Load(&m[idxPy]);
			}

			if (zC < nv) {
				idxMz = (idx + XC - nv) * Simd_x::sWide;
				tmp += tm2.Load(&m[idxMz]).zPermute();
			} else {
				idxMz = (idx - nv) * Simd_x::sWide;
				tmp += tm2.Load(&m[idxMz]);
			}

			if (zC >= ZC - nv) {
				idxPz = (idx - YC + nv) * Simd_x::sWide;
				tmp += tm2.Load(&m[idxPz]).zPermute();
			} else {
				idxPz = ((idx + nv) * Simd_x::sWide;
				tmp += tm2.Load(&m[idxPz]);
			}

			tmp -= mel*Simd_x(6.0);
			tmp *= Simd_x(COV[nv-1]);
			lap += tmp;

		} //end neighbour loop nv

		tm2 = mel*mel;
		tm3 = !mel;
/*
#if	defined(__AVX512F__)
			mPx = opCode(add_pd, opCode(castsi512_pd, opCode(shuffle_epi32, opCode(castpd_si512, mPy), _MM_PERM_BADC)), mPy);
#elif	defined(__AVX__)
		mPx = opCode(add_pd, opCode(permute_pd, mPy, 0b00000101), mPy);
#else
		mPx = opCode(add_pd, opCode(shuffle_pd, mPy, mPy, 0b00000001), mPy);
#endif
*/
		// FIXME Separar potenciales
		switch	(VQcd & VQCD_TYPE) {

			default:
			case	VQCD_PQ_ONLY:
			tm4 = lap - ((tm3 - z2)*Simd_x(LL))*mel;
			break;

			case	VQCD_0:
			tm2 = opCode(vqcd0_ps,mel)/sqrt(tm3*(tm3*tm3));	// FIXME vqcd0_ps won't work properly for double

	 		tm4 = (lap + tm2*zNVec) - ((tm3 - z2)*Simd_x(LL))*mel;
	 		break;

			case	VQCD_1:
			tm4 = (lap + zQVec) - ((tm3 - z2)*Simd_x(LL))*mel;
			break;

			case	VQCD_1_PQ_2:
			tm4 = (lap + zQVec) - (((tm3*tm3) - z4)*(tm3*Simd_x(LaLa)))*mel;
			break;

			case	VQCD_2:
			tm4 = (lap - (Simd_x(zQ)*(mel - zRVec))) - ((tm3 - z2)*Simd_x(LL))*mel;
			break;

			case	VQCD_1N2:
			tm4 = (lap + (zNVec*mel)) - ((tm3 - z2)*Simd_x(LL))*mel;
			break;

		}
//mMx -> tm4 mPx -> tm3 mPy -> tm2
		tm2.Load(&v[idxP0]);
		// FIXME With potentials
		switch	(VQcd & VQCD_DAMP) {

			default:
			case	VQCD_NONE:
#if	defined(__AVX512F__) || defined(__FMA__)
			tmp = tm4.fma(Simd_x(dzc), tm2);
#else
			tmp = tm2 + (tm4*Simd_x(dzc));
#endif
			break;

			case	VQCD_DAMP_RHO:
			{
				//New implementation
				tmp = mel*tm2;
				// FIXME This
#if	defined(__AVX__)// || defined(__AVX512F__)
				auto vecmv = opCode(add_pd, opCode(permute_pd, tmp, 0b00000101), tmp);
#else
				auto vecmv = opCode(add_pd, opCode(shuffle_pd, tmp, tmp, 0b00000001), tmp);
#endif

				// vecma = MA
				// mel = M, mMx = A
				tmp = mel*tm4;
				// FIXME This
#if	defined(__AVX__)// || defined(__AVX512F__)
				auto vecma = opCode(add_pd, opCode(permute_pd, tmp, 0b00000001), tmp);
#else
				auto vecma = opCode(add_pd, opCode(shuffle_pd, tmp, tmp, 0b00000001), tmp);
#endif

// FIXME This crap
#if	defined(__AVX512F__) || defined(__FMA__)
// A*dzc + mPy - epsi (M/|M|^2)(2*(vecmv-|M|^2/t) +vecma dzc)
tmp = opCode(sub_pd,
	opCode(fmadd_pd, mMx, opCode(set1_pd, dzc), mPy),
	opCode(mul_pd, opCode(mul_pd, opCode(set1_pd, epsi), opCode(div_pd, mel, mPx)),
		opCode(fmadd_pd, opCode(sub_pd,vecmv,opCode(div_pd,mPx,opCode(set1_pd, zR))), opCode(set1_pd, 2.0), opCode(mul_pd, vecma, opCode(set1_pd, dzc)))));
#else
tmp = opCode(sub_pd,
	opCode(add_pd, mPy, opCode(mul_pd, mMx, opCode(set1_pd, dzc))),
	opCode(mul_pd, opCode(mul_pd, opCode(set1_pd, epsi), opCode(div_pd, mel, mPx)),
		opCode(add_pd,
			opCode(mul_pd, opCode(sub_pd,vecmv,opCode(div_pd,mPx,opCode(set1_pd, zR))), opCode(set1_pd, 2.0)),
			opCode(mul_pd, vecma, opCode(set1_pd, dzc)))));
#endif
			}
			break;

			case	VQCD_DAMP_ALL:
#if	defined(__AVX512F__) || defined(__FMA__)
			tmp = tm2.fma(Simd_x(damp2), (tm4*Simd_x(damp1*dzc)));
#else
			tmp = (tm2*Simd_x(damp2)) + (tm4*Simd_x(damp1*dzc));
#endif
			break;
		}

		if (VQcd & VQCD_EVOL_RHO)
		{
			auto vecmv = mel*tmp;
			//FIXME Crap strikes again
#if	defined(__AVX__)// || defined(__AVX512F__)
			auto vecma = opCode(add_pd, opCode(permute_pd, vecmv, 0b00000101), vecmv);
#else
			auto vecma = opCode(add_pd, opCode(shuffle_pd, vecmv, vecmv, 0b00000001), vecmv);
#endif
			tmp = (mel*vecma)/tm3;
		}

		tmp.Save(&v[idxP0]);

		if (UpdateM == true) {
#if	defined(__AVX512F__) || defined(__FMA__)
			tm3 = tmp.fma(Simd_x(dzd), mel);
#else
			tm3 = mel + (tmp*Simd_x(dzd));
#endif
			tm3.Stream(&m2[idxP0]);
		}
	      }
	    }
	  }
	}
}

template<class Simd_x>
inline	void	updateMXeon(void * __restrict__ m_, const void * __restrict__ v_, const double dz, const double d,
			    const size_t x0, const size_t xF, const size_t y0, const size_t yF, const size_t z0, const size_t zF) {

	Simd_x::sData * __restrict__ m		= (Simd_x::sData * __restrict__) __builtin_assume_aligned (m_, Align);
	const Simd_x::sData * __restrict__ v	= (const Simd_x::sData * __restrict__) __builtin_assume_aligned (v_, Align);

	const Simd_x dzd(dz*d);

	#pragma omp parallel default(shared)
	{
		register Simd_x mIn, vIn, tmp;
		register size_t idxP0;

		#pragma omp parallel default(shared)
		{
			register Simd_x mIn, vIn, tmp;
			register size_t idxP0;

			#pragma omp for schedule(static)
			for (size_t zC = z0; zC < zF; zC ++)
			  for (size_t yC = y0; yC < yF; yC ++)
			    for (size_t xC = x0; xC < xF; xC ++)
			    {
				idxP0 = (xC + yC*Lx + zC*Lx*Ly) * Simd_x::sWide;
				vIn.Load(&v[idxP0]);
				mIn.Load(&m[idxP0]);
#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = dzd.fma(vIn, mIn);
#else
				tmp = mIn + (vIn * dzd);
#endif
				tmp.Save(&m[idxP0]);
			    }
		}
	}
}
