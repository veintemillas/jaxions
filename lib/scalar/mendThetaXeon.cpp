#include <cstdio>
#include <cmath>
#include "scalar/scalarField.h"
#include "enum-field.h"
#include "scalar/varNQCD.h"
#include "utils/parse.h"

#include "utils/triSimd.h"

#define opCode_P(x,y,...) x ## _ ## y (__VA_ARGS__)
#define opCode_N(x,y,...) opCode_P(x, y, __VA_ARGS__)
#define opCode(x,...) opCode_N(_PREFIX_, x, __VA_ARGS__)

#include <immintrin.h>

#ifdef  __AVX512F__
	#define Align 64
	#define _PREFIX_ _mm512
#else
	#if not defined(__AVX__) and not defined(__AVX2__)
		#define Align 16
		#define _PREFIX_ _mm
	#else
		#define Align 32
		#define _PREFIX_ _mm256
	#endif
#endif

//----------------------------------------------------------------------
//		CHECK JUMPS
//----------------------------------------------------------------------

//	THIS FUNCTION CHECKS THETA IN ORDER AND NOTES DOWN POSITIONS WITH JUMPS OF 2 PI
//  MARKS THEM DOWN INTO THE ST BIN ARRAY AS POSSIBLE PROBLEMATIC POINTS WITH GRADIENTS
//  TRIES TO MEND THE THETA DISTRIBUTION INTO MANY RIEMMAN SHEETS TO HAVE A CONTINUOUS FIELD

template<const bool zDir>
inline  size_t	mendThetaKernelXeon(void * __restrict__ m_, void * __restrict__ v_, const double z, const size_t Lx, const size_t Lz, const size_t Vo, const size_t Vf, FieldPrecision precision)
{
        const size_t Sf = Lx*Lx;
	size_t count = 0;

        if (precision == FIELD_DOUBLE)
        {
#ifdef  __AVX512F__
	#define _MData_ __m512d
	#define step 8
#elif   defined(__AVX__)
	#define _MData_ __m256d
	#define step 4
#else
	#define _MData_ __m128d
	#define step 2
#endif

		double * __restrict__ m = (double * __restrict__) __builtin_assume_aligned (m_, Align);
		double * __restrict__ v = (double * __restrict__) __builtin_assume_aligned (v_, Align);

		const double zP = M_PI*z;

#ifdef  __AVX512F__
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);

		const long long int __attribute__((aligned(Align))) shfLf[8] = { 1, 2, 3, 4, 5, 6, 7, 0 };
#elif   defined(__AVX__)
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);
#else
		const size_t XC = (Lx<<1);
		const size_t YC = (Lx>>1);
#endif
		const _MData_ pVec  = opCode(set1_pd, +zP);
		const _MData_ mVec  = opCode(set1_pd, -zP);
		const _MData_ vVec  = opCode(set1_pd, 2.*M_PI);
		const _MData_ cVec  = opCode(set1_pd, zP*2.);

#ifdef  __AVX512F__
                const auto vShLf  = opCode(load_si512, shfLf);
#endif

		_MData_ mel, mDf, mDp, mDm, mPx, vPx;

		for (size_t idx = Vo; idx < Vf; idx += step)
		{
			size_t X[2], idxPx, idxVx;

			{
				size_t tmi = idx/XC, tpi;

			        tpi = tmi/YC;
				X[1] = tmi - tpi*YC;
				X[0] = idx - tmi*XC;
			}

			if (X[0] == XC-step)
				idxPx = idx - XC + step;
			else
				idxPx = idx + step;

			idxVx = idxPx - Sf;

			mel = opCode(load_pd, &m[idx]);
			mPx = opCode(load_pd, &m[idxPx]);
			vPx = opCode(load_pd, &v[idxVx]);

			/*	X-Direction	*/

			mDf = opCode(sub_pd, mPx, mel);
#ifdef	__AVX512__
			auto pMask = opCode(cmp_pd_mask, mDf, pVec, _CMP_GE_OQ);
			auto mMask = opCode(cmp_pd_mask, mDf, mVec, _CMP_LT_OQ);

			mPx = opCode(mask_sub_pd, mPx, pMask, mPx, cVec);
			mPx = opCode(mask_add_pd, mPx, mMask, mPx, cVec);
			vPx = opCode(mask_sub_pd, vPx, pMask, vPx, vVec);
			vPx = opCode(mask_add_pd, vPx, mMask, vPx, vVec);

			pMask |= mMask;

			for (int i=1, int k=0; k<step; i<<=1, k++)
				count += (pMask & i) >> k;
#else	// AVX and SSE4.1

#ifdef	__AVX__
			mDp = opCode(cmp_pd, mDf, pVec, _CMP_GE_OQ);
			mDm = opCode(cmp_pd, mDf, mVec, _CMP_LT_OQ);
#else
			mDp = opCode(cmpge_pd, mDf, pVec);
			mDm = opCode(cmplt_pd, mDf, mVec);
#endif
			mPx = opCode(sub_pd, mPx,
				opCode(sub_pd,
					opCode(and_pd, mDp, cVec),
					opCode(and_pd, mDm, cVec)));

			vPx = opCode(sub_pd, vPx,
				opCode(sub_pd,
					opCode(and_pd, mDp, vVec),
					opCode(and_pd, mDm, vVec)));

			mDp = opCode(or_pd, mDp, mDm);

			for (int k=0; k<step; k++)
				count += reinterpret_cast<size_t&>(mDp[k]) & 1;
#endif	// AVX and SSE4.1
		}

		/*	Move forward in z	*/

		size_t idxPz = Vo + Sf, idxVz = Vo;

		mPx = opCode(load_pd, &m[idxPz]);

		const int nSplit  = commSize();

		if (zDir == false) {
			if (nSplit > 1) {
				// Exchange ghosts to get the right v
				const int rank    = commRank();
				const int nSplit  = commSize();
				const int fwdNeig = (rank + 1) % nSplit;
				const int bckNeig = (rank - 1 + nSplit) % nSplit;
				MPI_Request rSendBck, rRecvFwd;

				double tmp[step];

				MPI_Send_init(v,   step, MPI_DOUBLE, bckNeig, 2*rank+1,    MPI_COMM_WORLD, &rSendBck);
				MPI_Recv_init(tmp, step, MPI_DOUBLE, fwdNeig, 2*fwdNeig+1, MPI_COMM_WORLD, &rRecvFwd);
			
				MPI_Start(&rRecvFwd);
				MPI_Start(&rSendBck);

				MPI_Wait(&rSendBck, MPI_STATUS_IGNORE);
				MPI_Wait(&rRecvFwd, MPI_STATUS_IGNORE);

				MPI_Request_free(&rSendBck);
				MPI_Request_free(&rRecvFwd);

				vPx = opCode(load_pd, tmp);
			} else {
				vPx = opCode(load_pd, &v[0]);
			}
		} else {
			vPx = opCode(load_pd, &v[idxVz]);
		}

		mDf = opCode(sub_pd, mPx, mel);
#ifdef	__AVX512__
		pMask = opCode(cmp_pd_mask, mDf, pVec, _CMP_GE_OQ);
		mMask = opCode(cmp_pd_mask, mDf, mVec, _CMP_LT_OQ);

		mPx = opCode(mask_sub_pd, mPx, pMask, mPx, cVec);
		mPx = opCode(mask_add_pd, mPx, mMask, mPx, cVec);
		vPx = opCode(mask_sub_pd, vPx, pMask, vPx, vVec);
		vPx = opCode(mask_add_pd, vPx, mMask, vPx, vVec);

		pMask |= mMask;

		for (int i=1, int k=0; k<step; i<<=1, k++)
					count += (pMask & i) >> k;
#else	// AVX and SSE4.1

#ifdef	__AVX__
		mDp = opCode(cmp_pd, mDf, pVec, _CMP_GE_OQ);
		mDm = opCode(cmp_pd, mDf, mVec, _CMP_LT_OQ);
#else
		mDp = opCode(cmpge_pd, mDf, pVec);
		mDm = opCode(cmplt_pd, mDf, mVec);
#endif
		mPx = opCode(sub_pd, mPx,
			opCode(sub_pd,
				opCode(and_pd, mDp, cVec),
				opCode(and_pd, mDm, cVec)));

		vPx = opCode(sub_pd, vPx,
			opCode(sub_pd,
				opCode(and_pd, mDp, vVec),
				opCode(and_pd, mDm, vVec)));

		mDp = opCode(or_pd, mDp, mDm);

		for (int k=0; k<step; k++)
			count += reinterpret_cast<size_t&>(mDp[k]) & 1;
#endif	// AVX and SSE4.1

		if (zDir == false) {
			if (nSplit > 1 ) {
				// Exchange ghosts to store the right m/v
				const int rank    = commRank();
				const int fwdNeig = (rank + 1) % nSplit;
				const int bckNeig = (rank - 1 + nSplit) % nSplit;
				MPI_Request rSendFwd, rRecvBck;

				double mTmp[step], vTmp[step];
				opCode(store_pd, mTmp, mPx);
				opCode(store_pd, vTmp, vPx);

				MPI_Send_init(mTmp, step, MPI_DOUBLE, fwdNeig, 2*rank,    MPI_COMM_WORLD, &rSendFwd);
				MPI_Recv_init(m,    step, MPI_DOUBLE, bckNeig, 2*bckNeig, MPI_COMM_WORLD, &rRecvBck);

				MPI_Start(&rRecvBck);
				MPI_Start(&rSendFwd);

				MPI_Wait(&rSendFwd, MPI_STATUS_IGNORE);
				MPI_Wait(&rRecvBck, MPI_STATUS_IGNORE);

				MPI_Request_free(&rSendFwd);
				MPI_Request_free(&rRecvBck);

				MPI_Send_init(vTmp, step, MPI_DOUBLE, fwdNeig, 2*rank,    MPI_COMM_WORLD, &rSendFwd);
				MPI_Recv_init(v,    step, MPI_DOUBLE, bckNeig, 2*bckNeig, MPI_COMM_WORLD, &rRecvBck);

				MPI_Start(&rRecvBck);
				MPI_Start(&rSendFwd);

				MPI_Wait(&rSendFwd, MPI_STATUS_IGNORE);
				MPI_Wait(&rRecvBck, MPI_STATUS_IGNORE);

				MPI_Request_free(&rSendFwd);
				MPI_Request_free(&rRecvBck);
			} else {
				opCode(store_pd, &m[Sf], mPx);
				opCode(store_pd, &v[0],  vPx);
			}
		} else {
			opCode(store_pd, &m[idxPz], mPx);
			opCode(store_pd, &v[idxVz], vPx);
		}
#undef  _MData_
#undef  step
	}
	else if (precision == FIELD_SINGLE)
	{
#ifdef  __AVX512F__
	#define _MData_ __m512
	#define step 16
#elif   defined(__AVX__)
	#define _MData_ __m256
	#define step 8
#else
	#define _MData_ __m128
	#define step 4
#endif
		float * __restrict__ m = (float * __restrict__) __builtin_assume_aligned (m_, Align);
		float * __restrict__ v = (float * __restrict__) __builtin_assume_aligned (v_, Align);

		const float zP = M_PI*z;
#ifdef  __AVX512F__
		const size_t XC = (Lx<<4);
		const size_t YC = (Lx>>4);

		const int    __attribute__((aligned(Align))) shfLf[16] = { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0};

		const auto vShLf  = opCode(load_si512, shfLf);
#elif   defined(__AVX__)
		const size_t XC = (Lx<<3);
		const size_t YC = (Lx>>3);
#else
		const size_t XC = (Lx<<2);
		const size_t YC = (Lx>>2);
#endif
		const _MData_ pVec  = opCode(set1_ps, +zP);
		const _MData_ mVec  = opCode(set1_ps, -zP);
		const _MData_ vVec  = opCode(set1_ps, 2.*M_PI);
		const _MData_ cVec  = opCode(set1_ps, zP*2.);

                _MData_ mel, mDf, mDp, mDm, mPx, vPx;

		for (size_t idx = Vo; idx < Vf; idx += step)
		{
			size_t X[2], idxPx, idxVx;

			{
				size_t tmi = idx/XC, tpi;

			        tpi = tmi/YC;
				X[1] = tmi - tpi*YC;
				X[0] = idx - tmi*XC;
			}

			if (X[0] == XC-step)
				idxPx = idx - XC + step;
			else
				idxPx = idx + step;

			idxVx = idxPx - Sf;

			mel = opCode(load_ps, &m[idx]);
			mPx = opCode(load_ps, &m[idxPx]);
			vPx = opCode(load_ps, &v[idxVx]);

			/*	X-Direction	*/

			mDf = opCode(sub_ps, mPx, mel);
#ifdef	__AVX512__
			auto pMask = opCode(cmp_pd_mask, mDf, pVec, _CMP_GE_OQ);
			auto mMask = opCode(cmp_pd_mask, mDf, mVec, _CMP_LT_OQ);

			mPx = opCode(mask_sub_ps, mPx, pMask, mPx, cVec);
			mPx = opCode(mask_add_ps, mPx, mMask, mPx, cVec);
			vPx = opCode(mask_sub_ps, vPx, pMask, vPx, vVec);
			vPx = opCode(mask_add_ps, vPx, mMask, vPx, vVec);

			pMask |= mMask;

			for (int i=1, int k=0; k<step; i<<=1, k++)
				count += (pMask & i) >> k;
#else	// AVX and SSE4.1

#ifdef	__AVX__
			mDp = opCode(cmp_ps, mDf, pVec, _CMP_GE_OQ);
			mDm = opCode(cmp_ps, mDf, mVec, _CMP_LT_OQ);
#else
			mDp = opCode(cmpge_ps, mDf, pVec);
			mDm = opCode(cmplt_ps, mDf, mVec);
#endif
			mPx = opCode(sub_ps, mPx,
				opCode(sub_ps,
					opCode(and_ps, mDp, cVec),
					opCode(and_ps, mDm, cVec)));

			vPx = opCode(sub_ps, vPx,
				opCode(sub_ps,
					opCode(and_ps, mDp, vVec),
					opCode(and_ps, mDm, vVec)));

			mDp = opCode(or_ps, mDp, mDm);

			for (int k=0; k<step; k++)
				count += reinterpret_cast<int&>(mDp[k]) & 1;
#endif	// AVX and SSE4.1
			opCode(store_ps, &v[idxVx], vPx);
			opCode(store_ps, &m[idxPx], mPx);
		}

		/*	Move forward in z	*/

		size_t idxPz = Vo + Sf, idxVz = Vo;

		mPx = opCode(load_ps, &m[idxPz]);

		const int nSplit  = commSize();

		if (zDir == false) {
			if (nSplit > 1) {
				// Exchange ghosts to get the right v
				const int rank    = commRank();
				const int nSplit  = commSize();
				const int fwdNeig = (rank + 1) % nSplit;
				const int bckNeig = (rank - 1 + nSplit) % nSplit;
				MPI_Request rSendBck, rRecvFwd;

				float tmp[step];

				MPI_Send_init(v,   step, MPI_FLOAT, bckNeig, 2*rank+1,    MPI_COMM_WORLD, &rSendBck);
				MPI_Recv_init(tmp, step, MPI_FLOAT, fwdNeig, 2*fwdNeig+1, MPI_COMM_WORLD, &rRecvFwd);
			
				MPI_Start(&rRecvFwd);
				MPI_Start(&rSendBck);

				MPI_Wait(&rSendBck, MPI_STATUS_IGNORE);
				MPI_Wait(&rRecvFwd, MPI_STATUS_IGNORE);

				MPI_Request_free(&rSendBck);
				MPI_Request_free(&rRecvFwd);

				vPx = opCode(load_ps, tmp);
			} else {
				vPx = opCode(load_ps, &v[0]);
			}
		} else {
			vPx = opCode(load_ps, &v[idxVz]);
		}

		mPx = opCode(load_ps, &m[idxPz]);

		mDf = opCode(sub_ps, mPx, mel);
#ifdef	__AVX512__
		pMask = opCode(cmp_ps_mask, mDf, pVec, _CMP_GE_OQ);
		mMask = opCode(cmp_ps_mask, mDf, mVec, _CMP_LT_OQ);

		mPx = opCode(mask_sub_ps, mPx, pMask, mPx, cVec);
		mPx = opCode(mask_add_ps, mPx, mMask, mPx, cVec);
		vPx = opCode(mask_sub_ps, vPx, pMask, vPx, vVec);
		vPx = opCode(mask_add_ps, vPx, mMask, vPx, vVec);

		pMask |= mMask;

		for (int i=1, int k=0; k<step; i<<=1, k++)
			count += (pMask & i) >> k;
#else	// AVX and SSE4.1

#ifdef	__AVX__
		mDp = opCode(cmp_ps, mDf, pVec, _CMP_GE_OQ);
		mDm = opCode(cmp_ps, mDf, mVec, _CMP_LT_OQ);
#else
		mDp = opCode(cmpge_ps, mDf, pVec);
		mDm = opCode(cmplt_ps, mDf, mVec);
#endif
		mPx = opCode(sub_ps, mPx,
			opCode(sub_ps,
				opCode(and_ps, mDp, cVec),
				opCode(and_ps, mDm, cVec)));

		vPx = opCode(sub_ps, vPx,
			opCode(sub_ps,
				opCode(and_ps, mDp, vVec),
				opCode(and_ps, mDm, vVec)));

		mDp = opCode(or_ps, mDp, mDm);

		for (int k=0; k<step; k++)
			count += reinterpret_cast<int&>(mDp[k]) & 1;
#endif	// AVX and SSE4.1

		if (zDir == false) {
			if (nSplit > 1 ) {
				// Exchange ghosts to store the right m/v
				const int rank    = commRank();
				const int fwdNeig = (rank + 1) % nSplit;
				const int bckNeig = (rank - 1 + nSplit) % nSplit;
				MPI_Request rSendFwd, rRecvBck;

				float mTmp[step], vTmp[step];
				opCode(store_ps, mTmp, mPx);
				opCode(store_ps, vTmp, vPx);

				MPI_Send_init(mTmp, step, MPI_FLOAT, fwdNeig, 2*rank,    MPI_COMM_WORLD, &rSendFwd);
				MPI_Recv_init(m,    step, MPI_FLOAT, bckNeig, 2*bckNeig, MPI_COMM_WORLD, &rRecvBck);

				MPI_Start(&rRecvBck);
				MPI_Start(&rSendFwd);

				MPI_Wait(&rSendFwd, MPI_STATUS_IGNORE);
				MPI_Wait(&rRecvBck, MPI_STATUS_IGNORE);

				MPI_Request_free(&rSendFwd);
				MPI_Request_free(&rRecvBck);

				MPI_Send_init(vTmp, step, MPI_FLOAT, fwdNeig, 2*rank,    MPI_COMM_WORLD, &rSendFwd);
				MPI_Recv_init(v,    step, MPI_FLOAT, bckNeig, 2*bckNeig, MPI_COMM_WORLD, &rRecvBck);

				MPI_Start(&rRecvBck);
				MPI_Start(&rSendFwd);

				MPI_Wait(&rSendFwd, MPI_STATUS_IGNORE);
				MPI_Wait(&rRecvBck, MPI_STATUS_IGNORE);

				MPI_Request_free(&rSendFwd);
				MPI_Request_free(&rRecvBck);
			} else {
				opCode(store_ps, &m[Sf], mPx);
				opCode(store_ps, &v[0],  vPx);
			}
		} else {
			opCode(store_ps, &m[idxPz], mPx);
			opCode(store_ps, &v[idxVz], vPx);
		}

#undef  _MData_
#undef  step
	}
}

template<typename Float>
inline  size_t	mendThetaSingle(Float * __restrict__ m, Float * __restrict__ v, const double z, const size_t Lx, const size_t Sf, const size_t Vo, const size_t Vf, const int step)
{
	const double zP = M_PI*z;
	size_t count = 0;

	Float mDf, mel[step], mPx[step], vPx[step], mPy[step], vPy[step], mPz[step], vPz[step];

	int shf = 0, cnt = step;

	while (cnt != 1) {
		cnt >>= 1;
		shf++;
	}
		
	const size_t XC = (Lx<<shf);
	const size_t YC = (Lx>>shf);

	for (size_t idx = Vo; idx < Vf; idx += step)
	{
		memcpy (&mel[0], &m[idx], step*sizeof(Float));

		size_t X[2], idxPx, idxPy, idxPz = idx + Sf, idxVx, idxVy, idxVz = idx;

		{
			size_t tmi = idx/XC, tpi;

		        tpi = tmi/YC;
			X[1] = tmi - tpi*YC;
			X[0] = idx - tmi*XC;
		}

		if (X[0] == XC-step)
			idxPx = idx - XC + step;
		else
			idxPx = idx + step;

		idxVx = idxPx - Sf;

		if (X[1] == YC-1)
		{
			idxPy = idx - Sf + XC;
			idxVy = idxPy - Sf;

			memcpy (&mPy[0], &m[idxPy], step*sizeof(Float));
			memcpy (&vPy[0], &v[idxVy], step*sizeof(Float));

			Float mSave = mPy[0];
			Float vSave = vPy[0];

			for (int i = 0; i < step-1; i++) {
				mPy[i] = mPy[i+1];
				vPy[i] = vPy[i+1];
			}

			mPy[step-1] = mSave;
			vPy[step-1] = vSave;
		} else {
			idxPy = idx + XC;
			idxVy = idxPy - Sf;

			memcpy (&mPy[0], &m[idxPy], step*sizeof(Float));
			memcpy (&vPy[0], &v[idxVy], step*sizeof(Float));
		}

		memcpy (&mPx[0], &m[idxPx], step*sizeof(Float));
		memcpy (&vPx[0], &v[idxVx], step*sizeof(Float));
		memcpy (&mPz[0], &m[idxPz], step*sizeof(Float));
		memcpy (&vPz[0], &v[idxVz], step*sizeof(Float));

		/*	Vector loop	*/
		for (int i=0; i<step; i++) {

			/*	X-Direction	*/

			mDf = mPx[i] - mel[i];

			if (mDf > zP) {
				mPx[i] -= zP;
				vPx[i] -= 2.*M_PI;
				m[idxPx + i] = mPx[i];
				v[idxVx + i] = vPx[i];
				count++;
			} else if (mDf < -zP) {
				mPx[i] += zP;
				vPx[i] += 2.*M_PI;
				m[idxPx + i] = mPx[i];
				v[idxVx + i] = vPx[i];
				count++;
			}

			/*	Y-Direction	*/

			mDf = mPy[i] - mel[i];

			if (mDf > zP) {
				mPy[i] -= zP;
				vPy[i] -= 2.*M_PI;
				m[idxPy + i] = mPy[i];
				v[idxVy + i] = vPy[i];
				count++;
			} else if (mDf < -zP) {
				mPy[i] += zP;
				vPy[i] += 2.*M_PI;
				m[idxPy + i] = mPy[i];
				v[idxVy + i] = vPy[i];
				count++;
			}

			/*	Z-Direction	*/

			mDf = mPz[i] - mel[i];

			if (mDf > zP) {
				mPz[i] -= zP;
				vPz[i] -= 2.*M_PI;
				m[idxPz + i] = mPz[i];
				v[idxVz + i] = vPz[i];
				count++;
			} else if (mDf < -zP) {
				mPz[i] += zP;
				vPz[i] += 2.*M_PI;
				m[idxPz + i] = mPz[i];
				v[idxVz + i] = vPz[i];
				count++;
			}

		}

		if (X[1] == YC-1)
		{
			Float mSave = mPy[step-1];
			Float vSave = vPy[step-1];

			for (int i = 1; i < step; i++) {
				m[idxPy + i] = mPy[i-1];
				v[idxVy + i] = vPy[i-1];
			}

			m[idxPy] = mSave;
			v[idxVy] = vSave;
		}
	}
}

template<typename Float>
inline  size_t	mendThetaLine(Float * __restrict__ m, Float * __restrict__ v, const double z, const size_t Lx, const size_t Sf, const size_t slice, const int step)
{
	const double zP = M_PI*z;
	size_t count = 0;

	Float mDf, mel, mPy, vPy;

	int shf = 0, cnt = step;

	while (cnt != 1) {
		cnt >>= 1;
		shf++;
	}
		
	const size_t XC = (Lx<<shf);
	const size_t YC = (Lx>>shf);

	size_t cIdx, idxPy, idxVy, idx = slice*Sf;

	/*	Vector loop	*/
	for (int i=0; i<step; i++) {
		for (size_t lPos = 0; lPos < YC-1; lPos++) {
			size_t cIdx = idx + lPos*XC;

			mel = m[cIdx + i];

			idxPy = cIdx + XC;
			idxVy = idxPy - Sf;

			mPy = m[idxPy + i];
			vPy = v[idxVy + i];

			/*	Y-Direction	*/

			mDf = mPy - mel;

			if (mDf > zP) {
				mPy -= 2.*zP;
				vPy -= 2.*M_PI;
				m[idxPy + i] = mPy;
				v[idxVy + i] = vPy;
				count++;
			} else if (mDf < -zP) {
				mPy += 2.*zP;
				vPy += 2.*M_PI;
				m[idxPy + i] = mPy;
				v[idxVy + i] = vPy;
				count++;
			}
		}

		cIdx  = idx + Sf - XC;
		idxPy = idx;
		idxVy = idxPy - Sf;

		mPy = m[idxPy + ((i + 1)%step)];
		vPy = v[idxVy + ((i + 1)%step)];

		if (slice == 1) {
			printf ("(%lu --> %lu) %d (%d) %f %f %f\n", cIdx, idxPy, i, ((i+1)%step), mel, mPy, vPy);
		}

		/*	Y-Direction	*/

		mDf = mPy - mel;

		if (mDf > zP) {
			mPy -= 2.*zP;
			vPy -= 2.*M_PI;
			m[idxPy + ((i + 1)%step)] = mPy;
			v[idxVy + ((i + 1)%step)] = vPy;
			count++;
		} else if (mDf < -zP) {
			mPy += 2.*zP;
			vPy += 2.*M_PI;
			m[idxPy + ((i + 1)%step)] = mPy;
			v[idxVy + ((i + 1)%step)] = vPy;
			count++;
		}
	}

	return	count;
}

size_t	mendSliceXeon (Scalar *field, size_t slice)
{
	const double z  = *(field->zV());
	const size_t Sf =   field->Surf();

	size_t tJmps = 0;

	switch (field->Precision()) {
		case FIELD_DOUBLE:
		field->exchangeGhosts(FIELD_M);
		tJmps  = mendThetaLine  <double>(static_cast<double*>(field->mCpu()), static_cast<double*>(field->vCpu()), z, field->Length(), Sf, slice, Align/sizeof(double));
		break;

		case FIELD_SINGLE:
		field->exchangeGhosts(FIELD_M);
		tJmps  = mendThetaLine  <float> (static_cast<float *>(field->mCpu()), static_cast<float *>(field->vCpu()), z, field->Length(), Sf, slice, Align/sizeof(float));
		break;
	}

	return	tJmps;
}

bool	mendThetaXeon (Scalar *field)
{
	const double	z     = *(field->zV());
	size_t		tJmps = 0;
	size_t		cIdx  = 0;
	bool		wJmp  = false;

	for (size_t i=0; i<field->Depth(); i++) {
		do {
			tJmps  = mendSliceXeon(field, i);	// Updates the x=cte,z=cte line, so we can vectorize
			tJmps += mendThetaKernelXeon<true>(field->mCpu(), field->vCpu(), z, field->Length(), field->Depth(), cIdx, cIdx + field->Surf(), field->Precision());

			if (tJmps)
				wJmp = true;
		}	while	(tJmps != 0);

		cIdx += field->Surf();
	}

	do {
		tJmps  = mendSliceXeon(field, field->Depth());
		field->exchangeGhosts(FIELD_M);
		tJmps += mendThetaKernelXeon<false>(field->mCpu(), field->vCpu(), z, field->Length(), field->Depth(), cIdx, cIdx + field->Surf(), field->Precision());

		if (tJmps)
			wJmp = true;
	}	while	(tJmps != 0);

	return	wJmp;
}

