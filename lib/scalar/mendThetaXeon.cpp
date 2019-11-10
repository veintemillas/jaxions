#include <cstdio>
#include <cmath>
#include "scalar/scalarField.h"
#include "enum-field.h"
//#include "scalar/varNQCD.h"
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

/*	Connects all the points in a ZY plane with the whole volume	*/
/*	Parallelizes on ZY and vectorizes on Y				*/

inline  size_t	mendThetaKernelXeon(void * __restrict__ m_, void * __restrict__ v_, const double z, const size_t Lx, const size_t Lz, const size_t Sf, FieldPrecision precision)
{
	size_t count = 0;

        if (precision == FIELD_DOUBLE)
        {
#ifdef  __AVX512F__
	#define _MData_ __m512d
	#define step 8
	#define shf  3
#elif   defined(__AVX__)
	#define _MData_ __m256d
	#define step 4
	#define shf  2
#else
	#define _MData_ __m128d
	#define step 2
	#define shf  1
#endif

		double * __restrict__ m = (double * __restrict__) __builtin_assume_aligned (m_, Align);
		double * __restrict__ v = (double * __restrict__) __builtin_assume_aligned (v_, Align);

		const double zP = M_PI*z;

		const size_t XC = (Lx<<shf);
		const size_t YC = (Lx>>shf);

		const _MData_ pVec  = opCode(set1_pd, +zP);
		const _MData_ mVec  = opCode(set1_pd, -zP);
		const _MData_ vVec  = opCode(set1_pd, 2.*M_PI);
		const _MData_ cVec  = opCode(set1_pd, zP*2.);

		#pragma omp parallel default(shared) reduction(+:count)
		{
			size_t	idx, idxPx;
			_MData_ mel, mDf, mDc, mDp, mDm, mPx, vPx;

			/*	Collapse loops so OpenMP handles the plane YZ	*/
			#pragma omp for collapse(2) schedule(static)
			for (size_t zSl = 0; zSl < Lz; zSl++)
				for (size_t yLn = 0; yLn < YC; yLn++) {
					/*	Bulk loop in X	*/
					for(size_t xPt = 0; xPt < XC-step; xPt += step) {
						idx   = zSl*Sf + yLn*XC + xPt;
						idxPx = idx + step;

						mel = opCode(load_pd, &m[idx]);
						mPx = opCode(load_pd, &m[idxPx]);
						vPx = opCode(load_pd, &v[idxPx]);

						/*	X-Direction	*/

						mDf  = opCode(sub_pd, mPx, mel);
#ifdef	__AVX512F__
						auto pMask = opCode(cmp_pd_mask, mDf, pVec, _CMP_GE_OQ);
						auto mMask = opCode(cmp_pd_mask, mDf, mVec, _CMP_LT_OQ);
						auto mask = pMask | mMask;

						while	(mask != 0) {
							mPx = opCode(mask_sub_pd, mPx, pMask, mPx, cVec);
							vPx = opCode(mask_sub_pd, vPx, pMask, vPx, vVec);
							mPx = opCode(mask_add_pd, mPx, mMask, mPx, cVec);
							vPx = opCode(mask_add_pd, vPx, mMask, vPx, vVec);

							for (int k=0,i=1; k<step; k++,i<<=1)
								count += (mask & i) >> k;

							mDf  = opCode(sub_pd, mPx, mel);

							auto pMask = opCode(cmp_pd_mask, mDf, pVec, _CMP_GE_OQ);
							auto mMask = opCode(cmp_pd_mask, mDf, mVec, _CMP_LT_OQ);
							mask = pMask | mMask;
						}
#else
						size_t mask = 0;
						size_t msk[step];
	#ifdef	__AVX__
						mDp = opCode(cmp_pd, mDf, pVec, _CMP_GE_OQ);
						mDm = opCode(cmp_pd, mDf, mVec, _CMP_LT_OQ);
	#else
						mDp = opCode(cmpge_pd, mDf, pVec);
						mDm = opCode(cmplt_pd, mDf, mVec);
	#endif
						mDc = opCode(or_pd, mDp, mDm);
						opCode(store_pd, static_cast<double*>(static_cast<void*>(msk)), mDc);

						for (int k=0; k<step; k++)
							mask += msk[k] & 1;

						while	(mask != 0) {
							mPx = opCode(sub_pd, mPx, opCode(and_pd, mDp, cVec));
							vPx = opCode(sub_pd, vPx, opCode(and_pd, mDp, vVec));
							mPx = opCode(add_pd, mPx, opCode(and_pd, mDm, cVec));
							vPx = opCode(add_pd, vPx, opCode(and_pd, mDm, vVec));

							count += mask;

							mDf = opCode(sub_pd, mPx, mel);
	#ifdef	__AVX__
							mDp = opCode(cmp_pd, mDf, pVec, _CMP_GE_OQ);
							mDm = opCode(cmp_pd, mDf, mVec, _CMP_LT_OQ);
	#else
							mDp = opCode(cmpge_pd, mDf, pVec);
							mDm = opCode(cmplt_pd, mDf, mVec);
	#endif
							mDc = opCode(or_pd, mDp, mDm);

							opCode(store_pd, static_cast<double*>(static_cast<void*>(msk)), mDc);

							mask = 0;
							for (int k=0; k<step; k++)
								mask += msk[k] & 1;
						}
#endif	// AVX and SSE4.1

						opCode(store_pd, &m[idxPx], mPx);
						opCode(store_pd, &v[idxPx], vPx);
					}

					/*	Boundary	*/
					// I don't think the boundary is necessary
				}
		}	// End of parallel region

#undef  _MData_
#undef  step
#undef  shf
	}
	else if (precision == FIELD_SINGLE)
	{
#ifdef  __AVX512F__
	#define _MData_ __m512
	#define step 16
	#define shf  4
#elif   defined(__AVX__)
	#define _MData_ __m256
	#define step 8
	#define shf  3
#else
	#define _MData_ __m128
	#define step 4
	#define shf  2
#endif
		float * __restrict__ m = (float * __restrict__) __builtin_assume_aligned (m_, Align);
		float * __restrict__ v = (float * __restrict__) __builtin_assume_aligned (v_, Align);

		const float zP = M_PI*z;

		const size_t XC = (Lx<<shf);
		const size_t YC = (Lx>>shf);

		const _MData_ pVec  = opCode(set1_ps, +zP);
		const _MData_ mVec  = opCode(set1_ps, -zP);
		const _MData_ vVec  = opCode(set1_ps, 2.f*M_PI);
		const _MData_ cVec  = opCode(set1_ps, zP*2.f);

		#pragma omp parallel default(shared) reduction(+:count)
		{
			size_t	idx, idxPx;
			_MData_ mel, mDf, mDc, mDp, mDm, mPx, vPx;

			/*	Collapse loops so OpenMP handles the plane YZ	*/
			#pragma omp for collapse(2) schedule(static)
			for (size_t zSl = 0; zSl < Lz; zSl++) {
				for (size_t yLn = 0; yLn < YC; yLn++) {
					/*	Bulk loop in X	*/
					for(size_t xPt = 0; xPt < XC-step; xPt += step) {
						idx   = zSl*Sf + yLn*XC + xPt;
						idxPx = idx + step;

						mel = opCode(load_ps, &m[idx]);
						mPx = opCode(load_ps, &m[idxPx]);
						vPx = opCode(load_ps, &v[idxPx]);

						/*	X-Direction	*/
						mDf  = opCode(sub_ps, mPx, mel);
#ifdef	__AVX512F__
						auto pMask = opCode(cmp_ps_mask, mDf, pVec, _CMP_GE_OQ);
						auto mMask = opCode(cmp_ps_mask, mDf, mVec, _CMP_LT_OQ);
						auto mask  = opCode(kor, pMask, mMask);

						int  nChg = 0;
						for (int k=0,i=1; k<step; k++,i<<=1)
							nChg += (mask & i) >> k;

						while	(nChg != 0) {
							mPx = opCode(mask_sub_ps, mPx, pMask, mPx, cVec);
							vPx = opCode(mask_sub_ps, vPx, pMask, vPx, vVec);
							mPx = opCode(mask_add_ps, mPx, mMask, mPx, cVec);
							vPx = opCode(mask_add_ps, vPx, mMask, vPx, vVec);

							count += nChg;

							mDf  = opCode(sub_ps, mPx, mel);

							pMask = opCode(cmp_ps_mask, mDf, pVec, _CMP_GE_OQ);
							mMask = opCode(cmp_ps_mask, mDf, mVec, _CMP_LT_OQ);
							mask  = opCode(kor, pMask, mMask);

							nChg = 0;
							for (int k=0,i=1; k<step; k++,i<<=1)
								nChg += (mask & i) >> k;
						}
#else
						int mask = 0;
						int msk[step];
	#ifdef	__AVX__
						mDp = opCode(cmp_ps, mDf, pVec, _CMP_GE_OQ);
						mDm = opCode(cmp_ps, mDf, mVec, _CMP_LT_OQ);
	#else
						mDp = opCode(cmpge_ps, mDf, pVec);
						mDm = opCode(cmplt_ps, mDf, mVec);
	#endif
						mDc = opCode(or_ps, mDp, mDm);
						opCode(store_ps, static_cast<float*>(static_cast<void*>(msk)), mDc);

						for (int k=0; k<step; k++)
							mask += msk[k] & 1;

						while	(mask != 0) {
							mPx = opCode(sub_ps, mPx, opCode(and_ps, mDp, cVec));
							vPx = opCode(sub_ps, vPx, opCode(and_ps, mDp, vVec));
							mPx = opCode(add_ps, mPx, opCode(and_ps, mDm, cVec));
							vPx = opCode(add_ps, vPx, opCode(and_ps, mDm, vVec));

							count += mask;

							mDf = opCode(sub_ps, mPx, mel);
	#ifdef	__AVX__
							mDp = opCode(cmp_ps, mDf, pVec, _CMP_GE_OQ);
							mDm = opCode(cmp_ps, mDf, mVec, _CMP_LT_OQ);
	#else
							mDp = opCode(cmpge_ps, mDf, pVec);
							mDm = opCode(cmplt_ps, mDf, mVec);
	#endif
							mDc = opCode(or_ps, mDp, mDm);
							opCode(store_ps, static_cast<float*>(static_cast<void*>(msk)), mDc);

							mask = 0;
							for (int k=0; k<step; k++)
								mask += msk[k] & 1;
						}
#endif
						opCode(store_ps, &m[idxPx], mPx);
						opCode(store_ps, &v[idxPx], vPx);
					}
				}
			}
		}
#undef  _MData_
#undef  step
#undef  shf
	}

	return	count;
}

/*	Connects all the points in a Z line with the whole ZY plane	*/
/*	Parallelizes on Z						*/

template<typename Float, const int step>
inline  size_t	mendThetaSlice(Float * __restrict__ m, Float * __restrict__ v, const double z, const size_t Lx, const size_t Lz, const size_t Sf, size_t NN)
{
	const double zP = M_PI*z;
	size_t count = 0;

	size_t idx, idxPy, idxVy;
	Float mDf, mel, mPy, vPy;

	int shf = 0, cnt = step;

	while (cnt != 1) {
		cnt >>= 1;
		shf++;
	}

	const size_t XC = (Lx<<shf);
	const size_t YC = (Lx>>shf);

	/*	We go parallel on Z	*/
	#pragma omp parallel for private(idx,idxPy,idxVy,mDf,mel,mPy,vPy) reduction(+:count) schedule(static)
	for (size_t zSl = NN; zSl < Lz+NN; zSl++) {
		/*	Vectorization goes serial	*/
		for (int vIdx = 0; vIdx < step; vIdx++) {
			/*	Bulk Y, X=0 always	*/
			for (size_t yPt = 0; yPt < YC - 1; yPt++) {
				idx   = zSl*Sf + yPt*XC + vIdx;
				idxPy = idx + XC;
				idxVy = idxPy - Sf*NN;

				mel = m[idx];
				mPy = m[idxPy];
				vPy = v[idxVy];

				mDf = mPy - mel;

				while (abs(mDf) > zP)
				{
					if (mDf > zP) {
						mPy -= 2.*zP;
						vPy -= 2.*M_PI;
						count++;
					} else if (mDf < zP) {
						mPy += 2.*zP;
						vPy += 2.*M_PI;
						count++;
					}

					mDf = mPy - mel;
				}

				m[idxPy] = mPy;
				v[idxVy] = vPy;
			}
#if 1
			/*	Border, needs shifts	*/
			idx    = zSl*Sf;
			idxPy  = idx + ((vIdx+1) % step);
			idxVy  = idxPy - Sf*NN;
			idx   += Sf - XC + vIdx;

			mel = m[idx];
			mPy = m[idxPy];
			vPy = v[idxVy];

			mDf = mPy - mel;

			while (abs(mDf) > zP)
			{
				if (mDf > zP) {
					mPy -= 2.*zP;
					vPy -= 2.*M_PI;
					count++;
				} else if (mDf < zP) {
					mPy += 2.*zP;
					vPy += 2.*M_PI;
					count++;
				}

				mDf = mPy - mel;
			}

			m[idxPy] = mPy;
			v[idxVy] = vPy;
#endif
		}
	}

	return	count;
}

template<typename Float>
inline  size_t	mendThetaLine(Float * __restrict__ m, Float * __restrict__ v, const double z, const size_t Lz, const size_t Sf, size_t NN)
{
	const double zP = M_PI*z;
	size_t count = 0;

	Float mDf, mel, mPz, vPz;
	size_t idxPz, idxVz;

	/*	For MPI		*/
	const int nSplit  = commSize();
	const int rank    = commRank();
	//const int fwdNeig = (rank + 1) % nSplit;
	const int bckNeig = (rank - 1 + nSplit) % nSplit;


	for (int cRank = 0; cRank < commSize(); cRank++) {

		commSync();

		//const int cFwdNeig = (cRank + 1) % nSplit;
		const int cBckNeig = (cRank - 1 + nSplit) % nSplit;

		/*	Get the ghosts for slice 0							*/
		/*	It's cumbersome but we avoid exchanging the whole slice to get one point	*/

		if (commSize() == 1) {
			m[Sf*(NN-1)] = m[Sf*(Lz+NN-1)];
		} else {
			if (rank == cBckNeig) {
				MPI_Send(&m[Sf*(Lz+NN-1)], sizeof(Float), MPI_CHAR, cRank,   cRank, MPI_COMM_WORLD);
			}

			if (rank == cRank) {
				MPI_Recv(&m[Sf*(NN-1)]  , sizeof(Float), MPI_CHAR, bckNeig, cRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}

		/*	We run only one rank at a time, otherwise we can enter an infinite	*/
		/*	loop where one rank undoes what another did				*/

		if (rank == cRank) {
			for (size_t idx=0,i=0; i<Lz-1; i++,idx+=Sf) {

				idxPz = idx + Sf*NN;

				mel = m[idx + Sf*(NN-1)];

				mPz = m[idxPz];
				vPz = v[idx];

				/*	Z-Direction	*/

				mDf = mPz - mel;

				while (abs(mDf) > zP) {
					if (mDf > zP) {
						mPz -= 2.*zP;
						vPz -= 2.*M_PI;
						count++;
					} else {
						mPz += 2.*zP;
						vPz += 2.*M_PI;
						count++;
					}

					mDf = mPz - mel;
				}

				m[idxPz] = mPz;
				v[idx] = vPz;
			}
		}
	}

	return	count;
}

size_t	mendThetaXeon (Scalar *field)
{
	constexpr int	dStep = Align/sizeof(double);
	constexpr int	fStep = Align/sizeof(float);
	const double	z     = *(field->zV());
	size_t		tJmp = 0;

	switch (field->Precision()) {
		case	FIELD_DOUBLE:
		tJmp += mendThetaLine(static_cast<double*>(field->mCpu()), static_cast<double*>(field->vCpu()), z, field->Depth(), field->Surf(), field->getNg());
		tJmp += mendThetaSlice<double, dStep>(static_cast<double*>(field->mCpu()), static_cast<double*>(field->vCpu()), z, field->Length(), field->Depth(), field->Surf(), field->getNg());
		tJmp += mendThetaKernelXeon(field->mStart(), field->vCpu(), z, field->Length(), field->Depth(), field->Surf(), field->Precision());
		break;

		case	FIELD_SINGLE:
		tJmp += mendThetaLine(static_cast<float *>(field->mCpu()), static_cast<float *>(field->vCpu()), z, field->Depth(), field->Surf(), field->getNg());
		tJmp += mendThetaSlice<float, fStep>(static_cast<float *>(field->mCpu()), static_cast<float *>(field->vCpu()), z, field->Length(), field->Depth(), field->Surf(), field->getNg());
		tJmp += mendThetaKernelXeon(field->mStart(), field->vCpu(), z, field->Length(), field->Depth(), field->Surf(), field->Precision());
		break;

		default:
		break;
	}
	LogOut("mendTheta done mends = %lu\n",tJmp);
	return	tJmp;
}
