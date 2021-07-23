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
/*	TODO undo pinches */
/*		Current method uses scan in both +/- directions in X
			only mends if both directions give the same answer
			this will not mend correctly a patch if a line contains also a pinch */

/* Uses m2 as aux field to store F and B on a line/line basis
distribution is n-th thread nth 4 lines of m2 */
inline  size_t	mendThetaKernelXeon(void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_, void * __restrict__ s_, const double R, const size_t Lx, const size_t Lz, const size_t Sf, FieldPrecision precision)
{
	size_t countF = 0;
	size_t countB = 0;
	size_t count  = 0;
	size_t DLz    = commRank()*Lz;

	/* check that m2 is sufficient,
	we assume m2 is still the complex field with only 2 ghost regions, the minimum */
	if (omp_get_num_threads()*4 > 2*Sf*(Lz+2))
		LogError("Insufficient size of m2 for mendTheta. Race conditions possible and likely.");

	char *strdaa = static_cast<char *>(s_);
	#pragma omp parallel for schedule(static)
	for (size_t idx = 0; idx < Sf*Lz; idx++)
		strdaa[idx] = STRING_NOTHING;

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

		double * __restrict__ m  = (double * __restrict__) __builtin_assume_aligned (m_, Align);
		double * __restrict__ v  = (double * __restrict__) __builtin_assume_aligned (v_, Align);
		double * __restrict__ m2 = (double * __restrict__) __builtin_assume_aligned (m2_, Align);

		const double zP = M_PI*R;

		const size_t XC = (Lx<<shf);
		const size_t YC = (Lx>>shf);

		const _MData_ pVec  = opCode(set1_pd, +zP);
		const _MData_ mVec  = opCode(set1_pd, -zP);
		const _MData_ vVec  = opCode(set1_pd, 2.*M_PI);
		const _MData_ cVec  = opCode(set1_pd, zP*2.);

		#pragma omp parallel default(shared) reduction(+:countF,countB,count)
		{
			size_t	idx, idxPx, Vo, idx2, idx2B, iNx, iNxB;
			_MData_ mel, mDf, mDc, mDp, mDm, mPx, vPx;
			size_t	idxB, idxBMx;
			_MData_ melB, mBMx, mBDf, mBDp, mBDm, mBDc, vBMx;

			int nThread = omp_get_thread_num();
			size_t thread0 = nThread*4*XC;

			/*	Collapse loops so OpenMP handles the plane YZ	*/
			#pragma omp for collapse(2) schedule(static)
			for (size_t zSl = 0; zSl < Lz; zSl++) {
				for (size_t yLn = 0; yLn < YC; yLn++) {
					Vo = zSl*Sf;

					/*	Bulk loop in X	*/
					bool cha = false;
					size_t countF_line = 0;
					size_t countB_line = 0;
					/* copy m[0] into m2[0] and m2[0+2Sf]
					because we will use m2 as references to build F anf B
					mended lines */
					mel = opCode(load_pd, &m[Vo+yLn*XC]);
					opCode(store_pd, &m2[thread0], mel);
					opCode(store_pd, &m2[thread0+2*XC], mel);

					for(size_t xPt = 0; xPt < XC-step; xPt += step) {
						/* Forward (x,y,z) vs (x+step,y,z) */
						idx   = thread0 + xPt;                 // in m2 I do not need z,y info, just thread0
						idx2  = idx + step;                    // in m2
						idxPx = Vo + yLn*XC + xPt + step;      // in m I need the full index
						iNx   = (xPt/step + 1 + (yLn)*Lx + Vo); // will miss + vectorindex*YC*Lx
						mel = opCode(load_pd, &m2[idx]);        // note we read from m2
						mPx = opCode(load_pd, &m[idxPx]);
// TODO load only if neccesary?
						vPx = opCode(load_pd, &v[idxPx]);


						/* Backward (L-x,y,z) vs (L-x-step,y,z) */
						idx2B  = thread0 + 3*XC - xPt - step;
						idxB   = idx2B + step;
						if (xPt == 0)
								idxB -= XC; // wrap around the boundary
						idxBMx = Vo + (yLn+1)*XC - xPt - step;
						iNxB   = ((yLn+1)*Lx + Vo -xPt/step - 1);  // will miss + vectorindex*YC*Lx
						melB = opCode(load_pd, &m2[idxB]); // note we read the ref from the copy
						mBMx = opCode(load_pd, &m[idxBMx]);
						vBMx = opCode(load_pd, &v[idxBMx]);

						/*	X-Direction	*/

						mDf  = opCode(sub_pd, mPx, mel);
						mBDf = opCode(sub_pd, mBMx, melB);
#ifdef	__AVX512F__
						/* Forward */
						{
						auto pMask = opCode(cmp_pd_mask, mDf, pVec, _CMP_GE_OQ);
						auto mMask = opCode(cmp_pd_mask, mDf, mVec, _CMP_LT_OQ);
						auto mask = pMask | mMask;

						int  nChg = 0;
						for (int k=0,i=1; k<step; k++,i<<=1) {
							nChg += (mask & i) >> k;
							if ( (mask & i) >> k ) strdaa[iNx+k*YC*Lx]  = 1;
						}
						if (nChg > 0) cha = true;

						while	(mask != 0) {
							mPx = opCode(mask_sub_pd, mPx, pMask, mPx, cVec);
							vPx = opCode(mask_sub_pd, vPx, pMask, vPx, vVec);
							mPx = opCode(mask_add_pd, mPx, mMask, mPx, cVec);
							vPx = opCode(mask_add_pd, vPx, mMask, vPx, vVec);

							for (int k=0,i=1; k<step; k++,i<<=1)
								countF_line += (mask & i) >> k;

							mDf  = opCode(sub_pd, mPx, mel);

							auto pMask = opCode(cmp_pd_mask, mDf, pVec, _CMP_GE_OQ);
							auto mMask = opCode(cmp_pd_mask, mDf, mVec, _CMP_LT_OQ);
							mask = pMask | mMask;
						}

						/* Backward */
						pMask = opCode(cmp_pd_mask, mBDf, pVec, _CMP_GE_OQ);
						mMask = opCode(cmp_pd_mask, mBDf, mVec, _CMP_LT_OQ);
						mask = pMask | mMask;

						nChg = 0;
						for (int k=0,i=1; k<step; k++,i<<=1) {
							nChg += (mask & i) >> k;
							if ( (mask & i) >> k ) strdaa[iNxB+k*YC*Lx]  = 1;
						}
						if (nChg > 0) cha = true;

						while	(mask != 0) {
							mBMx = opCode(mask_sub_pd, mBMx, pMask, mBMx, cVec);
							vBMx = opCode(mask_sub_pd, vBMx, pMask, vBMx, vVec);
							mBMx = opCode(mask_add_pd, mBMx, mMask, mBMx, cVec);
							vBMx = opCode(mask_add_pd, vBMx, mMask, vBMx, vVec);

							for (int k=0,i=1; k<step; k++,i<<=1)
								countB_line += (mask & i) >> k;

							mBDf  = opCode(sub_pd, mBMx, melB);

							auto pMask = opCode(cmp_pd_mask, mBDf, pVec, _CMP_GE_OQ);
							auto mMask = opCode(cmp_pd_mask, mBDf, mVec, _CMP_LT_OQ);
							mask = pMask | mMask;
						}
						}
#else
						long mask = 0;
						long msk[step];
						long maskB = 0;
						long mskB[step];
	#ifdef	__AVX__
						mDp = opCode(cmp_pd, mDf, pVec, _CMP_GE_OQ);
						mDm = opCode(cmp_pd, mDf, mVec, _CMP_LT_OQ);
						mBDp = opCode(cmp_pd, mBDf, pVec, _CMP_GE_OQ);
						mBDm = opCode(cmp_pd, mBDf, mVec, _CMP_LT_OQ);

	#else
						mDp = opCode(cmpge_pd, mDf, pVec);
						mDm = opCode(cmplt_pd, mDf, mVec);
						mBDp = opCode(cmpge_pd, mBDf, pVec);
						mBDm = opCode(cmplt_pd, mBDf, mVec);
	#endif
						mDc = opCode(or_pd, mDp, mDm);
						mBDc = opCode(or_pd, mBDp, mBDm);
						opCode(store_pd, static_cast<double*>(static_cast<void*>(msk)), mDc);
						opCode(store_pd, static_cast<double*>(static_cast<void*>(mskB)), mBDc);

						for (int k=0; k<step; k++) {
							mask += msk[k] & 1;
							maskB += mskB[k] & 1;
							if (msk[k]  & 1) strdaa[iNx+k*YC*Lx]  = 1;
							if (mskB[k] & 1) strdaa[iNxB+k*YC*Lx] = 2;
						}
						if ((mask > 0) || (maskB > 0)) cha = true;


// if (zSl == 128 && yLn == 0)
// {
// 	double ma[step], me[step], mi[step], mo[step], mu[step], my[step];
// 	opCode(store_pd, static_cast<double*>(static_cast<void*>(ma)), mel);
// 	opCode(store_pd, static_cast<double*>(static_cast<void*>(me)), mPx);
// 	opCode(store_pd, static_cast<double*>(static_cast<void*>(mi)), melB);
// 	opCode(store_pd, static_cast<double*>(static_cast<void*>(mo)), mBMx);
// 	opCode(store_pd, static_cast<double*>(static_cast<void*>(mu)), mDf);
// 	opCode(store_pd, static_cast<double*>(static_cast<void*>(my)), mBDf);
// 	for (int k=0; k<step; k++) {
// 		printf("%d) %.5f %.5f %.5f %ld",k, ma[k]/R, me[k]/R, mu[k]/R, msk[k]);
// 		printf("| %.5f %.5f %.5f %ld -> cha %d\n ",mi[k]/R, mo[k]/R, my[k]/R, mskB[k], cha);
// }}
						/* Forward mask */
						while	(mask != 0) {
							mPx = opCode(sub_pd, mPx, opCode(and_pd, mDp, cVec));
							vPx = opCode(sub_pd, vPx, opCode(and_pd, mDp, vVec));
							mPx = opCode(add_pd, mPx, opCode(and_pd, mDm, cVec));
							vPx = opCode(add_pd, vPx, opCode(and_pd, mDm, vVec));

							countF_line += mask;

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
							for (int k=0; k<step; k++) {
								mask += msk[k] & 1;
								// if (msk[k]  & 1) strdaa[iNx+k*YC*Lx]  = 1; // strinctly not needed
							}
						}

						/* Backward mask */
						while	(maskB != 0) {
							mBMx = opCode(sub_pd, mBMx, opCode(and_pd, mBDp, cVec));
							vBMx = opCode(sub_pd, vBMx, opCode(and_pd, mBDp, vVec));
							mBMx = opCode(add_pd, mBMx, opCode(and_pd, mBDm, cVec));
							vBMx = opCode(add_pd, vBMx, opCode(and_pd, mBDm, vVec));

							countB_line += maskB;

							mBDf = opCode(sub_pd, mBMx, melB);
	#ifdef	__AVX__
							mBDp = opCode(cmp_pd, mBDf, pVec, _CMP_GE_OQ);
							mBDm = opCode(cmp_pd, mBDf, mVec, _CMP_LT_OQ);
	#else
							mBDp = opCode(cmpge_pd, mBDf, pVec);
							mBDm = opCode(cmplt_pd, mBDf, mVec);
#endif
							mBDc = opCode(or_pd, mBDp, mBDm);
							opCode(store_pd, static_cast<double*>(static_cast<void*>(mskB)), mBDc);

							maskB = 0;
							for (int k=0; k<step; k++) {
								maskB += mskB[k] & 1;
								// if (mskB[k] & 1) strdaa[iNxB+k*YC*Lx] = 2; // strictly not needed
							}
						}
#endif	// end AVX and SSE4.1

						opCode(store_pd, &m2[idx2], mPx);
						opCode(store_pd, &m2[idx2+XC], vPx);
						/* Backward mask stored in 2nd slice m2*/
						opCode(store_pd, &m2[idx2B], mBMx);
						// opCode(store_pd, &m2[idx2+3*Sf], vBMx);

					} // end ForBack x-loop

					countB += countB_line;
					countF += countF_line;
					/* compare forwards with backwards and write only if they coincide
					do nothing if nothing was changed in the line */
					if (cha){
						LogMsg(VERB_PARANOID,"[mT] z,y = %lu, %lu (+%lu*n) pre-mends F %lu B %lu",DLz+zSl,yLn,YC,countF_line,countB_line);
						for(size_t xPt = 0; xPt < XC-step; xPt += step) {

							/* checks if jumps where taken, else it breaks to avoid loads */
							bool compute = false;
							iNx   = (xPt/step + 1 + (yLn)*Lx + Vo);  // will miss + vectorindex*YC*Lx
							for (int k=0; k<step; k++) {
								if (strdaa[iNx+k*YC*Lx] != STRING_NOTHING)
									compute = true;
							}
							if (!compute)
								continue;

							idx   = Vo + yLn*XC + xPt;
							idxPx = idx + step;
							idx2  = thread0 + xPt + step;

							mel  = opCode(load_pd, &m[idxPx]);      // arg(phi)
							mPx  = opCode(load_pd, &m2[idx2]);      // corrected F
							mBMx = opCode(load_pd, &m2[idx2+2*XC]); // corrected B

						/* calculate mask
						1 if F = B (will mend)
						0 if F!= B (will not mend) */
						long mask = 0;
						long msk[step];
#ifdef	__AVX512F__
						auto pMask = opCode(cmp_pd_mask, mPx, mBMx, _CMP_EQ_OQ);
						// auto mMask = opCode(cmp_pd_mask, mPx, mBMx, _CMP_NEQ_OQ);
						// auto masks  = pMask | mMask;
						for (int k=0,i=1; k<step; k++,i<<=1)
							msk[k] = (pMask & i) >> k;
#else
	#ifdef	__AVX__
						melB = opCode(cmp_pd, mPx, mBMx, _CMP_EQ_OQ); // are cF and cB equal?
						vBMx = opCode(cmp_pd, mPx, mBMx, _CMP_NEQ_OQ); //are cF and cB different?
	#else
						melB = opCode(cmpeq_ps, mPx, mBMx);
						vBMx = opCode(cmpneq_ps, mPx, mBMx);
	#endif
						opCode(store_pd, static_cast<double*>(static_cast<void*>(msk)), melB);
#endif
						/* calculate mends */
						for (int k=0; k<step; k++) {
							if ((msk[k] & 1) && strdaa[iNx+k*YC*Lx] > 0 )
								mask++;
						}
						count += mask;

						/* store if necessary */
						if (mask >0){
							// double mm[step], mvB[step];
							double vv[step], mvF[step];
							// opCode(store_pd, static_cast<double*>(static_cast<void*>(mm)), mel);
							opCode(store_pd, static_cast<double*>(static_cast<void*>(mvF)), mPx);
							// opCode(store_pd, static_cast<double*>(static_cast<void*>(mvB)), mBMx);
							mPx  = opCode(load_pd, &m2[idx2+XC]);
							opCode(store_pd, static_cast<double*>(static_cast<void*>(vv)), mPx);
							for (int k=0; k<step; k++) {
								if ((msk[k] & 1) && (strdaa[iNx+k*YC*Lx] > 0) ) {
									strdaa[iNx+k*YC*Lx] |= STRING_WALL;
									// double co = R;
									// LogMsg(VERB_PARANOID,"[mT] z,y,x, k = %lu, %lu, %lu, %d; R = %f",zSl, yLn+k*YC, xPt/step,R, k);
									// LogMsg(VERB_PARANOID,"[mT] m %f (%f)-> m_mendF/B %f/%f", m[idxPx+k]/co, mm[k]/co, mvF[k]/co, mvB[k]/co);
									// size_t com = xPt/step < Lx - 2 ? idxPx+step+k : idxPx + step - XC +k;
									// LogMsg(VERB_PARANOID,"[mT] m[-1] %f m[+1] %f", m[idxPx-step+k]/co, m[com]/co); //ojo con +1
									// if (zSl == 128 && yLn == 0) {
									// 	printf("[mT] x %lu k %d mask %d msk[k] %d m %f (%f)-> m_mendF/B %f/%f ... \n", xPt/step, k, mask, msk[k], m[idxPx+k]/co, mm[k]/co, mvF[k]/co, mvB[k]/co);
									// 	size_t com = xPt/step < XC - 2 ? idxPx+step+k : idxPx + step - XC +k;
									// 	printf("m[-1] %f m[+1] %f \n", m[idxPx-step+k]/co, m[com]/co); //ojo con +1
									// }

									/* do this to store or the vector alternative below */
									static_cast<double*>(m_)[idxPx+k] = mvF[k];
									static_cast<double*>(v_)[idxPx+k] = vv[k];
								}
							}
						// /* vector alternative (copy of float version... needs to be adapted)*/
						// /* apply masks and save */
						// 	mel = opCode(add_pd,opCode(and_pd, mel, mBMx),
						// 											opCode(and_pd, mPx, melB));
						// 	opCode(store_pd, &m[idxPx], mel);
						//
						// 	mel  = opCode(load_pd, &v[idxPx]);      // v
						// 	mPx  = opCode(load_pd, &m2[idx2+Sf]);   // corrected F
						//
						// 	mel = opCode(add_pd,opCode(and_pd, mel, mBMx),
						// 											opCode(and_pd, mPx, melB));
						// 	opCode(store_pd, &v[idxPx], mel);
						}
						} // end correction x-loop
				}
				} //end y-loop
			} //end z-loop
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
		float * __restrict__ m  = (float * __restrict__) __builtin_assume_aligned (m_, Align);
		float * __restrict__ v  = (float * __restrict__) __builtin_assume_aligned (v_, Align);
		float * __restrict__ m2 = (float * __restrict__) __builtin_assume_aligned (m2_, Align);

		const float zP = M_PI*R;

		const size_t XC = (Lx<<shf);
		const size_t YC = (Lx>>shf);

		const _MData_ pVec  = opCode(set1_ps, +zP);
		const _MData_ mVec  = opCode(set1_ps, -zP);
		const _MData_ vVec  = opCode(set1_ps, 2.f*M_PI);
		const _MData_ cVec  = opCode(set1_ps, zP*2.f);

		#pragma omp parallel default(shared) reduction(+:countF,countB,count)
		{
			size_t	idx, idxPx, Vo, idx2, idx2B, iNx, iNxB;
			_MData_ mel, mDf, mDc, mDp, mDm, mPx, vPx;
			size_t	idxB, idxBMx;
			_MData_ melB, mBMx, mBDf, mBDp, mBDm, mBDc, vBMx;

			/* position of m2 to use */
			int nThread = omp_get_thread_num();
			size_t thread0 = nThread*4*XC;

			/*	Collapse loops so OpenMP handles the plane YZ	*/
			#pragma omp for collapse(2) schedule(static)
			for (size_t zSl = 0; zSl < Lz; zSl++) {
				for (size_t yLn = 0; yLn < YC; yLn++) {
					Vo = zSl*Sf;

					/*	Bulk loop in X	*/
					bool cha = false;
					size_t countF_line = 0;
					size_t countB_line = 0;
					/* copy m[0] into m2[0] and m2[0+2XC]
					because we will use m2 as references to build F anf B
					mended lines */
					mel = opCode(load_ps, &m[Vo+yLn*XC]);
					opCode(store_ps, &m2[thread0], mel);
					opCode(store_ps, &m2[thread0+2*XC], mel);

					for(size_t xPt = 0; xPt < XC-step; xPt += step) {
// if (zSl == 128 && yLn == 0 && 0)
// 	printf("z,y,x = %lu, %lu-%lu, %lu; R = %f ",zSl, yLn, yLn+(step-1)*YC, xPt/step,R);
						/* Forward (x,y,z) vs (x+step,y,z) */
						idx   = thread0 + xPt;                 // in m2 I do not need z,y info, just thread0
						idx2  = idx + step;                    // in m2
						idxPx = Vo + yLn*XC + xPt + step;      // in m I need the full index
						iNx   = (xPt/step + 1 + (yLn)*Lx + Vo); // will miss + vectorindex*YC*Lx
						mel = opCode(load_ps, &m2[idx]);        // note we read from m2
						mPx = opCode(load_ps, &m[idxPx]);
// TODO load only if neccesary?
						vPx = opCode(load_ps, &v[idxPx]);

						/* Backward (L-x,y,z) vs (L-x-step,y,z) */
						idx2B  = thread0 + 3*XC - xPt - step;
						idxB   = idx2B + step;
						if (xPt == 0)
								idxB -= XC; // wrap around the boundary
						idxBMx = Vo + (yLn+1)*XC - xPt - step;
						iNxB   = ((yLn+1)*Lx + Vo -xPt/step - 1);  // will miss + vectorindex*YC*Lx
						melB = opCode(load_ps, &m2[idxB]); // note we read the ref from the copy
						mBMx = opCode(load_ps, &m[idxBMx]);
						vBMx = opCode(load_ps, &v[idxBMx]);

// if (zSl == 128 && yLn == 0 && 0)
// 	printf(" load done! %lu %lu %lu %lu \n", idx, idxPx, idxB, idxBMx);

						/*	X-Direction	*/

						mDf  = opCode(sub_ps, mPx, mel);
						mBDf = opCode(sub_ps, mBMx, melB);
#ifdef	__AVX512F__
						/* Forward */
						{
						auto pMask = opCode(cmp_ps_mask, mDf, pVec, _CMP_GE_OQ);
						auto mMask = opCode(cmp_ps_mask, mDf, mVec, _CMP_LT_OQ);
						auto mask  = opCode(kor, pMask, mMask);

						int  nChg = 0;
						for (int k=0,i=1; k<step; k++,i<<=1) {
							nChg += (mask & i) >> k;
							if ( (mask & i) >> k ) strdaa[iNx+k*YC*Lx]  = 1;
						}
						if (nChg > 0) cha = true;
//if (cha){
//for (int k=0,i=1; k<step; k++,i<<=1)
//LogMsg(VERB_PARANOID,"zyx %lu %lu %lu k %d i %d mask %d nChg %d ",DLz+zSl,yLn+k*YC,xPt/step+1,k,i,mask, nChg);
//}


						while	(nChg != 0) {
							mPx = opCode(mask_sub_ps, mPx, pMask, mPx, cVec);
							vPx = opCode(mask_sub_ps, vPx, pMask, vPx, vVec);
							mPx = opCode(mask_add_ps, mPx, mMask, mPx, cVec);
							vPx = opCode(mask_add_ps, vPx, mMask, vPx, vVec);

							countF_line += nChg;

							mDf  = opCode(sub_ps, mPx, mel);

							pMask = opCode(cmp_ps_mask, mDf, pVec, _CMP_GE_OQ);
							mMask = opCode(cmp_ps_mask, mDf, mVec, _CMP_LT_OQ);
							mask  = opCode(kor, pMask, mMask);

							nChg = 0;
							for (int k=0,i=1; k<step; k++,i<<=1)
								nChg += (mask & i) >> k;
						}

						/* Backward */
						pMask = opCode(cmp_ps_mask, mBDf, pVec, _CMP_GE_OQ);
						mMask = opCode(cmp_ps_mask, mBDf, mVec, _CMP_LT_OQ);
						mask  = opCode(kor, pMask, mMask);

						nChg = 0;
						for (int k=0,i=1; k<step; k++,i<<=1) {
							nChg += (mask & i) >> k;
							if ( (mask & i) >> k ) strdaa[iNxB+k*YC*Lx] = 2;
						}
						if (nChg > 0) cha = true;

						while	(nChg != 0) {
							mBMx = opCode(mask_sub_ps, mBMx, pMask, mBMx, cVec);
							vBMx = opCode(mask_sub_ps, vBMx, pMask, vBMx, vVec);
							mBMx = opCode(mask_add_ps, mBMx, mMask, mBMx, cVec);
							vBMx = opCode(mask_add_ps, vBMx, mMask, vBMx, vVec);

							countB_line += nChg;

							mBDf = opCode(sub_ps, mBMx, melB);

							pMask = opCode(cmp_ps_mask, mBDf, pVec, _CMP_GE_OQ);
							mMask = opCode(cmp_ps_mask, mBDf, mVec, _CMP_LT_OQ);
							mask  = opCode(kor, pMask, mMask);

							nChg = 0;
							for (int k=0,i=1; k<step; k++,i<<=1)
								nChg += (mask & i) >> k;
						}
						}
#else
						int mask = 0;
						int msk[step];
						int maskB = 0;
						int mskB[step];
	#ifdef	__AVX__
						mDp = opCode(cmp_ps, mDf, pVec, _CMP_GE_OQ);
						mDm = opCode(cmp_ps, mDf, mVec, _CMP_LT_OQ);
						mBDp = opCode(cmp_ps, mBDf, pVec, _CMP_GE_OQ);
						mBDm = opCode(cmp_ps, mBDf, mVec, _CMP_LT_OQ);
	#else
						mDp = opCode(cmpge_ps, mDf, pVec);
						mDm = opCode(cmplt_ps, mDf, mVec);
						mBDp = opCode(cmpge_ps, mBDf, pVec);
						mBDm = opCode(cmplt_ps, mBDf, mVec);

	#endif
						mDc = opCode(or_ps, mDp, mDm);
						mBDc = opCode(or_ps, mBDp, mBDm);
						opCode(store_ps, static_cast<float*>(static_cast<void*>(msk)), mDc);
						opCode(store_ps, static_cast<float*>(static_cast<void*>(mskB)), mBDc);

						for (int k=0; k<step; k++) {
							mask += msk[k] & 1;
							maskB += mskB[k] & 1;
							if (msk[k]  & 1) strdaa[iNx+k*YC*Lx]  = 1;
							if (mskB[k] & 1) strdaa[iNxB+k*YC*Lx] = 2;
						}
						if ((mask > 0) || (maskB > 0)) cha = true;

// if (zSl == 128 && yLn == 0 && 0)
// {
// 	float ma[step], me[step], mi[step], mo[step], mu[step], my[step];
// 	opCode(store_ps, static_cast<float*>(static_cast<void*>(ma)), mel);
// 	opCode(store_ps, static_cast<float*>(static_cast<void*>(me)), mPx);
// 	opCode(store_ps, static_cast<float*>(static_cast<void*>(mi)), melB);
// 	opCode(store_ps, static_cast<float*>(static_cast<void*>(mo)), mBMx);
// 	opCode(store_ps, static_cast<float*>(static_cast<void*>(mu)), mDf);
// 	opCode(store_ps, static_cast<float*>(static_cast<void*>(my)), mBDf);
// 	for (int k=0; k<step; k++) {
// 		printf("%d) %.5f %.5f %.5f %3d",k, ma[k]/R, me[k]/R, mu[k]/R, msk[k]);
// 		printf("| %.5f %.5f %.5f %3d -> cha %d\n ",mi[k]/R, mo[k]/R, my[k]/R, mskB[k], cha);
// }}

						/* Forward mask */
						while	(mask != 0) {
							mPx = opCode(sub_ps, mPx, opCode(and_ps, mDp, cVec));
							vPx = opCode(sub_ps, vPx, opCode(and_ps, mDp, vVec));
							mPx = opCode(add_ps, mPx, opCode(and_ps, mDm, cVec));
							vPx = opCode(add_ps, vPx, opCode(and_ps, mDm, vVec));

							countF_line += mask;

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
							for (int k=0; k<step; k++){
								mask += msk[k] & 1;
								// if (msk[k]  & 1) strdaa[iNx+k*YC*Lx]  = 1; // strinctly not needed
							}
							if (mask > 0) cha = true;
						}

						/* Backward mask */
						while	(maskB != 0) {
							mBMx = opCode(sub_ps, mBMx, opCode(and_ps, mBDp, cVec));
							vBMx = opCode(sub_ps, vBMx, opCode(and_ps, mBDp, vVec));
							mBMx = opCode(add_ps, mBMx, opCode(and_ps, mBDm, cVec));
							vBMx = opCode(add_ps, vBMx, opCode(and_ps, mBDm, vVec));

							countB_line += maskB;

							mBDf = opCode(sub_ps, mBMx, melB);
	#ifdef	__AVX__
							mBDp = opCode(cmp_ps, mBDf, pVec, _CMP_GE_OQ);
							mBDm = opCode(cmp_ps, mBDf, mVec, _CMP_LT_OQ);
	#else
							mBDp = opCode(cmpge_ps, mBDf, pVec);
							mBDm = opCode(cmplt_ps, mBDf, mVec);
	#endif
							mBDc = opCode(or_ps, mBDp, mBDm);
							opCode(store_ps, static_cast<float*>(static_cast<void*>(mskB)), mBDc);

							maskB = 0;
							for (int k=0; k<step; k++){
								maskB += mskB[k] & 1;
								// if (mskB[k] & 1) strdaa[iNxB+k*YC*Lx] = 2; // strictly not needed
							}
							if (maskB > 0) cha = true;
						}
#endif	// end AVX and SSE4.1
						/* TODO if no jumps break to save stores? */

						/* Forward mask stored in 1st slice m2
						can be done in ghosts of v if lowmem
						because it has at least two slices of complex
						Note that the x = 0 slice will not be written because is never mended */

						opCode(store_ps, &m2[idx2],    mPx);
						opCode(store_ps, &m2[idx2+XC], vPx);
						/* Backward mask stored in 2nd slice m2*/
						opCode(store_ps, &m2[idx2B],   mBMx);    //
						// opCode(store_ps, &m2[idx2B+3*XC], vBMx); // not used

					} // end ForBack x-loop

					countB += countB_line;
					countF += countF_line;
					/* compare forwards with backwards and write only if they coincide
					do nothing if nothing was changed in the line */
					if (cha) {
						LogMsg(VERB_PARANOID,"[mT] z,y = %lu, %lu (+%lu*n) pre-mends F %lu B %lu",DLz+zSl,yLn,YC,countF_line,countB_line);
//if (zSl == 128 && yLn == 0 && 0) {
	// printf("[mT] z,y = %lu, %lu pre-mends F %lu B %lu\n",zSl,yLn,countF_line,countB_line);
	// size_t com = xPt/step < XC - 2 ? idxPx+step+k : idxPx + step - XC +k;
	// printf("m[-1] %f m[+1] %f \n", m[idxPx-step+k]/co, m[com]/co); //ojo con +1
// }

						for(size_t xPt = 0; xPt < XC-step; xPt += step) {

							/* checks if jumps where taken, else it breaks to avoid loads */
							bool compute = false;
							iNx   = (xPt/step + 1 + (yLn)*Lx + Vo);  // will miss + vectorindex*YC*Lx
							for (int k=0; k<step; k++) {
								if (strdaa[iNx+k*YC*Lx] != STRING_NOTHING)
									compute = true;
							}
							if (!compute)
								continue;

							idx   = Vo + yLn*XC + xPt;
							idxPx = Vo + yLn*XC + xPt + step;
							idx2  = thread0 + xPt + step;

							mel  = opCode(load_ps, &m[idxPx]);      // arg(phi)
							mPx  = opCode(load_ps, &m2[idx2]);      // corrected F
							mBMx = opCode(load_ps, &m2[idx2+2*XC]); // corrected B

							/* calculate mask
							1 if F = B (will mend)
							0 if F!= B (will not mend) */
							int mask = 0;
							int msk[step];
	#ifdef	__AVX512F__
							auto pMask = opCode(cmp_ps_mask, mPx, mBMx, _CMP_EQ_OQ);
							//auto mMask  = opCode(cmp_ps_mask, mPx, mBMx, _CMP_NEQ_OQ);
							//auto masks  = opCode(kor, pMask, mMask);
							for (int k=0,i=1; k<step; k++,i<<=1)
								msk[k] = (pMask & i) >> k;
	#else
		#ifdef	__AVX__
							melB = opCode(cmp_ps, mPx, mBMx, _CMP_EQ_OQ); // are cF and cB equal?
							vBMx = opCode(cmp_ps, mPx, mBMx, _CMP_NEQ_OQ); //are cF and cB different?
		#else
							melB = opCode(cmpeq_ps, mPx, mBMx);
							vBMx = opCode(cmpneq_ps, mPx, mBMx);
		#endif
							opCode(store_ps, static_cast<float*>(static_cast<void*>(msk)), melB);
	#endif
							/* calculate mends */
							for (int k=0; k<step; k++) {
								if ((msk[k] & 1) && (strdaa[iNx+k*YC*Lx] > 0) )
									mask++;
							}
							count += mask;

							/* store if necessary */
							if (mask >0){
								// float mm[step], mvB[step];
								float vv[step], mvF[step];
								// opCode(store_ps, static_cast<float*>(static_cast<void*>(mm)), mel);
								opCode(store_ps, static_cast<float*>(static_cast<void*>(mvF)), mPx);
								// opCode(store_ps, static_cast<float*>(static_cast<void*>(mvB)), mBMx);
								mPx  = opCode(load_ps, &m2[idx2+XC]);                                  // we have saved corrected v here
								opCode(store_ps, static_cast<float*>(static_cast<void*>(vv)), mPx);
								for (int k=0; k<step; k++) {
									if ((msk[k] & 1) && (strdaa[iNx+k*YC*Lx] > 0) ) {
										// float co = R;
										// if (zSl == 128 && yLn == 0) {
										// printf("[mT] x %lu y %lu mask %d msk[k] %d m %f (%f)-> m_mendF/B %f/%f ... %d \n", xPt/step, yLn+YC*k, mask, msk[k], m[idxPx+k]/co, mm[k]/co, mvF[k]/co, mvB[k]/co,strdaa[iNx+k*YC*Lx]);
										// size_t com = xPt/step < XC - 2 ? idxPx+step+k : idxPx + step - XC +k;
										// printf("m[-1] %f m[+1] %f \n", m[idxPx-step+k]/co, m[com]/co); //ojo con +1
										// }
										strdaa[iNx+k*YC*Lx] |= STRING_WALL;

										/* do this to store or the vector alternative below */
										static_cast<float*>(m_)[idxPx+k] = mvF[k];
										static_cast<float*>(v_)[idxPx+k] = vv[k];
									}
										/* vector alternative (problems at y=0?)*/
										/* apply masks and save */
											// mel = opCode(add_ps,opCode(and_ps, mel, vBMx),
											// 										opCode(and_ps, mPx, melB));
											// opCode(store_ps, &m[idxPx], mel);
											//
											// mel  = opCode(load_ps, &v[idxPx]);      // v
											// // mPx  = opCode(load_ps, &m2[idx2+XC]);   // corrected F
											//
											// mel = opCode(add_ps,opCode(and_ps, mel, vBMx),
											// 										opCode(and_ps, mPx, melB));
											// opCode(store_ps, &v[idxPx], mel);
								}

							}
						} // end correction x-loop
					} // end conditional correction x-loop

// if (zSl == 128 && yLn == 0 && 0)
// 	printf(" 128-0 done\n\n");

//	printf(" thread %d did zS %lu/%lu yL %lu/%lu\n",nThread, zSl, Lz, yLn, YC);
				} // end y-loop
			} // end z-loop
		} // end parallel region
#undef  _MData_
#undef  step
#undef  shf
	}

	return	count;
}

/*	Connects all the points in a Z line with the whole ZY plane	*/
/*	Parallelizes on Z						*/
/*  TODO iterate until a jumpless plane is found */


template<typename Float, const int step>
inline  size_t	mendThetaSlice(Float * __restrict__ m, Float * __restrict__ v, Float * __restrict__ m2, const double R, const size_t Lx, const size_t Lz, const size_t Sf, size_t NN)
{
	const double zP = M_PI*R;
	size_t countF = 0;
	size_t countB = 0;
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
	#pragma omp parallel for reduction(+:count,countF,countB) schedule(static)
	for (size_t zSl = NN; zSl < Lz+NN; zSl++) {

		size_t idx, idxPy, idxB, idxPyB, idx2, idx2P, idx2B, idx2PB;
		Float mDf, mel, mPy, vPy;
		Float mDfB, melB, mPyB, vPyB;
		size_t Vo = zSl*Sf;

		/* Each thread writes its own buffer in m2 */
		int nThread = omp_get_thread_num();
		size_t thread0 = nThread*Sf;
		m2[thread0]        = m[Vo];
		m2[thread0+2*step] = m[Vo];

		/*	Vectorization goes serial	*/
		for (int vIdx = 0; vIdx < step; vIdx++) {
			int vIdxB = step - 1 - vIdx;	// for the backwards line mend
			/*	Bulk Y, X=0 always	*/
			for (size_t yPt = 0; yPt < YC; yPt++) {
				size_t yPtB = YC - 1 - yPt; // for the backwards line mend

				idx   = Vo + yPt*XC + vIdx;
				idxPy = idx + XC;
				idx2  = yPt*XC + vIdx;
				idx2P = idx2 + XC;
				/* last yPt needs shifts */
			 	if (yPt == YC -1) {
					idx    = Vo;
					idxPy  = idx + ((vIdx+1) % step);
					idx   += Sf - XC + vIdx;
					idx2   = idx - Vo;
					idx2P  = idxPy - Vo;
				}

				idxB   = Vo + yPtB*XC + vIdxB;
				idxPyB = idxB + XC;
				idx2B  = yPtB*XC + vIdxB;
				idx2PB = idx2B + XC;
				/* last yPt needs shifts */
			 	if (yPtB == YC -1) {
					idxB    = Vo;
					idxPyB  = idxB + ((vIdxB+1) % step);
					idxB   += Sf - XC + vIdxB;
					idx2B   = idxB - Vo;
					idx2PB  = idxPyB - Vo;
				}


				// mel = m[idx];
				mel = m2[thread0+idx2];        // reference
				mPy = m[idxPy];        // value to compare
				vPy = v[idxPy-Sf*NN];  // value to shift together with value

				mDf = mPy - mel;

				while (abs(mDf) > zP)
				{
					if (mDf > zP) {
						mPy -= 2.*zP;
						vPy -= 2.*M_PI;
						countF++;
					} else if (mDf < zP) {
						mPy += 2.*zP;
						vPy += 2.*M_PI;
						countF++;
					}

					mDf = mPy - mel;
				}

				melB = m[idxB];        // value to compare
				mPyB = m2[thread0+2*step+idx2PB];     // reference
				vPyB = v[idxB-Sf*NN];  // value to shift together with value

				mDfB = melB - mPyB ;

				while (abs(mDfB) > zP)
				{
					if (mDfB > zP) {
						melB -= 2.*zP;
						vPyB -= 2.*M_PI;
						countB++;
					} else if (mDfB < zP) {
						melB += 2.*zP;
						vPyB += 2.*M_PI;
						countB++;
					}

					mDfB = melB - mPyB;
				}

				/* write two m2 */
				/* We shall never write z=0 */
				if (idx2P != 0) {
					m2[thread0+idx2P]      = mPy;
					m2[thread0+idx2P+step] = vPy;      // velocity shifted by step
				}
				if (idx2B != 0) {
					m2[thread0+idx2B+2*step] = melB;  // backwards shifted by 2step
					m2[thread0+idx2B+3*step] = vPyB;  // velocity shifted by 3step
				}

				/* old write */
				// if (idxPy != Vo) {
				// 	m[idxPy] = mPy;
				// 	v[idxPy-Sf*NN] = vPy;
				// }
			} // end yL loop
		} // end vector index loop vIdx

		/* If both corrections coincide write */
		for (int vIdx = 0; vIdx < step; vIdx++) {
			for (size_t yPt = 0; yPt < YC; yPt++) {

				idx   = Vo + yPt*XC + vIdx;
				idxPy = idx + XC;
				idx2  = yPt*XC + vIdx;
				idx2P = idx2 + XC;
				if (yPt == YC -1) {
					idx    = Vo;
					idxPy  = idx + ((vIdx+1) % step);
					idx   += Sf - XC + vIdx;
					idx2   = idx - Vo;
					idx2P  = idxPy - Vo;
				}
				/* writes only if z!=0 (z=0 excluded by idx2P) */
				if (m2[thread0+idx2P] == m2[thread0+idx2P+2*step])
				if (idx2P != 0)
				{
					/* count the mends */
					if (m[idxPy] != m2[thread0+idx2P])
						count++;

					m[idxPy]       = m2[thread0+idx2P];
					v[idxPy-Sf*NN] = m2[thread0+idx2P+step];
				}
			}
		}
	} //end z-loop

	size_t totF=0, totB=0, tot=0;
	MPI_Allreduce(&countF, &totF, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&countB, &totB, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&count, &tot, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
	LogMsg(VERB_HIGH,"[mT] ZY Slice done with mends F/B/Written %lu %lu %lu",totF,totB,tot);
	return	count;
}

/* TODO iterate until a jumpless line is found */

template<typename Float>
inline  size_t	mendThetaLine(Float * __restrict__ m, Float * __restrict__ v, const double R, const size_t Lz, const size_t Sf, size_t NN)
{
	const double zP = M_PI*R;
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
				LogMsg(VERB_PARANOID,"[mT] rank %d send %f for local_z=-1",cBckNeig,m[Sf*(Lz+NN-1)]);
			}

			if (rank == cRank) {
				MPI_Recv(&m[Sf*(NN-1)]  , sizeof(Float), MPI_CHAR, bckNeig, cRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				LogMsg(VERB_PARANOID,"[mT] rank %d received %f for local_z=-1",cRank,m[Sf*(NN-1)]);
			}
		}

		/*	We run only one rank at a time, otherwise we can enter an infinite	*/
		/*	loop where one rank undoes what another did				*/

		if (rank == cRank) {
			for (size_t idx=0,i=0; i<Lz; i++,idx+=Sf) {

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
			} // end for
		} // end rank if
	} // end z-loop

	return	count;
}

size_t	mendThetaXeon (Scalar *field)
{
	constexpr int	dStep = Align/sizeof(double);
	constexpr int	fStep = Align/sizeof(float);
	const double	R     = *(field->RV());
	size_t		tJmp = 0;
	unsigned long tota = 0;

	switch (field->Precision()) {
		case	FIELD_DOUBLE:
		tJmp += mendThetaLine(static_cast<double*>(field->mCpu()), static_cast<double*>(field->vCpu()), R, field->Depth(), field->Surf(), field->getNg());
		MPI_Allreduce(&tJmp, &tota, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
		LogMsg(VERB_HIGH,"[mT] line %lu jumps",tota);
		tJmp += mendThetaSlice<double, dStep>(static_cast<double*>(field->mCpu()), static_cast<double*>(field->vCpu()),static_cast<double*>(field->m2Cpu()),  R, field->Length(), field->Depth(), field->Surf(), field->getNg());
		MPI_Allreduce(&tJmp, &tota, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
		LogMsg(VERB_HIGH,"[mT] Slice %lu jumps",tota);
		if (tJmp > 0) LogMsg(VERB_HIGH,"[mT] Warnng: jumps in line/slice are not safe!");
		tJmp += mendThetaKernelXeon(field->mStart(), field->vCpu(), field->m2Cpu(), field->sData(), R, field->Length(), field->Depth(), field->Surf(), field->Precision());
		break;

		case	FIELD_SINGLE:
		tJmp += mendThetaLine(static_cast<float *>(field->mCpu()), static_cast<float *>(field->vCpu()), R, field->Depth(), field->Surf(), field->getNg());
		MPI_Allreduce(&tJmp, &tota, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
		LogMsg(VERB_HIGH,"[mT] line %lu jumps",tota);
		tJmp += mendThetaSlice<float, fStep>(static_cast<float *>(field->mCpu()), static_cast<float *>(field->vCpu()), static_cast<float *>(field->m2Cpu()), R, field->Length(), field->Depth(), field->Surf(), field->getNg());
		MPI_Allreduce(&tJmp, &tota, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
		LogMsg(VERB_HIGH,"[mT] Slice %lu jumps",tota);
		if (tJmp > 0) LogMsg(VERB_HIGH,"[mT] Warnng: jumps in line/slice are not safe!");
		tJmp += mendThetaKernelXeon(field->mStart(), field->vCpu(), field->m2Cpu(), field->sData(), R, field->Length(), field->Depth(), field->Surf(), field->Precision());
		break;

		default:
		break;
	}

	field->setSD(SD_MENDMAP);
	LogMsg(VERB_HIGH,"[mT] mend map created");

	MPI_Allreduce(&tJmp, &tota, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
	LogMsg(VERB_HIGH,"[mT] Volume %lu jumps",tota);


	LogMsg(VERB_NORMAL,"mendTheta done mends = %lu\n",tota);
	LogOut("mendTheta done mends = %lu\n",tota);
	return	tota;
}
