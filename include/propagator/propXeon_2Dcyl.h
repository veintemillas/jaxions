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
	2Dcyl version vector along X (z cylindrical coordinate), MPI along Z (radial)
*/
template<const VqcdType VQcd, bool UpdateM = true>
inline	void	propagateKernelXeon(const void * __restrict__ m_, void * __restrict__ v_, void * __restrict__ m2_, PropParms ppar, const double dz, const double c, const double d,
				    const size_t Vo, const size_t Vf, FieldPrecision precision, const unsigned int bSizeX, const unsigned int bSizeY, const unsigned int bSizeZ)
{

	const size_t NN    = ppar.Ng;
	const size_t Nx    = ppar.Lx;
	const size_t Nz    = ppar.Lz;
	const size_t Tz    = ppar.Tz;
	const size_t Sf    = Nx;
	const size_t NSf   = Sf*NN;
	const double *PC   = ppar.PC;		// laplacian coeffs
	const double *PCp   = ppar.PCp; // derivative coeffs

	const double R     = ppar.R;
	const double ood2  = ppar.ood2a;
	const double mA2   = ppar.massA2;
	const double gamma = ppar.gamma;
	const double LL    = ppar.lambda;
	const double Rpp   = ppar.Rpp;
	const double Rp    = ppar.Rp ;
	const double deti  = ppar.dectime ;
	const double RPQ   = ppar.RPQ ;

LogMsg(VERB_HIGH,"[pX2D] z0 %lu zF %lu bSizeX %d bSizeY %d bSizeZ %d [NN %d]",Vo/Sf, Vf/Sf, bSizeX, bSizeY, bSizeZ, NN);LogFlush();

	/* Linear term in the EOM contains T,Rpp,etc... */
	const double A     = Rpp - LL*RPQ*RPQ;

	if (Vo>Vf)
		return ;

//sponge
	const uint  nAbsZ   = 16;     // sponge width fast axis
	const uint  nAbsR   = 16;     // sponge width radial
	const float sigAbsZ = 0.5f;  // strength
	const float sigAbsR = 0.5f;

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

		const size_t Nc    = step;      // 4 complex for AVX512, 2 for AVX, 1 for SSE2
		const size_t Sfold = Nx/step;   // physical spacing between packed x values

		const double * __restrict__ m	= (const double * __restrict__) __builtin_assume_aligned(m_, Align);
		double * __restrict__ v		= (double * __restrict__) __builtin_assume_aligned(v_, Align);
		double * __restrict__ m2	= (double * __restrict__) __builtin_assume_aligned(m2_, Align);

		const double dzc = dz*c;
		const double dzd = dz*d;
		const double Rf  = R;
		const double R2  = Rf*Rf;
		const double zQ  = mA2*R2*Rf;

		double gasa;
		switch (VQcd & V_QCD) {
			case V_QCD2:
				gasa = (mA2*R2)/2.;
			break;

			default:
			case V_QCDC:
				gasa = (mA2*R2*R2);
			break;
		}

		const double zN = gasa;

		const double R4    = R2*R2;
		const double LaLa  = LL*2./R4;
		double GGGG        = gamma/Rf;
		if (deti > 0)
			GGGG *= R2/deti;
		const double mola  = GGGG*dzc/2.;
		const double damp1 = 1./(1.+mola);
		const double damp2 = (1.-mola)*damp1;
		const double epsi  = mola/(1.+mola);

		_MData_ COL[NN], COD[NN];
		for (size_t nv = 0; nv < NN; nv++) {
			COL[nv] = opCode(set1_pd, PC[nv]*ood2);
			COD[nv] = opCode(set1_pd, PCp[nv]*ood2);
		}

	#if	defined(__AVX512F__)
		const size_t XC = (Nx<<2);

		const double __attribute__((aligned(Align))) zQAux[8]  = { zQ, 0., zQ, 0., zQ, 0., zQ, 0. };
		const double __attribute__((aligned(Align))) zNAux[8]  = { zN,-zN, zN,-zN, zN,-zN, zN,-zN };
		const double __attribute__((aligned(Align))) zRAux[8]  = { Rf, 0., Rf, 0., Rf, 0., Rf, 0. };
		const double __attribute__((aligned(Align))) cjgAux[8] = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };

		const long long __attribute__((aligned(Align))) shfRg[8] = {6, 7, 0, 1, 2, 3, 4, 5};
		const long long __attribute__((aligned(Align))) shfLf[8] = {2, 3, 4, 5, 6, 7, 0, 1};

		const _MInt_ vShRg = opCode(load_si512, shfRg);
		const _MInt_ vShLf = opCode(load_si512, shfLf);
	#elif	defined(__AVX__)
		const size_t XC = (Nx<<1);

		const double __attribute__((aligned(Align))) zQAux[4]  = { zQ, 0., zQ, 0. };
		const double __attribute__((aligned(Align))) zNAux[4]  = { zN,-zN, zN,-zN };
		const double __attribute__((aligned(Align))) zRAux[4]  = { Rf, 0., Rf, 0. };
		const double __attribute__((aligned(Align))) cjgAux[4] = { 1.,-1., 1.,-1. };
	#else
		const size_t XC = Nx;

		const double __attribute__((aligned(Align))) zQAux[2]  = { zQ, 0. };
		const double __attribute__((aligned(Align))) zNAux[2]  = { zN,-zN };
		const double __attribute__((aligned(Align))) zRAux[2]  = { Rf, 0. };
		const double __attribute__((aligned(Align))) cjgAux[2] = { 1.,-1. };
	#endif

		const _MData_ zQVec = opCode(load_pd, zQAux);
		const _MData_ zNVec = opCode(load_pd, zNAux);
		const _MData_ zRVec = opCode(load_pd, zRAux);
		const _MData_ cjg   = opCode(load_pd, cjgAux);

		const uint z0 = Vo/(Nx);
		const uint zF = Vf/(Nx);

		const uint zTiles = (zF - z0 + bSizeZ - 1) / bSizeZ;
		const uint xTiles = (Nx  + bSizeX - 1) / bSizeX;

		#pragma omp parallel default(shared)
		{
			_MData_ tmp, mel, mPx, mMx, mPz, mMz, mPy, tmp2, lap;

			#pragma omp for collapse(2) schedule(static)
			for (uint zT = 0; zT < zTiles; zT++) {
				for (uint xT = 0; xT < xTiles; xT++) {

					const uint zBeg = z0 + zT * bSizeZ;
					const uint zEnd = (zBeg + bSizeZ < zF) ? (zBeg + bSizeZ) : zF;

					const uint xBeg = xT * bSizeX;
					const uint xEnd = (xBeg + bSizeX < Nx) ? (xBeg + bSizeX) : Nx;

					for (uint zC = zBeg; zC < zEnd; zC++) {
						for (uint xC = xBeg; xC < xEnd; xC += step) {

							size_t idxMx, idxPx, idxMz, idxPz, idxP0, idxV0;
							size_t idx = zC * Nx + xC;
							size_t zC_global = zC - NN + Nz * commRank();

							idxP0 =  (idx << 1);
							idxV0 =  ((idx - NSf) << 1);

							mel = opCode(load_pd, &m[idxP0]);
							lap = opCode(set1_pd, 0.0);

							const uint j        = xC / step;
							const uint zFastMax = j + (Nc - 1)*Sfold;

							for (size_t nv = 1; nv <= NN; nv++)
							{
								const _MData_ c_lap = COL[nv - 1];
								const _MData_ c_der = COD[nv - 1];

								/* ------------------ X direction (axial Zc) ------------------ */

								if (xC >= nv*step) {
									idxMx = ((idx - nv*step) << 1);
									mMx   = opCode(load_pd, &m[idxMx]);
								} else {
									const size_t idxWrap = ((zC*Nx + (xC + (Sfold - nv)*step)) << 1);

									alignas(Align) double bxWrap[2*step];
									alignas(Align) double bxOut [2*step];

									opCode(store_pd, bxWrap, opCode(load_pd, &m[idxWrap]));

									for (size_t q = 0; q < Nc; q++) {
										const uint xq = j + q*Sfold;

										if (xq < nv) {
											const uint xRef = nv - xq;

											const uint jRef = xRef % Sfold;
											const uint qRef = xRef / Sfold;

											const size_t idxRefScalar = ((zC*Nx + jRef*step + qRef) << 1);

											double re = m[idxRefScalar + 0];
											double im = m[idxRefScalar + 1];

											// BC: phi(-x) = phi*(x)
											bxOut[2*q + 0] =  re;
											bxOut[2*q + 1] = -im;
										} else {
											bxOut[2*q + 0] = bxWrap[2*(q-1) + 0];
											bxOut[2*q + 1] = bxWrap[2*(q-1) + 1];
										}
									}

									mMx = opCode(load_pd, bxOut);
								}

								if (j + nv < Sfold) {
									idxPx = ((idx + nv*step) << 1);
									mPx   = opCode(load_pd, &m[idxPx]);
								} else {
									const uint jWrap = j + nv - Sfold;

									alignas(Align) double bxWrap[2*step];
									alignas(Align) double bxOut [2*step];
									alignas(Align) double bxMel [2*step];

									const size_t idxWrap = ((zC*Nx + jWrap*step) << 1);
									opCode(store_pd, bxWrap, opCode(load_pd, &m[idxWrap]));
									opCode(store_pd, bxMel, mel);

									for (size_t q = 0; q < Nc; q++) {
										const uint xq = j + q*Sfold;
										const uint xp = xq + nv;

										if (xp >= Nx) {
											bxOut[2*q + 0] = bxMel[2*q + 0];
											bxOut[2*q + 1] = bxMel[2*q + 1];
										} else {
											bxOut[2*q + 0] = bxWrap[2*(q+1) + 0];
											bxOut[2*q + 1] = bxWrap[2*(q+1) + 1];
										}
									}

									mPx = opCode(load_pd, bxOut);
								}

								/* ------------------ Z direction (cylindrical rho) ------------------ */

								if (zC_global >= nv) {
									idxMz = ((idx - nv*Sf) << 1);
									mMz   = opCode(load_pd, &m[idxMz]);
								} else {
									idxMz = ((idx + (nv - zC_global)*Sf) << 1);
									mMz   = opCode(load_pd, &m[idxMz]);
								}

								if (zC_global + nv < Tz) {
									idxPz = ((idx + nv*Sf) << 1);
									mPz   = opCode(load_pd, &m[idxPz]);
								} else {
									mPz = mel;
								}

								/* ------------------ Cylindrical XZ operator ------------------ */

								if (zC_global == 0) {
									tmp = opCode(add_pd, mPx, mMx);
									tmp = opCode(add_pd, tmp, opCode(add_pd, mPz, mMz));
									tmp = opCode(add_pd, tmp, opCode(add_pd, mPz, mMz));
									tmp = opCode(add_pd, tmp, opCode(mul_pd, mel, opCode(set1_pd, -6.0)));
									tmp = opCode(mul_pd, tmp, c_lap);
								} else {
									tmp = opCode(add_pd, mPx, mMx);
									tmp = opCode(add_pd, tmp, opCode(add_pd, mPz, mMz));
									tmp = opCode(add_pd, tmp, opCode(mul_pd, mel, opCode(set1_pd, -4.0)));
									tmp = opCode(mul_pd, tmp, c_lap);

									tmp2 = opCode(sub_pd, mPz, mMz);
									tmp2 = opCode(mul_pd, tmp2,
											opCode(mul_pd, c_der, opCode(set1_pd, 1.0/double(zC_global))));

									tmp = opCode(add_pd, tmp, tmp2);
								}

								lap = opCode(add_pd, lap, tmp);
							} // end neighbour loop

							mPy = opCode(mul_pd, mel, mel);

	#if	defined(__AVX__)
							mPx = opCode(add_pd, opCode(permute_pd, mPy, 0b0101), mPy);
	#else
							mPx = opCode(add_pd, opCode(shuffle_pd, mPy, mPy, 0b0001), mPy);
	#endif

							/* mMx = acceleration */
							if (VQcd & V_EVOL_THETA)
								mMx = lap;
							else
								switch (VQcd & V_PQ) {
									case V_PQ1:
										mMx = opCode(sub_pd, lap,
													opCode(mul_pd, mel,
														opCode(sub_pd,
															opCode(mul_pd,
																opCode(sub_pd, mPx, opCode(set1_pd, R2)),
																opCode(set1_pd, LL)),
															opCode(set1_pd, A))));
									break;

									case V_PQ3:
										tmp2 = opCode(sub_pd, mPx, opCode(set1_pd, R2));
										tmp  = opCode(mul_pd, opCode(set1_pd, 2.*R2),
													opCode(sub_pd, opCode(set1_pd, 1.0),
														opCode(div_pd, opCode(set1_pd, Rf),
															opCode(sqrt_pd, mPx))));
										tmp = opCode(kkk_pd, tmp, tmp2, mPx, opCode(set1_pd, R2));

										mMx = opCode(sub_pd, lap,
													opCode(mul_pd, mel,
														opCode(sub_pd,
															opCode(mul_pd, tmp, opCode(set1_pd, LL)),
															opCode(set1_pd, A))));
									break;

									case V_PQ2:
										mMx = opCode(sub_pd, lap,
													opCode(mul_pd, mel,
														opCode(sub_pd,
															opCode(mul_pd,
																opCode(sub_pd,
																	opCode(mul_pd, mPx, mPx),
																	opCode(set1_pd, R4)),
																opCode(mul_pd, mPx, opCode(set1_pd, LaLa))),
															opCode(set1_pd, A))));
									break;
								}

							/* mMx = mMx + VQCD part */
							if (!(VQcd & V_EVOL_RHO))
								switch (VQcd & V_QCD) {
									case V_QCD1:
										mMx = opCode(add_pd, mMx, zQVec);
									break;

									case V_QCDV:
										mMx = opCode(add_pd, mMx,
													opCode(mul_pd, opCode(set1_pd, zQ),
														opCode(sub_pd, zRVec, mel)));
									break;

									case V_QCD2:
										mMx = opCode(add_pd, mMx, opCode(mul_pd, zNVec, mel));
									break;

									case V_QCDC:
										tmp2 = opCode(div_pd,
													opCode(vqcd0_pd, mel),
													opCode(sqrt_pd,
														opCode(mul_pd, mPx,
															opCode(mul_pd, mPx, mPx))));
										mMx = opCode(add_pd, mMx, opCode(mul_pd, zNVec, tmp2));
									break;

									default:
									case V_QCDL:
										tmp2 = opCode(div_pd,
													opCode(vqcd0_pd, mel),
													opCode(sqrt_pd,
														opCode(mul_pd, mPx,
															opCode(mul_pd, mPx, mPx))));
										mMx = opCode(add_pd, mMx, opCode(mul_pd, zNVec, tmp2));
									break;

									case V_QCD0:
									break;
								}

							mPy = opCode(load_pd, &v[idxV0]);

							/* Project accelerations and velocities if needed */
							if (VQcd & V_EVOL_THETA)
							{
	#if	defined(__AVX__)
								lap = opCode(permute_pd, opCode(mul_pd, mel, cjg), 0b0101);
								auto vecmv = opCode(mul_pd, mMx, lap);
								auto vecma = opCode(add_pd, opCode(permute_pd, vecmv, 0b0101), vecmv);
								vecmv = opCode(mul_pd, mPy, lap);
								vecmv = opCode(add_pd, opCode(permute_pd, vecmv, 0b0101), vecmv);
	#else
								lap = opCode(mul_pd, mel, cjg);
								lap = opCode(shuffle_pd, lap, lap, 0b0001);
								auto vecmv = opCode(mul_pd, mMx, lap);
								auto vecma = opCode(add_pd, opCode(shuffle_pd, vecmv, vecmv, 0b0001), vecmv);
								vecmv = opCode(mul_pd, mPy, lap);
								vecmv = opCode(add_pd, opCode(shuffle_pd, vecmv, vecmv, 0b0001), vecmv);
	#endif
								mMx = opCode(add_pd,
											opCode(div_pd, opCode(mul_pd, lap, vecma), mPx),
											opCode(mul_pd, mel, opCode(set1_pd, A)));

								mPy = opCode(add_pd,
											opCode(div_pd, opCode(mul_pd, lap, vecmv), mPx),
											opCode(mul_pd, mel, opCode(set1_pd, Rp)));
							}

							/* update velocities with/without damping */
							const bool inRhoSponge = (zC_global >= Tz - nAbsR);
							const bool inZSponge   = (zFastMax   >= Nx - nAbsZ);

							switch (VQcd & V_DAMP) {

								default:
								case V_NONE:
									if (!(inRhoSponge || inZSponge)) {
	#if defined(__AVX512F__) || defined(__FMA__)
										tmp = opCode(fmadd_pd, mMx, opCode(set1_pd, dzc), mPy);
	#else
										tmp = opCode(add_pd, mPy, opCode(mul_pd, mMx, opCode(set1_pd, dzc)));
	#endif
									} else {
										alignas(Align) double sigAux[2*step];

										double sigRho = 0.0;
										if (inRhoSponge) {
											const double u = double(zC_global - (Tz - nAbsR)) / double(nAbsR);
											sigRho = sigAbsR * u * u;
										}

										for (size_t q = 0; q < Nc; q++) {
											double sig = sigRho;

											if (inZSponge) {
												const uint zFast = j + q*Sfold;
												if (zFast >= Nx - nAbsZ) {
													const double u = double(zFast - (Nx - nAbsZ)) / double(nAbsZ);
													sig += sigAbsZ * u * u;
												}
											}

											sigAux[2*q + 0] = sig;
											sigAux[2*q + 1] = sig;
										}

										const _MData_ sigVec = opCode(load_pd, sigAux);
										const _MData_ one    = opCode(set1_pd, 1.0);
										const _MData_ half   = opCode(set1_pd, 0.5);
										const _MData_ dzv    = opCode(set1_pd, dzc);

										const _MData_ molaS  = opCode(mul_pd, half, opCode(mul_pd, sigVec, dzv));
										const _MData_ damp1S = opCode(div_pd, one, opCode(add_pd, one, molaS));
										const _MData_ damp2S = opCode(mul_pd, opCode(sub_pd, one, molaS), damp1S);

										tmp = opCode(add_pd,
													opCode(mul_pd, mPy, damp2S),
													opCode(mul_pd, opCode(mul_pd, mMx, dzv), damp1S));
									}
								break;

								case V_DAMP_RHO:
								{
									tmp = opCode(mul_pd, mel, mPy);
	#if	defined(__AVX__)
									auto vecmv = opCode(add_pd, opCode(permute_pd, tmp, 0b0101), tmp);
	#else
									auto vecmv = opCode(add_pd, opCode(shuffle_pd, tmp, tmp, 0b0001), tmp);
	#endif

									tmp = opCode(mul_pd, mel, mMx);
	#if	defined(__AVX__)
									auto vecma = opCode(add_pd, opCode(permute_pd, tmp, 0b0101), tmp);
	#else
									auto vecma = opCode(add_pd, opCode(shuffle_pd, tmp, tmp, 0b0001), tmp);
	#endif

	#if	defined(__AVX512F__) || defined(__FMA__)
									tmp = opCode(sub_pd,
										opCode(fmadd_pd, mMx, opCode(set1_pd, dzc), mPy),
										opCode(mul_pd,
											opCode(mul_pd, opCode(set1_pd, epsi), opCode(div_pd, mel, mPx)),
											opCode(fmadd_pd,
												opCode(sub_pd, vecmv, opCode(mul_pd, mPx, opCode(set1_pd, Rp))),
												opCode(set1_pd, 2.0),
												opCode(mul_pd, vecma, opCode(set1_pd, dzc)))));
	#else
									tmp = opCode(sub_pd,
										opCode(add_pd, mPy, opCode(mul_pd, mMx, opCode(set1_pd, dzc))),
										opCode(mul_pd,
											opCode(mul_pd, opCode(set1_pd, epsi), opCode(div_pd, mel, mPx)),
											opCode(add_pd,
												opCode(mul_pd,
													opCode(sub_pd, vecmv, opCode(mul_pd, mPx, opCode(set1_pd, Rp))),
													opCode(set1_pd, 2.0)),
												opCode(mul_pd, vecma, opCode(set1_pd, dzc)))));
	#endif
								}
								break;

								case V_DAMP_ALL:
	#if	defined(__AVX512F__) || defined(__FMA__)
									tmp = opCode(fmadd_pd, mPy, opCode(set1_pd, damp2),
											opCode(mul_pd, mMx, opCode(set1_pd, damp1*dzc)));
	#else
									tmp = opCode(add_pd,
											opCode(mul_pd, mPy, opCode(set1_pd, damp2)),
											opCode(mul_pd, mMx, opCode(set1_pd, damp1*dzc)));
	#endif
								break;
							}

							if (VQcd & V_EVOL_RHO)
							{
								auto vecmv = opCode(mul_pd, mel, tmp);
	#if	defined(__AVX__)
								auto vecma = opCode(add_pd, opCode(permute_pd, vecmv, 0b0101), vecmv);
	#else
								auto vecma = opCode(add_pd, opCode(shuffle_pd, vecmv, vecmv, 0b0001), vecmv);
	#endif
								tmp = opCode(div_pd, opCode(mul_pd, mel, vecma), mPx);
							}


	opCode(store_pd,  &v[idxV0], tmp);

	if constexpr (UpdateM)
							{
							#if	defined(__AVX512F__) || defined(__FMA__)
													mPx = opCode(fmadd_pd, tmp, opCode(set1_pd, dzd), mel);
							#else
													mPx = opCode(add_pd, mel, opCode(mul_pd, tmp, opCode(set1_pd, dzd)));
							#endif

							opCode(stream_pd, &m2[idxP0], mPx);
							}
						}
					}
				}
			}
		}
	#undef	_MData_
	#undef	step
	}
	else if (precision == FIELD_SINGLE) {
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

		const size_t Nc    = step;      // 8 complex for AVX512, 4 for AVX, 2 for SSE
		const size_t Sfold = Nx/step;   // physical spacing between packed x values

		const float * __restrict__ m	= (const float * __restrict__) __builtin_assume_aligned (m_, Align);
		float * __restrict__ v		= (float * __restrict__) __builtin_assume_aligned (v_, Align);
		float * __restrict__ m2		= (float * __restrict__) __builtin_assume_aligned (m2_, Align);

		const float dzc = dz*c;
		const float dzd = dz*d;
		const float Rf  = (float) R;
		const float R2  = Rf*Rf;
		const float zQ = (float) (mA2*R2*Rf);

		float gasa;
		switch	(VQcd & V_QCD) {
				case	V_QCD2:
				gasa = (float) (mA2*R2)/2;
				break;

				default:
				case	V_QCDC:
				gasa = (float) (mA2*R2*R2);
				break;
		}
		//For V_QCD2 & V_QCDC
		const float zN = gasa;

		const float R4 = R2*R2;
		const float LaLa = LL*2.f/R4;
		float GGGG = gamma/Rf;
		if (deti > 0)
			GGGG *= R2/deti;
		const float mola = GGGG*dzc/2.f;
		const float damp1 = 1.f/(1.f+mola);
		const float damp2 = (1.f-mola)*damp1;
		const float epsi = mola/(1.f+mola);

		_MData_ COL[NN], COD[NN];
		for (size_t nv = 0; nv < NN ; nv++){
			COL[nv]  = opCode(set1_ps, PC[nv]*ood2);
			COD[nv]  = opCode(set1_ps, PCp[nv]*ood2);
		}


#if	defined(__AVX512F__)
		const size_t XC = (Nx<<3); // we calculate with vectors of 8 complex values

		const float __attribute__((aligned(Align))) zQAux[16] = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zNAux[16] = { zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN, zN, -zN };
		const float __attribute__((aligned(Align))) zRAux[16] = { Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f };
		// TEMPO
		const float __attribute__((aligned(Align))) cjgAux[16]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 1.,-1. };

		const int   __attribute__((aligned(Align))) shfRg[16] = {14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
		const int   __attribute__((aligned(Align))) shfLf[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1};

		const _MInt_  vShRg  = opCode(load_si512, shfRg);
		const _MInt_  vShLf  = opCode(load_si512, shfLf);
#elif	defined(__AVX__)
		const size_t XC = (Nx<<2); // we calculate with vectors of 8 complex values

		const float __attribute__((aligned(Align))) zQAux[8]  = { zQ, 0.f, zQ, 0.f, zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zNAux[8]  = { zN, -zN, zN, -zN, zN, -zN, zN, -zN };
		const float __attribute__((aligned(Align))) zRAux[8]  = { Rf, 0.f, Rf, 0.f, Rf, 0.f, Rf, 0.f };

		const float __attribute__((aligned(Align))) cjgAux[8]  = { 1.,-1., 1.,-1., 1.,-1., 1.,-1. };
#else
		const size_t XC = (Nx<<1); // we calculate with vectors of 8 complex values

		const float __attribute__((aligned(Align))) zQAux[4]  = { zQ, 0.f, zQ, 0.f };
		const float __attribute__((aligned(Align))) zNAux[4]  = { zN, -zN, zN, -zN };
		const float __attribute__((aligned(Align))) zRAux[4]  = { Rf, 0.f, Rf, 0.f };

		const float __attribute__((aligned(Align))) cjgAux[4]  = { 1.,-1., 1.,-1. };
#endif
		const _MData_ zQVec  = opCode(load_ps, zQAux);
		const _MData_ zNVec  = opCode(load_ps, zNAux);
		const _MData_ zRVec  = opCode(load_ps, zRAux);
		const _MData_ cjg  = opCode(load_ps, cjgAux);

	/* no chuncking tricks
	xC, zC are coordinates of the complex field
	idx is float, not complex<float> so it runs up to 2*Nx*Nz

	we assume the field is folded as phi(x,z)
	rephi(0,0) imphi(0,0) rephi(0+S,0) imphi(0+S,0) rephi(0+2S,0) imphi(0+2S,0) ... rephi(0+(F-1)S,0) imphi(0+(F-1)S,0)
	rephi(1,0) imphi(1,0) rephi(1+S,0) imphi(1+S,0) rephi(1+2S,0) imphi(1+2S,0) ... rephi(1+(F-1)S,0) imphi(1+(F-1)S,0)
	...
	rephi(0,1) imphi(0,1) rephi(0+S,1) imphi(0+S,1) rephi(0+2S,1) imphi(0+2S,1) ... rephi(0+(F-1)S,1) imphi(0+(F-1)S,1)
	F is the size of the vector (mAlign/datasize)
	S is the vector index = (0...,Nx/F)
	*/

	const uint z0 = Vo/(Nx);
	const uint zF = Vf/(Nx);

	const uint zTiles = (zF - z0 + bSizeZ - 1) / bSizeZ;
	const uint xTiles = (Nx  + bSizeX - 1) / bSizeX;
// LogMsg(VERB_PARANOID,"[pX2D] z0 %lu zF %lu bSizeX %d bSizeY %d bSizeZ %d [NN %d]",Vo/Sf, Vf/Sf, bSizeX, bSizeY, bSizeZ, NN);LogFlush();

	#pragma omp parallel default(shared)
	{
	_MData_ tmp, mel, mPx, mMx, mPz, mMz, mPy, tmp2, lap;

	#pragma omp for collapse(2) schedule(static)
	for (uint zT = 0; zT < zTiles; zT++) {
		for (uint xT = 0; xT < xTiles; xT++) {

			const uint zBeg = z0 + zT * bSizeZ;
			const uint zEnd = (zBeg + bSizeZ < zF) ? (zBeg + bSizeZ) : zF;

			const uint xBeg = xT * bSizeX;
			const uint xEnd = (xBeg + bSizeX < Nx) ? xBeg + bSizeX : Nx;

			for (uint zC = zBeg; zC < zEnd; zC++) {                  // tile interior in z
				for (uint xC = xBeg; xC < xEnd; xC += step) {        // tile interior in x, SIMD stride

					size_t idxMx, idxPx, idxMz, idxPz, idxP0, idxV0;
					size_t idx = zC * Nx + xC;
					size_t zC_global = zC - NN + Nz * commRank();   // zC are measured from ghost, read z coordinate is z-NN

			idxP0 =  (idx << 1);        // to access memory with m casted as float we neeed to multiply by 2
			idxV0 =  (idx-NSf) << 1;    // m is ghosted, v is not

			mel = opCode(load_ps, &m[idxP0]);
			lap = opCode(set1_ps, 0.f);

			const uint j = xC / step;   // folded-vector index: 0..Sfold-1
			const uint zFastMax = j + (Nc-1)*Sfold;
			// const bool inZsponge = (zFastMax >= Nx - nAbsZ);
			// const bool inRhosponge = (zC_global >= Tz - nAbsR);

			for (size_t nv=1; nv <= NN; nv++)
			{
				const _MData_ c_lap = COL[nv - 1];
				const _MData_ c_der = COD[nv - 1];

				/* ------------------ X direction (axial Zc) ------------------ */


				if (xC >= nv*step) {
					idxMx = ((idx - nv*step) << 1);
					mMx   = opCode(load_ps, &m[idxMx]);
				} else {

					// wrapped vector j' = Sfold + j - nv
					const size_t idxWrap = ((zC*Nx + (xC + (Sfold - nv)*step)) << 1);

					// left boundary at x=0
					// -conj reflection for the circular loop
					// load reflected candidate (x = nv)
					size_t idxRef = ((zC*Nx + nv*step) << 1);
					_MData_ vRef = opCode(load_ps, &m[idxRef]);

					// load interior candidate (x = S - nv)
					size_t idxInt = ((zC*Nx + (Sfold - nv)*step) << 1);
					_MData_ vInt = opCode(load_ps, &m[idxInt]);

					// spill to temporary arrays
					// alignas(Align) float bxRef[2*step];
					// alignas(Align) float bxInt[2*step];
					// alignas(Align) float bxOut[2*step];
					alignas(Align) float bxWrap[2*step];
					alignas(Align) float bxOut [2*step];

					// opCode(store_ps, bxRef, vRef);
					// opCode(store_ps, bxInt, vInt);
					opCode(store_ps, bxWrap, opCode(load_ps, &m[idxWrap]));

					// build output lane-by-lane
					for (size_t q = 0; q < Nc; q++) {

					    // const uint xq = (xC/step) + q*Sfold;
							const uint xq = j + q*Sfold;
					    if (xq < nv) {
									// reflected ghost point: x = -(nv - xq)  ->  x_ref = nv - xq
					        // reflected: take from bxRef and apply BC φ(-x) = φ*(x)
					        // float re = bxRef[2*q + 0];
					        // float im = bxRef[2*q + 1];
									const uint xRef = nv - xq;

			            const uint jRef = xRef % Sfold;   // folded vector
			            const uint qRef = xRef / Sfold;   // lane inside that vector

			            const size_t idxRefScalar = ((zC*Nx + jRef*step + qRef) << 1);

			            float re = m[idxRefScalar + 0];
			            float im = m[idxRefScalar + 1];

					        bxOut[2*q + 0] = re;
					        bxOut[2*q + 1] = -im;
					    } else {
					        // interior: take from bxInt
					        bxOut[2*q + 0] = bxWrap[2*(q-1) + 0];
					        bxOut[2*q + 1] = bxWrap[2*(q-1) + 1];
					    }
					}

					// reload into SIMD register
					mMx = opCode(load_ps, bxOut);
				}

				if (j + nv < Sfold) {
				    idxPx = ((idx + nv*step) << 1);
				    mPx   = opCode(load_ps, &m[idxPx]);
				} else {
				    const uint jWrap = j + nv - Sfold;

				    alignas(Align) float bxWrap[2*step];
				    alignas(Align) float bxOut [2*step];

				    const size_t idxWrap = ((zC*Nx + jWrap*step) << 1);
				    opCode(store_ps, bxWrap, opCode(load_ps, &m[idxWrap]));

				    alignas(Align) float bxMel[2*step];
				    opCode(store_ps, bxMel, mel);

				    for (size_t q = 0; q < Nc; q++) {
				        const uint xq = j + q*Sfold;
				        const uint xp = xq + nv;

				        if (xp >= Nx) {
				            // outer boundary placeholder
				            bxOut[2*q + 0] = bxMel[2*q + 0];
				            bxOut[2*q + 1] = bxMel[2*q + 1];
				        } else {
				            bxOut[2*q + 0] = bxWrap[2*(q+1) + 0];
				            bxOut[2*q + 1] = bxWrap[2*(q+1) + 1];
				        }
				    }

				    mPx = opCode(load_ps, bxOut);
				}

				/* ------------------ Z direction (cylindrical rho) ------------------ */

				if (zC_global >= nv) {
					idxMz = ((idx - nv*Sf) << 1);
					mMz   = opCode(load_ps, &m[idxMz]);
				} else {
					// regular reflection across rho=0
					// rho<0 mirrored to rho>0
					// mMz = m[idx + (nv - zC_global)*Sf]
					idxMz = ((idx + (nv - zC_global)*Sf) << 1);
					mMz   = opCode(load_ps, &m[idxMz]);
				}

				if (zC_global + nv < Tz) {
					idxPz = ((idx + nv*Sf) << 1);
					mPz   = opCode(load_ps, &m[idxPz]);
				} else {
					// outer radial boundary, ideally absorbing
					// simplest placeholder:
					mPz = mel;
				}

				/* ------------------ Cylindrical XZ operator ------------------ */

				if (zC_global == 0) {
					// axis row: radial piece doubled
					tmp = opCode(add_ps, mPx, mMx);
					tmp = opCode(add_ps, tmp, opCode(add_ps, mPz, mMz));
					tmp = opCode(add_ps, tmp, opCode(add_ps, mPz, mMz));
					tmp = opCode(add_ps, tmp, opCode(mul_ps, mel, opCode(set1_ps, -6.f)));
					tmp = opCode(mul_ps, tmp, c_lap);
				} else {
					// standard cylindrical row
					tmp = opCode(add_ps, mPx, mMx);
					tmp = opCode(add_ps, tmp, opCode(add_ps, mPz, mMz));
					tmp = opCode(add_ps, tmp, opCode(mul_ps, mel, opCode(set1_ps, -4.f)));
					tmp = opCode(mul_ps, tmp, c_lap);

					tmp2 = opCode(sub_ps, mPz, mMz);
					tmp2 = opCode(mul_ps, tmp2, opCode(mul_ps, c_der, opCode(set1_ps, 1.f/float(zC_global))));

					tmp = opCode(add_ps, tmp, tmp2);
				}

				lap = opCode(add_ps, lap, tmp);

			} // end neighbour loop


			mPy = opCode(mul_ps, mel, mel);		// M1^2 M2^2

			// mPx is M1^2+M2^2 M1^2+M2^2
#if	defined(__AVX__)// || defined(__AVX512F__)
			mPx = opCode(add_ps, opCode(permute_ps, mPy, 0b10110001), mPy);
#else
			mPx = opCode(add_ps, opCode(shuffle_ps, mPy, mPy, 0b10110001), mPy);
#endif

			/* mMx = acceleration
						 = lap - Phi *
									( PQ-part
										- R''p/R )
			*/
			if (VQcd & V_EVOL_THETA)
				mMx = lap;
			else
				switch	(VQcd & V_PQ) {
					case V_PQ1:
						mMx = opCode(sub_ps, lap,
										opCode(mul_ps, mel,
											opCode(sub_ps,
												opCode(mul_ps, opCode(sub_ps, mPx, opCode(set1_ps, R2)),
													opCode(set1_ps, LL)),
												opCode(set1_ps, A))));
					break;
					case V_PQ3:
						// a = p^2 - R^2
						tmp2 = opCode(sub_ps, mPx, opCode(set1_ps, R2));
						// b = 2R^2(1-R/p)
						tmp  = opCode(mul_ps, opCode(set1_ps, 2*R2),
										opCode(sub_ps,opCode(set1_ps, 1.f),
											opCode(div_ps,opCode(set1_ps, Rf),
												opCode(sqrt_ps, mPx))));
						// tmp(cuadratic) if p > R and tmp2(quartic) if p < R
						tmp = opCode(kkk_ps,tmp,tmp2,mPx,opCode(set1_ps, R2));

						mMx = opCode(sub_ps, lap,
										opCode(mul_ps, mel,
											opCode(sub_ps,
												opCode(mul_ps, tmp,
													opCode(set1_ps, LL)),
												opCode(set1_ps, A))));
					break;
					case V_PQ2:
						mMx = opCode(sub_ps, lap,
										opCode(mul_ps, mel,
											opCode(sub_ps,
												opCode(mul_ps,
													opCode(sub_ps, opCode(mul_ps, mPx, mPx), opCode(set1_ps, R4)),
														opCode(mul_ps, mPx, opCode(set1_ps, LaLa))),
											opCode(set1_ps, A))));
					break;
				}
			/* mMx = mMx + VQCD part */
			if ( !(VQcd & V_EVOL_RHO) )
				switch	(VQcd & V_QCD) {
					case V_QCD1:
						mMx = opCode(add_ps, mMx, zQVec);
					break;
					case V_QCDV:
						mMx = opCode(add_ps, mMx, opCode(mul_ps, opCode(set1_ps, zQ), opCode(sub_ps, zRVec, mel)));
					break;
					case V_QCD2:
						mMx = opCode(add_ps, mMx, opCode(mul_ps,zNVec,mel));
					break;
					case V_QCDC:
						tmp2 = opCode(div_ps,
										opCode(vqcd0_ps,mel),
											opCode(sqrt_ps, opCode(mul_ps, mPx, opCode(mul_ps, mPx, mPx) ) ) ); //
						mMx = opCode(add_ps, mMx, opCode(mul_ps, zNVec, tmp2));
					break;
					default:
					case V_QCDL:
						tmp2 = opCode(div_ps,
										opCode(vqcd0_ps,mel),
											opCode(sqrt_ps, opCode(mul_ps, mPx, opCode(mul_ps, mPx, mPx) ) ) ); //
						mMx = opCode(add_ps, mMx, opCode(mul_ps, zNVec, tmp2));
					break;
					case V_QCD0:
					break;
				}


			mPy = opCode(load_ps, &v[idxV0]);

			/* Proyect accelerations (mMx) and velocities (mPy) if needed */

			if (VQcd & V_EVOL_THETA)
			{
#if	defined(__AVX__)// || defined(__AVX512F__)
				//0.-(-mi mr)
				lap = opCode(permute_ps, opCode(mul_ps, mel, cjg), 0b10110001);
				//1.- (ar ai)*(-mi mr) = (-ar*mi ai*mr)
				auto vecmv = opCode(mul_ps, mMx, lap);
				//2.- (-ar*mi ai*mr, -ar*mi ai*mr)
				auto vecma = opCode(add_ps, opCode(permute_ps, vecmv, 0b10110001), vecmv);
				//3.- (vr vi)*(-mi mr) = (-vr*mi vi*mr)
				vecmv = opCode(mul_ps, mPy, lap);
				//4.- (-vr*mi vi*mr, -vr*mi vi*mr)
				vecmv = opCode(add_ps, opCode(permute_ps, vecmv, 0b10110001), vecmv);
#else
				lap = opCode(mul_ps, mel, cjg);
				lap = opCode(shuffle_ps, lap, lap, 0b10110001);
				auto vecmv = opCode(mul_ps, mMx, lap);
				auto vecma = opCode(add_ps, opCode(shuffle_ps, vecmv, vecmv, 0b10110001), vecmv);
				vecmv = opCode(mul_ps, mPy, lap);
				vecmv = opCode(add_ps, opCode(shuffle_ps, vecmv, vecmv, 0b10110001), vecmv);
#endif
				//5.- (-ar*mi ai*mr, -ar*mi ai*mr)*(-mi mr)/|m|^2 + R''/R (mr mi)
				mMx   = opCode(add_ps,
									opCode(div_ps, opCode(mul_ps, lap, vecma), mPx),
										opCode(mul_ps, mel, opCode(set1_ps, A)));
				//6.- (-vr*mi vi*mr, -vr*mi vi*mr)*(-mi mr)/|m|^2 + R/R (mr mi)
				mPy   = opCode(add_ps,
									opCode(div_ps, opCode(mul_ps, lap, vecmv), mPx),
										opCode(mul_ps, mel, opCode(set1_ps, Rp)));
			}

			/* update velocities with/without damping */
const bool inRhoSponge = (zC_global >= Tz - nAbsR);
const bool inZSponge   = (j + (Nc-1)*Sfold >= Nx - nAbsZ);

			switch (VQcd & V_DAMP) {

				default:
				case	V_NONE:
if (!(inRhoSponge || inZSponge)) {
    // bulk: original update, zero overhead
#if defined(__AVX512F__) || defined(__FMA__)
    tmp = opCode(fmadd_ps, mMx, opCode(set1_ps, dzc), mPy);
#else
    tmp = opCode(add_ps, mPy, opCode(mul_ps, mMx, opCode(set1_ps, dzc)));
#endif
} else {
    alignas(Align) float sigAux[2*step];

    float sigRho = 0.f;
    if (inRhoSponge) {
        const float u = float(zC_global - (Tz - nAbsR)) / float(nAbsR);
        sigRho = sigAbsR * u * u;
    }

    for (size_t q = 0; q < Nc; q++) {
        float sig = sigRho;

        if (inZSponge) {
            const uint zFast = j + q*Sfold;
            if (zFast >= Nx - nAbsZ) {
                const float u = float(zFast - (Nx - nAbsZ)) / float(nAbsZ);
                sig += sigAbsZ * u * u;
            }
        }

        sigAux[2*q + 0] = sig;
        sigAux[2*q + 1] = sig;
    }

    const _MData_ sigVec = opCode(load_ps, sigAux);
    const _MData_ one    = opCode(set1_ps, 1.f);
    const _MData_ half   = opCode(set1_ps, 0.5f);
    const _MData_ dzv    = opCode(set1_ps, dzc);

    const _MData_ mola  = opCode(mul_ps, half, opCode(mul_ps, sigVec, dzv));
    const _MData_ damp1 = opCode(div_ps, one, opCode(add_ps, one, mola));
    const _MData_ damp2 = opCode(mul_ps, opCode(sub_ps, one, mola), damp1);

    tmp = opCode(add_ps,
                 opCode(mul_ps, mPy, damp2),
                 opCode(mul_ps, opCode(mul_ps, mMx, dzv), damp1));
}

				break;

				case	V_DAMP_RHO:
				{
// NEW implementation
// V = (V+Adt) - (epsi M/|M|^2)(2 MV+ MA*dt - 2 |M|^2/t)
// recall
// V=mPy     A=mMx    |M|^2=mPx      M=mel
// vecmv = MV
					tmp = opCode(mul_ps, mel, mPy);
#if	defined(__AVX__)// || defined(__AVX512F__)
					auto vecmv = opCode(add_ps, opCode(permute_ps, tmp, 0b10110001), tmp);
#else
					auto vecmv = opCode(add_ps, opCode(shuffle_ps, tmp, tmp, 0b10110001), tmp);
#endif

					// vecma = MA
					// mel = M, mMx = A
					tmp = opCode(mul_ps, mel, mMx);
#if	defined(__AVX__)// || defined(__AVX512F__)
					auto vecma = opCode(add_ps, opCode(permute_ps, tmp, 0b10110001), tmp);
#else
					auto vecma = opCode(add_ps, opCode(shuffle_ps, tmp, tmp, 0b10110001), tmp);
#endif

#if	defined(__AVX512F__) || defined(__FMA__)
// A*dzc + mPy - epsi (M/|M|^2)(2*(vecmv-|M|^2 R'/R) +vecma dzc)
tmp = opCode(sub_ps,
	opCode(fmadd_ps, mMx, opCode(set1_ps, dzc), mPy),
	opCode(mul_ps, opCode(mul_ps, opCode(set1_ps, epsi), opCode(div_ps, mel, mPx)),
		opCode(fmadd_ps, opCode(sub_ps,vecmv,opCode(mul_ps,mPx,opCode(set1_ps, Rp))), opCode(set1_ps, 2.f), opCode(mul_ps, vecma, opCode(set1_ps, dzc)))));
#else

tmp = opCode(sub_ps,
	opCode(add_ps, mPy, opCode(mul_ps, mMx, opCode(set1_ps, dzc))),
	opCode(mul_ps, opCode(mul_ps, opCode(set1_ps, epsi), opCode(div_ps, mel, mPx)),
		opCode(add_ps,
			opCode(mul_ps, opCode(sub_ps,vecmv,opCode(mul_ps,mPx,opCode(set1_ps, Rp))), opCode(set1_ps, 2.f)),
			opCode(mul_ps, vecma, opCode(set1_ps, dzc)))));
#endif
				}
				break;

				case	V_DAMP_ALL:
				// damping all directions implementation
#if	defined(__AVX512F__) || defined(__FMA__)
				tmp = opCode(fmadd_ps, mPy, opCode(set1_ps, damp2), opCode(mul_ps, mMx, opCode(set1_ps, damp1*dzc)));
#else
				tmp = opCode(add_ps, opCode(mul_ps, mPy, opCode(set1_ps, damp2)), opCode(mul_ps, mMx, opCode(set1_ps, damp1*dzc)));
#endif
				break;
			}

			// if only evolution along r is desired project v into rho (m) direction in complex space
			// we use vecma = m*v_update
			if (VQcd & V_EVOL_RHO)
			{
				//1.- (mr mi)*(vr vi) = (mr*vr vi*mi)
				auto vecmv = opCode(mul_ps, mel, tmp);
				//2.- (mr*vr+vi*mi, mr*vr+vi*mi)
#if	defined(__AVX__)// || defined(__AVX512F__)
				auto vecma = opCode(add_ps, opCode(permute_ps, vecmv, 0b10110001), vecmv);
#else
				auto vecma = opCode(add_ps, opCode(shuffle_ps, vecmv, vecmv, 0b10110001), vecmv);
#endif
				//3.- (mr*vr+vi*mi, mr*vr+vi*mi)*(mr,mi)/|m|^2 = [mr*vr+vi*mi]/|m|^2 * (mr,mi)
				tmp   = opCode(div_ps, opCode(mul_ps, mel, vecma), mPx);
			}

			opCode(store_ps,  &v[idxV0], tmp);

			if constexpr (UpdateM)
			{
			#if	defined(__AVX512F__) || defined(__FMA__)
						mPx = opCode(fmadd_ps, tmp, opCode(set1_ps, dzd), mel);
			#else
						mPx = opCode(add_ps, mel, opCode(mul_ps, tmp, opCode(set1_ps, dzd)));
			#endif
			opCode(stream_ps, &m2[idxP0], mPx);	// Avoids cache thrashing
			}
		}
		} // end zC,xC loops
	  } //
		}  // end zT zTile loop
		} // end parallel
#undef	_MData_
#undef	step
	}
}





inline void updateMXeon(void * __restrict__ m_,
                        const void * __restrict__ v_,
                        PropParms ppar,
                        const double dz,
                        const double d,
                        const size_t Vo,
                        const size_t Vf,
                        FieldPrecision precision,
                        const unsigned int bSizeX,
                        const unsigned int bSizeY,
                        const unsigned int bSizeZ)
{
	const size_t Nx = ppar.Lx;
	const size_t Sf = Nx;

	const uint z0 = Vo / Sf;
	const uint zF = Vf / Sf;

	const uint zTiles = (zF - z0 + bSizeZ - 1) / bSizeZ;
	const uint xTiles = (Nx  + bSizeX - 1) / bSizeX;

	if (precision == FIELD_DOUBLE)
	{
#if defined(__AVX512F__)
		#define _MData_ __m512d
		#define step 4
#elif defined(__AVX__)
		#define _MData_ __m256d
		#define step 2
#else
		#define _MData_ __m128d
		#define step 1
#endif

		double * __restrict__ m       = (double * __restrict__) __builtin_assume_aligned(m_, Align);
		const double * __restrict__ v = (const double * __restrict__) __builtin_assume_aligned(v_, Align);

		const double dzd = dz * d;

		#pragma omp parallel default(shared)
		{
			_MData_ mIn, vIn, tmp;

			#pragma omp for collapse(2) schedule(static)
			for (uint zT = 0; zT < zTiles; zT++) {
				for (uint xT = 0; xT < xTiles; xT++) {

					const uint zBeg = z0 + zT * bSizeZ;
					const uint zEnd = (zBeg + bSizeZ < zF) ? (zBeg + bSizeZ) : zF;

					const uint xBeg = xT * bSizeX;
					const uint xEnd = (xBeg + bSizeX < Nx) ? (xBeg + bSizeX) : Nx;

					for (uint zC = zBeg; zC < zEnd; zC++) {
						for (uint xC = xBeg; xC < xEnd; xC += step) {

							const size_t idx = zC * Nx + xC;

							const size_t idxM0 = (idx << 1);
							const size_t idxV0 = ((idx - Vo) << 1);

#if defined(__AVX512F__) || defined(__FMA__)
							vIn = opCode(load_pd, &v[idxV0]);
							mIn = opCode(load_pd, &m[idxM0]);
							tmp = opCode(fmadd_pd, opCode(set1_pd, dzd), vIn, mIn);
#else
							vIn = opCode(load_pd, &v[idxV0]);
							mIn = opCode(load_pd, &m[idxM0]);
							tmp = opCode(add_pd, mIn, opCode(mul_pd, opCode(set1_pd, dzd), vIn));
#endif
							opCode(store_pd, &m[idxM0], tmp);
						}
					}
				}
			}
		}

#undef _MData_
#undef step
	}
	else if (precision == FIELD_SINGLE)
	{
#if defined(__AVX512F__)
		#define _MData_ __m512
		#define step 8
#elif defined(__AVX__)
		#define _MData_ __m256
		#define step 4
#else
		#define _MData_ __m128
		#define step 2
#endif

		float * __restrict__ m       = (float * __restrict__) __builtin_assume_aligned(m_, Align);
		const float * __restrict__ v = (const float * __restrict__) __builtin_assume_aligned(v_, Align);

		const float dzd = (float)(dz * d);

		#pragma omp parallel default(shared)
		{
			_MData_ mIn, vIn, tmp;

			#pragma omp for collapse(2) schedule(static)
			for (uint zT = 0; zT < zTiles; zT++) {
				for (uint xT = 0; xT < xTiles; xT++) {

					const uint zBeg = z0 + zT * bSizeZ;
					const uint zEnd = (zBeg + bSizeZ < zF) ? (zBeg + bSizeZ) : zF;

					const uint xBeg = xT * bSizeX;
					const uint xEnd = (xBeg + bSizeX < Nx) ? (xBeg + bSizeX) : Nx;

					for (uint zC = zBeg; zC < zEnd; zC++) {
						for (uint xC = xBeg; xC < xEnd; xC += step) {

							const size_t idx = zC * Nx + xC;

							const size_t idxM0 = (idx << 1);
							const size_t idxV0 = ((idx - Vo) << 1);

#if defined(__AVX512F__) || defined(__FMA__)
							vIn = opCode(load_ps, &v[idxV0]);
							mIn = opCode(load_ps, &m[idxM0]);
							tmp = opCode(fmadd_ps, opCode(set1_ps, dzd), vIn, mIn);
#else
							vIn = opCode(load_ps, &v[idxV0]);
							mIn = opCode(load_ps, &m[idxM0]);
							tmp = opCode(add_ps, mIn, opCode(mul_ps, opCode(set1_ps, dzd), vIn));
#endif
							opCode(store_ps, &m[idxM0], tmp);
						}
					}
				}
			}
		}

#undef _MData_
#undef step
	}
}





#undef	opCode
#undef	opCode_N
#undef	opCode_P
#undef	Align
#undef	_PREFIX_
