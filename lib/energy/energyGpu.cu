#include "complexGpu.cuh"
#include "utils/index.cuh"

#include "enum-field.h"

//#include "scalar/varNQCD.h"
//#include "utils/parse.h"

#include "utils/reduceGpu.cuh"

#define	BLSIZE 512

using namespace gpuCu;
using namespace indexHelper;

template<const bool map, const VqcdType VQcd, typename Float>
static __device__ __forceinline__ void	energyCoreGpu(const uint idx, const complex<Float> * __restrict__ m, const complex<Float> * __restrict__ v, Float * __restrict__ m2,
						      const uint Lx, const uint Sf, const uint Vf, const Float Rpp, const Float iZ, const Float iZ2, const Float zQ, const Float lZ,
						      const Float o2, double *tR, const Float shift)
{
	uint X[3], idxPx, idxPy, idxPz, idxMx, idxMy, idxMz;

	idx2Vec(idx, X, Lx);

	if (X[0] == Lx-1)
		idxPx = idx - Lx+1;
	else
		idxPx = idx+1;

	if (X[0] == 0)
		idxMx = idx + Lx-1;
	else
		idxMx = idx-1;

	if (X[1] == Lx-1)
		idxPy = idx - Sf + Lx;
	else
		idxPy = idx + Lx;

	if (X[1] == 0)
		idxMy = idx + Sf - Lx;
	else
		idxMy = idx - Lx;

	idxPz = idx + Sf;
	idxMz = idx - Sf;

	complex<Float> tmp = m[idx];
	complex<Float> tp2 = tmp - shift;

	Float mod = tmp.real()*tmp.real() + tmp.imag()*tmp.imag();
	Float md2 = tp2.real()*tp2.real() + tp2.imag()*tp2.imag();
	Float mFac = iZ2*mod;
	Float mFc2 = iZ2*md2;
	Float iMod = 1./mod;

	complex<Float> mPx = (m[idxPx] - tmp)*conj(tmp)*iMod;
	complex<Float> mPy = (m[idxPy] - tmp)*conj(tmp)*iMod;
	complex<Float> mPz = (m[idxPz] - tmp)*conj(tmp)*iMod;
	complex<Float> mMx = (m[idxMx] - tmp)*conj(tmp)*iMod;
	complex<Float> mMy = (m[idxMy] - tmp)*conj(tmp)*iMod;
	complex<Float> mMz = (m[idxMz] - tmp)*conj(tmp)*iMod;
	complex<Float> vOm = v[idxMz]*conj(tmp)*iMod - gpuCu::complex<Float>(Rpp*iZ, 0.);

	Float rGrx = ((Float) (mFac*(mPx.real()*mPx.real() + mMx.real()*mMx.real())));
	Float tGrx = ((Float) (mFac*(mPx.imag()*mPx.imag() + mMx.imag()*mMx.imag())));
	Float rGry = ((Float) (mFac*(mPy.real()*mPy.real() + mMy.real()*mMy.real())));
	Float tGry = ((Float) (mFac*(mPy.imag()*mPy.imag() + mMy.imag()*mMy.imag())));
	Float rGrz = ((Float) (mFac*(mPz.real()*mPz.real() + mMz.real()*mMz.real())));
	Float tGrz = ((Float) (mFac*(mPz.imag()*mPz.imag() + mMz.imag()*mMz.imag())));
	Float rKin = ((Float) (mFac*vOm.real()*vOm.real()));
	Float tKin = ((Float) (mFac*vOm.imag()*vOm.imag()));

	Float tPot, rPot;

	switch (VQcd & V_PQ) {
		case	V_PQ1:
			rPot  = ((Float) (mFc2 - 1.)*(mFc2 - 1.));
			break;
		case	V_PQ2:
			rPot  = ((Float) (mFc2*mFc2 - 1.));
			rPot *= rPot;
			break;
	}

	switch (VQcd & V_QCD) {
		case	V_QCD0:
			tPot  = 0.;
			break;
		case	V_QCD1:
		case	V_QCDC:
			tPot  = (((Float) 1.) - tp2.real()/sqrt(md2));
			break;
		case	V_QCDV:
			tPot  = 0.5*(((Float) 1.) - tp2.real()*iZ)*(((Float) 1.) - tp2.real()*iZ) +
							0.5*(tp2.imag()*iZ)*(tp2.imag()*iZ) ;
			break;
		// case	V_QCDL:
		// 	rPot  = ((Float) (mFc2 - 1.)*(mFc2 - 1.));
		// 	tPot  = 0.;
		// 	break;
	}


	tR[RH_GRX] = (double) rGrx;
        tR[TH_GRX] = (double) tGrx;
        tR[RH_GRY] = (double) rGry;
        tR[TH_GRY] = (double) tGry;
        tR[RH_GRZ] = (double) rGrz;
        tR[TH_GRZ] = (double) tGrz;
        tR[RH_POT] = (double) rPot;
        tR[TH_POT] = (double) tPot;
        tR[RH_KIN] = (double) rKin;
        tR[TH_KIN] = (double) tKin;

	if (map == true) {
		m2[idxMz]  = o2*(tGrx + tGry + tGrz) + 0.5*tKin + zQ*tPot;
		m2[idx+Vf] = o2*(rGrx + rGry + rGrz) + 0.5*rKin + lZ*rPot;
	}
}

template<const bool map, const VqcdType VQcd, typename Float>
__global__ void	energyKernel(const complex<Float> * __restrict__ m, const complex<Float> * __restrict__ v, Float * __restrict__ m2, const uint Lx, const uint Sf, const uint V,
			     const Float Rpp, const Float iZ, const Float iZ2, const Float zQ, const Float lZ, const Float o2, double *eR, double *partial, const Float shift)
{
	uint idx = Sf + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));

	double tmp[10] = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };

	if	(idx < V)
		energyCoreGpu<map,VQcd,Float>(idx, m, v, m2, Lx, Sf, V, Rpp, iZ, iZ2, zQ, lZ, o2, tmp, shift);

	reduction<BLSIZE,double,10>   (eR, tmp, partial);
}

int	energyGpu	(const void * __restrict__ m, const void * __restrict__ v, void * __restrict__ m2, const double zR, const double Rpp, const double delta2, const double LL, const double aMass2, const double shift,
			 const VqcdType VQcd, const uint Lx, const uint Lz, const uint V, const uint S, FieldPrecision precision, double *eR, cudaStream_t &stream, const bool map)
{
	const uint Vm = V+S;
	const uint Lz2 = V/(Lx*Lx);
	dim3  gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3  blockSize(BLSIZE,1,1);
	const int nBlocks = gridSize.x*gridSize.y;

	const double o2  = 0.25/delta2;
	const double zQ  = aMass2*zR*zR;
	const double lZ  = 0.25*LL*zR*zR;

	double *tR, *partial;

	if ((cudaMalloc(&tR, sizeof(double)*10) != cudaSuccess) || (cudaMalloc(&partial, sizeof(double)*10*nBlocks*4) != cudaSuccess))
		return -1;

	if (precision == FIELD_DOUBLE)
	{
		const double iZ  = 1./zR;
		const double iZ2 = iZ*iZ;

		switch (VQcd & V_TYPE) {
			case	V_QCD0_PQ1:
				if (map == true)
					energyKernel<true, V_QCD0_PQ1><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<double>*>(m), static_cast<const complex<double>*>(v), static_cast<double*>(m2), Lx, S, Vm, Rpp, iZ, iZ2, zQ, lZ, o2, tR, partial, shift);
				else
					energyKernel<false,V_QCD0_PQ1><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<double>*>(m), static_cast<const complex<double>*>(v), static_cast<double*>(m2), Lx, S, Vm, Rpp, iZ, iZ2, zQ, lZ, o2, tR, partial, shift);
				break;

			case	V_QCD1_PQ1:
				if (map == true)
					energyKernel<true, V_QCD1_PQ1><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<double>*>(m), static_cast<const complex<double>*>(v), static_cast<double*>(m2), Lx, S, Vm, Rpp, iZ, iZ2, zQ, lZ, o2, tR, partial, shift);
				else
					energyKernel<false,V_QCD1_PQ1><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<double>*>(m), static_cast<const complex<double>*>(v), static_cast<double*>(m2), Lx, S, Vm, Rpp, iZ, iZ2, zQ, lZ, o2, tR, partial, shift);
				break;

			case	V_QCD1_PQ2:
				if (map == true)
					energyKernel<true, V_QCD1_PQ2><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<double>*>(m), static_cast<const complex<double>*>(v), static_cast<double*>(m2), Lx, S, Vm, Rpp, iZ, iZ2, zQ, lZ, o2, tR, partial, shift);
				else
					energyKernel<false,V_QCD1_PQ2><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<double>*>(m), static_cast<const complex<double>*>(v), static_cast<double*>(m2), Lx, S, Vm, Rpp, iZ, iZ2, zQ, lZ, o2, tR, partial, shift);
				break;

			case	V_QCDV_PQ1:
				if (map == true)
					energyKernel<true, V_QCDV_PQ1><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<double>*>(m), static_cast<const complex<double>*>(v), static_cast<double*>(m2), Lx, S, Vm, Rpp, iZ, iZ2, zQ, lZ, o2, tR, partial, shift);
				else
					energyKernel<false,V_QCDV_PQ1><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<double>*>(m), static_cast<const complex<double>*>(v), static_cast<double*>(m2), Lx, S, Vm, Rpp, iZ, iZ2, zQ, lZ, o2, tR, partial, shift);
				break;

			default:
				break;
		}
	}
	else if (precision == FIELD_SINGLE)
	{
		const float iZ  = 1./zR;
		const float iZ2 = iZ*iZ;

		switch (VQcd) {
			case	V_QCD0_PQ1:
				if (map == true)
					energyKernel<true, V_QCD0_PQ1><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<float>*>(m), static_cast<const complex<float>*>(v), static_cast<float*>(m2), Lx, S, Vm, (float) Rpp, iZ, iZ2, (float) zQ, (float) lZ, (float) o2, tR, partial, (float) shift);
				else
					energyKernel<false,V_QCD0_PQ1><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<float>*>(m), static_cast<const complex<float>*>(v), static_cast<float*>(m2), Lx, S, Vm, (float) Rpp, iZ, iZ2, (float) zQ, (float) lZ, (float) o2, tR, partial, (float) shift);
				break;

			case	V_QCD1_PQ1:
				if (map == true)
					energyKernel<true, V_QCD1_PQ1><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<float>*>(m), static_cast<const complex<float>*>(v), static_cast<float*>(m2), Lx, S, Vm, (float) Rpp, iZ, iZ2, (float) zQ, (float) lZ, (float) o2, tR, partial, (float) shift);
				else
					energyKernel<false,V_QCD1_PQ1><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<float>*>(m), static_cast<const complex<float>*>(v), static_cast<float*>(m2), Lx, S, Vm, (float) Rpp, iZ, iZ2, (float) zQ, (float) lZ, (float) o2, tR, partial, (float) shift);
				break;

			case	V_QCD1_PQ2:
				if (map == true)
					energyKernel<true, V_QCD1_PQ2><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<float>*>(m), static_cast<const complex<float>*>(v), static_cast<float*>(m2), Lx, S, Vm, (float) Rpp, iZ, iZ2, (float) zQ, (float) lZ, (float) o2, tR, partial, (float) shift);
				else
					energyKernel<false,V_QCD1_PQ2><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<float>*>(m), static_cast<const complex<float>*>(v), static_cast<float*>(m2), Lx, S, Vm, (float) Rpp, iZ, iZ2, (float) zQ, (float) lZ, (float) o2, tR, partial, (float) shift);
				break;

			case	V_QCDV_PQ1:
				if (map == true)
					energyKernel<true, V_QCDV_PQ1><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<float>*>(m), static_cast<const complex<float>*>(v), static_cast<float*>(m2), Lx, S, Vm, (float) Rpp, iZ, iZ2, (float) zQ, (float) lZ, (float) o2, tR, partial, (float) shift);
				else
					energyKernel<false,V_QCDV_PQ1><<<gridSize,blockSize,0,stream>>> (static_cast<const complex<float>*>(m), static_cast<const complex<float>*>(v), static_cast<float*>(m2), Lx, S, Vm, (float) Rpp, iZ, iZ2, (float) zQ, (float) lZ, (float) o2, tR, partial, (float) shift);
				break;

			default:
				break;
		}
	}

	cudaDeviceSynchronize();

	cudaMemcpy(eR, tR, sizeof(double)*10, cudaMemcpyDeviceToHost);
	cudaFree(tR); cudaFree(partial);

	eR[TH_GRX] *= o2;
	eR[TH_GRY] *= o2;
	eR[TH_GRZ] *= o2;
	eR[TH_KIN] *= .5;
	eR[TH_POT] *= zQ;
	eR[RH_GRX] *= o2;
	eR[RH_GRY] *= o2;
	eR[RH_GRZ] *= o2;
	eR[RH_KIN] *= .5;
	eR[RH_POT] *= lZ;

	return	0;
}
