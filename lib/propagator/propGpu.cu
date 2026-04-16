#include "kernelParms.cuh"
#include "complexGpu.cuh"
#include "utils/index.cuh"
#include "enum-field.h"
#include "propagator/prop-def-mac.h"


//#include "utils/parse.h"
//#include "scalar/varNQCD.h"

#define	BLSIZE 256
#define	BSSIZE 256

using namespace gpuCu;
using namespace indexHelper;

#include "cudaErrors.h"

template<typename Float, const VqcdType VQcd, bool UpdateM>
static __device__ __forceinline__
void propagateCoreGpu(
	const uint idx,
	const complex<Float>* __restrict__ m,
	complex<Float>* __restrict__ v,
	complex<Float>* __restrict__ m2,
	const Float z, const Float z2, const Float z4, const Float zQ,
	const Float gFac, const Float eps, const Float dp1, const Float dp2,
	const Float dzc, const Float dzd,
	const Float* __restrict__ ood2,
	const Float LL, const uint Lx, const uint Sf, const uint NN)
	{
	uint X[3], idxPx, idxPy, idxMx, idxMy;

	complex<Float> mel, a, tmp, zN;

	switch	(VQcd & V_QCD) {
			case	V_QCD2:
			zN = (Float) (zQ/z)/2 * complex<Float>(1,-1);
			break;

			default:
			case	V_QCDC:
			case	V_QCDL:
			zN = (Float) (zQ*z) * complex<Float>(1,-1);
			break;
	}

	mel = complex<Float>(0,0);
	tmp = m[idx];

#ifdef USE_2DCYL

	uint X0 = idx % Lx;

	complex<Float> malPx, malMx, malPz, malMz;

	for (size_t nv=1; nv <= NN; nv++)
	{
		if (X0 + nv >= Lx)
			malPx = m[idx]; 	// this cancels this term, here we would need absorbing boundary conditions ...
		else
			malPx = m[idx + nv];

		if (X0 < nv)
			malMx = conj(m[idx + (nv-X0)]); // symmetric boundary conditions around x=0
		else
			malMx = m[idx - nv];

		malPz = m[idx + nv*Lx]; // requires a special ghost at rank Np - 1
		malMz = m[idx - nv*Lx]; // requires a special ghost at rank 0

		const Float c_lap = ood2[nv - 1];
		const Float c_der = ood2[NN + nv - 1];

		if (X[0]==0)
			mel += (malPx+malMx+malPx+malMx + malPz+malMz - ((Float) 6.)*tmp)*c_lap ;
		else
			mel += (malPx+malMx+malPz+malMz - ((Float) 4.)*tmp)*c_lap + (malPz - malMz)/((Float) X[0])*c_der;
	}
#else

	idx2Vec(idx, X, Lx); // will only work for Lx=Ly

	for (size_t nv=1; nv <= NN; nv++)
	{
		if (X[0] + nv >= Lx)
			idxPx = idx + nv - Lx;
		else
			idxPx = idx + nv;

		if (X[0] < nv)
			idxMx = idx + Lx - nv;
		else
			idxMx = idx - nv;

		if (X[1] + nv >= Lx)
			idxPy = idx + nv*Lx - Sf;
		else
			idxPy = idx + nv*Lx;

		if (X[1] < nv)
			idxMy = idx + Sf - nv*Lx;
		else
			idxMy = idx - nv*Lx;

		const Float c_lap = ood2[nv - 1];
		mel += (m[idxMx] + m[idxPx] + m[idxPy] + m[idxMy] + m[idx+nv*Sf] + m[idx-nv*Sf] - ((Float) 6.)*tmp)*c_lap;
	}
#endif

	Float pot = tmp.real()*tmp.real() + tmp.imag()*tmp.imag();

	switch (VQcd & V_PQ) {
		default:
		case	V_PQ1:
			a  = mel - tmp*(((Float) LL)*(pot - z2));
			break;
		case	V_PQ2:
			a  = mel - tmp*pot*(((Float) LL)*(pot*pot - z4))*((Float) 2.)/z4;
			break;
		case	V_NONE:
			a  = mel ;
			break;
	}

	switch (VQcd & V_QCD) {
		default:
		case	V_QCD1:
			a  += complex<Float>(zQ,0);
			break;
		case	V_QCDV:
			a  += (z - tmp)*zQ;
			break;
		case	V_QCD2:
			a  -= tmp*zN;
			break;
		case	V_QCDC:
			{
			Float pota = 1/pot*sqrt(pot);
			mel = complex<Float>(tmp.imag()*tmp.imag(),tmp.imag()*tmp.real());
			a  -= pota*mel*zN;
			}
			break;
		case	V_QCDL:
			mel = complex<Float>(tmp.imag(),tmp.real());
			a += mel*atan2(tmp.real(),tmp.imag())*zN/pot;
			break;
		case	V_QCD0:
			break;
	}

#ifdef USE_2DCYL
	mel = v[idx-NN*Lx];
#else
	mel = v[idx-NN*Sf];
#endif

	switch (VQcd & V_DAMP) {
		case	V_NONE:
		mel += a*dzc;
		break;

		case	V_DAMP_RHO:
		{
			Float vec  = tmp.real()*mel.real() + tmp.imag()*mel.imag();
			Float vea  = tmp.real()*a.real()   + tmp.imag()*a.imag();
			a   += tmp*gFac;
			mel += a*dzc - (tmp/pot)*eps*(((Float) 2.)*vec + vea*dzc);
		}
		break;

		case	V_DAMP_ALL:
		mel = mel*dp2 + a*dp1*dzc;
		break;
	}


	if (VQcd & V_EVOL_RHO) {
		Float kReal = tmp.real()*mel.real() + tmp.imag()*mel.imag();
		mel *= kReal/pot;
	}

#ifdef USE_2DCYL
	v[idx-NN*Lx] = mel;
#else
	v[idx-NN*Sf] = mel;
#endif

	if constexpr (UpdateM)
	{
		mel *= dzd;
		tmp += mel;
		m2[idx] = tmp;
	}

}

template<typename Float, const VqcdType VQcd, bool UpdateM>
__global__ void	propagateKernel(const complex<Float> * __restrict__ m, complex<Float> * __restrict__ v, complex<Float> * __restrict__ m2, const Float z, const Float z2, const Float z4,
				const Float zQ, const Float gFac, const Float eps, const Float dp1, const Float dp2, const Float dzc, const Float dzd, const Float* __restrict__ ood2, const Float LL,
				const uint Lx, const uint Sf, const uint Vo, const uint Vf, const uint NN)
{
	//uint idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));
//	uint idx = Vo + (threadIdx.x + blockDim.x*blockIdx.x) + Sf*(threadIdx.y + blockDim.y*blockIdx.y);
//
//	if	(idx >= Vf)
//		return;
#ifdef USE_2DCYL
	uint X = threadIdx.x + blockDim.x * blockIdx.x;
	uint Z = threadIdx.y + blockDim.y * blockIdx.y;
	if (X >= Lx || Z >= Sf/Lx) return;
	uint idx = Vo + X + Lx*Z;
#else
	uint idx = Vo + (threadIdx.x + blockDim.x*blockIdx.x)
        	     + Sf*(threadIdx.y + blockDim.y*blockIdx.y);
	if (idx >= Vf) return;
#endif

	propagateCoreGpu<Float, VQcd, UpdateM>(idx, m, v, m2, z, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, LL, Lx, Sf, NN);
}

void	propagateGpu(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, PropParms ppar, const double dz, const double c, const double d,
				const uint Vo, const uint Vf, const VqcdType VQcd, FieldPrecision precision, const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream)
{
	if (Vo>Vf)
		return ;
/*
	const uint Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3	  gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3	  blockSize(BLSIZE,1,1);
*/
	bool UpdateM = (m2 != m);
	if (UpdateM)
		LogMsg(VERB_PARANOID,"[pG] propGPU called");
	else
		LogMsg(VERB_PARANOID,"[pG] updateVGPU called");

	//LogMsg(VERB_PARANOID,"[pG] dz %f c %f d %f Vo %lu Vf %lu VQcd %lu precision %d x y xBlock %lu %lu %lu",dz,c,d,Vo,Vf,VQcd,precision,xBlock,yBlock,zBlock);

	const uint Lx    = ppar.Lx;
	const uint Ly    = ppar.Ly;
	const uint Lz    = ppar.Lz;
#ifdef USE_2DCYL
        const uint Sf  = Lx*Lz;
        const uint Lz2 = (Vf-Vo)/Sf;
        dim3 gridSize((Lx+xBlock-1)/xBlock, (Lz+zBlock-1)/zBlock, 1);
	dim3 blockSize(xBlock, zBlock, 1);
#else
	const uint Sf  = Lx*Ly;
	const uint Lz2 = (Vf-Vo)/Sf;
	dim3 gridSize((Sf+xBlock-1)/xBlock, (Lz2+yBlock-1)/yBlock, 1);
	dim3 blockSize(xBlock, yBlock, 1);
#endif
//LogMsg(VERB_PARANOID,"[pG] gridsize %d %d %d ",(Sf+xBlock-1)/xBlock,(Lz2+yBlock-1)/yBlock,1);
	const uint NN    = ppar.Lap;

//	LogMsg(VERB_PARANOID,"[pG] allocate %d bits for lap/der coefficients NN = %d",2*NN*sizeof(double), NN);

//a
//	for (int i =0; i<NN; i++) {
//		LogMsg(VERB_PARANOID,"C_LAP[%d] %f C_DER[%d] %f", i, (ppar.PC)[i],i,(ppar.PCp)[i]);
//	}


//	LogMsg(VERB_HIGH,"m=%p v=%p m2=%p ",m, v, m2);LogFlush();

	LogMsg(VERB_HIGH, "grid=(%u,%u,%u) block=(%u,%u,%u)\n", (unsigned)gridSize.x, (unsigned)gridSize.y, (unsigned)gridSize.z,
 (unsigned)blockSize.x, (unsigned)blockSize.y, (unsigned)blockSize.z);LogFlush();

	if (precision == FIELD_DOUBLE)
	{
		const double dzc  = dz*c;
		const double dzd  = dz*d;
		const double zR   = ppar.R;
		const double z2   = zR*zR;
		const double z4   = z2*z2;
		const double zQ   = ppar.massA2*z2*zR;
		const double LL   = ppar.lambda;
		const double gFp1 = ppar.gamma/zR;
		const double gFac = gFp1/zR;
		const double gFp2 = gFp1*dzc/2.;
		const double eps  = gFp2/(1. + gFp2);
		const double dp1  =   1./(1. + gFp2);
		const double dp2  = (1. - gFp2)*dp1;

		double *ood2 = nullptr;
		cudaMalloc(&ood2, 2*NN*sizeof(double));

		std::vector<double> aux(2*NN);
		for (int i =0; i<NN; i++){
	        	aux[i] = (double) ((ppar.PC)[i]*ppar.ood2a);
						aux[NN+i] = (double) ((ppar.PCp)[i]*ppar.ood2a);
		}
		cudaMemcpy(ood2,aux.data(),2*NN*sizeof(double),cudaMemcpyHostToDevice);

		switch (VQcd) {

			DEFALLPROPTEM_K_GPU(double)

			default:
			return;
		}
		cudaFree(ood2);
	} else if (precision == FIELD_SINGLE) {
		const float dzc  = dz*c;
		const float dzd  = dz*d;
		const float zR   = ppar.R;
		const float z2   = zR*zR;
		const float z4   = z2*z2;
		const float zQ   = ppar.massA2*z2*zR; //(float) axionmass2(*z, nQcd, zthres, zrestore)*z2*zR;
		const float LL   = ppar.lambda;
		const float gFp1 = ppar.gamma/zR;
		const float gFac = gFp1/zR;
		const float gFp2 = gFp1*dzc/2.;
		const float eps  = gFp2/(1. + gFp2);
		const float dp1  =   1./(1. + gFp2);
		const float dp2  = (1. - gFp2)*dp1;

		float *ood2 = nullptr;
		cudaMalloc(&ood2, 2*NN*sizeof(float));

		std::vector<float> aux(2*NN);
		for (int i =0; i<NN; i++){
			aux[i]    = (float) ((ppar.PC)[i]*ppar.ood2a);
			aux[NN+i] = (float) ((ppar.PCp)[i]*ppar.ood2a);
		}
		cudaMemcpy(ood2,aux.data(),2*NN*sizeof(float),cudaMemcpyHostToDevice);

		switch (VQcd) {

			DEFALLPROPTEM_K_GPU(float)

			default:
			cudaFree(ood2);
			return;
		}
		cudaFree(ood2);

	}
	//cudaDeviceSynchronize();

	CudaCheckError();
}

template<typename cFloat, typename Float>
static __device__ void	__forceinline__ updateMCoreGpu(const uint idx, cFloat * __restrict__ m, const cFloat * __restrict__ v, const Float dzd, const uint Sf)
{
	cFloat mm = m[idx], vv = v[idx-Sf];

	mm += vv*dzd;
	m[idx] = mm;
}

template<typename cFloat, typename Float>
__global__ void	updateMKernel(cFloat * __restrict__ m, const cFloat * __restrict__ v, const Float dzd, const uint Lx, const uint Sf, const uint Vo, const uint Vf)
{
	//uint idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));
	uint idx = Vo + (threadIdx.x + blockDim.x*blockIdx.x) + Sf*(threadIdx.y + blockDim.y*blockIdx.y);

	if	(idx >= Vf)
		return;

	updateMCoreGpu<cFloat,Float>(idx, m, v, dzd, Sf);
}

void	updateMGpu(void * __restrict__ m, const void * __restrict__ v, const double dz, const double d, const uint Lx, const uint Vo, const uint Vf, FieldPrecision precision,
		   const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream, FieldType fType=FIELD_SAXION)
{
/*
	const uint Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3	gridSize((Lx*Lx+BSSIZE-1)/BSSIZE,Lz2,1);
	dim3	blockSize(BSSIZE,1,1);
*/
	const uint Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3 gridSize((Lx*Lx+xBlock-1)/xBlock, (Lz2+yBlock-1)/yBlock, 1);
	dim3 blockSize(xBlock, yBlock, 1);

	if (precision == FIELD_DOUBLE)
	{
		const double dzd  = dz*d;
		if (fType & FIELD_AXION)
			updateMKernel<<<gridSize,blockSize,0,stream>>> ((        double *) m, (const         double *) v, dzd, Lx, Lx*Lx, Vo, Vf);
		else
			updateMKernel<<<gridSize,blockSize,0,stream>>> ((complex<double>*) m, (const complex<double>*) v, dzd, Lx, Lx*Lx, Vo, Vf);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float dzd  = dz*d;
		if (fType & FIELD_AXION)
			updateMKernel<<<gridSize,blockSize,0,stream>>> ((        float  *) m, (const         float  *) v, dzd, Lx, Lx*Lx, Vo, Vf);
		else
			updateMKernel<<<gridSize,blockSize,0,stream>>> ((complex<float> *) m, (const complex<float> *) v, dzd, Lx, Lx*Lx, Vo, Vf);
	}
}
