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
		const Float LL, const uint Lx, const uint Lz, const uint Tz,
		const uint Sf, const uint NN)
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

	uint X0 = idx % Lx;                        // z coordinatea
	uint Z  = idx/Lx - NN;
	uint Z0_global = Sf + Z;         // radial coordinate, Sf is Nz*commRank()
	
	complex<Float> malPx, malMx, malPz, malMz;

	for (size_t nv=1; nv <= NN; nv++)
	{
			if (X0 + nv >= Lx)
				/* Smooth constant extrapolation at the open z boundary.  Do not
				 * impose a second conjugation fixed point on an isolated-loop IC. */
				malPx = tmp;
			else
				malPx = m[idx + nv];

			if (X0 < nv)
				malMx = conj(m[idx - X0 + (nv-X0)]); // phi(-x)=conj(phi(x))
		else
			malMx = m[idx - nv];


			if (Z0_global + nv < Tz) {
				malPz = m[idx + nv*Lx];
			} else {
				// Even reflection about the outer face rho=Tz-1/2.
				const uint rhoRef = 2*Tz - 1 - (Z0_global + nv);
				const uint localRef = rhoRef - Sf;
				const uint rowBase = idx - Z*Lx;
				malPz = m[rowBase + localRef*Lx];
			}

		//malMz = m[idx - nv*Lx]; // requires a special ghost at rank 0
		

			if (Z0_global < nv)
				malMz = m[idx - Z*Lx + (nv-Z)*Lx];
		else
			malMz = m[idx - nv*Lx];

		const Float c_lap = ood2[nv - 1];
		const Float c_der = ood2[NN + nv - 1];

		if (Z0_global == 0)
			mel += (malPx+malMx+malPz+malMz + malPz+malMz - ((Float) 6.)*tmp)*c_lap ;
		else
			mel += (malPx+malMx+malPz+malMz - ((Float) 4.)*tmp)*c_lap + (malPz - malMz)/((Float) Z0_global)*c_der;
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
		#ifdef USE_2DCYL
		{
			/* Match the CPU cylindrical propagator's absorbing layer.  Without
			 * this sponge, the conjugate/even outer reflections launch a
			 * lattice-scale wave which subsequently propagates back into the
			 * physical domain. */
			constexpr uint nAbsZ = 16;
			constexpr uint nAbsR = 16;
			constexpr Float sigAbsZ = Float(0.5);
			constexpr Float sigAbsR = Float(0.5);
			Float sigma = Float(0);
			if (Lx > nAbsZ && X0 >= Lx - nAbsZ) {
				const Float u = Float(X0 - (Lx - nAbsZ))/Float(nAbsZ);
				sigma += sigAbsZ*u*u;
			}
			if (Tz > nAbsR && Z0_global >= Tz - nAbsR) {
				const Float u = Float(Z0_global - (Tz - nAbsR))/Float(nAbsR);
				sigma += sigAbsR*u*u;
			}
			if (sigma > Float(0)) {
				const Float sponge = sigma*dzc/Float(2);
				const Float spongeDp1 = Float(1)/(Float(1) + sponge);
				const Float spongeDp2 = (Float(1) - sponge)*spongeDp1;
				mel = mel*spongeDp2 + a*(spongeDp1*dzc);
			} else
				mel += a*dzc;
		}
		#else
		mel += a*dzc;
		#endif
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
__global__ void	propagateKernel(const complex<Float> * __restrict__ m, complex<Float> * __restrict__ v, complex<Float> * __restrict__ m2, 
			const Float z, const Float z2, const Float z4, const Float zQ, const Float gFac, const Float eps, 
			const Float dp1, const Float dp2, const Float dzc, const Float dzd, const Float* __restrict__ ood2, const Float LL,
			const uint Lx, const uint Lz, const uint Tz, const uint Sf,
			const uint Vo, const uint Vf, const uint NN)
{
	//uint idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));
//	uint idx = Vo + (threadIdx.x + blockDim.x*blockIdx.x) + Sf*(threadIdx.y + blockDim.y*blockIdx.y);
//
//	if	(idx >= Vf)
//		return;
#ifdef USE_2DCYL
	uint X = threadIdx.x + blockDim.x * blockIdx.x;
	uint Z = threadIdx.y + blockDim.y * blockIdx.y;
	const uint Lz_2  = (Vf-Vo)/Lx; // # z-slices to calculate
	if (X >= Lx || Z >= Lz_2) return;
	uint idx = Vo + X + Lx*Z;
	// here Lz = Nz (per rank) and
	// Sf includes commRank(), Sf=Nz*commRank()!
#else
	uint idx = Vo + (threadIdx.x + blockDim.x*blockIdx.x)
        	     + Sf*(threadIdx.y + blockDim.y*blockIdx.y);
#endif
	if (idx >= Vf) return;
	propagateCoreGpu<Float, VQcd, UpdateM>(idx, m, v, m2, z, z2, z4, zQ, gFac, eps, dp1, dp2, dzc, dzd, ood2, LL, Lx, Lz, Tz, Sf, NN);
}

void	propagateGpu(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, PropParms ppar, 
			const double dz, const double c, const double d,
			const uint Vo, const uint Vf, const VqcdType VQcd, FieldPrecision precision, 
			const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream)
{
	if (Vo>Vf)
		return ;

	bool UpdateM = (m2 != m);
	if (UpdateM)
		LogMsg(VERB_HIGH,"[pG] propGPU called");
	else
		LogMsg(VERB_HIGH,"[pG] updateVGPU called");

	const uint Lx    = ppar.Lx;
	const uint Ly    = ppar.Ly;
	const uint Tz    = ppar.Tz;
#ifdef USE_2DCYL
	const uint Sf    = ppar.Lz*commRank();
	const uint Lz    = ppar.Lz;
	const uint Lz_2  = (Vf-Vo)/Lx; // # z-slices to calculate
	dim3 gridSize((Lx+xBlock-1)/xBlock, (Lz_2+zBlock-1)/zBlock, 1);
	dim3 blockSize(xBlock, zBlock, 1);
	LogMsg(VERB_HIGH,"[pG2D] Lx %lu Lz %lu z-slices %lu Sf(Nz*nMPI) %lu dz %f c %f d %f Vo %lu Vf %lu VQcd %lu precision %d x y z Block %lu %lu %lu",
		Lx,Lz,Lz_2,Sf,dz,c,d,Vo,Vf,VQcd,precision,xBlock,yBlock,zBlock);
#else
	const uint Lz    = ppar.Lz;
	const uint Sf  = Lx*Ly;
	const uint Lz2 = (Vf-Vo)/Sf;
	dim3 gridSize((Sf+xBlock-1)/xBlock, (Lz2+yBlock-1)/yBlock, 1);
	dim3 blockSize(xBlock, yBlock, 1);
	LogMsg(VERB_HIGH,"[pG] Lx %lu Lz %lu z-slices %lu Sf %lu dz %f c %f d %f Vo %lu Vf %lu VQcd %lu precision %d x y x Block %lu %lu %lu",
		Lx,Lz,Lz2,Sf,dz,c,d,Vo,Vf,VQcd,precision,xBlock,yBlock,zBlock);
#endif
     

//LogMsg(VERB_PARANOID,"[pG] gridsize %d %d %d ",(Sf+xBlock-1)/xBlock,(Lz2+yBlock-1)/yBlock,1);
	const uint NN    = ppar.Lap;

//	LogMsg(VERB_PARANOID,"[pG] allocate %d bits for lap/der coefficients NN = %d",2*NN*sizeof(double), NN);

//a
//	for (int i =0; i<NN; i++) {
//		LogMsg(VERB_PARANOID,"C_LAP[%d] %f C_DER[%d] %f", i, (ppar.PC)[i],i,(ppar.PCp)[i]);
//	}


//	LogMsg(VERB_HIGH,"m=%p v=%p m2=%p ",m, v, m2);LogFlush();

	LogMsg(VERB_HIGH, "grid=(%u,%u,%u) block=(%u,%u,%u)", (unsigned)gridSize.x, (unsigned)gridSize.y, (unsigned)gridSize.z,
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
__global__ void	updateMKernel(cFloat * __restrict__ m, const cFloat * __restrict__ v, 
	const Float dzd, const uint Lx, const uint Sf, const uint Vo, const uint Vf, const uint NN)
{
\
#ifdef USE_2DCYL
	uint X = threadIdx.x + blockDim.x * blockIdx.x;
	uint Z = threadIdx.y + blockDim.y * blockIdx.y;
	const uint Lz_2  = (Vf-Vo)/Lx; // # z-slices to calculate
	if (X >= Lx || Z >= Lz_2) return;
	uint idx = Vo + X + Lx*Z;
	uint m_v_gap = NN*Lx;
#else
	uint idx = Vo + (threadIdx.x + blockDim.x*blockIdx.x)
					+ Sf*(threadIdx.y + blockDim.y*blockIdx.y);
	uint m_v_gap = NN*Sf;
#endif
	if (idx >= Vf) return;
	updateMCoreGpu<cFloat,Float>(idx, m, v, dzd, m_v_gap);
}

void	updateMGpu(void * __restrict__ m, const void * __restrict__ v, PropParms ppar, 
		const double dz, const double d, const uint Vo, const uint Vf, FieldPrecision precision,
		const int xBlock, const int yBlock, const int zBlock, cudaStream_t &stream, FieldType fType=FIELD_SAXION)
{

		if (Vo>Vf)
			return ;
        
			LogMsg(VERB_HIGH,"[pG] updateMGPU called");
	
	const uint NN    = ppar.Lap;
        const uint Lx    = ppar.Lx;
        const uint Ly    = ppar.Ly;

		/* Warning : The usage of Sf depends on the build */
#ifdef USE_2DCYL
        const uint Sf    = ppar.Lz;
        const uint Lz_2  = (Vf-Vo)/Lx; // # z-slices to calculate
        dim3 gridSize((Lx+xBlock-1)/xBlock, (Lz_2+zBlock-1)/zBlock, 1);
        dim3 blockSize(xBlock, zBlock, 1);
    	LogMsg(VERB_HIGH,"[uMGpu2D] Lx %lu Lz_offset %lu Sf(Lz) %lu dz %f d %f Vo %lu Vf %lu precision %d x y z Block %lu %lu %lu",
			Lx,Lz_2,Sf,dz,d,Vo,Vf,precision,xBlock,yBlock,zBlock);
#else
        const uint Lz    = ppar.Lz;
        const uint Sf  = Lx*Ly;
        const uint Lz2 = (Vf-Vo)/Sf;
        dim3 gridSize((Sf+xBlock-1)/xBlock, (Lz2+yBlock-1)/yBlock, 1);
        dim3 blockSize(xBlock, yBlock, 1);
     	LogMsg(VERB_HIGH,"[uMGpu] Lx %lu Sf %lu dz %f d %f Vo %lu Vf %lu precision %d x y z Block %lu %lu %lu",
			Lx,Sf,dz,d,Vo,Vf,precision,xBlock,yBlock,zBlock);
#endif


	if (precision == FIELD_DOUBLE)
	{
		const double dzd  = dz*d;
		if (fType & FIELD_AXION)
			updateMKernel<<<gridSize,blockSize,0,stream>>> ((        double *) m, (const         double *) v, dzd, Lx, Sf, Vo, Vf,NN);
		else
			updateMKernel<<<gridSize,blockSize,0,stream>>> ((complex<double>*) m, (const complex<double>*) v, dzd, Lx, Sf, Vo, Vf,NN);
	}
	else if (precision == FIELD_SINGLE)
	{
		const float dzd  = dz*d;
		if (fType & FIELD_AXION)
			updateMKernel<<<gridSize,blockSize,0,stream>>> ((        float  *) m, (const         float  *) v, dzd, Lx, Sf, Vo, Vf,NN);
		else
			updateMKernel<<<gridSize,blockSize,0,stream>>> ((complex<float> *) m, (const complex<float> *) v, dzd, Lx, Sf, Vo, Vf, NN);
	}
}
