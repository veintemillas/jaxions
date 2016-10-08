#include"kernelParms.cuh"
#include"complexGpu.cuh"
#include"index.cuh"
#include<vector>

#include"scalarField.h"
#include"enum-field.h"
#include"RKParms.h"

#include"comms.h"

using namespace gpuCu;
using namespace indexHelper;

template<typename mFloat, typename cFloat>
__device__ void	updateMCoreGpu(const int idx, complex<mFloat> * __restrict__ m, const complex<mFloat> * __restrict__ v, const cFloat dz, const cFloat d, const int Sf)
{
	complex<cFloat> mel, tmp;

	mel = v[idx-Sf];
	tmp = m[idx];

	mel *= dz*d;
	tmp += mel;
	m[idx] = tmp;
//	m[idx] += v[idx]*dz*d;
}

template<typename mFloat, typename cFloat>
static __device__ __forceinline__ void	updateVCoreGpu(const int idx, const complex<mFloat> * __restrict__ m, complex<mFloat> * __restrict__ v, cFloat z, const cFloat dz,
						       const cFloat c, const cFloat delta2, const cFloat LL, const cFloat nQcd, const int Lx, const int Lz)
{
	const int &LX = Lx;
	const int &LZ = Lz;

	int idx2, X[3], Y[2];

	complex<cFloat> mel, a, tmp;

	idx2Vec(idx, X, LX);
	Y[0] = X[0]; Y[1] = X[1];

	X[0] = ((Y[0]+1)%LX);
	idx2 = vec2Idx(X, LX);
	tmp = m[idx2];
	mel = tmp;

	X[0] = ((Y[0]-1+LX)%LX);
	idx2 = vec2Idx(X, LX);
	tmp = m[idx2];
	mel += tmp;

	X[0] = Y[0];
	X[1] = ((Y[1]+1)%LX);
	idx2 = vec2Idx(X, LX);
	tmp = m[idx2];
	mel += tmp;

	X[1] = ((Y[1]-1+LX)%LX);
	idx2 = vec2Idx(X, LX);
	tmp = m[idx2];
	mel += tmp;

	X[1] = Y[1];
	X[2]++;
	idx2 = vec2Idx(X, LX);
	tmp = m[idx2];
	mel += tmp;

	X[2] -= 2;
	idx2 = vec2Idx(X, LX);
	tmp = m[idx2];
	mel += tmp;

	tmp = m[idx];
	a = (mel-((cFloat) 6.)*tmp)/(delta2) + ((cFloat) 9.)*pow(z,(cFloat) (nQcd+3)) - ((cFloat) LL)*tmp*(abs(tmp)*abs(tmp) - z*z);

	mel = v[idx-LZ];
	mel += a*dz*c;
	v[idx-LZ] = mel;
/*
	mel *= dz*d;
	tmp += mel;
	m2[idx] = tmp;
*/
/*	tmp = v[idx];
	tmp += a*dz*c;
	v[idx] = tmp;
	tmp *= dz*d;
	a = m[idx];
	a += tmp;
	m2[idx] = a;*/
}

template<typename mFloat, typename cFloat>
void	updateMCoreCpu(const int idx, std::complex<mFloat> * __restrict__ m, const std::complex<mFloat> * __restrict__ v, const cFloat dz, const cFloat d)
{
	m[idx] = (std::complex<mFloat>) (((std::complex<cFloat>) m[idx]) + ((std::complex<cFloat>) v[idx])*dz*d);
}

template<typename mFloat, typename cFloat>
void	updateVCoreCpu(const int idx, const std::complex<mFloat> * __restrict__ m, std::complex<mFloat> *v, cFloat z, const cFloat dz,
			 const cFloat c, const cFloat delta2, const cFloat LL, const cFloat nQcd, const int Lx, const int Lz)
{
	int idx2, X[3];

	std::complex<cFloat> mel, a;

	idx2Vec(idx, X, Lx);

	X[0] = ((X[0]+1)%Lx);
	idx2 = vec2Idx(X, Lx);
	mel = m[idx2];

	X[0] = ((X[0]-2+Lx)%Lx);
	idx2 = vec2Idx(X, Lx);
	mel += m[idx2];

	X[0] = ((X[0]+1)%Lx);
	X[1] = ((X[1]+1)%Lx);
	idx2 = vec2Idx(X, Lx);
	mel += m[idx2];

	X[1] = ((X[1]-2+Lx)%Lx);
	idx2 = vec2Idx(X, Lx);
	mel += m[idx2];

	X[1] = ((X[1]+1)%Lx);
	X[2] = ((X[2]+1)%Lz);
	idx2 = vec2Idx(X, Lx);
	mel += m[idx2];

	X[2] = ((X[2]-2+Lz)%Lz);
	idx2 = vec2Idx(X, Lx);
	mel += m[idx2];

	a = (mel-((cFloat) 6.)*((std::complex<cFloat>) m[idx]))/(delta2) + ((cFloat) 9.)*pow(z, (cFloat) (nQcd+3)) - ((cFloat) LL)*((std::complex<cFloat>) m[idx]) * (abs((std::complex<cFloat>) m[idx])*abs((std::complex<cFloat>) m[idx]) - z*z);

	v[idx] += a*dz*c;
//	m2[idx] = (std::complex<mFloat>) (((std::complex<cFloat>) m[idx]) + ((std::complex<cFloat>) v[idx])*dz*d);
}

template<typename mFloat, typename cFloat>
__global__ void	updateMKernel(complex<mFloat> * __restrict__ m, const complex<mFloat> * __restrict__ v, const cFloat dz, const cFloat d, const int Vo, const int Vf, const int Sf)
{
	//int idx = Vo + (threadIdx.x + blockIdx.x*blockDim.x);// + gridDim.x*blockDim.x*((threadIdx.y + blockIdx.y*blockDim.y) + gridDim.y*blockDim.y*(threadIdx.z + blockIdx.z*blockDim.z));
	int idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));// + gridDim.x*blockDim.x*((threadIdx.y + blockIdx.y*blockDim.y) + gridDim.y*blockDim.y*(threadIdx.z + blockIdx.z*blockDim.z));

	if	(idx >= Vf)
		return;

	updateMCoreGpu<mFloat,cFloat>(idx, m, v, dz, d, Sf);
}

template<typename mFloat, typename cFloat>
__global__ void	updateVKernel(const complex<mFloat> * __restrict__ m, complex<mFloat> * __restrict__ v, cFloat z, const cFloat dz, const cFloat c,
				const cFloat delta2, const cFloat LL, const cFloat nQcd, const int Lx, const int Lz, const int Vo, const int Vf)
{
	int idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));// + gridDim.x*blockDim.x*((threadIdx.y + blockIdx.y*blockDim.y) + gridDim.y*blockDim.y*(threadIdx.z + blockIdx.z*blockDim.z));

	if	(idx >= Vf)
		return;

	updateVCoreGpu<mFloat,cFloat>(idx, m, v, z, dz, c, delta2, LL, nQcd, Lx, Lz);
}

void	updateMGpu(void * __restrict__ m, const void * __restrict__ v, const double dz, const double d, const int Lx, const int Vo, const int Vf, FieldPrecision precision, cudaStream_t &stream)
{
	#define	BSSIZE 512
	const int Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3	gridSize((Lx*Lx+BSSIZE-1)/BSSIZE,Lz2,1);
	dim3	blockSize(BSSIZE,1,1);

	//dim3	gridSize((Lx+BSIZE-1)/BSIZE,(Lx+BSIZE-1)/BSIZE,(Lz+BSIZE-3)/BSIZE);

	//const int Lz2 = (Vf-Vo)/(Lx*Lx);
	//dim3	gridSize((Lx+BSIZE-1)/BSIZE,(Lx+BSIZE-1)/BSIZE,(Lz2+BSIZE-1)/BSIZE);
	//dim3	blockSize(BSIZE,BSIZE,BSIZE);

	if (precision == FIELD_DOUBLE)
	{
		updateMKernel<<<gridSize,blockSize,0,stream>>> ((complex<double> *) m, (const complex<double> *) v, (double) dz, (double) d, Vo, Vf, Lx*Lx);
	}
	else if (precision == FIELD_SINGLE)
	{
		updateMKernel<<<gridSize,blockSize,0,stream>>> ((complex<float> *) m, (const complex<float> *) v, (float) dz, (float) d, Vo, Vf, Lx*Lx);
	}
	else if (precision == FIELD_MIXED)
	{
		updateMKernel<<<gridSize,blockSize,0,stream>>> ((complex<float> *) m, (const complex<float> *) v, (double) dz, (double) d, Vo, Vf, Lx*Lx);
	}
}

void	updateVGpu(const void * __restrict__ m, void * __restrict__ v, double *z, const double dz, const double c,
		     const double delta2, const double LL, const double nQcd, const int Lx, const int Lz, const int Vo, const int Vf, FieldPrecision precision, cudaStream_t &stream)
{
	#define	BLSIZE 512
	const int Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3	gridSize((Lx*Lx+BSSIZE-1)/BSSIZE,Lz2,1);
	dim3	blockSize(BSSIZE,1,1);

	//dim3	gridSize((Lx+BSIZE-1)/BSIZE,(Lx+BSIZE-1)/BSIZE,(Lz+BSIZE-3)/BSIZE);

	//const int Lz2 = (Vf-Vo)/(Lx*Lx);
	//dim3	gridSize((Lx+BSIZE-1)/BSIZE,(Lx+BSIZE-1)/BSIZE,(Lz2+BSIZE-1)/BSIZE);
	//dim3	blockSize(BSIZE,BSIZE,BSIZE);

	if (precision == FIELD_DOUBLE)
	{
		double zR = *z;
		updateVKernel<<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, zR, (double) dz, (double) c, (double) delta2,
								 (double) LL, (double) nQcd, Lx, Lx*Lx, Vo, Vf);
	}
	else if (precision == FIELD_SINGLE)
	{
		float zR = *z;
		updateVKernel<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, zR, (float) dz, (float) c, (float) delta2,
								 (float) LL, (float) nQcd, Lx, Lx*Lx, Vo, Vf);
	}
	else if (precision == FIELD_MIXED)
	{
		double zR = *z;
		updateVKernel<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, zR, (double) dz, (double) c, (double) delta2,
								 (double) LL, (double) nQcd, Lx, Lx*Lx, Vo, Vf);
	}
}

void	updateMCpu(void * __restrict__ m, const void * __restrict__ v, const double dz, const double d, const int V, FieldPrecision precision)
{
	if (precision == FIELD_DOUBLE)
	{
		#pragma omp parallel for default(shared) private(dz,c,d,delta2) schedule(static)
		for (int idx = 0; idx < V; idx++)
			updateMCoreCpu(idx, (std::complex<double> *) m, (const std::complex<double> *) v, (double) dz, (double) d);
	}
	else if (precision == FIELD_SINGLE)
	{
		#pragma omp parallel for default(shared) private(dz,c,d,delta2) schedule(static)
		for (int idx = 0; idx < V; idx++)
			updateMCoreCpu(idx, (std::complex<float> *) m, (const std::complex<float> *) v, (float) dz, (float) d);
	}
	else if (precision == FIELD_MIXED)
	{
		#pragma omp parallel for default(shared) private(dz,c,d,delta2) schedule(static)
		for (int idx = 0; idx < V; idx++)
			updateMCoreCpu(idx, (std::complex<float> *) m, (const std::complex<float> *) v, (double) dz, (double) d);
	}
}

void	updateVCpu(const void * __restrict__ m, void *v, double *z, const double dz, const double c, const double d,
		     const double delta2, const double LL, const double nQcd, const int Lx, const int Lz, const int V, FieldPrecision precision)
{
	if (precision == FIELD_DOUBLE)
	{
		double zR = *z;
		#pragma omp parallel for default(shared) private(dz,c,d,delta2) schedule(static)
		for (int idx = 0; idx < V; idx++)
			updateVCoreCpu(idx, (const std::complex<double> *) m, (std::complex<double> *) v, zR, (double) dz, (double) c, (double) delta2, (double) LL, (double) nQcd, Lx, Lz);
	}
	else if (precision == FIELD_SINGLE)
	{
		float zR = *z;
		#pragma omp parallel for default(shared) private(dz,c,d,delta2) schedule(static)
		for (int idx = 0; idx < V; idx++)
			updateVCoreCpu(idx, (const std::complex<float> *) m, (std::complex<float> *) v, zR, (float) dz, (float) c, (float) delta2, (float) LL, (float) nQcd, Lx, Lz);
	}
	else if (precision == FIELD_MIXED)
	{
		double zR = *z;
		#pragma omp parallel for default(shared) private(dz,c,d,delta2) schedule(static)
		for (int idx = 0; idx < V; idx++)
			updateVCoreCpu(idx, (const std::complex<float> *) m, (std::complex<float> *) v, zR, (double) dz, (double) c, (double) delta2, (double) LL, (double) nQcd, Lx, Lz);
	}

	*z += dz*d;
}

class	PropLowMem
{
	private:

	const double c1, c2, c3, c4;	// The parameters of the Runge-Kutta-Nyström
	const double d1, d2, d3, d4;
	const double delta2, dz;
	const double nQcd, LL;
	const int Lx, Lz, V, S;

	FieldPrecision precision;

	Scalar	*axionField;

	void	propagateGpu(const double c, const double d);

	public:

		 PropLowMem(Scalar *field, const double LL, const double nQcd, const double delta, const double dz);
		~PropLowMem() {};

	void	runGpu	();
	void	runCpu	();
};

	PropLowMem::PropLowMem(Scalar *field, const double LL, const double nQcd, const double delta, const double dz) : axionField(field), dz(dz), Lx(field->Length()), Lz(field->eDepth()), V(field->Size()),
									S(field->Surf()), c1(C1), d1(D1), c2(C2), d2(D2), c3(C3), d3(D3), c4(C4), d4(D4), delta2(delta*delta), precision(field->Precision()), LL(LL), nQcd(nQcd)
{
}

void	PropLowMem::propagateGpu(const double c, const double d)
{
	const int ext = V + S;
	double *z = axionField->zV();

	updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c, delta2, LL, nQcd, Lx, Lz, 2*S, V, precision, ((cudaStream_t *)axionField->Streams())[2]);
	updateMGpu(axionField->mGpu(), axionField->vGpu(), dz, d, Lx, 3*S, V-S, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
	updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c, delta2, LL, nQcd, Lx, Lz, S, 2*S, precision, ((cudaStream_t *)axionField->Streams())[0]);
	updateVGpu(axionField->mGpu(), axionField->vGpu(), z, dz, c, delta2, LL, nQcd, Lx, Lz, V, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);
	cudaStreamSynchronize(((cudaStream_t *)axionField->Streams())[2]);
//	cudaStreamSynchronize(((cudaStream_t *)axionField->Streams())[1]);
	updateMGpu(axionField->mGpu(), axionField->vGpu(), dz, d, Lx, S, 3*S, precision, ((cudaStream_t *)axionField->Streams())[0]);
	updateMGpu(axionField->mGpu(), axionField->vGpu(), dz, d, Lx, V-S, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);

	*z += dz*d;

	cudaDeviceSynchronize();
}

void	PropLowMem::runGpu	()
{
	propagateGpu(c1, d1);
	propagateGpu(c2, d2);
	propagateGpu(c3, d3);
	propagateGpu(c4, d4);
//	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
}

void	PropLowMem::runCpu	()
{
	axionField->sendGhosts(FIELD_M);
        updateVCpu(axionField->mCpu(), axionField->vCpu(), axionField->zV(), dz, c1, d1, delta2, LL, nQcd, Lx, Lz, V, precision);
        updateMCpu(axionField->mCpu(), axionField->vCpu(), dz, d1, V, precision);
	axionField->sendGhosts(FIELD_M);
        updateVCpu(axionField->mCpu(), axionField->vCpu(), axionField->zV(), dz, c2, d2, delta2, LL, nQcd, Lx, Lz, V, precision);
        updateMCpu(axionField->mCpu(), axionField->vCpu(), dz, d2, V, precision);
	axionField->sendGhosts(FIELD_M);
        updateVCpu(axionField->mCpu(), axionField->vCpu(), axionField->zV(), dz, c3, d3, delta2, LL, nQcd, Lx, Lz, V, precision);
        updateMCpu(axionField->mCpu(), axionField->vCpu(), dz, d3, V, precision);
	axionField->sendGhosts(FIELD_M);
        updateVCpu(axionField->mCpu(), axionField->vCpu(), axionField->zV(), dz, c4, d4, delta2, LL, nQcd, Lx, Lz, V, precision);
        updateMCpu(axionField->mCpu(), axionField->vCpu(), dz, d4, V, precision);
}

void	propLowMem	(Scalar *field, const double dz, const double LL, const double nQcd, const double delta, bool gpu)
{
	PropLowMem *prop = new PropLowMem(field, LL, nQcd, delta, dz);

	if (gpu)
		prop->runGpu ();
	else
		prop->runCpu ();

	delete	prop;
}

