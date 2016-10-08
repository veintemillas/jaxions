#include"kernelParms.cuh"
#include"complexGpu.cuh"
#include"index.cuh"
#include<vector>

#include"scalarField.h"
#include"enum-field.h"
#include"RKParms.h"

#include"comms.h"

#ifdef	PROFILE
#include<cuda_profiler_api.h>
#endif

using namespace gpuCu;
using namespace indexHelper;

template<typename mFloat, typename cFloat>
static __device__ __forceinline__ void	propagateCoreGpu(const int idx, const complex<mFloat> * __restrict__ m, complex<mFloat> * __restrict__ v, complex<mFloat> * __restrict__ m2,
							 cFloat z, const cFloat dz, const cFloat c, const cFloat d, const cFloat delta2, const cFloat LL, const cFloat nQcd,
							 const int Lx, const int Lz)
{
	const int &LX = Lx;
	const int &LZ = Lz;

	int X[3], Y[2], idx2;

	complex<cFloat> mel, a, tmp;

	idx2Vec(idx, X, LX);

	Y[0] = X[0];
	Y[1] = X[1];

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
//	a = (mel-((cFloat) 6.)*m[idx])/(delta2) + ((cFloat) 9.)*pow(z,(cFloat) (nQcd+3)) - ((cFloat) LL)*m[idx]*(abs(m[idx])*abs(m[idx]) - z*z);


	mel = v[idx-LZ];
	mel += a*dz*c;
	v[idx-LZ] = mel;
	mel *= dz*d;
	tmp += mel;
	m2[idx] = tmp;

/*	tmp = v[idx];
	tmp += a*dz*c;
	v[idx] = tmp;
	tmp *= dz*d;
	a = m[idx];
	a += tmp;
	m2[idx] = a;*/
}

template<typename mFloat, typename cFloat>
void	propagateCoreCpu(const int idx, const std::complex<mFloat> * __restrict__ m, std::complex<mFloat> * __restrict__ m2, std::complex<mFloat> *v, cFloat z, const cFloat dz,
			 const cFloat c, const cFloat d, const cFloat delta2, const cFloat LL, const cFloat nQcd, const int Lx, const int Lz)
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
	m2[idx] = (std::complex<mFloat>) (((std::complex<cFloat>) m[idx]) + ((std::complex<cFloat>) v[idx])*dz*d);
}

template<typename mFloat, typename cFloat>
__global__ void	propagateKernel(const complex<mFloat> * __restrict__ m, complex<mFloat> * __restrict__ v, complex<mFloat> * __restrict__ m2, cFloat z, const cFloat dz, const cFloat c, const cFloat d,
				const cFloat delta2, const cFloat LL, const cFloat nQcd, const int Lx, const int Lz, const int Vo, const int Vf)
{
	int idx = Vo + (threadIdx.x + blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y));//((threadIdx.y + blockIdx.y*blockDim.y));// + gridDim.y*blockDim.y*(threadIdx.z + blockIdx.z*blockDim.z));

	if	(idx >= Vf)
		return;

	propagateCoreGpu<mFloat,cFloat>(idx, m, v, m2, z, dz, c, d, delta2, LL, nQcd, Lx, Lz);
/*
	if (blockIdx.x == 0)
		printf("%d V(%d %d) T %d B(%d %d)\n", idx, Vo, Vf, threadIdx.x, blockIdx.x, blockDim.x);

	if (blockIdx.x == gridDim.x-2)
		printf("%d V(%d %d) T %d B(%d %d)\n", idx, Vo, Vf, threadIdx.x, blockIdx.x, blockDim.x);
*/
}

void	propagateGpu(const void * __restrict__ m, void * __restrict__ v, void * __restrict__ m2, double *z, const double dz, const double c, const double d,
		     const double delta2, const double LL, const double nQcd, const int Lx, const int Lz, const int Vo, const int Vf, FieldPrecision precision, cudaStream_t &stream)
{
	#define	BLSIZE 512
	const int Lz2 = (Vf-Vo)/(Lx*Lx);
	dim3	gridSize((Lx*Lx+BLSIZE-1)/BLSIZE,Lz2,1);
	dim3	blockSize(BLSIZE,1,1);
	//dim3	gridSize((Lx+BSIZE-1)/BSIZE,(Lx+BSIZE-1)/BSIZE,(Lz+BSIZE-3)/BSIZE);
//	dim3	gridSize((Lx+BSIZE-1)/BSIZE,(Lx+BSIZE-1)/BSIZE,(Lz2+BSIZE-1)/BSIZE);
	//dim3	blockSize(BSIZE,BSIZE,BSIZE);

	if (precision == FIELD_DOUBLE)
	{
		double zR = *z;
		propagateKernel<<<gridSize,blockSize,0,stream>>> ((const complex<double> *) m, (complex<double> *) v, (complex<double> *) m2, zR, (double) dz, (double) c, (double) d,
								(double) delta2, (double) LL, (double) nQcd, Lx, Lx*Lx, Vo, Vf);
	}
	else if (precision == FIELD_SINGLE)
	{
		float zR = *z;
		propagateKernel<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2, zR, (float) dz, (float) c, (float) d,
								 (float) delta2, (float) LL, (float) nQcd, Lx, Lx*Lx, Vo, Vf);
	}
	else if (precision == FIELD_MIXED)
	{
//		double zR = *z;
//		propagateKernel<<<gridSize,blockSize,0,stream>>> ((const complex<float> *) m, (complex<float> *) v, (complex<float> *) m2, zR, (double) dz, (double) c, (double) d,
//								 (double) delta2, (double) LL, (double) nQcd, Lx, Lx*Lx, Vo, Vf);
	}
}

void	propagateCpu(const void * __restrict__ m, void *v, void * __restrict__ m2, double *z, const double dz, const double c, const double d,
		     const double delta2, const double LL, const double nQcd, const int Lx, const int Lz, const int V, FieldPrecision precision)
{
	if (precision == FIELD_DOUBLE)
	{
		double zR = *z;
		#pragma omp parallel for default(shared) private(dz,c,d,delta2) schedule(static)
		for (int idx = 0; idx < V; idx++)
			propagateCoreCpu(idx, (const std::complex<double> *) m, (std::complex<double> *) m2, (std::complex<double> *) v, zR, (double) dz, (double) c, (double) d, (double) delta2, (double) LL, (double) nQcd, Lx, Lz);
	}
	else if (precision == FIELD_SINGLE)
	{
		float zR = *z;
		#pragma omp parallel for default(shared) private(dz,c,d,delta2) schedule(static)
		for (int idx = 0; idx < V; idx++)
			propagateCoreCpu(idx, (const std::complex<float> *) m, (std::complex<float> *) m2, (std::complex<float> *) v, zR, (float) dz, (float) c, (float) d, (float) delta2, (float) LL, (float) nQcd, Lx, Lz);
	}
	else if (precision == FIELD_MIXED)
	{
		double zR = *z;
		#pragma omp parallel for default(shared) private(dz,c,d,delta2) schedule(static)
		for (int idx = 0; idx < V; idx++)
			propagateCoreCpu(idx, (const std::complex<float> *) m, (std::complex<float> *) m2, (std::complex<float> *) v, zR, (double) dz, (double) c, (double) d, (double) delta2, (double) LL, (double) nQcd, Lx, Lz);
	}

	*z += dz*d;
}

class	Propagator
{
	private:

	const double c1, c2, c3, c4;	// The parameters of the Runge-Kutta-Nyström
	const double d1, d2, d3, d4;
	const double delta2, dz;
	const double nQcd, LL;
	const int Lx, Lz, V, S;

	FieldPrecision precision;

	Scalar	*axionField;

	public:

		 Propagator(Scalar *field, const double LL, const double nQcd, const double delta, const double dz);
		~Propagator() {};

	void	runGpu	();
	void	runCpu	();
};

	Propagator::Propagator(Scalar *field, const double LL, const double nQcd, const double delta, const double dz) : axionField(field), dz(dz), Lx(field->Length()), Lz(field->eDepth()), V(field->Size()),
									S(field->Surf()), c1(C1), d1(D1), c2(C2), d2(D2), c3(C3), d3(D3), c4(C4), d4(D4), delta2(delta*delta), precision(field->Precision()), LL(LL), nQcd(nQcd)
{
}

void	Propagator::runGpu	()
{
	const int ext = V + S;
	double *z = axionField->zV();

        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, LL, nQcd, Lx, Lz, 2*S, V, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, LL, nQcd, Lx, Lz, S, 2*S, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c1, d1, delta2, LL, nQcd, Lx, Lz, V, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);
	*z += dz*d1;

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot

        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, LL, nQcd, Lx, Lz, 2*S, V, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M2);
        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, LL, nQcd, Lx, Lz, S, 2*S, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c2, d2, delta2, LL, nQcd, Lx, Lz, V, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);
	*z += dz*d2;

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot

        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, LL, nQcd, Lx, Lz, 2*S, V, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M);
        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, LL, nQcd, Lx, Lz, S, 2*S, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propagateGpu(axionField->mGpu(), axionField->vGpu(), axionField->m2Gpu(), z, dz, c3, d3, delta2, LL, nQcd, Lx, Lz, V, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);
	*z += dz*d3;

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot

        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, LL, nQcd, Lx, Lz, 2*S, V, precision, ((cudaStream_t *)axionField->Streams())[2]);
	axionField->exchangeGhosts(FIELD_M2);
        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, LL, nQcd, Lx, Lz, S, 2*S, precision, ((cudaStream_t *)axionField->Streams())[0]);
        propagateGpu(axionField->m2Gpu(), axionField->vGpu(), axionField->mGpu(), z, dz, c4, d4, delta2, LL, nQcd, Lx, Lz, V, ext, precision, ((cudaStream_t *)axionField->Streams())[1]);
	*z += dz*d4;

	cudaDeviceSynchronize();	// This is not strictly necessary, but simplifies things a lot
}

void	Propagator::runCpu	()
{
	axionField->sendGhosts(FIELD_M);
        propagateCpu(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), axionField->zV(), dz, c1, d1, delta2, LL, nQcd, Lx, Lz, V, precision);
	axionField->sendGhosts(FIELD_M2);
        propagateCpu(axionField->m2Cpu(),axionField->vCpu(), axionField->mCpu(),  axionField->zV(), dz, c2, d2, delta2, LL, nQcd, Lx, Lz, V, precision);
	axionField->sendGhosts(FIELD_M);
        propagateCpu(axionField->mCpu(), axionField->vCpu(), axionField->m2Cpu(), axionField->zV(), dz, c3, d3, delta2, LL, nQcd, Lx, Lz, V, precision);
	axionField->sendGhosts(FIELD_M2);
        propagateCpu(axionField->m2Cpu(),axionField->vCpu(), axionField->mCpu(),  axionField->zV(), dz, c4, d4, delta2, LL, nQcd, Lx, Lz, V, precision);
}

void	propagate	(Scalar *field, const double dz, const double LL, const double nQcd, const double delta, bool gpu)
{
	#ifdef PROFILE
	cudaProfilerStart();
	#endif

	Propagator *prop = new Propagator(field, LL, nQcd, delta, dz);

	if (gpu)
		prop->runGpu ();
	else
		prop->runCpu ();

	delete	prop;

	#ifdef PROFILE
	cudaProfilerStop();
	#endif
}

