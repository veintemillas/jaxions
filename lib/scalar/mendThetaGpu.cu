#include "kernelParms.cuh"
#include "complexGpu.cuh"
#include "utils/index.cuh"

#include "enum-field.h"
#include "scalar/scalarField.h"
#include "utils/utils.h"

using namespace gpuCu;
using namespace indexHelper;

//#include <cstdio>
//#include <cmath>
//#include "scalar/varNQCD.h"
//#include "utils/parse.h"
#include "utils/reduceGpu.cuh"

template <typename Float>
inline __device__ constexpr Float sgn(Float val) {
    return (Float) ((Float(0) < val) - (val < Float(0)));
}

/*	Connects all the points in a ZY plane with the whole volume	*/
/*	Parallelizes on ZY and vectorizes on Y				*/

template<typename Float>
static __device__ __forceinline__ uint mendBulkCore(const uint sIdx, Float * __restrict__ m, Float * __restrict__ v, const Float zP, const uint Lx, const uint Sf) {
	uint count = 0;

	for (uint ix = 0; ix<Lx-1; ix++) {

		uint idxPx = sIdx  + ix;
		uint idxVx = idxPx - Sf;

		Float mel, mPx, vPx, mDf;

		mel = m[sIdx];
		mPx = m[idxPx];
		vPx = v[idxVx];

		mDf = mPx - mel;

		while	(abs(mDf) > zP) {
			const Float sign = sgn(mDf);
			mPx -= sign*zP*2.;
			vPx -= sign*M_PI*2.;
			count++;
		}
			
		m[idxPx] = mPx;
		v[idxVx] = vPx;
	}

	return	count;
}

/*	Connects all the points in a Z line with the whole ZY plane	*/
/*	Parallelizes on Z						*/

template<typename Float>
static __device__ __forceinline__ uint mendSliceCore(const uint zIdx, Float * __restrict__ m, Float * __restrict__ v, const Float zP, const size_t Lx, const size_t Lz, const size_t Sf) {
	uint count = 0;

	for (int iy=0; iy<Lx-1; iy++) {
		uint idx   = zIdx*Sf + iy*Lx;
		uint idxPy = idx + Lx;
		uint idxVy = idx - Sf;

		Float mDf, mel, mPy, vPy;

		mel = m[idx];
		mPy = m[idxPy];
		vPy = v[idxVy];

		mDf = mPy - mel;

		while	(abs(mDf) > zP) {
			const Float sign = sgn(mDf);
			mPy -= sign*zP*2.;
			vPy -= sign*M_PI*2.;
			count++;
		}
			
		m[idxPy] = mPy;
		v[idxVy] = vPy;
	}

	/*	Border, needs shifts	*/

	uint idxVy = zIdx*Sf - Lx;
	uint idxPy = idxVy   + Sf;
	uint idx   = idxPy + Sf - Lx;

	Float mDf, mel, mPy, vPy;

	mel = m[idx];
	mPy = m[idxPy];
	vPy = v[idxVy];

	mDf = mPy - mel;

	while	(abs(mDf) > zP) {
		const Float sign = sgn(mDf);
		mPy -= sign*zP*2.;
		vPy -= sign*M_PI*2.;
		count++;
	}
		
	m[idxPy] = mPy;
	v[idxVy] = vPy;

	return	count;
}

template<typename Float>
__global__ void	mendLineCore(Float * __restrict__ m, Float * __restrict__ v, const Float zP, const uint Lz, const uint Sf, uint *count) {
	*count = 0;

	Float mDf, mel, mPz, vPz;
	size_t idxPz, idxVz;

	for (size_t idx=0,i=0; i<Lz; i++,idx+=Sf) {

		idxPz = idx + Sf;
		idxVz = idx;

		mel = m[idx];

		mPz = m[idxPz];
		vPz = v[idxVz];

		/*	Z-Direction	*/

		mDf = mPz - mel;

		while	(abs(mDf) > zP) {
			const Float sign = sgn(mDf);
			mPz -= sign*zP*2.;
			vPz -= sign*M_PI*2.;
			*count++;
		}
			
		m[idxPz] = mPz;
		v[idxVz] = vPz;
	}

	return;
}

template<typename Float>
__global__ void	mendBulkGpu	(Float * __restrict__ m, Float * __restrict__ v, const Float zP, const uint Lx, const uint Sf, uint *count, uint *partial) {
	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint tmp[1] = { 0 };

	if	(idx < Sf)
		*tmp = mendBulkCore<Float>(idx, m, v, zP, Lx, Sf);

	reduction<BSIZE,uint,1>   (count, tmp, partial);
}

template<typename Float>
__global__ void	mendSliceGpu	(Float * __restrict__ m, Float * __restrict__ v, const Float zP, const uint Lx, const uint Lz, const uint Sf, uint *count, uint *partial) {

	uint idx = threadIdx.x + blockDim.x*blockIdx.x;
	uint tmp[1] = { 0 };

	if	(idx < Lz)
		*tmp = mendSliceCore<Float>(idx, m, v, zP, Lx, Lz, Sf);

	reduction<BSIZE,uint,1>   (count, tmp, partial);
}

template<typename Float>
uint	mendLineGpu	(Float * __restrict__ m, Float * __restrict__ v, const Float zP, const uint Lz, const uint Sf, cudaStream_t &stream) {
	/*	For MPI		*/
	const int nSplit  = commSize();
	const int rank    = commRank();
	const int fwdNeig = (rank + 1) % nSplit;
	const int bckNeig = (rank - 1 + nSplit) % nSplit;

	uint count = 0;
	uint *cnt_d;

printf("Alloc\n"); fflush(stdout);
	if (cudaMalloc(&cnt_d, sizeof(uint)) != cudaSuccess)
		return  0;
		
	for (int cRank = 0; cRank < commSize(); cRank++) {

printf("Loop %d / %d\n", cRank, commSize()); fflush(stdout);
		commSync();

		const int cFwdNeig = (cRank + 1) % nSplit;
		const int cBckNeig = (cRank - 1 + nSplit) % nSplit;

		/*	Get the ghosts for slice 0							*/
		/*	It's cumbersome but we avoid exchanging the whole slice to get one point	*/

		// Copia GPU --> CPU

		if (commSize() == 1) {
			uint cnt = 0;
			cudaMemcpy(m, &m[Sf*Lz], sizeof(Float), cudaMemcpyDeviceToDevice);
			mendLineCore<<<1,1,0,stream>>>(m, v, zP, Lz, Sf, cnt_d);
			auto cErr = cudaDeviceSynchronize();

			if (cErr != cudaSuccess) {
				LogError("Error: %s\n", cudaGetErrorString(cErr));
printf("AAAAHH\n\n\n\n"); fflush(stdout);
				return	0;
			}

			cudaMemcpy(&cnt, cnt_d, sizeof(uint), cudaMemcpyDeviceToHost);
			count += cnt;
		} else {
			if (rank == cBckNeig) {
				Float tmp;
				cudaMemcpy(&tmp, &m[Sf*Lz], sizeof(Float), cudaMemcpyDeviceToHost);
				MPI_Send(&tmp, sizeof(Float), MPI_CHAR, cRank,   cRank, MPI_COMM_WORLD);
			}

			if (rank == cRank) {
				uint cnt = 0;
				Float tmp;
				MPI_Recv(&tmp, sizeof(Float), MPI_CHAR, bckNeig, cRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				cudaMemcpy(m, &tmp, sizeof(Float), cudaMemcpyHostToDevice);
				cudaMemset(cnt_d, 0, sizeof(uint));
				cudaDeviceSynchronize();
				mendLineCore<<<1,1,0,stream>>>(m, v, zP, Lz, Sf, cnt_d);
				cudaDeviceSynchronize();
				cudaMemcpy(&cnt, cnt_d, sizeof(uint), cudaMemcpyDeviceToHost);
				count += cnt;
			}
		}
	}

	cudaFree(cnt_d);

	return	count;
}

template<typename Float>
uint	mendThetaGpu	(Float * __restrict__ m, Float * __restrict__ v, const Float z, const uint Lx, const uint Lz, const uint Sf, cudaStream_t &stream) {

	dim3  gridSf((Sf+BSIZE-1)/BSIZE,1,1);
	dim3  gridLn((Lz+BSIZE-1)/BSIZE,1,1);
	dim3  blockSize(BSIZE,1,1);

	const int nBlocksLn = gridLn.x;
	const int nBlocksSf = gridSf.x;

	uint tJmp = 0;

	uint *partial, maxBlock = (nBlocksLn > nBlocksSf) ? nBlocksLn : nBlocksSf;

	if (cudaMalloc(&partial, sizeof(uint)*maxBlock*8) != cudaSuccess)
		return  0;

	const Float zP = z*M_PI;

	uint pTmp = 0;
printf("Line\n"); fflush(stdout);
	tJmp += mendLineGpu(m, v, zP, Lz, Sf, stream);
printf("Slice\n"); fflush(stdout);
	mendSliceGpu<Float><<<gridLn, blockSize, 0, stream>>>(m, v, zP, Lx, Lz, Sf, &pTmp, partial);
	cudaDeviceSynchronize();
	tJmp += pTmp;
	cudaMemset(partial, 0, sizeof(uint)*maxBlock*8);
	pTmp = 0;
printf("Bulk\n"); fflush(stdout);
	mendBulkGpu <Float><<<gridSf, blockSize, 0, stream>>>(m, v, zP, Lx, Sf, &pTmp, partial);
	cudaDeviceSynchronize();
printf("Done\n"); fflush(stdout);
	tJmp += pTmp;

	cudaFree(partial);

	return	tJmp;
}

uint	mendThetaGpu(Scalar *field) {
	double z = *field->zV();
	uint   tJmp = 0;

	switch(field->Precision()) {
		case	FIELD_SINGLE:
			tJmp = mendThetaGpu(static_cast<float *>(field->mGpu()), static_cast<float *>(field->vGpu()), (float) z, field->Length(), field->Depth(),
								field->Surf(), ((cudaStream_t *) field->Streams())[0]);
			break;

		case	FIELD_DOUBLE:
			tJmp = mendThetaGpu(static_cast<double*>(field->mGpu()), static_cast<double*>(field->vGpu()),         z, field->Length(), field->Depth(),
								field->Surf(), ((cudaStream_t *) field->Streams())[0]);
			break;
	}

	LogOut("mendTheta done mends = %lu\n", tJmp);

	return	tJmp;
}
