#include<cstdlib>
#include<cstring>
#include<complex>
#include<random>

#include"square.h"
#include"fftCuda.h"
#include"fftCode.h"
#include"enum-field.h"

#include"comms.h"

#ifdef	USE_GPU
	#include<cuda.h>
	#include<cuda_runtime.h>
	#include "cudaErrors.h"
#endif

#include<mpi.h>
#include<omp.h>

#ifdef	USE_XEON
	#include"xeonDefs.h"
#endif

#include "index.h"

using namespace std;

const std::complex<double> I(0.,1.);
const std::complex<float> If(0.,1.);

class	Scalar
{
	private:

	const int n1;
	const int n2;
	const int n3;

	const int Lz;
	const int Tz;
	const int Ez;
	const int v3;

	const int nSplit;

	const bool lowmem;

	DeviceType	device;
	FieldPrecision	precision;

	int	fSize;
	int	mAlign;

	double	*z;

	void	*m,   *v,   *m2;			// Cpu data
#ifdef	USE_GPU
	void	*m_d, *v_d, *m2_d;			// Gpu data

	void	*sStreams;
#endif
	void	recallGhosts(FieldIndex fIdx);		// Move the fileds that will become ghosts from the Cpu to the Gpu
	void	transferGhosts(FieldIndex fIdx);	// Copy back the ghosts to the Gpu

	void	scaleField(FieldIndex fIdx, double factor);
	void	randConf();
	void	smoothConf(const int iter, const double alpha);

	template<typename Float>
	void	iteraField(const int iter, const Float alpha);


	template<typename Float>
	void	momConf(const int kMax, const Float kCrit);

	public:

			 Scalar(const int nLx, const int nLz, FieldPrecision prec, DeviceType dev, const double zI, char fileName[], bool lowmem, const int nSp,
				ConfType cType, const int parm1, const double parm2);
			~Scalar();

	void		*mCpu() { return m; }
	const void	*mCpu() const { return m; }
	void		*vCpu() { return v; }
	const void	*vCpu() const { return v; }
	void		*m2Cpu() { return m2; }
	const void	*m2Cpu() const { return m2; }

#ifdef	USE_XEON
	__attribute__((target(mic))) void	*mXeon() { return mX; }
	__attribute__((target(mic))) const void	*mXeon() const { return mX; }
	__attribute__((target(mic))) void	*vXeon() { return vX; }
	__attribute__((target(mic))) const void	*vXeon() const { return vX; }
	__attribute__((target(mic))) void	*m2Xeon() { return m2X; }
	__attribute__((target(mic))) const void	*m2Xeon() const { return m2X; }
#endif

#ifdef	USE_GPU
	void		*mGpu() { return m_d; }
	const void	*mGpu() const { return m_d; }
	void		*vGpu() { return v_d; }
	const void	*vGpu() const { return v_d; }
	void		*m2Gpu() { return m2_d; }
	const void	*m2Gpu() const { return m2_d; }
#endif
	bool		LowMem()  { return lowmem; }

	int		Size()   { return n3; }
	int		Surf()   { return n2; }
	int		Length() { return n1; }
	int		Depth()  { return Lz; }
	int		eDepth() { return Ez; }
	int		eSize()  { return v3; }

	FieldPrecision	Precision() { return precision; }

	int		dataSize() { return fSize; }

	double		*zV() { return z; }
	const double	*zV() const { return z; }

	void		setZ(const double newZ) { *z = newZ; }

	void	foldField	();
	void	unfoldField	();
	void	unfoldField2D	(const int sZ);		// Just for the maps

	void	transferDev(FieldIndex fIdx);		// Move data to device (Gpu or Xeon)
	void	transferCpu(FieldIndex fIdx);		// Move data to Cpu

	void	sendGhosts(FieldIndex fIdx, CommOperation commOp);	// Send the ghosts in the Cpu using MPI, use this to exchange ghosts with Cpus
	void	exchangeGhosts(FieldIndex fIdx);	// Transfer ghosts from neighbouring ranks, use this to exchange ghosts with Gpus

	void	fftCpu(int sign);			// Fast Fourier Transform in the Cpu
	void	fftGpu(int sign);			// Fast Fourier Transform in the Gpu

	void	prepareCpu(int *window);		// Sets the field for a FFT, prior to analysis

	void	squareGpu();				// Squares the m2 field in the Gpu
	void	squareCpu();				// Squares the m2 field in the Cpu

	void	genConf	(ConfType cType, const int parm1, const double parm2);
#ifdef	USE_GPU
	void	*Streams() { return sStreams; }
#endif
};

	Scalar::Scalar(const int nLx, const int nLz, FieldPrecision prec, DeviceType dev, const double zI, char fileName[], bool lowmem, const int nSp, ConfType cType, const int parm1,
		       const double parm2) : nSplit(nSp), n1(nLx), n2(nLx*nLx), n3(nLx*nLx*nLz), Lz(nLz), Ez(nLz + 2), Tz(nSp*Lz), v3(nLx*nLx*(nLz + 2)), precision(prec), device(dev),
		       lowmem(lowmem)
{
	switch (prec)
	{
		case FIELD_DOUBLE:

		fSize = sizeof(double);
		break;

		case FIELD_SINGLE:

		fSize = sizeof(float);
		break;

		default:

		printf("Unrecognized precision\n");
		exit(1);
		break;
	}

	switch	(dev)
	{
		case DEV_XEON:
			printf("Using Xeon Phi 64 bytes alignment\n");
			mAlign = 64;
			break;

		case DEV_CPU:
			#if	defined(__AVX__) || defined(__AVX2__)
			printf("Using AVX 32 bytes alignment\n");
			mAlign = 32;
			#else
			printf("Using SSE 16 bytes alignment\n");
			mAlign = 16;
			#endif
			break;

		case DEV_GPU:
			mAlign = 16;
			break;
	}

	if (n2*fSize*2 % mAlign)
	{
		printf("Error: misaligned memory. Are you using an odd dimension?\n");
		exit(1);
	}

	const size_t	mBytes = v3*fSize*2;
	const size_t	vBytes = n3*fSize*2;

#ifdef	USE_XEON
	posix_memalign ((void**) &mX, mAlign, mBytes);
	posix_memalign ((void**) &vX, mAlign, vBytes);

	if (!lowmem)
		posix_memalign ((void**) &m2X, mAlign, mBytes);

	m = mX;
	v = vX;

	if (!lowmem)
		m2 = m2X;
#else
	posix_memalign ((void**) &m, mAlign, mBytes);
	posix_memalign ((void**) &v, mAlign, vBytes);

	if (!lowmem)
		posix_memalign ((void**) &m2, mAlign, mBytes);
#endif

	if (m == NULL)
	{
		printf("\n\nError: Couldn't allocate %lu bytes on host for the m field\n", mBytes);
		exit(1);
	}

	if (v == NULL)
	{
		printf("\n\nError: Couldn't allocate %lu bytes on host for the v field\n", vBytes);
		exit(1);
	}

	if (!lowmem)
	{
		if (m2 == NULL)
		{
			printf("\n\nError: Couldn't allocate %lu bytes on host for the m2 field\n", mBytes);
			exit(1);
		}
	}

	memset (m, 0, 2*fSize*v3);
	memset (v, 0, 2*fSize*n3);

	if (!lowmem)
		memset (m, 0, 2*fSize*v3);

	posix_memalign ((void **) &z, mAlign, sizeof(double));

	if (z == NULL)
	{
		printf("\n\nError: Couldn't allocate %d bytes on host for the z field\n", sizeof(double));
		exit(1);
	}

	if (device == DEV_GPU)
	{
#ifndef	USE_GPU
		printf ("Gpu support not built\n");
		exit   (1);
#else
		if (cudaMalloc(&m_d,  mBytes) != cudaSuccess)
		{
			printf("\n\nError: Couldn't allocate %lu bytes for the gpu field m\n", mBytes);
			exit(1);
		}

		if (cudaMalloc(&v_d,  vBytes) != cudaSuccess)
		{
			printf("\n\nError: Couldn't allocate %lu bytes for the gpu field v\n", vBytes);
			exit(1);
		}

		if (!lowmem)
			if (cudaMalloc(&m2_d, mBytes) != cudaSuccess)
			{
				printf("\n\nError: Couldn't allocate %lu bytes for the gpu field m2\n", mBytes);
				exit(1);
			}

		if ((sStreams = malloc(sizeof(cudaStream_t)*3)) == NULL)
		{
			printf("\n\nError: Couldn't allocate %lu bytes on host for the gpu streams\n", sizeof(cudaStream_t)*3);
			exit(1);
		}

		cudaStreamCreate(&((cudaStream_t *)sStreams)[0]);
		cudaStreamCreate(&((cudaStream_t *)sStreams)[1]);
		cudaStreamCreate(&((cudaStream_t *)sStreams)[2]);

//		if (!lowmem)
//			initCudaFFT(n1, Lz, prec);
#endif
	} else {
		initFFT(static_cast<void *>((static_cast<char *> (m) + 2*n2*fSize)), m2, n1, Tz, precision, lowmem);
	}

	*z = zI;

	/*	If present, read fileName	*/

	if (cType == CONF_NONE)
	{
		if (fileName == NULL)
		{
			printf("Sorry! Could not find initial Conditions\n");
			exit(1);
		}

		FILE *fileM = fopen(fileName,"r");

		if (fileM == 0)
		{
			printf("Sorry! Could not find initial Conditions\n");
			exit(1);
		}

		fread(static_cast<char *> (m) + 2*fSize*n2, fSize*2, n3, fileM);
		fclose(fileM);

		memcpy (v, static_cast<char *> (m) + 2*fSize*n2, 2*fSize*n3);
		scaleField (FIELD_M, zI);
	} else {
		genConf(cType, parm1, parm2);
	}

	if (dev == DEV_XEON)
	{
#ifndef	USE_XEON
		printf ("Xeon Phi support not built\n");
		exit   (1);
#else
		const int micIdx = commAcc();

		#pragma offload_transfer target(mic:micIdx) nocopy(mX : length(2*fSize*v3) AllocX)
		#pragma offload_transfer target(mic:micIdx) nocopy(vX : length(2*fSize*n3) AllocX)

		if (!lowmem)
		{
			#pragma offload_transfer target(mic:micIdx) nocopy(m2X : length(2*fSize*v3) AllocX)
		}
#endif
	}
}

	Scalar::~Scalar()
{
	if (m != NULL)
		free(m);

	if (v != NULL)
		free(v);

	if (m2 != NULL)
		free(m2);

	if (z != NULL)
		free(z);

	if (device == DEV_GPU)
	{
		#ifndef	USE_GPU
			printf ("Gpu support not built\n");
			exit   (1);
		#else
			if (m_d != NULL)
				cudaFree(m_d);

			if (v_d != NULL)
				cudaFree(v_d);

			if (m2_d != NULL)
				cudaFree(m2_d);

			cudaStreamDestroy(((cudaStream_t *)sStreams)[2]);
			cudaStreamDestroy(((cudaStream_t *)sStreams)[1]);
			cudaStreamDestroy(((cudaStream_t *)sStreams)[0]);

			if (sStreams != NULL)
				free(sStreams);

//			if (!lowmem)
//				closeCudaFFT();
		#endif
	} else {
//		if (!lowmem)
			closeFFT();
	}

	if (device == DEV_XEON)
	{
		#ifndef	USE_XEON
			printf ("Xeon Phi support not built\n");
			exit   (1);
		#else
			const int micIdx = commAcc();

			#pragma offload_transfer target(mic:micIdx) nocopy(mX : length(2*fSize*v3) FreeX)
			#pragma offload_transfer target(mic:micIdx) nocopy(vX : length(2*fSize*n3) FreeX)

			if (!lowmem)
			{
				#pragma offload_transfer target(mic:micIdx) nocopy(m2X : length(2*fSize*v3) FreeX)
			}
		#endif
	}
}

void	Scalar::transferDev(FieldIndex fIdx)	// Transfers only the internal volume
{
	if (device == DEV_GPU)
	{
		#ifndef	USE_GPU
			printf ("Gpu support not built\n");
			exit   (1);
		#else
			if (fIdx & 1)
				cudaMemcpy((((char *) m_d) + 2*n2*fSize), (((char *) m) + 2*n2*fSize),  2*n3*fSize, cudaMemcpyHostToDevice);

			if (fIdx & 2)
				cudaMemcpy(v_d,  v,  2*n3*fSize, cudaMemcpyHostToDevice);

			if ((fIdx & 4) & (!lowmem))
				cudaMemcpy((((char *) m2_d) + 2*n2*fSize), (((char *) m2) + 2*n2*fSize),  2*n3*fSize, cudaMemcpyHostToDevice);
		#endif
	} else if (device == DEV_XEON) {
		#ifndef	USE_XEON
			printf ("Xeon Phi support not built\n");
			exit   (1);
		#else
			const int micIdx = commAcc();

			if (fIdx & 1)
			{
				#pragma offload_transfer target(mic:micIdx) in(mX : length(2*v3*fSize) ReUseX)
			}

			if (fIdx & 2)
			{
				#pragma offload_transfer target(mic:micIdx) in(vX : length(2*n3*fSize) ReUseX)
			}

			if ((fIdx & 4) & (!lowmem))
			{
				#pragma offload_transfer target(mic:micIdx) in(m2X : length(2*v3*fSize) ReUseX)
			}
		#endif
	}
}

void	Scalar::transferCpu(FieldIndex fIdx)	// Transfers only the internal volume
{
	if (device == DEV_GPU)
	{
		#ifndef	USE_GPU
			printf ("Gpu support not built\n");
			exit   (1);
		#else
			if (fIdx & 1)
				cudaMemcpy(m,  m_d,  2*v3*fSize, cudaMemcpyDeviceToHost);

			if (fIdx & 2)
				cudaMemcpy(v,  v_d,  2*n3*fSize, cudaMemcpyDeviceToHost);

			if ((fIdx & 4) & (!lowmem))
				cudaMemcpy(m2, m2_d, 2*v3*fSize, cudaMemcpyDeviceToHost);
		#endif
	} else if (device == DEV_XEON) {
		#ifndef	USE_XEON
			printf ("Xeon Phi support not built\n");
			exit   (1);
		#else
			const int micIdx = commAcc();

			if (fIdx & 1)
			{
				#pragma offload_transfer target(mic:micIdx) out(mX : length(2*v3*fSize) ReUseX)
			}

			if (fIdx & 2)
			{
				#pragma offload_transfer target(mic:micIdx) out(vX : length(2*n3*fSize) ReUseX)
			}

			if ((fIdx & 4) & (!lowmem))
			{
				#pragma offload_transfer target(mic:micIdx) out(m2X : length(2*v3*fSize) ReUseX)
			}
		#endif
	}
}

void	Scalar::recallGhosts(FieldIndex fIdx)		// Copy to the Cpu the fields in the Gpu that are to be exchanged
{
	if (device == DEV_GPU)
	{
		#ifndef	USE_GPU
			printf ("Gpu support not built\n");
			exit   (1);
		#else
			if (fIdx & FIELD_M)
			{
				cudaMemcpyAsync((((char *) m) + 2*n2*fSize), (((char *) m_d) + 2*n2*fSize), 2*n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[0]);
				cudaMemcpyAsync((((char *) m) + 2*n3*fSize), (((char *) m_d) + 2*n3*fSize), 2*n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[1]);
			}
			else
			{
				cudaMemcpyAsync((((char *) m2) + 2*n2*fSize), (((char *) m2_d) + 2*n2*fSize), 2*n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[0]);
				cudaMemcpyAsync((((char *) m2) + 2*n3*fSize), (((char *) m2_d) + 2*n3*fSize), 2*n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[1]);
			}

			cudaStreamSynchronize(((cudaStream_t *)sStreams)[0]);
			cudaStreamSynchronize(((cudaStream_t *)sStreams)[1]);
		#endif
	} else if (device == DEV_XEON) {
		#ifndef	USE_XEON
			printf ("Xeon Phi support not built\n");
			exit   (1);
		#else
			const int micIdx = commAcc();

			if (fIdx & FIELD_M)
			{
				#pragma offload_transfer target(mic:micIdx) out(mX[2*n2*fSize:2*n2*fSize] : ReUseX)
				#pragma offload_transfer target(mic:micIdx) out(mX[2*n3*fSize:2*n2*fSize] : ReUseX)
			} else {
				#pragma offload_transfer target(mic:micIdx) out(m2X[2*n2*fSize:2*n2*fSize] : ReUseX)
				#pragma offload_transfer target(mic:micIdx) out(m2X[2*n3*fSize:2*n2*fSize] : ReUseX)
			}
		#endif
	}
}

void	Scalar::transferGhosts(FieldIndex fIdx)	// Transfers only the ghosts to the Gpu
{
	if (device == DEV_GPU)
	{
		#ifndef	USE_GPU
			printf ("Gpu support not built\n");
			exit   (1);
		#else
			if (fIdx & FIELD_M)
			{
				cudaMemcpyAsync( ((char *) m_d),                      ((char *) m),                     2*n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[0]);
				cudaMemcpyAsync((((char *) m_d)  + 2*(n3+n2)*fSize), (((char *) m)  + 2*(n3+n2)*fSize), 2*n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[1]);
			} else {
				cudaMemcpyAsync( ((char *) m2_d),                      ((char *) m2),                     2*n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[0]);
				cudaMemcpyAsync((((char *) m2_d)  + 2*(n3+n2)*fSize), (((char *) m2)  + 2*(n3+n2)*fSize), 2*n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[1]);
			}

			cudaStreamSynchronize(((cudaStream_t *)sStreams)[0]);
			cudaStreamSynchronize(((cudaStream_t *)sStreams)[1]);
		#endif
	} else if (device == DEV_XEON) {
		#ifndef	USE_XEON
			printf ("Xeon Phi support not built\n");
			exit   (1);
		#else
			const int micIdx = commAcc();

			if (fIdx & FIELD_M)
			{
				#pragma offload_transfer target(mic:micIdx) in(mX[0:2*n2*fSize] : ReUseX)
				#pragma offload_transfer target(mic:micIdx) in(mX[2*(n2+n3)*fSize:2*n2*fSize] : ReUseX)
			} else {
				#pragma offload_transfer target(mic:micIdx) in(m2X[0:2*n2*fSize] : ReUseX)
				#pragma offload_transfer target(mic:micIdx) in(m2X[2*(n2+n3)*fSize:2*n2*fSize] : ReUseX)
			}
		#endif
	}
}

void	Scalar::sendGhosts(FieldIndex fIdx, CommOperation opComm)
{
	static const int rank = commRank();
	static const int fwdNeig = (rank + 1) % nSplit;
	static const int bckNeig = (rank - 1 + nSplit) % nSplit;

	static const int ghostBytes = 2*n2*fSize;

	static MPI_Request 	rSendFwd, rSendBck, rRecvFwd, rRecvBck;	// For non-blocking MPI Comms

	/* Assign receive buffers to the right parts of m, v */

	void *sGhostBck, *sGhostFwd, *rGhostBck, *rGhostFwd;

	if (fIdx & FIELD_M)
	{
		sGhostBck = (void *) (((char *) m) + 2*n2*fSize);
		sGhostFwd = (void *) (((char *) m) + 2*n3*fSize);
		rGhostBck = m;
		rGhostFwd = (void *) (((char *) m) + 2*(n3+n2)*fSize);
	}
	else
	{
		sGhostBck = (void *) (((char *) m2) + 2*n2*fSize);
		sGhostFwd = (void *) (((char *) m2) + 2*n3*fSize);
		rGhostBck = m2;
		rGhostFwd = (void *) (((char *) m2) + 2*(n3+n2)*fSize);
	}

	switch	(opComm)
	{
		case	COMM_SEND:

			MPI_Send_init(sGhostFwd, ghostBytes, MPI_BYTE, fwdNeig, 2*rank,   MPI_COMM_WORLD, &rSendFwd);
			MPI_Send_init(sGhostBck, ghostBytes, MPI_BYTE, bckNeig, 2*rank+1, MPI_COMM_WORLD, &rSendBck);

			MPI_Start(&rSendFwd);
			MPI_Start(&rSendBck);

			break;

		case	COMM_RECV:

			MPI_Recv_init(rGhostFwd, ghostBytes, MPI_BYTE, fwdNeig, 2*fwdNeig+1, MPI_COMM_WORLD, &rRecvFwd);
			MPI_Recv_init(rGhostBck, ghostBytes, MPI_BYTE, bckNeig, 2*bckNeig,   MPI_COMM_WORLD, &rRecvBck);

			MPI_Start(&rRecvBck);
			MPI_Start(&rRecvFwd);

			break;

		case	COMM_SDRV:

			MPI_Send_init(sGhostFwd, ghostBytes, MPI_BYTE, fwdNeig, 2*rank,   MPI_COMM_WORLD, &rSendFwd);
			MPI_Send_init(sGhostBck, ghostBytes, MPI_BYTE, bckNeig, 2*rank+1, MPI_COMM_WORLD, &rSendBck);
			MPI_Recv_init(rGhostFwd, ghostBytes, MPI_BYTE, fwdNeig, 2*fwdNeig+1, MPI_COMM_WORLD, &rRecvFwd);
			MPI_Recv_init(rGhostBck, ghostBytes, MPI_BYTE, bckNeig, 2*bckNeig,   MPI_COMM_WORLD, &rRecvBck);

			MPI_Start(&rRecvBck);
			MPI_Start(&rRecvFwd);
			MPI_Start(&rSendFwd);
			MPI_Start(&rSendBck);

			break;

		case	COMM_WAIT:

			MPI_Wait(&rSendFwd, MPI_STATUS_IGNORE);
			MPI_Wait(&rSendBck, MPI_STATUS_IGNORE);
			MPI_Wait(&rRecvFwd, MPI_STATUS_IGNORE);
			MPI_Wait(&rRecvBck, MPI_STATUS_IGNORE);

			MPI_Request_free(&rSendFwd);
			MPI_Request_free(&rSendBck);
			MPI_Request_free(&rRecvFwd);
			MPI_Request_free(&rRecvBck);

//			MPI_Barrier(MPI_COMM_WORLD);

			break;
	}

/*
	{
		MPI_Send_init(sGhostFwd, ghostBytes, MPI_BYTE, fwdNeig, 2*rank,   MPI_COMM_WORLD, rSendFwd);
		MPI_Send_init(sGhostBck, ghostBytes, MPI_BYTE, bckNeig, 2*rank+1, MPI_COMM_WORLD, rSendBck);

		MPI_Recv_init(rGhostFwd, ghostBytes, MPI_BYTE, fwdNeig, 2*fwdNeig+1, MPI_COMM_WORLD, rRecvFwd);
		MPI_Recv_init(rGhostBck, ghostBytes, MPI_BYTE, bckNeig, 2*bckNeig,   MPI_COMM_WORLD, rRecvBck);

		MPI_Start(rRecvBck);
		MPI_Start(rRecvFwd);
		MPI_Start(rSendFwd);
		MPI_Start(rSendBck);
	}
*/
}

void	Scalar::exchangeGhosts(FieldIndex fIdx)
{
	recallGhosts(fIdx);
	sendGhosts(fIdx, COMM_SDRV);
	sendGhosts(fIdx, COMM_WAIT);
	transferGhosts(fIdx);
}

//	USAR TEMPLATES PARA ESTO

void	Scalar::foldField()
{
	int	shift;

	shift = mAlign/(2*fSize);

	switch (precision)
	{
		case FIELD_DOUBLE:

			for (int iz=0; iz < Lz; iz++)
			{
				memcpy (         m,                    ((char *)m) + 2*fSize*n2*(1+iz), 2*fSize*n2);
				memcpy (((char *)m) + 2*fSize*(n3+n2), ((char *)v) + 2*fSize*n2*iz,     2*fSize*n2);

				for (int iy=0; iy < n1/shift; iy++)
					for (int ix=0; ix < n1; ix++)
						for (int sy=0; sy<shift; sy++)
						{
							int oIdx = (iy+sy*(n1/shift))*n1 + ix;
							int dIdx = iz*n2 + iy*n1*shift + ix*shift + sy;

							((double *) m)[2*dIdx+n2]   = ((double *) m)[2*oIdx];
							((double *) m)[2*dIdx+n2+1] = ((double *) m)[2*oIdx+1];
							((double *) v)[2*dIdx]      = ((double *) m)[2*(oIdx+n2+n3)];
							((double *) v)[2*dIdx+1]    = ((double *) m)[2*(oIdx+n2+n3)+1];
						}
			}

			break;

		case FIELD_SINGLE:

			for (int iz=0; iz < Lz; iz++)
			{
				memcpy (         m,                    ((char *)m) + 2*fSize*n2*(1+iz), 2*fSize*n2);
				memcpy (((char *)m) + 2*fSize*(n3+n2), ((char *)v) + 2*fSize*n2*iz,     2*fSize*n2);

				for (int iy=0; iy < n1/shift; iy++)
					for (int ix=0; ix < n1; ix++)
						for (int sy=0; sy<shift; sy++)
						{
							int oIdx = (iy+sy*(n1/shift))*n1 + ix;
							int dIdx = iz*n2 + iy*n1*shift + ix*shift + sy;

							((float *) m)[2*dIdx+n2]   = ((float *) m)[2*oIdx];
							((float *) m)[2*dIdx+n2+1] = ((float *) m)[2*oIdx+1];
							((float *) v)[2*dIdx]      = ((float *) m)[2*(oIdx+n2+n3)];
							((float *) v)[2*dIdx+1]    = ((float *) m)[2*(oIdx+n2+n3)+1];
						}
			}

			break;
	}

	return;
}

void	Scalar::unfoldField()
{
	int	shift;

	shift = mAlign/(2*fSize);

	switch (precision)
	{
		case FIELD_DOUBLE:

			for (int iz=0; iz < Lz; iz++)
			{
				memcpy (         m,                    ((char *)m) + 2*fSize*n2*(1+iz), 2*fSize*n2);
				memcpy (((char *)m) + 2*fSize*(n3+n2), ((char *)v) + 2*fSize*n2*iz,     2*fSize*n2);

				for (int iy=0; iy < n1/shift; iy++)
					for (int ix=0; ix < n1; ix++)
						for (int sy=0; sy<shift; sy++)
						{
							int oIdx = iy*n1*shift + ix*shift + sy;
							int dIdx = iz*n2 + (iy+sy*(n1/shift))*n1 + ix;

							((double *) m)[2*dIdx+n2]   = ((double *) m)[2*oIdx];
							((double *) m)[2*dIdx+n2+1] = ((double *) m)[2*oIdx+1];
							((double *) v)[2*dIdx]      = ((double *) m)[2*(oIdx+n2+n3)];
							((double *) v)[2*dIdx+1]    = ((double *) m)[2*(oIdx+n2+n3)+1];
						}
			}

			break;

		case FIELD_SINGLE:

			for (int iz=0; iz < Lz; iz++)
			{
				memcpy (         m,                    ((char *)m) + 2*fSize*n2*(1+iz), 2*fSize*n2);
				memcpy (((char *)m) + 2*fSize*(n3+n2), ((char *)v) + 2*fSize*n2*iz,     2*fSize*n2);

				for (int iy=0; iy < n1/shift; iy++)
					for (int ix=0; ix < n1; ix++)
						for (int sy=0; sy<shift; sy++)
						{
							int oIdx = iy*n1*shift + ix*shift + sy;
							int dIdx = iz*n2 + (iy+sy*(n1/shift))*n1 + ix;

							((float *) m)[2*dIdx+n2]   = ((float *) m)[2*oIdx];
							((float *) m)[2*dIdx+n2+1] = ((float *) m)[2*oIdx+1];
							((float *) v)[2*dIdx]      = ((float *) m)[2*(oIdx+n2+n3)];
							((float *) v)[2*dIdx+1]    = ((float *) m)[2*(oIdx+n2+n3)+1];
						}
			}

			break;
	}

	return;
}

void	Scalar::unfoldField2D(const int sZ)
{
	int	shift;

	shift = mAlign/(2*fSize);

	switch (precision)
	{
		case FIELD_DOUBLE:

		for (int iy=0; iy < n1/shift; iy++)
			for (int ix=0; ix < n1; ix++)
				for (int sy=0; sy<shift; sy++)
				{
					int oIdx = (sZ+1)*n2 + iy*n1*shift + ix*shift + sy;
					int dIdx = (iy+sy*(n1/shift))*n1 + ix;

					((double *) m)[2*dIdx]   = ((double *) m)[2*oIdx];
					((double *) m)[2*dIdx+1] = ((double *) m)[2*oIdx+1];
				}

		break;

		case FIELD_SINGLE:

		for (int iy=0; iy < n1/shift; iy++)
			for (int ix=0; ix < n1; ix++)
				for (int sy=0; sy<shift; sy++)
				{
					int oIdx = (sZ+1)*n2 + iy*n1*shift + ix*shift + sy;
					int dIdx = (iy+sy*(n1/shift))*n1 + ix;

					((float *) m)[2*dIdx]   = ((float *) m)[2*oIdx];
					((float *) m)[2*dIdx+1] = ((float *) m)[2*oIdx+1];
				}

			break;
	}

	return;
}
//	USA M2, ARREGLAR LOWMEM
void	Scalar::prepareCpu(int *window)
{
	if (precision == FIELD_DOUBLE)
	{
		#pragma omp parallel for default(shared) schedule(static)
		for(int i=0; i < n3; i++)
			((std::complex<double> *) m2)[i] = I*(((std::complex<double> *) v)[i]/((std::complex<double> *) m)[i]).imag()*((double) window[i]);
	}
	else
	{
		#pragma omp parallel for default(shared) schedule(static)
		for(int i=0; i < n3; i++)
			((std::complex<float> *) m2)[i] = If*(((std::complex<float> *) v)[i]/((std::complex<float> *) m)[i]).imag()*((float) window[i]);
	}
}

// EN TEORIA USA M2, ARREGLAR LOWMEM
void	Scalar::squareGpu()
{
//	square(m2_d, n1, Lz, n3, precision);
}

// USA M2, ARREGLAR LOWMEM
void	Scalar::squareCpu()
{
	if (precision == FIELD_DOUBLE)
	{
		#pragma omp parallel for default(shared) schedule(static)
		for(int i=0; i < n3; i++)
			((std::complex<double> *) m2)[i] = pow(abs(((std::complex<double> *) m2)[i]/((double) n3)),2);
	}
	else
	{
		#pragma omp parallel for default(shared) schedule(static)
		for(int i=0; i < n3; i++)
			((std::complex<float> *) m2)[i] = pow(abs(((std::complex<float> *) m2)[i]/((float) n3)),2);
	}
}

/*	ARREGLAR PARA DIFERENTES PRECISIONES	*/
/*
void	Scalar::addZmom(int pz, int oPz, void *data, int sign)
{
	int zDiff = pz - oPz;
	int zBase = commRank()*Lz;

	if (zDiff == 0)
		return;

	switch (precision)
	{
		case FIELD_DOUBLE:
		{
			complex<double> phase[Lz];
			double sg = sign;

			#pragma omp parallel for default(shared) schedule(static)
			for (int zc = 0; zc < Lz; zc++)
			{
				int zTot = zBase + zc;
				phase[zc] = exp(I*(sg*(pz*zTot)));
			}


			#pragma omp parallel for default(shared) schedule(static)
			for (int idx = 0; idx < n2; idx++)
				for (int zc = 0; zc < Lz; zc++)
					((complex<double> *) data)[n2*(zc+1)+idx] *= phase[zc];
		}

		break;

		case FIELD_SINGLE:
		{
			complex<float> phase[Lz];
			float sg = sign;

			#pragma omp parallel for default(shared) schedule(static)
			for (int zc = 0; zc < Lz; zc++)
			{
				int zTot = zBase + zc;
				phase[zc] = exp(If*(sg*(pz*zTot)));
			}


			#pragma omp parallel for default(shared) schedule(static)
			for (int idx = 0; idx < n2; idx++)
				for (int zc = 0; zc < Lz; zc++)
					((complex<float> *) data)[n2*(zc+1)+idx] *= phase[zc];
		}

		break;
	}
}
*/

void	Scalar::fftCpu	(int sign)
{
	runFFT(sign);
}

/*	CODIGO VIEJO INUTIL, IGUAL PARA FFT GPU...
void	Scalar::fftCpu	(int sign)
{
	int	  oldPz = 0;

	runFFT(sign);

	switch (precision)
	{
		case FIELD_DOUBLE:

		for (int cz = 0; cz < Tz; cz++)
		{
			int pz = cz - (cz/(Tz >> 1))*Tz;
			int rk = cz/Lz;
			int vk = cz%Lz;

			double *mDs = (((double *) m) + (2*n2*(vk+1)));

			addZmom (pz, oldPz, m2, sign);

			oldPz = pz;

			#pragma omp parallel for default(shared) schedule(static)
			for (int idx = 0; idx < n2; idx++)
			{
				((complex<double> *) m)[idx] = complex<double>(0.,0.);

				for (int mz = 0; mz < Lz; mz++)
					((complex<double> *) m)[idx] += ((complex<double> *) m2)[idx + (mz+1)*n2];
			}

			MPI_Reduce(m, mDs, 2*n2, MPI_DOUBLE, MPI_SUM, rk, MPI_COMM_WORLD);
		}

		break;

		case FIELD_SINGLE:

		for (int cz = 0; cz < Tz; cz++)
		{
			int pz = cz - (cz/(Tz >> 1))*Tz;
			int rk = cz/Lz;
			int vk = cz%Lz;

			float *mDs = (((float *) m) + (n2*(vk+1)));

			addZmom (pz, oldPz, m2, sign);

			oldPz = pz;

			#pragma omp parallel for default(shared) schedule(static)
			for (int idx = 0; idx < n2; idx++)
			{
				((complex<float> *) m)[idx] = complex<float>(0.,0.);

				for (int mz = 0; mz < Lz; mz++)
					((complex<float> *) m)[idx] += ((complex<float> *) m2)[idx + (mz+1)*n2];
			}

			MPI_Reduce(m, mDs, 2*n2, MPI_FLOAT, MPI_SUM, rk, MPI_COMM_WORLD);
		}

		break;
	}
}
*/

// ESTO YA NO SE USA, LA FFT ES SIEMPRE EN LA CPU
void	Scalar::fftGpu	(int sign)
{
#ifdef	USE_GPU
	runCudaFFT(m2_d, sign);
#endif
}

void	Scalar::genConf	(ConfType cType, const int parm1, const double parm2)
{
	switch (cType)
	{
		case CONF_NONE:
		break;

		case CONF_KMAX:		// kMax = parm1, kCrit = parm2

//		if (device == DEV_CPU || device == DEV_XEON)	//	Do this always...
		{
			switch (precision)
			{
				case FIELD_DOUBLE:

				momConf(parm1, parm2);

				break;

				case FIELD_SINGLE:

				momConf(parm1, (float) parm2);

				break;

				default:

				printf("Wrong precision\n");
				exit(1);

				break;
			}

			fftCpu (1);	// FFTW_BACKWARD
		}

		exchangeGhosts(FIELD_M);

		break;

		case CONF_SMOOTH:	// iter = parm1, alpha = parm2

		switch (device)
		{
			case	DEV_XEON:
			case	DEV_CPU:

			randConf ();
			smoothConf (parm1, parm2);
			break;

			case	DEV_GPU:

//			randConfGpu (m, n1, Lz, n3, precision);
//			smoothConfGpu (this, iter, alpha);
			break;
		}

		break;

		default:

		printf("Configuration type not recognized\n");
		exit(1);

		break;
	}

	if (cType != CONF_NONE)
	{
		memcpy (v, static_cast<char *> (m) + 2*fSize*n2, 2*fSize*n3);
		scaleField (FIELD_M, *z);
	}
}

void	Scalar::randConf ()
{
	int	maxThreads = omp_get_max_threads();
	int	*sd = (int *) malloc(sizeof(int)*maxThreads);

	std::random_device seed;		// Totally random seed coming from memory garbage

	for (int i=0; i<maxThreads; i++)
		sd[i] = seed();

	switch (precision)
	{
		case FIELD_DOUBLE:

		#pragma omp parallel default(shared)
		{
			int nThread = omp_get_thread_num();

			//JAVIER commented next line, it seems to work
			//printf	("Thread %d got seed %d\n", nThread, sd[nThread]);

			std::mt19937_64 mt64(sd[nThread]);		// Mersenne-Twister 64 bits, independent per thread
			//JAVIER included negative values
			std::uniform_real_distribution<double> uni(-1.0, 1.0);

			#pragma omp for schedule(static)	// This is NON-REPRODUCIBLE, unless one thread is used. Alternatively one can fix the seeds
			for (int idx=2*n2; idx<2*(n2+n3); idx+=2)
			{
				((double *) m)[idx]   = uni(mt64);
				((double *) m)[idx+1] = uni(mt64);
			}
		}
		//JAVIER control print
		printf	("CHECK: %d, %d+1 got (%lf,%lf)  \n", 2*n2,2*n2,((double *) m)[2*n2],((double *) m)[2*n2+1]);

		break;

		case FIELD_SINGLE:

		#pragma omp parallel default(shared)
		{
			int nThread = omp_get_thread_num();

			//printf	("Thread %d got seed %d\n", nThread, sd[nThread]);

			std::mt19937_64 mt64(sd[nThread]);		// Mersenne-Twister 64 bits, independent per thread
			//JAVIER included negative values
			std::uniform_real_distribution<float> uni(-1.0, 1.0);

			#pragma omp for schedule(static)	// This is NON-REPRODUCIBLE, unless one thread is used. Alternatively one can fix the seeds
			for (int idx=2*n2; idx<2*(n2+n3); idx+=2)
			{
				((float *) m)[idx]   = uni(mt64);
				((float *) m)[idx+1] = uni(mt64);
			}
		}

		break;

		default:

		printf("Unrecognized precision\n");
		free(sd);
		exit(1);

		break;
	}

	free(sd);
}// End Scalar::randConf ()


template<typename Float>
void	Scalar::iteraField(const int iter, const Float alpha)
{
	const Float One = 1.;
	const Float OneSixth = (1./6.);

	exchangeGhosts(FIELD_M);

	complex<Float> *mCp = static_cast<complex<Float>*> (m);
	complex<Float> *vCp = static_cast<complex<Float>*> (v);
	//JAVIER
	//printf("smoothing check m[0]= (%lf,%lf)\n",  ((Float *) m)[n2] , ((Float *) m)[n2] );
	//printf("the same? ????? m[0]= (%lf,%lf)\n",  ((Float *) mCp)[n2] , ((Float *) mCp)[n2] ); yes

	for (int it=0; it<iter; it++)
	{
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)
			for (int idx=n2; idx<(n2+n3); idx++)
			{
				int iPx, iMx, iPy, iMy, iPz, iMz, X[3];
				indexXeon::idx2Vec (idx, X, n1);

				if (X[0] == 0)
				{
					iPx = idx + 1;
					iMx = idx + n1 - 1;
				} else {
					if (X[0] == n1 - 1)
					{
						iPx = idx - n1 + 1;
						iMx = idx - 1;
					} else {
						iPx = idx + 2;
						iMx = idx - 2;
					}
				}

				if (X[1] == 0)
				{
					iPx = idx + n1;
					iMx = idx + n2 - n1;
				} else {
					if (X[1] == n1 - 1)
					{
						iPx = idx - n2 + n1;
						iMx = idx - n1;
					} else {
						iPx = idx + n1;
						iMx = idx - n1;
					}
				}

				iPz = idx + n2;
				iMz = idx - n2;
				//Uses v to copy the smoothed configuration
				vCp[idx-n2]   = alpha*mCp[idx] + OneSixth*(One-alpha)*(mCp[iPx] + mCp[iMx] + mCp[iPy] + mCp[iMy] + mCp[iPz] + mCp[iMz]);
			}
		}
		//Copies v to m
		memcpy (static_cast<char *>(m) + 2*fSize*n2, v, 2*fSize*n3);
		exchangeGhosts(FIELD_M);

		//printf("smoothing check m[0]= (%lf,%lf)\n",  real(((complex<double> *) m)[n2]), real(mCp[n2]) ); both give the same
		printf("smoothing check m[0],m[1]= (%lf,%lf), (%lf,%lf)\n",  real(mCp[n2]), imag(mCp[n2]),real(mCp[n2+1]), imag(mCp[n2+1]) ); 
	}//END iteration loop
}

void	Scalar::smoothConf (const int iter, const double alpha)
{
	switch	(precision)
	{
		case	FIELD_DOUBLE:
		iteraField (iter, alpha);
		break;

		case	FIELD_SINGLE:
		iteraField (iter, (float) alpha);
		break;

		default:
		printf("Unrecognized precision\n");
		exit(1);
		break;
	}
}




template<typename Float>
void	Scalar::momConf (const int kMax, const Float kCrit)
{
	const int kMax2  = kMax*kMax;
	const Float Twop = 2.0*M_PI;

	complex<Float> *fM;

	if (!lowmem)
		fM = static_cast<complex<Float>*> (m2);
	else
		fM = static_cast<complex<Float>*> (static_cast<void*>(static_cast<char*>(m) + 2*fSize*n2));

	int	maxThreads = omp_get_max_threads();
	int	*sd = (int *) malloc(sizeof(int)*maxThreads);

	std::random_device seed;		// Totally random seed coming from memory garbage

	for (int i=0; i<maxThreads; i++)
		sd[i] = seed();

	#pragma omp parallel default(shared)
	{
		int nThread = omp_get_thread_num();

	//	printf	("Thread %d got seed %d\n", nThread, sd[nThread]);

		std::mt19937_64 mt64(sd[nThread]);		// Mersenne-Twister 64 bits, independent per thread
		std::uniform_real_distribution<Float> uni(0.0, 1.0);

		#pragma omp for schedule(static)
		for (int oz = 0; oz < Tz; oz++)
		{
			if (oz/Lz != commRank())
				continue;

			int pz = oz - (oz/(Tz >> 1))*Tz;

			for(int py = -(kMax-pz); py <= (kMax-pz); py++)
				for(int px = -(kMax-pz-py); px <= (kMax-pz-py); px++)
				{
					int idx  = n2 + ((px + n1)%n1) + ((py+n1)%n1)*n1 + ((pz+Tz)%Tz)*n2 - commRank()*n3;
					int modP = px*px + py*py + pz*pz;

					if (modP <= kMax2)
					{
						Float mP = sqrt(((Float) modP))/((Float) kCrit);
						Float vl = Twop*(uni(mt64));
						Float sc = (modP == 0) ? 1.0 : sin(mP)/mP;

						fM[idx] = complex<Float>(cos(vl), sin(vl))*sc;
					}
				}
		}
	}

	free(sd);
}

void	Scalar::scaleField (FieldIndex fIdx, double factor)
{
	switch (precision)
	{
		case FIELD_DOUBLE:
		{
			complex<double> *field;
			int vol = n3;

			switch (fIdx)
			{
				case FIELD_M:
				field = static_cast<complex<double>*> (m);
				vol = v3;
				break;

				case FIELD_V:
				field = static_cast<complex<double> *> (v);
				break;

				case FIELD_M2:
				if (lowmem) {
					printf ("Wrong field. Lowmem forbids the use of m2");
					return;
				}

				field = static_cast<complex<double> *> (m2);
				vol = v3;
				break;

				default:
				printf ("Wrong field. Valid possibilities: FIELD_M, FIELD_M2 and FIELD_V");
				return;
				break;
			}

			#pragma omp parallel for default(shared) schedule(static)
			for (int lpc = 0; lpc < vol; lpc++)
				field[lpc] *= factor;

			break;
		}

		case FIELD_SINGLE:
		{
			complex<float> *field;
			float fac = factor;
			int vol = n3;

			switch (fIdx)
			{
				case FIELD_M:
				field = static_cast<complex<float> *> (m);
				vol = v3;
				break;

				case FIELD_V:
				field = static_cast<complex<float> *> (v);
				break;

				case FIELD_M2:
				if (lowmem) {
					printf ("Wrong field. Lowmem forbids the use of m2");
					return;
				}

				field = static_cast<complex<float> *> (m2);
				vol = v3;
				break;

				default:
				printf ("Wrong field. Valid possibilities: FIELD_M, FIELD_M2 and FIELD_V");
				break;
			}

			#pragma omp parallel for default(shared) schedule(static)
			for (int lpc = 0; lpc < vol; lpc++)
				field[lpc] *= fac;

			break;
		}

		default:
		printf("Unrecognized precision\n");
		exit(1);
		break;
	}
}
