#include<cstdlib>
#include<cstring>
#include<complex>

#include"square.h"
#include"fft/fftCuda.h"
#include"fft/fftCode.h"
#include"enum-field.h"

#include"utils/flopCounter.h"

#include"comms/comms.h"

#ifdef	USE_GPU
	#include<cuda.h>
	#include<cuda_runtime.h>
	#include "cudaErrors.h"
#endif

#include "utils/memAlloc.h"
#include "utils/parse.h"

#include<mpi.h>
#include<omp.h>

#ifdef	USE_XEON
	#include"utils/xeonDefs.h"
#endif

#include "utils/index.h"

#define	_SCALAR_CLASS_

using namespace std;

const std::complex<double> I(0.,1.);
const std::complex<float> If(0.,1.);

class	Scalar
{
	private:

	const size_t n1;
	const size_t n2;
	const size_t n3;

	const size_t Lz;
	const size_t Tz;
	const size_t Ez;
	const size_t v3;

	const int  nSplit;
	const bool lowmem;

	DeviceType	device;
	FieldPrecision	precision;
	FieldType	fieldType;
	LambdaType	lambdaType;

	size_t	fSize;
	size_t	mAlign;
	//JAVI
	int sHift;

	double	*z;

	void	*m,   *v,   *m2;			// Cpu data
#ifdef	USE_GPU
	void	*m_d, *v_d, *m2_d;			// Gpu data

	void	*sStreams;
#endif
	void	recallGhosts(FieldIndex fIdx);		// Move the fileds that will become ghosts from the Cpu to the Gpu
	void	transferGhosts(FieldIndex fIdx);	// Copy back the ghosts to the Gpu

//	void	scaleField(FieldIndex fIdx, double factor);
//	void	randConf();
//	void	smoothConf(const size_t iter, const double alpha);
	//JAVIER
//	void	normaliseField(FieldIndex fIdx);

//	template<typename Float>
//	void	iteraField(const size_t iter, const Float alpha);

	//JAVIER
	template<typename Float>
	void	ENERGY(const Float zz, FILE *enWrite, Float &Grho1, Float &Gtheta1, Float &Vrho1, Float &Vtheta1, Float &Krho1, Float &Ktheta1); // TEST

	template<typename Float>
	void ENERGY2(const Float zz, FILE *enWrite, double &Grho1, double &Gtheta1, double &Vrho1, double &Vtheta1, double &Krho1, double &Ktheta1); // TEST

	template<typename Float>
	void energymapTheta(const Float zz, const int index, void *contbin, int numbins); // TEST


//	template<typename Float>
//	void	momConf(const int kMax, const Float kCrit);

	public:

			 Scalar(const size_t nLx, const size_t nLz, FieldPrecision prec, DeviceType dev, const double zI, char fileName[], bool lowmem, const int nSp,
				ConfType cType, const size_t parm1, const double parm2, FlopCounter *fCount);
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

	size_t		TotalSize() { return n3*nSplit; }
	size_t		Size()      { return n3; }
	size_t		Surf()      { return n2; }
	size_t		Length()    { return n1; }
	size_t		TotalDepth(){ return Lz*nSplit; }
	size_t		Depth()     { return Lz; }
	size_t		eSize()     { return v3; }
	size_t		eDepth()    { return Ez; }

	FieldPrecision	Precision() { return precision; }
	DeviceType	Device()    { return device; }
	LambdaType	Lambda()    { return lambdaType; }
	FieldType	Fieldo()    { return fieldType; }

	void		SetLambda(LambdaType newLambda) { lambdaType = newLambda; }

	size_t		DataSize () { return fSize; }
	size_t		DataAlign() { return mAlign; }
	//JAVI
	int		shift() { return sHift; }

	double		*zV() { return z; }
	const double	*zV() const { return z; }

	void		setZ(const double newZ) { *z = newZ; }

	void	setField	(FieldType field);
/*
	void	foldField	();
	void	unfoldField	();
	void	unfoldField2D	(const size_t sZ);	// Just for the maps
*/
	void	transferDev(FieldIndex fIdx);		// Move data to device (Gpu or Xeon)
	void	transferCpu(FieldIndex fIdx);		// Move data to Cpu

	void	sendGhosts(FieldIndex fIdx, CommOperation commOp);	// Send the ghosts in the Cpu using MPI, use this to exchange ghosts with Cpus
	void	exchangeGhosts(FieldIndex fIdx);	// Transfer ghosts from neighbouring ranks, use this to exchange ghosts with Gpus

	void	fftCpu(int sign);			// Fast Fourier Transform in the Cpu
	void	fftGpu(int sign);			// Fast Fourier Transform in the Gpu
	void	fftCpuSpectrum(int sign);			// Fast Fourier Transform in the Cpu for m2 [axion spectrum usage]

	void	prepareCpu(int *window);		// Sets the field for a FFT, prior to analysis
	//JAVIER
	//void	thetaz2m2(int *window);		// COPIES dTHETA/dz into m2	//not used
	void	theta2m2();//int *window);	// COPIES THETA + I dTHETA/dz     into m2
	double	maxtheta();								// RETURNS THE MAX VALUE OF THETA [OR IM m]
	double	thetaDIST(int numbins, void *thetabin);							// RETURNS (MAX THETA) AND BINNED DATA FOR THETA DISTRIBUTION
	void	denstom(); 	//

	void	squareGpu();				// Squares the m2 field in the Gpu
	void	squareCpu();				// Squares the m2 field in the Cpu

//	void	genConf	(ConfType cType, const size_t parm1, const double parm2);
	//JAVIER
//	void	writeENERGY (double zzz, FILE *enwrite);
	void	writeENERGY (double zzz, FILE *enwrite, double &Gfr, double &Gft, double &Vfr, double &Vft, double &Kfr, double &Kft); // TEST
	void	writeMAPTHETA (double zzz, int index, void *contbin, int numbins);

#ifdef	USE_GPU
	void	*Streams() { return sStreams; }
#endif
//	template<typename Float>
//	void	normaCOREField(const Float alpha);
};

#include "gen/genConf.h"

	Scalar::Scalar(const size_t nLx, const size_t nLz, FieldPrecision prec, DeviceType dev, const double zI, char fileName[], bool lowmem, const int nSp, ConfType cType, const size_t parm1,
		       const double parm2, FlopCounter *fCount) : nSplit(nSp), n1(nLx), n2(nLx*nLx), n3(nLx*nLx*nLz), Lz(nLz), Ez(nLz + 2), Tz(Lz*nSp), v3(nLx*nLx*(nLz + 2)), precision(prec), device(dev),
		       lowmem(lowmem)
{
	fieldType  = FIELD_SAXION;
	lambdaType = LAMBDA_Z2;

	switch (prec)
	{
		case FIELD_DOUBLE:

		fSize = sizeof(double)*2;
		break;

		case FIELD_SINGLE:

		fSize = sizeof(float)*2;
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
	//JAVI
	sHift = mAlign/fSize;

	if (n2*fSize % mAlign)
	{
		printf("Error: misaligned memory. Are you using an odd dimension?\n");
		exit(1);
	}

	const size_t	mBytes = v3*fSize;
	const size_t	vBytes = n3*fSize;

#ifdef	USE_XEON
	alignAlloc ((void**) &mX, mAlign, mBytes);
	alignAlloc ((void**) &vX, mAlign, vBytes);

	if (!lowmem)
		alignAlloc ((void**) &m2X, mAlign, mBytes);

	m = mX;
	v = vX;

	if (!lowmem)
		m2 = m2X;
	else
		m2 = m2X = NULL;
#else
	printf("Allocating m and v\n");
	alignAlloc ((void**) &m, mAlign, mBytes);
	alignAlloc ((void**) &v, mAlign, vBytes);

	if (!lowmem)
	{
	printf("Allocating m2\n");
		alignAlloc ((void**) &m2, mAlign, mBytes);
	}
	else
	{
	printf("LOWMEM!\n");
		m2 = NULL;
	}
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
	printf("set m,v=0\n");
	memset (m, 0, fSize*v3);
	memset (v, 0, fSize*n3);

	if (!lowmem)
		memset (m2, 0, fSize*v3);

	alignAlloc ((void **) &z, mAlign, mAlign);//sizeof(double));

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
	}// else {
//		initFFT(static_cast<void *>(static_cast<char *> (m) + n2*fSize), m2, n1, Tz, precision, lowmem);
//	}

	*z = zI;

	/*	If present, read fileName	*/

	if (cType == CONF_NONE)
	{
/*		if (fileName != NULL)
		{
			FILE *fileM = fopen(fileName,"r");

			if (fileM == 0)
			{
				printf("Sorry! Could not find initial Conditions\n");
				exit(1);
			}

			fread(static_cast<char *> (m) + fSize*n2, fSize*2, n3, fileM);
			fclose(fileM);

			memcpy (v, static_cast<char *> (m) + fSize*n2, fSize*n3);
//			scaleField (FIELD_M, zI);

			if (prec == FIELD_DOUBLE)
			{
				#pragma omp parallel for schedule(static)
				for (size_t i=n2; i<n3+n2; i++)
				{
					complex<double> tmp((i-n2)&1, (i-n2)&1);//1-((i-n2)&1));
					static_cast<complex<double>*>(m)[i] = tmp;
				}
			}
			else
			{
				#pragma omp parallel for schedule(static)
				for (size_t i=n2; i<n3+n2; i++)
				{
					complex<float> tmp((i-n2)&1, (i-n2)&1);//1-((i-n2)&1));
					static_cast<complex<float>*>(m)[i] = tmp;
				}
			}
			memcpy (v, static_cast<char *> (m) + fSize*n2, fSize*n3);
		}
*/
	} else {
		if (cType == CONF_KMAX)
			initFFT(static_cast<void *>(static_cast<char *> (m) + n2*fSize), m2, n1, Tz, precision, lowmem);

		genConf	(this, cType, parm1, parm2, fCount);
	}

	if (dev == DEV_XEON)
	{
#ifndef	USE_XEON
		printf ("Xeon Phi support not built\n");
		exit   (1);
#else
		const int micIdx = commAcc();

		#pragma offload_transfer target(mic:micIdx) nocopy(mX : length(fSize*v3) AllocX)
		#pragma offload_transfer target(mic:micIdx) nocopy(vX : length(fSize*n3) AllocX)

		if (!lowmem)
		{
			#pragma offload_transfer target(mic:micIdx) nocopy(m2X : length(fSize*v3) AllocX)
		}
#endif
	}
	//
	printf("FFTing m2\n");
	initFFTSpectrum(m2, n1, Tz, precision, lowmem);
}

// END SCALAR

	Scalar::~Scalar()
{
	printf ("Calling destructor...\n");
	fflush (stdout);
	if (m != NULL)
		trackFree(&m, ALLOC_ALIGN);

	if (v != NULL && fieldType == FIELD_SAXION)
		trackFree(&v, ALLOC_ALIGN);

	if (m2 != NULL)
		trackFree(&m2, ALLOC_ALIGN);

	if (z != NULL)
		trackFree((void **) &z, ALLOC_ALIGN);

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
			closeFFT();
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

			#pragma offload_transfer target(mic:micIdx) nocopy(mX : length(fSize*v3) FreeX)
			#pragma offload_transfer target(mic:micIdx) nocopy(vX : length(fSize*n3) FreeX)

			if (!lowmem)
			{
				#pragma offload_transfer target(mic:micIdx) nocopy(m2X : length(fSize*v3) FreeX)
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
				cudaMemcpy((((char *) m_d) + n2*fSize), (((char *) m) + n2*fSize),  n3*fSize, cudaMemcpyHostToDevice);

			if (fIdx & 2)
				cudaMemcpy(v_d,  v,  n3*fSize, cudaMemcpyHostToDevice);

			if ((fIdx & 4) & (!lowmem))
				cudaMemcpy((((char *) m2_d) + n2*fSize), (((char *) m2) + n2*fSize),  n3*fSize, cudaMemcpyHostToDevice);
		#endif
	} else if (device == DEV_XEON) {
		#ifndef	USE_XEON
			printf ("Xeon Phi support not built\n");
			exit   (1);
		#else
			const int micIdx = commAcc();

			if (fIdx & 1)
			{
				#pragma offload_transfer target(mic:micIdx) in(mX : length(v3*fSize) ReUseX)
			}

			if (fIdx & 2)
			{
				#pragma offload_transfer target(mic:micIdx) in(vX : length(n3*fSize) ReUseX)
			}

			if ((fIdx & 4) & (!lowmem))
			{
				#pragma offload_transfer target(mic:micIdx) in(m2X : length(v3*fSize) ReUseX)
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
				cudaMemcpy(m,  m_d,  v3*fSize, cudaMemcpyDeviceToHost);

			if (fIdx & 2)
				cudaMemcpy(v,  v_d,  n3*fSize, cudaMemcpyDeviceToHost);

			if ((fIdx & 4) & (!lowmem))
				cudaMemcpy(m2, m2_d, v3*fSize, cudaMemcpyDeviceToHost);
		#endif
	} else if (device == DEV_XEON) {
		#ifndef	USE_XEON
			printf ("Xeon Phi support not built\n");
			exit   (1);
		#else
			const int micIdx = commAcc();

			if (fIdx & 1)
			{
				#pragma offload_transfer target(mic:micIdx) out(mX : length(v3*fSize) ReUseX)
			}

			if (fIdx & 2)
			{
				#pragma offload_transfer target(mic:micIdx) out(vX : length(n3*fSize) ReUseX)
			}

			if ((fIdx & 4) & (!lowmem))
			{
				#pragma offload_transfer target(mic:micIdx) out(m2X : length(v3*fSize) ReUseX)
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
				cudaMemcpyAsync(static_cast<char *> (m) + n2*fSize, static_cast<char *> (m_d) + n2*fSize, n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[0]);
				cudaMemcpyAsync(static_cast<char *> (m) + n3*fSize, static_cast<char *> (m_d) + n3*fSize, n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[1]);
			}
			else
			{
				cudaMemcpyAsync(static_cast<char *> (m2) + n2*fSize, static_cast<char *> (m2_d) + n2*fSize, n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[0]);
				cudaMemcpyAsync(static_cast<char *> (m2) + n3*fSize, static_cast<char *> (m2_d) + n3*fSize, n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[1]);
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
				#pragma offload_transfer target(mic:micIdx) out(mX[n2*fSize:n2*fSize] : ReUseX)
				#pragma offload_transfer target(mic:micIdx) out(mX[n3*fSize:n2*fSize] : ReUseX)
			} else {
				#pragma offload_transfer target(mic:micIdx) out(m2X[n2*fSize:n2*fSize] : ReUseX)
				#pragma offload_transfer target(mic:micIdx) out(m2X[n3*fSize:n2*fSize] : ReUseX)
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
				cudaMemcpyAsync(static_cast<char *> (m_d),                 static_cast<char *> (m),                 n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[0]);
				cudaMemcpyAsync(static_cast<char *> (m_d) + (n3+n2)*fSize, static_cast<char *> (m) + (n3+n2)*fSize, n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[1]);
			} else {
				cudaMemcpyAsync(static_cast<char *> (m2_d),                 static_cast<char *> (m2),                    n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[0]);
				cudaMemcpyAsync(static_cast<char *> (m2_d) + (n3+n2)*fSize, static_cast<char *> (m2)  + (n3+n2)*fSize, n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[1]);
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
				#pragma offload_transfer target(mic:micIdx) in(mX[0:n2*fSize] : ReUseX)
				#pragma offload_transfer target(mic:micIdx) in(mX[(n2+n3)*fSize:n2*fSize] : ReUseX)
			} else {
				#pragma offload_transfer target(mic:micIdx) in(m2X[0:n2*fSize] : ReUseX)
				#pragma offload_transfer target(mic:micIdx) in(m2X[(n2+n3)*fSize:n2*fSize] : ReUseX)
			}
		#endif
	}
}

void	Scalar::sendGhosts(FieldIndex fIdx, CommOperation opComm)
{
	static const int rank = commRank();
	static const int fwdNeig = (rank + 1) % nSplit;
	static const int bckNeig = (rank - 1 + nSplit) % nSplit;

	const int ghostBytes = n2*fSize;

	static MPI_Request 	rSendFwd, rSendBck, rRecvFwd, rRecvBck;	// For non-blocking MPI Comms

	/* Assign receive buffers to the right parts of m, v */

	void *sGhostBck, *sGhostFwd, *rGhostBck, *rGhostFwd;

	if (fIdx & FIELD_M)
	{
		sGhostBck = static_cast<void *> (static_cast<char *> (m) + fSize*n2);
		sGhostFwd = static_cast<void *> (static_cast<char *> (m) + fSize*n3);
		rGhostBck = m;
		rGhostFwd = static_cast<void *> (static_cast<char *> (m) + fSize*(n3 + n2));
	}
	else
	{
		sGhostBck = static_cast<void *> (static_cast<char *> (m2) + fSize*n2);
		sGhostFwd = static_cast<void *> (static_cast<char *> (m2) + fSize*n3);
		rGhostBck = m2;
		rGhostFwd = static_cast<void *> (static_cast<char *> (m2) + fSize*(n3 + n2));
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
/*
void	Scalar::foldField()
{
	int	shift;

	shift = mAlign/fSize;
	printf("Foldfield mAlign=%d, fSize=%d, shift=%d, n2=%d ... ", mAlign, fSize, shift, n2);

	switch (precision)
	{
		case FIELD_DOUBLE:

			if (fieldType == FIELD_SAXION)
			{
				for (size_t iz=0; iz < Lz; iz++)
				{
					memcpy (                    m,                  static_cast<char *>(m) + fSize*n2*(1+iz), fSize*n2);
					memcpy (static_cast<char *>(m) + fSize*(n3+n2), static_cast<char *>(v) + fSize*n2*iz,     fSize*n2);

					for (size_t iy=0; iy < n1/shift; iy++)
						for (size_t ix=0; ix < n1; ix++)
							for (size_t sy=0; sy<shift; sy++)
							{
								size_t oIdx = (iy+sy*(n1/shift))*n1 + ix;
								size_t dIdx = iz*n2 + ((size_t) (iy*n1*shift + ix*shift + sy));

								static_cast<complex<double> *> (m)[dIdx+n2] = static_cast<complex<double> *> (m)[oIdx];
								static_cast<complex<double> *> (v)[dIdx]    = static_cast<complex<double> *> (m)[oIdx+n2+n3];
							}
				}
			} else {
				for (size_t iz=0; iz < Lz; iz++)
				{
					memcpy (                    m,                  static_cast<char *>(m) + fSize*n2*(1+iz), fSize*n2);
					memcpy (static_cast<char *>(m) + fSize*(n3+n2), static_cast<char *>(v) + fSize*n2*iz,     fSize*n2);

					for (size_t iy=0; iy < n1/shift; iy++)
						for (size_t ix=0; ix < n1; ix++)
							for (size_t sy=0; sy<shift; sy++)
							{
								size_t oIdx = (iy+sy*(n1/shift))*n1 + ix;
								size_t dIdx = iz*n2 + ((size_t) (iy*n1*shift + ix*shift + sy));

								static_cast<double *> (m)[dIdx+n2] = static_cast<double *> (m)[oIdx];
								static_cast<double *> (v)[dIdx]    = static_cast<double *> (m)[oIdx+n2+n3];
							}
				}
			}

			break;

		case FIELD_SINGLE:

			if (fieldType == FIELD_SAXION)
			{
				for (size_t iz=0; iz < Lz; iz++)
				{
					memcpy (                    m,                  static_cast<char *>(m) + fSize*n2*(1+iz), fSize*n2);
					memcpy (static_cast<char *>(m) + fSize*(n3+n2), static_cast<char *>(v) + fSize*n2*iz,     fSize*n2);

					for (size_t iy=0; iy < n1/shift; iy++)
						for (size_t ix=0; ix < n1; ix++)
							for (size_t sy=0; sy<shift; sy++)
							{
								size_t oIdx = (iy+sy*(n1/shift))*n1 + ix;
								size_t dIdx = iz*n2 + ((size_t) (iy*n1*shift + ix*shift + sy));

								static_cast<complex<float> *> (m)[dIdx+n2] = static_cast<complex<float> *> (m)[oIdx];
								static_cast<complex<float> *> (v)[dIdx]    = static_cast<complex<float> *> (m)[oIdx+n2+n3];
							}
				}
			} else {
				for (size_t iz=0; iz < Lz; iz++)
				{
					memcpy (                    m,                  static_cast<char *>(m) + fSize*n2*(1+iz), fSize*n2);
					memcpy (static_cast<char *>(m) + fSize*(n3+n2), static_cast<char *>(v) + fSize*n2*iz,     fSize*n2);

					for (size_t iy=0; iy < n1/shift; iy++)
						for (size_t ix=0; ix < n1; ix++)
							for (size_t sy=0; sy<shift; sy++)
							{
								size_t oIdx = (iy+sy*(n1/shift))*n1 + ix;
								size_t dIdx = iz*n2 + ((size_t) (iy*n1*shift + ix*shift + sy));

								static_cast<float *> (m)[dIdx+n2] = static_cast<float *> (m)[oIdx];
								static_cast<float *> (v)[dIdx]    = static_cast<float *> (m)[oIdx+n2+n3];
							}
				}
			}
			break;
	}
	printf("Done!\n");
	return;
}

void	Scalar::unfoldField()
{
	int	shift;

	shift = mAlign/fSize;
	//printf("Unfoldfield mAlign=%d, fSize=%d, shift=%d, n2=%d ... ", mAlign, fSize,shift,n2);

	switch (precision)
	{
		case FIELD_DOUBLE:

			if (fieldType == FIELD_SAXION)
			{
				for (size_t iz=0; iz < Lz; iz++)
				{
					memcpy (                    m,                  static_cast<char *>(m) + fSize*n2*(1+iz), fSize*n2);
					memcpy (static_cast<char *>(m) + fSize*(n3+n2), static_cast<char *>(v) + fSize*n2*iz,     fSize*n2);

					for (size_t iy=0; iy < n1/shift; iy++)
						for (size_t ix=0; ix < n1; ix++)
							for (size_t sy=0; sy<shift; sy++)
							{
								size_t oIdx = iy*n1*shift + ix*shift + sy;
								size_t dIdx = iz*n2 + (iy+sy*(n1/shift))*n1 + ix;

								static_cast<complex<double> *> (m)[dIdx+n2] = static_cast<complex<double> *> (m)[oIdx];
								static_cast<complex<double> *> (v)[dIdx]    = static_cast<complex<double> *> (m)[oIdx+n2+n3];
							}
				}
			} else {
				for (size_t iz=0; iz < Lz; iz++)
				{
					memcpy (                    m,                  static_cast<char *>(m) + fSize*n2*(1+iz), fSize*n2);
					memcpy (static_cast<char *>(m) + fSize*(n3+n2), static_cast<char *>(v) + fSize*n2*iz,     fSize*n2);

					for (size_t iy=0; iy < n1/shift; iy++)
						for (size_t ix=0; ix < n1; ix++)
							for (size_t sy=0; sy<shift; sy++)
							{
								size_t oIdx = iy*n1*shift + ix*shift + sy;
								size_t dIdx = iz*n2 + (iy+sy*(n1/shift))*n1 + ix;

								static_cast<double *> (m)[dIdx+n2] = static_cast<double *> (m)[oIdx];
								static_cast<double *> (v)[dIdx]    = static_cast<double *> (m)[oIdx+n2+n3];
							}
				}
			}
			break;

		case FIELD_SINGLE:

			if (fieldType == FIELD_SAXION)
			{
				for (size_t iz=0; iz < Lz; iz++)
				{
					memcpy (                    m,                  static_cast<char *>(m) + fSize*n2*(1+iz), fSize*n2);
					memcpy (static_cast<char *>(m) + fSize*(n3+n2), static_cast<char *>(v) + fSize*n2*iz,     fSize*n2);

					for (size_t iy=0; iy < n1/shift; iy++)
						for (size_t ix=0; ix < n1; ix++)
							for (size_t sy=0; sy<shift; sy++)
							{
								size_t oIdx = iy*n1*shift + ix*shift + sy;
								size_t dIdx = iz*n2 + (iy+sy*(n1/shift))*n1 + ix;

								static_cast<complex<float> *> (m)[dIdx+n2] = static_cast<complex<float> *> (m)[oIdx];
								static_cast<complex<float> *> (v)[dIdx]    = static_cast<complex<float> *> (m)[oIdx+n2+n3];
							}
				}
			} else {
				for (size_t iz=0; iz < Lz; iz++)
				{
					memcpy (                    m,                  static_cast<char *>(m) + fSize*n2*(1+iz), fSize*n2);
					memcpy (static_cast<char *>(m) + fSize*(n3+n2), static_cast<char *>(v) + fSize*n2*iz,     fSize*n2);

					for (size_t iy=0; iy < n1/shift; iy++)
						for (size_t ix=0; ix < n1; ix++)
							for (size_t sy=0; sy<shift; sy++)
							{
								size_t oIdx = iy*n1*shift + ix*shift + sy;
								size_t dIdx = iz*n2 + (iy+sy*(n1/shift))*n1 + ix;

								static_cast<float *> (m)[dIdx+n2] = static_cast<float *> (m)[oIdx];
								static_cast<float *> (v)[dIdx]    = static_cast<float *> (m)[oIdx+n2+n3];
							}
				}
			}
			break;
	}
	//printf("Done!\n");
	return;
}

void	Scalar::unfoldField2D(const size_t sZ)
{
	//unfolds m(slice[sZ]]) into buffer 1 and v(slice[sZ]) into buffer2
	int	shift;

	shift = mAlign/fSize;
	//printf("MAP: Unfold-2D mAlign=%d, fSize=%d, shift=%d, field = %d ", mAlign, fSize,shift, fieldType == FIELD_SAXION);
	//fflush(stdout);
	switch (precision)
	{
		case FIELD_DOUBLE:
		//printf("Case double n1/shift=%d, shift=%d ...", n1/shift, shift);
		if (fieldType == FIELD_SAXION)
		{
			for (size_t iy=0; iy < n1/shift; iy++)
				for (size_t ix=0; ix < n1; ix++)
					for (size_t sy=0; sy<shift; sy++)
					{
						size_t oIdx = (sZ)*n2 + iy*n1*shift + ix*shift + sy;
						size_t dIdx = (iy+sy*(n1/shift))*n1 + ix;
						//this copies m into buffer 1
						static_cast<complex<double> *> (m)[dIdx] = static_cast<complex<double> *> (m)[oIdx+n2];
						//this copies v into buffer last
						static_cast<complex<double> *> (m)[dIdx+n3+n2] = static_cast<complex<double> *> (v)[oIdx];
					}
		}
		else // FIELD_AXION
		{
			for (size_t iy=0; iy < n1/shift; iy++)
				for (size_t ix=0; ix < n1; ix++)
					for (size_t sy=0; sy<shift; sy++)
					{
						size_t oIdx = (sZ)*n2 + iy*n1*shift + ix*shift + sy;
						size_t dIdx = (iy+sy*(n1/shift))*n1 + ix;
						//this copies m into buffer 1
						static_cast<double *> (m)[dIdx] = static_cast<double *> (m)[oIdx+n2];
						//this copies v into buffer last
						static_cast<double *> (m)[dIdx+n3+n2] = static_cast<double *> (v)[oIdx];
					}
		}
		break;

		case FIELD_SINGLE:
		//printf("Case single n1/shift=%d, shift=%d ...", n1/shift, shift);
		if (fieldType == FIELD_SAXION)
		{
			for (size_t iy=0; iy < n1/shift; iy++)
				for (size_t ix=0; ix < n1; ix++)
					for (size_t sy=0; sy<shift; sy++)
					{
						size_t oIdx = (sZ)*n2 + iy*n1*shift + ix*shift + sy;
						size_t dIdx = (iy+sy*(n1/shift))*n1 + ix;
						//this copies m into buffer 1
						static_cast<complex<float> *> (m)[dIdx] = static_cast<complex<float> *> (m)[oIdx+n2];
						//this copies v into buffer last
						static_cast<complex<float> *> (m)[dIdx+n3+n2] = static_cast<complex<float> *> (v)[oIdx];
					}
		}
		else // FIELD_AXION
		{
			for (size_t iy=0; iy < n1/shift; iy++)
				for (size_t ix=0; ix < n1; ix++)
					for (size_t sy=0; sy<shift; sy++)
					{
						size_t oIdx = (sZ)*n2 + iy*n1*shift + ix*shift + sy;
						size_t dIdx = (iy+sy*(n1/shift))*n1 + ix;
						//this copies m into buffer 1
						static_cast<float *> (m)[dIdx] = static_cast<float *> (m)[oIdx+n2];
						//this copies v into buffer last
						static_cast<float *> (m)[dIdx+n3+n2] = static_cast<float *> (v)[oIdx];
					}
		}
		break;
	}

	return;
}
*/
//	USA M2, ARREGLAR LOWMEM
void	Scalar::prepareCpu(int *window)
{
	if (precision == FIELD_DOUBLE)
	{
		#pragma omp parallel for default(shared) schedule(static)
		for(size_t i=0; i < n3; i++)
			((complex<double> *) m2)[i] = I*(((std::complex<double> *) v)[i]/((std::complex<double> *) m)[i]).imag()*((double) window[i]);
	}
	else
	{
		#pragma omp parallel for default(shared) schedule(static)
		for(size_t i=0; i < n3; i++)
			((complex<float> *) m2)[i] = If*(((std::complex<float> *) v)[i]/((std::complex<float> *) m)[i]).imag()*((float) window[i]);
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
		for(size_t i=0; i < n3; i++)
			((std::complex<double> *) m2)[i] = pow(abs(((std::complex<double> *) m2)[i]/((double) n3)),2);
	}
	else
	{
		#pragma omp parallel for default(shared) schedule(static)
		for(size_t i=0; i < n3; i++)
			((std::complex<float> *) m2)[i] = pow(abs(((std::complex<float> *) m2)[i]/((float) n3)),2);
	}
}

//	USA M2, ARREGLAR LOWMEM
// void	Scalar::thetaz2m2(int *window)
// {
// 	if (precision == FIELD_DOUBLE)
// 	{
// 		#pragma omp parallel for default(shared) schedule(static)
// 		for(size_t i=0; i < n3; i++)
// 			((complex<double> *) m2)[i] = I*(((std::complex<double> *) v)[i]/((std::complex<double> *) m)[i]).imag();
// 	}
// 	else
// 	{
// 		#pragma omp parallel for default(shared) schedule(static)
// 		for(size_t i=0; i < n3; i++)
// 			((complex<float> *) m2)[i] = If*(((std::complex<float> *) v)[i]/((std::complex<float> *) m)[i]).imag();
// 	}
// }

//	USA M2, ARREGLAR LOWMEM
//	COPIES CONFORMAL FIELD AND DERIVATIVE FROM PQ FIELD
void	Scalar::theta2m2()//int *window)
{
	switch (fieldType)
	{
		case FIELD_SAXION:
					if (precision == FIELD_DOUBLE)
					{
						double za = (*z);
						double massfactor = 3.0 * pow(za, nQcd/2 + 1);

						#pragma omp parallel for default(shared) schedule(static)
						for(size_t i=0; i < n3; i++)
						{
						double thetaaux = arg(((std::complex<double> *) m)[i]);
							((complex<double> *) m2)[i] = thetaaux*massfactor*za
																						+ I*( ((((std::complex<double> *) v)[i]/((std::complex<double> *) m)[i]).imag())*za
																						      + thetaaux ) ;
						}
					}
					else
					{
						float zaf = *z ;
						float massfactor = 3.0 * pow(zaf, nQcd/2 + 1);

						#pragma omp parallel for default(shared) schedule(static)
						for(size_t i=0; i < n3; i++)
						{
						float thetaauxf = arg(((std::complex<float> *) m)[i]);
							((complex<float> *) m2)[i] = thetaauxf*massfactor*zaf
																					 + If*( ((((std::complex<float> *) v)[i]/((std::complex<float> *) m)[i]).imag())*zaf
																					      + thetaauxf);
						}
					}
		break;

		case FIELD_AXION:

					if (precision == FIELD_DOUBLE)
					{
						double massfactor = 3.0 * pow((*z), nQcd/2 + 1);

						#pragma omp parallel for default(shared) schedule(static)
						for(size_t i=0; i < n3; i++)
						{
							((complex<double> *) m2)[i] = ((static_cast<double*> (m))[i])*massfactor + I*((static_cast<double*> (v))[i]);
						}
					}
					else
					{
						float massfactor = 3.0 * pow((*z), nQcd/2 + 1);

						#pragma omp parallel for default(shared) schedule(static)
						for(size_t i=0; i < n3; i++)
						{
						float thetaauxf = arg(((std::complex<float> *) m)[i]);
							((complex<float> *) m2)[i] = ((static_cast<float*> (m))[i])*massfactor + If*((static_cast<float*> (v))[i]);
						}
					}
		break;


	}
}

//

//OLD VERSION, THETA
// void	Scalar::theta2m2()//int *window)
// {
//
// 	if (precision == FIELD_DOUBLE)
// 	{
// 		#pragma omp parallel for default(shared) schedule(static)
// 		for(size_t i=0; i < n3; i++)
// 			((complex<double> *) m2)[i] = arg(((std::complex<double> *) m)[i]) + I*(((std::complex<double> *) v)[i]/((std::complex<double> *) m)[i]).imag();//*((double) window[i]);
// 	}
// 	else
// 	{
// 		#pragma omp parallel for default(shared) schedule(static)
// 		for(size_t i=0; i < n3; i++)
// 			((complex<float> *) m2)[i] = arg(((std::complex<float> *) m)[i]) +  If*(((std::complex<float> *) v)[i]/((std::complex<float> *) m)[i]).imag();//*((float) window[i]);
// 	}
// }


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

void	Scalar::fftCpuSpectrum	(int sign)
{
	runFFTSpectrum(sign);
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
/*
			// ----------------------------------------------------------------------
			// 		INITIAL CONDITIONS
			// ----------------------------------------------------------------------


void	Scalar::genConf	(ConfType cType, const size_t parm1, const double parm2)
{

	fflush (stdout);

	switch (cType)
	{
		case CONF_NONE:
		break;
//------------------------------------------------------------------------------
		case CONF_KMAX:		// kMax = parm1, kCrit = parm2

//		if (device == DEV_CPU || device == DEV_XEON)	//	Do this always...
		{
			switch (precision)
			{
				case FIELD_DOUBLE:
					printf ("Generating conf (KMAX) (double prec) ... ");
					fflush (stdout);
					momConf(parm1, parm2);
					printf ("Normalising field ");
					normaliseField(FIELD_M);
					normaCOREField( (double) alpha );
					fflush (stdout);
					printf ("Done!\n");
				break;

				case FIELD_SINGLE:
					printf ("Generating conf (KMAX) (single prec) ... ");
					fflush (stdout);
					momConf(parm1, (float) parm2);
					printf ("Normalising field ");
					normaliseField(FIELD_M);
					normaCOREField( (float) alpha );
					printf ("Done!\n");
					fflush (stdout);
				break;

				default:
					printf("Wrong precision!\n");
					exit(1);
				break;
			}

			printf ("FFT ... ");
			fflush (stdout);
			fftCpu (1);	// FFTW_BACKWARD
			printf ("Done!\n");
			fflush (stdout);

		}

		exchangeGhosts(FIELD_M);

		break;
//------------------------------------------------------------------------------
		case CONF_TKACHEV:		// kMax = parm1, kCrit = parm2

//		if (device == DEV_CPU || device == DEV_XEON)	//	Do this always...
		{
			switch (precision)
			{
				case FIELD_DOUBLE:
				printf ("Generating conf (TKA) (double prec) ... ");
				fflush (stdout);
				momConf(parm1, parm2);

				break;

				case FIELD_SINGLE:
				printf ("Generating conf (TKA) (single prec) ... ");
				fflush (stdout);
				momConf(parm1, (float) parm2);

				break;

				default:
				printf("Wrong precision!\n");
				exit(1);

				break;
			}
			printf ("FFT ... ");
			fflush (stdout);
			fftCpu (1);	// FFTW_BACKWARD
			printf ("Done!\n");
			fflush (stdout);
		}
		exchangeGhosts(FIELD_M);
		break;
//------------------------------------------------------------------------------
		case CONF_SMOOTH:	// iter = parm1, alpha = parm2

		switch (device)
		{
			case	DEV_XEON:
			case	DEV_CPU:

			printf ("Generating random conf (SMOOTH) ...");
			fflush (stdout);
			randConf ();
			printf("Smoothing (sIter=%d) ... ",parm1);
			fflush (stdout);
			smoothConf (parm1, parm2);
			printf("Done !\n ");
			fflush (stdout);
			printf ("Normalising field ");
			switch (precision)
			{
				case FIELD_DOUBLE:
					printf ("(double prec) ... ");
					fflush (stdout);
					normaliseField(FIELD_M);
					normaCOREField( (double) alpha );
					printf ("Done!\n");
					fflush (stdout);
				break;

				case FIELD_SINGLE:
				printf ("(single prec) ... ");
				fflush (stdout);
				normaliseField(FIELD_M);
				normaCOREField( (float) alpha );
				printf ("Done!\n");
				fflush (stdout);
			break;

			case	DEV_GPU:
			break;
			default:
			printf("Wrong precision! \n");
			exit(1);
			break;
		}
//------------------------------------------------------------------------------
		break;

		default:

		printf("Configuration type not recognized\n");
		exit(1);

		break;

	} // END SwITCH TYPE KMAX vs SMOOTH ... & TKACHEV
//------------------------------------------------------------------------------
	if (cType != CONF_NONE)
	{
		// printf ("Normalising field ");
		// //JAVIER normalisation
		// switch (precision)
		// {
		// 	case FIELD_DOUBLE:
		//
		// 	printf ("(double prec) ... ");
		// 	fflush (stdout);
		//
		// 	normaliseField(FIELD_M);
		// 	normaCOREField( (double) alpha );
		//
		// 	printf ("Done!\n");
		// 	fflush (stdout);
		// 	break;
		//
		// 	case FIELD_SINGLE:
		//
		// 	printf ("(single prec) ... ");
		// 	fflush (stdout);
		//
		// 	normaliseField(FIELD_M);
		// 	normaCOREField( (float) alpha );
		//
		// 	printf ("Done!\n");
		// 	fflush (stdout);
		// 	break;
		//
		// 	default:
		// 	printf("Wrong precision! \n");
		// 	exit(1);
		// 	break;
		}

		//JAVIER
		printf("Copying m to v ... ");
		fflush (stdout);

		memcpy (v, static_cast<char *> (m) + fSize*n2, fSize*n3);

		printf("Scaling m to mu=z*m ... ");
		fflush (stdout);

//		scaleField (FIELD_M, *z);
		printf("Done!\n");
		fflush (stdout);
	}
}
/*
				//----------------------------------------------------------------------
				//		FUNCTIONS FOR INIT. COND.
				//----------------------------------------------------------------------


void	Scalar::randConf ()
{
	int	maxThreads = omp_get_max_threads();
	int	*sd;

	trackAlloc((void **) &sd, sizeof(int)*maxThreads);

	std::random_device seed;		// Totally random seed coming from memory garbage

	for (int i=0; i<maxThreads; i++)
		sd[i] = seed();

	switch (precision)
	{
		case FIELD_DOUBLE:
		printf(" (double prec) ");
		#pragma omp parallel default(shared)
		{
			int nThread = omp_get_thread_num();

			//JAVIER commented next line, it seems to work
			//printf	("Thread %d got seed %d\n", nThread, sd[nThread]);

			std::mt19937_64 mt64(sd[nThread]);		// Mersenne-Twister 64 bits, independent per thread
			//JAVIER included negative values
			std::uniform_real_distribution<double> uni(-1.0, 1.0);

			#pragma omp for schedule(static)	// This is NON-REPRODUCIBLE, unless one thread is used. Alternatively one can fix the seeds
			for (size_t idx=n2; idx<n2+n3; idx++)
				static_cast<complex<double>*> (m)[idx]   = complex<double>(uni(mt64), uni(mt64));
		}
		//JAVIER control print
		printf(" Done! ");
		//printf	("CHECK: %d, %d+1 got (%lf,%lf)  \n", 2*n2,2*n2, static_cast<double *> (m)[2*n2], static_cast<double *> (m)[2*n2+1]);

		break;

		case FIELD_SINGLE:
		printf(" (single prec) ");
		#pragma omp parallel default(shared)
		{
			int nThread = omp_get_thread_num();

			std::mt19937_64 mt64(sd[nThread]);		// Mersenne-Twister 64 bits, independent per thread
			//JAVIER included negative values
			std::uniform_real_distribution<float> uni(-1.0, 1.0);

			#pragma omp for schedule(static)	// This is NON-REPRODUCIBLE, unless one thread is used. Alternatively one can fix the seeds
			for (size_t idx=n2; idx<n2+n3; idx++)
				static_cast<complex<float>*> (m)[idx]   = complex<float>(uni(mt64), uni(mt64));

			//printf ("Thread %d finished loop ", nThread);
			fflush (stdout);
		}
		printf(" Done! ");
		break;

		default:

		printf("Unrecognized precision\n");
		trackFree((void **) &sd, ALLOC_TRACK);
		exit(1);

		break;
	}

	trackFree((void **) &sd, ALLOC_TRACK);
}// End Scalar::randConf ()


template<typename Float>
void	Scalar::iteraField(const size_t iter, const Float alpha)
{
	const Float One = 1.;
	const Float OneSixth = (1./6.);

	exchangeGhosts(FIELD_M);

	complex<Float> *mCp = static_cast<complex<Float>*> (m);
	complex<Float> *vCp = static_cast<complex<Float>*> (v);
	//JAVIER
	//printf("smoothing check m[0]= (%lf,%lf)\n",  ((Float *) m)[n2] , ((Float *) m)[n2] );
	//printf("the same? ????? m[0]= (%lf,%lf)\n",  ((Float *) mCp)[n2] , ((Float *) mCp)[n2] ); yes

	for (size_t it=0; it<iter; it++)
	{
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)
			for (size_t idx=0; idx<n3; idx++)
			{
				size_t iPx, iMx, iPy, iMy, iPz, iMz, X[3];
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
						iPx = idx + 1;
						iMx = idx - 1;
					}
				}

				if (X[1] == 0)
				{
					iPy = idx + n1;
					iMy = idx + n2 - n1;
				} else {
					if (X[1] == n1 - 1)
					{
						iPy = idx - n2 + n1;
						iMy = idx - n1;
					} else {
						iPy = idx + n1;
						iMy = idx - n1;
					}
				}

				iPz = idx + n2;
				iMz = idx - n2;
				//Uses v to copy the smoothed configuration
				vCp[idx]   = alpha*mCp[idx+n2] + OneSixth*(One-alpha)*(mCp[iPx+n2] + mCp[iMx+n2] + mCp[iPy+n2] + mCp[iMy+n2] + mCp[iPz+n2] + mCp[iMz+n2]);
				vCp[idx]   = vCp[idx]/abs(vCp[idx]);
			}
		}
		//Copies v to m
		memcpy (static_cast<char *>(m) + fSize*n2, v, fSize*n3);
		exchangeGhosts(FIELD_M);

		//printf("smoothing check m[0]= (%lf,%lf)\n",  real(((complex<double> *) m)[n2]), real(mCp[n2]) ); both give the same
		//printf("smoothing check m[0],m[1]= (%lf,%lf), (%lf,%lf)\n",  real(mCp[n2]), imag(mCp[n2]),real(mCp[n2+1]), imag(mCp[n2+1]) );
	}//END iteration loop
		//printf("smoothing check m[0],m[1]= (%lf,%lf), (%lf,%lf)\n",  real(mCp[n2]), imag(mCp[n2]),real(mCp[n2+1]), imag(mCp[n2+1]) );
}//END Scalar::iteraField

void	Scalar::smoothConf (const size_t iter, const double alpha)
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
	const Float Twop = 2.0*M_PI;

	complex<Float> *fM;

	if (!lowmem)
		fM = static_cast<complex<Float>*> (m2);
	else
		fM = static_cast<complex<Float>*> (static_cast<void*>(static_cast<char*>(m) + fSize*n2));

	int	maxThreads = omp_get_max_threads();
	int	*sd;

	trackAlloc((void **) &sd, sizeof(int)*maxThreads);


	std::random_device seed;		// Totally random seed coming from memory garbage

	for (int i=0; i<maxThreads; i++)
		sd[i] = 0;//seed();

	//printf("kMax,Tz,Lz,nSplit,commRank():%d,%lu,%lu,%lu,%d\n", kMax, Tz, Lz, nSplit,commRank());
	printf("kMax,Tz,Lz,nSplit,commRank():%d,%d,%d,%d,%d\n", kMax, Tz, Lz, nSplit,commRank());

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

			//printf("Thread %d got pz=%d -- py-range (%d,%d)\n", nThread,pz,-kMax,kMax);


			for(int py = -kMax; py <= kMax; py++)
			{
				for(int px = -kMax; px <= kMax; px++)
				{
					size_t idx  = n2 + ((px + n1)%n1) + ((py+n1)%n1)*n1 + ((pz+Tz)%Tz)*n2 - commRank()*n3;
					int modP = px*px + py*py + pz*pz;

					//printf("modP=%d,idx=%lu,idx=%d =%d\n",modP,idx,idx,kMax*kMax);

					if (modP <= kMax*kMax)
					{
						Float mP = sqrt(((Float) modP))/((Float) kCrit);
						Float vl = Twop*(uni(mt64));
						Float sc = (modP == 0) ? 1.0 : sin(mP)/mP;

						fM[idx] = complex<Float>(cos(vl), sin(vl))*sc;
						//printf("oz=%d,Tz=%lu,(px,py,pz)=(%d,%d,%d),idx=%lu, fM[idx]=(%f,%f)\n",oz,Tz,px,py,pz,idx,real(fM[idx]),imag(fM[idx]));

					}
				} //END  px loop
			}		//END  py loop
		}//END oz FOR
	}

	trackFree((void **) &sd, ALLOC_TRACK);
}
*/

void	Scalar::setField (FieldType fType)
{
	switch (fType)
	{
		case FIELD_AXION:
			if (fieldType == FIELD_SAXION)
			{
				trackFree(&v, ALLOC_ALIGN);

				switch (precision)
				{
					case FIELD_SINGLE:
					v = static_cast<void*>(static_cast<float*>(m) + 2*n2 + n3);
					break;

					case FIELD_DOUBLE:
					v = static_cast<void*>(static_cast<double*>(m) + 2*n2 + n3);
					break;
				}

				fSize /= 2;

				const size_t	mBytes = v3*fSize;

				//IF low mem was used before, it creates m2 COMPLEX
				if (lowmem)
				{
					#ifdef	USE_XEON
					alignAlloc ((void**) &m2X, mAlign, 2*mBytes);
					m2  = m2X;
					#else
					alignAlloc ((void**) &m2, mAlign, 2*mBytes);
					#endif
					//initFFTSpectrum(m2, n1, Tz, precision, lowmem);
					initFFTSpectrum(m2, n1, Tz, precision, 0);
				}
				//IF no lowmem was used, we kill m2 complex and create m2 real ... not used
				// #ifdef	USE_XEON
				// if (!lowmem)
				// {
				// 	trackFree(&m2X, ALLOC_ALIGN);
				// 	m2 = m2X = NULL;
				// }
				// alignAlloc ((void**) &m2X, mAlign, mBytes);
				// m2  = m2X;
				// #else
				// if (!lowmem)
				// {
				// 	trackFree(&m2, ALLOC_ALIGN);
				// 	m2 = NULL;
				// }
				// alignAlloc ((void**) &m2, mAlign, mBytes);
				// #endif

			}
			break;

		case	FIELD_SAXION:
		break;
	}

	fieldType = fType;
}

//void	Scalar::writeENERGY (double zzz, FILE *enwrite)
void	Scalar::writeENERGY (double zzz, FILE *enwrite, double &Gfr, double &Gft, double &Vfr, double &Vft, double &Kfr, double &Kft) // TEST
{
	switch	(precision)
	{
		case	FIELD_DOUBLE:
		{
//			ENERGY (zzz, enwrite);
			ENERGY (zzz, enwrite, Gfr, Gft, Vfr, Vft, Kfr, Kft); // TEST
		}
		break;

		case	FIELD_SINGLE:
		{
			//float Gr, Gt, Vr, Vt, Kr, Kt;  // TEST
			//
			// ENERGY (static_cast<float>(zzz), enwrite, Gr, Gt, Vr, Vt, Kr, Kt); // TEST
			// Gfr = static_cast<double>(Gr); // TEST
			// Gft = static_cast<double>(Gt); // TEST;
			// Vfr = static_cast<double>(Vr); // TEST;
			// Vft = static_cast<double>(Vt); // TEST;
			// Kfr = static_cast<double>(Kr); // TEST;
			// Kft = static_cast<double>(Kt); // TEST;
			//Alternative
			ENERGY2 (static_cast<float>(zzz), enwrite, Gfr, Gft, Vfr, Vft, Kfr, Kft); // TEST

		}
		break;

		default:
		printf("Unrecognized precision\n");
		exit(1);
		break;
	}
}


//JAVIER ENERGY
template<typename Float>
//void	Scalar::ENERGY(const Float zz, FILE *enWrite)
void	Scalar::ENERGY(const Float zz, FILE *enWrite, Float &Grho, Float &Gtheta, Float &Vrho, Float &Vtheta, Float &Krho, Float &Ktheta) // TEST
{
	int	shift;
	shift = mAlign/fSize;

	const Float deltaa2 = pow(sizeL/sizeN,2) ;

	exchangeGhosts(FIELD_M);

	complex<Float> *mCp = static_cast<complex<Float>*> (m);
	complex<Float> *vCp = static_cast<complex<Float>*> (v);

	//printf("ENERGY CALCULATOR\n");
	fflush (stdout);
	//LEAVES BOUNDARIES OUT OF THE LOOP FOR SIMPLICITY

	//SUM variables
	//Float Vrho1 = 0, Vtheta1 = 0, Krho1 = 0, Ktheta1 = 0, Grho1 = 0, Gtheta1=0;
	double Vrho1 = 0, Vtheta1 = 0, Krho1 = 0, Ktheta1 = 0, Grho1 = 0, Gtheta1=0; // TEST

	const Float invz	= 1.0/zz;
	const Float LLzz2 = LL*zz*zz/4.0 ;
	const Float z9QCD = 9.0*pow(zz,nQcd+2) ;



		#pragma omp parallel for default(shared) schedule(static) reduction(+:Vrho1,Vtheta1, Krho1, Ktheta1, Grho1, Gtheta1)
		for (size_t iz=0; iz < Lz; iz++)
		{
			Float modul, modfac	;
			size_t idd	;

			for (size_t iy=0; iy < n1/shift; iy++)
				for (size_t ix=0; ix < n1; ix++)
					for (size_t sy=0; sy<shift; sy++)
					{

//unfolded coordinates	//size_t oIdx = (iy+sy*(n1/shift))*n1 + ix;
//folded coordinates		//size_t dIdx = iz*n2 + ((size_t) (iy*n1*shift + ix*shift + sy));

					size_t dIdx = iz*n2 + ((size_t) (iy*n1*shift + ix*shift + sy));


					modul = abs(mCp[dIdx+n2]);
					modfac = modul*modul*invz*invz;
					//Vrho misses LLzz2 factor
					Vrho1 	+= pow(modfac-1.0,2)	;
					//Vtheta misses z9QCD factor
					Vtheta1 += 1-real(mCp[dIdx+n2])*invz	;
					//Krho1 misses 0.5 factor
					Krho1 += modfac*pow(real(vCp[dIdx]/mCp[dIdx+n2])-invz,2);
					//Krho1 misses 0.5 factor
					Ktheta1 += modfac*pow(imag(vCp[dIdx]/mCp[dIdx+n2]),2);

					//Grho1 misses a factor 3*0.5/4 delta^2
					//only computed in the z direction ... easy!
					Grho1 += modfac*pow(real((mCp[dIdx+2*n2]-mCp[dIdx])/mCp[dIdx+n2]),2);
					Gtheta1 += modfac*pow(imag((mCp[dIdx+2*n2]-mCp[dIdx])/mCp[dIdx+n2]),2);

					}
		}
	//RENORMALISE
	//Vrho misses LLzz2 factor
	Vrho1 	*= LLzz2/n3	;
	//Vtheta misses z9QCD factor
	Vtheta1 *= z9QCD/n3	;
	//Krho1 misses 0.5 factor
	Krho1 *= 0.5/n3;
	//Krho1 misses 0.5 factor
	Ktheta1 *= 0.5/n3;

	//Grho1 misses a factor 3*0.5/delta^2
	Grho1 *= 3.0*0.125/(deltaa2*n3);
	Gtheta1 *= 3.0*0.125/(deltaa2*n3);

	Vrho = Vrho1; Vtheta = Vtheta1; Krho = Krho1; Ktheta = Ktheta1; Grho = Grho1; Gtheta = Gtheta1;

/*
	fprintf(enWrite,  "%f %f %f %f %f %f %f \n", zz, Vrho1, Vtheta1, Krho1, Ktheta1, Grho1, Gtheta1);
	printf("ENERGY & PRINTED - - - Vr=%f Va=%f Kr=%f Ka=%f Gr=%f Ga=%f \n", Vrho1, Vtheta1, Krho1, Ktheta1, Grho1, Gtheta1);
	fflush (stdout);
*/
}

//JAVIER ENERGY
template<typename Float>
//void	Scalar::ENERGY(const Float zz, FILE *enWrite)
void	Scalar::ENERGY2(const Float zz, FILE *enWrite, double &Grho, double &Gtheta, double &Vrho, double &Vtheta, double &Krho, double &Ktheta) // TEST
{
	int	shift;
	shift = mAlign/fSize;

	const Float deltaa2 = pow(sizeL/sizeN,2) ;

	exchangeGhosts(FIELD_M);

	complex<Float> *mCp = static_cast<complex<Float>*> (m);
	complex<Float> *vCp = static_cast<complex<Float>*> (v);

	//printf("ENERGY CALCULATOR\n");
	fflush (stdout);
	//LEAVES BOUNDARIES OUT OF THE LOOP FOR SIMPLICITY

	//SUM variables
	//Float Vrho1 = 0, Vtheta1 = 0, Krho1 = 0, Ktheta1 = 0, Grho1 = 0, Gtheta1=0;
	Float Vrho1 = 0, Vtheta1 = 0, Krho1 = 0, Ktheta1 = 0, Grho1 = 0, Gtheta1=0; // TEST

	const Float invz	= 1.0/zz;
	const Float LLzz2 = LL*zz*zz/4.0 ;
	const Float z9QCD = 9.0*pow(zz,nQcd+2) ;



		#pragma omp parallel for default(shared) schedule(static) reduction(+:Vrho1,Vtheta1, Krho1, Ktheta1, Grho1, Gtheta1)
		for (size_t iz=0; iz < Lz; iz++)
		{
			Float modul, modfac	;
			size_t idd	;

			for (size_t iy=0; iy < n1/shift; iy++)
				for (size_t ix=0; ix < n1; ix++)
					for (size_t sy=0; sy<shift; sy++)
					{

//unfolded coordinates	//size_t oIdx = (iy+sy*(n1/shift))*n1 + ix;
//folded coordinates		//size_t dIdx = iz*n2 + ((size_t) (iy*n1*shift + ix*shift + sy));

					size_t dIdx = iz*n2 + ((size_t) (iy*n1*shift + ix*shift + sy));


					modul = abs(mCp[dIdx+n2]);
					modfac = modul*modul*invz*invz;
					//Vrho misses LLzz2 factor
					Vrho1 	+= pow(modfac-1.0,2)	;
					//Vtheta misses z9QCD factor
					Vtheta1 += 1-real(mCp[dIdx+n2])*invz	;
					//Krho1 misses 0.5 factor
					Krho1 += modfac*pow(real(vCp[dIdx]/mCp[dIdx+n2])-invz,2);
					//Krho1 misses 0.5 factor
					Ktheta1 += modfac*pow(imag(vCp[dIdx]/mCp[dIdx+n2]),2);

					//Grho1 misses a factor 3*0.5/4 delta^2
					//only computed in the z direction ... easy!
					Grho1 += modfac*pow(real((mCp[dIdx+2*n2]-mCp[dIdx])/mCp[dIdx+n2]),2);
					Gtheta1 += modfac*pow(imag((mCp[dIdx+2*n2]-mCp[dIdx])/mCp[dIdx+n2]),2);

					}
		}
	//RENORMALISE
	//Vrho misses LLzz2 factor
	Vrho1 	*= LLzz2/n3	;
	//Vtheta misses z9QCD factor
	Vtheta1 *= z9QCD/n3	;
	//Krho1 misses 0.5 factor
	Krho1 *= 0.5/n3;
	//Krho1 misses 0.5 factor
	Ktheta1 *= 0.5/n3;

	//Grho1 misses a factor 3*0.5/delta^2
	Grho1 *= 3.0*0.125/(deltaa2*n3);
	Gtheta1 *= 3.0*0.125/(deltaa2*n3);

	Vrho = Vrho1; Vtheta = Vtheta1; Krho = Krho1; Ktheta = Ktheta1; Grho = Grho1; Gtheta = Gtheta1;
/*
	fprintf(enWrite,  "%f %f %f %f %f %f %f \n", zz, Vrho1, Vtheta1, Krho1, Ktheta1, Grho1, Gtheta1);
	printf("ENERGY & PRINTED - - - Vr=%f Va=%f Kr=%f Ka=%f Gr=%f Ga=%f \n", Vrho1, Vtheta1, Krho1, Ktheta1, Grho1, Gtheta1);
	fflush (stdout);
*/
}

// ----------------------------------------------------------------------
// 		FUNCTION FOR AXION ENERGY ; MAKES AN ENERGY MAP IN M2
// ----------------------------------------------------------------------
void	Scalar::writeMAPTHETA (double zzz, const int index, void *contbin, int numbins)//, FILE *enwrite, double &Gfr, double &Gft, double &Vfr, double &Vft, double &Kfr, double &Kft) // TEST
{
	double maxcontrast ;
	switch	(precision)
	{
		case	FIELD_DOUBLE:
		{
		}
		break;

		case	FIELD_SINGLE:
		{
			energymapTheta (static_cast<float>(zzz), index, contbin, numbins); // TEST

		}
		break;

		default:
		printf("Unrecognized precision\n");
		exit(1);
		break;
	}
	return ;
}

template<typename Float>
//void	Scalar::ENERGY(const Float zz, FILE *enWrite)
void	Scalar::energymapTheta(const Float zz, const int index, void *contbin, int numbins)
{
	// THIS TEMPLATE IS TO BE CALLED UNFOLDED
	// COPIES THE CONTRAST INTO THE REAL PART OF M2 (WHICH IS COMPLEX)
	// TO USE THE POWER SPECTRUM AFTER
	// 	FILES DENSITY CONTRAST
	char stoCON[256];
	sprintf(stoCON, "out/con/con-%05d.txt", index);
	FILE *file_con ;
	file_con = NULL;
	file_con = fopen(stoCON,"w+");
	fprintf(file_con,  "# %d %f %f %f \n", sizeN, sizeL, sizeL/sizeN, zz );

	// 	CONSTANTS
	const Float deltaa2 = pow(sizeL/sizeN,2.)*2. ;
	const Float invz	= 1.0/(*z);
	const Float z9QCD4 = 9.0*pow((*z),nQcd+4.) ;

	//	AUX VARIABLES
	Float maxi = 0.;
	Float maxibin = 0.;
	double toti = 0.;

	exchangeGhosts(FIELD_M);


	if(fieldType == FIELD_AXION)
	{
		Float *mTheta = static_cast<Float*> (m);
		Float *mVeloc = static_cast<Float*> (v);
		// REAL VERSION
		//Float *mCONT = static_cast<Float*> (m2);
		// COMPLEX VERSION
		complex<Float> *mCONT = static_cast<complex<Float>*> (m2);
		//printf("ENERGY map theta \n");

		//SUM variables
		//Float Vrho1 = 0, Vtheta1 = 0, Krho1 = 0, Ktheta1 = 0, Grho1 = 0, Gtheta1=0;
		//Float Vrho1 = 0, Vtheta1 = 0, Krho1 = 0, Ktheta1 = 0, Grho1 = 0, Gtheta1=0; // TEST
		#pragma omp parallel for default(shared) schedule(static) reduction(max:maxi), reduction(+:toti)
		for (int iz=0; iz < Lz; iz++)
		{
			Float acu , grad ;
			size_t idx, idaux ;
			for (int iy=0; iy < n1; iy++)
			{
				int iyP = (iy+1)%n1;
				int iyM = (iy-1+n1)%n1;
				for (int ix=0; ix < n1; ix++)
				{
					int ixP = (ix+1)%n1;
					int ixM = (ix-1+n1)%n1;

					idx = ix + iy*n1+(iz+1)*n2 ;
					//KINETIC + POTENTIAL
					acu = mVeloc[idx-n2]*mVeloc[idx-n2]/2. + z9QCD4*(1.0-cos(mTheta[idx]*invz)) ;
					//GRADIENTS
					idaux = ixP + iy*n1+(iz+1)*n2 ;
					grad = pow(mTheta[idaux]-mTheta[idx],2);
					idaux = ixM + iy*n1+(iz+1)*n2 ;
					grad += pow(mTheta[idaux]-mTheta[idx],2);
					idaux = ix + iyP*n1+(iz+1)*n2 ;
					grad += pow(mTheta[idaux]-mTheta[idx],2);
					idaux = ix + iyM*n1+(iz+1)*n2 ;
					grad += pow(mTheta[idaux]-mTheta[idx],2);
					grad += pow(mTheta[idx+n2]-mTheta[idx],2);
					grad += pow(mTheta[idx-n2]-mTheta[idx],2);
					mCONT[idx] = acu + grad/deltaa2 ;
					//mCONT[idx] = acu ;

					toti += (double) mCONT[idx].real() ;
					if (mCONT[idx].real() > maxi)
					{
						maxi = mCONT[idx].real() ;
					}
				} //END X LOOP
			} //END Y LOOP
		} //END Z LOOP



		toti = toti/n3 ;
		#pragma omp parallel for default(shared) schedule(static)
		for (size_t idx=n2; idx < n3+n2; idx++)
		{
			mCONT[idx] = mCONT[idx].real()/toti	;
		}
		maxi = (Float) maxi/toti ;

		if (maxi >100.)
		{
			maxibin = 100.;
		}
		else
		{
			maxibin = maxi ;
		}
		//BIN delta from 0 to maxi+1
		size_t auxintarray[numbins] ;

		for(size_t bin = 0; bin < numbins ; bin++)
		{
		(static_cast<double *> (contbin))[bin] = 0.;
		auxintarray[bin] = 0;
		}
		// 	SAVE AVERAGE
		//	MAXIMUM VALUE OF ENERGY CONTRAST
		//	MAXIMUM VALUE TO BE BINNED
		(static_cast<double *> (contbin))[0] = toti;
		(static_cast<double *> (contbin))[1] = maxi;
		(static_cast<double *> (contbin))[2] = maxibin;



		Float norma = (maxi)/(numbins-3) ;
		for(size_t i=n2; i < n3+n2; i++)
		{
			int bin;
			bin = (mCONT[i].real()/norma)	;
			//(static_cast<double *> (contbin))[bin+2] += 1. ;
			if (bin<numbins)
			{
				auxintarray[bin] +=1;
			}
		}



		#pragma omp parallel for default(shared) schedule(static)
		for(size_t bin = 0; bin < numbins-3 ; bin++)
		{
			(static_cast<double *> (contbin))[bin+3] = (double) auxintarray[bin];
		}



		//PRINT 3D maps
		#pragma omp parallel for default(shared) schedule(static)
		for (size_t idx = 0; idx < n3; idx++)
		{
			size_t ix, iy, iz;
				if (mCONT[n2+idx].real() > 5.)
				{
					iz = idx/n2 ;
					iy = (idx%n2)/n1 ;
					ix = (idx%n2)%n1 ;
					#pragma omp critical
					{
						fprintf(file_con,   "%d %d %d %f \n", ix, iy, iz, mCONT[n2+idx].real() ) ;
					}
				}
		}



	}
	else // FIELD_SAXION
	{
			// DO NOTHING
	}


	printf("%d/?? - - - ENERGYdens = %f Max contrast = %f\n ", index, toti, maxi);
	fflush (stdout);
	return ;
}


// ----------------------------------------------------------------------
// 		FUNCTIONS FOR MAX THETA [works but integrated with next]
// ----------------------------------------------------------------------

double	Scalar::maxtheta()//int *window)
{
	double mymaxd = 0.;
	if (precision == FIELD_DOUBLE)
	{

		double tauxd;
		#pragma omp parallel for reduction(max:mymaxd)
		for(size_t i=0; i < n3; i++)
		{
			if(fieldType == FIELD_SAXION)
			{
				tauxd = abs(arg(((complex<double> *) m)[i]));
			}
			else // FILD_AXION
			{
				tauxd = abs(((double *) m)[i]/(*z));
			}
			if( tauxd > mymaxd )
			{
				mymaxd = tauxd ;
			}
		}
	}
	else // PRECISION SINGLE
	{
		float mymax = 0.f, taux;
		#pragma omp parallel for reduction(max:mymax)
		for(size_t i=0; i < n3; i++)
		{
			if(fieldType == FIELD_SAXION)
			{
				taux = abs(arg(((complex<float> *) m)[i]));
			}
			else // FILD_AXION
			{
				taux = abs(((float *) m)[i]/(*z));
			}
			if( taux > mymax )
			{
				mymax = taux ;
			}
		}
		mymaxd = (double) mymax;
	}
	return mymaxd ;
}

//----------------------------------------------------------------------
//		FUNCTION FOR THETA DISTRIBUTION [and max]
//----------------------------------------------------------------------
// 		THIS VERSION GIVES MAX IN EACH SLICE (used to fix the theta transition problem)
//
// double	Scalar::thetaDIST(void * thetabin)//int *window)
// {
//
// 	for(size_t iz = 0; iz < Lz+2; iz++)
// 	{
// 	(static_cast<double *> (thetabin))[iz] = 0.;
// 	}
//
// 	if (precision == FIELD_DOUBLE)
// 	{
// 			//double tauxd;
// 			if(fieldType == FIELD_SAXION)
// 			{
// 				#pragma omp parallel for //reduction(max:mymaxd)
// 				for(size_t iz=0; iz < Lz+2; iz++)
// 				{
// 					double tauxd ;
// 					size_t n2shift = iz*n2;
// 					for(size_t i=0; i < n2; i++)
// 					{
// 						tauxd = abs(arg(((complex<double> *) m)[i+n2shift]));
// 						if( tauxd > (static_cast<double *> (thetabin))[iz] )
// 						{
// 							(static_cast<double *> (thetabin))[iz] = tauxd ;
// 						}
// 					}
// 				}
// 			}
// 			else // FIELD_AXION // recall c_theta is saved
// 			{
// 				#pragma omp parallel for //reduction(max:mymaxd)
// 				for(size_t iz=0; iz < Lz+2; iz++)
// 				{
// 					double tauxd ;
// 					size_t n2shift = iz*n2;
// 					for(size_t i=0; i < n2; i++)
// 					{
// 						tauxd = abs(((double *) m)[i + n2shift]/(*z));
// 						if( tauxd > (static_cast<double *> (thetabin))[iz] )
// 						{
// 							(static_cast<double *> (thetabin))[iz] = tauxd ;
// 						}
// 					}
// 				}
// 			}
// 	}
// 	else // PRECISION SINGLE
// 	{
// 		if(fieldType == FIELD_SAXION)
// 		{
// 			#pragma omp parallel for //reduction(max:mymax)
// 			for(size_t iz=0; iz < Lz+2; iz++)
// 			{
// 				float taux;
// 				size_t n2shift = iz*n2;
// 				for(size_t i=0; i < n2; i++)
// 				{
// 					taux = abs(arg(((complex<float> *) m)[i+n2shift]));
// 					if( (double) taux > (static_cast<double *> (thetabin))[iz] )
// 					{
// 						(static_cast<double *> (thetabin))[iz] = (double) taux ;
// 					}
// 				}
// 			}
// 		}
// 		else	//FIELD_AXION
// 		{
// 			#pragma omp parallel for //reduction(max:mymax)
// 			for(size_t iz=0; iz < Lz+2; iz++)
// 			{
// 				double taux;
// 				size_t n2shift = iz*n2;
// 				for(size_t i=0; i < n2; i++)
// 				{
// 					taux = abs(((float *) m)[i+n2shift]/(*z));
// 					if( (double) taux > (static_cast<double *> (thetabin))[iz] )
// 					{
// 						(static_cast<double *> (thetabin))[iz] = (double) taux ;
// 					}
// 				}
// 			}
// 		}
// 	}
// 	double mymaxd = 0.;
// 	for(size_t iz = 1; iz < Lz+1; iz++)
// 	{
// 		if((static_cast<double *> (thetabin))[iz]>mymaxd)
// 		{
// 			mymaxd = (static_cast<double *> (thetabin))[iz];
// 		}
// 	}
//
// 	return mymaxd ;
// }

//----------------------------------------------------------------------
//		FUNCTION FOR THETA DISTRIBUTION [and max]
//----------------------------------------------------------------------
//		BINS THE DISTRIBUTION OF THETA

double	Scalar::thetaDIST(int numbins, void * thetabin)//int *window)
{
	double thetamaxi = maxtheta();

//	printf("hallo von inside %f\n", thetamaxi);

	double n2p = numbins/thetamaxi;

	for(size_t i = 0; i < numbins ; i++)
	{
	(static_cast<double *> (thetabin))[i] = 0.;
	}

	if (precision == FIELD_DOUBLE)
	{
			if(fieldType == FIELD_SAXION)
			{
	//			#pragma omp parallel for default(shared)
				for(size_t i=0; i < n3 ; i++)
				{
					int bin;
					bin =  n2p*abs(arg(((complex<double> *) m)[i+n2]));
					(static_cast<double *> (thetabin))[bin] += 1.;
				}
			}
			else	//FIELD_AXION
			{
	//			#pragma omp parallel for default(shared)
				for(size_t i=0; i < n3; i++)
				{
					int bin;
					bin = n2p*abs(((double *) m)[i+n2]/(*z));
					(static_cast<double *> (thetabin))[bin] += 1. ;
				}
			}
	}
	else // PRECISION SINGLE
	{
		float n2pf = numbins/thetamaxi;

		if(fieldType == FIELD_SAXION)
		{
//			#pragma omp parallel for default(shared)
			for(size_t i=0; i < n3 ; i++)
			{
				int bin;
				bin =  n2pf*abs(arg(((complex<float> *) m)[i+n2]));
				(static_cast<double *> (thetabin))[bin] += 1.;
			}
		}
		else	//FIELD_AXION
		{
//			#pragma omp parallel for default(shared)
			for(size_t i=0; i < n3; i++)
			{
				int bin;
				bin = n2pf*abs(((float *) m)[i+n2]/(*z));
				(static_cast<double *> (thetabin))[bin] += 1. ;
			}
		}
	}

	// #pragma omp critical
	// {
	// 	for(int n=0; n<numbins; n++)
	// 	{
	// 		static_cast<double*>(thetabin)[n] += thetabin_private[n];
	// 	}
	// }

//}//END PARALLEL

	return thetamaxi ;
}

//--------------------------------------------------------------------
//		DENS M2 -> M
//--------------------------------------------------------------------

//	FOR FINAL OUTPUT, COPIES M2 INTO M

void	Scalar::denstom()//int *window)
{
	double thetamaxi = maxtheta();

//	printf("hallo von inside %f\n", thetamaxi);

if(fieldType == FIELD_AXION)
{
	if (precision == FIELD_SINGLE)
	{
	float *mTheta = static_cast<float*> (m);
	complex<float> *mCONT = static_cast<complex<float>*> (m2);

	#pragma omp parallel for default(shared)
		for(size_t idx=0; idx < n3; idx++)
			{
				mTheta[n2+idx] = mCONT[n2+idx].real();
			}

	}
	else //	FIELD_DOUBLE
	{
	double *mThetad = static_cast<double*> (m);
	complex<double> *mCONTd = static_cast<complex<double>*> (m2);

	#pragma omp parallel for default(shared)
		for(size_t idx=0; idx < n3; idx++)
			{
				mThetad[n2+idx] = mCONTd[n2+idx].real();
			}
	}
	printf("dens to m ... done\n");

}
else
{
	printf("dens to m not available for SAXION\n");
}

}
