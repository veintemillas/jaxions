#include<cstdlib>
#include<vector>
#include<complex>

#include"square.h"
#include"fftCuda.h"
#include"fftCode.h"
#include"enum-field.h"
#include"index.h"

#include"comms.h"

#include<cuda.h>
#include<cuda_runtime.h>
#include<mpi.h>

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
	const int Ez;
	const int v3;

	const int nSplit;

	void	*m,   *v,   *m2;			// Cpu data
	void	*m_d, *v_d, *m2_d;			// Gpu data
	double	*z;

	void	*sStreams;

	const bool lowmem;

	FieldPrecision	precision;
	int	fSize;

	void	recallGhosts(FieldIndex fIdx);		// Move the fileds that will become ghosts from the Cpu to the Gpu
	void	transferGhosts(FieldIndex fIdx);	// Copy back the ghosts to the Gpu

	public:

			 Scalar(const int nLx, const int nLz, FieldPrecision prec, const double zI, bool lowmem, const int nSp);
			 Scalar(const int nLx, const int nLz, FieldPrecision prec, const double zI, char fileName[], bool lowmem, const int nSp);
			~Scalar();

	void		genInitial();

	void		*mCpu() { return m; }
	const void	*mCpu() const { return m; }
	void		*vCpu() { return v; }
	const void	*vCpu() const { return v; }
	void		*m2Cpu() { return m2; }
	const void	*m2Cpu() const { return m2; }

	void		*mGpu() { return m_d; }
	const void	*mGpu() const { return m_d; }
	void		*vGpu() { return v_d; }
	const void	*vGpu() const { return v_d; }
	void		*m2Gpu() { return m2_d; }
	const void	*m2Gpu() const { return m2_d; }

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

	void	transferGpu(FieldIndex fIdx);		// Move data to Gpu
	void	transferCpu(FieldIndex fIdx);		// Move data to Cpu

	void	sendGhosts(FieldIndex fIdx);		// Send the ghosts in the Cpu using MPI, use this to exchange ghosts with Cpus
	void	exchangeGhosts(FieldIndex fIdx);	// Transfer ghosts from neighbouring ranks, use this to exchange ghosts with Gpus

	void	fftCpu();				// Fast Fourier Transform in the Cpu
	void	fftGpu();				// Fast Fourier Transform in the Gpu

	void	prepareCpu(int *window);		// Sets the field for a FFT, prior to analysis

	void	squareGpu();				// Squares the m2 field in the Gpu
	void	squareCpu();				// Squares the m2 field in the Cpu

	void	*Streams() { return sStreams; }
};

	Scalar::Scalar(const int nLx, const int nLz, FieldPrecision prec, const double zI, bool lowmem, const int nSp=1) : nSplit(nSp), n1(nLx), n2(nLx*nLx), n3(nLx*nLx*nLz), Lz(nLz), Ez(nLz + 2), v3(nLx*nLx*(nLz + 2)), precision(prec), lowmem(lowmem)
{
	switch (prec)
	{
		case FIELD_DOUBLE:

		fSize = sizeof(double);
		break;

		case FIELD_SINGLE:
		case FIELD_MIXED:

		fSize = sizeof(float);
		break;

		default:

		printf("Unrecognized precision\n");
		exit(1);
		break;
	}

	m  = malloc(fSize*2*v3);
	v  = malloc(fSize*2*n3);

	if (!lowmem)
		m2 = malloc(fSize*2*v3);

//	initFFT(m2, n1);

	z = (double *) malloc(sizeof(double));

	if (cudaMalloc(&m_d,  2*fSize*v3) != cudaSuccess)
	{
		printf("\n\nError: Couldn't allocate memory for the gpu, array mGpu\n");
		exit(1);
	}

	if (cudaMalloc(&v_d,  2*fSize*n3) != cudaSuccess)
	{
		printf("\n\nError: Couldn't allocate memory for the gpu, array vGpu\n");
		exit(1);
	}

	if (!lowmem)
		if (cudaMalloc(&m2_d, 2*fSize*v3) != cudaSuccess)
		{
			printf("\n\nError: Couldn't allocate memory for the gpu, array m2Gpu\n");
			exit(1);
		}

	sStreams = malloc(sizeof(cudaStream_t)*3);

	cudaStreamCreate(&((cudaStream_t *)sStreams)[0]);
	cudaStreamCreate(&((cudaStream_t *)sStreams)[1]);
	cudaStreamCreate(&((cudaStream_t *)sStreams)[2]);


	/*	BORRAR		*/
	/*	Set dummy data	*/

	if ((precision == FIELD_SINGLE) || (precision == FIELD_MIXED))
	{
		float *ptM = ((float *) m) + n2*2;
		float *ptV = ((float *) v);

		for (int i=0; i<n3*2; i++)
		{
			ptM[i] = i*1.;
			ptV[i] = (n3-i)*1.;
		}
	}

	if (precision == FIELD_DOUBLE)
	{
		double *ptM = ((double *) m) + n2*2;
		double *ptV = ((double *) v);

		for (int i=0; i<n3*2; i++)
		{
			ptM[i] = i*1.;
			ptV[i] = (n3-i)*1.;
		}
	}
	*z = zI;
	/*	FIN BORRAR	*/
}

	Scalar::Scalar(const int nLx, const int nLz, FieldPrecision prec, const double zI, char fileName[], bool lowmem, const int nSp=1) : nSplit(nSp), n1(nLx), n2(nLx*nLx), n3(nLx*nLx*nLz),
															 Lz(nLz), Ez(nLz + 2), v3(nLx*nLx*(nLz + 2)), precision(prec), lowmem(lowmem)
{
	switch (prec)
	{
		case FIELD_DOUBLE:

		fSize = sizeof(double);
		break;

		case FIELD_SINGLE:
		case FIELD_MIXED:

		fSize = sizeof(float);
		break;

		default:

		printf("Unrecognized precision\n");
		exit(1);
		break;
	}

	unsigned long int data = v3;
	unsigned long int datV = n3;

	data *= 2*fSize;
	datV *= 2*fSize;

	if ((m = malloc(data)) == NULL)
	{
		printf("\n\nError: Couldn't allocate %lu bytes for the cpu field m\n", data);
		exit(1);
	}

	if ((v = malloc(datV)) == NULL)
	{
		printf("\n\nError: Couldn't allocate %lu bytes for the cpu field v\n", datV);
		exit(1);
	}

	if (!lowmem)
		if ((m2 = malloc(data)) == NULL)
		{
			printf("\n\nError: Couldn't allocate %lu bytes for the cpu field m2\n", data);
			exit(1);
		}

	FILE *fileM = fopen(fileName,"r");

	if (fileM == 0)
	{
		printf("Sorry! Could not find initial Conditions\n");
		exit(1);
	}

	fread((((char *) m) + 2*fSize*n2), fSize*2, n3, fileM);
	fclose(fileM);

	if (precision == FIELD_DOUBLE)
	{
		#pragma omp parallel for default(shared) schedule(static)
		for (int lpc = 0; lpc < n3; lpc++)
		{
			((std::complex<double> *)v)[lpc]  = ((std::complex<double> *)m)[lpc+n2];
			((std::complex<double> *)m)[lpc+n2] *= zI;
		}
	}
	else
	{
		#pragma omp parallel for default(shared) schedule(static)
		for (int lpc = 0; lpc < n3; lpc++)
		{
			((std::complex<float> *)v)[lpc]  = ((std::complex<float> *)m)[lpc+n2];
			((std::complex<float> *)m)[lpc+n2] *= zI;
		}
	}

//	initFFT(m2, n1);

	if ((z = (double *) malloc(sizeof(double))) == NULL)
	{
		printf("\n\nError: Couldn't allocate memory for the cpu, z field (really???)\n");
		exit(1);
	}

	if (cudaMalloc(&m_d,  data) != cudaSuccess)
	{
		printf("\n\nError: Couldn't allocate memory for the gpu, array mEGpu %d\n", sizeof(double)*v3/(1024*1024));
		exit(1);
	}

	if (cudaMalloc(&v_d,  datV) != cudaSuccess)
	{
		printf("\n\nError: Couldn't allocate memory for the gpu, array vEGpu %d\n", sizeof(double)*n3/(1024*1024));
		exit(1);
	}

	if (!lowmem)
		if (cudaMalloc(&m2_d, data) != cudaSuccess)
		{
			printf("\n\nError: Couldn't allocate memory for the gpu, array m2Gpu %d\n", sizeof(double)*v3/(1024*1024));
			exit(1);
		}

	*z = zI;

	if ((sStreams = malloc(sizeof(cudaStream_t)*3)) == NULL)
	{
		printf("\n\nError: Couldn't allocate memory for the cpu, streams\n");
		exit(1);
	}

	cudaStreamCreate(&((cudaStream_t *)sStreams)[0]);
	cudaStreamCreate(&((cudaStream_t *)sStreams)[1]);
	cudaStreamCreate(&((cudaStream_t *)sStreams)[2]);
}

	Scalar::~Scalar()
{
	if (m != NULL)
		free(m);

	if (v != NULL)
		free(v);

	if (m2 != NULL)
		free(m2);

	if (m_d != NULL)
		cudaFree(m_d);

	if (v_d != NULL)
		cudaFree(v_d);

	if (m2_d != NULL)
		cudaFree(m2_d);

	if (z != NULL)
		free(z);

	cudaStreamDestroy(((cudaStream_t *)sStreams)[2]);
	cudaStreamDestroy(((cudaStream_t *)sStreams)[1]);
	cudaStreamDestroy(((cudaStream_t *)sStreams)[0]);

	if (sStreams != NULL)
		free(sStreams);
//	closeFFT();
}

void	Scalar::transferGpu(FieldIndex fIdx)	// Transfers only the internal volume
{
	if (fIdx & 1)
		cudaMemcpy((((char *) m_d) + 2*n2*fSize), (((char *) m) + 2*n2*fSize),  2*n3*fSize, cudaMemcpyHostToDevice);
//		cudaMemcpy(m_d,  m,  2*n3*fSize, cudaMemcpyHostToDevice);

	if (fIdx & 2)
		//cudaMemcpy((((char *) v_d) + 2*n2*fSize), (((char *) v) + 2*n2*fSize),  2*n3*fSize, cudaMemcpyHostToDevice);
		cudaMemcpy(v_d,  v,  2*n3*fSize, cudaMemcpyHostToDevice);

	if ((fIdx & 4) & (!lowmem))
		cudaMemcpy((((char *) m2_d) + 2*n2*fSize), (((char *) m2) + 2*n2*fSize),  2*n3*fSize, cudaMemcpyHostToDevice);
//		cudaMemcpy(m2_d, m2, 2*n3*fSize, cudaMemcpyHostToDevice);
}

void	Scalar::transferCpu(FieldIndex fIdx)	// Transfers only the internal volume
{
	if (fIdx & 1)
		cudaMemcpy(m,  m_d,  2*v3*fSize, cudaMemcpyDeviceToHost);

	if (fIdx & 2)
		cudaMemcpy(v,  v_d,  2*n3*fSize, cudaMemcpyDeviceToHost);

	if ((fIdx & 4) & (!lowmem))
		cudaMemcpy(m2, m2_d, 2*v3*fSize, cudaMemcpyDeviceToHost);
}

void	Scalar::recallGhosts(FieldIndex fIdx)		// Copy to the Cpu the fields in the Gpu that are to be exchanged
{
//	if (nSplit == 1)
//		return;

	if (fIdx & FIELD_M)
	{
//		cudaMemcpyAsync(m,  m_d,   2*n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[0]);
//		cudaMemcpyAsync((((char *) m) + 2*(n3-n2)*fSize),  (((char *) m_d) + 2*(n3-n2)*fSize),  2*n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[1]);
		cudaMemcpyAsync((((char *) m) + 2*n2*fSize), (((char *) m_d) + 2*n2*fSize), 2*n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[0]);
		cudaMemcpyAsync((((char *) m) + 2*n3*fSize), (((char *) m_d) + 2*n3*fSize), 2*n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[1]);
	}
	else
	{
//		cudaMemcpyAsync(m2, m2_d,  2*n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[0]);
//		cudaMemcpyAsync((((char *) m2) + 2*(n3-n2)*fSize), (((char *) m2_d) + 2*(n3-n2)*fSize), 2*n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[1]);
		cudaMemcpyAsync((((char *) m2) + 2*n2*fSize), (((char *) m2_d) + 2*n2*fSize), 2*n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[0]);
		cudaMemcpyAsync((((char *) m2) + 2*n3*fSize), (((char *) m2_d) + 2*n3*fSize), 2*n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[1]);
	}

	cudaStreamSynchronize(((cudaStream_t *)sStreams)[0]);
	cudaStreamSynchronize(((cudaStream_t *)sStreams)[1]);
}

void	Scalar::transferGhosts(FieldIndex fIdx)	// Transfers only the ghosts to the Gpu
{
//	if (nSplit == 1)
//		return;

	if (fIdx & FIELD_M)
	{
//		cudaMemcpyAsync((((char *) m_d)  + 2*n3*fSize), (((char *) m)  + 2*n3*fSize),  2*n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[0]);
//		cudaMemcpyAsync((((char *) m_d)  + 2*(n3+n2)*fSize), (((char *) m)  + 2*(n3+n2)*fSize),  2*n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[1]);
		cudaMemcpyAsync( ((char *) m_d),                      ((char *) m),                     2*n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[0]);
		cudaMemcpyAsync((((char *) m_d)  + 2*(n3+n2)*fSize), (((char *) m)  + 2*(n3+n2)*fSize), 2*n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[1]);
	}
	else
	{
//		cudaMemcpyAsync((((char *) m2_d) + 2*n3*fSize), (((char *) m2) + 2*n3*fSize),  2*n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[0]);
//		cudaMemcpyAsync((((char *) m2_d) + 2*(n3+n2)*fSize), (((char *) m2) + 2*(n3+n2)*fSize),  2*n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[1]);
		cudaMemcpyAsync( ((char *) m2_d),                      ((char *) m2),                     2*n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[0]);
		cudaMemcpyAsync((((char *) m2_d)  + 2*(n3+n2)*fSize), (((char *) m2)  + 2*(n3+n2)*fSize), 2*n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[1]);
	}

	cudaStreamSynchronize(((cudaStream_t *)sStreams)[0]);
	cudaStreamSynchronize(((cudaStream_t *)sStreams)[1]);
}

void	Scalar::sendGhosts(FieldIndex fIdx)
{
//	if (nSplit == 1)
//		return;

	const int rank = commRank();
	const int fwdNeig = (rank + 1) % nSplit;
	const int bckNeig = (rank - 1 + nSplit) % nSplit;

	const int ghostBytes = 2*n2*fSize;

	/* Assign receive buffers to the right parts of m, v */

	void *sGhostBck, *sGhostFwd, *rGhostBck, *rGhostFwd;

	if (fIdx & FIELD_M)
	{
//		sGhostBck = m;
//		sGhostFwd = (void *) (((char *) m) + 2*(n3-n2)*fSize);
//		rGhostBck = (void *) (((char *) m) + 2*(n3+n2)*fSize);
//		rGhostFwd = (void *) (((char *) m) + 2*n3*fSize);
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
//		sGhostBck = m2;
//		sGhostFwd = (void *) (((char *) m2) + 2*(n3-n2)*fSize);
//		rGhostBck = (void *) (((char *) m2) + 2*(n3+n2)*fSize);
//		rGhostFwd = (void *) (((char *) m2) + 2*n3*fSize);
	}

/*	if (nSplit == 1)
	{
		cudaMemcpy(rGhostFwd, sGhostBck, 2*n2*fSize, cudaMemcpyHostToHost);
		cudaMemcpy(rGhostBck, sGhostFwd, 2*n2*fSize, cudaMemcpyHostToHost);
	}
	else*/
	{
		MPI_Request rSendFwd, rSendBck, rRecvFwd, rRecvBck;

		MPI_Send_init(sGhostFwd, ghostBytes, MPI_BYTE, fwdNeig, 2*rank,   MPI_COMM_WORLD, &rSendFwd);
		MPI_Send_init(sGhostBck, ghostBytes, MPI_BYTE, bckNeig, 2*rank+1, MPI_COMM_WORLD, &rSendBck);

		MPI_Recv_init(rGhostFwd, ghostBytes, MPI_BYTE, fwdNeig, 2*fwdNeig+1,   MPI_COMM_WORLD, &rRecvFwd);
		MPI_Recv_init(rGhostBck, ghostBytes, MPI_BYTE, bckNeig, 2*bckNeig, MPI_COMM_WORLD, &rRecvBck);

		MPI_Start(&rRecvBck);
		MPI_Start(&rRecvFwd);
		MPI_Start(&rSendFwd);
		MPI_Start(&rSendBck);

		MPI_Wait(&rSendFwd, MPI_STATUS_IGNORE);
		MPI_Wait(&rSendBck, MPI_STATUS_IGNORE);
		MPI_Wait(&rRecvFwd, MPI_STATUS_IGNORE);
		MPI_Wait(&rRecvBck, MPI_STATUS_IGNORE);

//	printf("Rank %d sends %e fwd to %d and %e bck to %d. Receives %e from %d and %e from %d\n", rank, ((float *) sGhostFwd)[0], fwdNeig, ((float *) sGhostBck)[0], bckNeig, ((float *) rGhostBck)[0], fwdNeig, ((float *) rGhostFwd)[0], bckNeig);

		MPI_Request_free(&rSendFwd);
		MPI_Request_free(&rSendBck);
		MPI_Request_free(&rRecvFwd);
		MPI_Request_free(&rRecvBck);

		MPI_Barrier(MPI_COMM_WORLD);
	}
}

void	Scalar::exchangeGhosts(FieldIndex fIdx)
{
	recallGhosts(fIdx);
	sendGhosts(fIdx);
	transferGhosts(fIdx);
}

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

void	Scalar::squareGpu()
{
	square(m2_d, n1, Lz, n3, precision);
}

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

void	Scalar::fftCpu	()
{
	runFFT();
}

void	Scalar::fftGpu	()
{
	runCudaFFT(m2_d);
}
