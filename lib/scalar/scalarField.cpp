#include<cstdlib>
#include<cstring>
#include<complex>
#include<chrono>

#include"enum-field.h"
#include"fft/fftCuda.h"
#include"fft/fftCode.h"

#include"scalar/scalarField.h"

#include"comms/comms.h"
#include"scalar/varNQCD.h"

#ifdef	USE_GPU
	#include<cuda.h>
	#include<cuda_runtime.h>
	#include "cudaErrors.h"
#endif

#include<mpi.h>
#include<omp.h>

#include "utils/utils.h"
#include "gen/genConf.h"

using namespace std;
using namespace profiler;

#define printMpi(...) do {		\
	if (!commRank()) {		\
	  printf(__VA_ARGS__);  	\
	  fflush(stdout); }		\
}	while (0)

const std::complex<double> I(0.,1.);
const std::complex<float> If(0.,1.);


	Scalar::Scalar(const size_t nLx, const size_t nLz, FieldPrecision prec, DeviceType dev, const double zI, bool lowmem, const int nSp, FieldType newType, LambdaType lType,
		       ConfType cType, const size_t parm1, const double parm2) : nSplit(nSp), n1(nLx), n2(nLx*nLx), n3(nLx*nLx*nLz), Lz(nLz), Ez(nLz + 2), Tz(Lz*nSp), v3(nLx*nLx*(nLz + 2)),
		       fieldType(newType), lambdaType(lType), precision(prec), device(dev), lowmem(lowmem)
{
LogOut("0");
	Profiler &prof = getProfiler(PROF_SCALAR);

	prof.start();

	size_t nData;

	folded 	   = false;

	switch (fieldType)
	{
		case FIELD_SAXION:
			nData = 2;
			break;

		case FIELD_AXION:
		case FIELD_WKB:
			nData = 1;
			break;

		default:
			LogError("Error: unrecognized field type");
			exit(1);
			break;
	}

	switch (prec)
	{
		case FIELD_DOUBLE:
			fSize = sizeof(double)*nData;
			break;

		case FIELD_SINGLE:
			fSize = sizeof(float)*nData;
			break;

		default:
			LogError("Error: unrecognized precision");
			exit(1);
			break;
	}

	switch	(dev)
	{
		case DEV_CPU:
			#ifdef	__AVX512F__
			LogMsg(VERB_NORMAL, "Using AVX-512 64 bytes alignment");
			mAlign = 64;
			#elif	defined(__AVX__) || defined(__AVX2__)
			LogMsg(VERB_NORMAL, "Using AVX 32 bytes alignment");
			mAlign = 32;
			#else
			LogMsg(VERB_NORMAL, "Using SSE 16 bytes alignment");
			mAlign = 16;
			#endif
			break;

		case DEV_GPU:
			LogMsg(VERB_NORMAL, "Using 16 bytes alignment for the Gpu");
			mAlign = 16;
			break;
	}

	shift = mAlign/fSize;

	if (n2*fSize % mAlign)
	{
		LogError("Error: misaligned memory. Are you using an odd dimension?");
		exit(1);
	}

	const size_t	mBytes = v3*fSize;
	const size_t	vBytes = n3*fSize;
LogOut("1");
	// MODIFIED BY JAVI
	// IN AXION MODE I WANT THE M AND V SPACES TO BE ALIGNED

	switch (fieldType)
	{
		case FIELD_SAXION:
			alignAlloc ((void**) &m, mAlign, mBytes);
			alignAlloc ((void**) &v, mAlign, vBytes);
			break;

		case FIELD_AXION:
		case FIELD_WKB:
			//alignAlloc ((void**) &m, mAlign, mBytes+vBytes);
			//this would allocate a full complex m space, a bit larger than m+v in real mode (mBytes+vBytes)
			//alignAlloc ((void**) &m, mAlign, 2*mBytes);
			//this allocates a slightly larger v to host FFTs in place
			alignAlloc ((void**) &m, mAlign, mBytes);
			alignAlloc ((void**) &v, mAlign, mBytes);
			break;

		default:
			LogError("Error: unrecognized field type");
			exit(1);
			break;
	}
	// MODIFICATION UNTIL HERE
	// NOTE THAT DOES NOT AFFECT CREATION IN SAXION MODE


	/*	This MUST be revised, otherwise
		simulations can segfault after
		the transition to theta due to
		lack of memory. The difference
		is small (a ghost region), but
		it must be taken into account.	*/

	//  M2 issue ;; we always allocate a complex m2 in theta mode!
	//	EVEN IF WE DO NOT SPECIFY lowmem
	switch (fieldType)
	{
		case FIELD_SAXION:
			if (!lowmem) {
				alignAlloc ((void**) &m2, mAlign, mBytes);
				memset (m2, 0, fSize*v3);
			} else
				m2 = nullptr;
			break;

		case FIELD_AXION:
			alignAlloc ((void**) &m2, mAlign, 2*mBytes);
			memset (m2, 0, 2*fSize*n3);
			break;

		case FIELD_WKB:
			m2 = nullptr;
			break;

		default:
			LogError("Error: unrecognized field type");
			exit(1);
			break;
	}

LogOut("2");
	if (m == NULL)
	{
		LogError ("Error: couldn't allocate %lu bytes on host for the m field", mBytes);
		exit(1);
	}

	if (v == NULL)
	{
		LogError ("Error: couldn't allocate %lu bytes on host for the v field", vBytes);
		exit(1);
	}

	if (!lowmem)
	{
		if (m2 == nullptr)
		{
			LogError ("Error: couldn't allocate %lu bytes on host for the m2 field", mBytes);
			exit(1);
		}
	}

LogOut("3");
	memset (m, 0, fSize*v3);
	// changed from memset (v, 0, fSize*n3);
	memset (v, 0, fSize*n3);

	commSync();

	alignAlloc ((void **) &z, mAlign, mAlign);

	if (z == NULL)
	{
		LogError ("Error: couldn't allocate %d bytes on host for the z field", sizeof(double));
		exit(1);
	}

	if (device == DEV_GPU)
	{
#ifndef	USE_GPU
		LogError ("Error: gpu support not built\n");
		exit   (1);
#else
		if (cudaMalloc(&m_d,  mBytes) != cudaSuccess)
		{
			LogError ("Error: couldn't allocate %lu bytes for the gpu field m", mBytes);
			exit(1);
		}

		if (cudaMalloc(&v_d,  vBytes) != cudaSuccess)
		{
			LogError ("Error: couldn't allocate %lu bytes for the gpu field v", vBytes);
			exit(1);
		}

		if (!lowmem)
			if (cudaMalloc(&m2_d, mBytes) != cudaSuccess)
			{
				LogError ("Error: couldn't allocate %lu bytes for the gpu field m2", mBytes);
				exit(1);
			}

		if ((sStreams = malloc(sizeof(cudaStream_t)*3)) == NULL)
		{
			LogError ("Error: couldn't allocate %lu bytes on host for the gpu streams", sizeof(cudaStream_t)*3);
			exit(1);
		}

		cudaStreamCreate(&((cudaStream_t *)sStreams)[0]);
		cudaStreamCreate(&((cudaStream_t *)sStreams)[1]);
		cudaStreamCreate(&((cudaStream_t *)sStreams)[2]);
#endif
	}

	*z = zI;

	/*	WKB fields won't trigger configuration read or FFT initialization	*/

	if (fieldType != FIELD_WKB) {

		AxionFFT::initFFT(prec);

		if (pType & PROP_SPEC) {
			AxionFFT::initPlan (this, FFT_SPSX,  FFT_FWDBCK, "SpSx");
			AxionFFT::initPlan (this, FFT_SPAX,  FFT_FWDBCK, "SpAx");
		}

		if (fieldType = FIELD_SAXION) {
			AxionFFT::initPlan (this, FFT_RtoC_M2toM2_SAXION_AXION, FFT_FWD, "pSpectrum_ax");
		}
		else
		{
			AxionFFT::initPlan (this, FFT_RtoC_M2toM2_AXION, FFT_FWD, "pSpectrum_ax");
		}

		if (!lowmem) {
			AxionFFT::initPlan (this, FFT_CtoC_MtoM2,	   FFT_FWD, "nSpecSxM");
			AxionFFT::initPlan (this, FFT_CtoC_VtoM2,	   FFT_FWD, "nSpecSxV");
			AxionFFT::initPlan (this, FFT_RtoC_M2toM2_SAXION,  FFT_FWD, "pSpectrum_sax");
		}

		/*	If present, read fileName	*/

		if (cType == CONF_NONE) {
			LogMsg (VERB_HIGH, "No configuration selected. Hope we are reading from a file...");

			if (fIndex == -1) {
				LogError ("Error: neither file nor initial configuration specified");
				exit(2);
			}

		} else {
			if (fieldType == FIELD_AXION) {
				LogError ("Configuration generation for axion fields not supported");
			} else {
				if (cType == CONF_KMAX || cType == CONF_TKACHEV)
					if (lowmem)
						AxionFFT::initPlan (this, FFT_CtoC_MtoM,  FFT_FWDBCK, "Init");
					else
						AxionFFT::initPlan (this, FFT_CtoC_MtoM2, FFT_FWDBCK, "Init");
				prof.stop();
				genConf	(this, cType, parm1, parm2);
				prof.start();
			}
		}
		prof.stop();
		prof.add(std::string("Init"), 0.0, (lowmem ? 2*mBytes+vBytes : mBytes+vBytes)*1e-9);
	} else {
		prof.stop();
		prof.add(std::string("Init"), 0.0, 2.e-9*mBytes);
	}
}

// END SCALAR

	Scalar::~Scalar()
{
	commSync();
	LogMsg (VERB_HIGH, "Rank %d Calling destructor...",commRank());

	if (m != nullptr)
		trackFree(&m, ALLOC_ALIGN);

	if (v != nullptr && fieldType == FIELD_SAXION) {
		trackFree(&v, ALLOC_ALIGN);
	}

	if (m2 != nullptr)
		trackFree(&m2, ALLOC_ALIGN);

	if (z != nullptr)
		trackFree((void **) &z, ALLOC_ALIGN);

	if (device == DEV_GPU)
	{
		#ifndef	USE_GPU
			LogError ("Error: gpu support not built");
			exit   (1);
		#else
			if (m_d != nullptr)
				cudaFree(m_d);

			if (v_d != nullptr)
				cudaFree(v_d);

			if (m2_d != nullptr)
				cudaFree(m2_d);

			cudaStreamDestroy(((cudaStream_t *)sStreams)[2]);
			cudaStreamDestroy(((cudaStream_t *)sStreams)[1]);
			cudaStreamDestroy(((cudaStream_t *)sStreams)[0]);

			if (sStreams != nullptr)
				free(sStreams);

			AxionFFT::closeFFT();
		#endif
	} else {
		AxionFFT::closeFFT();
	}
}

void	Scalar::transferDev(FieldIndex fIdx)	// Transfers only the internal volume
{
	if (device == DEV_GPU)
	{
		#ifndef	USE_GPU
			LogError ("Error: gpu support not built");
			exit   (1);
		#else
			if (fIdx & 1)
				cudaMemcpy((((char *) m_d) + n2*fSize), (((char *) m) + n2*fSize),  n3*fSize, cudaMemcpyHostToDevice);

			if (fIdx & 2)
				cudaMemcpy(v_d,  v,  n3*fSize, cudaMemcpyHostToDevice);

			if ((fIdx & 4) && (!lowmem))
				cudaMemcpy((((char *) m2_d) + n2*fSize), (((char *) m2) + n2*fSize),  n3*fSize, cudaMemcpyHostToDevice);
		#endif
	}
}

void	Scalar::transferCpu(FieldIndex fIdx)	// Transfers only the internal volume
{
	if (device == DEV_GPU)
	{
		#ifndef	USE_GPU
			LogError ("Error: gpu support not built");
			exit   (1);
		#else
			if (fIdx & 1)
				cudaMemcpy(m,  m_d,  v3*fSize, cudaMemcpyDeviceToHost);

			if (fIdx & 2)
				cudaMemcpy(v,  v_d,  n3*fSize, cudaMemcpyDeviceToHost);

			if ((fIdx & 4) && (!lowmem))
				cudaMemcpy(m2, m2_d, v3*fSize, cudaMemcpyDeviceToHost);
		#endif
	}
}

void	Scalar::recallGhosts(FieldIndex fIdx)		// Copy to the Cpu the fields in the Gpu that are to be exchanged
{
	if (device == DEV_GPU)
	{
		#ifndef	USE_GPU
			LogError ("Error: gpu support not built");
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
	}
}

void	Scalar::transferGhosts(FieldIndex fIdx)	// Transfers only the ghosts to the Gpu
{
	if (device == DEV_GPU)
	{
		#ifndef	USE_GPU
			LogError ("Error: gpu support not built");
			exit   (1);
		#else
			if (fIdx & FIELD_M)
			{
				cudaMemcpyAsync(static_cast<char *> (m_d),                 static_cast<char *> (m),                 n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[0]);
				cudaMemcpyAsync(static_cast<char *> (m_d) + (n3+n2)*fSize, static_cast<char *> (m) + (n3+n2)*fSize, n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[1]);
			} else {
				cudaMemcpyAsync(static_cast<char *> (m2_d),                 static_cast<char *> (m2),                  n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[0]);
				cudaMemcpyAsync(static_cast<char *> (m2_d) + (n3+n2)*fSize, static_cast<char *> (m2)  + (n3+n2)*fSize, n2*fSize, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[1]);
			}

			cudaStreamSynchronize(((cudaStream_t *)sStreams)[0]);
			cudaStreamSynchronize(((cudaStream_t *)sStreams)[1]);
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

			break;
	}
}

void	Scalar::exchangeGhosts(FieldIndex fIdx)
{
	recallGhosts(fIdx);
	sendGhosts(fIdx, COMM_SDRV);
	sendGhosts(fIdx, COMM_WAIT);
	transferGhosts(fIdx);
}

void	Scalar::setField (FieldType newType)
{
	if (fieldType == FIELD_WKB) {
		LogError("Warning: conversion from WKB field not supported");
		return;
	}

	switch (newType)
	{
		case FIELD_AXION:
			if (fieldType == FIELD_SAXION)
			{
				if (!lowmem) {
					trackFree(&m2, ALLOC_ALIGN);

					#ifdef	USE_GPU
					if (device == DEV_GPU)
						cudaFree(m2_d);
					#endif
				}

				m2 = v;
				#ifdef	USE_GPU
				if (device == DEV_GPU)
					m2_d = v_d;
				#endif

				switch (precision)
				{
					case FIELD_SINGLE:
					v = static_cast<void*>(static_cast<float*>(m) + 2*n2 + n3);

					#ifdef	USE_GPU
					if (device == DEV_GPU)
						v_d = static_cast<void*>(static_cast<float*>(m_d) + 2*n2 + n3);
					#endif

					break;

					case FIELD_DOUBLE:
					v = static_cast<void*>(static_cast<double*>(m) + 2*n2 + n3);

					#ifdef	USE_GPU
					if (device == DEV_GPU)
						v_d = static_cast<void*>(static_cast<double*>(m_d) + 2*n2 + n3);
					#endif

					break;
				}

				fSize /= 2;

				if (device != DEV_GPU)
					shift *= 2;

				const size_t	mBytes = v3*fSize;

				//if (lowmem)
				//AxionFFT::initPlan(this, FFT_RtoC_M2toM2, FFT_FWD, "pSpectrum");

				// IF low mem was used before, it creates m2 COMPLEX
/*				if (lowmem)
				{
					alignAlloc ((void**) &m2, mAlign, 2*mBytes);

					// Move this to the analysis
					// AxionFFT::initPlan(this, FFT_CtoC_M2toM2, FFT_FWD, "pSpectrum");

					#ifdef	USE_GPU
					if (cudaMalloc(&m2_d, 2*mBytes) != cudaSuccess)
					{
						LogError ("Error: couldn't allocate %lu bytes for the gpu field m2", 2*mBytes);
						exit(1);
					}
					#endif

				} else {
				// IF no lowmem was used, we kill m2 complex and create m2 real ... not used
					trackFree(&m2, ALLOC_ALIGN);
					m2 = nullptr;
					alignAlloc ((void**) &m2, mAlign, 2*mBytes);

				#ifdef	USE_GPU
					cudaFree(m2_d);
				#endif

				#ifdef	USE_GPU
					if (cudaMalloc(&m2_d, 2*mBytes) != cudaSuccess)
					{
						LogError ("Error: couldn't allocate %lu bytes for the gpu field m2", 2*mBytes);
						exit(1);
					}
				#endif
				}
*/

			//FFT for spectrums

			}
			break;

		case	FIELD_SAXION:
			if (fieldType == FIELD_AXION)
			{
				if (commRank() == 0)
					LogError ("Error: transformation from axion to saxion not supported");
			} else {
				fieldType = FIELD_SAXION;
			}
			break;
	}
	fieldType = newType;
}

void	Scalar::setFolded (bool foli)
{
	folded = foli ;
}

/*	These next two functions are to be
	removed, the only reason to keep them
	is because the code inside might be
	useful for a future gpu FFT
	implementation, but the chances of
	that happening are pretty slim		*/
/*	The problem of the theta spectral
	propagator might renew the interest
	in these functions			*/

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

/*	Follow all the functions written by Javier	*/
/*	These should be rewritten following the
	standards of the library, including logger,
	profiler, vector code, gpu support and
	outside the Scalar class, so it doesn't
	become a massively cluttered object		*/

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


//	COPIES c_theta into m2 AS PADDED REAL ARRAY
// 	M2 IS SHIFTED BY GHOSTS? check ...
//	USA M2, ARREGLAR LOWMEM
void	Scalar::theta2m2()//int *window)
{
	LogMsg (VERB_HIGH, "Function theta2m2 marked for future optimization or removal");

	// PADDING idx  = N*N *iz + N *iy + ix
	// 			TO idx' = N*N'*iz + N'*iy + ix = idx + 2*( iy + N*iz )
	//			where N'= 2(N/2+1) = N+2 (because our N's are always even)
	//			interestingly iy+N*iz is idx/N
	// 			THEREFORE PADDING IS JUST
	//			idx -> idx + 2(idx/N)
	size_t idx;

	switch (fieldType)
	{
		case FIELD_SAXION:
			if (precision == FIELD_DOUBLE)
			{
				double za = (*z);

				#pragma omp parallel for default(shared) schedule(static)
				for(size_t i=0; i <n3; i++)
				{
					double thetaaux = arg(((std::complex<double> *) m)[n2+i]);
					(static_cast<double*> (m2))[i + 2*(i/n1)] = thetaaux*za ;
				}
			}
			else // FIELD_SINGLE
			{
				float zaf = *z ;

				#pragma omp parallel for default(shared) schedule(static)
				for(size_t i=0; i < n3; i++)
				{
					float thetaauxf = arg(((complex<float> *) m)[n2+i]);
					(static_cast<float*> (m2))[i + 2*(i/n1)] = thetaauxf*zaf;
				}
			}
		break;

		case FIELD_AXION:
			if (precision == FIELD_DOUBLE)
			{
				#pragma omp parallel for default(shared) schedule(static)
				for(size_t i=0; i < n3; i++)
					(static_cast<double*> (m2))[ i + 2*(i/n1)] = ((static_cast<double*> (m))[n2+i]) ;
			}
			else	// FIELD_SINGLE
			{
				#pragma omp parallel for default(shared) schedule(static)
				for(size_t i=0; i < n3; i++)
					(static_cast<float*> (m2))[ i + 2*(i/n1)] = ((static_cast<float*> (m))[n2+i]) ;
			}
		break;
	}
}

//	COPIES c_theta_v (vtheta) into m2 AS PADDED REAL ARRAY
// 	M2 IS SHIFTED BY N2
//	USA M2, ARREGLAR LOWMEM
void	Scalar::vheta2m2()//int *window)
{
	LogMsg (VERB_HIGH, "Function vheta2m2 marked for future optimization or removal");

	switch (fieldType)
	{
		case FIELD_SAXION:
			if (precision == FIELD_DOUBLE)
			{
				double za = (*z);

				#pragma omp parallel for default(shared) schedule(static)
				for(size_t i=0; i < n3; i++)
				{
					double thetaaux = arg(((std::complex<double> *) m)[n2+i]);
					//((complex<double> *) m2)[i+n2] = 0.0 + I*( ((((complex<double>*) v)[i]/((complex<double>*) m)[i+n2]).imag())*za + thetaaux);
					(static_cast<double*> (m2))[ i + 2*(i/n1)] = (((((complex<double>*) v)[i]/((complex<double>*) m)[n2+i]).imag())*za + thetaaux);
				}
			}
			else // FIELD_SINGLE
			{
				float zaf = *z ;

				#pragma omp parallel for default(shared) schedule(static)
				for(size_t i=0; i < n3; i++)
				{
					float thetaauxf = arg(((complex<float> *) m)[i+n2]);
					//((complex<float> *) m2)[i+n2] = 0.f + If*(((((complex<float>*) v)[i]/((complex<float>*) m)[i+n2]).imag())*zaf + thetaauxf);
					(static_cast<float*> (m2))[ i + 2*(i/n1)] = (((((complex<float>*) v)[i]/((complex<float>*) m)[n2+i]).imag())*zaf + thetaauxf);
				}
			}
		break;

		case FIELD_AXION:
			if (precision == FIELD_DOUBLE)
			{

				#pragma omp parallel for default(shared) schedule(static)
				for(size_t i=0; i < n3; i++)
					//((complex<double> *) m2)[i+n2] = 0.0 + I*((static_cast<double*> (v))[i]);
					(static_cast<double*> (m2))[ i + 2*(i/n1)] = ((static_cast<double*> (v))[i]);
			}
			else	// FIELD_SINGLE
			{

				#pragma omp parallel for default(shared) schedule(static)
				for(size_t i=0; i < n3; i++)
					//((complex<float> *) m2)[i+n2] = 0.f + If*((static_cast<float*> (v))[i]);
					(static_cast<float*> (m2))[ i + 2*(i/n1)] = ((static_cast<float*> (v))[i]);
			}
		break;
	}	// END FIELDTYPE
}		// END vheta2m2

//	energy is copied in m2 after the ghost AS REAL
// 	the fourier transform R2C takes a real array padded
//	USA M2, ARREGLAR LOWMEM
// void	Scalar::padenergym2()//int *window)
// {
// 	LogMsg (VERB_HIGH, "Function padenergym2 marked for future optimization or removal");
//
// 	if (precision == FIELD_DOUBLE)
// 	{
// 		#pragma omp parallel for default(shared) schedule(static)
// 		for(size_t i=0; i < n3; i++)
// 		{
// 			((static_cast<double*> (m2))[n2 + i + 2*(i/n1)] = (((((complex<double>*) v)[i]/((complex<double>*) m)[i+n2]).imag())*za + thetaaux);
// 		}
// 	}
// 	else // FIELD_SINGLE
// 	{
// 		#pragma omp parallel for default(shared) schedule(static)
// 		for(size_t i=0; i < n3; i++)
// 		{
// 			float thetaauxf = arg(((complex<float> *) m)[i+n2]);
// 			//((complex<float> *) m2)[i+n2] = 0.f + If*(((((complex<float>*) v)[i]/((complex<float>*) m)[i+n2]).imag())*zaf + thetaauxf);
// 			((static_cast<float*> (m2))[n2 + i + 2*(i/n1)] = (((((complex<float>*) v)[i]/((complex<float>*) m)[i+n2]).imag())*zaf + thetaauxf);
// 		}
// 	}		// END padenergym2


// LEGACY FUNCTION COPYING c_theta*mass + I c_theta_z in m2 for the number spectrum
// SUPERSEEDED BY theta2m2 and vheta2m2 to work with MPI
void	Scalar::thetav2m2()//int *window)
{
	LogMsg (VERB_HIGH, "Function thetav2m2 marked for future optimization or removal");

	switch (fieldType)
	{
		case FIELD_SAXION:
			if (precision == FIELD_DOUBLE)
			{
				double za = (*z);
				//double massfactor = 3.0 * pow(za, nQcd/2 + 1);
				double massfactor = axionmass(za, nQcd,zthres, zrestore)*za;

				#pragma omp parallel for default(shared) schedule(static)
				for(size_t i=0; i < n3; i++)
				{
					double thetaaux = arg(((std::complex<double> *) m)[i+n2]);
					((complex<double> *) m2)[i+n2] = thetaaux*massfactor*za + I*( ((((complex<double>*) v)[i]/((complex<double>*) m)[i+n2]).imag())*za + thetaaux);
				}
			} else {
				float zaf = *z ;
				//float massfactor = 3.0 * pow(zaf, nQcd/2 + 1);
				float massfactor = (float) axionmass((double) zaf, nQcd,zthres, zrestore)*zaf;

				#pragma omp parallel for default(shared) schedule(static)
				for(size_t i=0; i < n3; i++)
				{
					float thetaauxf = arg(((complex<float> *) m)[i+n2]);
					((complex<float> *) m2)[i+n2] = thetaauxf*massfactor*zaf + If*(((((complex<float>*) v)[i]/((complex<float>*) m)[i+n2]).imag())*zaf + thetaauxf);
				}
			}
		break;

		case FIELD_AXION:
			if (precision == FIELD_DOUBLE)
			{
				//double massfactor = 3.0 * pow((*z), nQcd/2 + 1);
				double massfactor = axionmass((*z), nQcd,zthres, zrestore)*(*z);

				#pragma omp parallel for default(shared) schedule(static)
				for(size_t i=0; i < n3; i++)
					((complex<double> *) m2)[i+n2] = ((static_cast<double*> (m))[i+n2])*massfactor + I*((static_cast<double*> (v))[i]);
			}
			else
			{
				//float massfactor = 3.0 * pow((*z), nQcd/2 + 1);
				float zaf = (float) *z ;
				float massfactor = (float) axionmass((double) zaf, nQcd,zthres, zrestore)*zaf;

				#pragma omp parallel for default(shared) schedule(static)
				for(size_t i=0; i < n3; i++)
					((complex<float> *) m2)[i+n2] = ((static_cast<float*> (m))[i+n2])*massfactor + If*((static_cast<float*> (v))[i]);
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


//void	Scalar::writeENERGY (double zzz, FILE *enwrite)
void	Scalar::writeENERGY (double zzz, FILE *enwrite, double &Gfr, double &Gft, double &Vfr, double &Vft, double &Kfr, double &Kft) // TEST
{
	LogError ("Function writeENERGY has been deprecated, use writeEnergy instead");
	LogOut   ("Function writeENERGY has been deprecated and is marked for removal, use writeEnergy instead\n");

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
	LogError ("Function ENERGY has been deprecated, use energy instead");
	LogOut   ("Function ENERGY has been deprecated and is marked for removal, use energy instead\n");

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
	LogError ("Function ENERGY2 has been deprecated, use energy instead");
	LogOut   ("Function ENERGY2 has been deprecated and is marked for removal, use energy instead\n");

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
			Float modul, modfac;
			size_t idd;

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
	LogMsg (VERB_HIGH, "Function writeMAPTHETA marked for optimization or removal");
	LogMsg (VERB_NORMAL, "writeMAPTHETA");

	double maxcontrast ;
	switch	(precision)
	{
		case	FIELD_DOUBLE:
		{
		}
		break;

		case	FIELD_SINGLE:
		{
			//COMPUTES JAVIER DENSITY MAP AND BINS
			//energymapTheta (static_cast<float>(zzz), index, contbin, numbins); // TEST

			//USES WHATEVER IS IN M2, COMPUTES CONTRAST AND BINS []
			// USE AFTER ALEX'FUNCTION
			contrastbin(static_cast<float>(zzz), index, contbin, numbins);
		}
		break;

		default:
		printf("Unrecognized precision\n");
		exit(1);
		break;
	}
	return ;
}

//----------------------------------------------------------
//	DENSITY CONTRAST FUNCTION BY JAVIER< IT WORKS...
//----------------------------------------------------------

template<typename Float>
//void	Scalar::ENERGY(const Float zz, FILE *enWrite)
void	Scalar::energymapTheta(const Float zz, const int index, void *contbin, int numbins)
{
	LogMsg (VERB_HIGH, "Function energymapTheta marked for optimization or removal, please use energy instead");
	LogMsg (VERB_NORMAL, "energymapTheta");

	// THIS TEMPLATE IS TO BE CALLED UNFOLDED
	if (folded)
		{
			printMpi("EMT called Folded!\n");
			return;
		}
	// COPIES THE CONTRAST INTO THE REAL PART OF M2 (WHICH IS COMPLEX)
	// TO USE THE POWER SPECTRUM AFTER
	// 	FILES DENSITY CONTRAST


	// 	CONSTANTS
	const Float deltaa2 = pow(sizeL/sizeN,2.)*4. ;
	const Float invz	= 1.0/(*z);
	const Float piz2	= 3.1415926535898*(zz)*3.1415926535898*(zz);
	//const Float z9QCD4 = 9.0*pow((*z),nQcd+4.) ;
	const Float z9QCD4 = axionmass2((*z), nQcd, zthres, zrestore )*pow((*z),4);

	//	AUX VARIABLES
	Float maxi = 0.;
	Float maxibin = 0.;
	double maxid =0.;
	double toti = 0.;

	exchangeGhosts(FIELD_M);

	//SUM variables
	double contbin_local[numbins] ;
	double toti_global;
	double maxi_global;

	//printf("\n q1-%d",commRank());fflush(stdout);
	if(fieldType == FIELD_AXION)
	{
		Float *mTheta = static_cast<Float*> (m);
		Float *mVeloc = static_cast<Float*> (v);
		// REAL VERSION
		//Float *mCONT = static_cast<Float*> (m2);
		// COMPLEX VERSION
		complex<Float> *mCONT = static_cast<complex<Float>*> (m2);
		//printf("ENERGY map theta \n");

		#pragma omp parallel for default(shared) schedule(static) reduction(max:maxi), reduction(+:toti)
		for (size_t iz=0; iz < Lz; iz++)
		{
			//OLD VERSION
			// Float acu , grad ;
			// size_t idx, idaux ;
			// size_t iyP, iyM, ixP, ixM;
			// for (size_t iy=0; iy < n1; iy++)
			// {
			// 	iyP = (iy+1)%n1;
			// 	iyM = (iy-1+n1)%n1;
			// 	for (size_t ix=0; ix < n1; ix++)
			// 	{
			// 		ixP = (ix+1)%n1;
			// 		ixM = (ix-1+n1)%n1;
			//
			// 		idx = ix + iy*n1+(iz+1)*n2 ;
			// 		//KINETIC + POTENTIAL
			// 		acu = mVeloc[idx-n2]*mVeloc[idx-n2]/2. + z9QCD4*(1.0-cos(mTheta[idx]*invz)) ;
			// 		//GRADIENTS
			// 		idaux = ixP + iy*n1+(iz+1)*n2 ;
			// 		grad = pow(mTheta[idaux]-mTheta[idx],2);
			//
			// 		idaux = ixM + iy*n1+(iz+1)*n2 ;
			// 		grad += pow(mTheta[idaux]-mTheta[idx],2);
			//
			// 		idaux = ix + iyP*n1+(iz+1)*n2 ;
			// 		grad += pow(mTheta[idaux]-mTheta[idx],2);
			//
			// 		idaux = ix + iyM*n1+(iz+1)*n2 ;
			// 		grad += pow(mTheta[idaux]-mTheta[idx],2);
			// 		grad += pow(mTheta[idx+n2]-mTheta[idx],2);
			// 		grad += pow(mTheta[idx-n2]-mTheta[idx],2);
			// 		mCONT[idx-n2] = acu + grad/deltaa2 + If*grad/deltaa2;

					//mCONT[idx] = acu ;
					//printf("check im=0 %f %f\n", mCONT[idx].real(), mCONT[idx].imag());

			//NEW TEST VERSION
			Float acu , grad , asu;
			size_t idx, idaux ;
			size_t iyP, iyM, ixP, ixM;
			Float mC0, mXp, mXm, mYp, mYm, mZp, mZm ;

			for (size_t iy=0; iy < n1; iy++)
			{
				iyP = (iy+1)%n1;
				iyM = (iy-1+n1)%n1;
				for (size_t ix=0; ix < n1; ix++)
				{
					ixP = (ix+1)%n1;
					ixM = (ix-1+n1)%n1;

					idx   = ix  + iy*n1+(iz+1)*n2 ;
					mC0   = mTheta[ix  + iy*n1+(iz+1)*n2] ;
					//idaux = ixP + iy*n1+(iz+1)*n2 ;
					mXp   = mTheta[ixP + iy*n1+(iz+1)*n2] ;
					//idaux = ixM + iy*n1+(iz+1)*n2 ;
					mXm   = mTheta[ixM + iy*n1+(iz+1)*n2] ;
					//idaux = ix + iyP*n1+(iz+1)*n2 ;
					mYp   = mTheta[ix + iyP*n1+(iz+1)*n2] ;
					//idaux = ix + iyM*n1+(iz+1)*n2 ;
					mYm   = mTheta[ix + iyM*n1+(iz+1)*n2] ;
					mZp   = mTheta[idx+n2] ;
					mZm   = mTheta[idx-n2] ;

					//KINETIC + POTENTIAL
					acu = mVeloc[idx-n2]*mVeloc[idx-n2]/2.f + z9QCD4*(1.f-cos(mC0*invz)) ;
					grad = 0.f ;
					//GRADIENTS
					asu = (mXp-mC0)*(mXp-mC0);
					if (asu < piz2)
					grad += asu;

					asu = (mXm-mC0)*(mXm-mC0);
					if (asu < piz2)
					grad += asu;

					asu = (mYp-mC0)*(mYp-mC0);
					if (asu < piz2)
					grad += asu;

					asu = (mYm-mC0)*(mYm-mC0);
					if (asu < piz2)
					grad += asu;

					asu = (mZp-mC0)*(mZp-mC0);
					if (asu < piz2)
					grad += asu;

					asu = (mZm-mC0)*(mZm-mC0);
					if (asu < piz2)
					grad += asu;

					//mCONT[idx-n2] = acu + grad/deltaa2 + If*grad/deltaa2;
					mCONT[idx-n2] = acu + grad/deltaa2 + If*0.f;

					toti += (double) mCONT[idx].real() ;
					if (mCONT[idx].real() > maxi)
					{
						maxi = mCONT[idx].real() ;
					}
				} //END X LOOP
			} //END Y LOOP
		} //END Z LOOP

		//printf("\n q2-%d",commRank());fflush(stdout);
		maxid = (double) maxi;

		MPI_Allreduce(&toti, &toti_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&maxid, &maxi_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		//printf("\n %d gets tot=%f(%f) max=%f(%f) av=%f(%f)",
		//						commRank(),toti,toti_global,maxid,maxi_global,toti_global/(n3*nSplit),
		//						maxi_global*n3*nSplit/toti_global);
		fflush(stdout);
		toti = toti/(n3*nSplit);
		toti_global = toti_global/(n3*nSplit) ;

		// NORMALISED DENSITY
		#pragma omp parallel for default(shared) schedule(static)
		for (size_t idx=n2; idx < n3 + n2 ; idx++)
		{
			mCONT[idx] = mCONT[idx]/((Float) toti_global)	;
			//printf("check im=0 %f %f\n", mCONT[idx].real(), mCONT[idx].imag());
		}
		//printf("\n q4-%d",commRank());fflush(stdout);

		maxid = maxid/toti_global ;
		maxi_global = maxi_global/toti_global ;

		// //LINEAR BINNING CONSTRAINED TO 100
		//
		// if (maxi_global >100.)
		// {
		// 	maxibin = 100.;
		// }
		// else
		// {
		// 	maxibin = maxi_global ;
		// }
		//
		// //BIN delta from 0 to maxi+1
		// size_t auxintarray[numbins] ;
		// for(size_t bin = 0; bin < numbins ; bin++)
		// {
		// (static_cast<double *> (contbin_local))[bin] = 0.;
		// auxintarray[bin] = 0;
		// }
		// Float norma = (Float) (maxi_global/(numbins-3)) ;
		// for(size_t i=n2; i < n3+n2; i++)
		// {
		// 	int bin;
		// 	bin = (mCONT[i].real()/norma)	;
		// 	//(static_cast<double *> (contbin))[bin+2] += 1. ;
		// 	if (bin<numbins)
		// 	{
		// 		auxintarray[bin] +=1;
		// 	}
		// }

		//LOG BINNING from log10(10^-5) to log(maxi_global)

		maxibin = log10(maxi_global) ;

		//BIN delta from 0 to maxi+1
		size_t auxintarray[numbins] ;
		for(size_t bin = 0; bin < numbins ; bin++)
		{
		(static_cast<double *> (contbin_local))[bin] = 0.;
		auxintarray[bin] = 0;
		}

		Float norma = (Float) ((maxibin+5.)/(numbins-3)) ;
		for(size_t i=n2; i < n3+n2; i++)
		{
			int bin;
			bin = (log10(mCONT[i].real())+5.)/norma	;
			//(static_cast<double *> (contbin))[bin+2] += 1. ;
			if (0<=bin<numbins)
			{
				auxintarray[bin] +=1;
			}
		}

		//printf("\n q7-%d",commRank());fflush(stdout);

		#pragma omp parallel for default(shared) schedule(static)
		for(size_t bin = 0; bin < numbins-3 ; bin++)
		{
			(static_cast<double *> (contbin_local))[bin+3] = (double) auxintarray[bin];
		}

		// NO BORRAR!
		// //PRINT 3D maps

		//WITH NO MPI THIS WORKS FOR OUTPUT
		if (commRank() ==0)
		{
						char stoCON[256];
						sprintf(stoCON, "out/con/con-%05d.txt", index);
						FILE *file_con ;
						file_con = NULL;
						file_con = fopen(stoCON,"w+");
						fprintf(file_con,  "# %d %f %f %f %f %f \n", sizeN, sizeL, sizeL/sizeN, zz, toti_global, z9QCD4 );

						//PRINT 3D maps
						#pragma omp parallel for default(shared) schedule(static)
						for (size_t idx = 0; idx < n3; idx++)
						{
							size_t ix, iy, iz;
								if (mCONT[idx+n2].real() > 100.)
								{
									iz = idx/n2 ;
									iy = (idx%n2)/n1 ;
									ix = (idx%n2)%n1 ;
									#pragma omp critical
									{
										fprintf(file_con,   "%d %d %d %f %f %f %f %f \n", ix, iy, iz,
										mCONT[idx+n2].real(), mCONT[idx+n2].imag(),
										mVeloc[idx+n2]*mVeloc[idx+n2]/(2.f*toti_global) ,
										z9QCD4*(1.f-cos(mTheta[idx+n2]/zz))/toti_global,
										mTheta[idx+n2]/zz ) ;
									}
								}
						}
						fclose(file_con);

		}//END PRINT COMRANK 0

		//printf("\n q8-%d",commRank());fflush(stdout);

	}
	else // FIELD_SAXION
	{
			// DO NOTHING
	}

	// 	SAVE AVERAGE
	//	MAXIMUM VALUE OF ENERGY CONTRAST
	//	MAXIMUM VALUE TO BE BINNED

	MPI_Reduce(contbin_local, (static_cast<double *> (contbin)), numbins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	(static_cast<double *> (contbin))[0] = toti_global;
	(static_cast<double *> (contbin))[1] = maxi_global;
	//note that maxibin is log10(maxContrast)
	(static_cast<double *> (contbin))[2] = (double) maxibin;

	if (commRank() ==0)
	printMpi("%(Edens = %f delta_max = %f) ", toti_global, maxi_global);
	fflush (stdout);
	commRank();
	return ;
}

//----------------------------------------------------------
//	DENSITY CONTRAST BINING USING VECTORS
//	IT ASSUMES M2 ALREADY CONTAINS A DENSITY DISTRIBUTION
//----------------------------------------------------------

template<typename Float>
void	Scalar::contrastbin(const Float zz, const int index, void *contbin, int numbins)
{
	LogMsg (VERB_HIGH, "Function contrastbin marked for optimization or removal");
	LogMsg (VERB_NORMAL, "contrastbin()");
	// THIS TEMPLATE DOES NO NEED TO BE CALLED FOLDED

	//	AUX VARIABLES
	Float maxi = 0.;
	Float maxibin = 0.;
	double maxid =0.;
	double toti = 0.;
	const Float z9QCD2 = axionmass2((*z), nQcd, zthres, zrestore )*pow((*z),2);

	//SUM variables
	double contbin_local[numbins] ;
	double toti_global;
	double maxi_global;

	//COMPUTES AVERAGE AND MAX
	//IF M2 IS ALREADY CONTRAST DOES NOT CHANGE ANYTHING

	if(fieldType == FIELD_AXION)
	{
		Float *mTheta = static_cast<Float*> (m);
		Float *mVeloc = static_cast<Float*> (v);

		//OLD VERSION energy was saved as complex
		//complex<Float> *mCONT = static_cast<complex<Float>*> (m2);
		//new version notice no padding
		Float *mCONT = static_cast<Float*> (m2);
		//printMpi("COMP-");
		#pragma omp parallel for default(shared) schedule(static) reduction(max:maxi), reduction(+:toti)
		for (size_t idx=n2; idx < n3+n2; idx++)
		{
				toti += (double) mCONT[idx] ;
				if (mCONT[idx] > maxi)
				{
					maxi = mCONT[idx] ;
				}
		}
		// compute local average density
		toti = toti / n3;
		// pointer for the maximum density in double
		maxid = (double) maxi;

		LogOut("RED-");
		//SUMS THE DENSITIES OF ALL RANGES
		MPI_Allreduce(&toti, &toti_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		//GLOBAL MAXIMUM DENSITY
		MPI_Allreduce(&maxid, &maxi_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		//printMpi("(TG %f,MG %f)",toti_global,maxi_global);

		fflush(stdout);
		//CONVERT SUM OF DENSITIES INTO GLOBAL AVERAGE [divide by number of ranges]
		toti_global = toti_global/(nSplit) ;

		// float version
		Float averagedens = (Float) toti_global ;

		//printMpi("NORM-");
		//DENSITIES ARE CONVERTED INTO DENSITY CONTRAST
		#pragma omp parallel for default(shared) schedule(static)
		for (size_t idx=n2; idx < n3+n2; idx++)
			{
				mCONT[idx] = mCONT[idx]/(averagedens)	;
				//printf("check im=0 %f %f\n", mCONT[idx].real(), mCONT[idx].imag());
			}

		//MAX DENSITY ARE CONVERTED INTO MAXIMUM DENSITY CONTRAST
		maxi_global = maxi_global/toti_global ;
		maxibin = log10(maxi_global) ;

		//printMpi("(MG %f,maxbinlog %f)BIN-",maxi_global,maxibin);

		//BIN delta from 0 to maxi+1
		//size_t auxintarray[numbins] ;
		#pragma omp parallel for default(shared) schedule(static)
		for(size_t bin = 0; bin < numbins ; bin++)
		{
		(static_cast<double *> (contbin_local))[bin] = 0.;
		//auxintarray[bin] = 0;
		}

		Float minbinfloat = (Float) 5.	;
		Float norma = (Float) ((maxibin + minbinfloat)/(numbins-3)) ;

		//printMpi("BIN2-(%f) ",norma);
		#pragma omp parallel for default(shared) schedule(static)
		for(size_t i=n2; i < n3+n2; i++)
		{
			Float caca = (log10(mCONT[i]) + minbinfloat)/norma	;
			int bin = caca	;
			if ( (bin>0) && (bin < numbins-4))
			{
				//auxintarray[bin] +=1;
				#pragma omp atomic
				(static_cast<double *> (contbin_local))[bin+3] += 1.;
			}
		}
		// printMpi("BIN3-");
		// #pragma omp parallel for default(shared) schedule(static)
		// for(size_t bin = 0; bin < numbins-3 ; bin++)
		// {
		// 	(static_cast<double *> (contbin_local))[bin+3] = (double) auxintarray[bin];
		// }

		// NO BORRAR!
		// //PRINT 3D maps
		//printMpi("PRI-");
		//WITH NO MPI THIS WORKS FOR OUTPUT
		if (commRank() ==0)
		{

						char stoCON[256];
						sprintf(stoCON, "out/con/con-%05d.txt", index);
						FILE *file_con ;
						file_con = NULL;
						file_con = fopen(stoCON,"w+");
						fprintf(file_con,  "# %d %f %f %f %f %f \n",
						sizeN, sizeL, sizeL/sizeN, zz, toti_global,
						axionmass2((*z), nQcd, zthres, zrestore )*pow((*z),4) );
						size_t cues = n1/shift ;

						//PRINT 3D maps
						#pragma omp parallel for default(shared) schedule(static)
						for (size_t idx = 0; idx < n3; idx++)
						{
							size_t ix, iy, iz;
							size_t sy, iiy, fidx ;
								if (mCONT[idx+n2] > 100.)
								{
									iz = idx/n2 ;
									iy = (idx%n2)/n1 ;
									ix = (idx%n2)%n1 ;

									if (folded)
									{
										sy = iy/(cues)	;
										iiy = iy - sy*cues;
										fidx = iz*n2+ iiy*n1*shift + ix*shift + sy ;
									}
									else
									{
										size_t fidx = idx;
									}

									#pragma omp critical
									{
										//PRINT COORDINATES CONTRAST AND [?]
										// fprintf(file_con,   "%d %d %d %f %f %f %f %f \n", ix, iy, iz,
										// mCONT[idx].real(), mCONT[idx].imag(),
										// 0., 0., 0.) ;
										fprintf(file_con,   "%d %d %d %f %f %f %f %f \n", ix, iy, iz,
										mCONT[idx+n2], 0.,
										mVeloc[fidx]*mVeloc[fidx]/(2.f*toti_global) ,
										z9QCD2*(1.f-cos(mTheta[fidx+n2]/zz))/toti_global,
										mTheta[fidx+n2]/zz ) ;
									}
								}
						}
						fclose(file_con);

		}//END PRINT COMRANK 0

		//printf("\n q8-%d",commRank());fflush(stdout);

	}
	else // FIELD_SAXION
	{
			// DO NOTHING
	}

	// 	SAVE AVERAGE
	//	MAXIMUM VALUE OF ENERGY CONTRAST
	//	MAXIMUM VALUE TO BE BINNED
	//printMpi("RED-");
	MPI_Reduce(contbin_local, (static_cast<double *> (contbin)), numbins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	(static_cast<double *> (contbin))[0] = toti_global;
	(static_cast<double *> (contbin))[1] = maxi_global;
	//note that maxibin is log10(maxContrast)
	(static_cast<double *> (contbin))[2] = (double) maxibin;

	if (commRank() ==0)
	printMpi("%(Edens = %.0f delta_max = %.0f) ", toti_global, maxi_global);
	fflush (stdout);
	commRank();
	return ;
}



void	Scalar::padder()
{
		LogMsg (VERB_NORMAL, "Function padder is deprecated and has been marked for removal");
		LogMsg (VERB_NORMAL, "PADDER");
		// idx goes into idx + 2(idx/Lx)
		// copy same iz,ix, i.e. by same value of idx/Lx = fresa

		char *mB = static_cast<char *>(m2);

		for (int fresa = n1*Lz-1; fresa >=0; fresa--)
			memcpy (mB + fSize*fresa*(n1+2), mB + fSize*fresa*n1, fSize*n1);
}


// ----------------------------------------------------------------------
// 		FUNCTIONS FOR MAX THETA [works but integrated with next]
// ----------------------------------------------------------------------

double	Scalar::maxtheta()//int *window)
{
	LogMsg (VERB_HIGH, "Function maxtheta is deprecated and has been marked for removal, use find in utils/binner.h instead");
	//LogMsg (VERB_NORMAL, "maxtheta()");
	double mymaxd = 0.;
	double mymaxd_global = 0.;
	if (precision == FIELD_DOUBLE)
	{

		double tauxd;
		#pragma omp parallel for reduction(max:mymaxd)
		for(size_t i=0; i < n3; i++)
		{
			if(fieldType == FIELD_SAXION)
			{
				tauxd = abs(arg(((complex<double> *) m)[i+n2]));
			}
			else // FIELD_AXION
			{
				tauxd = abs(((double *) m)[i+n2]/(*z));
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
				taux = abs(arg(((complex<float> *) m)[i+n2]));
			}
			else // FILD_AXION
			{
				taux = abs(((float *) m)[i+n2]/(*z));
			}
			if( taux > mymax )
			{
				mymax = taux ;
			}
		}
		mymaxd = (double) mymax;
	}
	MPI_Allreduce(&mymaxd, &mymaxd_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	return mymaxd_global ;
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
//		ADDS THE DISTRIBUTION OF RHO IF IN SAXION MODE

double	Scalar::thetaDIST(int numbins, void * thetabin)//int *window)
{
	LogMsg (VERB_HIGH, "Function thetaDIST has been deprecated, use the Binner class instead");
	double thetamaxi = maxtheta();
	printMpi("MAXTHETA=%f\n",thetamaxi);fflush(stdout);
//	printf("hallo von inside %f\n", thetamaxi);

	double n2p = numbins/thetamaxi;
	double thetabin_local[2*numbins];

	for(size_t i = 0; i < 2*numbins ; i++)
		thetabin_local[i] = 0.;

	if (precision == FIELD_DOUBLE)
	{
			if(fieldType == FIELD_SAXION)
			{
	//			#pragma omp parallel for default(shared)
				for(size_t i=0; i < n3 ; i++)
				{
					int bin;
					bin =  n2p*abs(arg(((complex<double> *) m)[i+n2]));
					(static_cast<double *> (thetabin_local))[bin] += 1.;
					bin =  numbins*(abs(((complex<double> *) m)[i+n2])/((*z)*2.));
					if (bin < numbins)
					(static_cast<double *> (thetabin_local))[bin+numbins] += 1.;
				}
			}
			else	//FIELD_AXION
			{
	//			#pragma omp parallel for default(shared)
				for(size_t i=0; i < n3; i++)
				{
					int bin;
					bin = n2p*abs(((double *) m)[i+n2]/(*z));
					if ((bin > 2*numbins) || (bin < 0))
						LogOut ("Warning: bin outside acceptable range\n");
					else
						thetabin_local[bin] += 1. ;
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
				(static_cast<double *> (thetabin_local))[bin] += 1.;
				bin =  numbins*(abs(((complex<float> *) m)[i+n2])/((*z)*2.));
				if (bin < numbins)
				{
				(static_cast<double *> (thetabin_local))[numbins + bin] += 1.;
				}
			}
		}
		else	//FIELD_AXION
		{
//			#pragma omp parallel for default(shared)
			for(size_t i=0; i < n3; i++)
			{
				int bin;
				bin = n2pf*abs(((float *) m)[i+n2]/(*z));

				if ((bin > 2*numbins) || (bin < 0))
					LogOut ("Warning: bin outside acceptable range\n");
				else
					thetabin_local[bin] += 1. ;
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
	MPI_Reduce(thetabin_local, thetabin, 2*numbins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	//if (commRank()==0)
	//printf("out = %f\n",thetamaxi);
	return thetamaxi ;
}

//--------------------------------------------------------------------
//		DENS M2 -> M
//--------------------------------------------------------------------

//	FOR FINAL OUTPUT, COPIES M2 INTO M

void	Scalar::denstom()//int *window)
{
	LogMsg (VERB_HIGH, "Function denstom marked for optimization or removal");
	//double thetamaxi = maxtheta();

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
					mTheta[idx] = mCONT[n2+idx].real();
				}

		}
		else //	FIELD_DOUBLE
		{
		double *mThetad = static_cast<double*> (m);
		complex<double> *mCONTd = static_cast<complex<double>*> (m2);

		#pragma omp parallel for default(shared)
			for(size_t idx=0; idx < n3; idx++)
				{
					mThetad[idx] = mCONTd[n2+idx].real();
				}
		}
		commSync();
		printMpi("dens to m ... done\n");

	}
	else
	{
		printMpi("dens to m not available for SAXION\n");
	}

}

//--------------------------------------------------------------------
//		DENS M2 -> M
//--------------------------------------------------------------------

//	FOR FINAL OUTPUT, COPIES M2 (AS COMPLEX) INTO M2 (AS REAL TO BE EXPORTED BY writeEDens)

void	Scalar::autodenstom2()//int *window)
{
	LogMsg (VERB_HIGH, "Function autodenstom marked for optimization or removal");
	//double thetamaxi = maxtheta();

//	printf("hallo von inside %f\n", thetamaxi);

	if(fieldType == FIELD_AXION)
	{
		if (precision == FIELD_SINGLE)
		{
		float *mCONTREAL = static_cast<float*> (m2);
		complex<float> *mCONT = static_cast<complex<float>*> (m2);

		//#pragma omp parallel for default(shared)
			for(size_t idx=0; idx < n3; idx++)
				{
					mCONTREAL[idx] = mCONT[idx].real();
				}

		}
		else //	FIELD_DOUBLE
		{
		double *mCONTREALd = static_cast<double*> (m);
		complex<double> *mCONTd = static_cast<complex<double>*> (m2);

		//#pragma omp parallel for default(shared)
			for(size_t idx=0; idx < n3; idx++)
				{
					mCONTREALd[idx] = mCONTd[idx].real();
				}
		}
		commSync();
		printMpi("m2 (complex) to m2 (real)... done\n");

	}
	else
	{
		printMpi("dens to m not available for SAXION\n");
	}

}

//----------------------------------------------------------------------
//		CHECK JUMPS
//----------------------------------------------------------------------

//	THIS FUNCTION CHECKS THETA IN ORDER AND NOTES DOWN POSITIONS WITH JUMPS OF 2 PI
//  MARKS THEM DOWN INTO THE ST BIN ARRAY AS POSSIBLE PROBLEMATIC POINTS WITH GRADIENTS
//  TRIES TO MEND THE THETA DISTRIBUTION INTO MANY RIEMMAN SHEETS TO HAVE A CONTINUOUS FIELD

void	Scalar::mendtheta()//int *window)
{
	LogMsg (VERB_HIGH, "Function mendtheta marked for optimization or removal");
//	make sure field unfolded
// 	make sure ghosts sent

if(fieldType == FIELD_AXION)
{

	int counter = 0;

	if (precision == FIELD_SINGLE)
	{
	float *mTheta = static_cast<float*> (m);
	float *mVeloc = static_cast<float*> (v);
	float zPi = (*z)*3.14159265359 ;
	float Pi2 = 6.283185307179586  ;
	float P0, PN;

	for(size_t zP=0; zP < Lz-1; zP++)
	{
		for(size_t yP=0; yP < n1-1; yP++)
		{
			for(size_t xP=0; xP < n1-1; xP++)
			{
				P0 = mTheta[n2*(1+zP)+ n1*yP +xP] ;
				PN = mTheta[n2*(1+zP)+ n1*yP +xP+1] ;

				if (abs((PN - P0)) > zPi)
				{
						counter++ ;
						for (int nn = 1 ; nn <6 ; nn++)
						{
							if(abs( nn*2*zPi + PN - P0) < zPi)
							{
								mTheta[n2*(1+zP)+ n1*yP + xP + 1] += nn*2*zPi   ;
								mVeloc[n2*(  zP)+ n1*yP + xP + 1] += nn*Pi2     ;
								//printf("(%f->%f)",mTheta[n2*(1+zP)+ n1*yP +xP],)
								break;
							}
							if(abs(-nn*2*zPi + PN - P0) < zPi)
							{
								mTheta[n2*(1+zP)+ n1*yP + xP + 1] += - nn*2*zPi ;
								mVeloc[n2*(  zP)+ n1*yP + xP + 1] += - nn*Pi2   ;
								break ;
							}
						}
				}

			}//END X LINE
			//PREPARE Y+1 POINT
			P0 = mTheta[n2*(1+zP) + n1*yP] 			;
			PN = mTheta[n2*(1+zP) + n1*yP +n1] ;

				if (abs((PN - P0)) > zPi)
				{
						counter++ ;
						for (int nn = 1 ; nn <6 ; nn++)
						{
							if(abs( nn*2*zPi + PN - P0) < zPi)
							{
								mTheta[n2*(1+zP) + n1*yP +n1] += nn*2*zPi ;
								mVeloc[n2*(  zP) + n1*yP +n1] += nn*Pi2 ;
								break;
							}
							if(abs(-nn*2*zPi + PN - P0) < zPi)
							{
								mTheta[n2*(1+zP) + n1*yP +n1] += - nn*2*zPi ;
								mVeloc[n2*(  zP) + n1*yP +n1] += - nn*Pi2   ;
								break ;
							}
						}
				}


		}//END Y LINE
		//PREPARE Z+1 POINT
		P0 = mTheta[n2*(1+zP) ] ;
		PN = mTheta[n2*(1+zP) + n2] ;

			if (abs((PN - P0)) > zPi)
			{
					counter++ ;
					for (int nn = 1 ; nn <6 ; nn++)
					{
						if(abs( nn*2*zPi + PN - P0) < zPi)
						{
							mTheta[n2*(1+zP) + n2] += nn*2*zPi ;
							mVeloc[n2*(  zP) + n2] += nn*Pi2   ;
							break;
						}
						if(abs(-nn*2*zPi + PN - P0) < zPi)
						{
							mTheta[n2*(1+zP) + n2] += - nn*2*zPi ;
							mVeloc[n2*(  zP) + n2] += - nn*Pi2   ;
							break ;
						}
					}
			}
		}
	}
	else //	FIELD_DOUBLE
	{
	}
	commSync();
	printMpi("mendtheta done, mends = %d\n", counter);

}
else
{
	printMpi("mendtheta not available for SAXION\n");
}

}

//----------------------------------------------------------------------
//		AXITON FINDER
//----------------------------------------------------------------------
void	Scalar::writeAXITONlist (double contrastthreshold, void *idxbin, int numaxitons)
{
	LogMsg (VERB_HIGH, "Function writeAXITONlist marked for optimization or removal");
	switch	(precision)
	{
		case	FIELD_DOUBLE:
		{
		}
		break;

		case	FIELD_SINGLE:
		{
			//COMPUTES JAVIER DENSITY MAP AND BINS
			//energymapTheta (static_cast<float>(zzz), index, contbin, numbins); // TEST

			//USES WHATEVER IS IN M2, COMPUTES CONTRAST AND BINS []
			// USE WITH ALEX'FUNCTION
			axitonfinder(static_cast<float>(contrastthreshold), idxbin, numaxitons);
		}
		break;

		default:
		printf("Unrecognized precision\n");
		exit(1);
		break;
	}

}

// READS A DENSITY GRID IN M2 AND SEARCHES FOR LOCAL MAXIMA ABOVE A GIVEN CONTRAST
// RETURNS COORDINATES FOLDED AND UNFOLDED TO A POINTER

template<typename Float>
void	Scalar::axitonfinder(Float contrastthreshold, void *idxbin, int numaxitons)
{
	LogMsg (VERB_HIGH, "Function axitonfinder marked for optimization or removal");

	//array for idx
	size_t ar_local[numaxitons] ;
	//array for contrast comparisons
	Float  ct_local[numaxitons] ;
	// for folding issues
	size_t cues = n1/shift ;


	for(int i = 0; i < numaxitons ; i++)
	{
		ar_local[i] = 0 ;
		ct_local[i] = 0. ;
	}

	if(fieldType == FIELD_AXION)
	{
		//mCONT assumed normalised // i.e. contrastbin was called before
		Float *mCONT = static_cast<Float*> (m2);
		Float *mTheta = static_cast<Float*> (m);
		//float *mVeloc = static_cast<float*> (v);

		int size = 0 ;
		size_t iyP, iyM, ixP, ixM;
		size_t fidx ;
		size_t iz, iy, ix ;
		size_t idaux, ixyzAux	;

//		#pragma omp parallel for default(shared) schedule(static)
		for (size_t idx = 0; idx < n3; idx++)
		{
			//size_t ix, iy, iz;
			// ONE CAN USE DENSITY CONTRAST BUT HF FLUCTUATIONS MASK THE AXITONS
			// IT IS PERHAPS GOOD TO USE THE THETA FIELD INSTEAD
				if (mCONT[idx]> contrastthreshold)
				{
					// iz = idx/n2 ;
					// iy = (idx%n2)/n1 ;
					// ix = (idx%n2)%n1 ;

					Float val = mCONT[idx];

					iz = idx/n2 ;
					iy = (idx%n2)/n1 ;
					ix = (idx%n2)%n1 ;

					if (folded)
					{
						size_t sy = iy/(cues)	;
						size_t iiy = iy - sy*cues;
						fidx = iz*n2+ iiy*n1*shift + ix*shift + sy ;
					}
					else
					{
						fidx = idx;
					}

					if (abs(mTheta[n2+fidx]/(*z)) < 3.14)
					continue;

					ixyzAux = (ix+1)%n1;
					idaux = ixyzAux + iy*n1+(iz)*n2 ;
					if (mCONT[idaux] - val < 0)
					continue;

					ixyzAux = (ix-1+n1)%n1;
					idaux = ixyzAux + iy*n1+(iz)*n2 ;
					if (mCONT[idaux] - val < 0)
					continue;

					ixyzAux = (iy+1)%n1;
					idaux = ix + ixyzAux*n1+(iz)*n2 ;
					if (mCONT[idaux] - val < 0)
					continue;

					ixyzAux = (iy-1+n1)%n1;
					idaux = ix + ixyzAux*n1+(iz)*n2 ;
					if (mCONT[idaux] - val < 0)
					continue;

					// I CANNOT CHECK THE Z DIRECTION BECAUSE I DO NOT HAVE GHOSTS
					// IN THE CURRENT MPI IMPLEMENTATION
					// I NEVERTHELESS USE THIS WITHOUT MPI
					// CHANGE M2 TO HAVE GHOSTS? FFT, ETC...

					ixyzAux = (iz+1)%Lz;
					idaux = ix + iy*n1+(ixyzAux)*n2 ;
					if (mCONT[idaux] - val > 0)
					continue;

					ixyzAux = (iz-1+Lz)%Lz;
					idaux = ix + iy*n1+(ixyzAux)*n2 ;
					if (mCONT[idaux] - val > 0)
					continue;

					// IF IT REACHED HERE IT IS REALLY A MAXIMUM OF DENSITY
					// AND HAS A LARGE AMPLITUDE
				//	#pragma omp critical
				//	{
		   				int pos = size;
		   				while (pos > 0 && ct_local[pos - 1] < val)
							{
		      			pos--;
		      			if (pos < numaxitons-1)
								{
									ct_local[pos + 1] = ct_local[pos];
									ar_local[pos + 1] = ar_local[pos];
								}
		   				}
		   				if (size < numaxitons) size++;
		   				if (pos < size)
							{
								ct_local[pos] = val;
								ar_local[pos] = fidx ;
							}
				//	}

				}
		}
	}
	else
	{
		printMpi("axiton finder not available in SAXION mode\n");
		return;
	}

	printMpi("%d axitons: ", numaxitons );
	for(int i = 0; i<numaxitons; i++)
	{
		printMpi("%f(%d)", ct_local[i], ar_local[i]);
	}
	printMpi("\n");


	for(int i = 0; i < numaxitons ; i++)
	{
		(static_cast<size_t *> (idxbin))[i] = ar_local[i];
	}
	return ;
}


//----------------------------------------------------------------------
//		HALO UTILITIES
//----------------------------------------------------------------------

void	Scalar::loadHalo()
{
	LogOut   ("Deprecated function loadHalo(), marked for removal\n");
	LogError ("Deprecated function loadHalo(), marked for removal");

	printf("initFFThalo sending fSize=%d, n1=%d, Tz=%d\n", fSize, n1, Tz);
	// printf("| free v ");fflush(stdout);
	// trackFree(&v, ALLOC_ALIGN);
	//
 // 	const size_t	mBytes = v3*fSize;
 // 	printf("| realoc m2 ");
	// v = (fftwf_complex*) fftwf_malloc(sizeN*sizeN*(sizeN/2+1) * sizeof(fftwf_complex));
	//

//	initFFThalo(static_cast<void *>(static_cast<char *> (m) + n2*fSize), v, n1, Tz, precision);

}
/*
void	Scalar::fftCpuHalo	(FFTdir sign)
{

//	runFFThalo(sign);
}*/
