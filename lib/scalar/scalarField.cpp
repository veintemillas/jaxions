#include<cstdlib>
#include<cstring>
#include<complex>
#include<chrono>

#include"enum-field.h"
#include"fft/fftCuda.h"
#include"fft/fftCode.h"

#include"scalar/scalarField.h"

#include"comms/comms.h"
//#include"scalar/varNQCD.h"

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


	Scalar::Scalar(Cosmos *cm, const size_t nLx, const size_t nLz, FieldPrecision prec, DeviceType dev, const double zI, bool lowmem, const int nSp, FieldType newType, LambdaType lType, size_t Ngg)
		: n1(nLx), n2(nLx*nLx), n3(nLx*nLx*nLz), Lz(nLz), Tz(Lz*nSp), nSplit(nSp), Ng(Ngg),  Ez(nLz + 2*Ngg), v3(nLx*nLx*(nLz + 2*Ngg)),
		  device(dev), precision(prec), fieldType(newType), lambdaType(lType), lowmem(lowmem)
{
	Profiler &prof = getProfiler(PROF_SCALAR);

	prof.start();

	LogMsg(VERB_NORMAL,"[sca] Constructor Scalar");
	LogMsg(VERB_NORMAL,"[sca] Lx           =  %d",nLx);
	LogMsg(VERB_NORMAL,"[sca] Lz           =  %d",nLz);
	LogMsg(VERB_NORMAL,"[sca] nSplit       =  %d ",nSplit);
	LogMsg(VERB_NORMAL,"[sca] Field Type   =  %d (SAX/AX/WKB %d/%d/%d) ",newType,FIELD_SAXION,FIELD_AXION,FIELD_WKB);
	// LogMsg(VERB_NORMAL,"[sca] Lambda Type  =  %d (FIXED/Z2 %d/%d) ",lType,LAMBDA_FIXED,LAMBDA_Z2);
	LogMsg(VERB_NORMAL,"[sca] Precision    =  %d (SINGLE/DOUBLE %d/%d)",prec,FIELD_SINGLE,FIELD_DOUBLE);
	LogMsg(VERB_NORMAL,"[sca] Device       =  %d (CPU/GPU %d/%d)",dev,DEV_CPU,DEV_GPU);
	LogMsg(VERB_NORMAL,"[sca] Lowmem       =  %d ",lowmem);
	LogMsg(VERB_NORMAL,"[sca] Nghost       =  %d ", Ngg);

	if (cm == nullptr) {
		LogError("Error: no cosmological background defined!. Will exit with errors.");
		prof.stop();
		endComms();
		exit(1);
	}

	bckgnd = cm;
	// Ng = cm->ICData().Nghost ;
	// Ez = nLz + 2*Ng;
	// v3 = nLx*nLx*(nLz + 2*Ng);
	size_t nData;

	msa = sqrt(2.*bckgnd->Lambda())*bckgnd->PhysSize()/((double) nLx);


	LogMsg(VERB_NORMAL,"[sca] ZThRes  () %f",cm->ZThRes  ());
	LogMsg(VERB_NORMAL,"[sca] ZRestore() %f",cm->ZRestore());
	LogMsg(VERB_NORMAL,"[sca] PhysSize() %f",cm->PhysSize());
	LogMsg(VERB_NORMAL,"[sca] Lambda  () %f",cm->Lambda  ());
	LogMsg(VERB_NORMAL,"[sca] LamZ2Exp() %f",cm->LamZ2Exp());
	LogMsg(VERB_NORMAL,"[sca] Indi3   () %f",cm->Indi3   ());
	LogMsg(VERB_NORMAL,"[sca] Gamma   () %f",cm->Gamma   ());
	LogMsg(VERB_NORMAL,"[sca] QcdExp  () %f",cm->QcdExp  ());
	LogMsg(VERB_NORMAL,"[sca] QcdPot  () %d",cm->QcdPot  ());
	LogMsg(VERB_NORMAL,"[sca] Frw     () %f",cm->Frw     ());
	LogMsg(VERB_NORMAL,"[sca] Mink    () %d",cm->Mink    ());

	LogMsg(VERB_NORMAL,"[sca] ic.Nghost   %d",cm->ICData().Nghost   );
	LogMsg(VERB_NORMAL,"[sca] ic.icdrule  %d",cm->ICData().icdrule  );
	LogMsg(VERB_NORMAL,"[sca] ic.preprop  %d",cm->ICData().preprop  );
	LogMsg(VERB_NORMAL,"[sca] ic.icstudy  %d",cm->ICData().icstudy  );
	LogMsg(VERB_NORMAL,"[sca] ic.prepstL  %f",cm->ICData().prepstL  );
	LogMsg(VERB_NORMAL,"[sca] ic.prepcoe  %f",cm->ICData().prepcoe  );
	LogMsg(VERB_NORMAL,"[sca] ic.pregammo %f",cm->ICData().pregammo );
	LogMsg(VERB_NORMAL,"[sca] ic.prelZ2e  %f",cm->ICData().prelZ2e  );
	LogMsg(VERB_NORMAL,"[sca] ic.prevtype %d",cm->ICData().prevtype );
	LogMsg(VERB_NORMAL,"[sca] ic.normcore %d",cm->ICData().normcore );
	LogMsg(VERB_NORMAL,"[sca] ic.alpha    %f",cm->ICData().alpha    );
	LogMsg(VERB_NORMAL,"[sca] ic.siter    %d",cm->ICData().siter    );
	LogMsg(VERB_NORMAL,"[sca] ic.kcr      %f",cm->ICData().kcr      );
	LogMsg(VERB_NORMAL,"[sca] ic.kMax     %d",cm->ICData().kMax     );
	LogMsg(VERB_NORMAL,"[sca] ic.mode0    %f",cm->ICData().mode0    );
	LogMsg(VERB_NORMAL,"[sca] ic.zi       %f",cm->ICData().zi       );
	LogMsg(VERB_NORMAL,"[sca] ic.logi     %f",cm->ICData().logi     );
	LogMsg(VERB_NORMAL,"[sca] ic.cType    %d",cm->ICData().cType    );
	LogMsg(VERB_NORMAL,"[sca] ic.smvarTy  %d",cm->ICData().smvarType);
	LogMsg(VERB_NORMAL,"[sca] ic.mocoty   %d",cm->ICData().mocoty   );

	folded 	   = false;
	eReduced   = false;
	mmomspace 	 = false;
	vmomspace 	 = false;
	gsent = false;
	grecv = false;

	setCO(Ng);


	switch (fieldType)
	{
		case FIELD_SAXION:
		case FIELD_SX_RD:
			nData = 2;
			break;

		case FIELD_AXION_MOD:
		case FIELD_AX_MOD_RD:
		case FIELD_AXION:
		case FIELD_AX_RD:
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

//	switch	(dev)
//	{
//		case DEV_CPU:
			#ifdef	__AVX512F__
			LogMsg(VERB_NORMAL, "[sca] Using AVX-512 64 bytes alignment");
			mAlign = 64;
			#elif	defined(__AVX__) || defined(__AVX2__)
			LogMsg(VERB_NORMAL, "[sca] Using AVX 32 bytes alignment");
			mAlign = 32;
			#else
			LogMsg(VERB_NORMAL, "[sca] Using SSE 16 bytes alignment");
			mAlign = 16;
			#endif
//			break;

//		case DEV_GPU:
//			LogMsg(VERB_NORMAL, "Using 16 bytes alignment for the Gpu");
//			mAlign = 16;
//			break;
//	}

	shift = mAlign/fSize;

	if (n2*fSize % mAlign)
	{
		LogError("Error: misaligned memory. Are you using an odd dimension?");
		exit(1);
	}

	LogMsg(VERB_NORMAL, "[sca] v3 %d n3 %d", v3, (n2*(nLz + 2)));
	const size_t	mBytes = v3*fSize;
	const size_t	vBytes = (n2*(nLz + 2))*fSize;


	switch (fieldType)
	{
		case FIELD_SAXION:
		case FIELD_SX_RD:
			LogMsg(VERB_NORMAL, "[sca] allocating m, v, sData");
			alignAlloc ((void**) &m,   mAlign, mBytes);
			alignAlloc ((void**) &v,   mAlign, vBytes);
			trackAlloc ((void**) &str, n3);
			break;

		case FIELD_AXION_MOD:
		case FIELD_AX_MOD_RD:
		case FIELD_AXION:
		case FIELD_AX_RD:
		case FIELD_WKB:
			str = nullptr;
			//this allocates a slightly larger v to host FFTs in place
			LogMsg(VERB_NORMAL, "[sca] allocating m, v");
			alignAlloc ((void**) &m, mAlign, mBytes+vBytes);
			v = static_cast<void *>(static_cast<char *>(m) + mBytes );
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
				LogMsg(VERB_NORMAL, "[sca] allocating m2");
				alignAlloc ((void**) &m2, mAlign, mBytes);
				memset (m2, 0, fSize*v3);
			} else
				m2 = nullptr;
			break;

		case FIELD_AXION_MOD:
		case FIELD_AXION:
			LogMsg(VERB_NORMAL, "[sca] allocating m2");
			alignAlloc ((void**) &m2, mAlign, 2*mBytes);
			memset (m2, 0, 2*fSize*n3);
			break;

		case FIELD_SX_RD:
		case FIELD_AX_MOD_RD:
		case FIELD_AX_RD:
		case FIELD_WKB:
			m2 = nullptr;
			break;

		default:
			LogError("Error: unrecognized field type");
			exit(1);
			break;
	}

	statusM2 = M2_DIRTY;

	if (m == nullptr)
	{
		LogError ("Error: couldn't allocate %lu bytes on host for the m field", mBytes);
		exit(1);
	}

	if (v == nullptr)
	{
		LogError ("Error: couldn't allocate %lu bytes on host for the v field", vBytes);
		exit(1);
	}

	if (str == nullptr && (fieldType & (FIELD_SAXION != 0)))
	{
		LogError ("Error: couldn't allocate %lu bytes on host for the string map", n3);
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

	memset (m, 0, fSize*v3);
	memset (v, 0, fSize*(n2*(nLz + 2)));

	commSync();

	LogMsg(VERB_NORMAL, "[sca] allocating z, R");
	alignAlloc ((void **) &z, mAlign, mAlign);
	alignAlloc ((void **) &R, mAlign, mAlign);

	if (z == nullptr)
	{
		LogError ("Error: couldn't allocate %d bytes on host for the z field", sizeof(double));
		exit(1);
	}

	if (R == nullptr)
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
		if (fieldType == FIELD_SAXION) {
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
		} else {
			if (cudaMalloc(&m_d, 2*mBytes) != cudaSuccess)
			{
				LogError ("Error: couldn't allocate %lu bytes for the gpu field m", mBytes);
				exit(1);
			}

			v_d = static_cast<void *>(static_cast<char *>(m_d) + fSize*v3);
		}

		if (!lowmem || (fieldType & FIELD_AXION))
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

	/* Note the big difference zI is an obsolete parameter FIX ME */
	*z = cm->ICData().zi; //*z = zI;
	*R = 1.0;
	updateR();


	prof.stop();
	prof.add(std::string("Init Allocation"), 0.0, 0.0);

	/*	WKB fields won't trigger configuration read or FFT initialization	*/

	if (fieldType != FIELD_WKB && !(fieldType & FIELD_REDUCED)) {
		prof.start();
		AxionFFT::initFFT(prec);

		/* Backward needed for reduce-filter-map */
		AxionFFT::initPlan (this, FFT_PSPEC_AX,  FFT_FWDBCK, "pSpecAx");		// Spectrum for axion

		if (fieldType == FIELD_SAXION) {
			if (!lowmem) {
				AxionFFT::initPlan (this, FFT_SPSX,       FFT_FWDBCK,     "SpSx");
				AxionFFT::initPlan (this, FFT_PSPEC_SX,   FFT_FWDBCK,  "pSpecSx");
				AxionFFT::initPlan (this, FFT_RDSX_V,     FFT_FWDBCK,    "RdSxV");
				AxionFFT::initPlan (this, FFT_RHO_SX,     FFT_FWDBCK,    "RhoSx");
				AxionFFT::initPlan (this, FFT_CtoC_MtoM2, FFT_FWD,    "nSpecSxM");	// Only possible if lowmem == false
				AxionFFT::initPlan (this, FFT_CtoC_VtoM2, FFT_FWD,    "nSpecSxV");
			}
		}

		AxionFFT::initPlan (this, FFT_SPAX,       FFT_FWDBCK,  "SpAx");

		/* If spectral initSpectral plans
		at the moment this is always done which avoids some issues
		when reading configurations without the explicit flag */
		// AxionFFT::initPlan (this, FFT_SPSX,       FFT_FWDBCK,     "SpSx");

		/* If fspectral initSpectral plans*/
		if (fpectral) {
			LogMsg(VERB_NORMAL,"Initialising fspectral plans");
			// Saxion m inplace
			AxionFFT::initPlan (this, FFT_CtoC_MtoM, FFT_FWDBCK, "C2CM2M");
			// Saxion v inplace
			AxionFFT::initPlan (this, FFT_CtoC_VtoV, FFT_FWDBCK, "C2CV2V");
			AxionFFT::initPlan(this, FFT_CtoC_M2toM2, FFT_FWDBCK, "C2CM22M2");
			AxionFFT::initPlan(this, FFT_CtoC_M2toM, FFT_FWDBCK, "C2CM22M");
			// Axion m/v inplace
			AxionFFT::initPlan (this, FFT_RtoC_MtoM_WKB,  FFT_FWDBCK, "R2CM2M");
			AxionFFT::initPlan (this, FFT_RtoC_VtoV_WKB,  FFT_FWDBCK, "R2CV2V");
		}
		/*	If present, read fileName	*/

		ConfType cType = cm->ICData().cType;
		if (cType == CONF_NONE) {
			LogMsg (VERB_HIGH, "No configuration selected. Hope we are reading from a file...");

			if (fIndex == -1) {
				LogError ("Error: neither file nor initial configuration specified");
				exit(2);
			}

			prof.stop();
			prof.add(std::string("Init FFT"), 0.0, 0.0);
		} else {
			if (fieldType & FIELD_AXION) {
				LogError ("Configuration generation for axion fields not supported");
				prof.stop();
				prof.add(std::string("Init FFT"), 0.0, 0.0);
			} else {
				if ( !(cType == CONF_SMOOTH) ) {
					if (lowmem)
						AxionFFT::initPlan (this, FFT_CtoC_MtoM,  FFT_FWDBCK, "Init");
					else
						AxionFFT::initPlan (this, FFT_CtoC_MtoM2, FFT_FWDBCK, "Init");
				}
				prof.stop();

				prof.add(std::string("Init FFT"), 0.0, 0.0);
				genConf	(cm, this);
			}
		}
	}
}

// END SCALAR

	Scalar::~Scalar()
{
	commSync();
	LogMsg (VERB_HIGH, "Rank %d Calling destructor...",commRank());

	bckgnd = nullptr;

	if (m != nullptr)
		trackFree(m);

	if (v != nullptr && (fieldType & FIELD_SAXION))
		trackFree(v);

	if (m2 != nullptr)
		trackFree(m2);

	if (str != nullptr)
		trackFree(str);

	if (z != nullptr)
		trackFree((void *) z);

	if (R != nullptr)
		trackFree((void *) R);

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
		#endif
	}

	if ((fieldType & FIELD_REDUCED) == false)
		AxionFFT::closeFFT();
}

void	Scalar::transferDev(FieldIndex fIdx)	// Transfers only the internal volume
{
	if (device == DEV_GPU)
	{
		#ifndef	USE_GPU
			LogError ("Error: gpu support not built");
			exit   (1);
		#else
			if (fIdx & FIELD_M)
				cudaMemcpy((((char *) m_d) + n2*fSize), (((char *) m) + n2*fSize),  n3*fSize, cudaMemcpyHostToDevice);

			if (fIdx & FIELD_V)
				cudaMemcpy(v_d,  v,  n3*fSize, cudaMemcpyHostToDevice);

			if ((fIdx & FIELD_M2) && (!lowmem))
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
			if (fIdx & FIELD_M)
				cudaMemcpy(m,  m_d,  v3*fSize, cudaMemcpyDeviceToHost);

			if (fIdx & FIELD_V)
				cudaMemcpy(v,  v_d,  n3*fSize, cudaMemcpyDeviceToHost);

			if ((fIdx & FIELD_M2) && (!lowmem))
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
			if (fIdx & FIELD_M) {
				cudaMemcpyAsync(static_cast<char *> (m) + n2*fSize, static_cast<char *> (m_d) + n2*fSize, n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[0]);
				cudaMemcpyAsync(static_cast<char *> (m) + n3*fSize, static_cast<char *> (m_d) + n3*fSize, n2*fSize, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[1]);
			} else {
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
			if (fIdx & FIELD_M) {
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

void	Scalar::sendGhosts(FieldIndex fIdx, CommOperation opComm, size_t Nng)
{
	static const int rank = commRank();
	static const int fwdNeig = (rank + 1) % nSplit;
	static const int bckNeig = (rank - 1 + nSplit) % nSplit;

	/* we can send 1 (for energy) , or Ng (for propagator)*/
	const int ghostBytes = Ng*n2*fSize;

	static MPI_Request 	rSendFwd, rSendBck, rRecvFwd, rRecvBck;	// For non-blocking MPI Comms

	/* Assign receive buffers to the right parts of m, v */
LogMsg(VERB_DEBUG,"[sca] Called send Ghosts (COMM %d) slice %lu Ng %d",opComm, Nng, Ng);LogFlush();
	void *sGhostBck, *sGhostFwd, *rGhostBck, *rGhostFwd;

	if (fIdx & FIELD_M)
	{
		if (Nng > 0){
			//FIX ME in case one needs to transfer other slices with Nng
			sGhostBck = mStart();																																						//slice to be send back
			sGhostFwd = static_cast<void *> (static_cast<char *> (mStart()) + fSize*n3-ghostBytes);					//slice to be send forw
			rGhostBck = static_cast<void *> (static_cast<char *> (mFrontGhost()) + fSize*Ng*n2-ghostBytes);	//reception point
			rGhostFwd = static_cast<void *> (static_cast<char *> (mBackGhost()) + fSize*Ng*n2-ghostBytes);	//reception point
		} else {
			// from v to 1st slice for NNEIG; assumes Ng = 1
			sGhostBck = vGhost(); 																													//slice to be send back
			sGhostFwd = static_cast<void *> (static_cast<char *> (vGhost()) + fSize*(n2));	//slice to be send forw
			rGhostBck = mFrontGhost();
			rGhostFwd = mBackGhost();
		}
	}
	else
	{
		if (Nng > 0){
			sGhostBck = m2Start();
			sGhostFwd = static_cast<void *> (static_cast<char *> (m2Start()) + fSize*n3-ghostBytes);
			rGhostBck = static_cast<void *> (static_cast<char *> (m2FrontGhost()) + fSize*Ng*n2-ghostBytes);		//reception point
			rGhostFwd = static_cast<void *> (static_cast<char *> (m2BackGhost()) + fSize*Ng*n2-ghostBytes);	//reception point
		} else {
			// from v to 1st slice for NNEIG; assumes Ng = 1
			sGhostBck = vGhost();																									 					//slice to be send back
			sGhostFwd = static_cast<void *> (static_cast<char *> (vGhost()) + fSize*(n2));	//slice to be send forw
			rGhostBck = m2FrontGhost();
			rGhostFwd = m2BackGhost();
		}
	}


	switch	(opComm)
	{
		case	COMM_SEND:
LogMsg(VERB_PARANOID,"[COMM_TESTS] SEND");
			MPI_Send_init(sGhostFwd, ghostBytes, MPI_BYTE, fwdNeig, 2*rank,   MPI_COMM_WORLD, &rSendFwd);
			MPI_Send_init(sGhostBck, ghostBytes, MPI_BYTE, bckNeig, 2*rank+1, MPI_COMM_WORLD, &rSendBck);

			MPI_Start(&rSendFwd);
			MPI_Start(&rSendBck);
 			gsent = false;

			break;

		case	COMM_RECV:
LogMsg(VERB_PARANOID,"[COMM_TESTS] RECV");
			MPI_Recv_init(rGhostFwd, ghostBytes, MPI_BYTE, fwdNeig, 2*fwdNeig+1, MPI_COMM_WORLD, &rRecvFwd);
			MPI_Recv_init(rGhostBck, ghostBytes, MPI_BYTE, bckNeig, 2*bckNeig,   MPI_COMM_WORLD, &rRecvBck);

			MPI_Start(&rRecvBck);
			MPI_Start(&rRecvFwd);
 			grecv = false;
			break;

		case	COMM_SDRV:
LogMsg(VERB_PARANOID,"[COMM_TESTS] SDRV");
			MPI_Send_init(sGhostFwd, ghostBytes, MPI_BYTE, fwdNeig, 2*rank,   MPI_COMM_WORLD, &rSendFwd);
			MPI_Send_init(sGhostBck, ghostBytes, MPI_BYTE, bckNeig, 2*rank+1, MPI_COMM_WORLD, &rSendBck);
			MPI_Recv_init(rGhostFwd, ghostBytes, MPI_BYTE, fwdNeig, 2*fwdNeig+1, MPI_COMM_WORLD, &rRecvFwd);
			MPI_Recv_init(rGhostBck, ghostBytes, MPI_BYTE, bckNeig, 2*bckNeig,   MPI_COMM_WORLD, &rRecvBck);

			MPI_Start(&rRecvBck);
			MPI_Start(&rRecvFwd);
			MPI_Start(&rSendFwd);
			MPI_Start(&rSendBck);
			gsent = false;
			grecv = false;
			break;

		case	COMM_TESTS:
		{
		int flag1 = 0, flag2 = 0;
		int pest1 = MPI_Test(&rSendBck, &flag1, MPI_STATUS_IGNORE);
		int pest2 = MPI_Test(&rSendFwd, &flag2, MPI_STATUS_IGNORE);
		if (flag1 * flag2)
			gsent = true;
		else  gsent = false ;
LogMsg(VERB_PARANOID,"[COMM_TESTS] flag1/2 %d/%d [%d/%d] > gsent %d",flag1,flag2,pest1,pest2,gsent);
		}
		break;

	case	COMM_TESTR:
		{
		int flag1 = 0, flag2 = 0;
		int pest1 = MPI_Test(&rRecvFwd, &flag1, MPI_STATUS_IGNORE);
		int pest2 = MPI_Test(&rRecvBck, &flag2, MPI_STATUS_IGNORE);
		if (flag1 * flag2)
			grecv = true;
		else  grecv = false;
LogMsg(VERB_PARANOID,"[COMM_TESTR] flag1/2 %d/%d [%d/%d] > grecv %d",flag1,flag2,pest1,pest2,grecv);
		}
		break;

	case	COMM_WAIT:
LogMsg(VERB_PARANOID,"[COMM_TESTS] WAIT");
		MPI_Wait(&rSendFwd, MPI_STATUS_IGNORE);
		MPI_Wait(&rSendBck, MPI_STATUS_IGNORE);
		MPI_Wait(&rRecvFwd, MPI_STATUS_IGNORE);
		MPI_Wait(&rRecvBck, MPI_STATUS_IGNORE);
		gsent = true;
		grecv = true;
		MPI_Request_free(&rSendFwd);
		MPI_Request_free(&rSendBck);
		MPI_Request_free(&rRecvFwd);
		MPI_Request_free(&rRecvBck);
LogMsg(VERB_PARANOID,"[COMM_TESTS] FREE");
		break;

	}
}

void	Scalar::exchangeGhosts(FieldIndex fIdx)
{
LogMsg(VERB_PARANOID,"[sca] Exchange Ghosts");LogFlush();
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
		case FIELD_AXION_MOD:
		case FIELD_AXION:

				fSize /= 2;

				if (device != DEV_GPU)
					shift *= 2;

		break;

		case	FIELD_SAXION:
			if (fieldType & FIELD_AXION)
				LogError ("Error: transformation from axion to saxion not supported");
			else
				fieldType = FIELD_SAXION;
			break;

		default:
			LogError ("Error: transformation not supported");
			break;

	}
	fieldType = newType;
}

void	Scalar::setFolded (bool foli)
{
	folded = foli;
}

void	Scalar::setMMomSpace (bool foli)
{
	mmomspace = foli;
}
void	Scalar::setVMomSpace (bool foli)
{
	vmomspace = foli;
}
void	Scalar::setReduced (bool eRed, size_t nLx, size_t nLz)
{
	eReduced = eRed;

	if (eRed == true) {
		rLx = nLx;
		rLz = nLz;
	} else {
		rLx = n1;
		rLz = Lz;
	}
}

void	Scalar::updateR ()
{
	// updates scale factor R = z^frw
	// Minkowski frw = 0, Radiation frw = 1,
	// if Minkowski R=1 and there is no need
	// by default R=z and there is no need (one uses z for all purposes)
	if (!bckgnd->Mink())
		*R = pow(*z,frw);
}

double	Scalar::Rfromct (const double ct)
{
	// Returns scale factor R = z^frw for any conformal time ct
	// Minkowski frw = 0, Radiation frw = 1,
	return pow(ct,frw);
}

double	Scalar::LambdaP ()
{
	// Returns The value of Lambda with PRS trick IF needed
	// Minkowski frw = 0, Radiation frw = 1,
	double lbd  = bckgnd->Lambda();
	double llee = bckgnd->LamZ2Exp();
LogMsg(VERB_PARANOID,"[sca:LambdaP] LambdaPhysical %f Le %f",lbd,llee);
	if (LambdaT() == LAMBDA_FIXED)
		return  lbd;
	else if (LambdaT() == LAMBDA_Z2)
		return  lbd/pow(*R,llee);
}

double	Scalar::Msa ()
{
	// Returns The value of Msa with PRS trick, or Physical strings
	// Minkowski frw = 0, Radiation frw = 1,
	// double &lbd = bckgnd->Lambda();
	// double llee = bckgnd->LamZ2Exp();
	// if (LambdaT() == LAMBDA_FIXED)
	// 	return  sqrt(2.0*LambdaP()) * (*R) * bckgnd->PhysSize()/Length() ;
	// else if (LambdaT() == LAMBDA_Z2)
LogMsg(VERB_PARANOID,"[sca:msa] LambdaPhysical %f ",LambdaP() );
		return  sqrt(2.0*LambdaP()) * (*R) * bckgnd->PhysSize()/Length() ;
}

double  Scalar::HubbleMassSq  ()
{
	// R''/R = frw(frw-1)/z^2
	// since we have R=z^frw
	//except in the case where frw = 0,1
	int fr = (int) bckgnd->Frw();
	return (fr == 0 || fr == 1) ? 0.0 : (bckgnd->Frw())*(bckgnd->Frw()-1.0)/(*RV()*(*RV())) ;
}

double  Scalar::HubbleConformal  ()
{
	// R'/R = frw/z
	// since we have R=z^frw
	//except in the case where frw = 0,1
	int fr = (int) bckgnd->Frw();
	return (fr == 0 || fr == 1) ? 0.0 : (bckgnd->Frw())/(*RV()) ;
}

double	Scalar::AxionMass  () {

	double aMass;
	double RNow      = *RV();
	// ZThRes is applied to R, not z
	// change the names?
	double &zThRes   = bckgnd->ZThRes();
	double &zRestore = bckgnd->ZRestore();
	double &indi3    = bckgnd->Indi3();
	double &nQcd     = bckgnd->QcdExp();

        if ((RNow > zThRes) &&  (zThRes < zRestore))
        {
                aMass = indi3*pow(zThRes, nQcd*0.5);
                if (RNow > zRestore)
                        aMass *= pow(RNow/zRestore, nQcd*0.5);
        }
        else
                aMass = indi3*pow(RNow, nQcd*0.5);

        return aMass;
}

double	Scalar::AxionMassSq() {

	double aMass;
	double RNow      = *RV();
	double &zThRes   = bckgnd->ZThRes();
	double &zRestore = bckgnd->ZRestore();
	double &indi3    = bckgnd->Indi3();
	double &nQcd     = bckgnd->QcdExp();

        if ((RNow > zThRes) &&  (zThRes < zRestore))
        {
                aMass = indi3*indi3*pow(zThRes, nQcd);
                if (RNow > zRestore)
                        aMass *= pow(RNow/zRestore, nQcd);
        }
        else
                aMass = indi3*indi3*pow(RNow, nQcd);

        return aMass;
}

// time integral of the axion mass^2 R^n assuming it is a truncated power law
// and R = z^frw
double	Scalar::IAxionMassSqn(double z0, double z, int nn) {

	double aMass;
	double RNow      = *RV();
	double &zThRes   = bckgnd->ZThRes();
	double &zRestore = bckgnd->ZRestore();
	double &indi3    = bckgnd->Indi3();
	double &nQcd     = bckgnd->QcdExp();

	// assume zrestore is infty
	// exponent of the time integral
	if (z > z0){
		if (z <= zThRes){
			double exponent = (bckgnd->Frw())*(nQcd+nn) +1;
			double inte = (pow(z, exponent)-pow(z0, exponent))*indi3*indi3/(exponent);
			return inte;
		}
		if ( (z > zThRes) && (z0 < zThRes)){
			double exponent = (bckgnd->Frw())*(nQcd+nn) +1;
			double inte = (pow(zThRes, exponent)-pow(z0, exponent))*indi3*indi3/(exponent);
			exponent = (bckgnd->Frw())*(nn) +1;
			inte += (pow(z, exponent)-pow(zThRes, exponent))*indi3*indi3/(exponent);
			return inte;
		}
		if ( z > zThRes && (z0 >= zThRes)){
			double exponent = (bckgnd->Frw())*(nn) +1;
			double inte = (pow(z, exponent)-pow(z0, exponent))*indi3*indi3/(exponent);
			return inte;
		}
	} else {
		if (z0 <= zThRes){
			double exponent = (bckgnd->Frw())*(nQcd+nn) +1;
			double inte = (pow(z, exponent)-pow(z0, exponent))*indi3*indi3/(exponent);
			return inte;
		}
		if ( (z0 > zThRes) && (z < zThRes)){
			double exponent = (bckgnd->Frw())*(nn) +1;
			double inte = (pow(zThRes, exponent)-pow(z0, exponent))*indi3*indi3/(exponent);
			exponent = (bckgnd->Frw())*(nQcd+nn) +1;
			inte += (pow(z, exponent)-pow(zThRes, exponent))*indi3*indi3/(exponent);
			return inte;
		}
		if ( z0 > zThRes && (z >= zThRes)){
			double exponent = (bckgnd->Frw())*(nn) +1;
			double inte = (pow(z, exponent)-pow(z0, exponent))*indi3*indi3/(exponent);
			return inte;
		}
	}
}

// 2nd time integral of the axion mass^2 R^n assuming it is a truncated power law
// and R = z^frw
double	Scalar::IIAxionMassSqn(double z0, double z, int nn) {

	double aMass;
	double RNow      = *RV();
	double &zThRes   = bckgnd->ZThRes();
	double &zRestore = bckgnd->ZRestore();
	double &indi3    = bckgnd->Indi3();
	double &nQcd     = bckgnd->QcdExp();

	// assume zrestore is infty
	// exponent of the time integral
	if (z>z0){
		if (z <= zThRes){
			double exponent = (bckgnd->Frw())*(nQcd+nn) +1;
			double inte = (pow(z, exponent+1)-pow(z0, exponent+1))*indi3*indi3/(exponent*(exponent+1))
										- pow(z0, exponent)*(z-z0)*indi3*indi3/(exponent);
			return inte;
		}
		if ( (z > zThRes) && (z0 < zThRes)){
			double exponent = (bckgnd->Frw())*(nQcd+nn) +1;
			double inte = (pow(zThRes, exponent+1)-pow(z0, exponent+1))*indi3*indi3/(exponent*(exponent+1))
										- pow(z0, exponent)*(zThRes-z0)*indi3*indi3/(exponent);
			exponent = (bckgnd->Frw())*(nn) +1;
			inte += (pow(z, exponent+1)-pow(zThRes, exponent+1))*indi3*indi3/(exponent*(exponent+1))
										- pow(zThRes, exponent)*(z-zThRes)*indi3*indi3/(exponent);
			return inte;
		}
		if ( z > zThRes && (z0 >= zThRes)){
			double exponent = (bckgnd->Frw())*(nn) +1;
			double inte = (pow(z, exponent+1)-pow(z0, exponent+1))*indi3*indi3/(exponent*(exponent+1))
										- pow(z0, exponent)*(z-z0)*indi3*indi3/(exponent);
			return inte;
		}
	} else {
		if (z0 <= zThRes){
			double exponent = (bckgnd->Frw())*(nQcd+nn) +1;
			double inte = (pow(z, exponent+1)-pow(z0, exponent+1))*indi3*indi3/(exponent*(exponent+1))
										- pow(z0, exponent)*(z-z0)*indi3*indi3/(exponent);
			return inte;
		}
		if ( (z0 > zThRes) && (z < zThRes)){
			double exponent = (bckgnd->Frw())*(nn) +1;
			double inte = (pow(zThRes, exponent+1)-pow(z0, exponent+1))*indi3*indi3/(exponent*(exponent+1))
										- pow(z0, exponent)*(zThRes-z0)*indi3*indi3/(exponent);
			exponent = (bckgnd->Frw())*(nQcd+nn) +1;
			inte += (pow(z, exponent+1)-pow(zThRes, exponent+1))*indi3*indi3/(exponent*(exponent+1))
										- pow(zThRes, exponent)*(z-zThRes)*indi3*indi3/(exponent);
			return inte;
		}
		if ( z > zThRes && (z0 >= zThRes)){
			double exponent = (bckgnd->Frw())*(nn) +1;
			double inte = (pow(z, exponent+1)-pow(z0, exponent+1))*indi3*indi3/(exponent*(exponent+1))
										- pow(z0, exponent)*(z-z0)*indi3*indi3/(exponent);
			return inte;
		}
	}

}
// Saxion mass squared, perhaps the following functions could be rewriten to use this one
double  Scalar::SaxionMassSq  ()
{

	double lbd   = LambdaP();

	auto   &pot = bckgnd->QcdPot();

	switch  (pot & VQCD_TYPE) {
		case    VQCD_0:
		case    VQCD_1:
		case    VQCD_2:
		case    VQCD_1N2:
		case    VQCD_PQ_ONLY:
			return 2.*lbd;
			break;

		case    VQCD_1_PQ_2:
		case    VQCD_1_PQ_2_DRHO:
			return  8.*lbd;
			break;

		default :
			return  0;
			break;
	}

	return  0.;
}

double	Scalar::dzSize	   () {
	double zNow = *zV();
	double RNow = *RV();
	double oodl = ((double) n1)/bckgnd->PhysSize();
	double mAx2 = AxionMassSq();
	double &lbd = bckgnd->Lambda();
	double msaa = sqrt(2.*bckgnd->Lambda())*bckgnd->PhysSize()/((double) n1);
	double mAfq = 0.;
	auto   &pot = bckgnd->QcdPot();
	double llee = bckgnd->LamZ2Exp();

        if ((fieldType & FIELD_AXION) || (fieldType == FIELD_WKB))
                return  std::min(wDz/sqrt(mAx2*(RNow*RNow) + 12.*(oodl*oodl)),zNow/10.);
         else
                mAfq = sqrt(mAx2*(RNow*RNow) + 12.*oodl*oodl);

        double mSfq = 0.;

				mAfq = std::max(mAfq,HubbleMassSq());

				double facto = 1.;
        if ((pot & VQCD_TYPE) == VQCD_1_PQ_2)
                facto = 2. ;

				mSfq = sqrt(2.*lbd*pow(RNow,2.0-llee)*facto*facto + 12.*oodl*oodl);

        return  std::min(wDz/std::max(mSfq,mAfq),zNow/10.);
}

// Fix for arbitrary background
double Scalar::SaxionShift()
{
	double lbd   = bckgnd->Lambda();
	double alpha = AxionMassSq()/lbd;

	if (LambdaT() == LAMBDA_Z2)
		alpha *= (*R)*(*R);

	double discr = 4./3.-9.*alpha*alpha;

	return	((discr > 0.) ? ((2./sqrt(3.))*cos(atan2(sqrt(discr),3.0*alpha)/3.0)-1.) : ((2./sqrt(3.))*cosh(atanh(sqrt(-discr)/(3.0*alpha))/3.0)-1.));
}

double  Scalar::Saskia  ()
{
	auto   &pot = bckgnd->QcdPot();

	switch  (pot & VQCD_TYPE) {
		case    VQCD_PQ_ONLY:
			return 0.0;
			break;

		case    VQCD_1:
			return SaxionShift();
			break;

		case    VQCD_1_PQ_2:
		case    VQCD_1_PQ_2_DRHO:
		{
			double  lbd = bckgnd->Lambda();
			if (LambdaT() == LAMBDA_Z2)
				return  rsvPQ2(AxionMassSq()/lbd*(*R)*(*R));
			else
				return  rsvPQ2(AxionMassSq()/lbd);
			break;
		}

		case    VQCD_2:
		case    VQCD_0:
			return  0.;
			break;

		// This is yet to be computed
		case    VQCD_1N2:
			return  0.;
			break;

		default :
			return  0;
			break;
	}

	return  0.;
}

double	Scalar::AxionMass  (const double RNow) {

	double aMass;
	double &zThRes   = bckgnd->ZThRes();
	double &zRestore = bckgnd->ZRestore();
	double &indi3    = bckgnd->Indi3();
	double &nQcd     = bckgnd->QcdExp();

        if ((RNow > zThRes) &&  (zThRes < zRestore))
        {
                aMass = indi3*pow(zThRes, nQcd*0.5);
                if (RNow > zRestore)
                        aMass *= pow(RNow/zRestore, nQcd*0.5);
        }
        else
                aMass = indi3*pow(RNow, nQcd*0.5);

        return aMass;
}

double	Scalar::AxionMassSq(const double RNow) {

	double aMass;
	double &zThRes   = bckgnd->ZThRes();
	double &zRestore = bckgnd->ZRestore();
	double &indi3    = bckgnd->Indi3();
	double &nQcd     = bckgnd->QcdExp();

        if ((RNow > zThRes) &&  (zThRes < zRestore))
        {
                aMass = indi3*indi3*pow(zThRes, nQcd);
                if (RNow > zRestore)
                        aMass *= pow(RNow/zRestore, nQcd);
        }
        else
                aMass = indi3*indi3*pow(RNow, nQcd);

        return aMass;
}

// Saxion mass squared, perhaps the following functions could be rewriten to use this one
double  Scalar::SaxionMassSq  (const double RNow)
{

	double lbd   = bckgnd->Lambda();
	double llee  = bckgnd->LamZ2Exp();
	if (LambdaT() == LAMBDA_Z2)
		lbd /= pow(RNow,llee);

	auto   &pot = bckgnd->QcdPot();

	switch  (pot & VQCD_TYPE) {
		case    VQCD_PQ_ONLY:
		case    VQCD_0:
		case    VQCD_1:
		case    VQCD_2:
		case    VQCD_1N2:
			return 2.*lbd;
			break;

		case    VQCD_1_PQ_2:
		case    VQCD_1_PQ_2_DRHO:
			return  8.*lbd;
			break;

		default :
			return  0;
			break;
	}

	return  0.;
}

double	Scalar::dzSize	   (const double RNow) {
	double oodl = ((double) n1)/bckgnd->PhysSize();
	double mAx2 = AxionMassSq();
	double &lbd = bckgnd->Lambda();
	double msaa = sqrt(2.*bckgnd->Lambda())*bckgnd->PhysSize()/((double) n1);
	double mAfq = 0.;
	auto   &pot = bckgnd->QcdPot();

        if ((fieldType & FIELD_AXION) || (fieldType == FIELD_WKB))
                return  wDz/sqrt(mAx2*(RNow*RNow) + 12.*(oodl*oodl));
         else
                mAfq = sqrt(mAx2*(RNow*RNow) + 12.*oodl*oodl);

        double mSfq = 0.;

				mAfq = std::max(mAfq,HubbleMassSq());

        double facto = 1.;
        if ((pot & VQCD_TYPE) == VQCD_1_PQ_2)
                facto = 2. ;

        switch (lambdaType) {
                case    LAMBDA_Z2:
                        mSfq = sqrt(facto*facto*msaa*msaa + 12.)*oodl;
                        break;

                case    LAMBDA_FIXED:
                        mSfq = sqrt(2.*lbd*(RNow*RNow)*facto*facto + 12.*oodl*oodl);
                        break;
        }

        return  wDz/std::max(mSfq,mAfq);
}

double Scalar::SaxionShift(const double RNow)
{
	double &lbd = bckgnd->Lambda();
	double alpha = AxionMassSq()/(lbd*RNow*RNow);
	double discr = 4./3.-9.*alpha*alpha;

	return	((discr > 0.) ? ((2./sqrt(3.))*cos(atan2(sqrt(discr),3.0*alpha)/3.0)-1.) : ((2./sqrt(3.))*cosh(atanh(sqrt(-discr)/(3.0*alpha))/3.0)-1.));
}

double  Scalar::Saskia  (const double RNow)
{
	double &lbd = bckgnd->Lambda();
	auto   &pot = bckgnd->QcdPot();

	switch  (pot & VQCD_TYPE) {
		case    VQCD_PQ_ONLY:
			return 0.0;
			break;

		case    VQCD_1:
			return SaxionShift();
			break;

		case    VQCD_1_PQ_2:
		case    VQCD_1_PQ_2_DRHO:
			return  rsvPQ2(AxionMassSq()/(lbd*RNow*RNow));
			break;

		case    VQCD_2:
		case    VQCD_0:
			return  0.;
			break;

		case    VQCD_1N2:
			return  0.;
			break;


		default :
			return  0;
			break;
	}

	return  0.;
}

void	Scalar::setCO(size_t newN)
{
	co.resize(newN); co.assign(newN, 0.);

	switch(newN)
	{
		case 0:
		break;

		case 1:
		default:
			co = {1.}  ;
			break;
		case 2:
			co = {4./3., -1./12.};
			break;
		case 3:
			co = {1.5, -3./20.0,1./90.};
			break;
		case 4:
			co = {1.6, -0.2, 8./315., -1./560.};
			break;
		case 5:
			co = {5./3., -5./21., 5./126., -5./1008., 1./3150.};
			break;
	}
}

/*	Follow all the functions written by Javier	*/
/*	These should be rewritten following the
	standards of the library, including logger,
	profiler, vector code, gpu support and
	outside the Scalar class, so it doesn't
	become a massively cluttered object		*/

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

	if(fieldType & FIELD_AXION)
	{
		//mCONT assumed normalised // i.e. contrastbin was called before
		Float *mCONT = static_cast<Float*> (m2);
		Float *mTheta = static_cast<Float*> (m);
		//float *mVeloc = static_cast<float*> (v);

		int size = 0 ;
//		size_t iyP, iyM, ixP, ixM;
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
		printMpi("%f(%zu)", ct_local[i], ar_local[i]);
	}
	printMpi("\n");


	for(int i = 0; i < numaxitons ; i++)
	{
		(static_cast<size_t *> (idxbin))[i] = ar_local[i];
	}
	return ;
}
