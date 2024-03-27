#include<cstdlib>
#include<cstring>
#include<complex>
#include<chrono>

#include"enum-field.h"
#include"fft/fftCuda.h"
#include"fft/fftCode.h"
#include "scalar/folder.h"



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

#define NMPICHUNK 10

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



	LogMsg(VERB_NORMAL,"[sca] ZThRes  () %e",cm->ZThRes  ());
	LogMsg(VERB_NORMAL,"[sca] ZRestore() %e",cm->ZRestore());
	LogMsg(VERB_NORMAL,"[sca] PhysSize() %e",cm->PhysSize());
	LogMsg(VERB_NORMAL,"[sca] Lambda  () %e",cm->Lambda  ());
	LogMsg(VERB_NORMAL,"[sca] LamZ2Exp() %e",cm->LamZ2Exp());
	LogMsg(VERB_NORMAL,"[sca] Indi3   () %e",cm->Indi3   ());
	if (cm->UeC()){
		LogMsg(VERB_NORMAL,"[sca] using QCD cosmology!");
		cm->Setup();
	}
	else {
		LogMsg(VERB_NORMAL,"[sca] Gamma   () %e",cm->Gamma   ());
		LogMsg(VERB_NORMAL,"[sca] QcdExp  () %e",cm->QcdExp  ());
		LogMsg(VERB_NORMAL,"[sca] QcdPot  () %d",cm->QcdPot  ());
		LogMsg(VERB_NORMAL,"[sca] Frw     () %e",cm->Frw     ());
		LogMsg(VERB_NORMAL,"[sca] Mink    () %d",cm->Mink    ());
	}
	LogMsg(VERB_NORMAL,"[sca] ic.Nghost   %d",cm->ICData().Nghost   );
	LogMsg(VERB_NORMAL,"[sca] ic.icdrule  %d",cm->ICData().icdrule  );
	LogMsg(VERB_NORMAL,"[sca] ic.preprop  %d",cm->ICData().preprop  );
	LogMsg(VERB_NORMAL,"[sca] ic.icstudy  %d",cm->ICData().icstudy  );
	LogMsg(VERB_NORMAL,"[sca] ic.prepstL  %e",cm->ICData().prepstL  );
	LogMsg(VERB_NORMAL,"[sca] ic.prepcoe  %e",cm->ICData().prepcoe  );
	LogMsg(VERB_NORMAL,"[sca] ic.pregammo %e",cm->ICData().pregammo );
	LogMsg(VERB_NORMAL,"[sca] ic.prelZ2e  %e",cm->ICData().prelZ2e  );
	LogMsg(VERB_NORMAL,"[sca] ic.prevtype %d",cm->ICData().prevtype );
	LogMsg(VERB_NORMAL,"[sca] ic.dumpicmeas %d",cm->ICData().dumpicmeas );
	LogMsg(VERB_NORMAL,"[sca] ic.normcore %d",cm->ICData().normcore );
	LogMsg(VERB_NORMAL,"[sca] ic.alpha    %e",cm->ICData().alpha    );
	LogMsg(VERB_NORMAL,"[sca] ic.siter    %d",cm->ICData().siter    );
	LogMsg(VERB_NORMAL,"[sca] ic.kcr      %e",cm->ICData().kcr      );
	LogMsg(VERB_NORMAL,"[sca] ic.kMax     %d",cm->ICData().kMax     );
	LogMsg(VERB_NORMAL,"[sca] ic.mode0    %e",cm->ICData().mode0    );
	LogMsg(VERB_NORMAL,"[sca] ic.beta     %e",cm->ICData().beta     );
	LogMsg(VERB_NORMAL,"[sca] ic.zi       %e",cm->ICData().zi       );
	LogMsg(VERB_NORMAL,"[sca] ic.logi     %e",cm->ICData().logi     );
	LogMsg(VERB_NORMAL,"[sca] ic.cType    %d",cm->ICData().cType    );
	LogMsg(VERB_NORMAL,"[sca] ic.smvarTy  %d",cm->ICData().smvarType);
	LogMsg(VERB_NORMAL,"[sca] ic.mocoty   %d",cm->ICData().mocoty   );

	folded 	   = false;
	M2folded 	 = false;
	eReduced   = false;
	mmomspace 	 = false;
	vmomspace 	 = false;

	setCO(Ng);


	switch (fieldType)
	{
		case FIELD_SAXION:
		case FIELD_SX_RD:
		case FIELD_NAXION:
			nData = 2;
			break;

		case FIELD_AXION_MOD:
		case FIELD_AX_MOD_RD:
		case FIELD_AXION:
		case FIELD_AX_RD:
		case FIELD_WKB:
		case FIELD_PAXION:
		case FIELD_FAXION:
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
			trackAlloc ((void**) &str, n3);
			break;

		case FIELD_FAXION:
			LogMsg(VERB_NORMAL, "[sca] allocating theta, vheta, rho, vho, gra");
			alignAlloc ((void**) &m,   mAlign, mBytes); // will not compute grads
			alignAlloc ((void**) &v,   mAlign, mBytes); //
			alignAlloc ((void**) &rho, mAlign, mBytes);
			alignAlloc ((void**) &vho, mAlign, mBytes);
			alignAlloc ((void**) &g,   mAlign, 3*mBytes);
			trackAlloc ((void**) &str, n3);
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
		case FIELD_FAXION:
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

	LogMsg(VERB_NORMAL, "[sca] Setting m,v to 0");
	memset (m, 0, fSize*v3);
	memset (v, 0, fSize*(n2*(nLz + 2)));
	if (fieldType == FIELD_FAXION){
		memset (rho, 0, fSize*v3);
		memset (vho, 0, fSize*v3);
		memset (g, 0, 3*fSize*v3);
	}



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

	/* Note the big difference zI is an obsolete parameter FIX ME */
	*z = cm->ICData().zi;
	*R = 1.0;
	updateR();

	LogFlush();

	/* CPU allocation */

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

	prof.stop();
	prof.add(std::string("Init Allocation"), 0.0, 0.0);


	LogMsg(VERB_NORMAL, "[sca] Initialise FFT plans");LogFlush();

	/*	WKB fields won't trigger configuration read or FFT initialization	*/

	if (fieldType != FIELD_WKB && !(fieldType & FIELD_REDUCED)) {
		prof.start();
		AxionFFT::initFFT(prec);

		/* For spectra, reducer, genConf */
		AxionFFT::initPlan (this, FFT_PSPEC_AX,  FFT_FWDBCK, "pSpecAx");

		if (fieldType == FIELD_SAXION) {
			if (!lowmem) {
				AxionFFT::initPlan (this, FFT_SPSX,       FFT_FWDBCK,     "SpSx");
				AxionFFT::initPlan (this, FFT_RDSX_V,     FFT_FWDBCK,    "RdSxV");
			}
		}

		/* If spectral initSpectral plans
		at the moment this is always done which avoids some issues
		when reading configurations without the explicit flag */
		// AxionFFT::initPlan (this, FFT_SPSX,       FFT_FWDBCK,     "SpSx");

		/* If fspectral initSpectral plans*/
		if (fpectral) {
			LogMsg(VERB_NORMAL,"Initialising fspectral plans");
			// Saxion m inplace
			AxionFFT::initPlan (this, FFT_CtoC_MtoM,   FFT_FWDBCK, "C2CM2M");
			// Saxion v inplace
			AxionFFT::initPlan (this, FFT_CtoC_VtoV,   FFT_FWDBCK, "C2CV2V");
			AxionFFT::initPlan (this, FFT_CtoC_M2toM2, FFT_FWDBCK, "C2CM22M2");
			AxionFFT::initPlan (this, FFT_CtoC_M2toM,  FFT_FWDBCK, "C2CM22M");
			// Axion m/v inplace and m2/m
			// for WKB? for fspectral axion
			AxionFFT::initPlan (this, FFT_RtoC_MtoM_WKB,  FFT_FWDBCK, "R2CM2M");
			AxionFFT::initPlan (this, FFT_RtoC_VtoV_WKB,  FFT_FWDBCK, "R2CV2V");
			AxionFFT::initPlan (this, FFT_RtoC_M2toM,     FFT_FWDBCK, "R2CM22M");
			AxionFFT::initPlan (this, FFT_RtoC_M2toV,     FFT_FWDBCK, "R2CM22V");
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
					LogMsg(VERB_NORMAL,"Skipping initialisation of FFT, do it in genconf!");
				}
				prof.stop();

				prof.add(std::string("Init FFT"), 0.0, 0.0);
				genConf	(cm, this);
			}
		}
		LogFlush();
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
	size_t Gc = Ng*n2*fSize;  // Number of chars of the ghost region
	size_t Tc = n3*fSize;			// Number of chars to transfer
	if (device == DEV_GPU)
	{
		#ifndef	USE_GPU
			LogError ("Error: gpu support not built");
			exit   (1);
		#else
			LogMsg(VERB_HIGH,"[sca] Transfering to device %d (M/V/MV=%d,%d,%d) cMDTH %lu ",fIdx,FIELD_M,FIELD_V,FIELD_MV,cudaMemcpyHostToDevice);
			if (Folded())
			{
				Folder munge(this);
				munge(UNFOLD_ALL);
			}
			if (fIdx & FIELD_M)
				cudaMemcpy((((char *) m_d) + Gc), (((char *) m) + Gc),  Tc, cudaMemcpyHostToDevice);

			if (fIdx & FIELD_V)
				cudaMemcpy(v_d,  v,  Tc, cudaMemcpyHostToDevice);

			if ((fIdx & FIELD_M2) && (!lowmem))
				cudaMemcpy((((char *) m2_d) + Gc), (((char *) m2) + Gc),  Tc, cudaMemcpyHostToDevice);
		#endif
	}
}

void	Scalar::transferCpu(FieldIndex fIdx)	// Copies all the array to the CPU
{
	size_t Gc  = Ng*n2*fSize;           // Number of chars of the ghost region
	size_t Tc  = v3*fSize;              // Number of chars to transfer
	size_t Tvc = (n2*(Lz+2))*fSize;     // Number of chars to transfer

	if (device == DEV_GPU)
	{
		#ifndef	USE_GPU
			LogError ("Error: gpu support not built");
			exit   (1);
		#else
			LogMsg(VERB_HIGH,"[sca] Transferring to CPU %d (M/V/MV=%d,%d,%d) cMDTH %lu ",fIdx,FIELD_M,FIELD_V,FIELD_MV,cudaMemcpyDeviceToHost);
			if (fIdx & FIELD_M)
				cudaMemcpy(m,  m_d,  Tc, cudaMemcpyDeviceToHost);

			if (fIdx & FIELD_V){
				cudaMemcpy(v,  v_d,  Tvc, cudaMemcpyDeviceToHost);
				LogMsg(VERB_HIGH,"[sca] v transferred %p to %p (%zu bytes)",v_d,v,Tvc);
			}
			if ((fIdx & FIELD_M2) && (!lowmem))
				cudaMemcpy(m2, m2_d, Tc, cudaMemcpyDeviceToHost);
			
			CudaCheckError();
		#endif
	}
}

void	Scalar::recallGhosts(FieldIndex fIdx)		// Copy to the Cpu the slices of the Gpu that are to be exchanged
{
	size_t Gc  = Ng*n2*fSize;           // Number of chars of the ghost region
	                                    // The same than the chars we have to transfer
	size_t Tc  = n3*fSize;              // Number of chars of the total physical unghosted array

	if (device == DEV_GPU)
	{
		#ifndef	USE_GPU
			LogError ("Error: gpu support not built");
			exit   (1);
		#else
			if (fIdx & FIELD_M) {
				cudaMemcpyAsync(static_cast<char *> (m) + Gc, static_cast<char *> (m_d) + Gc, Gc, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[0]);
				cudaMemcpyAsync(static_cast<char *> (m) + Tc, static_cast<char *> (m_d) + Tc, Gc, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[1]);
			} else {
				cudaMemcpyAsync(static_cast<char *> (m2) + Gc, static_cast<char *> (m2_d) + Gc, Gc, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[0]);
				cudaMemcpyAsync(static_cast<char *> (m2) + Tc, static_cast<char *> (m2_d) + Tc, Gc, cudaMemcpyDeviceToHost, ((cudaStream_t *)sStreams)[1]);
			}

			cudaStreamSynchronize(((cudaStream_t *)sStreams)[0]);
			cudaStreamSynchronize(((cudaStream_t *)sStreams)[1]);
		#endif
	}
}

void	Scalar::transferGhosts(FieldIndex fIdx)	// Copy to the GPU the slices of the CPU that HAVE BEEN UPDATED
{
	size_t Gc  = Ng*n2*fSize;           // Number of chars of the ghost region
																			// The same than the chars we have to transfer
	size_t Tc  = n3*fSize;              // Number of chars of the total physical unghosted array
	size_t Lc  = Gc+Tc;                 // Number of chars before the lastghost region (1 ghost region+physical vol)
	if (device == DEV_GPU)
	{
		#ifndef	USE_GPU
			LogError ("Error: gpu support not built");
			exit   (1);
		#else
			if (fIdx & FIELD_M) {
				cudaMemcpyAsync(static_cast<char *> (m_d),       static_cast<char *> (m),      Gc, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[0]);
				cudaMemcpyAsync(static_cast<char *> (m_d)  + Lc, static_cast<char *> (m) + Lc, Gc, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[1]);
			} else {
				cudaMemcpyAsync(static_cast<char *> (m2_d),      static_cast<char *> (m2),     Gc, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[0]);
				cudaMemcpyAsync(static_cast<char *> (m2_d) + Lc, static_cast<char *> (m2)+ Lc, Gc, cudaMemcpyHostToDevice, ((cudaStream_t *)sStreams)[1]);
			}

			cudaStreamSynchronize(((cudaStream_t *)sStreams)[0]);
			cudaStreamSynchronize(((cudaStream_t *)sStreams)[1]);
		#endif
	}
}

void	Scalar::sendGeneral(CommOperation opComm, size_t count, MPI_Datatype dataType, void* sendBufferB, void* receiveBufferF, void* sendBufferF, void* receiveBufferB)
{
	/* Sends, receives, waits, etc... up to two buffers between ranks
	sendBufferB: pointer to buffer to be sent backwards (0->R-1,1->0,2->1,etc.)
	receBufferF: pointer to buffer to be received FROM forward rank (in 0 from 1, in 1 from 2, etc..)

	sendBufferF: pointer to buffer to be sent forwards  (0->1,1->2,etc.)
	receBufferB: pointer to buffer to be received FROM backwards rank (in 0 from R-1, in 1 from 0, etc..)

	if sendBufferB==receBufferF we deactivate this exchange
	if sendBufferF==receBufferB we deactivate this exchange
	*/
	bool sendB = false;
	bool sendF = false;

	if (sendBufferB != receiveBufferF)
		sendB = true;
	if (sendBufferF != receiveBufferB)
		sendF = true;

	static const int rank = commRank();
	static const int fwdNeig = (rank + 1) % nSplit;
	static const int bckNeig = (rank - 1 + nSplit) % nSplit;

	/* Calculate chunks to comply with INT_MAX in MPI */
	const size_t MAX_CHUNK  = 2147483584;
	int sizeDataType;
	MPI_Type_size(dataType,&sizeDataType);
	const size_t countBytes = count*sizeDataType;
	const size_t nchunks    = (countBytes % MAX_CHUNK == 0) ? countBytes/MAX_CHUNK : 1 + (countBytes/MAX_CHUNK);
	int lastchunk     = (countBytes - (nchunks-1)*MAX_CHUNK); // if lastchunk = 0 it won't be used

	/* Assign receive buffers to the right parts of m, v */
	LogMsg(VERB_HIGH, "[sca] Called send General (COMM %d)",opComm);
	LogMsg(VERB_PARANOID,"[sca] count %lu sizeof(MPIDATA) %d countBytes %lu, MAX_CHUNK %lu, #chunks %lu lastchunk %d", count, sizeDataType, countBytes, MAX_CHUNK, nchunks, lastchunk);
	LogFlush();

	if (nchunks > NMPICHUNK){
		LogError("Number of ghost chunks %d larger than precompiled maximum %d. Change NMPICHUNK in scalarField.cpp and recompile again!", nchunks, NMPICHUNK);
		exit(0);
	}

	int sendBytes[nchunks];
	/* Initialises ALL 4 request-arrays
	I am not sure if different calls will overwrite
	please use only 1 call at a time */
	static MPI_Request reqSendBck[NMPICHUNK], reqRecvFwd[NMPICHUNK];
	static MPI_Request reqSendFwd[NMPICHUNK], reqRecvBck[NMPICHUNK];
	void *sGhostBck[nchunks], *rGhostFwd[nchunks];
	void *sGhostFwd[nchunks], *rGhostBck[nchunks];

	for (int n = 0; n < nchunks; n++)
	{
		sendBytes[n] = (int) MAX_CHUNK;
		if (n == nchunks-1 && lastchunk > 0)
			sendBytes[n] = lastchunk;

		if (sendB) {
		sGhostBck[n] = static_cast<void *> (static_cast<char *> (sendBufferB)    + n*MAX_CHUNK );
		rGhostFwd[n] = static_cast<void *> (static_cast<char *> (receiveBufferF) + n*MAX_CHUNK );
		}

		if (sendF) {
		sGhostFwd[n] = static_cast<void *> (static_cast<char *> (sendBufferF)    + n*MAX_CHUNK );
		rGhostBck[n] = static_cast<void *> (static_cast<char *> (receiveBufferB) + n*MAX_CHUNK );
		}

		LogMsg(VERB_PARANOID,"[sca] n = %d size %d %p %p %p %p", n, sendBytes[n], sGhostBck[n], sGhostFwd[n], rGhostBck[n], rGhostFwd[n]);
	}

	switch	(opComm)
	{
		case	COMM_SEND:
LogMsg(VERB_PARANOID,"[COMM_TESTS] SEND");
			for (int n=0; n<nchunks ;n++) {
				if (sendB)
					MPI_Send_init(sGhostBck[n], sendBytes[n], MPI_BYTE, bckNeig, 2*rank+1, MPI_COMM_WORLD, &(reqSendBck[n]));
				if (sendF)
					MPI_Send_init(sGhostFwd[n], sendBytes[n], MPI_BYTE, fwdNeig, 2*rank,   MPI_COMM_WORLD, &(reqSendFwd[n]));
			}
			for (int n=0; n<nchunks ;n++) {
				if (sendB)
					MPI_Start(&reqSendBck[n]);
				if (sendF)
					MPI_Start(&reqSendFwd[n]);
			}
			break;

		case	COMM_RECV:
LogMsg(VERB_PARANOID,"[COMM_TESTS] RECV");
			for (int n=0; n<nchunks ;n++) {
				if (sendB)
					MPI_Recv_init(rGhostFwd[n], sendBytes[n], MPI_BYTE, fwdNeig, 2*fwdNeig+1, MPI_COMM_WORLD, &(reqRecvFwd[n]));
				if (sendF)
					MPI_Recv_init(rGhostBck[n], sendBytes[n], MPI_BYTE, bckNeig, 2*bckNeig,   MPI_COMM_WORLD, &(reqRecvBck[n]));
			}
			for (int n=0; n<nchunks ;n++) {
				if (sendB)
					MPI_Start(&reqRecvFwd[n]);
				if (sendF)
					MPI_Start(&reqRecvBck[n]);
			}
			break;

		case	COMM_SDRV:
			LogMsg(VERB_PARANOID,"[COMM_TESTS] SDRV");
			for (int n=0; n<nchunks ;n++) {
				if (sendB) {
					MPI_Send_init(sGhostBck[n], sendBytes[n], MPI_BYTE, bckNeig, 2*rank+1,    MPI_COMM_WORLD, &(reqSendBck[n]));
					MPI_Recv_init(rGhostFwd[n], sendBytes[n], MPI_BYTE, fwdNeig, 2*fwdNeig+1, MPI_COMM_WORLD, &(reqRecvFwd[n]));
				}
				if (sendF){
					MPI_Send_init(sGhostFwd[n], sendBytes[n], MPI_BYTE, fwdNeig, 2*rank,      MPI_COMM_WORLD, &(reqSendFwd[n]));
					MPI_Recv_init(rGhostBck[n], sendBytes[n], MPI_BYTE, bckNeig, 2*bckNeig,   MPI_COMM_WORLD, &(reqRecvBck[n]));
				}
			}
			for (int n=0; n<nchunks ;n++) {
				if (sendB){
					MPI_Start(&(reqSendBck[n]));
					MPI_Start(&(reqRecvFwd[n]));
				}
				if (sendF){
					MPI_Start(&(reqSendFwd[n]));
					MPI_Start(&(reqRecvBck[n]));
				}
			}
			LogMsg(VERB_PARANOID,"[COMM_TESTS] SDRV Done");LogFlush();
			break;


	case	COMM_WAIT:
LogMsg(VERB_PARANOID,"[COMM_TESTS] WAIT");
		for (int n=0; n<nchunks ;n++) {
			if (sendB){
				MPI_Wait(&(reqSendBck[n]), MPI_STATUS_IGNORE);
				MPI_Wait(&(reqRecvFwd[n]), MPI_STATUS_IGNORE);
			}
			if (sendF){
				MPI_Wait(&(reqSendFwd[n]), MPI_STATUS_IGNORE);
				MPI_Wait(&(reqRecvBck[n]), MPI_STATUS_IGNORE);
			}
		}
		for (int n=0; n<nchunks ;n++) {
			if (sendB){
				MPI_Request_free(&(reqSendBck[n]));
				MPI_Request_free(&(reqRecvFwd[n]));
			}
			if (sendF){
				MPI_Request_free(&(reqSendFwd[n]));
				MPI_Request_free(&(reqRecvBck[n]));
			}
		}
LogMsg(VERB_PARANOID,"[COMM_TESTS] FREE");
		break;
	}
}

void	Scalar::sendGhosts2(FieldIndex fIdx, CommOperation opComm, int ng)
{
	/* Exchange ghosts
	by default ng = Ng */
	int rNg = (ng == -1) ? Ng : ng;
	if (rNg > Ng){
		LogError("too many ghost slices requested to be exhanged! ng %d > Ng = %d. Only Ng will be exchanged!",ng,Ng);
		rNg = Ng;
	}

	const size_t ghostBytes = rNg*n2*fSize;
	LogMsg(VERB_PARANOID,"[sca] sendGhosts2 ghostBytes %lu GByte %e",ghostBytes,ghostBytes/1.e9);

	void *sB, *rF, *sF, *rB;
	if (fIdx == FIELD_M){
		sB = mStart();
		rF = mBackGhost(); // slice after m
		sF = static_cast<void *> (static_cast<char *> (mStart()) +fSize*n3-ghostBytes);
		rB = mFrontGhost() + (Ng-rNg)*n2*fSize;  // mCpu
	} else {
		if (fIdx & FIELD_V) {
			sB = vStart();
			rF = vBackGhost(); // slice after m
			sF = static_cast<void *> (static_cast<char *> (vStart()) +fSize*n3-ghostBytes);
			rB = vFrontGhost() + (Ng-rNg)*n2*fSize;  // mCpu
		} else {
			sB = m2Start();
			rF = m2BackGhost(); // slice after m
			sF = static_cast<void *> (static_cast<char *> (m2Start()) +fSize*n3-ghostBytes);
			rB = m2FrontGhost() + (Ng-rNg)*n2*fSize;  // mCpu
		}
	}
	Scalar::sendGeneral(opComm, ghostBytes, MPI_BYTE, sB, rF, sF, rB);
}

void	Scalar::sendGhosts(FieldIndex fIdx, CommOperation opComm)
{
	static const int rank = commRank();
	static const int fwdNeig = (rank + 1) % nSplit;
	static const int bckNeig = (rank - 1 + nSplit) % nSplit;

	/* If ghostbytes > INT_MAX split the communication */
	// const size_t MAX_CHUNK  = 2147483648;
	// OJO this is NOT 2^31! is the maximum multiple of 64 below 2^31
	const size_t MAX_CHUNK  = 2147483584;
	const size_t ghostBytes = Ng*n2*fSize;
	size_t nchunks    = 1 + (ghostBytes/MAX_CHUNK);
	int lastchunk     = (ghostBytes - (nchunks-1)*MAX_CHUNK);

	/* Assign receive buffers to the right parts of m, v */
	LogMsg(VERB_HIGH,"[sca] Called send Ghosts (COMM %d)",opComm);
	LogMsg(VERB_PARANOID,"[sca] #Ghost regions Ng = %d, ghostBytes %lu, MAX_CHUNK %lu, #chunks %lu lastchunk %d", Ng, ghostBytes, MAX_CHUNK, nchunks, lastchunk);
	LogFlush();

	if (nchunks > NMPICHUNK){
		LogError("Number of ghost chunks %d larger than precompiled maximum %d. Change NMPICHUNK in scalarField.cpp and recompile again!", nchunks, NMPICHUNK);
		exit(0);
	}


	int sendBytes[nchunks];
	static MPI_Request rSendFwd[NMPICHUNK],  rSendBck[NMPICHUNK],   rRecvFwd[NMPICHUNK],   rRecvBck[NMPICHUNK];	// For non-blocking MPI Comms
	void      *sGhostBck[nchunks], *sGhostFwd[nchunks], *rGhostBck[nchunks], *rGhostFwd[nchunks];

	for (int n = 0; n < nchunks; n++)
	{
		sendBytes[n] = (int) MAX_CHUNK;
		if (n == nchunks-1)
			sendBytes[n] = lastchunk;

		if (fIdx & FIELD_M)
		{
				sGhostBck[n] = static_cast<void *> (static_cast<char *> (mStart())     + n*MAX_CHUNK                         );                       //slice to be send back
				sGhostFwd[n] = static_cast<void *> (static_cast<char *> (mStart())     + n*MAX_CHUNK + fSize*n3-ghostBytes   ); //slice to be send forw
				rGhostBck[n] = static_cast<void *> (static_cast<char *> (mFrontGhost())+ n*MAX_CHUNK + fSize*Ng*n2-ghostBytes);       //reception point
				rGhostFwd[n] = static_cast<void *> (static_cast<char *> (mBackGhost()) + n*MAX_CHUNK + fSize*Ng*n2-ghostBytes);        //reception point
		}
		else
		{
			if (fIdx & FIELD_V)
			{
					sGhostBck[n] = static_cast<void *> (static_cast<char *> (vStart())      + n*MAX_CHUNK                          );//slice to be send back
					sGhostFwd[n] = static_cast<void *> (static_cast<char *> (vStart())      + n*MAX_CHUNK + fSize*n3-ghostBytes    );					//slice to be send forw
					rGhostBck[n] = static_cast<void *> (static_cast<char *> (vFrontGhost()) + n*MAX_CHUNK + fSize*Ng*n2-ghostBytes );	//reception point
					rGhostFwd[n] = static_cast<void *> (static_cast<char *> (vBackGhost())  + n*MAX_CHUNK + fSize*Ng*n2-ghostBytes );	//reception point
			} else {
					sGhostBck[n] = static_cast<void *> (static_cast<char *> (m2Start())      + n*MAX_CHUNK) ;
					sGhostFwd[n] = static_cast<void *> (static_cast<char *> (m2Start())      + n*MAX_CHUNK + fSize*n3-ghostBytes);
					rGhostBck[n] = static_cast<void *> (static_cast<char *> (m2FrontGhost()) + n*MAX_CHUNK + fSize*Ng*n2-ghostBytes);		//reception point
					rGhostFwd[n] = static_cast<void *> (static_cast<char *> (m2BackGhost())  + n*MAX_CHUNK + fSize*Ng*n2-ghostBytes);	//reception point
			}
		}
		LogMsg(VERB_PARANOID,"[sca] n = %d size %d %p %p %p %p", n, sendBytes[n], sGhostBck[n], sGhostFwd[n], rGhostBck[n], rGhostFwd[n]);
	}

	switch	(opComm)
	{
		case	COMM_SEND:
LogMsg(VERB_PARANOID,"[COMM_TESTS] SEND");
			for (int n=0; n<nchunks ;n++) {
				MPI_Send_init(sGhostFwd[n], sendBytes[n], MPI_BYTE, fwdNeig, 2*rank,   MPI_COMM_WORLD, &(rSendFwd[n]));
				MPI_Send_init(sGhostBck[n], sendBytes[n], MPI_BYTE, bckNeig, 2*rank+1, MPI_COMM_WORLD, &(rSendBck[n]));
			}
			for (int n=0; n<nchunks ;n++) {
				MPI_Start(&rSendFwd[n]);
				MPI_Start(&rSendBck[n]);
			}
			break;

		case	COMM_RECV:
LogMsg(VERB_PARANOID,"[COMM_TESTS] RECV");
			for (int n=0; n<nchunks ;n++) {
				MPI_Recv_init(rGhostFwd[n], sendBytes[n], MPI_BYTE, fwdNeig, 2*fwdNeig+1, MPI_COMM_WORLD, &(rRecvFwd[n]));
				MPI_Recv_init(rGhostBck[n], sendBytes[n], MPI_BYTE, bckNeig, 2*bckNeig,   MPI_COMM_WORLD, &(rRecvBck[n]));
			}
			for (int n=0; n<nchunks ;n++) {
				MPI_Start(&rRecvBck[n]);
				MPI_Start(&rRecvFwd[n]);
			}
			break;

		case	COMM_SDRV:
LogMsg(VERB_PARANOID,"[COMM_TESTS] SDRV");
			for (int n=0; n<nchunks ;n++) {
				MPI_Send_init(sGhostFwd[n], sendBytes[n], MPI_BYTE, fwdNeig, 2*rank,      MPI_COMM_WORLD, &(rSendFwd[n]));
				MPI_Send_init(sGhostBck[n], sendBytes[n], MPI_BYTE, bckNeig, 2*rank+1,    MPI_COMM_WORLD, &(rSendBck[n]));
				MPI_Recv_init(rGhostFwd[n], sendBytes[n], MPI_BYTE, fwdNeig, 2*fwdNeig+1, MPI_COMM_WORLD, &(rRecvFwd[n]));
				MPI_Recv_init(rGhostBck[n], sendBytes[n], MPI_BYTE, bckNeig, 2*bckNeig,   MPI_COMM_WORLD, &(rRecvBck[n]));
			}
			for (int n=0; n<nchunks ;n++) {
				MPI_Start(&(rRecvBck[n]));
				MPI_Start(&(rRecvFwd[n]));
				MPI_Start(&(rSendFwd[n]));
				MPI_Start(&(rSendBck[n]));
			}
LogMsg(VERB_PARANOID,"[COMM_TESTS] SDRV Done");LogFlush();
			break;


	case	COMM_WAIT:
LogMsg(VERB_PARANOID,"[COMM_TESTS] WAIT");
		for (int n=0; n<nchunks ;n++) {
			MPI_Wait(&(rSendFwd[n]), MPI_STATUS_IGNORE);
			MPI_Wait(&(rSendBck[n]), MPI_STATUS_IGNORE);
			MPI_Wait(&(rRecvFwd[n]), MPI_STATUS_IGNORE);
			MPI_Wait(&(rRecvBck[n]), MPI_STATUS_IGNORE);
		}
		for (int n=0; n<nchunks ;n++) {
			MPI_Request_free(&(rSendFwd[n]));
			MPI_Request_free(&(rSendBck[n]));
			MPI_Request_free(&(rRecvFwd[n]));
			MPI_Request_free(&(rRecvBck[n]));
		}
LogMsg(VERB_PARANOID,"[COMM_TESTS] FREE");
		break;

	}
}

void	Scalar::exchangeGhosts(FieldIndex fIdx)
{
LogMsg(VERB_PARANOID,"[sca] Exchange Ghosts (fIdx %d)",fIdx);LogFlush();
	recallGhosts(fIdx);
	sendGhosts2(fIdx, COMM_SDRV);
	sendGhosts2(fIdx, COMM_WAIT);
	transferGhosts(fIdx);
LogMsg(VERB_PARANOID,"[sca] Exchange Ghosts Done!");LogFlush();
}













void	Scalar::setField (FieldType newType)
{
	LogMsg(VERB_NORMAL,"\n",newType);
	LogMsg(VERB_NORMAL,"[sca] Called setField to %d !",newType);

	if (fieldType == FIELD_WKB) {
		LogError("Warning: conversion from WKB field not supported");
		return;
	}

	switch (newType)
	{
		case FIELD_AXION_MOD:
		case FIELD_AXION:
		if (fieldType & FIELD_AXION){
			LogError ("Error: transformation from axion to axion irrelevant");
			break;
		}
				fSize /= 2;

				//if (device != DEV_GPU)
				shift *= 2;
		fieldType = newType;

		LogMsg(VERB_NORMAL,"[sca] Field set to AXION !");
		break;

		case	FIELD_SAXION:
			if (fieldType & FIELD_AXION)
				LogError ("Error: transformation from axion to saxion not supported");
			else
				fieldType = FIELD_SAXION;
			break;

		case	FIELD_NAXION:
			if ( !(fieldType & FIELD_AXION) )
				LogError ("Error: transformation to naxion only available from axion mode");
			else
				fSize *= 2;

				if (device != DEV_GPU)
					shift /= 2;

				fieldType = FIELD_NAXION;
				LogMsg(VERB_NORMAL,"[sca] Field set to NAXION !");
			break;

		case	FIELD_PAXION:
			if ( !(fieldType & FIELD_AXION) )
				LogError ("Error: transformation to paxion only available from axion mode");
			else
				fieldType = FIELD_PAXION;
				LogMsg(VERB_NORMAL,"[sca] Field set to PAXION (nothing was done at the Scalar class-level)!");
			break;


		default:
			LogError ("Error: transformation not supported");
			break;

	}
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

// GENERAL BACKGROUND UPDATE REQUIRED!
double	Scalar::Rfromct (const double ct)
{
	// Returns scale factor R = R(ct) conformal time ct
	return bckgnd->R(ct);
}

void	Scalar::updateR ()
{
	// updates scale factor R = z^frw
	*R = Rfromct(*z);
}

double	Scalar::LambdaP ()
{
	// Returns The value of Lambda with PRS trick IF needed
	return bckgnd->LambdaP(*z);
}

double	Scalar::Msa ()
{
	// Returns The value of Msa with PRS trick, or Physical strings
LogMsg(VERB_PARANOID,"[sca:msa] LambdaPhysical %.2f msa %.2f",LambdaP(),sqrt(2.0*LambdaP()) * (*R) * bckgnd->PhysSize()/Length() );
		return  sqrt(2.0*LambdaP()) * (*R) * bckgnd->PhysSize()/Length() ;
}

double  Scalar::HubbleMassSq  ()
{
	return bckgnd->Rpp(*z);
}

double  Scalar::Rpp  ()
{
	// R''/R
	return bckgnd->Rpp(*z);
}

double  Scalar::HubbleConformal  ()
{
	// R'/R = frw/z
	// since we have R=z^frw
	//except in the case where frw = 0,1
	int fr = (int) bckgnd->Frw();
	double Rp = (fr == 0 || fr == 1) ? 0.0 : (bckgnd->Frw())/(*zV());
	LogMsg(VERB_PARANOID,"[sca:HC] %.2e (legacy!)",Rp);
	return Rp ;
}

double	Scalar::AxionMass  () {
	return std::sqrt(bckgnd->AxionMass2(*z));
}

double	Scalar::AxionMassSq() {
	return bckgnd->AxionMass2(*z);
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
	return bckgnd->SaxionMass2(*z);
}

double	Scalar::dzSize	   () {
	return dzSize	(*zV());
}

/* TODO Need to update everything to zNow */
double	Scalar::dzSize	   (double zNow) {
	double RNow = Rfromct(zNow);
	double oodl = ((double) n1)/bckgnd->PhysSize();
	double mAx2 = AxionMassSq(zNow);
	double lamP = bckgnd->LambdaP(zNow);
	double mAfq = 0.;
	auto   &pot = bckgnd->QcdPot();

	double dct   = 0.0;
	double RNext = RNow;

	switch (fieldType) {
		case FIELD_SAXION:
		{
			double mSfq = 0.;
			mAfq = sqrt(mAx2*(RNow*RNow) + 12.*oodl*oodl);
			mAfq = std::max(mAfq,HubbleMassSq());
			double facto = 1.;
			if ((pot & V_PQ) == V_PQ2)
							facto = 2. ;
			mSfq = std::sqrt(2.*lamP*RNow*RNow*facto*facto + 12.*oodl*oodl);
			dct = wDz/std::max(mSfq,mAfq) ;
		}
		break;

		case FIELD_AXION:
		case FIELD_AXION_MOD:
		case FIELD_WKB:
			dct = wDz/std::sqrt(mAx2*(RNow*RNow) + 12.*(oodl*oodl));
		break;

		case FIELD_NAXION:
		case FIELD_PAXION:
			{
				/* w = k^2//(\sqrt(m2+k2)+m2) is the safe formula */
				double w =(12.*oodl*oodl)/(std::sqrt(12*oodl*oodl + mAx2*RNow*RNow) + std::sqrt(mAx2)*RNow) ;
				dct = wDz/w ;
				LogMsg(VERB_PARANOID,"[sca:ct] PNaxion w %e wDz %e mAx2 %e R %e",w,wDz,std::sqrt(mAx2),RNow);
			}
		break;
		default:
			dct = 0.0;
			LogError(" dz set to 0 because FIELD is undefined!");
		break;
	}

	/* Do not allow to jump over z/10 or Rc */
	// if (fieldType != FIELD_PAXION){
	// 	LogMsg(VERB_PARANOID,"[sca:ct] >> ct forced to ct/10 ",dct,zNow);
	// 	dct = std::min(dct,zNow/10.);
	// 	}

	RNext = Rfromct(zNow + dct);
	if ( (RNow < bckgnd->ZThRes()) && (RNext > bckgnd->ZThRes()) && (bckgnd->Frw() != 0.0))
		{
			/*This would set the next point to Rc but it can be stuck*/
			// dct = pow(bckgnd->ZThRes(),1.0/bckgnd->Frw()) - zNow;
			/* Rather, go a ministep ahead */
			double zNext = zNow + dct;
			dct = pow(bckgnd->ZThRes()+ 0.001*(RNext-bckgnd->ZThRes()),1.0/bckgnd->Frw()) - zNow;
			LogMsg(VERB_PARANOID,"[sca:ct] >> dct decreased to %e just jump over Rc [ct,ct_Next,ct+sct]",dct, zNow, zNext,zNow+dct);
		}
	LogMsg(VERB_HIGH,"[sca:ct] dct = %e ct = %e",dct,zNow);
	return dct;
}



double Scalar::SaxionShift()
{
	double alpha = AxionMassSq()/bckgnd->LambdaP (*z);
	double discr = 4./3.-9.*alpha*alpha;
	return	((discr > 0.) ? ((2./sqrt(3.))*cos(atan2(sqrt(discr),3.0*alpha)/3.0)-1.) : ((2./sqrt(3.))*cosh(atanh(sqrt(-discr)/(3.0*alpha))/3.0)-1.));
}

double  Scalar::Saskia  ()
{
	auto   &pot = bckgnd->QcdPot();

	switch  (pot & V_QCD) {
		case    V_QCD0:
		case    V_QCDV:
			return 0.0;
		break;

		case    V_QCD1:
			if 			((pot & V_PQ) == V_PQ1){
				double sh = SaxionShift();
				LogMsg(VERB_PARANOID,"[sca:Saskia] Shift PQ1 %e",sh);
				return sh;
			}
			else if ((pot & V_PQ) == V_PQ2)
			{
				double sh = rsvPQ2(AxionMassSq()/bckgnd->LambdaP (*z));
				LogMsg(VERB_PARANOID,"[sca:Saskia] Shift PQ2 %e",sh);
				return sh ;
			}

		break;

		// This is yet to be computed
		case    V_QCD2:
			return  0.;
		break;

		default :
			return  0;
			break;
	}

	return  0.;
}

double	Scalar::AxionMass  (const double RNow) {

	return std::sqrt(bckgnd->AxionMass2(RNow));
}

double	Scalar::AxionMassSq(const double ct) {
	return bckgnd->AxionMass2(ct);
}

double  Scalar::SaxionMassSq  (const double RNow)
{
	return bckgnd->SaxionMass2(RNow);
}

double Scalar::SaxionShift(const double ct)
{
	double alpha = AxionMassSq(ct)/bckgnd->LambdaP(ct);
	double discr = 4./3.-9.*alpha*alpha;

	return	((discr > 0.) ? ((2./sqrt(3.))*cos(atan2(sqrt(discr),3.0*alpha)/3.0)-1.) : ((2./sqrt(3.))*cosh(atanh(sqrt(-discr)/(3.0*alpha))/3.0)-1.));
}

double  Scalar::Saskia  (const double ct)
{
	auto   &pot = bckgnd->QcdPot();

	switch  (pot & V_QCD) {
		case    V_QCD0:
		case    V_QCDV:
			return 0.0;
		break;

		case    V_QCD1:
			if 			((pot & V_PQ) == V_PQ1){
				double sh = SaxionShift();
				LogMsg(VERB_PARANOID,"[sca:Saskia] Shift PQ1 %e",sh);
				return sh;
			}
			else if ((pot & V_PQ) == V_PQ2)
			{
				double sh = rsvPQ2(AxionMassSq()/bckgnd->LambdaP (*z));
				LogMsg(VERB_PARANOID,"[sca:Saskia] Shift PQ2 %e",sh);
				return sh ;
			}

		break;

		// This is yet to be computed
		case    V_QCD2:
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
