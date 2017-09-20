//#include <unordered_map>
#include <map>
#include <fftw3-mpi.h>
#include "fft/fftCode.h"

namespace AxionFFT {

	static bool init       = false;
	static bool useThreads = false;
	static std::map <std::string,FFTplan>   fftPlans;

	void	FFTplan::importWisdom() {

		auto	myRank = commRank();

		LogMsg (VERB_NORMAL, "Importing wisdom");

		switch (prec) {
			case	FIELD_SINGLE:
				if (myRank == 0) {
				        if (fftwf_import_wisdom_from_filename("../fftWisdom.single") == 0) {
				                LogMsg (VERB_NORMAL, "Warning: could not import wisdom from fftWisdom.single");
						return;
					}
				}

				fftwf_mpi_broadcast_wisdom(MPI_COMM_WORLD);
				break;

			case	FIELD_DOUBLE:
				if (myRank == 0) {
				        if (fftw_import_wisdom_from_filename("../fftWisdom.double") == 0) {
				                LogMsg (VERB_NORMAL, "Warning: could not import wisdom from fftWisdom.double");
						return;
					}
				}

				fftw_mpi_broadcast_wisdom(MPI_COMM_WORLD);
				break;
		}
		LogMsg (VERB_NORMAL, "Wisdom successfully imported");
	}

	void	FFTplan::exportWisdom() {

		LogMsg (VERB_NORMAL, "Importing wisdom");

		auto	myRank = commRank();

		switch (prec) {
			case	FIELD_SINGLE:
				fftwf_mpi_gather_wisdom(MPI_COMM_WORLD);
				if (myRank == 0) fftwf_export_wisdom_to_filename("../fftWisdom.single");
				break;

			case	FIELD_DOUBLE:
				fftw_mpi_gather_wisdom(MPI_COMM_WORLD);
				if (myRank == 0) fftw_export_wisdom_to_filename("../fftWisdom.double");
				break;
		}
		LogMsg (VERB_NORMAL, "Wisdom successfully exported");
	}

		FFTplan::FFTplan	(Scalar * axion, FFTtype type, FFTdir dFft) : type(type), dFft(dFft), prec(axion->Precision()), Lx(axion->Length()), Lz(axion->TotalDepth()) {

		/*	Import wisdom	*/
		importWisdom();

		switch (prec) {

			case	FIELD_SINGLE:
			{
				fftwf_complex *m   = static_cast<fftwf_complex*>(axion->mCpu())  + axion->Surf();
				fftwf_complex *v   = static_cast<fftwf_complex*>(axion->vCpu());
				fftwf_complex *m2  = static_cast<fftwf_complex*>(axion->m2Cpu()) + axion->Surf();
				float	      *mR  = static_cast<float *>       (axion->vCpu())  + axion->Surf();
				float	      *mS  = static_cast<float *>       (axion->m2Cpu()) + (axion->Surf()>>1);
				fftwf_complex *oR  = static_cast<fftwf_complex*>(static_cast<void*>(mR));

				// FOR SPECTRUM GOES WITHOUT GHOSTS
				// CASE AXION WILL USE M2 (WHICH IS INITIATED AS v)
				// THIS MUST BE PLANNED AT THE BEGGINING OF SCALAR
				float	      	*mA2  = static_cast<float *>       (axion->vCpu())  ;
				fftwf_complex *oA2  = static_cast<fftwf_complex*>(static_cast<void*>(mA2));

				// CASE SAXION WILL USE M2 ONLY WORKING IN !lowmem
				// THIS MUST BE PLANNED AT THE BEGGINING OF SCALAR and will become useless after because m2 is deleted
				float	      	*mS2  = static_cast<float *>       (axion->m2Cpu())  ;
				fftwf_complex *oS2  = static_cast<fftwf_complex*>(static_cast<void*>(mS2));

				// WKB pointers // ONLY USED IN AXION MODE knowing what you do
				// the whole M+V space of (n3+2n2) + n3 [+2n2 extras that I add] floats is divided into
				// (n3+2n2)+(n3+2n2) to host the padded data
				float	      	*WKBm  = static_cast<float *>       (axion->mCpu())  ;
				float	      	*WKBv  = static_cast<float *>       (axion->vCpu())  ;
				float	      	*WKBm2  = static_cast<float *>       (axion->m2Cpu())  ;
				fftwf_complex *WKBmC  = static_cast<fftwf_complex*>(static_cast<void*>(WKBm));
				fftwf_complex *WKBvC  = static_cast<fftwf_complex*>(static_cast<void*>(WKBv));
				fftwf_complex *WKBm2C  = static_cast<fftwf_complex*>(static_cast<void*>(WKBm2));

				switch	(type) {
					case	FFT_CtoC_MtoM:

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m, m, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m, m, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE));
						break;

					case	FFT_CtoC_M2toM2:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create C->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m2, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN));
						break;

					case	FFT_CtoC_MtoM2:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create C->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m,  m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m,  MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE));
						break;

					case	FFT_CtoC_VtoM2:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create C->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, v,  m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m2, v,  MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE));
						break;

					//NEW for SPECTRUM
					case	FFT_RtoC_M2toM2_AXION:
									/* For test, the backward plan requires ghosts	*/
						//if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mA2, oA2, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mR, oR, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
						//if (dFft & FFT_BCK)
						//	planBackward = static_cast<void *>(fftwf_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, oA2, mA2,  MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN));
						break;

					case	FFT_RtoC_M2toM2_SAXION:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create R->C plan with m2 in lowmem runs");
							exit(0);
						}
									/* For test, the backward plan requires ghosts	*/
						//if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mS2, oS2, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mS, oR, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));

						break;

					//FOR WKB PROGRAM
					case	FFT_RtoC_MtoM_WKB:
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, WKBm, WKBmC, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, WKBmC, WKBm, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN));
						break;
					case	FFT_RtoC_VtoV_WKB:
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, WKBv, WKBvC, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, WKBvC, WKBv, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN));
						break;
					case	FFT_RtoC_M2toM2_WKB:
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, WKBm2, WKBm2C, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, WKBm2C, WKBm2, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN));
						break;

					case	FFT_SPSX:

						if (axion->LowMem()) {
							LogError ("The spectral propagator doesn't work with lowmem");
							exit(0);
						}

						planForward  = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m,  m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
						planBackward = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m2, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN));
						break;

					case	FFT_SPAX:
						planForward  = static_cast<void *>(fftwf_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mR,  oR, MPI_COMM_WORLD,  FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
						planBackward = static_cast<void *>(fftwf_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, oR,  mR, MPI_COMM_WORLD,  FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN));
						break;
				}
			}
			break;

			case	FIELD_DOUBLE:
			{
				fftw_complex *m   = static_cast<fftw_complex*>(axion->mCpu())  +  axion->Surf();
				fftw_complex *v   = static_cast<fftw_complex*>(axion->vCpu());
				fftw_complex *m2  = static_cast<fftw_complex*>(axion->m2Cpu()) +  axion->Surf();
				double	     *mR  = static_cast<double *>     (axion->vCpu())  +  axion->Surf();
				double	     *mS  = static_cast<double *>     (axion->m2Cpu()) + (axion->Surf()>>1);
				fftw_complex *oR  = static_cast<fftw_complex*>(axion->vCpu())  +  axion->Surf();

				switch	(type) {
					case	FFT_CtoC_MtoM:

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m, m, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m, m, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE));
						break;

					case	FFT_CtoC_M2toM2:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create C->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m2, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN));
						break;

					case	FFT_CtoC_MtoM2:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create C->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m,  m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m,  MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE));
						break;

					case	FFT_CtoC_VtoM2:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create C->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, v,  m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m2, v,  MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE));
						break;

					case	FFT_RtoC_M2toM2_AXION:

						// FIXME choose FWD or BCK
						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mR, oR, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mR, oR, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
						break;

					case	FFT_RtoC_M2toM2_SAXION:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create R->C plan with m2 in lowmem runs");
							exit(0);
						}

						// FIXME choose FWD or BCK
						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mS, oR, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mS, oR, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
						break;


					case	FFT_SPSX:

						if (axion->LowMem()) {
							LogError ("The spectral propagator doesn't work with lowmem");
							exit(0);
						}

						planForward  = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m,  m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
						planBackward = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m2, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN));
						break;

					case	FFT_SPAX:
						planForward  = static_cast<void *>(fftw_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mR,  oR, MPI_COMM_WORLD,  FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
						planBackward = static_cast<void *>(fftw_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, oR,  mR, MPI_COMM_WORLD,  FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN));
						break;
				}
			}
			break;
		}

		/*	Export wisdom	*/
		exportWisdom();

	}

//		FFTplan::~FFTplan() {
//			LogError ("Calling destructor");	// Forces flush
//			printf("Destructor!!\n"); fflush(stdout);
/*		switch (axion->Precision()) {
			case	FIELD_SINGLE:
				if (dFft & FFT_FWD) {
//				if (planForward != nullptr) {
					printf ("Pointer %p %p %p\n", planForward, nullptr, NULL); fflush(stdout);
					fftwf_destroy_plan (static_cast<fftwf_plan>(planForward));
				}

//				if (planBackward != nullptr) {
				if (dFft & FFT_BCK) {
					printf ("Pointer %p %p %p\n", planBackward, nullptr, NULL); fflush(stdout);
					fftwf_destroy_plan (static_cast<fftwf_plan>(planBackward));
				}
				break;

			case	FIELD_DOUBLE:
				if (dFft & FFT_FWD)
					fftw_destroy_plan (static_cast<fftw_plan>(planForward));

				if (dFft & FFT_BCK)
					fftw_destroy_plan (static_cast<fftw_plan>(planBackward));
				break;
		}*/
//	}

	void	FFTplan::run	(FFTdir cDir)
	{
		LogMsg (VERB_HIGH, "Executing FFT");
		switch (prec) {
			case	FIELD_SINGLE:
				switch (cDir) {
					case	FFT_FWD:
						fftwf_execute(static_cast<fftwf_plan>(planForward));
						break;

					case	FFT_BCK:
						fftwf_execute(static_cast<fftwf_plan>(planBackward));
						break;
				}
				break;

			case	FIELD_DOUBLE:
				switch (cDir) {
					case	FFT_FWD:
						fftw_execute(static_cast<fftw_plan>(planForward));
						break;

					case	FFT_BCK:
						fftw_execute(static_cast<fftw_plan>(planBackward));
						break;
				}
				break;
		}
	}

	void	initFFT		(FieldPrecision prec) {
		LogMsg (VERB_NORMAL, "Initializing FFT using %d MPI ranks...", commSize());

		if (init == true) {
			LogMsg (VERB_HIGH, "FFT already initialized");
			return;
		}
/*
		int  *fftInitThreads;
		void *fftPlanThreads;
		void *fftInitMpi;

		switch (prec)
		{
			case FIELD_DOUBLE:
				fftInitThreads = &fftw_init_threads;
				fftPlanThreads = &fftw_plan_with_nthreads;
				fftInitMpi     = &fftw_mpi_init;
				break;

			case FIELD_SINGLE:
				fftInitThreads = &fftwf_init_threads;
				fftPlanThreads = &fftwf_plan_with_nthreads;
				fftInitMpi     = &fftwf_mpi_init;
				break;

			default:
				LogError ("Invalid precision");
				return;
				break;
		}

		if (!(*fftInitThreads()))
		{
			LogError ("Error initializing FFT with threads");
			LogError ("FFT will use one thread");
			useThreads = false;
			*fftInitMpi();
		} else {
			int nThreads = omp_get_max_threads();
			LogMsg (VERB_NORMAL, "Using %d threads for the FFT", nThreads);
			*fftInitMpi();
			*fftwPlanThreads(nThreads);
		}

*/
		switch (prec)
		{
			case FIELD_DOUBLE:

				if (!fftw_init_threads())
				{
					LogError ("Error initializing FFT with threads");
					LogError ("FFT will use one thread");
					useThreads = false;
					fftw_mpi_init();
				} else {
					int nThreads = omp_get_max_threads();
					LogMsg (VERB_NORMAL, "Using %d threads for the FFT", nThreads);
					fftw_mpi_init();
					fftw_plan_with_nthreads(nThreads);
				}

				break;

			case FIELD_SINGLE:

				if (!fftwf_init_threads())
				{
					LogError ("Error initializing FFT with threads");
					LogError ("FFT will use one thread");
					fflush (stdout);
					useThreads = false;
					fftwf_mpi_init();
				} else {
					int nThreads = omp_get_max_threads();
					LogMsg (VERB_NORMAL, "Using %d threads for the FFT", nThreads);
					fftwf_mpi_init();
					fftwf_plan_with_nthreads(nThreads);
				}

				break;

			default:
				LogError ("Invalid precision");
				return;
				break;

		}

		init = true;
	}

	void	initPlan	(Scalar * axion, FFTtype type, FFTdir dFft, std::string name) {

		LogMsg (VERB_NORMAL, "Creating FFT plan %s", name.c_str());

		if (fftPlans.find(name) == fftPlans.end()) {
			FFTplan myPlan(axion, type, dFft);
			fftPlans.insert(std::make_pair(name, std::move(myPlan)));

			LogMsg (VERB_NORMAL, "Plan %s successfully inserted", name.c_str());
		} else {
			LogMsg (VERB_NORMAL, "Plan %s already exists, ommitted", name.c_str());
		}
	}

	FFTplan&	fetchPlan	(std::string name) {
		return fftPlans[name];
	}

	void	removePlan		(std::string name) {

		if (fftPlans.find(name) == fftPlans.end()) {
			LogError ("Error removing plan %s: not found", name.c_str());
			return;
		}

		auto &myPlan = fftPlans[name];
		auto dFft    = myPlan.Direction();

		LogMsg (VERB_NORMAL, "Removing plan %s", name.c_str());

//		LogOut ("Plan %s, F%d %p B%d %p\n", name.c_str(), dFft & FFT_FWD, myPlan.PlanFwd(), dFft & FFT_BCK, myPlan.PlanBack());

		switch (myPlan.Precision()) {

			case FIELD_SINGLE:

				if (dFft & FFT_FWD)
					fftwf_destroy_plan(static_cast<fftwf_plan>(myPlan.PlanFwd()));

				if (dFft & FFT_BCK)
					fftwf_destroy_plan(static_cast<fftwf_plan>(myPlan.PlanBack()));
				break;

			case FIELD_DOUBLE:

				if (dFft & FFT_FWD)
					fftw_destroy_plan(static_cast<fftw_plan>(myPlan.PlanFwd()));

				if (dFft & FFT_BCK)
					fftw_destroy_plan(static_cast<fftw_plan>(myPlan.PlanBack()));

				break;
		}

		return;
	}


	void	closeFFT		() {

		LogMsg (VERB_NORMAL, "Closing FFT plans");

		if (init == false) {
			LogMsg (VERB_HIGH, "FFT already closed or not initialized");
			return;
		}

		FieldPrecision	prec;

		for (auto fft = fftPlans.cbegin(); fft != fftPlans.cend(); ) {
			auto name = (*fft).first;
			auto plan = (*fft).second;

			prec = plan.Precision();
			removePlan(name);
			fft = fftPlans.erase(fft);	//name

			LogMsg (VERB_NORMAL, "Plan %s closed", name.c_str());
		}

		switch (prec) {
			case FIELD_SINGLE:
				if (useThreads)
					fftwf_cleanup_threads();
				else
					fftwf_cleanup();
				break;

			case FIELD_DOUBLE:
				if (useThreads)
					fftw_cleanup_threads();
				else
					fftw_cleanup();
				break;

			default:
				LogError ("Invalid precision");
				return;
				break;
		}

		LogMsg (VERB_NORMAL, "FFT successfully closed");
		init = false;
	}
}
