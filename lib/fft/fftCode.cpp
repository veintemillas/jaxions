//#include <unordered_map>
#include <map>
#include <fftw3-mpi.h>
#include "fft/fftCode.h"
#include "utils/parse.h"

namespace AxionFFT {

	static bool init       = false;
	static bool useThreads = false;
	static std::map <std::string,FFTplan>   fftPlans;

	void	FFTplan::importWisdom() {

		static	bool imported = false;

		auto	myRank = commRank();

		if (imported)
			return;

		LogMsg (VERB_NORMAL, "Importing wisdom");

		switch (prec) {
			case	FIELD_SINGLE:
				if (myRank == 0) {
					char wisName[2048];
					sprintf (wisName, "%s/fftWisdom.single", wisDir);
				        if (fftwf_import_wisdom_from_filename(wisName) == 0) {
				                LogMsg (VERB_NORMAL, "Warning: could not import wisdom from %s/fftWisdom.single", wisDir);
						return;
					}
				}

				fftwf_mpi_broadcast_wisdom(MPI_COMM_WORLD);
				break;

			case	FIELD_DOUBLE:
				if (myRank == 0) {
					char wisName[2048];
					sprintf (wisName, "%s/fftWisdom.double", wisDir);
				        if (fftw_import_wisdom_from_filename(wisName) == 0) {
				                LogMsg (VERB_NORMAL, "Warning: could not import wisdom from %s/fftWisdom.double", wisDir);
						return;
					}
				}

				fftw_mpi_broadcast_wisdom(MPI_COMM_WORLD);
				break;
		}
		LogMsg (VERB_NORMAL, "Wisdom successfully imported");
		imported = true;
	}

	void	FFTplan::exportWisdom() {

		LogMsg (VERB_NORMAL, "Exporting wisdom");

		auto	myRank = commRank();

		char wisName[2048];

		switch (prec) {
			case	FIELD_SINGLE:
				sprintf (wisName, "%s/fftWisdom.single", wisDir);
				fftwf_mpi_gather_wisdom(MPI_COMM_WORLD);
				if (myRank == 0) fftwf_export_wisdom_to_filename(wisName);
				break;

			case	FIELD_DOUBLE:
				sprintf (wisName, "%s/fftWisdom.double", wisDir);
				fftw_mpi_gather_wisdom(MPI_COMM_WORLD);
				if (myRank == 0) fftw_export_wisdom_to_filename(wisName);
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
				fftwf_complex *m     = static_cast<fftwf_complex*>(axion->mCpu())  + axion->Surf();
				fftwf_complex *v     = static_cast<fftwf_complex*>(axion->vCpu());
				fftwf_complex *m2    = static_cast<fftwf_complex*>(axion->m2Cpu());
				fftwf_complex *m2R   = static_cast<fftwf_complex*>(axion->m2Cpu()) + (axion->Size()>>1) + axion->Surf();

				// Power spectrum/spectral propagator in axion when the field was created as saxion
				float	      *mR    = static_cast<float *>       (axion->vCpu());

				// Power spectrum in saxion and the other axion cases
				float	      *mA    = static_cast<float *>       (axion->m2Cpu());
				float	      *mAR   = static_cast<float *>       (axion->m2Cpu()) +  axion->Size()     + axion->Surf()*2;

				// WKB
				float	      *mNoGr = static_cast<float *>       (axion->mCpu());
				fftwf_complex *mNoGc = static_cast<fftwf_complex*>(axion->mCpu());

				switch	(type) {
					case	FFT_CtoC_MtoM:

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m, m, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m, m, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE));
						break;

					case	FFT_CtoC_M2toM2:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create C->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m2, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN));
						break;

					case	FFT_CtoC_MtoM2:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create C->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m,  m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m,  MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE));
						break;

					case	FFT_CtoC_VtoM2:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create C->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, v,  m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m2, v,  MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE));
						break;

					case	FFT_RDSX_V:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create C->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, v,  m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m2, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE));
						break;

					case	FFT_SPAX:
					case	FFT_PSPEC_AX:
LogOut("Muahahaha2\n");
						if (axion->Field() == FIELD_SAXION) {

							if (dFft & FFT_FWD)
								planForward  = static_cast<void *>(fftwf_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mR, v, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
LogOut("Muahahaha3\n");
							if (dFft & FFT_BCK)
								planBackward = static_cast<void *>(fftwf_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, v, mR, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN));
LogOut("Muahahaha4\n");
						} else {

							if (dFft & FFT_FWD)
								planForward  = static_cast<void *>(fftwf_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mA, m2, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
							if (dFft & FFT_BCK)
								planBackward = static_cast<void *>(fftwf_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, m2, mA, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN));
						}
						break;

					case	FFT_SPSX:
					case	FFT_RDSX_M:

						if (axion->LowMem()) {
							LogError ("The spectral propagator doesn't work with lowmem");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m,  m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m2, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN));
						break;

					case	FFT_PSPEC_SX:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create R->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mA, m2, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT));
						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, m2, mA, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN));
						break;

					case	FFT_RHO_SX:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create R->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mAR, m2R, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT));
						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, m2R, mAR, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN));
						break;

					case	FFT_RtoC_MtoM_WKB:

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mNoGr, mNoGc, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT));
						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, mNoGc, mNoGr, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN));
						break;

					case	FFT_RtoC_VtoV_WKB:

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mR, v, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT));
						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, v, mR, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN));
						break;

					case	FFT_RtoC_M2toM2_WKB:

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftwf_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mA, m2, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT));
						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftwf_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, m2, mA, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN));
						break;
				}
			}
			break;

			case	FIELD_DOUBLE:
			{
				fftw_complex *m     = static_cast<fftw_complex*>(axion->mCpu())  + axion->Surf();
				fftw_complex *v     = static_cast<fftw_complex*>(axion->vCpu());
				fftw_complex *m2    = static_cast<fftw_complex*>(axion->m2Cpu());
				fftw_complex *m2R   = static_cast<fftw_complex*>(axion->m2Cpu()) + (axion->Size()>>1) + axion->Surf();

				// Power spectrum/spectral propagator in axion when the field was created as saxion
				double	      *mR   = static_cast<double*>      (axion->vCpu());

				// Power spectrum in saxion and the other axion cases
				double	      *mA   = static_cast<double*>      (axion->m2Cpu());
				double	      *mAR  = static_cast<double*>      (axion->m2Cpu()) +  axion->Size()     + axion->Surf()*2;

				// WKB
				double	     *mNoGr = static_cast<double*>      (axion->mCpu());
				fftw_complex *mNoGc = static_cast<fftw_complex*>(axion->mCpu());

				switch	(type) {
					case	FFT_CtoC_MtoM:

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m, m, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m, m, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE));
						break;

					case	FFT_CtoC_M2toM2:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create C->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m2, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN));
						break;

					case	FFT_CtoC_MtoM2:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create C->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m,  m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m,  MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE));
						break;

					case	FFT_CtoC_VtoM2:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create C->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, v,  m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m2, v,  MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE));
						break;

					case	FFT_RDSX_V:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create C->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, v,  m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_ESTIMATE));

						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m2, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE));
						break;

					case	FFT_SPAX:
					case	FFT_PSPEC_AX:
						if (axion->Field() == FIELD_SAXION) {

							if (dFft & FFT_FWD)
								planForward  = static_cast<void *>(fftw_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mR, v, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
							if (dFft & FFT_BCK)
								planBackward = static_cast<void *>(fftw_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, v, mR, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN));
						} else {

							if (dFft & FFT_FWD)
								planForward  = static_cast<void *>(fftw_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mA, m2, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
							if (dFft & FFT_BCK)
								planBackward = static_cast<void *>(fftw_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, m2, mA, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN));
						}
						break;

					case	FFT_SPSX:
                                        case    FFT_RDSX_M:

						if (axion->LowMem()) {
							LogError ("The spectral propagator doesn't work with lowmem");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m,  m2, MPI_COMM_WORLD, FFTW_FORWARD,  FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT));
						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_3d(Lz, Lx, Lx, m2, m2, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN));
						break;

					case	FFT_PSPEC_SX:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create R->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mA, m2, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT));
						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, m2, mA, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN));
						break;

					case    FFT_RHO_SX:

						if (axion->m2Cpu() == nullptr) {
							LogError ("Can't create R->C plan with m2 in lowmem runs");
							exit(0);
						}

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mAR, m2R, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT));
						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, m2R, mAR, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN));
						break;

					case	FFT_RtoC_MtoM_WKB:

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mNoGr, mNoGc, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT));
						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, mNoGc, mNoGr, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN));
						break;
					case	FFT_RtoC_VtoV_WKB:

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mR, v, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT));
						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, v, mR, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN));
						break;
					case	FFT_RtoC_M2toM2_WKB:

						if (dFft & FFT_FWD)
							planForward  = static_cast<void *>(fftw_mpi_plan_dft_r2c_3d(Lz, Lx, Lx, mA, m2, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT));
						if (dFft & FFT_BCK)
							planBackward = static_cast<void *>(fftw_mpi_plan_dft_c2r_3d(Lz, Lx, Lx, m2, mA, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN));
						break;

				}
			}
			break;
		}

		/*	Export wisdom	*/
		exportWisdom();
	}

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
