#include <cmath>
#include <algorithm>
#include <complex>
#include <cstring>

#include <omp.h>
#include <mpi.h>

#include "spectrum/spectrum.h"
#include "scalar/folder.h"
#include "comms/comms.h"
#include "fft/fftCode.h"

// uses sD, m and perhaps v to create a mask in m2
// possibly outputs a 2D map of the mask
// possibly outputs a 3D map of the mask (using strings or stringCo functions)

void	SpecBin::masker	(SpectrumMaskType mask, int neigh){

	switch (mask)
	{
		case SPMASK_FLAT :
		case SPMASK_VIL :
		case SPMASK_VIL2 :
		case SPMASK_SAXI :
			LogError("[Spectrum nRun] These masks are not yet implemented");
		break;

		case SPMASK_TEST :
		default:
			switch (fPrec)
			{
				case FIELD_SINGLE :
				SpecBin::masker<float,SPMASK_TEST> (neigh);
				break;

				case FIELD_DOUBLE :
				SpecBin::masker<double,SPMASK_TEST> (neigh);
				break;

				default :
				LogError("[Spectrum nRun] precision not reconised.");
				break;
			}
		break;
	}
}

template<typename Float, SpectrumMaskType mask>
void	SpecBin::masker	(int neigh) {

	if	(field->Folded())
	{
		Folder	munge(field);
		munge(UNFOLD_ALL);
	}

	field->sendGhosts(FIELD_M,COMM_SDRV);
	field->sendGhosts(FIELD_M, COMM_WAIT);

	switch (fType) {

		case	FIELD_SAXION:
		{
			char *strdaa = static_cast<char *>(static_cast<void *>(field->sData()));
			// JAVI PROPOSAL I think would be easy to modify propkernel Xeon to do the loops vectorised
			std::complex<Float> *ma     = static_cast<std::complex<Float>*>(field->mStart());
			std::complex<Float> *va     = static_cast<std::complex<Float>*>(field->vCpu());
			Float *m2sa                 = static_cast<Float *>(field->m2Cpu());

			/* set to 0 two new ghost regions in m2half*/
			Float *m2sax                = static_cast<Float *>(field->m2half());
			size_t surfi = Ly*(Ly+2) ;
			#pragma omp parallel for schedule(static)
			for (size_t odx=0; odx < surfi; odx++) {
				m2sax[odx] = 0 ;
			}

			/* MPI rank and position of the last slice that we will send to the next rank */
			int myRank = commRank();
			int nsplit = (int) (field->TotalDepth()/field->Depth()) ;
			static const int fwdNeig = (myRank + 1) % nSplit;
			size_t voli = Ly*(Ly+2)*(Lz-1) ;
			const int ghostBytes = (Ly*(Ly+2))*(field->DataSize());
			static MPI_Request 	rSendFwd, rRecvBck;
			void *sGhostFwd = static_cast<void *>(m2sa + voli);
			void *rGhostBck = static_cast<void *>(m2sax);

			// optimizar!
			#pragma omp parallel for schedule(static)
			for (size_t iz=Lz-1; iz >= 0; iz--) {
				size_t zo = Ly*(Ly+2)*iz ;
				size_t zoM = Ly*(Ly+2)*(iz+1) ;
				size_t zi = Ly*Ly*iz ;

				for (size_t iy=0; iy < Ly; iy++) {
					size_t yo = (Ly+2)*iy ;
					size_t yoM = (Ly+2)*((iy+1)%Ly) ;
					size_t yi = Ly*iy ;

					for (size_t ix=0; ix < Ly; ix++) {

						/* position in the mask (with padded zeros for the FFT) and in the stringData */
						size_t odx = ix + yo + zo;
						size_t idx = ix + yi + zi;

						/* initialise to zero the mask */
						m2sa[odx] = 0;

						switch(mask){
							case SPMASK_FLAT:
							case SPMASK_VIL:
							case SPMASK_VIL2:
							case SPMASK_SAXI:
								LogMsg(VERB_NORMAL,"These masks are automatic! why did you run this function??");
							break;
							case SPMASK_TEST:
									if ( (strdaa[idx] & STRING_ONLY) != 0)
									{
										m2sa[odx] = 1;
										if (strdaa[idx] & (STRING_XY))
										{
											m2sa[((ix + 1) % Ly) + yo + zo] = 1;
											m2sa[ix + yoM + zo] = 1;
											m2sa[((ix + 1) % Ly) + yoM + zo] = 1;
										}

										if (strdaa[idx] & (STRING_YZ))
										{
											m2sa[ix + yoM + zo] = 1;
											m2sa[ix + yo + zoM] = 1;
											m2sa[ix + yoM + zoM] = 1;
										}

										if (strdaa[idx] & (STRING_ZX))
										{
											m2sa[ix + yo + zoM] = 1;
											m2sa[((ix + 1) % Ly) + yo + zo] = 1;
											m2sa[((ix + 1) % Ly) + yo + zoM] = 1;
										}
									}
							break;
						}  //end mask
					}    // end loop x
				}      // end loop y

				if (Lz = Ly-1) //given to one thread only I hope
				{
					/* Send ghosts from lastslicem2 -> mhalf */
					MPI_Send_init(sGhostFwd, ghostBytes, MPI_BYTE, fwdNeig, rank,   MPI_COMM_WORLD, &rSendFwd);
					MPI_Recv_init(rGhostBck, ghostBytes, MPI_BYTE, bckNeig, bckNeig,   MPI_COMM_WORLD, &rRecvBck);
					MPI_Start(&rSendFwd);
					MPI_Start(&rRecvBck);
				}

			}        // end loop y

			/* makes sure the ghosts have arrived */
			MPI_Wait(&rSendFwd, MPI_STATUS_IGNORE);
			MPI_Wait(&rRecvBck, MPI_STATUS_IGNORE);

			/* frees */
			MPI_Request_free(&rSendFwd);
			MPI_Request_free(&rRecvBck);

			/* Fuse ghost and local info 1st surfi */
			#pragma omp parallel for schedule(static)
			for (size_t odx=0; odx < surfi; odx++) {
				if (m2sax[odx] == 1)
					m2sa[odx] = m2sax[odx] ; // if it was 1 still 1, otherwise 1
			}

			/* Fourier transform */
			// r2c FFT in m2
			auto &myPlan = AxionFFT::fetchPlan("pSpecSx");
			myPlan.run(FFT_FWD);

			/* Filter */
			switch (fPrec) {
				case	FIELD_SINGLE:
						filterFFT<float> (neigh);
					break;

				case	FIELD_DOUBLE:
						filterFFT<double> (neigh);
					break;

				default:
					LogError ("Wrong precision");
					break;
			}

			/* iFFT */
			myPlan.run(FFT_BCK);

			/* play */

			/* mask in m2 */
			// one can use an energy plot to display it!
}
