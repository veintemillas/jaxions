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

#include "utils/kgvops.h"

using namespace profiler;

void	SpecBin::fillCosTable () {

	const double	ooLx   = 1./Ly;
	const double	factor = (2.*Ly*Ly)/(field->BckGnd()->PhysSize()*field->BckGnd()->PhysSize());

	cosTable.resize(kMax+1);
	cosTable2.resize(kMax+1);

	cosTable[0] = 0.0;
	cosTable2[0] = 1.0;
	#pragma omp parallel for schedule(static)
	for (size_t k=1; k<kMax+1; k++){
		cosTable[k] = factor*(1.0 - cos(M_PI*(2*k)*ooLx));
		cosTable2[k] = 2*(1.0 - cos(M_PI*(2*k)*ooLx))/pow(M_PI*(2*k)*ooLx,2.0);
	}

}

template<typename Float, const SpectrumType sType, const bool spectral>
void	SpecBin::fillBins	() {

	Profiler &prof = getProfiler(PROF_SPEC);
	prof.start();

	LogMsg(VERB_HIGH,"[FB] Filling beans sType %d ",sType) ;LogFlush();
	using cFloat = std::complex<Float>;

	/* The factor that will multiply the |ft|^2, taken to be L^3/(2 N^6) */
	const double norm = (field->BckGnd()->PhysSize()*field->BckGnd()->PhysSize()*field->BckGnd()->PhysSize()) /
			    (2.*(((double) field->TotalSize())*((double) field->TotalSize())));
	const int mIdx = commThreads();

	size_t	zBase = (Ly/commSize())*commRank();

	std::vector<double>	tBinK;
	std::vector<double>	tBinG;
	std::vector<double>	tBinV;
	std::vector<double>	tBinVnl;
#ifdef USE_NN_BINS
	std::vector<double>	tBinNK;
	std::vector<double>	tBinNG;
	std::vector<double>	tBinNV;
	std::vector<double>	tBinNVnl;
#endif
	std::vector<double>	tBinP;
	std::vector<double>	tBinPS;
	std::vector<double>	tBinNN;
	std::vector<double>	tBinAK;

	switch (sType) {
		case	SPECTRUM_K:
		case	SPECTRUM_KS:
		case	SPECTRUM_KK:
			tBinK.resize(powMax*mIdx);
			tBinK.assign(powMax*mIdx, 0);
#ifdef USE_NN_BINS
			tBinNK.resize(powMax*mIdx);
			tBinNK.assign(powMax*mIdx, 0);
#endif
			break;

		case	SPECTRUM_P:
			tBinP.resize(powMax*mIdx);
			tBinP.assign(powMax*mIdx, 0);
			break;
		case	SPECTRUM_PS:
			tBinPS.resize(powMax*mIdx);
			tBinPS.assign(powMax*mIdx, 0);
			break;

		case 	SPECTRUM_NN:
			binNN.resize(powMax); binNN.assign(powMax, 0.);
			tBinNN.resize(powMax*mIdx);
			tBinNN.assign(powMax*mIdx, 0);
			break;

		case 	SPECTRUM_AK:
			binAK.resize(powMax); binAK.assign(powMax, 0.);
			tBinAK.resize(powMax*mIdx);
			tBinAK.assign(powMax*mIdx, 0);
			break;

		case	SPECTRUM_G:
		case	SPECTRUM_GaSadd:
		case	SPECTRUM_GaS:
		case	SPECTRUM_GG:
			tBinG.resize(powMax*mIdx);
			tBinG.assign(powMax*mIdx, 0);
#ifdef USE_NN_BINS
			tBinNG.resize(powMax*mIdx);
			tBinNG.assign(powMax*mIdx, 0);
#endif
			break;

		case	SPECTRUM_VV:
			tBinV.resize(powMax*mIdx);
			tBinV.assign(powMax*mIdx, 0);
#ifdef USE_NN_BINS
			tBinNV.resize(powMax*mIdx);
			tBinNV.assign(powMax*mIdx, 0);
#endif
			break;

		case	SPECTRUM_VNL:
			tBinVnl.resize(powMax*mIdx);
			tBinVnl.assign(powMax*mIdx, 0);
#ifdef USE_NN_BINS
			tBinNVnl.resize(powMax*mIdx);
			tBinNVnl.assign(powMax*mIdx, 0);
#endif
			break;

		default:
			tBinG.resize(powMax*mIdx);
			tBinV.resize(powMax*mIdx);
			tBinG.assign(powMax*mIdx, 0);
			tBinV.assign(powMax*mIdx, 0);
#ifdef USE_NN_BINS
			tBinNG.resize(powMax*mIdx);
			tBinNV.resize(powMax*mIdx);
			tBinNG.assign(powMax*mIdx, 0);
			tBinNV.assign(powMax*mIdx, 0);
#endif
			break;
	}

	#pragma omp parallel
	{
		int  tIdx = omp_get_thread_num ();

		#pragma omp for schedule(static)
		for (size_t idx=0; idx<nModeshc; idx++) {
			size_t tmp = idx/Lx;
			int    kx  = idx - tmp*Lx;
			int    ky  = tmp/Tz;
			int    kz  = tmp - ((size_t) ky)*Tz;
			ky += zBase;	// For MPI, transposition makes the Y-dimension smaller

			//ASSUMES THAT THE FFTS FOR SPECTRA ARE ALWAYS OF r2c type
			//and thus always in reduced format with half+1 of the elements in x

			// if (kx > static_cast<int>(hLx)) kx -= static_cast<int>(Ly); // half complex, this line is not needed
			if (ky > static_cast<int>(hLy)) ky -= static_cast<int>(Ly);
			if (kz > static_cast<int>(hTz)) kz -= static_cast<int>(Tz);

			double k2    = (double) kx*kx + ky*ky + kz*kz;

			//BINOPTION 1
			size_t myBin = floor(sqrt(k2));
			//BINOPTION 2
			// size_t myBin = floor(sqrt(k2)+0.5);

			// LogOut ("Check %lu (%d %d %d) bin out of range %lu > %lu\n", idx, kx, ky, kz, myBin, powMax);

			if (myBin > powMax) {
				LogError ("Error: point %lu (%d %d %d) bin out of range %lu > %lu\n", idx, kx, ky, kz, myBin, powMax);
				continue;
			}

			// JAVI CHANGED for easiness of interpretation
			// if (spectral)
			// 	k2 *= (4.*M_PI*M_PI)/(field->BckGnd()->PhysSize()*field->BckGnd()->PhysSize());
			// else
			// 	k2  = cosTable[abs(kx)] + cosTable[abs(ky)] + cosTable[abs(kz)];

			double		w = 1.0;
			double 		m, m2;

			switch	(sType) {
				case	SPECTRUM_K:
				case	SPECTRUM_KK:
				case	SPECTRUM_G:
				case	SPECTRUM_GG:
				case 	SPECTRUM_V:
				case 	SPECTRUM_GV:
				case 	SPECTRUM_GaS:
				case 	SPECTRUM_GaSadd:
				case 	SPECTRUM_VV:
				case 	SPECTRUM_VNL:
					k2 *= (4.*M_PI*M_PI)/(field->BckGnd()->PhysSize()*field->BckGnd()->PhysSize());
					w  = sqrt(k2 + mass2);
					m  = std::abs(static_cast<cFloat *>(field->m2Cpu())[idx]);
					m2 = 0.;
					break;

				case	SPECTRUM_KS:
				case	SPECTRUM_GS:
				case 	SPECTRUM_VS:
				case 	SPECTRUM_GVS:
					k2 *= (4.*M_PI*M_PI)/(field->BckGnd()->PhysSize()*field->BckGnd()->PhysSize());
					w  = sqrt(k2 + mass2Sax);
					m  = std::abs(static_cast<cFloat *>(field->m2Cpu())[idx]);
					m2 = 0.;
					break;

				case 	SPECTRUM_P:
				case 	SPECTRUM_PS:
					m  = std::abs(static_cast<cFloat *>(field->m2Cpu())[idx]);
					m2 = 0.;
				break;

				case SPECTRUM_NN:
				case SPECTRUM_AK:
					m = 1;
				break;
			}

			/* FFTS are assumed outcome of FFT r2c
				if c2c this needs some changes */
			// recall hLx - 1 = N/2
			if ((kx == 0) || (kx == static_cast<int>(hLx - 1)))
				m2 = m*m;
			else
				m2 = 2*m*m;

			double		mw;

			switch	(sType) {

				/* Saxion mode, the derivative, gradient and mass(top suscep)
				 are already included and we do not divide by w because it
				 can be zero; it needs to be done outside the program */
				case	SPECTRUM_KK:
					tBinK.at(myBin + powMax*tIdx) += m2;
#ifdef USE_NN_BINS
					tBinNK.at(myBin + powMax*tIdx) += m2/w;
#endif
					break;
				case	SPECTRUM_VV:
					tBinV.at(myBin + powMax*tIdx) += m2;
#ifdef USE_NN_BINS
					tBinNV.at(myBin + powMax*tIdx) += m2/w;
#endif
					break;
				case  SPECTRUM_GaS:
				case  SPECTRUM_GaSadd:
					switch (controlxyz){
						// tmp is not used anymore
						case 1:
							tmp = (size_t) std::abs(ky);
							break;
						case 2:
							tmp = (size_t) std::abs(kz);
							break;
						case 0:
						default:
							tmp = kx;
							break;
					}
					tBinG.at(myBin + powMax*tIdx) += m2/cosTable2[tmp];
#ifdef USE_NN_BINS
					tBinNG.at(myBin + powMax*tIdx) += m2/(cosTable2[tmp]*w);
#endif
					break;
				/* is possible to account for the finite difference formula
				by using the folloging line for the gradients
					// tBinG.at(myBin + powMax*tIdx) += mw/cosTable2[kx];
					*/

				/* Axion mode or only Saxion*/
				case	SPECTRUM_K:
				case	SPECTRUM_KS:
				 	// mw = m2/w;
					tBinK.at(myBin + powMax*tIdx) += m2;
#ifdef USE_NN_BINS
					tBinNK.at(myBin + powMax*tIdx) += m2/w;
#endif
					break;
				case	SPECTRUM_G:
				case	SPECTRUM_GS:
					// mw = m2/w;
					tBinG.at(myBin + powMax*tIdx) += m2*k2;
#ifdef USE_NN_BINS
					tBinNG.at(myBin + powMax*tIdx) += m2*k2/w;
#endif
					break;
				case	SPECTRUM_V:
					// mw = m2/w;
					tBinV.at(myBin + powMax*tIdx) += m2*mass2;
#ifdef USE_NN_BINS
					tBinNV.at(myBin + powMax*tIdx) += m2*mass2/w;
#endif
					break;
				case	SPECTRUM_VNL:
					tBinVnl.at(myBin + powMax*tIdx) += m2*mass2;
#ifdef USE_NN_BINS
					tBinNVnl.at(myBin + powMax*tIdx) += m2*mass2/w;
#endif
					break;
				case	SPECTRUM_VS:
					// mw = m2/w;
					tBinV.at(myBin + powMax*tIdx) += m2*mass2Sax;
#ifdef USE_NN_BINS
					tBinNV.at(myBin + powMax*tIdx) += m2*mass2Sax/w;
#endif
					break;
				case	SPECTRUM_GVS:
					// mw = m2/w;
					tBinG.at(myBin + powMax*tIdx) += m2*k2;
					tBinV.at(myBin + powMax*tIdx) += m2*mass2Sax;
#ifdef USE_NN_BINS
					tBinNG.at(myBin + powMax*tIdx) += mw*k2/w;
					tBinNV.at(myBin + powMax*tIdx) += mw*mass2Sax/w;
#endif
					break;

				case	SPECTRUM_GV:
					// mw = m2/w;
					tBinG.at(myBin + powMax*tIdx) += m2*k2;
					tBinV.at(myBin + powMax*tIdx) += m2*mass2;
#ifdef USE_NN_BINS
					tBinNG.at(myBin + powMax*tIdx) += m2*k2/w;
					tBinNV.at(myBin + powMax*tIdx) += m2*mass2/w;
#endif
					break;

				/* energy spectra */
				case	SPECTRUM_P:
					tBinP.at(myBin + powMax*tIdx) += m2;
					break;
				case	SPECTRUM_PS:
					tBinPS.at(myBin + powMax*tIdx) += m2;
					break;

				/* number of modes */
				case	SPECTRUM_NN:
					tBinNN.at(myBin + powMax*tIdx) += m2;
					break;

				/* averaged k2 in the bin */
				case	SPECTRUM_AK:
					tBinAK.at(myBin + powMax*tIdx) += m2*k2;
					break;

			}
		}

		#pragma omp for schedule(static)
		for (uint j=0; j<powMax; j++) {
			for (int i=0; i<mIdx; i++) {

				switch	(sType) {
					case	SPECTRUM_K:
					case	SPECTRUM_KK:
					case	SPECTRUM_KS:
						binK[j] += tBinK[j + i*powMax]*norm;
#ifdef USE_NN_BINS
						binNK[j] += tBinNK[j + i*powMax]*norm;
#endif
						break;
					case	SPECTRUM_G:
					case	SPECTRUM_GG:
					case	SPECTRUM_GaS:
					case	SPECTRUM_GaSadd:
						binG[j] += tBinG[j + i*powMax]*norm;
#ifdef USE_NN_BINS
						binNG[j] += tBinNG[j + i*powMax]*norm;
#endif
						break;
					case	SPECTRUM_VV:
						binV[j] += tBinV[j + i*powMax]*norm;
#ifdef USE_NN_BINS
						binNV[j] += tBinNV[j + i*powMax]*norm;
#endif
						break;
					case	SPECTRUM_VNL:
						binVnl[j] += tBinVnl[j + i*powMax]*norm;
#ifdef USE_NN_BINS
						binNVnl[j] += tBinNVnl[j + i*powMax]*norm;
#endif
						break;
					case	SPECTRUM_P:
						binP[j] += tBinP[j + i*powMax]*norm;
						break;
					case	SPECTRUM_PS:
						binPS[j] += tBinPS[j + i*powMax]*norm;
						break;
					case	SPECTRUM_NN:
						binNN[j] += tBinNN[j + i*powMax];
						break;
					case	SPECTRUM_AK:
						binAK[j] += tBinAK[j + i*powMax];
						break;

					default:
						binG[j] += tBinG[j + i*powMax]*norm;
						binV[j] += tBinV[j + i*powMax]*norm;
#ifdef USE_NN_BINS
						binNG[j] += tBinNG[j + i*powMax]*norm;
						binNV[j] += tBinNV[j + i*powMax]*norm;
#endif
						break;

				}
			}
		}
	}

switch	(sType) {
		case	SPECTRUM_K:
		case	SPECTRUM_KK:
		case	SPECTRUM_KS:
			std::copy_n(binK.begin(), powMax, tBinK.begin());
			MPI_Allreduce(tBinK.data(), binK.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#ifdef USE_NN_BINS
			std::copy_n(binNK.begin(), powMax, tBinNK.begin());
			MPI_Allreduce(tBinNK.data(), binNK.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
			break;

		case	SPECTRUM_P:
			std::copy_n(binP.begin(), powMax, tBinP.begin());
			MPI_Allreduce(tBinP.data(), binP.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			break;

		case	SPECTRUM_PS:
			std::copy_n(binPS.begin(), powMax, tBinPS.begin());
			MPI_Allreduce(tBinPS.data(), binPS.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			break;

		case	SPECTRUM_NN:
			std::copy_n(binNN.begin(), powMax, tBinNN.begin());
			MPI_Allreduce(tBinNN.data(), binNN.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			break;
		case	SPECTRUM_AK:
			std::copy_n(binAK.begin(), powMax, tBinAK.begin());
			MPI_Allreduce(tBinAK.data(), binAK.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			break;

		case	SPECTRUM_GaSadd:
		// we do not do anything, just keep each binG with its local sum
		// only when G or GaS are called we do the MPI reduce
			break;
		case	SPECTRUM_G:
		case	SPECTRUM_GaS:
		// now we assume that all 3 grad squared are in binG
		// we can reduce among ranks
			std::copy_n(binG.begin(), powMax, tBinG.begin());
			MPI_Allreduce(tBinG.data(), binG.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#ifdef USE_NN_BINS
			std::copy_n(binNG.begin(), powMax, tBinNG.begin());
			MPI_Allreduce(tBinNG.data(), binNG.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
			break;

		case	SPECTRUM_VV:
			std::copy_n(binV.begin(), powMax, tBinV.begin());
			MPI_Allreduce(tBinV.data(), binV.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#ifdef USE_NN_BINS
			std::copy_n(binNV.begin(), powMax, tBinNV.begin());
			MPI_Allreduce(tBinNV.data(), binNV.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
			break;

		case	SPECTRUM_VNL:
			std::copy_n(binVnl.begin(), powMax, tBinVnl.begin());
			MPI_Allreduce(tBinVnl.data(), binVnl.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#ifdef USE_NN_BINS
			std::copy_n(binNVnl.begin(), powMax, tBinNVnl.begin());
			MPI_Allreduce(tBinNVnl.data(), binNVnl.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
			break;

		default:
		case	SPECTRUM_GV:
			std::copy_n(binG.begin(), powMax, tBinG.begin());
			MPI_Allreduce(tBinG.data(), binG.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			std::copy_n(binV.begin(), powMax, tBinV.begin());
			MPI_Allreduce(tBinV.data(), binV.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#ifdef USE_NN_BINS
			std::copy_n(binNG.begin(), powMax, tBinNG.begin());
			MPI_Allreduce(tBinNG.data(), binNG.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			std::copy_n(binNV.begin(), powMax, tBinNV.begin());
			MPI_Allreduce(tBinNV.data(), binNV.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
			break;
	}
	LogMsg(VERB_NORMAL,"[FB] Bins filled sType = %d", sType) ;
	prof.stop();
	prof.add(std::string("Fill Bins"), 0.0, 0.0);
}



/* computes efficiently (int) floor(sqrt(aux)); */
int inflsq(double aux)
{
	int ts = 0 ;
	while (ts*ts <= aux)
		ts++;
	ts--;
	return ts;
}

/* masks a spherical ball around a set of points with mpi
   using strdata matrix */
void	SpecBin::maskball	(double radius_mask, char DEFECT_LABEL, char MASK_LABEL) {

	Profiler &prof = getProfiler(PROF_SPEC);
	prof.start();

	int rz = (int) floor(radius_mask);
	// LogOut("[MB] MASKBALL %.2f>%d(sdlab) %lu (>>mask) %d", radius_mask, rz, DEFECT_LABEL, MASK_LABEL) ;
	LogMsg(VERB_NORMAL,"[MB] MASKBALL %.2f > %d (sdlab) %d (>>mask) %d", radius_mask, rz, DEFECT_LABEL, MASK_LABEL) ;
	/* Build the ball, loop around the center of the ball
		ball is set of coordinates for which
		d^2 = x^2+y^2+z^2 < radius_mask^2
		and consists on a loop structure
		z in (z0 - rz, z0 + rz)
		y in (y0 - ry(z-z0), y0 + ry(z))
		x in (x0 - rz(z-z0,y-y0), x0 + rx(z-z0,y-y0))
		we only need to compute the
		functions rz,ry,yx
		rz = floor(radius_mask)
		ry = floor( sqrt(radius_mask^2-(z-z0)^2) )
		rx = floor( sqrt(radius_mask^2-(z-z0)^2)-(y-y0)^2) )

		mpi z splitting implies the z-loop will be broken in the frontier
		if this is the case
			calculate how many points to mask further in the z direction
			and send them to a ghost
			receive and mask

			for mpi reasons do first the first and last rz slices,
			arrange ghosts and send
			compute bulk
			force receive and compute ghosts again

			problem with race conditions -> atomic updates
		 */

	char *strdaa = static_cast<char *>(static_cast<void *>(field->sData()));

	if	(field->Folded())
	{
		Folder	munge(field);
		munge(UNFOLD_ALL);
	}

	/*clear ghost region
		Compute 1st last slice
		compose ghosts
		send ghosts */

		LogMsg(VERB_PARANOID,"[MB] clear g") ;LogFlush();


		/* ghosts */
		short *mFG   = static_cast<short *>(field->mFrontGhost());
		short *mBG   = static_cast<short *>(field->mBackGhost());
		#pragma omp parallel for
		for (size_t idx = 0 ; idx < LyLy*2; idx++){
			mFG[idx] = 0;
			mBG[idx] = 0;
		}

		/* send info from mCpu/mBackGhost to mBackGhost+slice/mCpu+slice */
		const int sliceBytes = LyLy*sizeof(short);
		void *sB = field->mFrontGhost();
		void *rF = static_cast<void *> (static_cast<char *> (field->mBackGhost()) + LyLy*sizeof(short));
		void *sF = field->mBackGhost();
		void *rB = static_cast<void *> (static_cast<char *> (field->mCpu())       + LyLy*sizeof(short));
		LogMsg(VERB_PARANOID,"[MB] sizeofshort %d ",sizeof(short)) ;LogFlush();
		int mp = 0;
		int ml = 0;
		int mr = 0;
		LogMsg(VERB_PARANOID,"[MB] boundary ",radius_mask) ;LogFlush();
		// avoids errors with thin simulations
		size_t lzmin = Lz-rz;
		size_t lzmax = Lz+rz;
		if (rz > Lz){
			lzmin = 0;
			lzmax = Lz;
		}

		#pragma omp parallel for reduction(+:mp,mr,ml)
		for (size_t liz = lzmin ; liz < lzmax; liz++) {
			int z = liz % Lz ;
			size_t zi = Ly*Ly*z ;
			for (int y=0; y < Ly; y++) {
				size_t yi = Ly*y ;
				for (int x=0; x < Ly; x++) {
					size_t idx = zi + yi + x;

					short aux;
					short load;
					/* Signals by any defect */
					if (strdaa[idx] & DEFECT_LABEL){

						int rzmin = -rz;
						int rzmax = rz;
						if (rz > Lz){ rzmin = 0;  rzmax = Lz -1 ; }

						for (int dz = rzmin; dz <= rzmax; dz++) {
							int zii = z+dz;
							double raux = (radius_mask * radius_mask) - ((double) (dz*dz));
							// int ry = (int) floor(sqrt(raux));
							int ry = inflsq(raux);
							for (int dy=-ry; dy <= ry; dy++) {
								int yii = (Ly+y+dy) % Ly;
								// int rx = (int) floor(sqrt(aux -(dy)*(dy)));
								int rx = inflsq(raux- ((double) (dy*dy)));
								for (int dx=-rx; dx <= rx; dx++) {
									int xii = (Ly+x+dx) % Ly;

									if (z + dz < 0)
									{
										ml++;
										load = (short) -(z+dz);
										#pragma omp atomic read
										 	aux = mFG[(size_t) Ly*yii+xii];

										if ( load > aux ){
											#pragma omp atomic write
												mFG[(size_t) Ly*yii+xii] = load;
										}
									}
									else {
										if (z + dz >= (int) Lz)
										{
											mr++;
											load = (short) (z + dz + 1 - (int) Lz);
											#pragma omp atomic read
												aux = mBG[(size_t) (Ly*yii+xii)];

											if (load > aux){
												#pragma omp atomic write
													mBG[(size_t) (Ly*yii+xii)] = load;
											}
										}
										else
										{
											mp++;
											#pragma omp atomic update
											strdaa[(size_t) (LyLy*zii + Ly*yii + xii)] |= MASK_LABEL;
										}
									}
								}
							}
						}
					} //end if defect!

				}
			}
		} // end bulk loop


	LogMsg(VERB_PARANOID,"[MB] send ghosts") ;LogFlush();
	field->sendGeneral(COMM_SDRV, sliceBytes, MPI_BYTE, sB, rF, sF, rB);

	// Avoid bulk if thin slice
	int mpb = 0;
	if ( Lz > rz)
	{
		LogMsg(VERB_PARANOID,"[MB] bulk") ;LogFlush();

		/* compute bulk
		distribute in xy
		*/

		#pragma omp parallel for reduction(+:mpb)
		for (size_t z = rz; z < Lz-rz; z++) {
			size_t zi = Ly*Ly*z ;
			for (size_t y=0; y < Ly; y++) {
				size_t yi = Ly*y ;
				for (size_t x=0; x < Ly; x++) {
					size_t idx = zi + yi + x;

					/* Signals by any defect */
					if (strdaa[idx] & DEFECT_LABEL){
						for (int dz = -rz; dz <= rz; dz++) {
							double raux = radius_mask*radius_mask- (double) ((dz)*(dz));
							int ry = inflsq(raux);
							for (int dy=-ry; dy <= ry; dy++) {
								int yii = (Ly+y+dy) % Ly;
								int rx = inflsq(raux -(dy*dy));
								for (int dx=-rx; dx <= rx; dx++) {
									int xii = (Ly+x+dx) % Ly;
									mpb++;
									size_t uidx = (z+dz)*LyLy + Ly*yii + xii;
									#pragma omp atomic update
									strdaa[uidx] |= MASK_LABEL;
								}
							}
						}
					} //end if defect!

				}
			}
		} // end bulk loop
	}

	/* receive ghosts and mark the mask */

	LogMsg(VERB_PARANOID,"[MB] receive? ... ") ;LogFlush();
	field->sendGeneral(COMM_WAIT, sliceBytes, MPI_BYTE, sB, rF, sF, rB);
	LogMsg(VERB_PARANOID,"[MB] yes!") ;LogFlush();
	int irF = 0;
	int irB = 0;

	if (Lz > 1){
		#pragma omp parallel for reduction(+:irF,irB)
		for (size_t idx = 0 ; idx < LyLy; idx++){
			if (mFG[idx+LyLy] > 0) {
				short Dz = mFG[idx+LyLy];
				irB += Dz;
				for (size_t dz =0; dz < Dz; dz++)
					strdaa[idx + dz*LyLy] |= MASK_LABEL;
			}
			if (mBG[idx+LyLy] > 0){
				short Dz = mBG[idx+LyLy];
				irF += Dz;
				for (size_t dz = 0; dz < Dz; dz++)
					strdaa[idx + LyLy*(Lz-1-dz)] |= MASK_LABEL;
			}
		}
	}

	LogMsg(VERB_NORMAL,"[MB] done! masked points bulk %d border %d, (MPIed <-0 %d, MPIed Lz-> %d) ",mpb, mp, ml, mr) ;
	LogMsg(VERB_NORMAL,"[MB] received from back %d (slices 0...r) from forward %d (Lz-1... Lz-r-1)",irB,irF) ;LogFlush();

	prof.stop();
	prof.add(std::string("Mask Ball"), 0.0, 0.0);
}





void	SpecBin::pRun	() {

	Profiler &prof = getProfiler(PROF_SPEC);


	size_t dSize    = (size_t) (field->Precision());
	size_t dataLine = dSize*Ly;
	size_t Sm	= Ly*Lz;

	char *m2 = static_cast<char *>(field->m2Cpu());
	char *mA = m2;

	LogMsg(VERB_NORMAL,"[pRun] Called with status field->statusM2()=%d",field->m2Status()) ;
	LogMsg(VERB_NORMAL,"[pRun]             status field->statusM2h()=%d",field->m2hStatus()) ;

	if ((field->m2Status() != M2_ENERGY) && (field->m2Status() != M2_ENERGY_FFT)) {
		if ((field->m2hStatus() != M2_ENERGY)){
			LogError ("Power spectrum requires previous calculation of the energy. Ignoring pRun request.");
			return;
		} else {
			mA = static_cast<char *>(field->m2half());
		}
	}


	/*If we do not have the energy fft in m2, calculate it*/
	if (field->m2Status() != M2_ENERGY_FFT) {
		// contrast bin is assumed in m2 or m2h
		// pad in place
			prof.start();
		for (int sl=Sm-1; sl>=0; sl--) {
			auto	oOff = sl*dSize*(Ly);
			auto	fOff = sl*dSize*(Ly+2);
			memmove	(m2+fOff, mA+oOff, dataLine);
		}
			prof.stop();
				prof.add(std::string("memmove"), 0.0, 0.0);

			prof.start();
		auto &myPlan = AxionFFT::fetchPlan("pSpecAx");
		myPlan.run(FFT_FWD);
			prof.stop();
				prof.add(std::string("pSpecAx"), 0.0, 0.0);
	}

	/* empty the bins */
	binP.assign(powMax, 0.);

	// the function gives the same in spectral or !spectral
	switch (fPrec) {
		case	FIELD_SINGLE:
			if (spec)
				fillBins<float,  SPECTRUM_P, true> ();
			else
				fillBins<float,  SPECTRUM_P, false>();
			break;

		case	FIELD_DOUBLE:
			if (spec)
				fillBins<double,  SPECTRUM_P, true> ();
			else
				fillBins<double,  SPECTRUM_P, false>();
			break;

		default:
			LogError ("Wrong precision");
			break;
	}

	field->setM2     (M2_ENERGY_FFT);

}

// axion number spectrum

void	SpecBin::nRun	(SpectrumMaskType mask, nRunType nrt){

	switch (mask)
	{
		case SPMASK_FLAT :
				switch (fPrec)
				{
					case FIELD_SINGLE :
					SpecBin::nRun<float,SPMASK_FLAT> (nrt);
					break;

					case FIELD_DOUBLE :
					SpecBin::nRun<double,SPMASK_FLAT> (nrt);
					break;

					default :
					LogError("[Spectrum nRun] precision not reconised.");
					break;
				}
				break;

		case SPMASK_VIL :
				switch (fPrec)
				{
					case FIELD_SINGLE :
					SpecBin::nRun<float,SPMASK_VIL> (nrt);
					break;

					case FIELD_DOUBLE :
					SpecBin::nRun<double,SPMASK_VIL> (nrt);
					break;

					default :
					LogError("[Spectrum nRun] precision not reconised.");
					break;
				}
			break;

			case SPMASK_VIL2 :
					switch (fPrec)
					{
						case FIELD_SINGLE :
						SpecBin::nRun<float,SPMASK_VIL2> (nrt);
						break;

						case FIELD_DOUBLE :
						SpecBin::nRun<double,SPMASK_VIL2> (nrt);
						break;

						default :
						LogError("[Spectrum nRun] precision not reconised.");
						break;
					}
		break;

		case SPMASK_SAXI :
				switch (fPrec)
				{
					case FIELD_SINGLE :
					SpecBin::nRun<float,SPMASK_SAXI> (nrt);
					break;

					case FIELD_DOUBLE :
					SpecBin::nRun<double,SPMASK_SAXI> (nrt);
					break;

					default :
					LogError("[Spectrum nRun] precision not reconised.");
					break;
				}
		break;

		case SPMASK_REDO :
				switch (fPrec)
				{
					case FIELD_SINGLE :
					SpecBin::nRun<float,SPMASK_REDO> (nrt);
					break;

					case FIELD_DOUBLE :
					SpecBin::nRun<double,SPMASK_REDO> (nrt);
					break;

					default :
					LogError("[Spectrum nRun] precision not reconised.");
					break;
				}
		break;

		case SPMASK_DIFF :
				switch (fPrec)
				{
					case FIELD_SINGLE :
					SpecBin::nRun<float,SPMASK_DIFF> (nrt);
					break;

					case FIELD_DOUBLE :
					SpecBin::nRun<double,SPMASK_DIFF> (nrt);
					break;

					default :
					LogError("[Spectrum nRun] precision not reconised.");
					break;
				}
		break;

		case SPMASK_GAUS :
				switch (fPrec)
				{
					case FIELD_SINGLE :
					SpecBin::nRun<float,SPMASK_GAUS> (nrt);
					break;

					case FIELD_DOUBLE :
					SpecBin::nRun<double,SPMASK_GAUS> (nrt);
					break;

					default :
					LogError("[Spectrum nRun] precision not reconised.");
					break;
				}
		break;

		case SPMASK_BALL :
				switch (fPrec)
				{
					case FIELD_SINGLE :
					SpecBin::nRun<float,SPMASK_BALL> (nrt);
					break;

					case FIELD_DOUBLE :
					SpecBin::nRun<double,SPMASK_BALL> (nrt);
					break;

					default :
					LogError("[Spectrum nRun] precision not reconised.");
					break;
				}
		break;

		case SPMASK_AXIT :
				switch (fPrec)
				{
					case FIELD_SINGLE :
					SpecBin::nRun<float,SPMASK_AXIT> (nrt);
					break;

					case FIELD_DOUBLE :
					SpecBin::nRun<double,SPMASK_AXIT> (nrt);
					break;

					default :
					LogError("[Spectrum nRun] precision not reconised.");
					break;
				}
		break;

		case SPMASK_AXIT2 :
				switch (fPrec)
				{
					case FIELD_SINGLE :
					SpecBin::nRun<float,SPMASK_AXIT2> (nrt);
					break;

					case FIELD_DOUBLE :
					SpecBin::nRun<double,SPMASK_AXIT2> (nrt);
					break;

					default :
					LogError("[Spectrum nRun] precision not reconised.");
					break;
				}
		break;

		default:
		LogError("[Spectrum nRun] SPMASK not recognised!");
		break;
	}
}


template<typename Float, SpectrumMaskType mask>
void	SpecBin::nRun	(nRunType nrt) {

	Profiler &prof = getProfiler(PROF_SPEC);

	LogMsg(VERB_HIGH,"[nRun] Called with mask %d nrt %d",mask,nrt);
	/* test if everything we need is there in the different cases */
	switch(mask)
	{
		case SPMASK_REDO:
			if ((field->sDStatus() & SD_MASK))
				LogMsg(VERB_NORMAL,"nRun with SPMASK_REDO ok SPMASK=%d field->statusSD()=%d",SPMASK_REDO,field->sDStatus()) ;
			else{
			LogMsg(VERB_NORMAL,"nRun with SPMASK_REDO but SPMASK=%d field->statusSD()=%d ... EXIT!",SPMASK_REDO,field->sDStatus()) ;
			return ;
			}
		break;
		case SPMASK_GAUS:
			if ((field->sDStatus() & SD_MASK))
				LogMsg(VERB_NORMAL,"nRun with SPMASK_GAUS ok SPMASK=%d field->statusSD()=%d",SPMASK_GAUS,field->sDStatus()) ;
			else{
			LogMsg(VERB_NORMAL,"nRun with SPMASK_GAUS but SPMASK=%d field->statusSD()=%d ... EXIT!",SPMASK_GAUS,field->sDStatus()) ;
			return ;
			}
		break;
		case SPMASK_DIFF:
			if ((field->sDStatus() & SD_MASK))
				LogMsg(VERB_NORMAL,"nRun with SPMASK_DIFF ok SPMASK=%d field->statusSD()=%d",SPMASK_DIFF,field->sDStatus()) ;
			else{
			LogMsg(VERB_NORMAL,"nRun with SPMASK_DIFF but SPMASK=%d field->statusSD()=%d ... EXIT!",SPMASK_DIFF,field->sDStatus()) ;
			return ;
			}
		break;
		case SPMASK_AXIT2:
			if ((field->sDStatus() & SD_AXITONMASK))
				LogMsg(VERB_NORMAL,"nRun with SPMASK_AXIT2 ok SPMASK=%d field->statusSD()=%d",SPMASK_AXIT2,field->sDStatus()) ;
			else{
			LogMsg(VERB_NORMAL,"nRun with SPMASK_DIFF but SPMASK=%d field->statusSD()=%d ... EXIT!",SPMASK_AXIT2,field->sDStatus()) ;
			return ;
			}
		break;
		default:
		LogMsg(VERB_NORMAL,"nRun with self-contained mask") ;
		break;

	}

		prof.start();
	binK.assign(powMax, 0.);
	binG.assign(powMax, 0.);
		// // TEMP INDIVIDUAL XYZ
		// binGy.assign(powMax, 0.);
		// binGz.assign(powMax, 0.);
	binV.assign(powMax, 0.);
	binVnl.assign(powMax, 0.);
#ifdef USE_NN_BINS
	binNK.assign(powMax, 0.);
	binNG.assign(powMax, 0.);
	binNV.assign(powMax, 0.);
	binNVnl.assign(powMax, 0.);
#endif
	binPS.assign(powMax, 0.);
	binP.assign(powMax, 0.);
	if (mask == SPMASK_SAXI)
		binPS.assign(powMax, 0.);

	prof.stop();
		prof.add(std::string("assign"), 0.0, 0.0);

	// using cFloat = std::complex<Float>;
	std::complex<Float> zaskaF((Float) zaskar, 0.);
	Float zaskaFF = (Float) zaskar;

	if	(field->Folded())
	{
		Folder	munge(field);
		munge(UNFOLD_ALL);
	}

	field->sendGhosts(FIELD_M, COMM_SDRV);
	field->sendGhosts(FIELD_M, COMM_WAIT);

	switch (fType) {
		case	FIELD_SAXION:
		{
			// FIX ME vectorise loops?
			// std::complex<Float> *ma     = static_cast<std::complex<Float>*>(field->mStart());
			std::complex<Float> *ma     = static_cast<std::complex<Float>*>(field->mCpu())+(field->getNg()-1)*field->Surf();
			std::complex<Float> *va     = static_cast<std::complex<Float>*>(field->vCpu());
			Float *m2sa                 = static_cast<Float *>(field->m2Cpu());
			// Float *m2sax                = static_cast<Float *>(field->m2Cpu()) + field->eSize();
			Float *m2sax                = static_cast<Float *>(field->m2half());
			char *strdaa = static_cast<char *>(static_cast<void *>(field->sData()));

			// r2c FFT in m2
			auto &myPlan = AxionFFT::fetchPlan("pSpecAx");

			// FIX ME vectorise
			/* Kinetic energy includes subgrid correction */

			if (nrt & NRUN_K){

				buildc_k(field, PFIELD_M2, zaskar, mask, true);

				/* corrected */
				LogMsg(VERB_HIGH,"[nRun] FFT") ;
					prof.start();
				myPlan.run(FFT_FWD);
					prof.stop();
						prof.add(std::string("pSpecAx"), 0.0, 0.0);

				// FIX ME FOR SAXI
				LogMsg(VERB_HIGH,"[nRun] bin") ;
				if (spec)
					fillBins<Float,  SPECTRUM_KK, true> ();
				else
					fillBins<Float,  SPECTRUM_KK, false>();
			}

			/* Kinetic energy OLD VERSION*/
			if (nrt & NRUN_CK)
			{
				LogMsg(VERB_HIGH,"[nRun] C loop") ;

				buildc_k(field, PFIELD_M2, zaskar, mask, false);

				/* uncorrected */
				LogMsg(VERB_HIGH,"[nRun] FFT") ;
					prof.start();
				myPlan.run(FFT_FWD);
					prof.stop();
						prof.add(std::string("pSpecAx"), 0.0, 0.0);

				// FIX ME FOR SAXI
				LogMsg(VERB_HIGH,"[nRun] bin") ;
				if (spec)
					fillBins<Float,  SPECTRUM_KK, true> ();
				else
					fillBins<Float,  SPECTRUM_KK, false>();

				/* overwrites K! */
			}

			/* Potential energy*/
			if ( (nrt & NRUN_V) && (mass2 > 0.0))
			{
					LogMsg(VERB_HIGH,"[nRun] V loop") ;
					buildc_v (field, PFIELD_M2, zaskar, mask);

					// POTENTIAL:
					LogMsg(VERB_HIGH,"[nRun] FFT") ;
						prof.start();
					myPlan.run(FFT_FWD);
						prof.stop();
							prof.add(std::string("pSpecAx"), 0.0, 0.0);

					if (spec)
						fillBins<Float,  SPECTRUM_VNL, true> ();
					else
						fillBins<Float,  SPECTRUM_VNL, false>();

					// linear version
					// Copy m2aux -> m2
					// we move real, not complex numbers
					size_t dataTotalSize2 = field->Precision()*field->eSize();
					char *m2C  = static_cast<char *>(field->m2Cpu());
					char *m2Ch = static_cast<char *>(field->m2half());
						prof.start();
					memmove	(m2C, m2Ch, dataTotalSize2);
						prof.stop();
							prof.add(std::string("memmove"), 0.0, 0.0);

					// POTENTIAL LINEAR:
					LogMsg(VERB_HIGH,"[nRun] FFT") ;
						prof.start();
					myPlan.run(FFT_FWD);
						prof.stop();
							prof.add(std::string("pSpecAx"), 0.0, 0.0);

					if (spec)
						fillBins<Float,  SPECTRUM_V, true> ();
					else
						fillBins<Float,  SPECTRUM_V, false>();
			}
			else
			LogMsg(VERB_HIGH,"[nRun] axion mass 0 (%.6e) -> no V loop ",mass2) ;

			/* Potential energy OLD VERSION*/
			if ( (nrt & NRUN_CV) && (mass2 > 0.0) )
			{

					/* Potential ... WARNING: still experimental */
					// FIX me potential dependency

					// conformal mass square root of topological susceptibility
					// we use a factor of more because by default 1/2 is included in fillbins
					// because of the kin and grad terms
				Float mass = (Float) std::sqrt(mass2);
				Float iR   = (Float) 1/Rscale;
				Float iR2   = (Float) 1/(Rscale*Rscale);
					prof.start();
				if (mass > 0.0)
				{
					#pragma omp parallel for schedule(static)
					for (size_t iz=0; iz < Lz; iz++) {
						size_t zo = Ly*(Ly+2)*iz ;
						size_t zi = Ly*Ly*iz ;
						size_t zp = Ly*Ly*(iz+1) ;
						for (size_t iy=0; iy < Ly; iy++) {
							size_t yo = (Ly+2)*iy ;
							size_t yi = Ly*iy ;
							size_t yp = Ly*((iy+1)%Ly) ;
							for (size_t ix=0; ix < Ly; ix++) {
								size_t odx = ix + yo + zo; size_t idx = ix + yi + zi;
								size_t iyM = ix + yp + zi; size_t izM = ix + yi + zp;

								switch(mask){
									case SPMASK_FLAT:
											// cosine version
											// m2sa[odx] = std::sqrt(2*(1.-std::real( (ma[idx]-zaskaF)/std::abs(ma[idx]-zaskaF)))  );
											// linear version, matches better with NR axion number although it is not accurate
											//
											// m2sa[odx] = mass*std::abs(std::arg(ma[idx]-zaskaF));
											m2sa[odx] = mass*Rscale*std::arg(ma[idx]-zaskaF);
											break;
									case SPMASK_REDO:
											if (strdaa[idx] & STRING_MASK){
													m2sa[odx] = 0 ;
											}
											else{
													m2sa[odx]  = mass*Rscale*std::arg(ma[idx]-zaskaF);
											}
											break;
									case SPMASK_VIL:
											m2sa[odx]  = mass*(std::abs(ma[idx]-zaskaF))*std::arg(ma[idx]-zaskaF);
											break;
										case SPMASK_VIL2:
												m2sa[odx]  = mass*(std::pow(std::abs(ma[idx]-zaskaF),2)*iR)*std::arg(ma[idx]-zaskaF);
												break;
										case SPMASK_SAXI:
										// what do I do here?
												break;
								} //end mask
							}
						}
					} //end volume loop

					// POTENTIAL:
					myPlan.run(FFT_FWD);

					if (spec)
						fillBins<Float,  SPECTRUM_VV, true> ();
					else
						fillBins<Float,  SPECTRUM_VV, false>();
					}
					prof.stop();
						prof.add(std::string("Build V"), 0.0, 0.0);
			} // potential

			/* Gradient energy */
			if (nrt & NRUN_G)
			{
				/* Gradient X */
				LogMsg(VERB_HIGH,"[nRun] GX loop") ;
				buildc_gx (field, PFIELD_M2, zaskar, mask, true);

				// r2c FFT in m2
				LogMsg(VERB_HIGH,"[nRun] FFT") ;
					prof.start();
				myPlan.run(FFT_FWD);
					prof.stop();
						prof.add(std::string("pSpecAx"), 0.0, 0.0);

				controlxyz = 0 ;
				if (spec)
					fillBins<Float,  SPECTRUM_GaSadd, true> ();
				else
					fillBins<Float,  SPECTRUM_GaSadd, false>();

				// // TEMP INDIVIDUAL XYZ
				// controlxyz = 0 ;
				// fillBins<Float,  SPECTRUM_GaS, false>();
				// binGy = binG;
				// binG.assign(powMax, 0.);

				/* Gradient YZ */
				LogMsg(VERB_HIGH,"[nRun] GYZ loop") ;
				buildc_gyz (field, PFIELD_M2, zaskar, mask, true);

					// GRADIENT Y:
					LogMsg(VERB_HIGH,"[nRun] FFT") ;
						prof.start();
					myPlan.run(FFT_FWD);
						prof.stop();
							prof.add(std::string("pSpecAx"), 0.0, 0.0);

					controlxyz = 1;
					if (spec)
						fillBins<Float,  SPECTRUM_GaSadd, true> ();
					else
						fillBins<Float,  SPECTRUM_GaSadd, false>();

					// // TEMP INDIVIDUAL XYZ
					// fillBins<Float,  SPECTRUM_GaS, false>();
					// binGz = binG;
					// binG.assign(powMax, 0.);


					// GRADIENT Z:
					// Copy m2aux -> m2
					// we move real, not complex numbers
						prof.start();
					size_t dataTotalSize2 = field->Precision()*field->eSize();
					char *m2C  = static_cast<char *>(field->m2Cpu());
					char *m2Ch = static_cast<char *>(field->m2half());
					memmove	(m2C, m2Ch, dataTotalSize2);
						prof.stop();
							prof.add(std::string("memmove"), 0.0, 0.0);

					/* unpad m2 in place if SPMASK_GAUS/DIFF */
							prof.start();
						if (mask & (SPMASK_GAUS|SPMASK_DIFF)){
							size_t dl = Ly*field->Precision();
							size_t pl = (Ly+2)*field->Precision();
							size_t ss	= Ly*Lz;

							for (size_t sl=1; sl<LyLz; sl++) {
								size_t	oOff = sl*dl;
								size_t	fOff = sl*pl;
								memmove	(m2C+oOff, m2C+fOff, dl);
								}
						}
							prof.stop();
								prof.add(std::string("unpad"), 0.0, 0.0);


					LogMsg(VERB_HIGH,"[nRun] FFT") ;
						prof.start();
					myPlan.run(FFT_FWD);
						prof.stop();
							prof.add(std::string("pSpecAx"), 0.0, 0.0);

					controlxyz = 2;
					if (spec)
						fillBins<Float,  SPECTRUM_GaS, true> ();
					else
						fillBins<Float,  SPECTRUM_GaS, false>();
					// //	TEMP INDIVIDUAL XYZ
					// fillBins<Float,  SPECTRUM_GaS, false>();

			}

			/* Gradient energy OLD VERSION*/
			if (nrt & NRUN_CG)
			{
				/* Gradient X */
				LogMsg(VERB_HIGH,"[nRun] CGX loop") ;
				// switch (mask){
				// 	default:
				// 	case SPMASK_FLAT:
				// 		buildc_gx<SPMASK_FLAT,false> (field, PFIELD_M2, zaskar);
				// 	break;
				// 	case SPMASK_REDO:
				// 		buildc_gx<SPMASK_REDO,false> (field, PFIELD_M2, zaskar);
				// 	break;
				// 	case SPMASK_GAUS:
				// 	case SPMASK_DIFF:
				// 		buildc_gx<SPMASK_DIFF,false> (field, PFIELD_M2, zaskar);
				// 	break;
				// 	case SPMASK_VIL:
				// 		buildc_gx<SPMASK_VIL,false> (field, PFIELD_M2, zaskar);
				// 	break;
				// 	case SPMASK_VIL2:
				// 		buildc_gx<SPMASK_VIL2,false> (field, PFIELD_M2, zaskar);
				// 	break;
				// }
				/* Gradient X */
					prof.start();
				#pragma omp parallel for schedule(static)
				for (size_t iz=0; iz < Lz; iz++) {
					size_t zo = Ly*(Ly+2)*iz ;
					size_t zi = LyLy*(iz+1) ;
					for (size_t iy=0; iy < Ly; iy++) {
						size_t yo = (Ly+2)*iy ;
						size_t yi = Ly*iy ;
						for (size_t ix=0; ix < Ly; ix++) {
							size_t odx = ix + yo + zo; size_t idx = ix + yi + zi; size_t ixM = ((ix + 1) % Ly) + yi + zi;

							switch(mask){
								case SPMASK_FLAT:
										m2sa[odx] = (2*Rscale/depta)*std::imag((ma[ixM]-ma[idx])/(ma[ixM]+ma[idx]-zaskaF-zaskaF));
										break;
								case SPMASK_REDO:
										if (strdaa[idx-LyLy] & STRING_MASK)
												m2sa[odx] = 0 ;
										else
												m2sa[odx] = (2*Rscale/depta)*std::imag((ma[ixM]-ma[idx])/(ma[ixM]+ma[idx]-zaskaF-zaskaF));
										break;
								case SPMASK_GAUS:
								case SPMASK_DIFF:
										/* keep the mask in mhalf so only one load is possible */
										m2sa[odx] = m2sax[idx-LyLy]*(2*Rscale/depta)*std::imag((ma[ixM]-ma[idx])/(ma[ixM]+ma[idx]-zaskaF-zaskaF));;
										break;
								case SPMASK_VIL:
										m2sa[odx] = (2*std::abs(ma[idx]-zaskaF)/depta)*std::imag((ma[ixM]-ma[idx])/(ma[ixM]+ma[idx]-zaskaF-zaskaF));
										break;
								case SPMASK_VIL2:
										m2sa[odx] = (2*std::pow(std::abs(ma[idx]-zaskaF),2)/Rscale/depta)*std::imag((ma[ixM]-ma[idx])/(ma[ixM]+ma[idx]-zaskaF-zaskaF));
										break;
								case SPMASK_SAXI:
										m2sa[odx]  =  std::real(va[idx-LyLy]) ;
										m2sax[odx] =  std::imag(va[idx-LyLy]);
								break;

							} //end mask
						}
					}
				} // end last volume loop
					prof.stop();
						prof.add(std::string("Build GX"), 0.0, 0.0);


				// r2c FFT in m2
				LogMsg(VERB_HIGH,"[nRun] FFT") ;
					prof.start();
				myPlan.run(FFT_FWD);
					prof.stop();
						prof.add(std::string("pSpecAx"), 0.0, 0.0);

				controlxyz = 0;
				if (spec)
					fillBins<Float,  SPECTRUM_GaSadd, true> ();
				else
					fillBins<Float,  SPECTRUM_GaSadd, false>();


				/* Gradient YZ */
				// LogMsg(VERB_HIGH,"[nRun] CGYZ loop") ;
				// switch (mask){
				// 	default:
				// 	case SPMASK_FLAT:
				// 		buildc_gyz<SPMASK_FLAT,false> (field, PFIELD_M2, zaskar);
				// 	break;
				// 	case SPMASK_REDO:
				// 		buildc_gyz<SPMASK_REDO,false> (field, PFIELD_M2, zaskar);
				// 	break;
				// 	case SPMASK_GAUS:
				// 	case SPMASK_DIFF:
				// 		buildc_gyz<SPMASK_DIFF,false> (field, PFIELD_M2, zaskar);
				// 	break;
				// 	case SPMASK_VIL:
				// 		buildc_gyz<SPMASK_VIL,false> (field, PFIELD_M2, zaskar);
				// 	break;
				// 	case SPMASK_VIL2:
				// 		buildc_gyz<SPMASK_VIL2,false> (field, PFIELD_M2, zaskar);
				// 	break;
				// }
				prof.start();
			#pragma omp parallel for schedule(static)
				for (size_t iz=0; iz < Lz; iz++) {
					size_t zo = Ly*(Ly+2)*iz ;
					size_t zi = LyLy*(iz+1) ;
					size_t zp = LyLy*(iz+2) ;
					for (size_t iy=0; iy < Ly; iy++) {
						size_t yo = (Ly+2)*iy ;
						size_t yi = Ly*iy ;
						size_t yp = Ly*((iy+1)%Ly) ;
						for (size_t ix=0; ix < Ly; ix++) {
							size_t odx = ix + yo + zo; size_t idx = ix + yi + zi;
							size_t iyM = ix + yp + zi; size_t izM = ix + yi + zp;

							switch(mask){
								case SPMASK_FLAT:
										m2sa[odx]  = (2*Rscale/depta)*std::imag((ma[iyM]-ma[idx])/(ma[iyM]+ma[idx]-zaskaF-zaskaF));
										m2sax[odx] = (2*Rscale/depta)*std::imag((ma[izM]-ma[idx])/(ma[izM]+ma[idx]-zaskaF-zaskaF));
										break;
								case SPMASK_REDO:
										if (strdaa[idx-LyLy] & STRING_MASK){
												m2sa[odx] = 0 ;
												m2sax[odx] = 0 ;
										}
										else{
											m2sa[odx]  = (2*Rscale/depta)*std::imag((ma[iyM]-ma[idx])/(ma[iyM]+ma[idx]-zaskaF-zaskaF));
											m2sax[odx] = (2*Rscale/depta)*std::imag((ma[izM]-ma[idx])/(ma[izM]+ma[idx]-zaskaF-zaskaF));
										}
										break;
								case SPMASK_GAUS:
								case SPMASK_DIFF:
										/* The mask in m2sax[idx] will be destroyed but a race condition is prevented by padding m2sax */
										m2sa[odx]  = m2sax[idx-LyLy]*(2*Rscale/depta)*std::imag((ma[iyM]-ma[idx])/(ma[iyM]+ma[idx]-zaskaF-zaskaF));
										m2sax[idx-LyLy] = m2sax[idx-LyLy]*(2*Rscale/depta)*std::imag((ma[izM]-ma[idx])/(ma[izM]+ma[idx]-zaskaF-zaskaF));
										break;
								case SPMASK_VIL:
										m2sa[odx]  = (2*std::abs(ma[idx]-zaskaF)/depta)*std::imag((ma[iyM]-ma[idx])/(ma[iyM]+ma[idx]-zaskaF-zaskaF));
										m2sax[odx] = (2*std::abs(ma[idx]-zaskaF)/depta)*std::imag((ma[izM]-ma[idx])/(ma[izM]+ma[idx]-zaskaF-zaskaF));
										break;
									case SPMASK_VIL2:
											m2sa[odx]  = (2*std::pow(std::abs(ma[idx]-zaskaF),2)/Rscale/depta)*std::imag((ma[iyM]-ma[idx])/(ma[iyM]+ma[idx]-zaskaF-zaskaF));
											m2sax[odx] = (2*std::pow(std::abs(ma[idx]-zaskaF),2)/Rscale/depta)*std::imag((ma[izM]-ma[idx])/(ma[izM]+ma[idx]-zaskaF-zaskaF));
											break;
									case SPMASK_SAXI:
											m2sa[odx]  =  std::real(ma[idx-LyLy]) ;
											m2sax[odx] =  std::imag(ma[idx-LyLy]);
											break;
							} //end mask
						}
					}
				} //end volume loop
				prof.stop();
					prof.add(std::string("Build GYZ"), 0.0, 0.0);

					// GRADIENT Y:
					LogMsg(VERB_HIGH,"[nRun] FFT") ;
						prof.start();
					myPlan.run(FFT_FWD);
						prof.stop();
							prof.add(std::string("pSpecAx"), 0.0, 0.0);

					controlxyz = 1;
					if (spec)
						fillBins<Float,  SPECTRUM_GaSadd, true> ();
					else
						fillBins<Float,  SPECTRUM_GaSadd, false>();

					// GRADIENT Z:
					// Copy m2aux -> m2
					// we move real, not complex numbers
					size_t dataTotalSize2 = field->Precision()*field->eSize();
					char *m2C  = static_cast<char *>(field->m2Cpu());
					char *m2Ch = static_cast<char *>(field->m2half());
						prof.start();
					memmove	(m2C, m2Ch, dataTotalSize2);
						prof.stop();
							prof.add(std::string("memmove"), 0.0, 0.0);

					/* unpad m2 in place if SPMASK_GAUS/DIFF */
							prof.start();
						if (mask & (SPMASK_GAUS|SPMASK_DIFF)){
							size_t dl = Ly*field->Precision();
							size_t pl = (Ly+2)*field->Precision();
							size_t ss	= Ly*Lz;

							for (size_t sl=1; sl<LyLz; sl++) {
								size_t	oOff = sl*dl;
								size_t	fOff = sl*pl;
								memmove	(m2C+oOff, m2C+fOff, dl);
								}
						}
						prof.stop();
							prof.add(std::string("unpad"), 0.0, 0.0);

						LogMsg(VERB_HIGH,"[nRun] FFT") ;
							prof.start();
						myPlan.run(FFT_FWD);
							prof.stop();
								prof.add(std::string("pSpecAx"), 0.0, 0.0);

					controlxyz = 2;
					if (spec)
						fillBins<Float,  SPECTRUM_GaS, true> ();
					else
						fillBins<Float,  SPECTRUM_GaS, false>();
			}

			field->setM2     (M2_DIRTY);
		}
		break;

		case	FIELD_AXION_MOD:
		case	FIELD_AXION:
		{
			auto &myPlan = AxionFFT::fetchPlan("pSpecAx");

			char *mO = static_cast<char *>(field->mStart());
			char *vO = static_cast<char *>(field->vCpu());
			char *mF = static_cast<char *>(field->m2Cpu());

			Float *m   = static_cast<Float*>(field->mStart());
			Float *v   = static_cast<Float*>(field->vCpu());
			Float *m2  = static_cast<Float*>(field->m2Cpu());

			char *strdaa = static_cast<char *>(static_cast<void *>(field->sData()));

			size_t dataLine = field->DataSize()*Ly;
			size_t Sm	= Ly*Lz;

			// Copy m -> m2 with padding

			if (nrt & (NRUN_G | NRUN_CG | NRUN_V | NRUN_CV) )
			{
				LogMsg(VERB_HIGH,"[nRun] GV loop (Axion)") ;
				if (mask & SPMASK_FLAT){
					#pragma omp parallel for schedule(static)
					for (uint sl=0; sl<Sm; sl++) {
						auto	oOff = sl*field->DataSize()* Ly;
						auto	fOff = sl*field->DataSize()*(Ly+2);
						memcpy	(mF+fOff, mO+oOff, dataLine);
					}
				} else {
					#pragma omp parallel for schedule(static)
					for (size_t iz=0; iz < Lz; iz++) {
						size_t zo = Ly*(Ly+2)*iz ;
						size_t zi = Ly*Ly*iz ;
						for (size_t iy=0; iy < Ly; iy++) {
							size_t yo = (Ly+2)*iy ;
							size_t yi = Ly*iy ;
							for (size_t ix=0; ix < Ly; ix++) {
								size_t odx = ix + yo + zo; size_t idx = ix + yi + zi;

								switch(mask){
									case SPMASK_AXIT:
											m2[odx] = m[idx]*0.5*(1-std::tanh(5*(m2[idx])-1)) ;
										break;
									default:
									case SPMASK_AXIT2:
											if (strdaa[idx] & STRING_MASK)
													m2[odx] = 0 ;
											else
													m2[odx] = m[idx];
										break;
								} //end mask
						}}} // end last volume loop
				}

				myPlan.run(FFT_FWD);

				if (spec)
					fillBins<Float,  SPECTRUM_GV, true> ();
				else
					fillBins<Float,  SPECTRUM_GV, false>();
			}

			if (nrt & (NRUN_K | NRUN_CK ) )
			{
				LogMsg(VERB_HIGH,"[nRun] K loop (Axion)") ;
				if (mask & SPMASK_FLAT){
					// Copy v -> m2 with padding
					#pragma omp parallel for schedule(static)
					for (uint sl=0; sl<Sm; sl++) {
						auto	oOff = sl*field->DataSize()* Ly;
						auto	fOff = sl*field->DataSize()*(Ly+2);
						memcpy	(mF+fOff, vO+oOff, dataLine);
					}
				} else {
					#pragma omp parallel for schedule(static)
					for (size_t iz=0; iz < Lz; iz++) {
						size_t zo = Ly*(Ly+2)*iz ;
						size_t zi = Ly*Ly*iz ;
						for (size_t iy=0; iy < Ly; iy++) {
							size_t yo = (Ly+2)*iy ;
							size_t yi = Ly*iy ;
							for (size_t ix=0; ix < Ly; ix++) {
								size_t odx = ix + yo + zo; size_t idx = ix + yi + zi;

								switch(mask){
									case SPMASK_AXIT:
											m2[odx] = m[idx]*0.5*(1-std::tanh(5*(m2[idx])-1)) ;
										break;
									default:
									case SPMASK_AXIT2:
											if (strdaa[idx] & STRING_MASK)
													m2[odx] = 0 ;
											else
													m2[odx] = v[idx];
										break;
								} //end mask
						}}} // end last volume loop
				}

				myPlan.run(FFT_FWD);

				if (spec)
					fillBins<Float,  SPECTRUM_K, true> ();
				else
					fillBins<Float,  SPECTRUM_K, false>();
			}
		/* If cosine potential the energy is
		R^-4 [ (psi')^2/2 + (grad phi)^2/2 + m2 R^4 (1-cos(psi/R)) ]
		in the linear regime the potential term is simply
		m2 R^2 psi^2/2
		the non0-linear term can be written as
		m2 R^4 (2 sin^2(psi/2R))
		which suggests to generalise
		m2 R^2/2 psi^2 -> m2 R^2/2 (4R^2 sin^2(psi/2R))
		and compute the FT not of psi, but of 2R sin(psi/2R)

		However, we will use the binning method SPECTRUM_VV
		which requires to multiply by the mass-prefactor
		as well, i.e.

		*/

			if ( (nrt & (NRUN_S | NRUN_CS)) && (mass2 > 0.0))
			{
				LogMsg(VERB_HIGH,"[nRun] Vnl loop (Axion)") ;
				Float R2   = (Float) Rscale*2;
				Float iR2  = 1/R2;
					#pragma omp parallel for schedule(static)
					for (size_t iz=0; iz < Lz; iz++) {
						size_t zo = Ly*(Ly+2)*iz ;
						size_t zi = Ly*Ly*iz ;
						for (size_t iy=0; iy < Ly; iy++) {
							size_t yo = (Ly+2)*iy ;
							size_t yi = Ly*iy ;
							for (size_t ix=0; ix < Ly; ix++) {
								size_t odx = ix + yo + zo; size_t idx = ix + yi + zi;

								switch(mask){
									default:
									case SPMASK_FLAT:
												m2[odx] = R2*std::sin(m[idx] * iR2);
											break;
									case SPMASK_AXIT:
												m2[odx] = R2*std::sin(m[idx] * iR2)*0.5*(1-std::tanh(5*(m2[idx])-1)) ;
											break;
									case SPMASK_AXIT2:
											if (strdaa[idx] & STRING_MASK)
													m2[odx] = 0 ;
											else
													m2[odx] = R2*std::sin(m[idx] * iR2);
											break;
								} //end mask
						}}} // end last volume loop

				myPlan.run(FFT_FWD);

				if (spec)
					fillBins<Float,  SPECTRUM_VNL, true> ();
				else
					fillBins<Float,  SPECTRUM_VNL, false>();
			}

			field->setM2     (M2_DIRTY);
		}
		break;

		case	FIELD_WKB:
		default:
		LogError ("Error: Field not supported");
		return;
		break;
	}
}


void	SpecBin::nSRun	() {
	// saxion spectrum

	binK.assign(powMax, 0.);
	binG.assign(powMax, 0.);
	binV.assign(powMax, 0.);

	if	(field->Folded())
	{
		Folder	munge(field);
		munge(UNFOLD_ALL);
	}

	switch (fType) {
		default:
		case FIELD_AXION:
		case FIELD_AXION_MOD:
		case FIELD_WKB:
				LogError ("Error: Wrong field called to numberSaxionSpectrum: no Saxion information!!");
		return;

		case	FIELD_SAXION:
		{
			// nPts = Lx*Ly*Lz;
			switch (fPrec) {
				case FIELD_SINGLE:
				{
					std::complex<float> *ma     = static_cast<std::complex<float>*>(field->mStart());
					std::complex<float> *va     = static_cast<std::complex<float>*>(field->vCpu());
					float *m2sa                 = static_cast<float *>(field->m2Cpu());
					float *m2sax                = static_cast<float *>(field->m2Cpu()) + (Ly+2)*Ly*Lz;

					#pragma omp parallel for schedule(static)
					for (size_t iz=0; iz < Lz; iz++) {
						size_t zo = Ly*(Ly+2)*iz ;
						size_t zi = Ly*Ly*iz ;
						for (size_t iy=0; iy < Ly; iy++) {
							size_t yo = (Ly+2)*iy ;
							size_t yi = Ly*iy ;
							for (size_t ix=0; ix < Ly; ix++) {
								size_t odx = ix + yo + zo;
								size_t idx = ix + yi + zi;

								float modu = std::abs(ma[idx]-zaskaf);
								// float modu = std::abs(ma[idx]);
								m2sa[odx] = std::real(va[idx]*modu/(ma[idx]-zaskaf)) ;
								// m2sa[odx] = real(va[idx]*modu/(ma[idx])) ;
								m2sax[odx] = modu - Rscale ;
							}
						}
					}
					// LogOut("[debug] 0 and 0 %f %f \n", m2sa[0], m2sax[0]);
					// LogOut("[debug] -1 and -1 %f %f \n", m2sa[(Ly+2)*Ly*(Lz-1)+(Ly+2)*(Ly-1)+Ly-1], m2sax[(Ly+2)*Ly*(Lz-1)+(Ly+2)*(Ly-1)+Ly-1]);
				}
				break;

				case FIELD_DOUBLE:
				{
					std::complex<double> *ma     = static_cast<std::complex<double>*>(field->mStart());
					std::complex<double> *va     = static_cast<std::complex<double>*>(field->vCpu());
					double *m2sa            = static_cast<double *>(field->m2Cpu());
					double *m2sax            = static_cast<double *>(field->m2Cpu())+(Ly+2)*Ly*Lz;

					#pragma omp parallel for schedule(static)
					for (size_t iz=0; iz < Lz; iz++) {
						size_t zo = Ly*(Ly+2)*iz ;
						size_t zi = Ly*Ly*iz ;
						for (size_t iy=0; iy < Ly; iy++) {
							size_t yo = (Ly+2)*iy ;
							size_t yi = Ly*iy ;
							for (size_t ix=0; ix < Ly; ix++) {
								size_t odx = ix + yo + zo;
								size_t idx = ix + yi + zi;

								double modu = std::abs(ma[idx]-zaska);
								// double modu = abs(ma[idx]);
								m2sa[odx] = std::real(va[idx]*modu/(ma[idx]-zaska)) ;
								// m2sa[odx] = real(va[idx]*modu/(ma[idx])) ;
								m2sax[odx] = modu - Rscale ;
							}
						}
					}
				}
				break;

				default:
					LogError ("Wrong precision");
					break;
			}//End prec switch


			// r2c FFT in m2

			auto &myPlan = AxionFFT::fetchPlan("pSpecAx");
			myPlan.run(FFT_FWD);


			if (fPrec == FIELD_SINGLE) {
				if (spec)
					fillBins<float,  SPECTRUM_KS, true> ();
				else
					fillBins<float,  SPECTRUM_KS, false>();
			} else {
				if (spec)
					fillBins<double, SPECTRUM_KS, true> ();
				else
					fillBins<double, SPECTRUM_KS, false>();
			}



			// Copy m2aux -> m2
			size_t dataTotalSize = field->Precision()*field->eSize();
			char *mA  = static_cast<char *>(field->m2Cpu());
			char *mAh = static_cast<char *>(field->m2half());
			memmove	(mA, mAh, dataTotalSize);

			// float *m2sa                 = static_cast<float *>(field->m2Cpu());
			// float *m2sax                = static_cast<float *>(field->m2Cpu()) + (Ly+2)*Ly*Lz;
			// LogOut("[debug] 0 and 0 %f %f \n", m2sa[0], m2sax[0]);
			// LogOut("[debug] -1 and -1 %f %f \n", m2sa[(Ly+2)*Ly*(Lz-1)+(Ly+2)*(Ly-1)+Ly-1], m2sax[(Ly+2)*Ly*(Lz-1)+(Ly+2)*(Ly-1)+Ly-1]);

			myPlan.run(FFT_FWD);

			if (fPrec == FIELD_SINGLE) {
				if (spec)
					fillBins<float,  SPECTRUM_GVS, true> ();
				else
					fillBins<float,  SPECTRUM_GVS, false>();
			} else {
				if (spec)
					fillBins<double, SPECTRUM_GVS, true> ();
				else
					fillBins<double, SPECTRUM_GVS, false>();
			}

			field->setM2     (M2_DIRTY);
		}
		break;
	}
}

void	SpecBin::nmodRun	() {

	if (fPrec == FIELD_SINGLE) {
		if (spec)
			fillBins<float,  SPECTRUM_NN, true> ();
		else
			fillBins<float,  SPECTRUM_NN, false>();
	} else {
		if (spec)
			fillBins<double, SPECTRUM_NN, true> ();
		else
			fillBins<double, SPECTRUM_NN, false>();
	}
}

void	SpecBin::avekRun	() {

	if (fPrec == FIELD_SINGLE) {
		if (spec)
			fillBins<float,  SPECTRUM_AK, true> ();
		else
			fillBins<float,  SPECTRUM_AK, false>();
	} else {
		if (spec)
			fillBins<double, SPECTRUM_AK, true> ();
		else
			fillBins<double, SPECTRUM_AK, false>();
	}
}


/* SMOOTHER */
/* SMOOTHS THE M2 FIELD WITH A GIVEN FILTER IN FOURIER SPACE */

void	SpecBin::smoothFourier	(double length, FilterIndex filter) {
	LogMsg (VERB_NORMAL, "[sF] Called smoothFourier");LogFlush();

	char *m2  = chosechar(PFIELD_M2);
	char *m2h = chosechar(PFIELD_M2H);
	auto &myPlan = AxionFFT::fetchPlan("pSpecAx");

	bool ok = false;
	if ((field->m2Status() == M2_ENERGY_FFT) && (field->m2hStatus() == M2_ENERGY_FFT))
	{
		LogMsg (VERB_PARANOID, "[sF] FFTs already there");LogFlush();
		ok = true;
	}
	else if ((field->m2Status() == M2_ENERGY_FFT) && (field->m2hStatus() != M2_ENERGY_FFT) && !ok)
	{
		LogMsg (VERB_PARANOID, "[sF] FFT in M2, copy to M2h");LogFlush();
		memmove(m2h, m2, dataTotalSize);
		ok = true;
		field->setM2h(M2_ENERGY_FFT);
	}
	else if ((field->m2Status() != M2_ENERGY_FFT) && (field->m2hStatus() == M2_ENERGY_FFT) && !ok)
	{
		LogMsg (VERB_PARANOID, "[sF] FFT in M2h, copy to M2");LogFlush();
		memmove(m2, m2h, dataTotalSize);
		ok = true;
		field->setM2(M2_ENERGY_FFT);
	}
	else if ((field->m2Status() == M2_ENERGY) && !ok)
	{
		LogMsg (VERB_PARANOID, "[sF] Energy in M2. Pad, FFT and copy to M2");LogFlush();
		pad(PFIELD_M2,PFIELD_M2);
		myPlan.run(FFT_FWD);
		memmove(m2h, m2, dataTotalSize);
		ok = true;
		field->setM2(M2_ENERGY_FFT);
		field->setM2h(M2_ENERGY_FFT);
	}
	else if ((field->m2hStatus() == M2_ENERGY) && !ok)
	{
		LogMsg (VERB_PARANOID, "[sF] Energy in M2h. Pad to M2, FFT and copy to M2h");LogFlush();
		pad(PFIELD_M2H,PFIELD_M2);
		myPlan.run(FFT_FWD);
		memmove(m2h, m2, dataTotalSize);
		ok = true;
		field->setM2(M2_ENERGY_FFT);
		field->setM2h(M2_ENERGY_FFT);
	}
	else
	{
		LogError ("Error: smoothFourier called with no energy/FFT in m2/m2h what?!");
		return ;
		/* */
	}

	LogMsg (VERB_PARANOID, "[sF] Start filter");LogFlush();
	switch(fPrec)
	{
		case FIELD_SINGLE:
			switch(filter){
					case FILTER_GAUSS:
						smoothFourier<float,FILTER_GAUSS>(length);
						break;
					case FILTER_TOPHAT:
						smoothFourier<float,FILTER_TOPHAT>(length);
						break;
					case FILTER_SHARPK:
						smoothFourier<float,FILTER_SHARPK>(length);
						break;
			}
			break;
		case FIELD_DOUBLE:
			switch(filter){
					case FILTER_GAUSS:
						smoothFourier<double,FILTER_GAUSS>(length);
						break;
					case FILTER_TOPHAT:
						smoothFourier<double,FILTER_TOPHAT>(length);
						break;
					case FILTER_SHARPK:
						smoothFourier<double,FILTER_SHARPK>(length);
						break;
			}
			break;
	}
	myPlan.run(FFT_BCK);
	field->setM2(M2_ENERGY_SMOOTH);
	LogMsg(VERB_HIGH,"[sF] Filtered!");
}


	/* smooths in Fourier space */


template<typename Float, FilterIndex filter>
void	SpecBin::smoothFourier	(double length) {

	using cFloat = std::complex<Float>;

	Float normn3 = 1./ ((double) field->TotalSize());
	Float k0R  = k0*length;
	Float pref ;

	switch(filter){
		case FILTER_GAUSS:
			LogMsg(VERB_HIGH,"[sF] Filter Gaussian - R = %.2e [ADM]",length);
			pref = 0.5*k0R*k0R;
			break;
		case FILTER_TOPHAT:
			LogMsg(VERB_HIGH,"[sF] Filter TopHat - R = %.2e [ADM]",length);
			normn3 *= 3.;
			break;
		case FILTER_SHARPK:
			pref = k0R*k0R;
			LogMsg(VERB_HIGH,"[sF] Filter SharpK - R = %.2e [ADM]",length);
			break;
	}

	LogMsg(VERB_PARANOID,"k0R %e pref %e normn3 %e",k0R,pref,normn3);

	// Float *m2 = static_cast<Float*> (field->m2Cpu());

	#pragma omp parallel for schedule(static)
		for (size_t idx=0; idx<nModeshc; idx++) {
			/* hC modes are stored as Idx = kx + kz*hLx * ky*hLx*Tz
				with hLx = Ly/2 + 1 here called Lx by Alex ... */
			int kz = idx/Lx; 				//aux kz + ky*Tz
			int kx = idx - kz*Lx;
			int ky = kz/Tz;
			kz -= ky*Tz;
			ky += zBase;	// For MPI, transposition makes the Y-dimension smaller

			// if (kx > static_cast<int>(hLy)) kx -= static_cast<int>(Ly); // by definition never true
			if (ky > static_cast<int>(hLy)) ky -= static_cast<int>(Ly);
			if (kz > static_cast<int>(hTz)) kz -= static_cast<int>(Tz);
			Float k2    = (Float) kx*kx + ky*ky + kz*kz;

			switch(filter){
				case FILTER_GAUSS:
					static_cast<cFloat *>(field->m2Cpu())[idx] *= exp(-pref*k2) * normn3;
					// k2 = exp(-pref*k2) * normn3;
					// m2[2*idx]   *= k2;
					// m2[2*idx+1] *= k2;
					break;
				case FILTER_TOPHAT:
					k2 = k0R*sqrt(k2);
					// k2 = ((std::sin(k2)-k2*std::cos(k2)) * normn3);
					// m2[2*idx]   *= k2;
					// m2[2*idx+1] *= k2;
					if (k2 > 0)
						static_cast<cFloat *>(field->m2Cpu())[idx] *= (normn3 * (sin(k2)-k2*cos(k2))/(k2*k2*k2) );
					else
						static_cast<cFloat *>(field->m2Cpu())[idx] *= (normn3/3.);

					break;
				case FILTER_SHARPK:
					if (pref*k2 > 1)
						static_cast<cFloat *>(field->m2Cpu())[idx] = ((cFloat) (0.,0.));
					else
						static_cast<cFloat *>(field->m2Cpu())[idx] *= normn3;
						// m2[2*idx]   = 0;
						// m2[2*idx+1] = 0;


					break;
			}
		}
		LogMsg(VERB_HIGH,"[sF] Done!");
}


	/* pads in general */


int	SpecBin::pad	(PadIndex origin, PadIndex dest){

	/* recall that Lx = Lx/2 +1, we use Ly for the would-be Lx*/
	size_t padVol   = (Ly+2)*Ly*Lz;
	size_t fromSVol = Ly*Ly*(Lz+field->getNg());
	size_t eVol     = Ly*Ly*(Lz+2*field->getNg());

	LogMsg (VERB_HIGH,  "[pad] from %d to %d",origin, dest);
	LogMsg (VERB_PARANOID, "[pad] padVol %d fromSVol %d eVol %d [-LxLxLz]",2*Ly*Lz, Ly*Ly*field->getNg(), 2*Ly*Ly*field->getNg());

	if ((dest & PFIELD_START) && (fromSVol > padVol) ){
		LogMsg(VERB_HIGH,"[pad] Padding unsafe, (Lz+Ng)Ly < (Ly+2)(Lz) and padded date overruns FIELD");
		LogError("[pad] Padding unsafe, perhaps ok! perhaps not...");
	} else {
		LogMsg (VERB_HIGH, "[pad] Padding safe (Lz+Ng)Ly>(Ly+2)(Lz)");
	}

	char *oR = chosechar(origin);
	char *dE = chosechar(dest);

	/* pad = expand > start from the end
	write before read if : end of write (sl*pl+dl) is before beggining of write (sl*dl) */
	if ( origin - dest == 0 ){
		/* sl*pl+dl > sl*dl always > SAFE*/
		for (int sl = LyLz-1; sl>=0; sl--) {
			memmove (dE + sl*pl, oR + sl*dl, dl);
		}
		return 1;
	}
	else if ( origin - dest == 1){
		/* From start to 0
		-write before read if : end of write (sl*pl+dl) is before beggining of write (sl*dl)
		-offset of Ng*Lx*dl = Ghost slices
		sl*pl + dl > Ng*Lx*dl + sl*dl
		sl*2  + dl > Ng*Lx*dl
		sl > (Ng*Lx-1)*dl/2
		this is always larger than the ghost region
		where there is nothing to overwrite > SAFE
		Can also be done from 0 onwards I think */
		for (int sl = LyLz-1; sl>=0; sl--) {
			memmove (dE + sl*pl, oR + sl*dl, dl);
		}
		return 1;
	} /* from  0 to Start : from the end */
	else if ( dest - origin == 1){
		for (int sl = LyLz-1; sl>=0; sl--)
			memmove (dE + sl*pl, oR + sl*dl, dl);
		return 1;
	}

	/* if padding into a different field > parallel */
	if (std::abs(origin - dest)>1)
	{
		#pragma omp parallel for schedule(static)
		for (size_t sl=0; sl < LyLz; sl++) {
			memmove	(dE + sl*pl, oR + sl*dl, dl);
		}
		return 1 ;
	}
	LogError("[Pad] could not pad anything!");
	return 0;
}


	/* unpads in general */


int	SpecBin::unpad	(PadIndex origin, PadIndex dest)
{
	/* recall that Lx = Lx/2 +1, we use Ly for the would-be Lx*/
	size_t padVol   = (Ly+2)*Ly*Lz;
	size_t fromSVol = Ly*Ly*(Lz+field->getNg());
	size_t eVol     = Ly*Ly*(Lz+2*field->getNg());

	LogMsg (VERB_HIGH,  "[upad] from %d to %d",origin, dest);
	LogMsg (VERB_PARANOID, "[upad] padVol %d fromSVol %d eVol %d [-LxLxLz]",2*Ly*Lz, Ly*Ly*field->getNg(), 2*Ly*Ly*field->getNg());

	if ((origin & PFIELD_START) && (fromSVol > padVol) ){
		LogMsg(VERB_HIGH,"[upad] Unpadding unsafe, (Lz+Ng)Ly < (Ly+2)(Lz) and padded date overruns FIELD");
		LogError("[upad] Padding unsafe, perhaps ok! perhaps not...");
	} else {
		LogMsg (VERB_HIGH, "[upad] Unpadding safe (Lz+Ng)Ly>(Ly+2)(Lz)");
	}

	char *oR = chosechar(origin);
	char *dE = chosechar(dest);

	/* unpad is compressing > start from beggining */
	if ( origin - dest == 0 ){
		/* In place
				> start from the beggining
					possible parallel by slice chunks?
						possible issue only if origin has elements outside natural range */
		LogMsg (VERB_PARANOID, "[upad] In place");
		LogMsg (VERB_PARANOID, "[upad] LyLz %d dl %d pl %d ",LyLz,dl,pl);
		for (size_t sl=0; sl < LyLz; sl++) {
			memmove	(dE + sl*dl, oR + sl*pl, dl);
		}
		return 1;
	}
	else if ( origin - dest == 1 ){
		/* From Start to 0 */
		LogMsg (VERB_PARANOID, "[upad] From S to 0");
		LogMsg (VERB_PARANOID, "[upad] LyLz %d dl %d pl %d ",LyLz,dl,pl);
		for (size_t sl=0; sl < LyLz; sl++) {
			memmove	(dE + sl*dl, oR + sl*pl, dl);
		}
		return 1;
	}
	else if ( dest - origin == 1){
		/* from  0 to Start (NLxLx): in place + all translation
			    0   - (Lx+2) Lx Lz; line starts at (Lx+2)*l , l in (0,LxLz)
			Lx*Lx*N - Lx Lx (Lz+N); line starts at Lx*Lx*N + Lx*l
			sure overlap unpad in place and memmove all
		*/
		LogMsg (VERB_PARANOID, "[upad] Forward! First in place, second move all");
		LogMsg (VERB_PARANOID, "[upad] dataBareSize %d",dataBareSize);
		for (int sl = 1; sl < LyLz; sl++)
			memmove (oR + sl*dl, oR + sl*pl, dl);
		memmove (dE, oR, dataBareSize);
		return 1;
	}

	/* if unpadding into a different field > parallel */
	if (std::abs(origin - dest)>1)
	{
		#pragma omp parallel for schedule(static)
		for (size_t sl=0; sl < LyLz; sl++) {
			memmove	(dE + sl*dl, oR + sl*pl, dl);
		}
		return 1 ;
	}
	LogError("[unPad] could not unpad anything!");
	return 0;
}

char*	SpecBin::chosechar	(PadIndex start){
switch (start){
	case PFIELD_M:
		return static_cast<char*>(field->mCpu());
		break;
	case PFIELD_MS:
		return static_cast<char*>(field->mStart());
		break;
	case PFIELD_V:
		return static_cast<char*>(field->vCpu());
		break;
	case PFIELD_VS:
		return static_cast<char*>(field->vStart());
		break;
	case PFIELD_M2:
		return static_cast<char*>(field->m2Cpu());
		break;
	case PFIELD_M2S:
		return static_cast<char*>(field->m2Start());
		break;
	case PFIELD_M2H:
		return static_cast<char*>(field->m2half());
		break;
	case PFIELD_M2HS:
		return static_cast<char*>(field->m2hStart());
		break;
	}
}




/* FILTERFFT */
/* multiplies the Fourier modes of m2 by a Gaussian filter; obsolete?? */

template<typename Float>
void	SpecBin::filterFFT	(double neigh) {
	LogMsg (VERB_NORMAL, "[FilterFFT] Called filterFFT with M2 status %d", field->m2Status());

	using cFloat = std::complex<Float>;

	//const int mIdx = commThreads();

	//prefactor is (2 pi^2 neigh^2/N^2)
	//double prefac = 2.0*M_PI*M_PI*neigh*neigh/field->Surf() ;
	double prefac = 0.5*M_PI*M_PI*neigh*neigh/field->Surf() ;

	LogMsg (VERB_NORMAL, "[FilterFFT] filterBins with %.3f neighbours, prefa = %f", neigh, prefac);

	//std::complex<Float> * m2ft = static_cast<std::complex<Float>*>(axion->m2Cpu());

	const double normn3 = field->TotalSize();

	#pragma omp parallel
	#pragma omp for schedule(static)
		for (size_t idx=0; idx<nModeshc; idx++) {

			int kz = idx/Lx;
			int kx = idx - kz*Lx;
			int ky = kz/Tz;

			//JAVI ASSUMES THAT THE FFTS FOR SPECTRA ARE ALWAYS OF r2c type
			//and thus always in reduced format with half+1 of the elements in x

			kz -= ky*Tz;
			ky += zBase;	// For MPI, transposition makes the Y-dimension smaller

			// if (kx > static_cast<int>(hLy)) kx -= static_cast<int>(Ly); // by definition never true
			if (ky > static_cast<int>(hLy)) ky -= static_cast<int>(Ly);
			if (kz > static_cast<int>(hTz)) kz -= static_cast<int>(Tz);

			double k2    = kx*kx + ky*ky + kz*kz;
			static_cast<cFloat *>(field->m2Cpu())[idx] *= (exp(-prefac*k2)/normn3);
		}
}

void	SpecBin::filter (size_t neigh) {

	Profiler &prof = getProfiler(PROF_SPEC);
	prof.start();

	LogMsg (VERB_NORMAL, "[Filter] Called filter with M2 status %d", field->m2Status());
	// FFT of contrast bin is assumed in m2 (with ghost bytes)
	// filter with a Gaussian over n neighbours
	// exp(- ksigma^2/2)
	// k = 2Pi* n/ L    [n labels mode number]
	// sigma = delta* number of neighbours
	// ksigma^2/2 = 2 pi^2 [n^2]/N^2 * (neighbour)^2


	switch (fPrec) {
		case	FIELD_SINGLE:
				filterFFT<float> ( (double) neigh);
			break;

		case	FIELD_DOUBLE:
				filterFFT<double> ( (double) neigh);
			break;

		default:
			LogError ("[Filter] Wrong precision");
			break;
	}

	LogMsg (VERB_NORMAL, "[Filter] FFT m2 inplace -> ");
	auto &myPlan = AxionFFT::fetchPlan("pSpecAx");
	myPlan.run(FFT_BCK);
	LogMsg (VERB_NORMAL, "[Filter] -> filtered density map in m2!");

	LogMsg (VERB_NORMAL, "[Filter] reducing map [cherrypicking]");
	// reducemap consists on reorganising items of the filtered density map
	// we outputthem as a bin to use print bin
	// or as a reduced density map ?


	size_t seta = (size_t) neigh ;
	size_t newNx = Ly/seta ;
	size_t newNz = Lz/seta ;

	LogMsg (VERB_NORMAL, "[Filter] seta %d newNx %d newNz %d [cherrypicking]",seta, newNx, newNz);

	switch (fPrec) {
		case	FIELD_SINGLE:
		{
			float *mCon = static_cast<float *>(static_cast<void*>(field->m2Cpu()));
			//size_t topa = newNx*newNx*newNz ;

			for (size_t iz=0; iz < newNz; iz++) {
				size_t laz = Ly*(Ly+2)*iz*seta ;
				size_t sz = newNx*newNx*iz ;
				for (size_t iy=0; iy < newNx; iy++) {
					size_t lay = (Ly+2)*iy*seta ;
					size_t sy = newNx*iy ;
					for (size_t ix=0; ix < newNx; ix++) {
						size_t idx = ix + sy + sz ;
						size_t odx = ix*seta + lay + laz ;
						mCon[idx] = mCon[odx] ;
					}
				}
			}
		}
		break;

		case	FIELD_DOUBLE:
		{
			double *mCon = static_cast<double *>(static_cast<void*>(field->m2Cpu()));

			for (size_t iz=0; iz < newNz; iz++) {
				size_t laz = Ly*(Ly+2)*iz*seta ;
				size_t sz = newNx*newNx*iz ;
				for (size_t iy=0; iy < newNx; iy++) {
					size_t lay = (Ly+2)*iy*seta ;
					size_t sy = newNx*iy ;
					for (size_t ix=0; ix < newNx; ix++) {
						size_t idx = ix + sy + sz ;
						size_t odx = ix*seta + lay + laz ;
						mCon[idx] = mCon[odx] ;
					}
				}
			}
		}
		break;

	}
	field->setM2(M2_ENERGY_RED);

	prof.stop();
	prof.add(std::string("filter"), 0.0, 0.0);
}







/* masker functions
	they use a radius mask to
	mask a certain quantity in a certain "mask" way
	and intend a certain output out*/

void	SpecBin::masker	(double radius_mask, SpectrumMaskType mask, StatusM2 out, bool l_cummask){

	switch (mask)
	{
		case SPMASK_FLAT :
		case SPMASK_SAXI :
			LogError("[masker] This masks is automatic, nothing will be done");
			return;
		break;

		case SPMASK_VIL2 :
		switch (fPrec)
			{
				case FIELD_SINGLE :
				SpecBin::masker<float,SPMASK_VIL2> (radius_mask, out, l_cummask);
				break;

				case FIELD_DOUBLE :
				SpecBin::masker<double,SPMASK_VIL2> (radius_mask, out, l_cummask);
				break;

				default :
				LogError("[masker] precision not reconised.");
				break;
			}
		break;

		case SPMASK_VIL :
		switch (fPrec)
			{
				case FIELD_SINGLE :
				SpecBin::masker<float,SPMASK_VIL> (radius_mask, out, l_cummask);
				break;

				case FIELD_DOUBLE :
				SpecBin::masker<double,SPMASK_VIL> (radius_mask, out, l_cummask);
				break;

				default :
				LogError("[masker] precision not reconised.");
				break;
			}
		break;

		case SPMASK_AXIT :
		switch (fPrec)
			{
				case FIELD_SINGLE :
				SpecBin::masker<float,SPMASK_AXIT> (radius_mask, out, l_cummask);
				break;

				case FIELD_DOUBLE :
				SpecBin::masker<double,SPMASK_AXIT> (radius_mask, out, l_cummask);
				break;

				default :
				LogError("[masker] precision not reconised.");
				break;
			}
		break;

		case SPMASK_AXIT2 :
		switch (fPrec)
			{
				case FIELD_SINGLE :
				SpecBin::masker<float,SPMASK_AXIT2> (radius_mask, out, l_cummask);
				break;

				case FIELD_DOUBLE :
				SpecBin::masker<double,SPMASK_AXIT2> (radius_mask, out, l_cummask);
				break;

				default :
				LogError("[masker] precision not reconised.");
				break;
			}
		break;

		case SPMASK_GAUS :
			switch (fPrec)
			{
				case FIELD_SINGLE :
				SpecBin::masker<float,SPMASK_GAUS> (radius_mask, out, l_cummask);
				break;

				case FIELD_DOUBLE :
				SpecBin::masker<double,SPMASK_GAUS> (radius_mask, out, l_cummask);
				break;

				default :
				LogError("[masker] precision not reconised.");
				break;
			}
		break;

		case SPMASK_DIFF :
			switch (fPrec)
			{
				case FIELD_SINGLE :
				SpecBin::masker<float,SPMASK_DIFF> (radius_mask, out, l_cummask);
				break;

				case FIELD_DOUBLE :
				SpecBin::masker<double,SPMASK_DIFF> (radius_mask, out, l_cummask);
				break;

				default :
				LogError("[masker] precision not reconised.");
				break;
			}
		break;

		case SPMASK_BALL :
			switch (fPrec)
			{
				case FIELD_SINGLE :
				SpecBin::masker<float,SPMASK_BALL> (radius_mask, out, l_cummask);
				break;

				case FIELD_DOUBLE :
				SpecBin::masker<double,SPMASK_BALL> (radius_mask, out, l_cummask);
				break;

				default :
				LogError("[masker] precision not reconised.");
				break;
			}
		break;

		case SPMASK_REDO :
		default:
			switch (fPrec)
			{
				case FIELD_SINGLE :
				SpecBin::masker<float,SPMASK_REDO> (radius_mask, out, l_cummask);
				break;

				case FIELD_DOUBLE :
				SpecBin::masker<double,SPMASK_REDO> (radius_mask, out, l_cummask);
				break;

				default :
				LogError("[masker] precision not reconised.");
				break;
			}
		break;
	}
}


/* MASKER */



template<typename Float, SpectrumMaskType mask>
void	SpecBin::masker	(double radius_mask, StatusM2 out, bool l_cummask) {

	Profiler &prof = getProfiler(PROF_SPEC);
	char PROFLABEL[256];
	sprintf(PROFLABEL, "mask %d", mask);

	if (field->LowMem()){
			LogMsg(VERB_NORMAL,"[masker] masker called in lowmem! exit!\n");
			return;
	}

	switch (mask)
	{
		case SPMASK_REDO:
			LogMsg(VERB_NORMAL,"[masker] masker REDO (Field = %d, sDStatus= %d)\n",field->Field(),field->sDStatus());
			if (field->Field() != FIELD_SAXION || !(field->sDStatus() & SD_MAP)){
					LogMsg(VERB_NORMAL,"[masker] masker called without string map! (Field = %d, sDStatus= %d)\n",field->Field(),field->sDStatus());
					return;
			}
		break;

		case SPMASK_GAUS:
			LogMsg(VERB_NORMAL,"[masker] masker GAUSS (Field = %d, sDStatus= %d)",field->Field(),field->sDStatus());
			if (field->Field() != FIELD_SAXION || !(field->sDStatus() & SD_MAP)){
					LogMsg(VERB_NORMAL,"[masker] masker called without string map! (Field = %d, sDStatus= %d)\n",field->Field(),field->sDStatus());
					return;
			}
		break;

		case SPMASK_DIFF:
			LogMsg(VERB_NORMAL,"[masker] masker DIFF (Field = %d, sDStatus= %d)\n",field->Field(),field->sDStatus());
			if (field->Field() != FIELD_SAXION || !(field->sDStatus() & SD_MAP)){
					LogMsg(VERB_NORMAL,"[masker] masker called without string map! (Field = %d, sDStatus= %d)\n",field->Field(),field->sDStatus());
					return;
			}
		break;

		case SPMASK_VIL:
			LogMsg(VERB_NORMAL,"[masker] masker Vil \n");
		break;

		case SPMASK_VIL2:
			LogMsg(VERB_NORMAL,"[masker] masker Vil2 \n");
		break;

		case SPMASK_AXIT:
		case SPMASK_AXIT2:
		case SPMASK_AXITV:
			// LogMsg(VERB_NORMAL,"[masker] Axiton M2status %d ! \n",field->m2Status());
			// if (field->sDStatus() == SD_AXITONMASK){
			// 	LogMsg(VERB_NORMAL,"[masker] masker called but mask already in place! exit!\n");
			// }
			if ( !(field->m2Status() == M2_ENERGY) ){
				if ( !(field->m2hStatus() == M2_ENERGY)){
					LogError("[masker] Axiton masker called without energy in M2 or in M2h (status %d and %d) ! exit!\n",field->m2Status(),field->m2hStatus());
					return;
					} else {
					LogMsg(VERB_HIGH,"[masker] Axiton masker called;  Energy in M2h (status %d)!\n", field->m2hStatus());
					}
			}
		break;

		case SPMASK_BALL:
			LogMsg(VERB_NORMAL,"[masker] Ball test");
		break;

		default:
			LogMsg(VERB_NORMAL,"\n[masker] Mask not available! exit!\n");
			return;
		break;
	}

	LogMsg(VERB_NORMAL,"[masker] cummask %d",l_cummask);LogFlush();

	if	(field->Folded())
	{
		Folder	munge(field);
		munge(UNFOLD_ALL);
	}


	switch (fType) {

		case	FIELD_SAXION:
		{
			/* If a plaquete is pierced by string marks the 4 vertices

				The only complication is that if the main vertex is in Lz-1
				and is a YZ or XZ plaquette we need to mark points in the next MPI
				rank. For this reason we MPI_send the last slice forward from z=Lz-1
				to m2half and then use it to mark the first slice of the next rank,
				i.e. slice z=0.

				For large slices we need to split the MPI communication into chunks
				less than INT_MAX.
				This is an unnecesary overhead of sizeof(single,double)*8
				because we only need 1 bit per point (mark? true or false).
				Thus we can most likely use just one slice of bool in 1 MPI_Send

			*/


			Float RR = (Float) Rscale;
			std::complex<Float> zaskaF((Float) zaskar, 0.);
			char *strdaa = static_cast<char *>(static_cast<void *>(field->sData()));
			Float *m2F                 = static_cast<Float *>(field->m2Cpu());
			/* set to 0 one ghost region in m2half*/
			Float *m2hF                = static_cast<Float *>(field->m2half());
			Float *m2sF                = static_cast<Float *>(field->m2Start());
			/* auxiliary; Last volume in m2 where size fits, size is 2*(Lz+2Ng)LxLy*/
			Float *m2aF                = static_cast<Float *>(field->m2Cpu()) + (field->eSize()*2-field->Size()) ;
			std::complex<Float>* mc         = static_cast<std::complex<Float> *>(field->mStart());

			/* For ghosting purposes */
			char *m2hb                = static_cast<char *>(field->m2half());
			char *m2h1b               = static_cast<char *>(field->m2half()) + field->Length()*(2+field->Length());

			/* For padding uses */
			char *m2C  = static_cast<char *>(field->m2Cpu());
			char *m2sC = static_cast<char *>(field->m2Start());
			char *m2hC = static_cast<char *>(field->m2half());
			char *m2aC = static_cast<char *>(static_cast<void *>(m2aF)) ;

			/* Clears completely m2 in SAXION MODE; only half in AXION MODE */
			memset (m2C, 0, field->eSize()*field->DataSize());

			/* Build mask (Vi,Vi2) or pre-mask */
			{
			/* Create the first mask in m2 with sData or rho field,
			in REDO, GAUSS, DIFF, BALL,
			1st slice Lz-1 to be send, then the rest */
			bool mpi = false;
			void *empty;
			size_t sliceBytes = (Ly*(Ly+2));

				prof.start();
			if (mask & (SPMASK_REDO|SPMASK_GAUS|SPMASK_DIFF|SPMASK_BALL))
			{
				size_t iz  = Lz-1;
				size_t zo  = Ly2Ly*iz ;
				size_t zoM = Ly2Ly*(iz+1) ;
				size_t zi  = LyLy*iz ;

				#pragma omp parallel for schedule(static)
				for (size_t iy=0; iy < Ly; iy++) {
					size_t yo  = (Ly+2)*iy ;
					size_t yoM = (Ly+2)*((iy+1)%Ly) ;
					size_t yi  = Ly*iy ;
					for (size_t ix=0; ix < Ly; ix++) {
						size_t odx = ix + yo + zo;
						size_t idx = ix + yi + zi;

						/* removes the mask if present
						watch out updates outside thread region of the loop
						(different y) */
						strdaa[idx] = strdaa[idx] & STRING_DEFECT;
								if ( (strdaa[idx] & STRING_ONLY) != 0)
								{
									#pragma omp atomic write
									m2F[odx] = 1;
									if (strdaa[idx] & (STRING_XY))
									{
										#pragma omp atomic write
										m2F[((ix + 1) % Ly) + yo  + zo] = 1;
										#pragma omp atomic write
										m2F[ix              + yoM + zo] = 1;
										#pragma omp atomic write
										m2F[((ix + 1) % Ly) + yoM + zo] = 1;
									}
									if (strdaa[idx] & (STRING_YZ))
									{
										#pragma omp atomic write
										m2F[ix + yoM + zo] = 1;
										#pragma omp atomic write
										m2h1b[ix + yo ] = 1;
										#pragma omp atomic write
										m2h1b[ix + yoM] = 1;
									}
									if (strdaa[idx] & (STRING_ZX))
									{
										#pragma omp atomic write
										m2F[((ix + 1) % Ly) + yo + zo]  = 1;
										#pragma omp atomic write
										m2h1b[ix              + yo] = 1;
										#pragma omp atomic write
										m2h1b[((ix + 1) % Ly) + yo] = 1;
									}
								}
					}    // end loop x
				}      // end loop z

			field->sendGeneral(COMM_SDRV, sliceBytes, MPI_BYTE, empty, empty, static_cast<void *>(m2h1b), static_cast<void *>(m2hb));
			mpi = true;
			}
			size_t iiz_start = mpi ? 1 : 0;

			/* We invert the order of the loops because usually Ly > Lz
			we dont collapse to avoid recalculating y0,y0M,yi,iz,zo,zoM,zi*/
			#pragma omp parallel for schedule(static)
			for (size_t iy=0; iy < Ly; iy++) {
				size_t yo  = (Ly+2)*iy ;
				size_t yoM = (Ly+2)*((iy+1)%Ly) ;
				size_t yi  = Ly*iy ;
				for (size_t iiz=iiz_start; iiz < Lz; iiz++) {
					size_t iz  = Lz-1-iiz;
					size_t zo  = Ly2Ly*iz ;
					size_t zoM = Ly2Ly*(iz+1) ;
					size_t zi  = LyLy*iz ;
					for (size_t ix=0; ix < Ly; ix++) {
						size_t odx = ix + yo + zo;
						size_t idx = ix + yi + zi;

						switch(mask){
							case SPMASK_VIL:
									m2F[odx] = std::abs(mc[idx]-zaskaF)/RR;
									break;
							case SPMASK_VIL2:
									m2F[odx] = pow(std::abs(mc[idx]-zaskaF)/RR,2);
									break;

							case SPMASK_REDO:
							case SPMASK_GAUS:
							case SPMASK_DIFF:
							case SPMASK_BALL:
							/* removes the mask if present
							needs atomic updates if other threads can be at work
							just when updating different values of y */
							strdaa[idx] = strdaa[idx] & STRING_DEFECT;
									if ( (strdaa[idx] & STRING_ONLY) != 0)
									{
										#pragma omp atomic write
										m2F[odx] = 1;
										if (strdaa[idx] & (STRING_XY))
										{
											#pragma omp atomic write
											m2F[((ix + 1) % Ly) + yo  + zo] = 1;
											#pragma omp atomic write
											m2F[ix              + yoM + zo] = 1;
											#pragma omp atomic write
											m2F[((ix + 1) % Ly) + yoM + zo] = 1;
										}
										if (strdaa[idx] & (STRING_YZ))
										{
											#pragma omp atomic write
											m2F[ix + yoM + zo ] = 1;
											#pragma omp atomic write
											m2F[ix + yo  + zoM] = 1;
											#pragma omp atomic write
											m2F[ix + yoM + zoM] = 1;
										}
										if (strdaa[idx] & (STRING_ZX))
										{
											#pragma omp atomic write
											m2F[((ix + 1) % Ly) + yo + zo ] = 1;
											#pragma omp atomic write
											m2F[ix              + yo + zoM] = 1;
											#pragma omp atomic write
											m2F[((ix + 1) % Ly) + yo + zoM] = 1;
										}
									}
							break;
						}  //end mask
					}    // end loop x
				}      // end loop z
			}        // end loop y

			prof.stop();
			{
				char LABEL[256];
				sprintf(LABEL, "M0.PREP (%s)", PROFLABEL);
				prof.add(std::string(LABEL), 0.0, 0.0);
			}

			/* For string based masks:
			1 - receive info from previous rank
			2 - Fuse ghost and local info inthe 1st surface */
			if (mpi)
			{
					prof.start();
				field->sendGeneral(COMM_WAIT, sliceBytes, MPI_BYTE, empty, empty, static_cast<void *>(m2h1b), static_cast<void *>(m2hb));
					prof.stop();
				char LABEL[256];
					sprintf(LABEL, "M1.WAIT (%s)", PROFLABEL);
						prof.add(std::string(LABEL), 0.0, 0.0);

				// int myRank = commRank();
				// int own = 0;
				// int news = 0;
				// int overlap = 0;
				// LogMsg(VERB_PARANOID,"[sp] Fusing ghosts between ranks");
				// #pragma omp parallel for reduction(+:own,news,overlap) collapse(2)
					prof.start();
				#pragma omp parallel for collapse(2)
					for (size_t iy=0; iy < Ly; iy++) {
						for (size_t ix=0; ix < Ly; ix++) {
							size_t odx = ix+(Ly+2)*iy;
							// if (m2F[odx] > 0.5)
							// 	own++;
							if (m2hb[odx] > 0){
								// news++;
								// if (m2F[odx] > 0.5)
								// 		overlap++;
								m2F[odx] = 1 ;
							}
				}}
				prof.stop();
			char LABEL2[256];
				sprintf(LABEL2, "M2.FUSE (%s)", PROFLABEL);
					prof.add(std::string(LABEL2), 0.0, 0.0);

				// LogMsg(VERB_PARANOID,"[sp] rank %d own %d new %d overlap %d",myRank,own,news,overlap);
			}

			/* if maskball we set the mask also in strdaa
			ideally we could do this from the beggining because we will mask m2
			after the call to mask ball; the overhead is not huge */
			if (mask == SPMASK_BALL)
			{
				size_t mm0 = 0;
				size_t mm1 = 0;
				size_t mm2 = 0;
				LogMsg(VERB_PARANOID,"[MB] correct strdata (STRING_DEFECT %d) ",STRING_DEFECT) ;LogFlush();
				#pragma omp parallel default(shared) reduction(+:mm0,mm1,mm2)
				{
					size_t X[3];
					size_t idx;
					#pragma omp for schedule(static)
					for (size_t odx=0; odx < Ly2Ly*Lz; odx++)
					{
						if (m2F[odx] > 0)
						{
							mm0++;
							indexXeon::idx2VecPad (odx, X, Ly);
							idx = indexXeon::vec2Idx(X, Ly);
							// LogOut("count %d odx %lu idx %lu X %lu %lu %lu st %d m2 %.2f\n",mm0, odx, idx, X[0],X[1],X[2],strdaa[idx],m2F[odx]);
							/* */
							if (X[0] < Ly){
								if (idx > V){
									LogError("strdaa overflow! odx %lu idx %lu X %lu %lu %lu\n",odx, idx, X[0],X[1],X[2]);
								}

								/* we mark with string-only those newly marked */
								if ( (strdaa[idx] & STRING_ONLY) != 0 )
									mm1++;
								else
								{
									strdaa[idx] |= STRING_ONLY;      // we mark all strings XY, YZ, XZ because is none of them
									mm2++;
								}
							}
						}
					}
				}
				LogMsg(VERB_PARANOID,"[MB] masked points (@M2) %lu (un/corrected in stdata) %lu/%lu",mm0,mm1,mm2) ;LogFlush();
			}

		} // MPI defs are contained here


			/* Load FFT for Wtilde and REDO,GAUS */
			auto &myPlan = AxionFFT::fetchPlan("pSpecAx");

			/* At this point all pre-masks are m2 (padded)
			Hereforth the DIFF, REDO and GAUS diverge;
			Vil requires nothing  */

			switch(mask){
				case SPMASK_VIL:
				case SPMASK_VIL2:
				break;

				case SPMASK_DIFF:
					{
							prof.start();
						/* unpad m2 in place */
						for (size_t sl=1; sl<LyLz; sl++) {
							size_t	oOff = sl*dl;
							size_t	fOff = sl*pl;
							memmove	(m2C+oOff, m2C+fOff, dl);
						}
							prof.stop();
								prof.add(std::string("memmove"), 0.0, 0.0);

						/* produce a first mask*/

						#pragma omp parallel for schedule(static)
						for (size_t idx=0; idx < field->Size(); idx++) {
							if ( m2F[idx] > 0.5 ){
								strdaa[idx] |= STRING_MASK ; // mask stored M=1-W
								}
						}

						/* shift away from ghost zone*/
						memmove	(m2sC, m2C, dataBareSize);

						/* Smooth with fixed mask */
						const Float OneSixth = 1./6.;

						/* how many iterations? FIX ME*/
						size_t iter = radius_mask*radius_mask;
						size_t radius_mask_size_t = (size_t) radius_mask;

						/* Size of the ghost region in Float (assume this funciton is called in Saxion mode)*/
						size_t GR = field->getNg()*LyLy*(field->DataSize()/field->Precision());

						/* I need char pointers to the last slice, backghost, both Float and std::complex_Float */

						char *m2lsC  = static_cast<char *>(field->m2Start()) + field->Surf()*(field->Depth()-1)*field->Precision();
						char *m2bgC  = static_cast<char *>(field->m2Start()) + field->Size()*field->Precision();
						char *m2lscC = static_cast<char *>(field->m2Start()) + field->Surf()*(field->Depth()-field->getNg())*field->DataSize();
						char *m2bgcC = static_cast<char *>(field->m2BackGhost());

						size_t SurfTotalSize = LyLy*field->Precision();

						/* Smoothing iteration loop */
						for (size_t it=0; it<iter+1; it++)
						{
							/* exchange ghosts Float using the Complex-based function */
									// copy last slice into the last std::complex slice (will be sent)
									// the first slice is already at position m2Start
									memcpy	(m2lscC, m2lsC, SurfTotalSize);
									field->exchangeGhosts(FIELD_M2);
									// move the slice received at the std::complex backghost into the Float-backghost
									memmove	(m2bgC, m2bgcC, SurfTotalSize);
									// move the slice received at the frontghost into the Float-Frontghost (defined by m2Start)
									memmove	(m2C+(GR-LyLy)*field->Precision(), m2C, SurfTotalSize);

							/* Smooth and copy to m2_aux m2aF
							(m2h does not work because is precisely the Float-backghost) */
							#pragma omp parallel for default(shared) schedule(static)
							for (size_t idx=GR; idx< GR+V; idx++)
							{
								size_t idxu = idx-GR;

								/* String plaquette vertices do not get diffused away*/
								if( strdaa[idxu] & STRING_MASK )
								{
									// LogOut("X %d Y%d Z %d SD %d m2 %f %f\n",X[0],X[1],X[2], (uint8_t) strdaa[idxu],m2F[idx],m2hF[idxu]);
									m2aF[idxu] = 1;
								}
								else {
									size_t X[3], O[4];
									indexXeon::idx2VecNeigh (idx, X, O, Ly);
									if (it < radius_mask_size_t)
									{ /* First round mask propagates the mask at 1 point per iteration*/
										if ((m2F[idx+LyLy] > 0) | (m2F[idx-LyLy] > 0) | (m2F[O[0]] > 0) | (m2F[O[1]] > 0) | (m2F[O[2]] > 0) | (m2F[O[3]] > 0))
										{
											/* we can test removing this */
											strdaa[idxu] |= STRING_MASK ;
											m2aF[idxu] = 1;
										} else {
											m2aF[idxu] = 0;
										}
									}
									else
									{ /* Next iterations smooth the mask */
										//copies the smoothed configuration into the auxiliary volume
										m2aF[idxu]   = OneSixth*(m2F[O[0]] + m2F[O[1]] + m2F[O[2]] + m2F[O[3]] + m2F[idx+LyLy] + m2F[idx-LyLy]);
									}
								} // end if not mask

							}
							LogMsg(VERB_PARANOID,"[masker] smoothing end iteration %d",it);
							/* bring smoothed configuration back to m2 */
							memmove (m2sC, m2aC, dataBareSize);

						} // end iteration loop

					/* copy to mh2 and pad to m2*/
					memcpy (m2hC, m2sC, dataBareSize);
					for (size_t sl=1; sl<LyLz; sl++) {
						size_t	oOff = sl*dl;
						size_t	fOff = sl*pl;
						memmove	(m2C+fOff, m2hC+oOff, dl);
					}

					} // end internal SPMASK_DIFF
				break;

				case SPMASK_REDO:
				case SPMASK_GAUS:
					{
						/* Fourier transform */
						// r2c FFT in m2
						prof.start();
						myPlan.run(FFT_FWD);

						/* Filter */
						double rsend = radius_mask;
						if (ng0calib > 0.0)
							rsend *= ng0calib;

						switch (fPrec) {
							case	FIELD_SINGLE:
									filterFFT<float> (rsend);
								break;

							case	FIELD_DOUBLE:
									filterFFT<double> (rsend);
								break;

							default:
								LogError ("Wrong precision");
								break;
						}

						/* iFFT gives desired antimask (unnormalised) */
						myPlan.run(FFT_BCK);

							/* save a padded copy in m2h */
						memcpy	(m2hC, m2C, dataTotalSize);

						prof.stop();
							prof.add(std::string("M3.RED"), 0.0, 0.0);

					} // end internal CASE REDO and GAUS case switch
				break;

				case SPMASK_BALL:
					{
						/* masks a ball of radius rm around STRING_DEFECT
						   and labels it STRING_MASK*/
							prof.start();
						maskball	(radius_mask, STRING_ONLY, STRING_MASK);
							prof.stop();
								prof.add(std::string("M3.maskball"), 0.0, 0.0);
					} // end internal CASE REDO and GAUS case switch
				break;
			} // end switch

// DEBUG
// int imask=0;
// #pragma omp parallel for reduction(+:imask)
// for (size_t idx = 0 ; idx < V; idx++){
// 	if (strdaa[idx] & STRING_MASK)
// 		imask++;}
// LogMsg(VERB_PARANOID,"DEBUG imask %d",imask);

			/* Produce the final mask in m2
			and unpadded in m2h
			remember that REDO, GAUS are padded
			diff already unpadded in m2h
			ball is only in strdaa */

			/* Constants for mask_REDO, DIFF */
			Float maskcut = 0.5;

			if (mask & (SPMASK_REDO | SPMASK_GAUS))
			{
				if (ng0calib < 0.0)
				{
				maskcut = (Float) std::abs(radius_mask);
					if (radius_mask < 4)
						maskcut = (0.42772052 -0.05299264*maskcut)/(maskcut*maskcut);
					else
						maskcut = (0.22619714 -0.00363601*maskcut)/(maskcut*maskcut);
					if (radius_mask > 8)
						maskcut = (0.21)/(radius_mask*radius_mask);
				} else {
					Float rsend = ng0calib* radius_mask;
					maskcut = 2.5*4/(2*M_PI*rsend*rsend)*std::exp(-2./(ng0calib*ng0calib));
				}

			}
			/* Constant for mask_GAUS */
			Float cc = 2.*M_PI*radius_mask*radius_mask ; // Uncalibrated

			LogMsg(VERB_HIGH,"maskcut %f",maskcut);
				prof.start();

			size_t V = field->Size() ;
			size_t Ly2 = (Ly+2);
			#pragma omp parallel for schedule(static)
			for (size_t idx=0; idx < field->Size(); idx++) {
				size_t X[3];
				indexXeon::idx2Vec (idx, X, Ly);
				size_t oidx = X[2]*Ly2Ly + X[1]*(Ly+2) + X[0];

				switch(mask){
					case SPMASK_VIL:
					case SPMASK_VIL2:
							/* Here only unpad to m2h*/
							m2hF[idx] = m2F[oidx];
							//m2F[oidx]  = 1. - m2F[oidx];
							break;
					case SPMASK_REDO:
							/* Top hat mask */
							if ( m2F[oidx] > maskcut ) {
								strdaa[idx] |= STRING_MASK ;
								m2F[oidx] = 0; // Axion ANTIMASK
								m2hF[idx] = 0; // Axion MASK
							}
							else {
								m2F[oidx] = 1;
								m2hF[idx] = 1;
							}
							break;
						case SPMASK_BALL:
								if ( strdaa[idx] &  STRING_MASK ) {
									m2F[oidx] = 0; // Axion ANTIMASK
									m2hF[idx] = 0; // Axion MASK
								}
								else {
									m2F[oidx] = 1;
									m2hF[idx] = 1;
								}
								break;
						case SPMASK_DIFF:
								/* Diffused mask */
								if ( m2hF[idx] > maskcut )     //0.5 // remember m2hF is already unpadded->needs idx
									strdaa[idx] |= STRING_MASK ; // Note that we store a CORE string mask here defined by the 0.5
								m2hF[idx] = 1 - m2hF[idx];     // Axion mask unpadded
								m2F[oidx] = m2hF[idx];         // Axion mask padded
								break;
					case SPMASK_GAUS:
							/* Exponentially suppressed mask */
							if ( m2F[oidx] > maskcut ){
								strdaa[idx] |= STRING_MASK ;
								// Linear version
								m2F[oidx] = 0;
								m2hF[idx] = 0;
							} else {
								m2F[oidx] = 1 - m2F[oidx]/maskcut;
								m2hF[idx] = m2F[oidx];
							}
							// exponential version
							// m2F[oidx] = exp(-cc*m2F[oidx]);
							// m2hF[idx] = m2F[oidx];
							break;
					default:
					LogError("[sp] Unknown mask!");
					break;
				} //end internal switch
			} // end idx loop
			prof.stop();
			char LABEL3[256];
				sprintf(LABEL3, "M5.END (%s)", PROFLABEL);
					prof.add(std::string(LABEL3), 0.0, 0.0);

			/* masks in m2 already padded
			 but we have saved unpadded maps in m2h */

// DEBUG
// imask=0;
// #pragma omp parallel for reduction(+:imask)
// for (size_t idx = 0 ; idx < V; idx++){
// 	if (strdaa[idx] & STRING_MASK)
// 		imask++;}
// LogMsg(VERB_PARANOID,"DEBUG 2 imask %d",imask);


			/* Calculate the FFT of the mask */
				prof.start();
			myPlan.run(FFT_FWD);
				prof.stop();
					prof.add(std::string("M6.WFFT"), 0.0, 0.0);


			binP.assign(powMax, 0.);
			switch (fPrec) {
				case	FIELD_SINGLE:
					if (spec)
						fillBins<float,  SPECTRUM_P, true> ();
					else
						fillBins<float,  SPECTRUM_P, false>();
					break;

				case	FIELD_DOUBLE:
					if (spec)
						fillBins<double,  SPECTRUM_P, true> ();
					else
						fillBins<double,  SPECTRUM_P, false>();
					break;

				default:
					LogError ("Wrong precision");
					break;
			}

			// remove unnecessary factor 1/2 in fillBins
			for(size_t i=0; i<powMax; i++) binP.at(i) *= 2.;


		/* Depends on out */
		prof.start();
		switch (out) {
					default:
					case M2_MASK:
						/* To print maps from m2*/
						memcpy	(m2C, m2hC , dataBareSize);
						field->setM2(M2_MASK);
						break;
					case M2_ANTIMASK:
						size_t V = field->Size() ;
						#pragma omp parallel for schedule(static)
						for (size_t idx=0; idx < V; idx++)
							m2F[idx] = 1-m2hF[idx];
						field->setM2(M2_ANTIMASK);
						break;
		}
			prof.stop();
				prof.add(std::string("M7.OUT"), 0.0, 0.0);

		if (mask & (SPMASK_DIFF | SPMASK_GAUS | SPMASK_REDO)){
			field->setSD(SD_MAPMASK);
		}


		}
		break; //case saxion ends


		case FIELD_AXION:
		{

			// thinking about axiton finder
			// energy is required in m2
			char *strdaa = static_cast<char *>(static_cast<void *>(field->sData()));
			Float *m2sa                 = static_cast<Float *>(field->m2Cpu());
			Float *m2sax                = static_cast<Float *>(field->m2half());
			Float *mm                   = static_cast<Float *>(field->mStart());
			char *mA  = static_cast<char *>(field->m2Cpu());
			char *mAS = static_cast<char *>(field->m2half());


			// makes a copy of the energy density of axions in m2_2
			if (field->m2Status() == M2_ENERGY){
				LogMsg(VERB_NORMAL,"[masker] Copy ENERGy to M2h");
				memmove (mAS, mA, dataBareSize);
				field->setM2h(M2_ENERGY);
				}
			if ((field->m2Status() != M2_ENERGY) && (field->m2hStatus() == M2_ENERGY)){
				LogMsg(VERB_NORMAL,"[masker] Copy ENERGY to M2");
				memmove (mA, mAS, dataBareSize);
				field->setM2(M2_ENERGY);
			}

			// threshold of the energy density [energy ]
			Float RRRRRR = (Float) *field->RV();
			Float ethres = (Float) 2*field->AxionMassSq();
			if (mask == SPMASK_AXITV)
				ethres = (Float) 0.5*M_PI*M_PI*field->AxionMassSq();

			Float iR     = 1/RRRRRR;
			Float tthres = std::sqrt(12/ethres)*iR/field->Delta();
			if( tthres > 3)
				tthres = 3;

			// l_cummask, Axit2 mask if previous point has this label
			StringType ST_CS = STRING_XY_POSITIVE;
			if (l_cummask) {
				ST_CS = STRING_WALL;
				LogMsg(VERB_NORMAL,"[masker axion] Cumulative masking (%d, WALL=64,...)",ST_CS);
				}
				// counter for masked points

			int mp = 0;
			int mt = 0;
			int ms = 0;
			#pragma omp parallel for schedule(static) reduction(+:mp,mt,ms)
			for (size_t idx=0; idx < field->Size(); idx++) {
				size_t X[3];
				indexXeon::idx2Vec (idx, X, Ly);
				size_t oidx = X[2]*Ly2Ly + X[1]*(Ly+2) + X[0];

				switch(mask){
					case SPMASK_AXIT:
					/* The last condition allows to mask if the point was masked before,
					i.e. not refresing the mask! */
					if (strdaa[idx] & ST_CS)
						mt++;
					if( (m2sax[idx] > ethres) || ( std::abs(mm[idx]*iR) > tthres) || (strdaa[idx] & ST_CS)){
					// if( (m2sax[idx] > ethres) ){
						mp++;
						strdaa[idx] = STRING_WALL;
						m2sa[oidx] = 0.0;
					} else {
						ms++;
						m2sa[oidx] = m2sax[idx];
						strdaa[idx] = STRING_NOTHING;
					}
					break;

					case SPMASK_AXIT2:
						/* The last condition allows to mask if the point was masked before,
						i.e. not refresing the mask! */
						if (strdaa[idx] & ST_CS)
							mt++;
						if( (m2sax[idx] > ethres) || ( std::abs(mm[idx]*iR) > tthres) || (strdaa[idx] & ST_CS) ){
							mp++;
							strdaa[idx] = STRING_WALL;
							m2sa[oidx] = 1.0;
						} else {
							ms++;
							m2sa[oidx] = 0.0;
							strdaa[idx] = STRING_NOTHING;
						}
					break;

					case SPMASK_AXITV:
						m2sa[oidx] = m2sax[idx]*0.5*(1-std::tanh(5*(m2sax[idx]/ethres - 1)));
					break;

				} //end mask switch
			}    // end loop idx

			int mp_g = 0;
			int mt_g = 0;
			int ms_g = 0;
			MPI_Allreduce(&mp, &mp_g, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&mt, &mt_g, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&ms, &ms_g, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
			LogMsg(VERB_NORMAL,"[masker axion] %d points masked, %d from previous mask, %d unmasked" ,mp_g,mt_g,ms_g);

			switch (mask)
			{
				case SPMASK_AXIT:
				{
							maskball	(radius_mask, STRING_WALL, STRING_MASK);

							mp = 0;
							#pragma omp parallel for schedule(static) reduction(+:mp)
							for (size_t idx=0; idx < field->Size(); idx++) {
								if (strdaa[idx] & STRING_MASK){
									mp++;
									size_t X[3];
									indexXeon::idx2Vec (idx, X, Ly);
									size_t oidx = X[2]*Ly2Ly + X[1]*(Ly+2) + X[0];
									m2sa[oidx] = 0;
								}
							}    // end loop idx
							field->setM2(M2_ENERGY_AXI);

							mp_g = 0;
							MPI_Allreduce(&mp, &mp_g, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
							LogMsg(VERB_NORMAL,"[masker axion] %d points masked after maskball (all ranks) ",mp_g);

							LogMsg(VERB_PARANOID,"[masker] computing PS");
							auto &myPlan = AxionFFT::fetchPlan("pSpecAx");
							myPlan.run(FFT_FWD);
							field->setM2(M2_ENERGY_MASK_AXI_FFT);

							LogMsg(VERB_PARANOID,"[masker] filling masked eA spectrum bins (masked)");
							binP.assign(powMax, 0.);
							switch (fPrec) {
								case	FIELD_SINGLE:
									if (spec)
										fillBins<float,  SPECTRUM_P, true> ();
									else
										fillBins<float,  SPECTRUM_P, false>();
									break;

								case	FIELD_DOUBLE:
									if (spec)
										fillBins<double,  SPECTRUM_P, true> ();
									else
										fillBins<double,  SPECTRUM_P, false>();
									break;

								default:
									LogError ("Wrong precision");
									break;
							}
				}
				break;

				case SPMASK_AXIT2:
				{
							/* Fourier transform */
							// r2c FFT in m2
							auto &myPlan = AxionFFT::fetchPlan("pSpecAx");
							myPlan.run(FFT_FWD);

							field->setM2(M2_MASK_AXI2_FFT);
							field->setSD(SD_AXITONMASK);

							/* bin the axion energy spectrum */
							LogMsg(VERB_NORMAL,"[masker] AXIT2 filter");
							/* Filter */
							switch (fPrec) {
								case	FIELD_SINGLE:
										filterFFT<float> (radius_mask);
									break;

								case	FIELD_DOUBLE:
										filterFFT<double> (radius_mask);
									break;

								default:
									LogError ("Wrong precision");
									break;
							}

							/* iFFT */
							myPlan.run(FFT_BCK);
							field->setM2(M2_MASK);

							/* Make plots if needed */

							/* Generate mask */
									/* */
									Float rsend = ng0calib* radius_mask;
									// Float rsend = radius_mask;
									Float maskcut = 2.5*4/(2*M_PI*rsend*rsend)*std::exp(-2./(ng0calib*ng0calib));


									int mp = 0;
									/* I apply the cut to the unpadded */
									#pragma omp parallel for schedule(static)
									for (size_t idx=0; idx < V; idx++) {
										size_t X[3];
										indexXeon::idx2Vec (idx, X, Ly);
										size_t oidx = X[2]*Ly2Ly + X[1]*(Ly+2) + X[0];

										if ( m2sa[oidx] > maskcut ) {
											mp++;
											strdaa[idx] |= STRING_MASK ;
											m2sa[oidx] = 0.0;
										} else {
											// use this if you want to output mask
											// strdaa[idx] = 0 ;
											// use this if you interested in the psp of the masked field directly
											m2sa[oidx] = m2sax[idx];
										}
									}
							field->setM2(M2_ENERGY_AXI);
							int mp_g = 0;
							MPI_Allreduce(&mp, &mp_g, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
							LogMsg(VERB_NORMAL,"[masker axion] %d points masked after smoothing",mp_g);

							/* Calculate the FFT of the masked field */
							myPlan.run(FFT_FWD);
							field->setM2(M2_ENERGY_MASK_AXI_FFT);

							binP.assign(powMax, 0.);
							switch (fPrec) {
								case	FIELD_SINGLE:
									if (spec)
										fillBins<float,  SPECTRUM_P, true> ();
									else
										fillBins<float,  SPECTRUM_P, false>();
									break;

								case	FIELD_DOUBLE:
									if (spec)
										fillBins<double,  SPECTRUM_P, true> ();
									else
										fillBins<double,  SPECTRUM_P, false>();
									break;

								default:
									LogError ("Wrong precision");
									break;
							}

				}
				break;

				case SPMASK_AXITV:
				{

							field->setM2(M2_ENERGY_AXI);

							LogMsg(VERB_PARANOID,"[masker] computing PS");
							auto &myPlan = AxionFFT::fetchPlan("pSpecAx");
							myPlan.run(FFT_FWD);
							field->setM2(M2_ENERGY_MASK_AXI_FFT);

							LogMsg(VERB_PARANOID,"[masker] filling masked eA spectrum bins (masked)");
							binP.assign(powMax, 0.);
							switch (fPrec) {
								case	FIELD_SINGLE:
									if (spec)
										fillBins<float,  SPECTRUM_P, true> ();
									else
										fillBins<float,  SPECTRUM_P, false>();
									break;

								case	FIELD_DOUBLE:
									if (spec)
										fillBins<double,  SPECTRUM_P, true> ();
									else
										fillBins<double,  SPECTRUM_P, false>();
									break;

								default:
									LogError ("Wrong precision");
									break;
							}
				}
				break;
				default:
				LogError("[masker axion] Error: Axion mode but no axiton mask!!");
				break;
		} // end case mask

		/* Depends on out */
		switch (out) {
					default:
					case M2_ENERGY:
						memcpy (mA, mAS, dataBareSize);
						field->setM2(M2_ENERGY);
						break;

					case M2_MASK:
					#pragma omp parallel for schedule(static)
					for (size_t idx=0; idx < V; idx++)
						if (strdaa[idx] & STRING_MASK)
							m2sa[idx] = 0;
						else
							m2sa[idx] = 1;
					field->setM2(M2_MASK);
						break;

					case M2_ANTIMASK:
					#pragma omp parallel for schedule(static)
					for (size_t idx=0; idx < V; idx++)
						if (strdaa[idx] & STRING_MASK)
							m2sa[idx] = 1;
						else
							m2sa[idx] = 0;
					field->setM2(M2_ANTIMASK);
						break;
		}

		}
		break ; //ends case axion

		default:
		LogError("[masker] Error: Masker template called with no saxion mode!");
		break ;
	} // end case saxion-axion

}	// end MASKER






/* build correction matrices */

void	SpecBin::matrixbuilder() {
	switch (fPrec)
	{
		case FIELD_SINGLE :
		SpecBin::matrixbuilder<float>();
		break;

		case FIELD_DOUBLE :
		SpecBin::matrixbuilder<double>();
		break;

		default :
		LogError("[Spectrum matrixbuilder] precision not recognised.");
		break;
	}
}

template<typename Float>
void	SpecBin::matrixbuilder() {

	//if (field->sDStatus() != SD_STDWMAP){
	//		LogOut("[matrixbuilder] matrixbuilder called without string map! exit!\n");
	//		return;
	//}
	if (field->LowMem()){
			LogOut("[matrixbuilder] matrixbuilder called in lowmem! exit!\n");
			return;
	}

	// calculate phase space density (stored in binNN), which will be used below
	// this only has to be done once in the simulation > can we do it only once?
	if (spec)
		fillBins<Float,  SPECTRUM_NN, true> ();
	else
		fillBins<Float,  SPECTRUM_NN, false>();

	// extend powmax such that it becomes a multiple of the number of MPI partitions.
	size_t powMaxPad = powMax/commSize()+1;
	size_t iBase = powMaxPad*commRank();
	double vol = field->BckGnd()->PhysSize()*field->BckGnd()->PhysSize()*field->BckGnd()->PhysSize();
	double norm = 1./vol;
	double coeJ = vol/(8.*M_PI);

	//As an alternative way, a new vector is used as a send buffer
	std::vector<double>	sbuf;
	sbuf.resize(powMaxPad*powMax);
	sbuf.assign(powMaxPad*powMax,0);

	switch (fType) {
		case	FIELD_SAXION:
		{
			//double *m2sa = static_cast<double *>(field->m2Cpu());
			// split i direction to MPI processes
			// resulting matrix M_ij is of the form (powMaxPad*Nrank x powMax)
			// the exccess part in i should be cut later.
			#pragma omp parallel for schedule(static)
			for (size_t i=0; i<powMaxPad; i++) {
				size_t is = iBase + i;
				for (size_t j=0; j<powMax; j++) {
					size_t indM = i*powMax+j;
					//m2sa[indM] = 0;
					for (size_t k=0; k<powMax; k++) {
						double J = 0;
						if (k==0) {
							if (j==0) {
								J = (is==0)?vol:0;
							} else {
								J = (is==j)?vol/binNN.at(is):0;
							}
						} else {
							if (j==0) {
								J = (is==k)?vol/binNN.at(k):0;
							} else {
								int diffkj = static_cast<int>(j) - static_cast<int>(k);
								if (is==0) {
									J = (j==k)?vol/binNN.at(j):0;
								} else if (is>=std::abs(diffkj) && is<=j+k && is < powMax) {
									J = coeJ/(is*j*k);
								} else {
									J = 0;
								}
							}
						}
						sbuf.at(indM) += norm*binP.at(k)*J;
						//m2sa[indM] += norm*binP.at(k)*J;
					}
				}
			}

			void * buf = field->m2Cpu();
			size_t charlengh = powMaxPad*powMax*sizeof(double);
			//MPI_Allgather(buf, charlengh, MPI_CHAR, buf, charlengh, MPI_CHAR, MPI_COMM_WORLD);
			// MPI_Allgather(static_cast<void *>(sbuf.data()[0]), powMaxPad*powMax, MPI_DOUBLE, buf, charlengh, MPI_CHAR, MPI_COMM_WORLD);
			//or simply use MPI_Gather ?
			MPI_Gather(static_cast<void *>(&sbuf.data()[0]), charlengh, MPI_CHAR, buf, charlengh, MPI_CHAR, 0, MPI_COMM_WORLD);
		}
		break; //case saxion ends

		default:
		LogError("[matrixbuilder] Error: matrixbuilder template called with no saxion mode!");
		break ;
	}
}

/* The following function just calculate power spectrum of |W|^2 and store it in binP. */

void	SpecBin::wRun	(SpectrumMaskType mask){

	switch (mask)
	{
		case SPMASK_FLAT :
			LogError("[Spectrum wRun] Error: we don't need the power spectrum of W in FLAT masking mode.");
			break;

		case SPMASK_VIL :
			switch (fPrec)
			{
				case FIELD_SINGLE :
				SpecBin::wRun<float,SPMASK_VIL> ();
				break;

				case FIELD_DOUBLE :
				SpecBin::wRun<double,SPMASK_VIL> ();
				break;

				default :
				LogError("[Spectrum wRun] precision not reconised.");
				break;
			}
			break;

		case SPMASK_VIL2 :
			switch (fPrec)
				{
					case FIELD_SINGLE :
					SpecBin::wRun<float,SPMASK_VIL2> ();
					break;

					case FIELD_DOUBLE :
					SpecBin::wRun<double,SPMASK_VIL2> ();
					break;

					default :
					LogError("[Spectrum wRun] precision not reconised.");
					break;
				}
			break;

		case SPMASK_SAXI :
			LogError("[Spectrum wRun] Error: we don't need the power spectrum of W in SAXI mode.");
			break;

		case SPMASK_REDO :
			switch (fPrec)
				{
					case FIELD_SINGLE :
					SpecBin::wRun<float,SPMASK_REDO> ();
					break;

					case FIELD_DOUBLE :
					SpecBin::wRun<double,SPMASK_REDO> ();
					break;

					default :
					LogError("[Spectrum wRun] precision not reconised.");
					break;
				}
			break;

		default:
		LogError("[Spectrum wRun] SPMASK not recognised!");
		break;
	}
}

template<typename Float, SpectrumMaskType mask>
void	SpecBin::wRun	() {

	binP.assign(powMax, 0.);

  std::complex<Float> zaskaF((Float) zaskar, 0.);

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
			std::complex<Float> *ma     = static_cast<std::complex<Float>*>(field->mStart());
			std::complex<Float> *va     = static_cast<std::complex<Float>*>(field->vCpu());
			Float *m2sa                 = static_cast<Float *>(field->m2Cpu());
			// Float *m2sax                = static_cast<Float *>(field->m2Cpu()) + field->eSize();
			// Float *m2sax                = static_cast<Float *>(field->m2half());
			Float *sd                   = static_cast<Float *>(field->sData());

			// identify the mask function
			#pragma omp parallel for schedule(static)
			for (size_t iz=0; iz < Lz; iz++) {
				size_t zo = Ly2Ly*iz ;
				size_t zi = LyLy*iz ;
				for (size_t iy=0; iy < Ly; iy++) {
					size_t yo = (Ly+2)*iy ;
					size_t yi = Ly*iy ;
					for (size_t ix=0; ix < Ly; ix++) {
						size_t odx = ix + yo + zo;
						size_t idx = ix + yi + zi;
						//size_t ixM = ((ix + 1) % Ly) + yi + zi;
						switch(mask){
							case SPMASK_VIL:
									m2sa[odx] = std::abs(ma[idx]-zaskaF)/Rscale;
									break;
							case SPMASK_VIL2:
									m2sa[odx] = std::pow(std::abs(ma[idx]-zaskaF)/Rscale,2);
									break;
							case SPMASK_REDO:
									//assume the map of W was already stored in stringdata
									// issue!! this will not work!
									m2sa[odx] = sd[idx];
									break;
							default:
									m2sa[odx] = 1.;
									break;
						} //end mask
					}
				}
			}

			// r2c FFT in m2
			auto &myPlan = AxionFFT::fetchPlan("pSpecAx");
			myPlan.run(FFT_FWD);
			if (spec)
				fillBins<Float,  SPECTRUM_P, true> ();
			else
				fillBins<Float,  SPECTRUM_P, false>();
			// remove unnecessary factor 1/2 in fillBins
			for(size_t i=0; i<powMax; i++) binP.at(i) *= 2.;

    }
    break;

    case	FIELD_AXION_MOD:
		case	FIELD_AXION:
		LogError ("[Spectrum wRun] Error: Theta only field not supported in wRun.");
		return;
		break;

		case	FIELD_WKB:
		LogError ("[Spectrum wRun] Error: WKB field not supported in wRun.");
		return;
		break;
  }
}
