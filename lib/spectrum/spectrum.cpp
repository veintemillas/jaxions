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

	using cFloat = std::complex<Float>;

	/* The factor that will multiply the |ft|^2, taken to be L^3/(2 N^6) */
	const double norm = (field->BckGnd()->PhysSize()*field->BckGnd()->PhysSize()*field->BckGnd()->PhysSize()) /
			    (2.*(((double) field->TotalSize())*((double) field->TotalSize())));
	const int mIdx = commThreads();

	size_t	zBase = (Ly/commSize())*commRank();

	std::vector<double>	tBinK;
	std::vector<double>	tBinG;
	std::vector<double>	tBinV;
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
			break;

		case	SPECTRUM_VV:
			tBinV.resize(powMax*mIdx);
			tBinV.assign(powMax*mIdx, 0);
			break;

		default:
			tBinG.resize(powMax*mIdx);
			tBinV.resize(powMax*mIdx);
			tBinG.assign(powMax*mIdx, 0);
			tBinV.assign(powMax*mIdx, 0);
			break;
	}

	#pragma omp parallel
	{
		int  tIdx = omp_get_thread_num ();

		#pragma omp for schedule(static)
		for (size_t idx=0; idx<nPts; idx++) {
			size_t tmp = idx/Lx;
			int    kx  = idx - tmp*Lx;
			int    ky  = tmp/Tz;

			int    kz  = tmp - ((size_t) ky)*Tz;

			ky += zBase;	// For MPI, transposition makes the Y-dimension smaller

			//ASSUMES THAT THE FFTS FOR SPECTRA ARE ALWAYS OF r2c type
			//and thus always in reduced format with half+1 of the elements in x

			if (kx > static_cast<int>(hLx)) kx -= static_cast<int>(Lx);
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
					break;
				case	SPECTRUM_VV:
					tBinV.at(myBin + powMax*tIdx) += m2;
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
					break;
				/* is possible to account for the finite difference formula
				by using the folloging line for the gradients
					// tBinG.at(myBin + powMax*tIdx) += mw/cosTable2[kx];
					*/

				/* Axion mode or only Saxion*/
				case	SPECTRUM_K:
				case	SPECTRUM_KS:
				 	mw = m2/w;
					tBinK.at(myBin + powMax*tIdx) += mw;
					break;
				case	SPECTRUM_G:
				case	SPECTRUM_GS:
					mw = m2/w;
					tBinG.at(myBin + powMax*tIdx) += mw*k2;
					break;
				case	SPECTRUM_V:
					mw = m2/w;
					tBinV.at(myBin + powMax*tIdx) += mw*mass2;
					break;
				case	SPECTRUM_VS:
					mw = m2/w;
					tBinV.at(myBin + powMax*tIdx) += mw*mass2Sax;
					break;
				case	SPECTRUM_GVS:
					mw = m2/w;
					tBinG.at(myBin + powMax*tIdx) += mw*k2;
					tBinV.at(myBin + powMax*tIdx) += mw*mass2Sax;
					break;

				case	SPECTRUM_GV:
					mw = m2/w;
					tBinG.at(myBin + powMax*tIdx) += mw*k2;
					tBinV.at(myBin + powMax*tIdx) += mw*mass2;
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
						break;
					case	SPECTRUM_G:
					case	SPECTRUM_GG:
					case	SPECTRUM_GaS:
					case	SPECTRUM_GaSadd:
						binG[j] += tBinG[j + i*powMax]*norm;
						break;
					case	SPECTRUM_VV:
						binV[j] += tBinV[j + i*powMax]*norm;
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
			break;

		case	SPECTRUM_VV:
			std::copy_n(binV.begin(), powMax, tBinV.begin());
			MPI_Allreduce(tBinV.data(), binV.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			break;

		default:
			std::copy_n(binG.begin(), powMax, tBinG.begin());
			MPI_Allreduce(tBinG.data(), binG.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			std::copy_n(binV.begin(), powMax, tBinV.begin());
			MPI_Allreduce(tBinV.data(), binV.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			break;
	}
}

void	SpecBin::pRun	() {

	size_t dSize    = (size_t) (field->Precision());
	size_t dataLine = dSize*Ly;
	size_t Sm	= Ly*Lz;

	char *mA = static_cast<char *>(field->m2Cpu());

	LogMsg(VERB_NORMAL,"[pRun] Called with status field->statusM2()=%d",field->m2Status()) ;

	if ((field->m2Status() != M2_ENERGY) && (field->m2Status() != M2_ENERGY_FFT)) {
		LogError ("Power spectrum requires previous calculation of the energy. Ignoring pRun request.");
		return;
	}

	if (field->m2Status() == M2_ENERGY) {
		// contrast bin is assumed in m2 (without ghost bytes)
		// Add the f@*&#ng padding plus ghost region, no parallelization
		for (int sl=Sm-1; sl>=0; sl--) {
			auto	oOff = sl*dSize*(Ly);
			auto	fOff = sl*dSize*(Ly+2);
			memmove	(mA+fOff, mA+oOff, dataLine);
		}



		if (field->Field() == FIELD_SAXION) {
			auto &myPlan = AxionFFT::fetchPlan("pSpecSx");
			myPlan.run(FFT_FWD);
		} else {
			auto &myPlan = AxionFFT::fetchPlan("pSpecAx");
			myPlan.run(FFT_FWD);
		}
	}

	//issue?
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

void	SpecBin::nRun	(SpectrumMaskType mask){

	switch (mask)
	{
		case SPMASK_FLAT :
				switch (fPrec)
				{
					case FIELD_SINGLE :
					SpecBin::nRun<float,SPMASK_FLAT> ();
					break;

					case FIELD_DOUBLE :
					SpecBin::nRun<double,SPMASK_FLAT> ();
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
					SpecBin::nRun<float,SPMASK_VIL> ();
					break;

					case FIELD_DOUBLE :
					SpecBin::nRun<double,SPMASK_VIL> ();
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
						SpecBin::nRun<float,SPMASK_VIL2> ();
						break;

						case FIELD_DOUBLE :
						SpecBin::nRun<double,SPMASK_VIL2> ();
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
					SpecBin::nRun<float,SPMASK_SAXI> ();
					break;

					case FIELD_DOUBLE :
					SpecBin::nRun<double,SPMASK_SAXI> ();
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
					SpecBin::nRun<float,SPMASK_REDO> ();
					break;

					case FIELD_DOUBLE :
					SpecBin::nRun<double,SPMASK_REDO> ();
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
void	SpecBin::nRun	() {

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
		default:
		LogMsg(VERB_NORMAL,"nRun with self-contained mask") ;
		break;

	}


	binK.assign(powMax, 0.);
	binG.assign(powMax, 0.);
	binV.assign(powMax, 0.);

	if (mask == SPMASK_SAXI)
	{
		binV.assign(powMax, 0.);
		binPS.assign(powMax, 0.);
	}

	// using cFloat = std::complex<Float>;
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
			// JAVI PROPOSAL I think would be easy to modify propkernel Xeon to do the loops vectorised
			std::complex<Float> *ma     = static_cast<std::complex<Float>*>(field->mStart());
			std::complex<Float> *va     = static_cast<std::complex<Float>*>(field->vCpu());
			Float *m2sa                 = static_cast<Float *>(field->m2Cpu());
			// Float *m2sax                = static_cast<Float *>(field->m2Cpu()) + field->eSize();
			Float *m2sax                = static_cast<Float *>(field->m2half());
			char *strdaa = static_cast<char *>(static_cast<void *>(field->sData()));

			// optimizar!
			#pragma omp parallel for schedule(static)
			for (size_t iz=0; iz < Lz; iz++) {
				size_t zo = Ly*(Ly+2)*iz ;
				size_t zi = Ly*Ly*iz ;
				for (size_t iy=0; iy < Ly; iy++) {
					size_t yo = (Ly+2)*iy ;
					size_t yi = Ly*iy ;
					for (size_t ix=0; ix < Ly; ix++) {
						size_t odx = ix + yo + zo; size_t idx = ix + yi + zi; size_t ixM = ((ix + 1) % Ly) + yi + zi;

						switch(mask){
							case SPMASK_FLAT:
									// m2sa[odx] = Rscale*std::imag(va[idx]/(ma[idx]-zaskaf))+std::arg(ma[idx]) ;
									m2sa[odx] = Rscale*std::imag( va[idx]/(ma[idx]-zaskaF) );
									m2sax[odx] = (2*Rscale/depta)*std::imag((ma[ixM]-ma[idx])/(ma[ixM]+ma[idx]-zaskaF-zaskaF));
									break;
							case SPMASK_REDO:
									if (strdaa[idx] & STRING_MASK){
											m2sa[odx] = 0 ;
											m2sax[odx] = 0 ;
									}
									else{
											m2sa[odx] = Rscale*std::imag( va[idx]/(ma[idx]-zaskaF) );
											m2sax[odx] = (2*Rscale/depta)*std::imag((ma[ixM]-ma[idx])/(ma[ixM]+ma[idx]-zaskaF-zaskaF));
									}
									break;
							case SPMASK_VIL:
									// m2sa[odx] = std::abs(ma[idx]-zaskaf)*(std::imag(va[idx]/(ma[idx]-zaskaf))+std::arg(ma[idx])/Rscale) ;
									m2sa[odx] =       std::abs(ma[idx]-zaskaF)      *(std::imag(va[idx]/(ma[idx]-zaskaF))) ;
									m2sax[odx] = (2*std::abs(ma[idx]-zaskaF)/depta)*std::imag((ma[ixM]-ma[idx])/(ma[ixM]+ma[idx]-zaskaF-zaskaF));
									break;
							case SPMASK_VIL2:
									// m2sa[odx] = std::abs(ma[idx]-zaskaf)*(std::imag(va[idx]/(ma[idx]-zaskaf))+std::arg(ma[idx])/Rscale) ;
									m2sa[odx] =       std::pow(std::abs(ma[idx]-zaskaF),2)/Rscale*(std::imag(va[idx]/(ma[idx]-zaskaF))) ;
									m2sax[odx] = (2*std::pow(std::abs(ma[idx]-zaskaF),2)/Rscale/depta)*std::imag((ma[ixM]-ma[idx])/(ma[ixM]+ma[idx]-zaskaF-zaskaF));
									break;
							case SPMASK_SAXI:
									// m2sa[odx] = std::abs(ma[idx]-zaskaf)*(std::imag(va[idx]/(ma[idx]-zaskaf))+std::arg(ma[idx])/Rscale) ;
									m2sa[odx]  =  std::real(va[idx]) ;
									m2sax[odx] =  std::imag(va[idx]);
							break;

						} //end mask
					}
				}
			}

			// r2c FFT in m2
			auto &myPlan = AxionFFT::fetchPlan("pSpecSx");
			myPlan.run(FFT_FWD);

			// KINETIC PART
			if (spec)
				fillBins<Float,  SPECTRUM_KK, true> ();
			else
				fillBins<Float,  SPECTRUM_KK, false>();


			// GRADIENT X
			// Copy m2aux -> m2
			size_t dSize    = (size_t) (field->Precision());
			//size_t dataTotalSize = dSize*(Ly+2)*Ly*Lz;
			size_t dataTotalSize2 = dSize*Ly*Ly*(Lz+2);
			char *mA = static_cast<char *>(field->m2Cpu());
			memmove	(mA, mA+dataTotalSize2, dataTotalSize2);

			// r2c FFT in m2
			myPlan.run(FFT_FWD);

			// This inits G bin and fills GX bins but does not reduce
			// in the saxi case, we need to fill like SPECTRUM_K again
			if (mask == SPMASK_SAXI) {
				//SAXI
				std::copy_n(binK.begin(), powMax, binV.begin());
				binK.assign(powMax, 0.);
				fillBins<Float,  SPECTRUM_K, false>();
			} else {
			if (spec)
				fillBins<Float,  SPECTRUM_GaSadd, true> ();
			else
				fillBins<Float,  SPECTRUM_GaSadd, false>();
			}

			// optimizar!
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
									m2sa[odx]  = (2*Rscale/depta)*std::imag((ma[iyM]-ma[idx])/(ma[iyM]+ma[idx]-zaskaF-zaskaF));
									m2sax[odx] = (2*Rscale/depta)*std::imag((ma[izM]-ma[idx])/(ma[izM]+ma[idx]-zaskaF-zaskaF));
									break;
							case SPMASK_REDO:
									if (strdaa[idx] & STRING_MASK){
											m2sa[odx] = 0 ;
											m2sax[odx] = 0 ;
									}
									else{
										m2sa[odx]  = (2*Rscale/depta)*std::imag((ma[iyM]-ma[idx])/(ma[iyM]+ma[idx]-zaskaF-zaskaF));
										m2sax[odx] = (2*Rscale/depta)*std::imag((ma[izM]-ma[idx])/(ma[izM]+ma[idx]-zaskaF-zaskaF));
									}
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
										m2sa[odx]  =  std::real(ma[idx]) ;
										m2sax[odx] =  std::imag(ma[idx]);
										break;
						} //end mask
					}
				}
			}


			// GRADIENT Y:
			myPlan.run(FFT_FWD);

			// this adds the gradient Y bins into binG
			if (mask == SPMASK_SAXI) {
				fillBins<Float,  SPECTRUM_G, false>();
				std::copy_n(binG.begin(), powMax, binPS.begin());
			} else {
controlxyz = 1;
			if (spec)
				fillBins<Float,  SPECTRUM_GaSadd, true> ();
			else
				fillBins<Float,  SPECTRUM_GaSadd, false>();
			}

			// GRADIENT Z:
			// Copy m2aux -> m2
			memmove	(mA, mA+dataTotalSize2, dataTotalSize2);

			myPlan.run(FFT_FWD);
			// this adds the gradient Z bins into binG and reduces the final sum!

			if (mask == SPMASK_SAXI) {
					binG.assign(powMax, 0.);
					fillBins<Float,  SPECTRUM_G, false>();
					std::copy_n(binG.begin(), powMax, binPS.begin());
				} else {
controlxyz = 2;
				if (spec)
					fillBins<Float,  SPECTRUM_GaS, true> ();
				else
					fillBins<Float,  SPECTRUM_GaS, false>();
			}

				// potential!!
				// experimental!!
				//potential dependent!
				// fix

				// conformal mass square root of topological susceptibility
				// we use a factor of more because by default 1/2 is included in fillbins
				// because of the kin and grad terms
			Float mass = (Float) std::sqrt(mass2);
			Float iR   = (Float) 1/Rscale;
			Float iR2   = (Float) 1/(Rscale*Rscale);

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
				}


				// POTENTIAL:
				myPlan.run(FFT_FWD);

				// this adds the gradient Y bins into binG
				if (mask != SPMASK_SAXI) {
					if (spec)
						fillBins<Float,  SPECTRUM_VV, true> ();
					else
						fillBins<Float,  SPECTRUM_VV, false>();
				}
			} // potential

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

			size_t dataLine = field->DataSize()*Ly;
			size_t Sm	= Ly*Lz;

			// Copy m -> m2 with padding
			#pragma omp parallel for schedule(static)
			for (uint sl=0; sl<Sm; sl++) {
				auto	oOff = sl*field->DataSize()* Ly;
				auto	fOff = sl*field->DataSize()*(Ly+2);
				memcpy	(mF+fOff, mO+oOff, dataLine);
			}

			myPlan.run(FFT_FWD);

			if (spec)
				fillBins<Float,  SPECTRUM_GV, true> ();
			else
				fillBins<Float,  SPECTRUM_GV, false>();

			// Copy v -> m2 with padding
			#pragma omp parallel for schedule(static)
			for (uint sl=0; sl<Sm; sl++) {
				auto	oOff = sl*field->DataSize()* Ly;
				auto	fOff = sl*field->DataSize()*(Ly+2);
				memcpy	(mF+fOff, vO+oOff, dataLine);
			}

			myPlan.run(FFT_FWD);

			if (spec)
				fillBins<Float,  SPECTRUM_K, true> ();
			else
				fillBins<Float,  SPECTRUM_K, false>();

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
					std::complex<float> *ma     = static_cast<std::complex<float>*>(field->mCpu())  + field->Surf();
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
					std::complex<double> *ma     = static_cast<std::complex<double>*>(field->mCpu())  + field->Surf();
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

			auto &myPlan = AxionFFT::fetchPlan("pSpecSx");
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
			size_t dSize    = (size_t) (field->Precision());
			size_t dataTotalSize = dSize*(Ly+2)*Ly*Lz;
			char *mA = static_cast<char *>(field->m2Cpu());
			memmove	(mA, mA+dataTotalSize, dataTotalSize);

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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

template<typename Float>
void	SpecBin::filterFFT	(double neigh) {
	LogMsg (VERB_NORMAL, "[Filter] Called filterFFT with M2 status %d", field->m2Status());

	using cFloat = std::complex<Float>;

	//const int mIdx = commThreads();

	size_t	zBase = (Ly/commSize())*commRank();

	//prefactor is (2 pi^2 neigh^2/N^2)
	//double prefac = 2.0*M_PI*M_PI*neigh*neigh/field->Surf() ;
	double prefac = 0.5*M_PI*M_PI*neigh*neigh/field->Surf() ;

	LogMsg (VERB_NORMAL, "filterBins with %.3f neighbours, prefa = %f", neigh, prefac);

	//complex<Float> * m2ft = static_cast<complex<Float>*>(axion->m2Cpu());

	const double normn3 = field->TotalSize();

	#pragma omp parallel
	#pragma omp for schedule(static)
		for (size_t idx=0; idx<nPts; idx++) {

			int kz = idx/Lx;
			int kx = idx - kz*Lx;
			int ky = kz/Tz;

			//JAVI ASSUMES THAT THE FFTS FOR SPECTRA ARE ALWAYS OF r2c type
			//and thus always in reduced format with half+1 of the elements in x

			kz -= ky*Tz;
			ky += zBase;	// For MPI, transposition makes the Y-dimension smaller

			if (kx > static_cast<int>(hLx)) kx -= static_cast<int>(Lx);
			if (ky > static_cast<int>(hLy)) ky -= static_cast<int>(Ly);
			if (kz > static_cast<int>(hTz)) kz -= static_cast<int>(Tz);

			double k2    = kx*kx + ky*ky + kz*kz;
			static_cast<cFloat *>(field->m2Cpu())[idx] *= (exp(-prefac*k2)/normn3);
		}
}

void	SpecBin::filter (size_t neigh) {

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
}


/* masker functions*/

void	SpecBin::masker	(double radius_mask, SpectrumMaskType mask){

	switch (mask)
	{
		case SPMASK_FLAT :
		case SPMASK_VIL :
		case SPMASK_VIL2 :
		case SPMASK_SAXI :
			LogError("[masker] These masks are not yet implemented");
		break;

		case SPMASK_AXIT :
		switch (fPrec)
			{
				case FIELD_SINGLE :
				SpecBin::masker<float,SPMASK_AXIT> (radius_mask);
				break;

				case FIELD_DOUBLE :
				SpecBin::masker<double,SPMASK_AXIT> (radius_mask);
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
				SpecBin::masker<float,SPMASK_AXIT2> (radius_mask);
				break;

				case FIELD_DOUBLE :
				SpecBin::masker<double,SPMASK_AXIT2> (radius_mask);
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
				SpecBin::masker<float,SPMASK_REDO> (radius_mask);
				break;

				case FIELD_DOUBLE :
				SpecBin::masker<double,SPMASK_REDO> (radius_mask);
				break;

				default :
				LogError("[masker] precision not reconised.");
				break;
			}
		break;
	}
}



template<typename Float, SpectrumMaskType mask>
void	SpecBin::masker	(double radius_mask) {

	switch (mask)
	{
		case SPMASK_REDO:
			LogMsg(VERB_NORMAL,"[masker] masker REDO (Field = %d, sDStatus= %d)\n",field->Field(),field->sDStatus());
			if (field->Field() != FIELD_SAXION || !(field->sDStatus() & SD_MAP)){
					LogMsg(VERB_NORMAL,"[masker] masker called without string map! (Field = %d, sDStatus= %d)\n",field->Field(),field->sDStatus());
					return;
			}
		break;

		case SPMASK_AXIT:
		case SPMASK_AXIT2:
			LogMsg(VERB_NORMAL,"[masker] Axiton M2status %d ! \n",field->m2Status());
			if ( !(field->m2Status() == M2_ENERGY)){
					LogMsg(VERB_NORMAL,"[masker] Axiton masker called without energy M2status %d ! exit!\n",field->m2Status());
					return;
			}
		break;
		default:
			LogMsg(VERB_NORMAL,"[masker] Mask not available! exit!\n");
			return;
		break;
	}


	if (field->LowMem()){
			LogMsg(VERB_NORMAL,"[masker] masker called in lowmem! exit!\n");
			return;
	}

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
			Float *m2sa                 = static_cast<Float *>(field->m2Cpu());

			/* set to 0 one ghost region in m2half*/
			Float *m2sax                = static_cast<Float *>(field->m2half());
			size_t surfi = Ly*(Ly+2) ;
			// #pragma omp parallel for schedule(static)
			// for (size_t odx=0; odx < surfi; odx++) {
			// 	m2sax[odx] = 0 ;
			// }
			memset (field->m2half(), 0, surfi*field->Precision());


			/* MPI rank and position of the last slice that we will send to the next rank */
			int myRank = commRank();
			int nsplit = (int) (field->TotalDepth()/field->Depth()) ;
			static const int fwdNeig = (myRank + 1) % nsplit;
			static const int bckNeig = (myRank - 1 + nsplit) % nsplit;

			size_t voli = Ly*(Ly+2)*(Lz-1) ;
			const int ghostBytes = (Ly*(Ly+2))*(field->Precision());
			static MPI_Request 	rSendFwd, rRecvBck;
			void *sGhostFwd = static_cast<void *>(m2sa + voli);
			void *rGhostBck = static_cast<void *>(m2sax);

			// memset (field->m2Cpu(), 0, field->eSize()*field->DataSize());

			// optimizar!
			#pragma omp parallel for schedule(static)
			for (size_t iiz=0; iiz < Lz; iiz++) {
				size_t iz = Lz-1-iiz;
				size_t zo = Ly*(Ly+2)*iz ;
				size_t zoM = Ly*(Ly+2)*(iz+1) ;
				size_t zi = Ly*Ly*iz ;
				// printf("zo %lu zoM %lu zi %lu\n",zo,zoM,zi);

				for (size_t iy=0; iy < Ly; iy++) {
					size_t yo = (Ly+2)*iy ;
					size_t yoM = (Ly+2)*((iy+1)%Ly) ;
					size_t yi = Ly*iy ;
					// printf("yo %lu yoM %lu yi %lu\n",yo,yoM,yi);

					for (size_t ix=0; ix < Ly; ix++) {

						/* position in the mask (with padded zeros for the FFT) and in the stringData */
						size_t odx = ix + yo + zo;
						size_t idx = ix + yi + zi;

						// printf("odx %lu idx %lu yi %lu\n", odx, idx);
						/* initialise to zero the mask */

						m2sa[odx] = 0;

						switch(mask){
							case SPMASK_FLAT:
							case SPMASK_VIL:
							case SPMASK_VIL2:
							case SPMASK_SAXI:
								LogOut("These masks are automatic! why did you run this function??\n");
								LogMsg(VERB_NORMAL,"These masks are automatic! why did you run this function??");
								//exit!
							break;

							case SPMASK_REDO:
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

				if (iz == Lz-1) //given to one thread only I hope
				{
					/* Send ghosts from lastslicem2 -> mhalf */
					MPI_Send_init(sGhostFwd, ghostBytes, MPI_BYTE, fwdNeig, myRank,   MPI_COMM_WORLD, &rSendFwd);
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

			commSync();

			/* Fourier transform */
			// r2c FFT in m2
			auto &myPlan = AxionFFT::fetchPlan("pSpecSx");
			myPlan.run(FFT_FWD);

			/* bin the raw mask function */
							// LogMsg(VERB_NORMAL,"[masker] filling  bins");
							// binP.assign(powMax, 0.);
							//
							// switch (fPrec) {
							// 	case	FIELD_SINGLE:
							// 		if (spec)
							// 			fillBins<float,  SPECTRUM_P, true> ();
							// 		else
							// 			fillBins<float,  SPECTRUM_P, false>();
							// 		break;
							//
							// 	case	FIELD_DOUBLE:
							// 		if (spec)
							// 			fillBins<double,  SPECTRUM_P, true> ();
							// 		else
							// 			fillBins<double,  SPECTRUM_P, false>();
							// 		break;
							//
							// 	default:
							// 		LogError ("Wrong precision");
							// 		break;
							// }

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



			/* we needed it padded for the FFT
        unpad for plots but ... save in m22 ? */

				size_t dl = Ly*field->Precision();
				size_t pl = (Ly+2)*field->Precision();
				size_t ss	= Ly*Lz;
				char *mAS = static_cast<char *>(field->m2Cpu());
				char *mAH = static_cast<char *>(field->m2half());

				size_t dataTotalSize = (Ly+2)*Ly*Lz*field->Precision();

				/* if a save is needed */
					// memcpy	(mAH, mAS, dataTotalSize);

				/* unpad in place? or operate later with pads? */
				for (size_t sl=1; sl<ss; sl++) {
					size_t	oOff = sl*dl;
					size_t	fOff = sl*pl;
					memmove	(mAS+oOff, mAS+fOff, dl);
				}


			/* mask in m2 (unpadded) (we label M2_ENERGY for plotting reasons)
			and m2half padded to use with the spectrum */
			// field->setM2(M2_ENERGY);

			/* return mask in binary to strData following a criterion */
			{
				// the critical value has been calibrated to be ... (for sigma in 1-8)
				// min (radius_mask)^2 = 0.42772052 -0.05299264*radius_mask for radius_mask<4
				// min (radius_mask)^2 = 0.22619714 -0.00363601*radius_mask for radius_mask>4

					Float maskcut = (Float) std::abs(radius_mask);
					if (radius_mask < 4)
						maskcut = (0.42772052 -0.05299264*maskcut)/(maskcut*maskcut);
					else
						maskcut = (0.22619714 -0.00363601*maskcut)/(maskcut*maskcut);

					if (radius_mask > 8)
						maskcut = (0.22619714 -0.00363601*8)/(radius_mask*radius_mask);

					size_t vol = field->Size() ;
					#pragma omp parallel for schedule(static)
					for (size_t idx=0; idx < vol; idx++) {
						if ( m2sa[idx] > maskcut ) {
							strdaa[idx] |= STRING_MASK ; // mask stored M=1-W
							m2sa[idx] = 0; // in m2 we use W
						}
						else {
							m2sa[idx] = 1;
						}
					}

			}

			/* pad m2 inplace */
			for (size_t sl=1; sl<ss; sl++) {
				size_t isl = ss-sl;
				size_t	oOff = isl*dl;
				size_t	fOff = isl*pl;
				// LogOut("A %lu ",sl);
				memmove	(mAS+fOff, mAS+oOff, dl);
			}

			/* Calculate the FFT of the mask */
			myPlan.run(FFT_FWD);

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

		field->setSD(SD_MAPMASK);
		field->setM2(M2_DIRTY); // M2_MASK_FT

		}
		break; //case saxion ends

		case FIELD_AXION:
		{

			// thinking about axiton finder
			// energy is required in m2
			char *strdaa = static_cast<char *>(static_cast<void *>(field->sData()));
			Float *m2sa                 = static_cast<Float *>(field->m2Cpu());
			Float *m2sax                = static_cast<Float *>(field->m2half());
			char *mA = static_cast<char *>(field->m2Cpu());
			char *mAS = static_cast<char *>(field->m2half());
			size_t vola = field->Size() ;

			// makes a copy of the energy density of axions in m2_2

			memcpy (mAS, mA, vola*field->Precision());

			// threshold of the energy density
			Float RRRRRR = (Float) *field->RV();
			Float ethres = (Float) field->AxionMassSq()*RRRRRR*RRRRRR;

			#pragma omp parallel for schedule(static)
			for (size_t idx=0; idx < vola; idx++) {

				switch(mask){
					case SPMASK_AXIT:
						if( m2sa[idx] > ethres){
							// strdaa[idx] = 1;
							m2sa[idx] = 0.0;
						}
						// else
						// 	strdaa[idx] = 0;
					break;

					case SPMASK_AXIT2:
						if( m2sa[idx] > ethres)
							m2sa[idx] = 1.0;
						else
							m2sa[idx] = 0.0;
					break;
				} //end mask switch
			}    // end loop idx

			// pad in place
			size_t dSize    = (size_t) (field->Precision());
			size_t dataLine = dSize*Ly;
			size_t Sm	= Ly*Lz;
			for (int sl=Sm-1; sl>=0; sl--) {
				auto	oOff = sl*dSize*(Ly);
				auto	fOff = sl*dSize*(Ly+2);
				memmove	(mA+fOff, mA+oOff, dataLine);
			}

			/* Fourier transform */
			// r2c FFT in m2
			auto &myPlan = AxionFFT::fetchPlan("pSpecAx");
			myPlan.run(FFT_FWD);

			switch(mask)
			{
				case SPMASK_AXIT:
					field->setM2(M2_ENERGY_MASK_AXI_FFT);
				break;
				case SPMASK_AXIT2:
					field->setM2(M2_MASK_AXI2_FFT);
				break;
			}

			switch (mask)
			{

				case SPMASK_AXIT:
				{
							/* bin the axion energy spectrum */
							LogMsg(VERB_NORMAL,"[masker] filling masked eA spectrum bins (masked)");
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
						/* copy m2 back into place */
						memcpy (mA, mAS, vola*field->Precision());
						field->setM2(M2_ENERGY);

						// field still contains M2_ENERGY
				}
				break;

				case SPMASK_AXIT2:
				{
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

							/* unpadd */
							size_t dl = Ly*field->Precision();
							size_t pl = (Ly+2)*field->Precision();
							size_t ss	= Ly*Lz;
							size_t dataTotalSize = (Ly+2)*Ly*Lz*field->Precision();
							for (size_t sl=1; sl<ss; sl++) {
								size_t	oOff = sl*dl;
								size_t	fOff = sl*pl;
								memmove	(mA+oOff, mA+fOff, dl);
							}

							/* Generate mask */
							{
									Float maskcut = (Float) std::abs(radius_mask);
									if (radius_mask < 4)
										maskcut = (0.42772052 -0.05299264*maskcut)/(maskcut*maskcut);
									else
										maskcut = (0.22619714 -0.00363601*maskcut)/(maskcut*maskcut);

									if (radius_mask > 8)
										maskcut = (0.22619714 -0.00363601*8)/(radius_mask*radius_mask);

									/* I apply the cut to the unpadded */
									size_t vol = field->Size() ;
									#pragma omp parallel for schedule(static)
									for (size_t idx=0; idx < vol; idx++) {
										if ( m2sa[idx] > maskcut ) {
											// strdaa[idx] |= 1 ; // mask stored M=1-W
											m2sa[idx] = 0;
										}
										else {
											// strdaa[idx] = 0 ;
											//m2sa[idx] = 1; // use this if you interested in the mask
											m2sa[idx] = m2sax[idx]; // use this if you interested in the psp directly
										}
									}
							field->setM2(M2_ENERGY_AXI);// use this if you interested in the psp directly

							}
							/* pad m2 inplace */
							for (size_t sl=1; sl<ss; sl++) {
								size_t isl = ss-sl;
								size_t	oOff = isl*dl;
								size_t	fOff = isl*pl;
								memmove	(mA+fOff, mA+oOff, dl);
							}

							/* Calculate the FFT of the mask */
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

						/* copy m2 back into place */
						memcpy (mA, mAS, vola*field->Precision());
						// field->setSD(SD_AXITONMASK);
						field->setM2(M2_ENERGY);
				}
				break;

				default:
				LogError("[masker] Error: Axion mode but no axiton mask!!");
				break;
		} // end case mask
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
				size_t zo = Ly*(Ly+2)*iz ;
				size_t zi = Ly*Ly*iz ;
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
			auto &myPlan = AxionFFT::fetchPlan("pSpecSx");
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
