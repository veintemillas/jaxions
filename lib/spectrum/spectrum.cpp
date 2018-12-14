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

	switch (sType) {
		case	SPECTRUM_K:
		case	SPECTRUM_KS:
			tBinK.resize(powMax*mIdx);
			tBinK.assign(powMax*mIdx, 0);
			break;

		case	SPECTRUM_P:
			tBinP.resize(powMax*mIdx);
			tBinP.assign(powMax*mIdx, 0);
			break;
		case	SPECTRUM_PS:
		case 	SPECTRUM_NN:
			tBinPS.resize(powMax*mIdx);
			tBinPS.assign(powMax*mIdx, 0);
			break;
		// case  SPECTRUM_GaS: 					//uses PS to add to G
		// tBinPS.resize(powMax*mIdx);
		// tBinPS.assign(powMax*mIdx, 0);
		// 	break;
		case	SPECTRUM_G:
		case	SPECTRUM_GaSadd:
		case	SPECTRUM_GaS:
			tBinG.resize(powMax*mIdx);
			tBinG.assign(powMax*mIdx, 0);
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
			size_t myBin = floor(sqrt(k2));

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

			k2 *= (4.*M_PI*M_PI)/(field->BckGnd()->PhysSize()*field->BckGnd()->PhysSize());

			double		w = 1.0;
			double 		m, m2;

			switch	(sType) {
				case	SPECTRUM_K:
				case	SPECTRUM_G:
				case 	SPECTRUM_V:
				case 	SPECTRUM_GV:
				case 	SPECTRUM_GaS:
				case 	SPECTRUM_GaSadd:
				case 	SPECTRUM_P:
				case 	SPECTRUM_PS:
					w  = sqrt(k2 + mass);
					m  = std::abs(static_cast<cFloat *>(field->m2Cpu())[idx]);
					m2 = 0.;
					break;

				case	SPECTRUM_KS:
				case	SPECTRUM_GS:
				case 	SPECTRUM_VS:
				case 	SPECTRUM_GVS:
					w  = sqrt(k2 + massSax);
					m  = std::abs(static_cast<cFloat *>(field->m2Cpu())[idx]);
					m2 = 0.;
					break;
				case SPECTRUM_NN:
					m = 1;
					break;
			}
			//double		w  = sqrt(k2 + mass);


			// FFTS are assumed outcome of FFT r2c
			// if c2c this needs some changes
			// recall hLx - 1 = N/2
			if ((kx == 0) || (kx == static_cast<int>(hLx - 1)))
				m2 = m*m;
			else
				m2 = 2*m*m;

			// if ((kx == hLx - 1) || (ky == hLy) || (kz == hLz)){
			// 	m2=0;
			// 	LogOut("c ");
			// }

			double		mw = m2/w;


			switch	(sType) {
				case	SPECTRUM_K:
				case	SPECTRUM_KS:
					tBinK.at(myBin + powMax*tIdx) += mw;
					break;

				case	SPECTRUM_P:
					tBinP.at(myBin + powMax*tIdx) += m2;
					break;
				case	SPECTRUM_PS:
					tBinPS.at(myBin + powMax*tIdx) += m2;
					break;
				case	SPECTRUM_G:
				case	SPECTRUM_GS:
					tBinG.at(myBin + powMax*tIdx) += mw*k2;
					break;
				// the gradient is already included in m2
				// so I do not need to include the k2 factor here!
				case  SPECTRUM_GaS:
				case  SPECTRUM_GaSadd:
					//the cosine correction accounts for the finite difference formula
					// tBinG.at(myBin + powMax*tIdx) += mw/cosTable2[kx];
					tBinG.at(myBin + powMax*tIdx) += mw;
					break;

				case	SPECTRUM_V:
					tBinV.at(myBin + powMax*tIdx) += mw*mass;
					break;
				case	SPECTRUM_VS:
					tBinV.at(myBin + powMax*tIdx) += mw*massSax;
					break;

				case	SPECTRUM_GVS:
					tBinG.at(myBin + powMax*tIdx) += mw*k2;
					tBinV.at(myBin + powMax*tIdx) += mw*massSax;
					break;

				case	SPECTRUM_GV:
					tBinG.at(myBin + powMax*tIdx) += mw*k2;
					tBinV.at(myBin + powMax*tIdx) += mw*mass;
					break;
				case	SPECTRUM_NN:
					tBinPS.at(myBin + powMax*tIdx) += m2;
					break;
			}
		}

		#pragma omp for schedule(static)
		for (uint j=0; j<powMax; j++) {
			for (int i=0; i<mIdx; i++) {

				switch	(sType) {
					case	SPECTRUM_K:
					case	SPECTRUM_KS:
						binK[j] += tBinK[j + i*powMax]*norm;
						break;
					case	SPECTRUM_P:
						binP[j] += tBinP[j + i*powMax]*norm;
						break;
					case	SPECTRUM_PS:
						binPS[j] += tBinPS[j + i*powMax]*norm;
						break;
					case	SPECTRUM_G:
					case	SPECTRUM_GaS:
					case	SPECTRUM_GaSadd:
						binG[j] += tBinG[j + i*powMax]*norm;
						break;
					case	SPECTRUM_NN:
						binPS[j] += tBinPS[j + i*powMax];
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
		case	SPECTRUM_KS:
			std::copy_n(binK.begin(), powMax, tBinK.begin());
			MPI_Allreduce(tBinK.data(), binK.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			break;

		case	SPECTRUM_P:
			std::copy_n(binP.begin(), powMax, tBinP.begin());
			MPI_Allreduce(tBinP.data(), binP.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			break;

		case	SPECTRUM_PS:
		case	SPECTRUM_NN:
			std::copy_n(binPS.begin(), powMax, tBinPS.begin());
			MPI_Allreduce(tBinPS.data(), binPS.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			break;
		case	SPECTRUM_GaSadd:
		// we do not do anything, just keep each binG with its local sum
			break;
		case	SPECTRUM_G:
		case	SPECTRUM_GaS:
		// now we assume that all 3 grad squared are in binG
		// we can reduce among ranks
			std::copy_n(binG.begin(), powMax, tBinG.begin());
			MPI_Allreduce(tBinG.data(), binG.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			break;

		// we sum tBinPS into tBinG and reduce into binG / tBinG remains
		// std::copy_n(binPS.begin(), powMax, tBinG.begin());
		// MPI_Allreduce(tBinG.data(), binG.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		// 	break;
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

		case SPMASK_TEST :
				switch (fPrec)
				{
					case FIELD_SINGLE :
					SpecBin::nRun<float,SPMASK_TEST> ();
					break;

					case FIELD_DOUBLE :
					SpecBin::nRun<double,SPMASK_TEST> ();
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

	if ((mask == SPMASK_TEST) && (field->sDStatus() & SD_MASK))
	{
		LogMsg(VERB_NORMAL,"nRun with SPMASK_TEST ok SPMASK=%d field->statusSD()=%d",SPMASK_TEST,field->sDStatus()) ;
	} else {
		LogMsg(VERB_NORMAL,"nRun with SPMASK_TEST but SPMASK=%d field->statusSD()=%d ... EXIT!",SPMASK_TEST,field->sDStatus()) ;
		return ;
	}


	binK.assign(powMax, 0.);
	binG.assign(powMax, 0.);
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
							case SPMASK_TEST:
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
				fillBins<Float,  SPECTRUM_K, true> ();
			else
				fillBins<Float,  SPECTRUM_K, false>();


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
							case SPMASK_TEST:
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


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

template<typename Float>
void	SpecBin::filterFFT	(double neigh) {

	using cFloat = std::complex<Float>;

	//const int mIdx = commThreads();

	size_t	zBase = (Ly/commSize())*commRank();

	//prefactor is (2 pi^2 neigh^2/N^2)
	//double prefac = 2.0*M_PI*M_PI*neigh*neigh/field->Surf() ;
	double prefac = 0.5*M_PI*M_PI*neigh*neigh/field->Surf() ;

	LogMsg (VERB_NORMAL, "filterBins with %d neighbours, prefa = %f", neigh, prefac);

	//complex<Float> * m2ft = static_cast<complex<Float>*>(axion->m2Cpu());

	const double normn3 = field->TotalSize();

	#pragma omp parallel
	{
		//int  tIdx = omp_get_thread_num ();

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
			static_cast<cFloat *>(field->m2Cpu())[idx] *= exp(-prefac*k2)/normn3;
		}

	}
}

void	SpecBin::filter (int neigh) {

	LogMsg (VERB_NORMAL, "Filter assumes m2 contains FFT r2c");
	// FFT of contrast bin is assumed in m2 (with ghost bytes)
	// filter with a Gaussian over n neighbours
	// exp(- ksigma^2/2)
	// k = 2Pi* n/ L    [n labels mode number]
	// sigma = delta* number of neighbours
	// ksigma^2/2 = 2 pi^2 [n^2]/N^2 * (neighbour)^2


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

	LogMsg (VERB_NORMAL, "FFT m2 inplace -> ");
	auto &myPlan = AxionFFT::fetchPlan("pSpecAx");
	myPlan.run(FFT_BCK);
	LogMsg (VERB_NORMAL, "-> filtered density map in m2!");

	LogMsg (VERB_NORMAL, "reducing map [cherrypicking]");
	// reducemap consists on reorganising items of the filtered density map
	// we outputthem as a bin to use print bin
	// or as a reduced density map ?


	size_t seta = (size_t) neigh ;
	size_t newNx = Ly/seta ;
	size_t newNz = Lz/seta ;

	switch (fPrec) {
		case	FIELD_SINGLE:
		{
			float *mCon = static_cast<float *>(static_cast<void*>(field->m2Cpu()));	// FIXME breaks for double precision
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
			double *mCon = static_cast<double *>(static_cast<void*>(field->m2Cpu()));	// FIXME breaks for double precision

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

		case SPMASK_TEST :
		default:
			switch (fPrec)
			{
				case FIELD_SINGLE :
				SpecBin::masker<float,SPMASK_TEST> (radius_mask);
				break;

				case FIELD_DOUBLE :
				SpecBin::masker<double,SPMASK_TEST> (radius_mask);
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

	if ( !(field->sDStatus() & SD_MAP)){
			LogOut("[masker] masker called without string map! exit!\n");
			return;
	}
	if (field->LowMem()){
			LogOut("[masker] masker called in lowmem! exit!\n");
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
					memcpy	(mAS+oOff, mAS+fOff, dl);
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
				memcpy	(mAS+fOff, mAS+oOff, dl);
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

		field->setSD(SD_MAPMASK);
		field->setM2(M2_DIRTY); // M2_MASK_FT

		}
		break; //case saxion ends

		case FIELD_AXION:
				LogError("[masker] Error: Masker template not available in axion mode (yet)!");
		break;

		default:
		LogError("[masker] Error: Masker template called with no saxion mode!");
		break ;
	}
}

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

	// calculate phase space density (stored in binPS), which will be used below
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

	switch (fType) {
		case	FIELD_SAXION:
		{
			double *m2sa = static_cast<double *>(field->m2Cpu());
			// split i direction to MPI processes
			// resulting matrix M_ij is of the form (powMaxPad*Nrank x powMax)
			// the exccess part in i should be cut later.
			#pragma omp parallel for schedule(static)
			for (size_t i=0; i<powMaxPad; i++) {
				size_t is = iBase + i;
				for (size_t j=0; j<powMax; j++) {
					size_t indM = i*powMax+j;
					m2sa[indM] = 0;
					for (size_t k=0; k<powMax; k++) {
						double J = 0;
						if (k==0) {
							if (j==0) {
								J = (is==0)?vol:0;
							} else {
								J = (is==j)?vol/binPS.at(is):0;
							}
						} else {
							if (j==0) {
								J = (is==k)?vol/binPS.at(k):0;
							} else {
								int diffkj = static_cast<int>(j) - static_cast<int>(k);
								if (is==0) {
									J = vol/binPS.at(j);
								} else if (is>=std::abs(diffkj) && is<=j+k && is < powMax) {
									J = coeJ/(is*j*k);
								} else {
									J = 0;
								}
							}
						}
						m2sa[indM] += norm*binP.at(k)*J;
						// if(k==1) m2sa[indM] = J;
					}
				}
			}

			void * buf = field->m2Cpu();
			//MPI_Allgather(m2sa, powMaxPad*powMax, MPI_DOUBLE, m2sa, powMaxPad*powMax, MPI_DOUBLE, MPI_COMM_WORLD);
			size_t charlengh = powMaxPad*powMax*sizeof(double);
			MPI_Allgather(buf, charlengh, MPI_CHAR, buf, charlengh, MPI_CHAR, MPI_COMM_WORLD);
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

		case SPMASK_TEST :
			switch (fPrec)
				{
					case FIELD_SINGLE :
					SpecBin::wRun<float,SPMASK_TEST> ();
					break;

					case FIELD_DOUBLE :
					SpecBin::wRun<double,SPMASK_TEST> ();
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
							case SPMASK_TEST:
									//assume the map of W was already stored in stringdata
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
