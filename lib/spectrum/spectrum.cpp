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
	const double	factor = (2.*Ly*Ly)/(sizeL*sizeL);

	cosTable.resize(kMax+1);

	#pragma omp parallel for schedule(static)
	for (int k=0; k<kMax+1; k++)
		cosTable[k] = factor*(1.0 - cos(M_PI*(2*k)*ooLx));

}

template<typename Float, const SpectrumType sType, const bool spectral>
void	SpecBin::fillBins	() {

	using cFloat = std::complex<Float>;

	const int mIdx = commThreads();

	size_t	zBase = Lz*commRank();

	std::vector<double>	tBinK;
	std::vector<double>	tBinG;
	std::vector<double>	tBinV;
	std::vector<double>	tBinP;

	switch (sType) {
		case	SPECTRUM_K:
			tBinK.resize(powMax*mIdx);
			tBinK.assign(powMax*mIdx, 0);
			break;

		case	SPECTRUM_P:
			tBinP.resize(powMax*mIdx);
			tBinP.assign(powMax*mIdx, 0);
			break;

		default:
			tBinG.resize(powMax*mIdx);
			tBinV.resize(powMax*mIdx);
			tBinG.assign(powMax*mIdx, 0);
			tBinV.assign(powMax*mIdx, 0);
			break;
	}

	const double fc   = ((fType == FIELD_SAXION) ? 1.0 : 2.0);

	#pragma omp parallel
	{
		int  tIdx = omp_get_thread_num ();
		auto hSf  = (field->Surf() >> 1);
		auto Sf   =  field->Surf();

		#pragma omp for schedule(static)
		for (size_t idx=0; idx<nPts; idx++) {

			int kz = idx/Lx;
			int kx = idx - kz*Lx;
			int ky = kz/Tz;

			// TODO Saxion WRONG, fcc ---> 1 para SAXION
			//JAVI ASSUMES THAT THE FFTS FOR SPECTRA ARE ALWAYS OF r2c type
			//and thus always in reduced format with half+1 of the elements in x
			double fcc = 2.0 ;
			if( kx == 0 )
			fcc = 1.0;
			if( kx == hLx - 1 )
			fcc = 1.0;

			kz -= ky*Tz;
			ky += zBase;	// For MPI, transposition makes the Y-dimension smaller

			if (kx > hLx) kx -= static_cast<int>(Lx);
			if (ky > hLy) ky -= static_cast<int>(Ly);
			if (kz > hTz) kz -= static_cast<int>(Tz);

			double k2    = kx*kx + ky*ky + kz*kz;
			size_t myBin = floor(sqrt(k2));

			if (myBin > powMax)
				printf ("Vaya bin!! %d %d %d ==> %llu\n", kx, ky, kz, myBin);

			if (spectral)
				k2 *= (4.*M_PI*M_PI)/(sizeL*sizeL);
			else
				k2  = cosTable[abs(kx)] + cosTable[abs(ky)] + cosTable[abs(kz)];

			double		w  = sqrt(k2 + mass);
			double		m  = abs(static_cast<cFloat *>(field->m2Cpu())[idx+hSf]);
			//double		m2 = fc*m*m;
			double		m2 = fcc*m*m;
			double		mw = m2/w;

			switch	(sType) {
				case	SPECTRUM_K:
					tBinK[myBin + powMax*tIdx] += mw;
					break;

				case	SPECTRUM_P:
					tBinP[myBin + powMax*tIdx] += m2;
					break;

				default:
					tBinG[myBin + powMax*tIdx] += mw*k2;
					tBinV[myBin + powMax*tIdx] += mw*mass;
					break;
			}
		}

		// if (fType == FIELD_AXION) {
		// 	#pragma omp for schedule(static)
		// 	for (size_t idx=0; idx<Sf; idx++) {	// Two surfaces, kx = 0 and kx = hLx
		// 		size_t eIdx = idx*Lx;		// Strided access in the YZ plane
		//
		// 		int kz = idx;
		// 		int ky = kz/Tz;
		//
		// 		kz -= ky*Tz;
		// 		ky += zBase;	// For MPI, transposition makes the Y-dimension smaller
		//
		// 		if (ky > hLy) ky -= static_cast<int>(Ly);
		// 		if (kz > hTz) kz -= static_cast<int>(Tz);
		//
		// 		double k20   = ky*ky + kz*kz;			// We compute both momenta at the same time
		// 		double k2m   = ky*ky + kz*kz + hLx*hLx;
		// 		size_t m0Bin = floor(sqrt(k20));
		// 		size_t mmBin = floor(sqrt(k2m));
		//
		// 		if (spectral) {
		// 			const double fSpc = (4.*M_PI*M_PI)/(sizeL*sizeL);
		//
		// 			k20 *= fSpc;
		// 			k2m *= fSpc;
		// 		} else {
		// 			k20 = cosTable[0]   + cosTable[abs(ky)] + cosTable[abs(kz)];
		// 			k2m = cosTable[hLx] + cosTable[abs(ky)] + cosTable[abs(kz)];
		// 		}
		//
		// 		double w0  = sqrt(k20 + mass);
		// 		double wm  = sqrt(k2m + mass);
		// 		double m0  = abs(static_cast<cFloat *>(field->m2Cpu())[eIdx+hSf]);
		// 		double mm  = abs(static_cast<cFloat *>(field->m2Cpu())[eIdx+hSf+hLx]);
		// 		double m20 = m0*m0;
		// 		double mw0 = m0*m0/w0;
		// 		double m2m = mm*mm;
		// 		double mwm = mm*mm/wm;
		//
		// 		switch	(sType) {
		// 			case	SPECTRUM_K:
		// 				tBinK[m0Bin] -= mw0;
		// 				tBinK[mmBin] -= mwm;
		// 				break;
		//
		// 			case	SPECTRUM_P:
		// 				tBinP[m0Bin] -= m20;
		// 				tBinP[mmBin] -= m2m;
		// 				break;
		//
		// 			default:
		// 				tBinG[m0Bin] -= mw0*k20;
		// 				tBinG[mmBin] -= mwm*k2m;
		// 				tBinV[m0Bin] -= mw0*mass;
		// 				tBinV[mmBin] -= mwm*mass;
		// 				break;
		// 		}
		// 	}
		// }

		const double norm = (sizeL*sizeL*sizeL)/(2.*(field->TotalSize()*field->TotalSize()));

		#pragma omp for schedule(static)
		for (int j=0; j<powMax; j++) {
			for (int i=0; i<mIdx; i++) {

				switch	(sType) {
					case	SPECTRUM_K:
						binK[j] += tBinK[j + i*powMax]*norm;
						break;

					case	SPECTRUM_P:
						binP[j] += tBinP[j + i*powMax]*norm;
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
			std::copy_n(binK.begin(), powMax, tBinK.begin());
			MPI_Allreduce(tBinK.data(), binK.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			break;

		case	SPECTRUM_P:
			std::copy_n(binP.begin(), powMax, tBinP.begin());
			MPI_Allreduce(tBinP.data(), binP.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			break;

		default:
			std::copy_n(binG.begin(), powMax, tBinG.begin());
			MPI_Allreduce(tBinG.data(), binG.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			std::copy_n(binV.begin(), powMax, tBinV.begin());
			MPI_Allreduce(tBinV.data(), binV.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			break;
	}
}

void	SpecBin::nRun	() {

	if	(field->Folded())
	{
		Folder	munge(field);
		munge(UNFOLD_ALL);
	}

	switch (fPrec) {
		case	FIELD_SINGLE:
			switch (fType) {
				case	FIELD_SAXION:
				{
					auto &planM = AxionFFT::fetchPlan("nSpecSxM");
					planM.run(FFT_FWD);
					if (spec)
						fillBins<float, SPECTRUM_GV, true> ();
					else
						fillBins<float, SPECTRUM_GV, false>();
					auto &planV = AxionFFT::fetchPlan("nSpecSxV");
					planV.run(FFT_FWD);
					if (spec)
						fillBins<float, SPECTRUM_K, true> ();
					else
						fillBins<float, SPECTRUM_K, false>();
				}
				break;

				case	FIELD_AXION:
				{
					auto &myPlan = AxionFFT::fetchPlan("pSpecAx");

					char *mO = static_cast<char *>(field->mCpu())  + field->Surf()*field->DataSize();
					char *vO = static_cast<char *>(field->vCpu());
					char *mF = static_cast<char *>(field->m2Cpu()) + field->Surf()*field->DataSize();

					size_t dataLine = field->DataSize()*Ly;
					size_t Sm	= Ly*Lz;

					// Copy m -> m2 with padding
					#pragma omp parallel for schedule(static)
					for (int sl=0; sl<Sm; sl++) {
						auto	oOff = sl*field->DataSize()*Ly;
						auto	fOff = sl*field->DataSize()*(Ly+2);
						memcpy	(mF+fOff, mO+oOff, dataLine);
					}

					myPlan.run(FFT_FWD);

					if (spec)
						fillBins<float, SPECTRUM_GV, true> ();
					else
						fillBins<float, SPECTRUM_GV, false>();

					// Copy v -> m2 with padding
					#pragma omp parallel for schedule(static)
					for (int sl=0; sl<Sm; sl++) {
						auto	oOff = sl*field->DataSize()*Ly;
						auto	fOff = sl*field->DataSize()*(Ly+2);
						memcpy	(mF+fOff, vO+oOff, dataLine);
					}

					//myPlan.run(FFT_BCK);
					myPlan.run(FFT_FWD);

					if (spec)
						fillBins<float, SPECTRUM_K, true> ();
					else
						fillBins<float, SPECTRUM_K, false>();
				}
				break;

				case	FIELD_WKB:
				LogError ("Error: WKB field not supported");
				return;
				break;
			}
			break;

		case	FIELD_DOUBLE:
			switch (fType) {
				case	FIELD_SAXION:
				{
					auto &planM = AxionFFT::fetchPlan("nSpecSxM");
					planM.run(FFT_FWD);
					if (spec)
						fillBins<double, SPECTRUM_GV, true> ();
					else
						fillBins<double, SPECTRUM_GV, false>();
					auto &planV = AxionFFT::fetchPlan("nSpecSxV");
					planV.run(FFT_FWD);
					if (spec)
						fillBins<double, SPECTRUM_K, true> ();
					else
						fillBins<double, SPECTRUM_K, false>();
				}
				break;

				case	FIELD_AXION:
				{
					auto &myPlan = AxionFFT::fetchPlan("pSpecAx");

					double *mO = static_cast<double *>(field->mCpu())  + field->Surf();
					double *vO = static_cast<double *>(field->vCpu());
					double *mF = static_cast<double *>(field->m2Cpu()) + field->Surf();

					size_t dataLine = field->DataSize()*Ly;
					size_t Sm	= Ly*Lz;

					// Copy m -> m2 with padding
					#pragma omp parallel for schedule(static)
					for (int sl=0; sl<Sm; sl++) {
						auto	oOff = sl*field->DataSize()*Ly;
						auto	fOff = sl*field->DataSize()*(Ly+2);
						memcpy	(mF+fOff, mO+oOff, dataLine);
					}
					//myPlan.run(FFT_BCK);
					myPlan.run(FFT_FWD);	//FIXME BCK for tests

					if (spec)
						fillBins<double, SPECTRUM_GV, true> ();
					else
						fillBins<double, SPECTRUM_GV, false>();

					// Copy m -> m2 with padding
					#pragma omp parallel for schedule(static)
					for (int sl=0; sl<Sm; sl++) {
						auto	oOff = sl*field->DataSize()*Ly;
						auto	fOff = sl*field->DataSize()*(Ly+2);
						memcpy	(mF+fOff, vO+oOff, dataLine);
					}
					//FFT_FWD
					myPlan.run(FFT_FWD);	//FIXME BCK for tests
					if (spec)
						fillBins<double, SPECTRUM_K, true> ();
					else
						fillBins<double, SPECTRUM_K, false>();
				}
				break;

				case	FIELD_WKB:
				LogError ("Error: WKB field not supported");
				return;
				break;
			}
			break;
	}
}

void	SpecBin::pRun	() {

	size_t dataLine = field->DataSize()*Ly;
	size_t Sf	= field->Surf();
	size_t Sm	= Ly*Lz;

	char *mA = static_cast<char *>(field->m2Cpu()) + field->Surf()*field->DataSize();

	// contrast bin is assumed in m2 (with ghost bytes)
	// Add the f@*&#ng padding plus ghost region, no parallelization
	for (int sl=Sm-1; sl>=0; sl--) {
		auto	oOff = sl*field->DataSize()*(Ly);
		auto	fOff = sl*field->DataSize()*(Ly+2);
		memcpy	(mA+fOff, mA+oOff, dataLine);
	}

	if (field->Field() == FIELD_SAXION) {
		auto &myPlan = AxionFFT::fetchPlan("pSpecSxP");
		myPlan.run(FFT_FWD);
	} else {
		auto &myPlan = AxionFFT::fetchPlan("pSpecAx");
		myPlan.run(FFT_FWD);
	}

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
	}
}
