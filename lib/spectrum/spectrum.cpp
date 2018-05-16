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

	#pragma omp parallel for schedule(static)
	for (int k=0; k<kMax+1; k++)
		cosTable[k] = factor*(1.0 - cos(M_PI*(2*k)*ooLx));

}

template<typename Float, const SpectrumType sType, const bool spectral>
void	SpecBin::fillBins	() {

	using cFloat = std::complex<Float>;

	const double norm = (field->BckGnd()->PhysSize()*field->BckGnd()->PhysSize()*field->BckGnd()->PhysSize()) /
			    (2.*(((double) field->TotalSize())*((double) field->TotalSize())));
	const int mIdx = commThreads();

	size_t	zBase = (Ly/commSize())*commRank();

	std::vector<double>	tBinK;
	std::vector<double>	tBinG;
	std::vector<double>	tBinV;
	std::vector<double>	tBinP;

	switch (sType) {
		case	SPECTRUM_K:
		case	SPECTRUM_KS:
			tBinK.resize(powMax*mIdx);
			tBinK.assign(powMax*mIdx, 0);
			break;

		case	SPECTRUM_P:
		case	SPECTRUM_PS:
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

			if (kx > hLx) kx -= static_cast<int>(Lx);
			if (ky > hLy) ky -= static_cast<int>(Ly);
			if (kz > hTz) kz -= static_cast<int>(Tz);

			double k2    = kx*kx + ky*ky + kz*kz;
			size_t myBin = floor(sqrt(k2));

			if (myBin > powMax) {
				LogError ("Error: point %lu (%d %d %d) bin out of range %lu > %lu\n", idx, kx, ky, kz, myBin, powMax);
				continue;
			}

			if (spectral)
				k2 *= (4.*M_PI*M_PI)/(field->BckGnd()->PhysSize()*field->BckGnd()->PhysSize());
			else
				k2  = cosTable[abs(kx)] + cosTable[abs(ky)] + cosTable[abs(kz)];

			double		w = 1.0;

			switch	(sType) {
				case	SPECTRUM_K:
				case	SPECTRUM_G:
				case 	SPECTRUM_V:
				case 	SPECTRUM_GV:
					w  = sqrt(k2 + mass);
					break;

				case	SPECTRUM_KS:
				case	SPECTRUM_GS:
				case 	SPECTRUM_VS:
				case 	SPECTRUM_GVS:
					w  = sqrt(k2 + massSax);
					break;
			}
			//double		w  = sqrt(k2 + mass);

			double		m  = abs(static_cast<cFloat *>(field->m2Cpu())[idx]);
			double		m2 = 0.;

			// FFTS are assumed outcome of FFT r2c
			// if c2c this needs some changes

			if ((kx == 0) || (kx == hLx - 1))
				m2 = m*m;
			else
				m2 = 2.*m*m;

			double		mw = m2/w;

			switch	(sType) {
				case	SPECTRUM_K:
				case	SPECTRUM_KS:
					tBinK.at(myBin + powMax*tIdx) += mw;
					break;

				case	SPECTRUM_P:
				case	SPECTRUM_PS:
					tBinP.at(myBin + powMax*tIdx) += m2;
					break;

				case	SPECTRUM_G:
				case	SPECTRUM_GS:
					tBinG.at(myBin + powMax*tIdx) += mw*k2;
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

			}
		}

		#pragma omp for schedule(static)
		for (int j=0; j<powMax; j++) {
			for (int i=0; i<mIdx; i++) {

				switch	(sType) {
					case	SPECTRUM_K:
					case	SPECTRUM_KS:
						binK[j] += tBinK[j + i*powMax]*norm;
						break;

					case	SPECTRUM_P:
					case	SPECTRUM_PS:
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
		case	SPECTRUM_KS:
			std::copy_n(binK.begin(), powMax, tBinK.begin());
			MPI_Allreduce(tBinK.data(), binK.data(), powMax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			break;

		case	SPECTRUM_P:
		case	SPECTRUM_PS:
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

	field->setM2     (M2_ENERGY_FFT);
}

// axion number spectrum
void	SpecBin::nRun	() {

	if	(field->Folded())
	{
		Folder	munge(field);
		munge(UNFOLD_ALL);
	}

	switch (fType) {
		case	FIELD_SAXION:
		{

			switch (fPrec) {
				case FIELD_SINGLE:
				{
					std::complex<float> *ma     = static_cast<std::complex<float>*>(field->mCpu())  + field->Surf();
					std::complex<float> *va     = static_cast<std::complex<float>*>(field->vCpu());
					float *m2sa                 = static_cast<float *>(field->m2Cpu());
					//DEBUG float *m2sa =           static_cast<float *>(static_cast<void*>(field->m2Cpu()));
					//DEBUG LogOut("Saxion spectrum single Lx %d Ly %d Lz %d ",Lx, Ly, Lz);
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
								// misses a factor of conformal time to be K
								m2sa[odx] = ztime*imag(va[idx]/ma[idx])+arg(ma[idx]) ;
							}
						}
					}
				}
				break;

				case FIELD_DOUBLE:
				{
					std::complex<double> *ma     = static_cast<std::complex<double>*>(field->mCpu())  + field->Surf();
					std::complex<double> *va     = static_cast<std::complex<double>*>(field->vCpu());
					double *m2sa            = static_cast<double *>(field->m2Cpu());
					//float *m2sa =           static_cast<float *>(static_cast<void*>(field->m2Cpu()));

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

								m2sa[odx] = ztime*imag(va[idx]/ma[idx]) ;
							}
						}
					}
				}
				break;
			}//End prec switch

			// r2c FFT in m2
			auto &myPlan = AxionFFT::fetchPlan("pSpecSx");
			myPlan.run(FFT_FWD);

			// for the moment it fills K bins only
			if (fPrec == FIELD_SINGLE) {
				if (spec)
					fillBins<float,  SPECTRUM_K, true> ();
				else
					fillBins<float,  SPECTRUM_K, false>();
			} else {
				if (spec)
					fillBins<double, SPECTRUM_K, true> ();
				else
					fillBins<double, SPECTRUM_K, false>();
			}

			field->setM2     (M2_DIRTY);
		}
		break;

		case	FIELD_AXION_MOD:
		case	FIELD_AXION:
		{
			auto &myPlan = AxionFFT::fetchPlan("pSpecAx");

			char *mO = static_cast<char *>(field->mCpu())  + field->Surf()*field->DataSize();
			char *vO = static_cast<char *>(field->vCpu());
			char *mF = static_cast<char *>(field->m2Cpu());

			size_t dataLine = field->DataSize()*Ly;
			size_t Sm	= Ly*Lz;

			// Copy m -> m2 with padding
			#pragma omp parallel for schedule(static)
			for (int sl=0; sl<Sm; sl++) {
				auto	oOff = sl*field->DataSize()* Ly;
				auto	fOff = sl*field->DataSize()*(Ly+2);
				memcpy	(mF+fOff, mO+oOff, dataLine);
			}

			myPlan.run(FFT_FWD);

			if (fPrec == FIELD_SINGLE) {
				if (spec)
					fillBins<float,  SPECTRUM_GV, true> ();
				else
					fillBins<float,  SPECTRUM_GV, false>();
			} else {
				if (spec)
					fillBins<double, SPECTRUM_GV, true> ();
				else
					fillBins<double, SPECTRUM_GV, false>();
			}

			// Copy v -> m2 with padding
			#pragma omp parallel for schedule(static)
			for (int sl=0; sl<Sm; sl++) {
				auto	oOff = sl*field->DataSize()* Ly;
				auto	fOff = sl*field->DataSize()*(Ly+2);
				memcpy	(mF+fOff, vO+oOff, dataLine);
			}

			myPlan.run(FFT_FWD);

			if (fPrec == FIELD_SINGLE) {
				if (spec)
					fillBins<float,  SPECTRUM_K, true> ();
				else
					fillBins<float,  SPECTRUM_K, false>();
			} else {
				if (spec)
					fillBins<double, SPECTRUM_K, true> ();
				else
					fillBins<double, SPECTRUM_K, false>();
			}

			field->setM2     (M2_DIRTY);
		}
		break;

		case	FIELD_WKB:
		LogError ("Error: WKB field not supported");
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
		case FIELD_AXION:
		case FIELD_AXION_MOD:
		case FIELD_WKB:
				LogError ("Error: Wrong field called to numberSaxionSpectrum: no Saxion information!!");
		return;
		break;

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

								float modu = std::abs(ma[idx]);
								m2sa[odx] = real(va[idx]*modu/ma[idx]) ;
								m2sax[odx] = modu - ztime ;
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

								double modu = abs(ma[idx]);
								m2sa[odx] = real(va[idx]*modu/ma[idx]) ;
								m2sax[odx] = modu - ztime ;
							}
						}
					}
				}
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

template<typename Float>
void	SpecBin::filterFFT	(int neigh) {

	using cFloat = std::complex<Float>;

	const int mIdx = commThreads();

	size_t	zBase = (Ly/commSize())*commRank();

	//prefactor is (2 pi^2 neigh^2/N^2)
	//double prefac = 2.0*M_PI*M_PI*neigh*neigh/field->Surf() ;
	double prefac = 0.5*M_PI*M_PI*neigh*neigh/field->Surf() ;

	LogMsg (VERB_NORMAL, "filterBins with %d neighbours, prefa = %f", neigh, prefac);

	//complex<Float> * m2ft = static_cast<complex<Float>*>(axion->m2Cpu());

	const double normn3 = field->TotalSize();

	#pragma omp parallel
	{
		int  tIdx = omp_get_thread_num ();

		#pragma omp for schedule(static)
		for (size_t idx=0; idx<nPts; idx++) {

			int kz = idx/Lx;
			int kx = idx - kz*Lx;
			int ky = kz/Tz;

			//JAVI ASSUMES THAT THE FFTS FOR SPECTRA ARE ALWAYS OF r2c type
			//and thus always in reduced format with half+1 of the elements in x

			kz -= ky*Tz;
			ky += zBase;	// For MPI, transposition makes the Y-dimension smaller

			if (kx > hLx) kx -= static_cast<int>(Lx);
			if (ky > hLy) ky -= static_cast<int>(Ly);
			if (kz > hTz) kz -= static_cast<int>(Tz);

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
	}

	LogMsg (VERB_NORMAL, "FFT m2 inplace -> ");
	auto &myPlan = AxionFFT::fetchPlan("pSpecAx");
	myPlan.run(FFT_BCK);
	LogMsg (VERB_NORMAL, "-> filtered density map in m2!");

	LogMsg (VERB_NORMAL, "reducing map [cherrypicking]");
	// reducemap consists on reorganising items of the filtered density map
	// we outputthem as a bin to use print bin
	// or as a reduced density map ?

	float *mCon = static_cast<float *>(static_cast<void*>(field->m2Cpu()));	// FIXME breaks for double precision
	size_t seta = (size_t) neigh ;
	size_t newNx = Ly/seta ;
	size_t newNz = Lz/seta ;
	size_t topa = newNx*newNx*newNz ;

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
