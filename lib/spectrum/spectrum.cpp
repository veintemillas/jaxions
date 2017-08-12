#include "enum-field.h"
#include "scalar/scalarField.h"
#include "fft/fftCode.h"

#include "spectrum/spectrum.h"

#include <omp.h>
#include <mpi.h>

void	SpecBin::fillCosTable () {

	const double	ooLx   = 1./Lx;
	const double	factor = 2./(sizeL*sizeL*ooLx*ooLx);

	cosTable.resize(kMax+2);

	#pragma omp parallel for schedule(static)
	for (int k=0; k<kMax+2; k++)
		cosTable[k] = factor*(1.0 - cos(M_PI*(2*k)*ooLx));
}

template<typename cFloat, const SpectrumType sType, const bool spectral>
void	SpecBin::fillBins	() {
	int mIdx;
	#pragma omp parallel
	{ mIdx = omp_num_threads(); }

	size_t	zBase = Lz*commRank();

	std:array<double>	tBinK;
	std:array<double>	tBinG;
	std:array<double>	tBinV;
	std:array<double>	tBinP;

	switch (sType) {
		case	SPECTRUM_K:
			tBinK.resize(powMax*mIdx);
			tBinK.fill(0);
			break;

		case	SPECTRUM_P:
			tBinP.resize(powMax*mIdx);
			tBinP.fill(0);
			break;

		default:
			tBinG.resize(powMax*mIdx);
			tBinV.resize(powMax*mIdx);
			tBinG.fill(0);
			tBinV.fill(0);
			break;
	}

	const double fc   = ((fType == FIELD_SAXION) ? 1.0 : 2.0);

	#pragma omp parallel
	{
		int tIdx = omp_get_thread_num ();

		#pragma omp for schedule(static)	// REVISA PARA AXION!!!
		for (size_t idx=0; idx<nPts; idx++) {

			size_t kz = idx/Lx;	// FFTW outputs a transpose array Ly x Lz x Lx with Lx = Ly
			size_t kx = idx - kz*Lx;
			size_t ky = kz/Lz;

			kz -= ky*Lz;
			kz += zBase;	// For MPI

			if (kx > hLx) kx -= Lx;
			if (ky > hLy) ky -= Ly;
			if (kz > hTz) kz -= Tz;

			double k2    = kx*kx + ky*ky + kz*kz;
			size_t myBin = floor(sqrt(k2));


			if (spectral)	// Includes norm sizeL^3/(2 volume^2)
				k2 *= (4.*M_PI*M_PI)/(sizeL*sizeL);
			else
				k2  = cosTable[abs(kx)] + cosTable[abs(ky)] + cosTable[abs(kz)];

			double		w  = sqrt(k2 + mass);
			double		m  = abs(static_cast<cFloat *>(field->m2Cpu())[idx]);
			double		m2 = m*m;
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

		#pragma omp single
		if (fType == FIELD_AXION) {
			if (zBase == 0) {
				double w  = sqrt(mass);
				double m  = abs(static_cast<cFloat *>(field->m2Cpu())[0]);
				double m2 = m*m;
				double mw = m*m/w;

				switch	(sType) {
					case	SPECTRUM_K:
						tBinK[0] -= mw;
						break;

					case	SPECTRUM_P:
						tBinK[0] -= m2;
						break;

					default:
						tBinG[0] -= mw*k2;
						tBinV[0] -= mw*mass;
						break;
				}
			} else if (zBase == (Lz*(commSize()>>1))) {
				size_t idx   = hLx + Lx*(hLz + Lz*hLy);
				double k2    = 2*hLy*hLy + hLz*hLz;
				size_t myBin = floor(sqrt(k2));

				if (spectral)
					k2 *= (4.*M_PI*M_PI)/(sizeL*sizeL);
				else
					k2  = cosTable[hLy] + cosTable[hLy] + cosTable[hLz];

				double w  = sqrt(k2 + mass);
				double m  = abs(static_cast<cFloat *>(field->m2Cpu())[idx]);
				double m2 = m*m;
				double mw = m2/w;

				switch	(sType) {
					case	SPECTRUM_K:
						tBinK[myBin] -= mw;
						break;

					case	SPECTRUM_P:
						tBinK[myBin] -= m2;
						break;

					default:
						tBinG[myBin] -= mw*k2;
						tBinV[myBin] -= mw*mass;
						break;
				}
			}
		}

		#pragma omp for schedule(static)
		for (int j=0; j<powMax; j++)
			for (int i=0; i<mIdx; i++) {
				const double norm = (sizeL*sizeL*sizeL)/(2.*field->TotalSize());

				switch	(sType) {
					case	SPECTRUM_K:
						binK[j] += tBinK[j + i*N]*norm;
						break;

					case	SPECTRUM_P:
						binP[j] += tBinP[j + i*N]*norm;
						break;

					default:
						binG[j] += tBinG[j + i*N]*norm;
						binV[j] += tBinV[j + i*N]*norm;
						break;
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
	switch (fPrec) {
		case	FIELD_SINGLE:
			switch (fType) {
				case	FIELD_SAXION:
				{
					auto &planM = AxionFFT::fetch("nSpecSxM");
					planM.run(FFT_FWD);
					fillBins<complex<float>, SPECTRUM_G | SPECTRUM_V, spec>();
					auto &planV = AxionFFT::fetch("nSpecSxV");
					planV.run(FFT_FWD);
					fillBins<complex<float>, SPECTRUM_K, spec>();
				}
				break;

				case	FIELD_AXION:
				{
					auto &myPlan = AxionFFT::fetch("nSpecAx");

					float *mO = static_cast<float *>(field->mCpu())  + field->Surf(); 
					float *vO = static_cast<float *>(field->vCpu()); 
					float *mF = static_cast<float *>(field->m2Cpu()) + field->Surf(); 

					size_t dataLine = field->DataSize()*Lx;

					// Copy m -> m2 with padding
					#pragma omp parallel for schedule(static)
					for (int sl=0; sl<Sf; sl++) {
						auto	oOff = sl*axionField->DataSize()*Lx;
						auto	fOff = sl*axionField->DataSize()*(Lx+2);
						memcpy	(mF+fOff, mO+oOff, dataLine);
					}

					myPlan.run(FFT_FWD);
					fillBins<float, SPECTRUM_G | SPECTRUM_V, spec>();

					// Copy m -> m2 with padding
					#pragma omp parallel for schedule(static)
					for (int sl=0; sl<Sf; sl++) {
						auto	oOff = sl*axionField->DataSize()*Lx;
						auto	fOff = sl*axionField->DataSize()*(Lx+2);
						memcpy	(mF+fOff, vO+oOff, dataLine);
					}

					myPlan.run(FFT_FWD);
					fillBins<float, SPECTRUM_K, spec>();
				}
				break;
			}
			break;

		case	FIELD_DOUBLE:
			switch (fType) {
				case	FIELD_SAXION:
				{
					auto &planM = AxionFFT::fetch("nSpecSxM");
					planM.run(FFT_FWD);
					fillBins<complex<double>, SPECTRUM_G | SPECTRUM_V, spec>();
					auto &planV = AxionFFT::fetch("nSpecSxV");
					planV.run(FFT_FWD);
					fillBins<complex<double>, SPECTRUM_K, spec>();
				}
				break;

				case	FIELD_AXION:
				{
					auto &myPlan = AxionFFT::fetch("nSpecAx");

					double *mO = static_cast<double *>(field->mCpu())  + field->Surf(); 
					double *vO = static_cast<double *>(field->vCpu()); 
					double *mF = static_cast<double *>(field->m2Cpu()) + field->Surf(); 

					size_t dataLine = field->DataSize()*Lx;

					// Copy m -> m2 with padding
					#pragma omp parallel for schedule(static)
					for (int sl=0; sl<Sf; sl++) {
						auto	oOff = sl*axionField->DataSize()*Lx;
						auto	fOff = sl*axionField->DataSize()*(Lx+2);
						memcpy	(mF+fOff, mO+oOff, dataLine);
					}

					myPlan.run(FFT_FWD);
					fillBins<double, SPECTRUM_G | SPECTRUM_V, spec>();

					// Copy m -> m2 with padding
					#pragma omp parallel for schedule(static)
					for (int sl=0; sl<Sf; sl++) {
						auto	oOff = sl*axionField->DataSize()*Lx;
						auto	fOff = sl*axionField->DataSize()*(Lx+2);
						memcpy	(mF+fOff, vO+oOff, dataLine);
					}

					myPlan.run(FFT_FWD);
					fillBins<double, SPECTRUM_K, spec>();
				}
				break;
			}
			break;
	}
}

void	SpecBin::pRun	() {
	switch (fPrec) {
		case	FIELD_SINGLE:
			switch (fType) {
				case	FIELD_SAXION:
				{
					auto &planM = AxionFFT::fetch("pSpecSx");
					planM.run(FFT_FWD);
					fillBins<complex<float>, SPECTRUM_P, spec>();
				}
				break;

				case	FIELD_AXION:
				{
					auto &myPlan = AxionFFT::fetch("pSpecAx");

					float *mA = static_cast<float *>(field->m2Cpu()) + field->Surf(); 

					size_t dataLine = field->DataSize()*Lx;

					// Add the f@*&#ng padding, no parallelization
					for (int sl=Sf-1; sl>=0; sl--) {
						auto	oOff = sl*axionField->DataSize()*Lx;
						auto	fOff = sl*axionField->DataSize()*(Lx+2);
						memcpy	(mA+fOff, mA+oOff, dataLine);
					}

					myPlan.run(FFT_FWD);
					fillBins<float, SPECTRUM_P, spec>();
				}
				break;
			}
			break;

		case	FIELD_DOUBLE:
			switch (fType) {
				case	FIELD_SAXION:
				{
					auto &planM = AxionFFT::fetch("pSpecSx");
					planM.run(FFT_FWD);
					fillBins<complex<double>, SPECTRUM_P, spec>();
				}

				case	FIELD_AXION:
				{
					auto &myPlan = AxionFFT::fetch("pSpecAx");

					double *mA = static_cast<double *>(field->m2Cpu()) + field->Surf(); 

					size_t dataLine = field->DataSize()*Lx;

					// Add the f@*&#ng padding, no parallelization
					for (int sl=Sf-1; sl>=0; sl--) {
						auto	oOff = sl*axionField->DataSize()*Lx;
						auto	fOff = sl*axionField->DataSize()*(Lx+2);
						memcpy	(mA+fOff, mA+oOff, dataLine);
					}

					myPlan.run(FFT_FWD);
					fillBins<double, SPECTRUM_P, spec>();
				}
				break;
			}
			break;
	}
}
