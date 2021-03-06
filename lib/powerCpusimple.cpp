#include <complex>

#include "scalar/scalarField.h"
#include "utils/index.h"
#include "energy/energy.h"
#include "scalar/varNQCD.h"
#include <omp.h>

#include <fftw3-mpi.h>
#include "comms/comms.h"
#include "utils/utils.h"
#include "fft/fftCode.h"


//#include<mpi.h>

using namespace std;


//--------------------------------------------------
// BIN NUMBER SPECTRUM Gradient and Votential part
//--------------------------------------------------

template<typename Float>
void	BinSpectrumGV (const complex<Float> *ft, double *binarray, size_t n1, size_t Lz, size_t Tz, size_t powmax, int kmax, double mass2)
{
	//printf("sizeL=%f , ",sizeL);fflush(stdout);
	const size_t n2 = n1*n1 ;
	const size_t n3 = n1*n1*Lz ;

	double norma = pow(sizeL,3.)/(2.*pow((double) n1,6.)) ;

	// DISCRETE VERSION OF K2
	double minus1costab [kmax+1] ;
	double id2 = 2.0/pow(sizeL/sizeN,2);

	#pragma omp parallel for default(shared) schedule(static)
	for (int i=0; i < kmax + 2; i++)
	{
		minus1costab[i] = id2*(1.0 - cos(((double) (6.2831853071796*i)/n1)));
	}
	//printf("mtab , ");fflush(stdout);


	// MPI FFT STUFF
	int rank = commRank();
	size_t local_1_start = rank*Lz;
	// MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// ptrdiff_t alloc_local, local_n0, local_0_start, local_n1, local_1_start;
	//
	// alloc_local = fftw_mpi_local_size_3d_transposed(
	// 						 Tz, n1, n1, MPI_COMM_WORLD,
	// 						 &local_n0, &local_0_start,
	// 						 &local_n1, &local_1_start);

	//printf ("BIN rank=%d - transpo - local_ny_start=%lld \n", rank,  local_1_start);
	fflush(stdout);

	#pragma omp parallel
	{

		int tid = omp_get_thread_num();
		// LOCAL THREAD BINS
		double spectrumG_private[powmax];
		double spectrumV_private[powmax];

		for (int i=0; i < powmax; i++)
		{
			spectrumG_private[i] = 0.0;
			spectrumV_private[i] = 0.0;
		}

		// FFT MPI NON TRANSPOSED
		// GLOBALLY
		// kdx = iX + iY*n1 + iZ*n2 (i < n1)
		// K = (iZ, iY, iX) or (iZ-n1, iY-n1, iX-n1) [the smallest]
		// LOCAL DISTRIBUTION on Z
		// kdx = iX + iY*n1 + iZ*n2 (iZ < Lz)
		// K = (iZ+rank*Lz, iY, iX) or (iZ+rank*Lz-n1, iY-n1, iX-n1) [the smallest]
		//
		// FFT MPI TRANSPOSED
		// GLOBALLY
		// kdx = iX + iZ*n1 + iY*n2 (i < n1)
		// K = (iZ, iY, iX) or (iZ-n1, iY-n1, iX-n1) [the smallest]
		// LOCAL DISTRIBUTION on Y
		// kdx = iX + iZ*n1 + iY*n2 (iZ < Lz)
		// K = (iZ, iY+rank*Lz, iX) or (iZ-n1, iY++rank*Lz-n1, iX-n1) [the smallest]

		//PADDING
		// As for the complex transforms, improved performance can be obtained by
		// specifying that the output is the transpose of the input or vice versa
		// (see Section 6.4.3 [Transposed distributions],page58).In our L×M×N r2c example,
		// including FFTW_TRANSPOSED_OUT inthe  ags means that
		// the input would be a padded L×M×2(N/2+1) real array distributed over the L dimension,
		// the output would be a M×L×(N/2 + 1) complex array distributed over the M dimension.

		size_t kdx;
		int bin;
		size_t iz, iy, ix;
		int kz, ky, kx;
		double k2, w;

		size_t n1pad = n1/2 + 1;
		size_t n2pad = n1*n1pad;
		size_t n3pad = Lz*n2pad;

	  #pragma omp barrier

		#pragma omp for schedule(static)
		for (size_t kdx = 0; kdx< n3pad; kdx++)
		{
			// ASSUMED TRANSPOSED
			// COMPLEX WITHOUT REDUNDANCY
			// I.E. ix does not belong to (0, Lx-1)
			// but to (0, Lx/2)
			// point has already the n2 ghost into account
			// therefore
			// idx = ix + (Lx/2)*iz + Lx*(Lx/2)*iy
			iy = kdx/n2pad + local_1_start;
			iz = (kdx%n2pad)/n1pad ;
			ix = kdx%n1pad ;
			ky = (int) iy;
			kz = (int) iz;
			kx = (int) ix;
			if (kz>n1/2) {kz = kz-n1; }
			if (ky>n1/2) {ky = ky-n1; }
			if (kx>n1/2) {kx = kx-n1; }

			k2 = kz*kz + ky*ky + kx*kx;
			bin  = (int) floor(sqrt(k2)) 	;

			if (spectral) //CONTINUUM DEFINITION
			{
				k2 =	(39.47841760435743/(sizeL*sizeL)) * k2;
			}
			else //LATICE DEFINITION
			{
				k2 =	(minus1costab[abs(kx)]+minus1costab[abs(ky)]+minus1costab[abs(kz)]);
			}

			w = sqrt(k2 + mass2);


			spectrumG_private[bin] += pow(abs(ft[kdx]),2)*k2/(w);
			spectrumV_private[bin] += pow(abs(ft[kdx]),2)*mass2/w;

		}// END LOOP


		#pragma omp critical
		{
			for(int n=0; n<powmax; n++)
			{
				//binarray[n] += spectrumK_private[n];
				binarray[powmax*4	+ n] += spectrumG_private[n];
				binarray[powmax*5 + n] += spectrumV_private[n];
      }
		}

	}//parallel

	for(int n=0; n<powmax; n++)
	{
		////binarray[n] *= norma;
		// binarray[powmax   + n] = (double) powmax   + n;
		// binarray[powmax*2 + n] = (double) 2*powmax   + n;
		 binarray[powmax*4 + n] *= norma;
		 binarray[powmax*5 + n] *= norma;
	}

	MPI_Reduce(&binarray[powmax*4], &binarray[powmax], 2*powmax, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

}

template<typename Float>
void	BinSpectrumK (const complex<Float> *ft, double *binarray, size_t n1, size_t Lz, size_t Tz, size_t powmax, int kmax, double mass2)
{
	//printf("sizeL=%f , ",sizeL);fflush(stdout);
	const size_t n2 = n1*n1 ;
	const size_t n3 = n1*n1*Lz ;

	double norma = pow(sizeL,3.)/(2.*pow((double) n1,6.)) ;

	// DISCRETE VERSION OF K2
	double minus1costab [kmax+1] ;
	double id2 = 2.0/pow(sizeL/sizeN,2);

	#pragma omp parallel for default(shared) schedule(static)
	for (int i=0; i < kmax + 2; i++)
	{
		minus1costab[i] = id2*(1.0 - cos(((double) (6.2831853071796*i)/n1)));
	}
	//printf("mtab , ");fflush(stdout);


	// MPI FFT STUFF
	int rank = commRank();
	size_t local_1_start = rank*Lz;

	#pragma omp parallel
	{

		int tid = omp_get_thread_num();

		// LOCAL THREAD BINS
		double spectrumK_private[powmax];

		for (int i=0; i < powmax; i++)
		{
			spectrumK_private[i] = 0.0;
		}

		size_t kdx;
		int bin;
		size_t iz, iy, ix;
		int kz, ky, kx;
		double k2, w;
		size_t n1pad = n1/2 + 1;
		size_t n2pad = n1*n1pad;
		size_t n3pad = Lz*n2pad;

		#pragma omp barrier

		#pragma omp for schedule(static)
		for (size_t kdx = 0; kdx< n3pad; kdx++)
		{
			// // ASSUMED TRANSPOSED
			// iy = kdx/n2 + local_1_start;
			// iz = (kdx%n2)/n1 ;
			// ix = kdx%n1 ;
			// ky = (int) iy;
			// kz = (int) iz;
			// kx = (int) ix;
			// if (kz>n1/2) {kz = kz-n1; }
			// if (ky>n1/2) {ky = ky-n1; }
			// if (kx>n1/2) {kx = kx-n1; }

			// ASSUMED TRANSPOSED
			// COMPLEX WITHOUT REDUNDANCY
			// I.E. ix does not belong to (0, Lx-1)
			// but to (0, Lx/2)
			// point has already the n2 ghost into account
			// therefore
			// idx = ix + (Lx/2)*iz + Lx*(Lx/2)*iy
			iy = kdx/n2pad + local_1_start;
			iz = (kdx%n2pad)/n1pad ;
			ix = kdx%n1pad ;
			ky = (int) iy;
			kz = (int) iz;
			kx = (int) ix;
			if (kz>n1/2) {kz = kz-n1; }
			if (ky>n1/2) {ky = ky-n1; }
			if (kx>n1/2) {kx = kx-n1; }

			k2 = kz*kz + ky*ky + kx*kx;
			bin  = (int) floor(sqrt(k2)) 	;

			if (spectral) //CONTINUUM DEFINITION
			{
				k2 =	(39.47841760435743/(sizeL*sizeL)) * k2;
			}
			else //LATICE DEFINITION
			{
				k2 =	(minus1costab[abs(kx)]+minus1costab[abs(ky)]+minus1costab[abs(kz)]);
			}
			w = sqrt(k2 + mass2);

			spectrumK_private[bin] += pow(abs(ft[kdx]),2)/(w);

		}// END LOOP


		#pragma omp critical
		{
			for(int n=0; n<powmax; n++)
			{
				binarray[powmax*3 + n] += spectrumK_private[n];
      }
		}

	}//parallel

	for(int n=0; n<powmax; n++)
	{
		binarray[powmax*3 + n] *= norma;
	}

	MPI_Reduce(&binarray[powmax*3], &binarray[0], powmax, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	//printf(" ... GV Axion spectrum printed\n");
}

//  Copies theta into m2
//	axion->thetha2m2
//	FFT m2 inplace -> [theta]_k
//	Bin power spectrum GthetaPS
//	|mode|^2 * k^2/w
//	Bin power spectrum VthetaPS
//	|mode|^2 * m^2/w
//
void	spectrumUNFOLDED(Scalar *axion)
{
	const size_t n1 = axion->Length();
	const size_t Lz = axion->Depth();
	const size_t Tz = axion->TotalSize();
	const int kmax = n1/2 -1;
	int powmax = floor(1.733*kmax)+2 ;

	const int fSize = axion->DataSize();
	// SETS 3 ARRAYS TO ZERO AT THE BEGGINING OF M
	//memset (axion->mCpu(), 0, 6*fSize*powmax);
	for(size_t dix=0; dix < 6*fSize*powmax; dix++)
	{
		static_cast<double*> (axion->mCpu())[dix] = 0. ;
	}

	double mass2 = axionmass2((*axion->zV()), nQcd, zthres, zrestore)*(*axion->zV())*(*axion->zV());

	auto &myPlan = AxionFFT::fetchPlan("pSpectrum_ax");
	// 	2 STEP SCHEME FOR MPI // OUTPUTS TO M

	// 	FIRST G AND V
	//	COPIES c_theta into RE[m2], IM[m2] = 0
	axion->theta2m2();

	//	FFT m2 inplace ->
	myPlan.run(FFT_FWD);

	switch(axion->Precision())
	{
		case FIELD_DOUBLE:
		BinSpectrumGV<double>(static_cast<const complex<double>*>(axion->m2Cpu()) ,
		static_cast<double*>(axion->mCpu()), n1, Lz, Tz, powmax, kmax, mass2);
		break;

		case FIELD_SINGLE:
		BinSpectrumGV<float>(static_cast<const complex<float>*>(axion->m2Cpu()) ,
		static_cast<double*>(axion->mCpu()), n1, Lz, Tz, powmax, kmax, mass2);
		break;

		default:
		printf ("Not a valid precision.\n");
		break;
	}

	// 	SECOND KIN
	//	COPIES vheta into RE[m2], IM[m2] = 0
	axion->vheta2m2();
	//	FFT m2 inplace ->
	myPlan.run(FFT_FWD);

	switch(axion->Precision())
	{
		case FIELD_DOUBLE:
		BinSpectrumK<double>(static_cast<const complex<double>*>(axion->m2Cpu()) ,
		static_cast<double*>(axion->mCpu()), n1, Lz, Tz, powmax, kmax, mass2);
		break;

		case FIELD_SINGLE:
		BinSpectrumK<float>(static_cast<const complex<float>*>(axion->m2Cpu()) ,
		static_cast<double*>(axion->mCpu()), n1, Lz, Tz, powmax, kmax, mass2);
		break;

		default:
		printf ("Not a valid precision.\n");
		break;
	}



}

//--------------------------------------------------------------------------------
//					POWER SPECTRUM
//--------------------------------------------------------------------------------
//		pSpectrumUNFOLDED<double>(static_cast<const complex<double>*>(axion->m2Cpu()) + axion->Surf(), spectrumK, spectrumG, spectrumV, n1, powmax, kmax);

template<typename Float>
void	BinSpectrum (const complex<Float> *ft, double *binarray, size_t n1, size_t Lz, size_t Tz, size_t powmax, int kmax, double mass2)
{
	//printf("sizeL=%f , ",sizeL);fflush(stdout);
	const size_t n2 = n1*n1 ;
	const size_t n3 = n1*n1*Lz ;

	double norma = pow(sizeL,3.)/(2.*pow((double) n1,6.)) ;


	// MPI FFT STUFF
	int rank = commRank();
	size_t local_1_start = rank*Lz;


	#pragma omp parallel default(shared)
	{

		int tid = omp_get_thread_num();

		// LOCAL THREAD BINS
		double spectrumK_private[powmax];

		for (int i=0; i < powmax; i++)
		{
			spectrumK_private[i] = 0.0;
		}

		size_t kdx;
		int bin;
		size_t iz, iy, ix;
		int kz, ky, kx;
		double k2, w;
		size_t n1pad = n1/2 + 1;
		size_t n2pad = n1*n1pad;
		size_t n3pad = Lz*n2pad;

		#pragma omp barrier

		#pragma omp for schedule(static)
		for (size_t kdx = 0; kdx< n3pad; kdx++)
		{
			// ASSUMED TRANSPOSED
			// COMPLEX WITHOUT REDUNDANCY
			// I.E. ix does not belong to (0, Lx-1)
			// but to (0, Lx/2)
			// point has already the n2 ghost into account
			// therefore
			// idx = ix + (Lx/2)*iz + Lx*(Lx/2)*iy
			iy = kdx/n2pad + local_1_start;
			iz = (kdx%n2pad)/n1pad ;
			ix = kdx%n1pad ;
			ky = (int) iy;
			kz = (int) iz;
			kx = (int) ix;
			if (kz>n1/2) {kz = kz-n1; }
			if (ky>n1/2) {ky = ky-n1; }
			if (kx>n1/2) {kx = kx-n1; }

			k2 = kz*kz + ky*ky + kx*kx;
			bin  = (int) floor(sqrt(k2)) 	;

			//spectrumK_private[bin] += 1.0 ;
			spectrumK_private[bin] += pow(abs(ft[kdx]),2);

		}// END LOOP


		#pragma omp critical
		{
			for(int n=0; n<powmax; n++)
			{
				binarray[powmax + n] += spectrumK_private[n];
      }
		}

	}//parallel

	for(int n=0; n<powmax; n++)
	{
		binarray[powmax + n] *= norma;
	}

	MPI_Reduce(&binarray[powmax], &binarray[0], powmax, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	//printf(" ... GV Axion spectrum printed\n");
}




void	powerspectrumUNFOLDED(Scalar *axion)
{

	const size_t n1 = axion->Length();
	const size_t Lz = axion->Depth();
	const size_t Tz = axion->TotalSize();
	const int kmax = n1/2 -1;
	int powmax = floor(1.733*kmax)+2 ;
	const double delta = sizeL/sizeN;
	const int fSize = axion->DataSize();
	// SETS 1 ARRAY TO ZERO AT THE BEGGINING OF M
	//memset (axion->mCpu(), 0, 2*fSize*powmax);
	for(size_t dix=0; dix < 2*fSize*powmax; dix++)
	{
		static_cast<double*> (axion->mCpu())[dix] = 0. ;
	}

	double mass2 = axionmass2((*axion->zV()), nQcd, zthres, zrestore)*(*axion->zV())*(*axion->zV());
	double eRes[10];
	// 	New scheme

	// ASSUMES THAT ENERGY M2 FIELD WAS ALREADY CREATED BY THE DENSITY ANALYSIS PART
	// pads the data for the r2c
	axion->padder();

	//	FFT m2 inplace ->
	auto &myPlan = AxionFFT::fetchPlan("pSpectrum_ax");

	myPlan.run(FFT_FWD);

	switch(axion->Precision())
	{
		case FIELD_DOUBLE:
		BinSpectrum<double>(static_cast<const complex<double>*>(axion->m2Cpu()) ,
		static_cast<double*>(axion->mCpu()), n1, Lz, Tz, powmax, kmax, mass2);
		break;

		case FIELD_SINGLE:
		BinSpectrum<float>(static_cast<const complex<float>*>(axion->m2Cpu())  ,
		static_cast<double*>(axion->mCpu()), n1, Lz, Tz, powmax, kmax, mass2);
		break;

		default:
		printf ("Not a valid precision.\n");
		break;
	}
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

//--------------------------------------------------
// BINS EXP[iTHETA] FIELD TO CHECK INITIAL SPECTRUM
//--------------------------------------------------

template<typename Float>
void	BinFT2expitheta (const complex<Float> *ft, double *binarray, size_t n1, size_t Lz, size_t Tz, size_t powmax, int kmax, double mass2)
{
	//printf("sizeL=%f , ",sizeL);fflush(stdout);
	const size_t n2 = n1*n1 ;
	const size_t n3 = n1*n1*Lz ;

	double norma = 1/(pow((double) n1,6.)) ;

	// MPI FFT STUFF
	int rank = commRank();
	size_t local_1_start = rank*Lz;
	// MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// ptrdiff_t alloc_local, local_n0, local_0_start, local_n1, local_1_start;
	//
	// alloc_local = fftw_mpi_local_size_3d_transposed(
	// 						 Tz, n1, n1, MPI_COMM_WORLD,
	// 						 &local_n0, &local_0_start,
	// 						 &local_n1, &local_1_start);

	//printf ("BIN rank=%d - transpo - local_ny_start=%lld \n", rank,  local_1_start);
	fflush(stdout);

	#pragma omp parallel
	{

		int tid = omp_get_thread_num();
		// LOCAL THREAD BINS
		double spectrum_private[powmax];

		for (int i=0; i < powmax; i++)
		{
			spectrum_private[i] = 0.0;
		}

		size_t kdx;
		int bin;
		size_t iz, iy, ix;
		int kz, ky, kx;
		double k2, w;

	  #pragma omp barrier

		#pragma omp for schedule(static)
		for (size_t kdx = 0; kdx< n3; kdx++)
		{
			// ASSUMED TRANSPOSED
			iy = kdx/n2 + local_1_start;
			iz = (kdx%n2)/n1 ;
			ix = kdx%n1 ;
			ky = (int) iy;
			kz = (int) iz;
			kx = (int) ix;
			if (kz>n1/2) {kz = kz-n1; }
			if (ky>n1/2) {ky = ky-n1; }
			if (kx>n1/2) {kx = kx-n1; }

			k2 = kz*kz + ky*ky + kx*kx;
			bin  = (int) floor(sqrt(k2)) 	;

			spectrum_private[bin] += pow(abs(ft[kdx]),2);

		}// END LOOP


		#pragma omp critical
		{
			for(int n=0; n<powmax; n++)
			{
				//binarray[n] += spectrumK_private[n];
				binarray[powmax + n] += spectrum_private[n];
      }
		}

	}//parallel

	for(int n=0; n<powmax; n++)
	{
		////binarray[n] *= norma;
		// binarray[powmax   + n] = (double) powmax   + n;
		// binarray[powmax*2 + n] = (double) 2*powmax   + n;
		 binarray[powmax + n] *= norma;
	}

	MPI_Reduce(&binarray[powmax], &binarray[0], powmax, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

}

void	powerspectrumexpitheta(Scalar *axion)
{

	const size_t n1 = axion->Length();
	const size_t Lz = axion->Depth();
	const size_t Tz = axion->TotalSize();
	const int kmax = n1/2 -1;
	int powmax = floor(1.733*kmax)+2 ;
	const double delta = sizeL/sizeN;
	const int fSize = axion->DataSize();
	// SETS 1 ARRAY TO ZERO AT THE BEGGINING OF M
	//memset (axion->mCpu(), 0, 2*fSize*powmax);
	for(size_t dix=0; dix < 2*fSize*powmax; dix++)
	{
		static_cast<double*> (axion->mCpu())[dix] = 0. ;
	}

	double mass2 = axionmass2((*axion->zV()), nQcd, zthres, zrestore)*(*axion->zV())*(*axion->zV());
	double eRes[10];
	size_t n3 = axion->Size();
	size_t n2 = axion->Surf();
	double z_now  = (*axion->zV())	;
	// COPIES Exp[iTheta] into M2 without ghost

	if (axion->Precision() == FIELD_SINGLE)
	{
			if ( axion->Field() == FIELD_SAXION)
			{
					if (!(axion->LowMem()))
					{
						complex<float> *mIn  = static_cast<complex<float>*> (axion->mCpu())  + axion->Surf();
						complex<float> *mOut = static_cast<complex<float>*> (axion->m2Cpu()) + axion->Surf();
						#pragma omp parallel for default(shared) schedule(static)
						for (size_t idx=0; idx < n3; idx++)
						{
							mOut[idx] = mIn[idx]/abs(mIn[idx]);
						}
					}
					else
					{
						LogError("No m2 in lowmem ... stop\n");
						return;
					}
			}
			else // ( axion->Field() == FIELD_AXION)
			{
				float *mIn  = static_cast<float*> (axion->mCpu()) + axion->Surf();
				complex<float> *mOut = static_cast<complex<float>*> (axion->m2Cpu()) + axion->Surf();
				#pragma omp parallel for default(shared) schedule(static)
				for (size_t idx=0; idx < n3; idx++)
				{
					float theta = mIn[idx]/z_now ;
					mOut[idx] = std::complex<float>(cos(theta), sin(theta))	;
				}
			}
			//	FFT m2 inplace ->
		auto &myPlan = AxionFFT::fetchPlan("pSpectrum");
		myPlan.run(FFT_BCK);
	}
	else {LogError("No double precision yet!\n");return ;  }

	switch(axion->Precision())
	{
		case FIELD_DOUBLE:
		BinFT2expitheta<double>(static_cast<const complex<double>*>(axion->m2Cpu()) + axion->Surf(),
		static_cast<double*>(axion->mCpu()), n1, Lz, Tz, powmax, kmax, mass2);
		break;

		case FIELD_SINGLE:
		BinFT2expitheta<float>(static_cast<const complex<float>*>(axion->m2Cpu()) + axion->Surf(),
		static_cast<double*>(axion->mCpu()), n1, Lz, Tz, powmax, kmax, mass2);
		break;

		default:
		printf ("Not a valid precision.\n");
		break;
	}
}
