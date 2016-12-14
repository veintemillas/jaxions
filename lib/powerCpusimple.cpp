#include <complex>

#include "scalar/scalarField.h"
#include "utils/index.h"
#include "utils/parse.h"
#include "energy/energyMap.h"

using namespace std;


//--------------------------------------------------
// NUMBER SPECTRUM
//--------------------------------------------------

template<typename Float>
void	nSpectrumUNFOLDED (const complex<Float> *ft, void *spectrumK, void *spectrumG, void *spectrumV,	int n1, int powmax, int kmax, double mass2)
{
	double voli = 1.0/((double) n1*n1*n1) ;

	#pragma omp parallel for default(shared) schedule(static)
	for (int i=0; i < powmax; i++)
	{
		(static_cast<double *> (spectrumK))[i] = 0.0;
		(static_cast<double *> (spectrumG))[i] = 0.0;
		(static_cast<double *> (spectrumV))[i] = 0.0;
	}
//	Bin power spectrum
//  KthetaPS |mode|^2 / w
// 	(c^2 + d^2)/w

//	Bin power spectrum GthetaPS
//	|mode|^2 * k^2/w
// 	(a^2 + b^2)/w

//	Bin power spectrum VthetaPS
//	|mode|^2 * m^2/w
// 	(a^2 + b^2) * m^2/w/w

//	The FFT is entangled
//	FFT[i] = (R , I) = (a[i] - d[i], b[i] + c[i])

//	BUT theta, theta_z are real so elements i and N-i are complex conjugate
//	a[i] + I b[i] = a[N-i] - I b[N-i]
//	c[i] + I d[i] = c[N-i] - I d[N-i]
//	THEREFORE
//	FFT[i]++(FFT[N-i])* = (a[i] -d[i] ++ a[N-i] -d[N-i], b[i] + c[i] ++ - b[N-i] - c[N-i])
//	FFT[i]++(FFT[N-i])* = +2(a[i], b[i]) = 2 theta_k

//	FFT[i]--(FFT[N-i])* = (a[i] -d[i] -- a[N-i] -d[N-i], b[i] + c[i] -- - b[N-i] - c[N-i])
//	FFT[i]--(FFT[N-i])* = 2(-d[i], c[i]) = 2 I (c[i], d[i]) = 2 I theta_z_k

//  Note that the Fourier coefficients of theta_z (with minus) and theta (with +) are explcitly those from an imagnary and real field

// 	Kthetak = |FFT[i]-(FTT[N-i])*|^2    /w	/8
//	Gthetak = |FFT[i]+(FTT[N-i])*|^2 k^2/w	/8
//	Vthetak = |FFT[i]+(FTT[N-i])*|^2 m^2/w	/8

	#pragma omp parallel
	{
		double spectrumK_private[powmax];
		double spectrumG_private[powmax];
		double spectrumV_private[powmax];

	
		for (int i=0; i < powmax; i++)
		{
			spectrumK_private[i] = 0.0;
			spectrumG_private[i] = 0.0;
			spectrumV_private[i] = 0.0;
		}
	
		#pragma omp parallel for default(shared)
		for (int kz = 0; kz<kmax + 1; kz++)
		{
			int bin;

			int iz = (n1+kz)%n1 ;
			int nz = (n1-kz)%n1 ;

			complex<Float> ftk, ftmk;

			for (int ky = -kmax; ky<kmax + 1; ky++)
			{
				int iy = (n1+ky)%n1 ;
				int ny = (n1-ky)%n1 ;

				for	(int kx = -kmax; kx<kmax + 1; kx++)
				{
					int ix = (n1+kx)%n1 ;
					int nx = (n1-kx)%n1 ;

					double k2 =	kx*kx + ky*ky + kz*kz;
					int bin  = (int) floor(sqrt(k2)) 	;
					k2 =	(39.47842/(sizeL*sizeL)) * k2;
					//CONTINUUM DEFINITION
					//double w = (double) sqrt(k2 + mass2);
					//LATICE DEFINITION
					double w = (double) sqrt(k2 + mass2);

					ftk = ft[ix+iy*n1+iz*n1*n1]; // Era ft2
					ftmk = conj(ft[nx+ny*n1+nz*n1*n1]);

					spectrumK_private[bin] += pow(abs(ftk - ftmk),2)/w;
					spectrumG_private[bin] += pow(abs(ftk + ftmk),2)*k2/w;
					spectrumV_private[bin] += pow(abs(ftk + ftmk),2)*mass2/w;
				}//x

			}//y
		}//z

		#pragma omp critical
		{
			for(int n=0; n<powmax; n++)
			{
				static_cast<double*>(spectrumK)[n] += spectrumK_private[n];
				static_cast<double*>(spectrumG)[n] += spectrumG_private[n];
				static_cast<double*>(spectrumV)[n] += spectrumV_private[n];
      }
		}

	}//parallel

	#pragma omp parallel for default(shared)
	for(int n=0; n<powmax; n++)
	{
		static_cast<double*>(spectrumK)[n] *= voli;
		static_cast<double*>(spectrumG)[n] *= voli;
		static_cast<double*>(spectrumV)[n] *= voli;
	}

	printf(" ... Axion spectrum printed\n");
}



//  Copies theta into m2
//	axion->thetha2m2
//	FFT m2 inplace -> [theta]_k
//	Bin power spectrum GthetaPS
//	|mode|^2 * k^2/w
//	Bin power spectrum VthetaPS
//	|mode|^2 * m^2/w

void	spectrumUNFOLDED(Scalar *axion, void *spectrumK, void *spectrumG, void *spectrumV)
{
	const int n1 = axion->Length();
	const int kmax = n1/2 -1;
	int powmax = floor(1.733*kmax)+2 ;
	//const double z = axion->zV();
	double mass2 = 9.*pow((*axion->zV()),nQcd+2);

	// 	New scheme

	//  Copies theta + I theta_z into m2
			axion->theta2m2();
	//  FFT[m2] = FFT[theta] + I*FFT[theta_z]
	//					= a + I b		 + I (c + I d)
	//	MAKE SURE SAME ORDER OF MAGNITUDE! MULTIPLY BY a clever factor?

	//	FFT m2 inplace ->
			axion->fftCpuSpectrum(1);

	switch(axion->Precision())
	{
		case FIELD_DOUBLE:
		nSpectrumUNFOLDED<double>(static_cast<const complex<double>*>(axion->m2Cpu()), spectrumK, spectrumG, spectrumV, n1, powmax, kmax, mass2);
		break;

		case FIELD_SINGLE:
		nSpectrumUNFOLDED<float>(static_cast<const complex<float>*>(axion->m2Cpu()), spectrumK, spectrumG, spectrumV, n1, powmax, kmax, mass2);
		break;

		default:
		printf ("Not a valid precision.\n");
		break;
	}
}

//--------------------------------------------------------------------------------
//					POWER SPECTRUM
//--------------------------------------------------------------------------------
//		pSpectrumUNFOLDED<double>(static_cast<const complex<double>*>(axion->m2Cpu()), spectrumK, spectrumG, spectrumV, n1, powmax, kmax);
template<typename Float>
void	pSpectrumUNFOLDED (const complex<Float> *ft, void *spectrumT, void *spectrumN, void *spectrumV,	int n1, int powmax, int kmax)
{
	double voli = 1.0/((double) n1*n1*n1) ;

	#pragma omp parallel for default(shared) schedule(static)
	for (int i=0; i < powmax; i++)
	{
		(static_cast<double *> (spectrumT))[i] = 0.0;
		(static_cast<double *> (spectrumN))[i] = 0.0;
		(static_cast<double *> (spectrumV))[i] = 0.0;
	}

	#pragma omp parallel
	{
	
		double spectrumK_private[powmax];
		double spectrumG_private[powmax];
		double spectrumV_private[powmax];

	
		for (int i=0; i < powmax; i++)
		{
			spectrumK_private[i] = 0.0;
			spectrumG_private[i] = 0.0;
			spectrumV_private[i] = 0.0;
		}	

		#pragma omp parallel for default(shared)
		for (int kz = 0; kz<kmax + 1; kz++)
		{
			int bin;

			int iz = (n1+kz)%n1 ;
			int nz = (n1-kz)%n1 ;

			complex<Float> ftk, ftmk;

			for (int ky = -kmax; ky<kmax + 1; ky++)
			{
				int iy = (n1+ky)%n1 ;
				int ny = (n1-ky)%n1 ;

				for	(int kx = -kmax; kx<kmax + 1; kx++)
				{
					int ix = (n1+kx)%n1 ;
					int nx = (n1-kx)%n1 ;

					double k2 =	kx*kx + ky*ky + kz*kz;
					int bin  = (int) floor(sqrt(k2)) 	;

					ftk = ft[ix+iy*n1+iz*n1*n1]; // Era ft2
					ftmk = conj(ft[nx+ny*n1+nz*n1*n1]);

					spectrumT_private[bin] += pow(abs(ftk + ftmk),2);
					spectrumN_private[bin] += 1.;
					spectrumV_private[bin] += pow(abs(ftk + ftmk),2);
				}//x

			}//y
		}//z

		#pragma omp critical
		{
			for(int n=0; n<powmax; n++)
			{
				static_cast<double*>(spectrumT)[n] += spectrumT_private[n];
				static_cast<double*>(spectrumN)[n]  = spectrumN_private[n];
				static_cast<double*>(spectrumV)[n] += spectrumV_private[n];
      }
		}

	}//parallel


	#pragma omp parallel for default(shared)
	for(int n=0; n<powmax; n++)
	{
		static_cast<double*>(spectrumT)[n] *= voli;
		static_cast<double*>(spectrumV)[n] *= voli;
	}

	printf(" ... power spectrum printed\n");
}

void	powerspectrumUNFOLDED(Scalar *axion, void *spectrumK, void *spectrumG, void *spectrumV, FlopCounter *fCount)
{
	const int n1 = axion->Length();
	const int kmax = n1/2 -1;
	int powmax = floor(1.733*kmax)+2 ;
	double delta = sizeL/sizeN ;
	//const double z = axion->zV();
	//double mass2 = 9.*pow((*axion->zV()),nQcd+2);

	// 	New scheme

	//  Copies energy_theta + I potential_energy_theta into m2
	// 	energyMap	(Scalar *field, const double LL, const double nQcd, const double delta, DeviceType dev, FlopCounter *fCount)
			energyMap	(axion, nQcd, delta, axion->Device(), fCount); ////// CHECKKKK!!!!

	//	FFT m2 inplace ->
			axion->fftCpuSpectrum(1);

	switch(axion->Precision())
	{
		case FIELD_DOUBLE:
		pSpectrumUNFOLDED<double>(static_cast<const complex<double>*>(axion->m2Cpu()), spectrumK, spectrumG, spectrumV, n1, powmax, kmax);
		break;

		case FIELD_SINGLE:
		pSpectrumUNFOLDED<float>(static_cast<const complex<float>*>(axion->m2Cpu()), spectrumK, spectrumG, spectrumV, n1, powmax, kmax);
		break;

		default:
		printf ("Not a valid precision.\n");
		break;
	}
}
