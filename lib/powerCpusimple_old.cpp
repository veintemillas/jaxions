#include <complex>

#include "scalar/scalarField.h"
#include "utils/index.h"
#include "utils/parse.h"
#include "energy/energyMap.h"
#include"scalar/varNQCD.h"
#include <omp.h>

using namespace std;


//--------------------------------------------------
// NUMBER SPECTRUM
//--------------------------------------------------

template<typename Float>
void	BinSpectrumVG (const complex<Float> *ft, void *spectrumV, void *spectrumG,	int n1, int powmax, int kmax, double mass2)
{
	printf("sizeL=%f , ",sizeL);fflush(stdout);
	double norma = pow(sizeL,3.)/(4.*pow((double) n1,6.)) ;

	double minus1costab [kmax+1] ;

	double id2 = 2.0/pow(sizeL/sizeN,2);

	#pragma omp parallel for default(shared) schedule(static)
	for (int i=0; i < kmax + 2; i++)
	{
		minus1costab[i] = id2*(1.0 - cos(((double) (6.2831853071796*i)/n1)));
	}
	printf("mtab , ");fflush(stdout);
	#pragma omp parallel for default(shared) schedule(static)
	for (int i=0; i < powmax; i++)
	{
		(static_cast<double *> (spectrumK))[i] = 0.0;
		(static_cast<double *> (spectrumG))[i] = 0.0;
		(static_cast<double *> (spectrumV))[i] = 0.0;
	}
//
	 printf("0ing , ");fflush(stdout);

	#pragma omp parallel default(shared)
	{

		double spectrumG_private[powmax];
		double spectrumV_private[powmax];

		for (int i=0; i < powmax; i++)
		{
			spectrumG_private[i] = 0.0;
			spectrumV_private[i] = 0.0;
		}

		size_t idx, midx;
		int bin;
		int kz, ky, kx;
		size_t iz, nz, iy, ny, ix, nx;
		double k2, w;
		complex<Float> ftk, ftmk;

	  #pragma omp barrier

		// ASSUMES FFT TRANSPOSED kY, kZ, kX
		// SHIFTED IN RANK I.E. kZ_global = kZ_local + rank*Lz

		// FROM THAT ONE CAN GET MOMENTA IN (0,N-1)
		// AND THEN CHOOSE (-N/2+1,N/2-1)
		// AND FINALLY BIN

		// kdx 0 < n2*Lz
		// kdx = kY*n2 + kZ*n1 + kx
		// kY = kdx / n2
		// kX = 
		#pragma omp for schedule(static)
		for (int kz = 0; kz<kmax + 2; kz++)
		{

			iz = (n1+kz)%n1 ;
			nz = (n1-kz)%n1 ;

			for (int ky = -kmax; ky<kmax + 1; ky++)
			{
				iy = (n1+ky)%n1 ;
				ny = (n1-ky)%n1 ;

				for	(int kx = -kmax; kx<kmax + 1; kx++)
				{
					ix = (n1+kx)%n1 ;
					nx = (n1-kx)%n1 ;

					k2 =	kx*kx + ky*ky + kz*kz;
					bin  = (int) floor(sqrt(k2)) 	;

					//CONTINUUM DEFINITION
					//k2 =	(39.47842/(sizeL*sizeL)) * k2;
					//double w = (double) sqrt(k2 + mass2);
					//LATICE DEFINITION
					//this first instance of w is aux
					k2 =	(minus1costab[abs(kx)]+minus1costab[abs(ky)]+minus1costab[abs(kz)]);
					w = sqrt(k2 + mass2);
					//k2 =	(39.47841760435743/(sizeL*sizeL)) * k2;

					idx  = ix+iy*n1+iz*n1*n1;
					midx = nx+ny*n1+nz*n1*n1;

					ftk = ft[idx];
					ftmk = conj(ft[midx]);

					if(!(kz==0||kz==kmax+1))
					{
					// -k is in the negative kx volume
					// it not summed in the for loop so include a factor of 2
					spectrumK_private[bin] += 2.*pow(abs(ftk - ftmk),2)/w;
					spectrumG_private[bin] += 2.*pow(abs(ftk + ftmk),2)*k2/(mass2*w);		//mass2 is included
					spectrumV_private[bin] += 2.*pow(abs(ftk + ftmk),2)/w;								//mass2 is included
					}
					else
					{
					// -k is in the kz=0 so both k and -k will be summed in the loop
					spectrumK_private[bin] += pow(abs(ftk - ftmk),2)/w;
					spectrumG_private[bin] += pow(abs(ftk + ftmk),2)*k2/(mass2*w);		//mass2 is included
					spectrumV_private[bin] += pow(abs(ftk + ftmk),2)/w;								//mass2 is included
					}
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
		static_cast<double*>(spectrumK)[n] *= norma;
		static_cast<double*>(spectrumG)[n] *= norma;
		static_cast<double*>(spectrumV)[n] *= norma;
	}

	printf(" ... Axion spectrum printed\n");
}


template<typename Float>
void	BinSpectrumK (const complex<Float> *ft, void *spectrumV, void *spectrumG,	int n1, int powmax, int kmax, double mass2)
{
	printf("sizeL=%f , ",sizeL);fflush(stdout);
	double norma = pow(sizeL,3.)/(4.*pow((double) n1,6.)) ;

	double minus1costab [kmax+1] ;

	double id2 = 2.0/pow(sizeL/sizeN,2);

	#pragma omp parallel for default(shared) schedule(static)
	for (int i=0; i < kmax + 2; i++)
	{
		minus1costab[i] = id2*(1.0 - cos(((double) (6.2831853071796*i)/n1)));
	}
	printf("mtab , ");fflush(stdout);
	#pragma omp parallel for default(shared) schedule(static)
	for (int i=0; i < powmax; i++)
	{
		(static_cast<double *> (spectrumK))[i] = 0.0;
		(static_cast<double *> (spectrumG))[i] = 0.0;
		(static_cast<double *> (spectrumV))[i] = 0.0;
	}
//
	 printf("0ing , ");fflush(stdout);

	#pragma omp parallel default(shared)
	{

		int tid = omp_get_thread_num();


		double spectrumK_private[powmax];
		double spectrumG_private[powmax];
		double spectrumV_private[powmax];

		for (int i=0; i < powmax; i++)
		{
			spectrumK_private[i] = 0.0;
			spectrumG_private[i] = 0.0;
			spectrumV_private[i] = 0.0;
		}

		size_t idx, midx;
		int bin;
		int kz, ky, kx;
		size_t iz, nz, iy, ny, ix, nx;
		double k2, w;
		complex<Float> ftk, ftmk;

	  #pragma omp barrier

		#pragma omp for schedule(static)
		for (int kz = 0; kz<kmax + 2; kz++)
		{

			iz = (n1+kz)%n1 ;
			nz = (n1-kz)%n1 ;

			for (int ky = -kmax; ky<kmax + 1; ky++)
			{
				iy = (n1+ky)%n1 ;
				ny = (n1-ky)%n1 ;

				for	(int kx = -kmax; kx<kmax + 1; kx++)
				{
					ix = (n1+kx)%n1 ;
					nx = (n1-kx)%n1 ;

					k2 =	kx*kx + ky*ky + kz*kz;
					bin  = (int) floor(sqrt(k2)) 	;

					//CONTINUUM DEFINITION
					//k2 =	(39.47842/(sizeL*sizeL)) * k2;
					//double w = (double) sqrt(k2 + mass2);
					//LATICE DEFINITION
					//this first instance of w is aux
					k2 =	(minus1costab[abs(kx)]+minus1costab[abs(ky)]+minus1costab[abs(kz)]);
					w = sqrt(k2 + mass2);
					//k2 =	(39.47841760435743/(sizeL*sizeL)) * k2;

					idx  = ix+iy*n1+iz*n1*n1;
					midx = nx+ny*n1+nz*n1*n1;

					ftk = ft[idx];
					ftmk = conj(ft[midx]);

					if(!(kz==0||kz==kmax+1))
					{
					// -k is in the negative kx volume
					// it not summed in the for loop so include a factor of 2
					spectrumK_private[bin] += 2.*pow(abs(ftk - ftmk),2)/w;
					spectrumG_private[bin] += 2.*pow(abs(ftk + ftmk),2)*k2/(mass2*w);		//mass2 is included
					spectrumV_private[bin] += 2.*pow(abs(ftk + ftmk),2)/w;								//mass2 is included
					}
					else
					{
					// -k is in the kz=0 so both k and -k will be summed in the loop
					spectrumK_private[bin] += pow(abs(ftk - ftmk),2)/w;
					spectrumG_private[bin] += pow(abs(ftk + ftmk),2)*k2/(mass2*w);		//mass2 is included
					spectrumV_private[bin] += pow(abs(ftk + ftmk),2)/w;								//mass2 is included
					}
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
		static_cast<double*>(spectrumK)[n] *= norma;
		static_cast<double*>(spectrumG)[n] *= norma;
		static_cast<double*>(spectrumV)[n] *= norma;
	}

	printf(" ... Axion spectrum printed\n");
}



void	spectrumUNFOLDED(Scalar *axion, void *spectrumK, void *spectrumG, void *spectrumV)
{
	const int n1 = axion->Length();
	const int kmax = n1/2 -1;
	int powmax = floor(1.733*kmax)+2 ;
	//const double z = axion->zV();
	//double mass2 = 9.*pow((*axion->zV()),nQcd+2.);
	double mass2 = axionmass2((*axion->zV()), nQcd, 1.5, 3.0)*(*axion->zV())*(*axion->zV());

	// 	New New scheme
	//  DIFFERENT FOR SAXION AND AXION

	//--------------------------------
	// 	FIRST POTENTIAL AND GRADIENT
	//--------------------------------
	switch (axion->Field())
	{
		case FIELD_SAXION:
							// copy theta into m2
							// FFT in place
							// fill power spectrum potential and gradient
			break;

		case FIELD_AXION:

							// c_THETA AND c_THETA_z already in place
							// FFT theta to m2
							axion->FFTtheta2m2();
							// fill power spectrum potential and gradient
							switch(axion->Precision())
						{
							case FIELD_DOUBLE:
							BinSpectrumVG<double>(static_cast<const complex<double>*>(axion->m2Cpu()), spectrumV, spectrumG , n1, powmax, kmax, mass2);
							break;

							case FIELD_SINGLE:
							BinSpectrumVG<float>(static_cast<const complex<float>*>(axion->m2Cpu()), spectrumV, spectrumG, n1, powmax, kmax, mass2);
							break;

							default:
							printf ("Not a valid precision.\n");
							break;
						}

			break;

		default:
							printf("Unrecognized field type\n");
							exit(1);
			break;
	}

	//--------------------------------
	// 	SECOND KINETIC
	//--------------------------------
	switch (axion->Field())
	{
		case FIELD_SAXION:
							// copy theta into m2
							// FFT in place
							// fill power spectrum potential and gradient
			break;

		case FIELD_AXION:

							// c_THETA AND c_THETA_z already in place
							// FFT theta to m2
							axion->FFTvtheta2m2();
							// fill power spectrum potential and gradient
							switch(axion->Precision())
						{
							case FIELD_DOUBLE:
							BinSpectrumK<double>(static_cast<const complex<double>*>(axion->m2Cpu()), spectrumK, n1, powmax, kmax, mass2);
							break;

							case FIELD_SINGLE:
							BinSpectrumK<float>(static_cast<const complex<float>*>(axion->m2Cpu()), spectrumK, n1, powmax, kmax, mass2);
							break;

							default:
							printf ("Not a valid precision.\n");
							break;
						}

			break;

		default:
							printf("Unrecognized field type\n");
							exit(1);
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
	double norma = pow(sizeL,3.)/(4.*pow((double) n1,6.)) ;

	#pragma omp parallel for default(shared) schedule(static)
	for (int i=0; i < powmax; i++)
	{
		(static_cast<double *> (spectrumT))[i] = 0.0;
		(static_cast<double *> (spectrumN))[i] = 0.0;
		(static_cast<double *> (spectrumV))[i] = 0.0;
	}

	#pragma omp parallel default(shared)
	{
		int tid = omp_get_thread_num();

		double spectrumT_private[powmax];
		double spectrumN_private[powmax];
		double spectrumV_private[powmax];


		for (int i=0; i < powmax; i++)
		{
			spectrumT_private[i] = 0.0;
			spectrumN_private[i] = 0.0;
			spectrumV_private[i] = 0.0;
		}

		size_t idx, midx;
		int kz, ky, kx;
		size_t iz, nz, iy, ny, ix, nx;
		int bin;
		double k2;
		complex<Float> ftk, ftmk;

		#pragma omp for schedule(static)
		for (int kz = 0; kz<kmax + 2; kz++)
		{
			//printf("thread %d gets kz=%d\n",tid, kz);

			iz = (n1+kz)%n1 ;
			nz = (n1-kz)%n1 ;

			for (int ky = -kmax; ky<kmax + 2; ky++)
			{
				iy = (n1+ky)%n1 ;
				ny = (n1-ky)%n1 ;

				for	(int kx = -kmax; kx<kmax + 2; kx++)
				{
					ix = (n1+kx)%n1 ;
					nx = (n1-kx)%n1 ;

					k2 =	kx*kx + ky*ky + kz*kz;
					bin  = (int) floor(sqrt(k2)) 	;

					idx  = ix+iy*n1+iz*n1*n1;
					midx = nx+ny*n1+nz*n1*n1;

					ftk = ft[idx];
					ftmk = conj(ft[midx]);

					//printf("ftk(%d,%d,%d)=%f + I*%f - ftmk(%d,%d,%d)=%f + I*%f\n", iz,iy,ix,ftk.real(),ftk.imag(),nz,ny,nx,ftmk.real(),ftmk.imag());
					if(!(kz==0||kz==kmax+1))
					{
					// -k is in the negative kx volume
					// it not summed in the for loop so include a factor of 2
					spectrumT_private[bin] += 2.*((double) pow(abs(ftk + ftmk),2));
					spectrumN_private[bin] += 2.;
					spectrumV_private[bin] += 2.*((double) pow(abs(ftk - ftmk),2));
					}
					else
					{
					//printf("ID %d kz(%d) ftk(%d,%d,%d)=%f + I*%f - ftmk(%d,%d,%d)=%f + I*%f\n", tid, kz,iz,iy,ix,ftk.real(),ftk.imag(),nz,ny,nx,ftmk.real(),ftmk.imag());
					spectrumT_private[bin] += (double) pow(abs(ftk + ftmk),2);
					spectrumN_private[bin] += 1.;
					spectrumV_private[bin] += (double) pow(abs(ftk - ftmk),2);
					}
				}//x

			}//y
		}//z

		#pragma omp critical
		{
			//printf("thread %d gets %f %f %f %f %f\n", tid, spectrumT_private[0],spectrumT_private[1],spectrumT_private[2],spectrumT_private[3],spectrumT_private[4]);
			for(int n=0; n<powmax; n++)
			{
				static_cast<double*>(spectrumT)[n] += spectrumT_private[n];
				static_cast<double*>(spectrumN)[n] += spectrumN_private[n];
				static_cast<double*>(spectrumV)[n] += spectrumV_private[n];
      }
//			printf("thread %d sums %f %f %f %f %f\n", spectrumT[0],spectrumT[1],spectrumT[2],spectrumT[3],spectrumT[4]);

		}

	}//parallel


	#pragma omp parallel for default(shared)
	for(int n=0; n<powmax; n++)
	{
		static_cast<double*>(spectrumT)[n] *= norma;
		static_cast<double*>(spectrumV)[n] *= norma;
	}

	//printf(" ... power spectrum printed (L=%f, norma = %f)\n",sizeL,norma);
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

			//PATCH
			if ( axion->Field() == FIELD_SAXION)
			{
				energyMap	(axion, nQcd, delta, axion->Device(), fCount); ////// CHECKKKK!!!!
			}
			else
			{
			//ASSUMES THAT M2 FIELD WAS ALREADY CREATED BY THE DENSITY ANALYSIS PART
			//which is already normalised by the average density
			}
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
