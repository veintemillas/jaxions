#ifndef	_WKB_CLASS_
	#define	_WKB_CLASS_
  #include <cmath>
  #include <complex>
  #include <cstring>

	#include "fft/fftCode.h"
  #include "scalar/scalarField.h"
  #include "scalar/varNQCD.h"
  #include "utils/index.h"
  #include "utils/utils.h"
  #include "energy/energy.h"
  #include "fft/fftCode.h"
  #include <gsl/gsl_sf_hyperg.h>
  #include <gsl/gsl_sf_gamma.h>

  using namespace std;

  //--------------------------------------------------
  // WKB CLASS
  //--------------------------------------------------
  class WKB
  {
  private:


    const size_t n3;
		const size_t v3;
		const size_t rLx, Ly, Lz, Sm, hLy, hLz, hTz, Tz, nModes, zBase ;
		const size_t dataLine , dataLineC, datavol;

    const double zini ;
		const double delta ;
		const double amass2zini2 = axionmass2(zini, nQcd, zthres, zrestore)*zini*zini ;

		FieldPrecision fPrec ;
		FieldType fType ;

		// pointers for the axion matrices
		// I need :
		// axion1 m+surf, v, m2
		// axion2 m, v
		void *mIn  	 ;
		void *vIn    ;
		void *m2In 	 ;
		void *mAux	 ;
		void *vAux  ;


  public:
		// constructor?
    WKB(Scalar* axion, Scalar* axion2);

		// THIS FUNCTION COMPUTES THE FFT COEFFICIENTS AT A TIME newz > zini
		// hay que definir cFloat, aunque se usa poco, casi todas las operaciones son en double, por si acaso
		// de momento curro la version float
    void doWKB(double zend)
    {
      printf("WKBing... ");

			// label 1 for ini, 2 for end
			double amass2zend2 = axionmass2(zend, nQcd, zthres, zrestore)	;
			double amass2zini1 = amass2zini2/zend ;
			double amass2zend1 = amass2zend2/zini ;
			double zetabase1   = 0.25*(nQcd+2.)*amass2zini1 ;
			double zetabase2   = 0.25*(nQcd+2.)*amass2zend1 ;
			double phibase1    = 2.*zini/(4.+nQcd) 	;
			double phibase2    = 2.*zend/(4.+nQcd) 	;
			double n2p1    		 = 1.+nQcd/2. 	;
			double nn1   		 	 = 1./(2.+nQcd)+0.5 	;
			double nn2   		   = nn1 + 0.5						;



			if (fPrec = FIELD_SINGLE)
			{
				// OLD STUFF
				// // las FT estan en Axion2 [COMPLEX & TRANSPOSED_OUT], defino punteros
				// float	      	 *mA2  = static_cast<float *>      (axion2->mCpu()) ;
				// float	      	 *vA2  = static_cast<float *>      (axion2->vCpu()) ;
				// // las FT[newz] las mando a axion[m2] y v
				// //
				// complex<float> *m2A1C = static_cast<complex<float>*>(axion->m2Cpu());
				// complex<float> *vA1C  = static_cast<complex<float>*>(axion->vCpu());
				// //
				// // tambien necesitare punteros float a m y v de axion1
				// // para copiar el resultado final
				// float	      	 *mA1  = static_cast<float *>      (axion->mCpu()+axion->Surf()) ;
				// float	      	 *vA1  = static_cast<float *>      (axion->vCpu()) ;
				// float	      	 *m2A1  = static_cast<float *>     (axion->m2Cpu()) ;
				//
				// // pointers for padding
				// char *mchar1 = static_cast<char *>(axion->mCpu())  + axion->Surf()*axion->DataSize();
				// char *vchar1 = static_cast<char *>(axion->vCpu());
				// char *m2char1 = static_cast<char *>(axion->m2Cpu());




				// las FT estan en Axion2 [COMPLEX & TRANSPOSED_OUT], defino punteros
				float	      	 *mA2  = static_cast<float *>(mAux) ;
				float	      	 *vA2  = static_cast<float *>(vAux) ;
				// las FT[newz] las mando a axion[m2] y v
				//
				complex<float> *m2A1C = static_cast<complex<float>*>(m2In);
				complex<float> *vA1C  = static_cast<complex<float>*>(vIn);
				//
				// tambien necesitare punteros float a m y v de axion1
				// para copiar el resultado final
				float	      	 *mA1  = static_cast<float *>(mIn) + Ly*Ly;
				float	      	 *vA1  = static_cast<float *>(vIn) ;
				float	      	 *m2A1 = static_cast<float *>(m2In) ;

				// pointers for padding ... maybe these are not needed? we can use the void pointers?
				char *mchar1 = static_cast<char *>(static_cast<void*>(mA1));
				char *vchar1 = static_cast<char *>(vIn);
				char *m2char1 = static_cast<char *>(m2In);

				auto &myPlanm21 = AxionFFT::fetchPlan("fftWKB_axion_m2");

				size_t	zBase = Lz*commRank();

			#pragma omp parallel
			  {
					#pragma omp for schedule(static)
					for (size_t idx=0; idx<nModes; idx++)
						{
							//rLx is n1/2+1, reduced number of modes for r2c
							int kz = idx/rLx;
							int kx = idx - kz*rLx;
							int ky = kz/Tz;

							kz -= ky*Tz;
							ky += zBase;	// For MPI, transposition makes the Y-dimension smaller

							// kx can never be that large
							//if (kx > hLx) kx -= static_cast<int>(Lx);
							if (ky > hLy) ky -= static_cast<int>(Ly);
							if (kz > hTz) kz -= static_cast<int>(Tz);

							// momentum2
							int mom = kx*kx + ky*ky + kz*kz	;
							double k2    =  (double) mom		;
							k2 *= (4.*M_PI*M_PI)/(sizeL*sizeL);

							// frequencies
							double w1  = sqrt(k2 + amass2zini2);
							double w2  = sqrt(k2 + amass2zend2);
							// adiabatic parameters
							double zeta1 = zetabase1/(w1*w1*w1);
							double zeta2  = zetabase2/(w2*w2*w2);

							// useful variables?
							double ooI = sqrt(w1/w2);

							double phi ;
							// WKB phase
							if (k2 == 0.)
								{
									phi = phibase2*w2-phibase2*w1 ;
								}
								else
								{
									phi =  phibase2*w2*(1.+n2p1*gsl_sf_hyperg_2F1(1., nn1, nn2, -amass2zend2/k2 ))
									      -phibase1*w1*(1.+n2p1*gsl_sf_hyperg_2F1(1., nn1, nn2, -amass2zini2/k2 ));
								}
							// phasor
							complex<double> pha = exp(i*phi)	;

							// initial conditions of the mode
							// in principle this could be done only once...
							complex<double> M0, D0, ap, am ;
							M0 = (complex<double>) mA2[idx]	;
							D0 = (complex<double>) vA2[idx]/(i*w1)	;

							// we could save these
							ap = 0.5*(M0*(1.,-zeta1)+D0)	;
							am = 0.5*(M0*(1., zeta1)-D0)	;

							// propagate
							ap *= ooI*pha 			;
							am *= ooI*conj(pha) ;
							M0 = ap + am	;
							D0 = ap - am + i*zeta2*M0

							// save in axion1 m2
							m2A1C[idx] = (complex<float>) M0 ;
							// save in axion1 v
							vA1[idx]   = (complex<float>) i*w2*D0 ;

					}
				}

				// FFT in place in m2 of axion1
				myPlanm21.run(FFT_BCK);

						LogOut ("copying psi m2 unpadded -> m padded ");
						#pragma omp parallel for schedule(static)
						for (size_t sl=0; sl<Sm; sl++) {
						auto	oOff = sl*dataLine;
						auto	fOff = sl*dataLineC;
						memcpy	(mchar1+oOff ,  m2char1+fOff, dataLine);
						}
						LogOut ("and FT(psi_z) v->m2 ");
						memcpy	(m2char1, vchar1, datavol);
						LogOut ("done!\n");

				// FFT in place in m2 of axion1
				myPlanm21.run(FFT_BCK);

				// transfer m2 into v
						LogOut ("copying psi_z m2 padded -> v unpadded ");
						//Copy m,v -> m2,v2 with padding
						#pragma omp parallel for schedule(static)
						for (size_t sl=0; sl<Sm; sl++) {
						auto	oOff = sl*dataLine;
						auto	fOff = sl*dataLineC;
						memcpy	(vchar1+oOff ,  m2char1+fOff, dataLine);
						}
						LogOut ("done!\n");

			}
			else
			{
				//precision double not supported yet
			}



			printf("Transfermatrix build completed from %f until %f\n", zini, zend);



		} // END DOWKB





  };



#endif
