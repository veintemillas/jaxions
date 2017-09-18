//#include "scalar/scalarField.h"

// void	WKBUNFOLDED(Scalar *axion, void *spectrumK, double zend, const double nnQCD, const double length);
// //this function takes evolves theta by an WKB approximation until zend;
// //the reult is stored in m2 in the format ma(zend)*pis + i*psi'
// //where psi=theta*z
// //the number spectrum resulting from the WKB is stored in spectrumK
// //it is up to a factor t_1^2/f_a^2

#ifndef	_WKB_CLASS_
	#define	_WKB_CLASS_
  #include <cmath>
  #include <complex>
  #include <cstring>

  #include "scalar/scalarField.h"
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

    const size_t n1;
    const size_t n3;
		const size_t rLx, Ly, Lz, hLy, hLz, hTz, Tz, nModes, kMax, powMax;

    const double zini ;
		const double amasszini2 = axionmass2(zini, nQcd, zthres, zrestore)*zini*zini ;

    double gg1 ;
    double delta ;



    double mass2z ;
    double mass2M ;

		double mass2M ;

		FieldPrecision fPrec ;

    int powmax;
    const int kmax;

    template <typename Float>
    void propP(const complex<Float> m2, const complex<Float> v2, complex<Float> &m, complex<Float> &v, double k2, double zend)
    {
			//alarm
			double amasszend2 = axionmass2(zini, nQcd, zthres, zrestore)*zend*zend ;
			double omgI   = sqrt(k2 + amasszini2);
			double omega  = sqrt(k2 + amasszend2);
      //alarm

			// old stuff
      // double omgPrI = 9./2.*(2.+nQcd)*pow(zini,nQcd+1.)/omgI;
      // double omgPr  = 9./2.*(2.+nQcd)*pow(zend,nQcd+1.)/omega;
      // double ooI = sqrt(omgI/omega);
      // double ooPr = omgPr/(2.*omega);
			// // these are essentially the derivatives of omega = sqrt{k2+axionmass2*z^2} -> (d axionmass2*z^2/dz )1/omega
		  double omgPrI = 0.5*(2.+nQcd)*amasszini2/(omgI*zini);
	    double omgPr  = 0.5*(2.+nQcd)*amasszend2/(omega*zend);
			// useful variables?
			double ooI = sqrt(omgI/omega);
      double ooPr = omgPr/(2.*omega);

			double mRe = double real(m2)	;
			double mIm = double imag(m2)	;
			double vRe = double real(v2)	;
			double vIm = double imag(v2)	;

      double tan1 = -1./omgI*(vRe/mRe + 0.5*omgPrI/omgI);
      double tan2 =  1./omgI*(vIm/mIm + 0.5*omgPrI/omgI);

      //this is a dummy version to test without gsl available
      //it does not produce the right physics!!
      //once GSL is available remove the next two lines & uncomment the next block
      //you also need to change gg1 in the constructor!!
      // Float phi;
      // phi = 6./(4.+nQCD)*(pow(zz, 2.+nQCD/2.) - pow(zM, 2.+nQCD/2.));


      //THIS IS THE RIGHT STUFF
      // the phase is the integral of omega
			// if k2 = 0 is analitical
			// integral of mass*z -> (mass*z^2)/(nQcd/2+2)
      Float phi;
      if (mom == 0.)
        {
        //phi = 6./(4.+nQcd)*(pow(zend, 2.+nQcd/2.) - pow(zini, 2.+nQcd/2.));
				phi = (axionmass(zend, nQcd, zthres, zrestore)*zend*zend - axionmass(zini, nQcd, zthres, zrestore)*zend*zend )/(2.+nQcd/2.)
        }

      else
        {
        //alarm
        //Float argi = 9.*pow(zini,nQcd+2.)/mom;
        //Float argz = 9.*pow(zend,nQcd+2.)/mom;
				double argi = amasszini2/k2 ;
        double argz = amasszend2/k2 ;

        Float f21zi, f21zz;
        //alarm indi
        if(argi< 1.)
          f21zi = gsl_sf_hyperg_2F1(1./2., 1./(2.+nQcd), 1.+1./(2.+nQcd), -1.*argi );
        else
          f21zi = -2./pow(argi+1.,0.5)/nQcd*gsl_sf_hyperg_2F1(0.5, 1., 3./2.-1./(2.+nQcd), 1./(argi+1.))
            +1./pow(argi+1., 1./(nQcd+2.))*gg1*gsl_sf_hyperg_2F1(1./(2.+nQcd), 0.5+1./(2.+nQcd), 0.5+1./(2.+nQcd),1./(argi+1));

        if(argz< 1.)
          f21zz = gsl_sf_hyperg_2F1(1./2., 1./(2.+nQcd), 1.+1./(2.+nQcd), -1.*argz );
        else
          f21zz = -2./pow(argz+1.,0.5)/nQcd*gsl_sf_hyperg_2F1(0.5, 1., 3./2.-1./(2.+nQcd), 1./(argz+1.))
            +1./pow(argz+1., 1./(nQcd+2.))*gg1*gsl_sf_hyperg_2F1(1./(2.+nQcd), 0.5+1./(2.+nQcd), 0.5+1./(2.+nQcd),1./(argz+1));

        phi = 2./(4.+nQcd)*(zend*sqrt(9.*pow(zend,nQcd+2.)+mom) - zini*sqrt(9.*pow(zini,nQcd+2.)+mom))
          +(nQcd+2.)/(nQcd+4.)*sqrt(mom)*(zend*f21zz - zini*f21zi);

        }

      //transfere functions
      Float t1re = ooI*(cos(phi) - tan1*sin(phi));
      Float t1im = ooI*(cos(phi) + tan2*sin(phi));
      Float t2re = -1.*ooI*(cos(phi)*(omega*tan1+ooPr) + sin(phi)*(omega-ooPr*tan1));
      Float t2im =     ooI*(cos(phi)*(omega*tan2-ooPr) - sin(phi)*(omega+ooPr*tan2));

      //save real part
      if(mRe == 0.)
        {
    mRe = vRe/sqrt(omega*omgI)*sin(phi)/static_cast<Float>(n3);
    vRe = vRe/sqrt(omega*omgI)*(omega*cos(phi) - ooPr*sin(phi))/static_cast<Float>(n3);
        }
      else{
        mRe = t1re*mRe/static_cast<Float>(n3);
        vRe = t2re*mRe/static_cast<Float>(n3);
      }

      //save imaginary part
      if(mIm==0.)
        {
    mIm = vIm/sqrt(omega*omgI)*sin(phi)/static_cast<Float>(n3);
    vIm = vIm/sqrt(omega*omgI)*(omega*cos(phi) - ooPr*sin(phi))/static_cast<Float>(n3);
        }
      else{
        mIm = t1im*mIm/static_cast<Float>(n3);
        vIm = t2im*mIm/static_cast<Float>(n3);
      }
  };


  public:
		// constructor?
    WKB(Scalar* axion, Scalar* axion2);


		// THIS FUNCTION COMPUTES THE FFT COEFFICIENTS AT A TIME newz > zini
		// hay que definir cFloat, aunque se usa poco, casi todas las operaciones son en double, por si acaso
		// de momento curro la version float
    void doWKB(double newz)
    {
      printf("WKBing... ");

			double amasszend2 = axionmass2(zend, nQcd, zthres, zrestore)

			if (precision = FIELD_SINGLE)
			{
				// las FT estan en Axion2 [COMPLEX & TRANSPOSED_OUT], defino punteros
				float	      	 *mA2  = static_cast<float *>      (axion2->mCpu()) ;
				float	      	 *vA2  = static_cast<float *>      (axion2->vCpu()) ;
				// las FT[newz] las mando a axion[m2] y v
				//
				complex<float> *m2A1C = static_cast<complex<float>*>(axion->m2Cpu());
				complex<float> *vA1C  = static_cast<complex<float>*>(axion->vCpu());
				//
				// tambien necesitare punteros float a m y v de axion1
				// para copiar el resultado final
				float	      	 *mA1  = static_cast<float *>      (axion->mCpu()+axion->Surf()) ;
				float	      	 *vA1  = static_cast<float *>      (axion->vCpu()) ;

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

							if (kx > hLx) kx -= static_cast<int>(Lx);
							if (ky > hLy) ky -= static_cast<int>(Ly);
							if (kz > hTz) kz -= static_cast<int>(Tz);

							double k2    = kx*kx + ky*ky + kz*kz;

							//if (spectral)
								k2 *= (4.*M_PI*M_PI)/(sizeL*sizeL);
							//else
							//	k2  = cosTable[abs(kx)] + cosTable[abs(ky)] + cosTable[abs(kz)];

							// initial conditions for a given mode
							// psi  mA2[idx];
							// psi' vA2[idx];

							//

							double amasszend2 = axionmass2(zini, nQcd, zthres, zrestore)*zend*zend ;
							double omgI   = sqrt(k2 + amasszini2);
							double omega  = sqrt(k2 + amasszend2);

							double omgPrI = 0.5*(2.+nQcd)*amasszini2/(omgI*zini);
							double omgPr  = 0.5*(2.+nQcd)*amasszend2/(omega*zend);

							// useful variables?
							double ooI = sqrt(omgI/omega);
							double ooPr = omgPr/(2.*omega);

							double mRe = double real(m2)	;
							double mIm = double imag(m2)	;
							double vRe = double real(v2)	;
							double vIm = double imag(v2)	;

							double tan1 = -1./omgI*(vRe/mRe + 0.5*omgPrI/omgI);
							double tan2 =  1./omgI*(vIm/mIm + 0.5*omgPrI/omgI);

											Float phi;
											if (mom == 0.)
												{
												//phi = 6./(4.+nQcd)*(pow(zend, 2.+nQcd/2.) - pow(zini, 2.+nQcd/2.));
												phi = (axionmass(zend, nQcd, zthres, zrestore)*zend*zend - axionmass(zini, nQcd, zthres, zrestore)*zend*zend )/(2.+nQcd/2.)
												}

											else
												{
												//alarm
												//Float argi = 9.*pow(zini,nQcd+2.)/mom;
												//Float argz = 9.*pow(zend,nQcd+2.)/mom;
												double argi = amasszini2/k2 ;
												double argz = amasszend2/k2 ;

												Float f21zi, f21zz;
												//alarm indi
												if(argi< 1.)
													f21zi = gsl_sf_hyperg_2F1(1./2., 1./(2.+nQcd), 1.+1./(2.+nQcd), -1.*argi );
												else
													f21zi = -2./pow(argi+1.,0.5)/nQcd*gsl_sf_hyperg_2F1(0.5, 1., 3./2.-1./(2.+nQcd), 1./(argi+1.))
														+1./pow(argi+1., 1./(nQcd+2.))*gg1*gsl_sf_hyperg_2F1(1./(2.+nQcd), 0.5+1./(2.+nQcd), 0.5+1./(2.+nQcd),1./(argi+1));

												if(argz< 1.)
													f21zz = gsl_sf_hyperg_2F1(1./2., 1./(2.+nQcd), 1.+1./(2.+nQcd), -1.*argz );
												else
													f21zz = -2./pow(argz+1.,0.5)/nQcd*gsl_sf_hyperg_2F1(0.5, 1., 3./2.-1./(2.+nQcd), 1./(argz+1.))
														+1./pow(argz+1., 1./(nQcd+2.))*gg1*gsl_sf_hyperg_2F1(1./(2.+nQcd), 0.5+1./(2.+nQcd), 0.5+1./(2.+nQcd),1./(argz+1));

												phi = 2./(4.+nQcd)*(zend*sqrt(9.*pow(zend,nQcd+2.)+mom) - zini*sqrt(9.*pow(zini,nQcd+2.)+mom))
													+(nQcd+2.)/(nQcd+4.)*sqrt(mom)*(zend*f21zz - zini*f21zi);

												}


												//transfere functions
												Float t1re = ooI*(cos(phi) - tan1*sin(phi));
												Float t1im = ooI*(cos(phi) + tan2*sin(phi));
												Float t2re = -1.*ooI*(cos(phi)*(omega*tan1+ooPr) + sin(phi)*(omega-ooPr*tan1));
												Float t2im =     ooI*(cos(phi)*(omega*tan2-ooPr) - sin(phi)*(omega+ooPr*tan2));

												//save real part
												if(mRe == 0.)
													{
											mRe = vRe/sqrt(omega*omgI)*sin(phi)/static_cast<Float>(n3);
											vRe = vRe/sqrt(omega*omgI)*(omega*cos(phi) - ooPr*sin(phi))/static_cast<Float>(n3);
													}
												else{
													mRe = t1re*mRe/static_cast<Float>(n3);
													vRe = t2re*mRe/static_cast<Float>(n3);
												}

												//save imaginary part
												if(mIm==0.)
													{
											mIm = vIm/sqrt(omega*omgI)*sin(phi)/static_cast<Float>(n3);
											vIm = vIm/sqrt(omega*omgI)*(omega*cos(phi) - ooPr*sin(phi))/static_cast<Float>(n3);
													}
												else{
													mIm = t1im*mIm/static_cast<Float>(n3);
													vIm = t2im*mIm/static_cast<Float>(n3);
												}

					}
				}
			}
			else
			{
				//precision double not supported yet
			}
		  printf("Transfermatrix build completed from %f until %f\n", zini, zend);
		} // END DOWKB





  }


  };

#endif
