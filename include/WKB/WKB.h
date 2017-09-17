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

    double zini ;
    double zend ;
    double gg1 ;
    double delta ;

    double mass2z ;
    double mass2M ;


    int powmax;
    const int kmax;

    template <typename Float>
    void propP(Float &mRe, Float &vRe, Float &mIm, Float &vIm, Float mom, Float mass2z, Float mass2M)
    {
      //alarm
      Float omega  = sqrt(mom + mass2z);
      Float omgI   = sqrt(mom + mass2M);
      //alarm

      Float omgPrI = 9./2.*(2.+nQcd)*pow(zini,nQcd+1.)/omgI;
      Float omgPr  = 9./2.*(2.+nQcd)*pow(zend,nQcd+1.)/omega;
      Float ooI = sqrt(omgI/omega);
      Float ooPr = omgPr/(2.*omega);

      Float tan1 = -1./omgI*(vRe/mRe + 0.5*omgPrI/omgI);
      Float tan2 =  1./omgI*(vIm/mIm + 0.5*omgPrI/omgI);

      //this is a dummy version to test without gsl available
      //it does not produce the right physics!!
      //once GSL is available remove the next two lines & uncomment the next block
      //you also need to change gg1 in the constructor!!
      // Float phi;
      // phi = 6./(4.+nQCD)*(pow(zz, 2.+nQCD/2.) - pow(zM, 2.+nQCD/2.));


      //THIS IS THE RIGHT STUFF
      //the phase
      Float phi;
      if (mom == 0.)
        {
        phi = 6./(4.+nQcd)*(pow(zend, 2.+nQcd/2.) - pow(zini, 2.+nQcd/2.));
        }

      else
        {
        //alarm
        Float argi = 9.*pow(zini,nQcd+2.)/mom;
        Float argz = 9.*pow(zend,nQcd+2.)/mom;

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

    WKB(Scalar* axion, Scalar* axion2, double zend);

    template <typename Float>
    void doWKB()
    {
      printf("WKBing... ");
  // reset power spectrum
  #pragma omp parallel
    {

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

  		double k2 = kx*kx + ky*ky + kz*kz;
  		int bin  = (int) floor(sqrt(k2));

  		//ftk = ft[ix+iy*n1+iz*n1*n1];
  		//ftmk = conj(ft[nx+ny*n1+nz*n1*n1]);

  		//disentangle
  		//correct for the multiplication factor in the fourier transforms
  		Float mRe =  0.5*real(ftk+ftmk)/mass2M;
  		Float mIm =  0.5*imag(ftk+ftmk)/mass2M;
  		Float vRe = -0.5*imag(ftk-ftmk);
  		Float vIm = -0.5*real(ftk-ftmk);

  		//evolve
  		//evolve does not update the time variable of axion -> should we change that?
  		Float mom = pow(2.*M_PI/(n1*delta),2)*(kx*kx + ky*ky + kz*kz);
  		propP(mRe, vRe, mIm, vIm, mom, mass2z, mass2M);

  		//add the new correction factor
  		//write data back
  		//ft[ix+iy*n1+iz*n1*n1] = (mRe*mass2z - vIm) + I*(mIm*mass2z + vRe);
  		//ft[nx+ny*n1+nz*n1*n1] = (mRe*mass2z + vIm) + I*(vRe - mIm*mass2z);

  		//power spectrum
  		// Float ft_out = abs((vRe + I*vIm)/zz - (mRe + I*mIm)/(zz*zz));
  		// ft_out = pow(ft_out,2);
  		// ft_out = ft_out*mom/(2.*pow(2.*M_PI,3));

  		//account for hemitian redundancy
  		// if(!(kz==0||kz==kmax))
  		//   {
  		//     ft_out = 2*ft_out;
  		//   }


  	  }//x
  	  }//y
      }//z


    }//parallel

    printf("WKB completed from %f until %f\n", zini, zend);
  }


  };

#endif
