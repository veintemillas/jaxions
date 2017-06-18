#include <cmath>
#include <complex>
#include <cstring>
//#include <gsl/gsl_sf_hyperg.h>
//#include <gsl/gsl_sf_gamma.h>

#include "scalar/scalarField.h"
#include "utils/index.h"
#include "utils/parse.h"
#include "energy/energyMap.h"

using namespace std;

//--------------------------------------------------
// WKB CLASS
//--------------------------------------------------
template <typename Float>
class WKB
{
private:
  complex<Float>* ft;
  Float* spec;

  const Float zM;
  const Float zz;
  const Float nQCD;
  const Float delta;

  const int n1;
  const int n3;
  const int kmax;
  const complex<Float> I;

  Float mass2M;
  Float mass2z;
  Float gg1;

  int powmax;

  void propP(Float &mRe, Float &vRe, Float &mIm, Float &vIm, Float mom)
{
    Float omega  = sqrt(mom + mass2z);
    Float omgI   = sqrt(mom + mass2M);
    Float omgPr  = 9./2.*(2.+nQCD)*pow(zz,nQCD+1.)/omega;
    Float omgPrI = 9./2.*(2.+nQCD)*pow(zM,nQCD+1.)/omgI;
    Float ooI = sqrt(omgI/omega);
    Float ooPr = omgPr/(2.*omega);

    Float tan1 = -1./omgI*(vRe/mRe + 0.5*omgPrI/omgI);
    Float tan2 =  1./omgI*(vIm/mIm + 0.5*omgPrI/omgI);

    //this is a dummy version to test without gsl available
    //it does not produce the right physics!!
    //once GSL is available remove the next two lines & uncomment the next block
    //you also need to change gg1 in the constructor!!
    Float phi;
    phi = 6./(4.+nQCD)*(pow(zz, 2.+nQCD/2.) - pow(zM, 2.+nQCD/2.));


    //THIS IS THE RIGHT STUFF
    //the phase
    // Float phi;
    // if (mom == 0.)
    //   {
    // 	phi = 6./(4.+nQCD)*(pow(zz, 2.+nQCD/2.) - pow(zM, 2.+nQCD/2.));
    //   }

    // else
    //   {
    // 	Float argi = 9.*pow(zM,nQCD+2.)/mom;
    // 	Float argz = 9.*pow(zz,nQCD+2.)/mom;

    // 	Float f21zi, f21zz;
    // 	if(argi< 1.)
    // 	  f21zi = gsl_sf_hyperg_2F1(1./2., 1./(2.+nQCD), 1.+1./(2.+nQCD), -1.*argi );
    // 	else
    // 	  f21zi = -2./pow(argi+1.,0.5)/nQCD*gsl_sf_hyperg_2F1(0.5, 1., 3./2.-1./(2.+nQCD), 1./(argi+1.))
    // 	    +1./pow(argi+1., 1./(nQCD+2.))*gg1*gsl_sf_hyperg_2F1(1./(2.+nQCD), 0.5+1./(2.+nQCD), 0.5+1./(2.+nQCD),1./(argi+1));

    // 	if(argz< 1.)
    // 	  f21zz = gsl_sf_hyperg_2F1(1./2., 1./(2.+nQCD), 1.+1./(2.+nQCD), -1.*argz );
    // 	else
    // 	  f21zz = -2./pow(argz+1.,0.5)/nQCD*gsl_sf_hyperg_2F1(0.5, 1., 3./2.-1./(2.+nQCD), 1./(argz+1.))
    // 	    +1./pow(argz+1., 1./(nQCD+2.))*gg1*gsl_sf_hyperg_2F1(1./(2.+nQCD), 0.5+1./(2.+nQCD), 0.5+1./(2.+nQCD),1./(argz+1));

    // 	phi = 2./(4.+nQCD)*(zz*sqrt(9.*pow(zz,nQCD+2.)+mom) - zM*sqrt(9.*pow(zM,nQCD+2.)+mom))
    // 	  +(nQCD+2.)/(nQCD+4.)*sqrt(mom)*(zz*f21zz - zM*f21zi);

    //   }

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
  WKB(Scalar* axion , void* spectrumK, double zend, const double nnQCD, const double length);
  void doWKB()
  {
    printf("\nDoing WKB... ");
// reset power spectrum
#pragma omp parallel
  {
    Float spectrumK_private[powmax];
    Float count[powmax];

    for (int i=0; i < powmax; i++)
      {
    	  spectrumK_private[i] = 0.0;
	      count[i] = 0.;
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

		double k2 = kx*kx + ky*ky + kz*kz;
		int bin  = (int) floor(sqrt(k2));

		ftk = ft[ix+iy*n1+iz*n1*n1];
		ftmk = conj(ft[nx+ny*n1+nz*n1*n1]);

		//disentangle
		//correct for the multiplication factor in the fourier transforms
		Float mRe =  0.5*real(ftk+ftmk)/mass2M;
		Float mIm =  0.5*imag(ftk+ftmk)/mass2M;
		Float vRe = -0.5*imag(ftk-ftmk);
		Float vIm = -0.5*real(ftk-ftmk);

		//evolve
		//evolve does not update the time variable of axion -> should we change that?
		Float mom = pow(2.*M_PI/(n1*delta),2)*(kx*kx + ky*ky + kz*kz);
		propP(mRe, vRe, mIm, vIm, mom);

		//add the new correction factor
		//write data back
		ft[ix+iy*n1+iz*n1*n1] = (mRe*mass2z - vIm) + I*(mIm*mass2z + vRe);
		ft[nx+ny*n1+nz*n1*n1] = (mRe*mass2z + vIm) + I*(vRe - mIm*mass2z);

		//power spectrum
		Float ft_out = abs((vRe + I*vIm)/zz - (mRe + I*mIm)/(zz*zz));
		ft_out = pow(ft_out,2);
		ft_out = ft_out*mom/(2.*pow(2.*M_PI,3));

		//account for hemitian redundancy
		if(!(kz==0||kz==kmax))
		  {
		    ft_out = 2*ft_out;
		    count[bin] += 1;
		  }

		spectrumK_private[bin] +=ft_out;
		count[bin] += 1;

	  }//x
	  }//y
    }//z

    #pragma omp critical
    {
      for(int n=0; n<powmax; n++)
	     {
	        spec[n] += spectrumK_private[n]/count[n];
    	 }
    }

  }//parallel

  printf("WKB completed from %f until %f\n", zM, zz);
}


};

//----------CONSTUCTOR----------
template <typename Float>
WKB<Float>::WKB(Scalar* axion, void* spectrumK, double zend, const double nnQCD, const double length):
  zM(static_cast<Float>(*axion->zV())), zz(static_cast<Float>(zend)), nQCD(static_cast<Float>(nnQCD)), n1(axion->Length()), n3(axion->Size()), kmax(n1/2-1), delta(static_cast<Float>(length/n1)), I((0.,1.))
{
  ft = static_cast<complex<Float>*>(axion->m2Cpu());
  spec = static_cast<Float*>(spectrumK);

  powmax = floor(1.733*kmax)+2;
  mass2M = 9.*pow(zM, nQCD+2.);
  mass2z = 9.*pow(zz, nQCD+2.);

  //THIS ALSO NEEDS TO BE CHANGED!!
  //gg1 = gsl_sf_gamma(1./2.-1./(nQCD+2.))*gsl_sf_gamma(1.+1./(nQCD+2.))/gsl_sf_gamma(1./2.);
  gg1 = 1.;

  memset(spec, 0, sizeof(Float)*powmax);

};

//----------propagation----------
//  template <typename Float>
// void WBK<Float>::propP(Float &mRe, Float &vRe, Float &mIm, Float &vIm, Float mom)



//----------full propagator----------
//template <typename Float>
//void WBK<Float>::doWKB()



 void	WKBUNFOLDED(Scalar *axion, void *spectrumK, double zend, const double nnQCD, const double length)
{

	//THIS IS THE END OF THE SIMULATION
	//double zfinal = (double) (*axion->zV());
	// zend is the end of the WKB

	//const int n1 = axion->Length();
	//const int kmax = n1/2 -1;
	//int powmax = floor(1.733*kmax)+2 ;
	//const double z = axion->zV();

	// 	New scheme

	//  Copies c_theta + I c_theta_z into m2

	// IF PQ FIELD
  axion->theta2m2();

	// IF theta FIELD
	//	axion->theta2m2axion();

	//  FFT[m2] = FFT[theta] + I*FFT[theta_z]
	//					= a + I b		 + I (c + I d)
	//	MAKE SURE SAME ORDER OF MAGNITUDE! MULTIPLY BY a clever factor?

	//	FFT m2 inplace ->

  axion->fftCpuSpectrum(1);

  switch(axion->Precision())
    {
    case FIELD_DOUBLE:
      WKB<double>* wkb;
      wkb = new WKB<double> (axion , spectrumK, zend, nnQCD, length);
      wkb->doWKB();
      break;

    case FIELD_SINGLE:
      WKB<float>* wkbF;
      wkbF = new WKB<float> (axion , spectrumK, zend, nnQCD, length);
      wkbF->doWKB();
      break;

    default:
      printf ("Not a valid precision.\n");
      break;

    }

 axion->fftCpuSpectrum(1);

}


 //remember to adjust header!
