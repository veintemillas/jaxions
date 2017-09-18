#include <cmath>
#include <complex>
#include <cstring>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>


#include "WKB/WKB.h"

//----------CONSTUCTOR----------
WKB::WKB(Scalar* axion, Scalar* axion2, double zendo):
  n1(axion->Length()), n3(axion->TotalSize()), zini((*axion->zV())), zend(zendo),
  kmax(n1/2-1), delta(sizeL/n1)
{
  powmax = floor(1.733*kmax)+2;
  //gg1 = gsl_sf_gamma(1./2.-1./(nQcd+2.))*gsl_sf_gamma(1.+1./(nQcd+2.))/gsl_sf_gamma(1./2.);
  gg1 = 1.;


  LogOut ("gg1 = %f\n",gg1);
  LogOut ("Planning 2 ... ");
  // plans in axionAUX
  // destroy input but nothing is in there
  AxionFFT::initPlan (axion2, FFT_RtoC_MtoM_WKB,  FFT_FWD, "fftWKB_axion2_m");
  AxionFFT::initPlan (axion2, FFT_RtoC_VtoV_WKB,  FFT_FWD, "fftWKB_axion2_v");
  LogOut ("done!\n");

  // pointers for copying 1->2
  // note that in axion2 v starts in m[v3]
  char *ma1 = static_cast<char *>(axion->mCpu())  + axion->Surf()*axion->DataSize();
  char *va1 = static_cast<char *>(axion->vCpu());

  char *ma2 = static_cast<char *>(axion2->mCpu())  ;
  char *va2 = static_cast<char *>(axion2->mCpu())  + axion2->eSize()*axion2->DataSize();

  // note Lz can be different from n1 if running MPI
  size_t Lz = axion->Depth();
  size_t dataLine = axion->DataSize()*n1;
  size_t Sm	= n1*Lz;

  LogOut ("copying 1->pad2 [warning!!! this destroys original input!!]... ");
  // Copy m,v -> m2,v2 with padding
  #pragma omp parallel for schedule(static)
  for (int sl=0; sl<Sm; sl++) {
    auto	oOff = sl*axion->DataSize()*n1;
    auto	fOff = sl*axion->DataSize()*(n1+2);
    memcpy	(ma2+fOff, ma1+oOff, dataLine);
    memcpy	(va2+fOff, va1+oOff, dataLine);
  }
  LogOut ("done!\n");

  LogOut ("Planning 1 ... ");
  // plans in axionIN§
  // destroy input but is safely copied
  AxionFFT::initPlan (axion, FFT_RtoC_MtoM_WKB,  FFT_BCK, "fftWKB_axion1_m");
  AxionFFT::initPlan (axion, FFT_RtoC_VtoV_WKB,  FFT_BCK, "fftWKB_axion1_v");
  LogOut ("done!!\n ");


  auto &myPlanm2 = AxionFFT::fetchPlan("fftWKB_axion2_m");
  auto &myPlanv2 = AxionFFT::fetchPlan("fftWKB_axion2_v");

  auto &myPlanm1 = AxionFFT::fetchPlan("fftWKB_axion1_m");
  auto &myPlanv1 = AxionFFT::fetchPlan("fftWKB_axion1_v");

  LogOut ("FFTWing m,v in axion1 ... ");
  myPlanm1.run(FFT_BCK);
  myPlanv1.run(FFT_BCK);
  LogOut ("done!!\n ");


  LogOut ("FFTWing m,v in axion2 ... ");
  myPlanm2.run(FFT_FWD);
  myPlanv2.run(FFT_FWD);
  LogOut ("done!!\n ");
  //
  // LogOut ("ready to WKB! \n");

};




//remember to adjust header!
