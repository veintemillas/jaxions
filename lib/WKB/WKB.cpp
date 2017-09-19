#include <cmath>
#include <complex>
#include <cstring>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>


#include "WKB/WKB.h"

//----------CONSTUCTOR----------
WKB::WKB(Scalar* axion, Scalar* axion2):
  Ly(axion->Length()), Lz(axion->Depth()), n3(axion->TotalSize()), v3(axion->eSize()), zini((*axion->zV())),
  delta(sizeL/Ly), fPrec(axion->Precision()), Tz(axion->TotalDepth()),
  nModes(axion->Size()), fType(axion->Field()), hLy(Ly >> 1), hLz (Lz >> 1), hTz(Tz >>1) , rLx (Ly >> 1 + 1),
  zBase (Lz*commRank())
{
  // THIS CONSTRUCTOR COPIES M1, V1 INTO M2, V2 OF AN AXION 2 AND COMPUTES FFT INPLACE TRASPOSED_OUT
  // PREPARES THE FFT IN AXION2 TO BUILD THE FIELD AT ANY OTHER TIME
  // THIS IS DONE WITH THE FUNCITONS DEFINED IN WKB.h

  LogOut ("Planning in axion2 ... ");
  // plans in axionAUX
  // destroys m2 but nothing is in there
  AxionFFT::initPlan (axion2, FFT_RtoC_MtoM_WKB,  FFT_FWD, "fftWKB_axion2_m");
  AxionFFT::initPlan (axion2, FFT_RtoC_VtoV_WKB,  FFT_FWD, "fftWKB_axion2_v");
  LogOut ("done!\n");

  // pointers for copying 1->2
  char *ma1 = static_cast<char *>(axion->mCpu())  + axion->Surf()*axion->DataSize();
  char *va1 = static_cast<char *>(axion->vCpu());

  char *ma2 = static_cast<char *>(axion2->mCpu())  ;
  char *va2 = static_cast<char *>(axion2->vCpu())  ;

  // note Lz can be different from n1 if running MPI
  size_t dataLine = axion->DataSize()*Ly;
  size_t Sm	= Ly*Lz;

  LogOut ("copying 1->2 ");
  //Copy m,v -> m2,v2 with padding
  #pragma omp parallel for schedule(static)
  for (int sl=0; sl<Sm; sl++) {
  auto	oOff = sl*axion->DataSize()*Ly;
  auto	fOff = sl*axion->DataSize()*(Ly+2);
  memcpy	(ma2+fOff, ma1+oOff, dataLine);
  memcpy	(va2+fOff, va1+oOff, dataLine);
  }
  LogOut ("done!\n");

  // LogOut ("Planning 1 ... [warning!!! this destroys original input!!]... ");
  // // plans in axionIN§
  // // destroy input but is safely copied
  // AxionFFT::initPlan (axion, FFT_RtoC_MtoM_WKB,  FFT_BCK, "fftWKB_axion1_m");
  // AxionFFT::initPlan (axion, FFT_RtoC_VtoV_WKB,  FFT_BCK, "fftWKB_axion1_v");
  // LogOut ("done!!\n");

  LogOut ("Planning IN AXION1_M2 ");
  AxionFFT::initPlan (axion, FFT_RtoC_M2toM2_WKB,  FFT_BCK, "fftWKB_axion_m2");
  auto &myPlanm21 = AxionFFT::fetchPlan("fftWKB_axion_m2");
  LogOut ("done!!\n");


  auto &myPlanm2 = AxionFFT::fetchPlan("fftWKB_axion_m2");
  auto &myPlanv2 = AxionFFT::fetchPlan("fftWKB_axion_v2");

  LogOut ("FFTWing m2,v2 inplace ... ");
  myPlanm2.run(FFT_FWD);
  myPlanv2.run(FFT_FWD);
  LogOut ("done!!\n ");
  //
  LogOut ("ready to WKB! \n");

 //  // THIS IS OLD CODE, JUST IN CASE
 //  // USING M2 MODE
 //  powmax = floor(1.733*kmax)+2;
 //  //gg1 = gsl_sf_gamma(1./2.-1./(nQcd+2.))*gsl_sf_gamma(1.+1./(nQcd+2.))/gsl_sf_gamma(1./2.);
 //  gg1 = 1.;
 //
 //  // OLD CODE, DO NOT KNOW WHY IT FAILS for 128
 //  LogOut ("gg1 = %f\n",gg1);
 //  LogOut ("Planning 2 ... ");
 //  // plans in axionAUX
 //  // destroys m2 but nothing is in there
 //  AxionFFT::initPlan (axion, FFT_RtoC_M2toM2_WKB,  FFT_FWD, "fftWKB_axion_m2");
 //  AxionFFT::initPlan (axion, FFT_RtoC_V2toV2_WKB,  FFT_FWD, "fftWKB_axion_v2");
 //  LogOut ("done!\n");
 //
 //  // pointers for copying 1->2
 //  char *ma1 = static_cast<char *>(axion->mCpu())  + axion->Surf()*axion->DataSize();
 //  char *va1 = static_cast<char *>(axion->vCpu());
 //
 //  char *ma2 = static_cast<char *>(axion->m2Cpu())  ;
 //  char *va2 = static_cast<char *>(axion->m2Cpu())  + axion->eSize()*axion->DataSize();
 //
 //  // note Lz can be different from n1 if running MPI
 //  size_t Lz = axion->Depth();
 //  size_t dataLine = axion->DataSize()*n1;
 //  size_t Sm	= n1*Lz;
 //
 //  LogOut ("copying 1->2 ");
 //  //Copy m,v -> m2,v2 with padding
 //  #pragma omp parallel for schedule(static)
 //  for (int sl=0; sl<Sm; sl++) {
 //  auto	oOff = sl*axion->DataSize()*n1;
 //  auto	fOff = sl*axion->DataSize()*(n1+2);
 //  memcpy	(ma2+fOff, ma1+oOff, dataLine);
 //  memcpy	(va2+fOff, va1+oOff, dataLine);
 //  }
 // LogOut ("done!\n");
 //
 //
 //  LogOut ("Planning 1 ... [warning!!! this destroys original input!!]... ");
 //  // plans in axionIN§
 //  // destroy input but is safely copied
 //  AxionFFT::initPlan (axion, FFT_RtoC_MtoM_WKB,  FFT_BCK, "fftWKB_axion_m1");
 //  AxionFFT::initPlan (axion, FFT_RtoC_VtoV_WKB,  FFT_BCK, "fftWKB_axion_v1");
 //  LogOut ("done!!\n");
 //
 //
 //  auto &myPlanm2 = AxionFFT::fetchPlan("fftWKB_axion_m2");
 //  auto &myPlanv2 = AxionFFT::fetchPlan("fftWKB_axion_v2");
 //
 //  // auto &myPlanm1 = AxionFFT::fetchPlan("fftWKB_axion1_m");
 //  // auto &myPlanv1 = AxionFFT::fetchPlan("fftWKB_axion1_v");
 //
 //  LogOut ("FFTWing m2,v2 inplace ... ");
 //  myPlanm2.run(FFT_FWD);
 //  myPlanv2.run(FFT_FWD);
 //  LogOut ("done!!\n ");
 //  //
 //  LogOut ("ready to WKB! \n");

};




//remember to adjust header!
