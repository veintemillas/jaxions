#include <cmath>
#include <complex>
#include <cstring>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>

#include "WKB/WKB.h"

namespace AxionWKB {

	//calculates 2F1 for z < -1
	double h2F1 (double a, double b, double c, double z) {
		if (fabs(z) < 1.) {
			return	gsl_sf_hyperg_2F1(a, b, c, z);
		} else {
			double coef1, coef2;

			coef1 = gsl_sf_gamma(c)*gsl_sf_gamma(b-a)*pow(1-z,-a)/(gsl_sf_gamma(b)*gsl_sf_gamma(c-a));
			coef2 = gsl_sf_gamma(c)*gsl_sf_gamma(a-b)*pow(1-z,-b)/(gsl_sf_gamma(a)*gsl_sf_gamma(c-b));

			return coef1*gsl_sf_hyperg_2F1(a,c-b,a-b+1.,1./(1.-z))+coef2*gsl_sf_hyperg_2F1(b,c-a,b-a+1.,1./(1.-z));
		}
	}


  //calculates v2*2F1 in the physical region
	double v2h2F1 (double a, double b, double c, double z) {
    if (abs(z) < 1) {
      // if (v2 < some interesting small value limit) {
		// 	return	simplified formula
		// }
    // else
    //{
			return z*gsl_sf_hyperg_2F1(a, b, c, 1.0-z) ;
		}
    else
    {
      return 0. ;
    }
	}

	// WKB::WKB(Scalar *field, Scalar *tmp): field(field), tmp(tmp), Ly(field->Length()), Lz(field->Depth()), Sm(Ly*Lz), zIni((*field->zV())), fPrec(field->Precision()),
	// 				      Tz(field->TotalDepth()), nModes(field->Size()), hLy(Ly/2), hLz(Lz/2), hTz(Tz/2) , rLx(Ly/2 + 1)
	WKB::WKB(Scalar *field, Scalar *tmp): field(field), tmp(tmp), Ly(field->Length()), Lz(field->Depth()), zIni((*field->zV())), fPrec(field->Precision()),
	Tz(field->TotalDepth()), nModes(field->eSize()), Sm(field->Length()*field->Depth()), hLy(field->Length()/2), hLz(field->Depth()/2),
	hTz(field->TotalDepth()/2), rLx(field->Length()/2+1)
	{
		if (field->Field() == FIELD_SAXION) {
			LogError("Error: WKB only available for axion/WKB fields. Ignoring request");
			return;
		}

		// hLy = Ly/2	;
		// hLz = Lz/2		;
		// hTz = Tz/2 ;
		// rLx = Ly/2 +1 ;

		// THIS CONSTRUCTOR COPIES M1, V1 INTO M2, V2 OF AN AXION 2 AND COMPUTES FFT INPLACE TRASPOSED_OUT
		// PREPARES THE FFT IN AXION2 TO BUILD THE FIELD AT ANY OTHER TIME
		// THIS IS DONE WITH THE FUNCITONS DEFINED IN WKB.h

		LogOut ("Planning in axion2 ... ");
		AxionFFT::initPlan (tmp, FFT_RtoC_MtoM_WKB,  FFT_FWD, "WKB m");
		AxionFFT::initPlan (tmp, FFT_RtoC_VtoV_WKB,  FFT_FWD, "WKB v");
		LogOut ("done!\n");

		// pointers for copying 1->2
		char *mOr = static_cast<char *>(field->mCpu()) + field->Surf()*field->DataSize();
		char *vOr = static_cast<char *>(field->vCpu());

		char *mDt = static_cast<char *>(tmp->mCpu());
		char *vDt = static_cast<char *>(tmp->vCpu());

		// note Lz can be different from n1 if running MPI
		size_t dataLine = field->DataSize()*Ly;

		LogOut ("copying 1->2 ");
		//Copy m,v -> m2,v2 with padding
		#pragma omp parallel for schedule(static)
		for (int sl=0; sl<Sm; sl++) {
			auto	oOff = sl*field->DataSize()*Ly;
			auto	fOff = sl*field->DataSize()*(Ly+2);
			memcpy	(mDt+fOff, mOr+oOff, dataLine);
			memcpy	(vDt+fOff, vOr+oOff, dataLine);
		}
		LogOut ("done!\n");

		auto &myPlanM = AxionFFT::fetchPlan("WKB m");
		auto &myPlanV = AxionFFT::fetchPlan("WKB v");

		LogOut (" FFTWing AXION2 m inplace ... ");
		myPlanM.run(FFT_FWD);
		LogOut ("done!!\n");

		LogOut (" FFTWing AXION2 v inplace ... ");
		myPlanV.run(FFT_FWD);
		LogOut ("done!!\n ");

		LogOut ("Planning IN axion1 m2 ");
		AxionFFT::initPlan (field, FFT_RtoC_M2toM2_WKB, FFT_BCK, "WKB p");	// Momenta coefficients
		LogOut ("done!!\n");

		LogOut (" ready to WKB! \n");
	};

	// THIS FUNCTION COMPUTES THE FFT COEFFICIENTS AT A TIME newz > zini
	void WKB::operator()(double zEnd) {
		if (field->Field() == FIELD_SAXION) {
			LogError("Error: WKB only available for axion/WKB fields. Ignoring request");
			return;
		}

		switch (fPrec) {
			case FIELD_SINGLE:
			doWKB<float> (zEnd);
			break;

			case FIELD_DOUBLE:
			doWKB<double> (zEnd);
			break;

			default:
			LogError("Error: WKB undefined precision. Nothing done.");
			break;
		}

		LogMsg(VERB_NORMAL, "Transfer matrix build completed from %f until %f", zIni, zEnd);
	}

	template<typename cFloat>
	void	WKB::doWKB(double zEnd) {

		// label 1 for ini, 2 for end
		double aMass2zIni2 = axionmass2(zIni, nQcd, zthres, zrestore)*zIni*zIni ;
		double aMass2zEnd2 = axionmass2(zEnd, nQcd, zthres, zrestore)*zEnd*zEnd ;
		double aMass2zIni1 = aMass2zIni2/zEnd;
		double aMass2zEnd1 = aMass2zEnd2/zIni;
		double zBase1      = 0.25*(nQcd+2.)*aMass2zIni1;
		double zBase2      = 0.25*(nQcd+2.)*aMass2zEnd1;
		double phiBase1	   = 2.*zIni/(4.+nQcd);
		double phiBase2	   = 2.*zEnd/(4.+nQcd);
		double n2p1        = 1.+nQcd/2.;
		double nn1         = 1./(2.+nQcd)+0.5;
		double nn2         = 1./(2.+nQcd)+1.0;

		// las FT estan en Axion2 [COMPLEX & TRANSPOSED_OUT], defino punteros
		//cFloat	 *mAux = static_cast<cFloat *>(tmp->mCpu());
		//cFloat	 *vAux = static_cast<cFloat *>(tmp->vCpu());
		std::complex<cFloat> *mAux  = static_cast<std::complex<cFloat>*>(tmp->mCpu());
		std::complex<cFloat> *vAux  = static_cast<std::complex<cFloat>*>(tmp->vCpu());

		// las FT[newz] las mando a axion[m2] y v
		std::complex<cFloat> *m2IC  = static_cast<std::complex<cFloat>*>(field->m2Cpu());
		std::complex<cFloat> *vInC  = static_cast<std::complex<cFloat>*>(field->vCpu());
		//
		// tambien necesitare punteros float a m y v de axion1
		cFloat	      	 *mIn  = static_cast<cFloat *>(field->mCpu()) + field->Surf();	// Theta ghosts
		cFloat	      	 *vIn  = static_cast<cFloat *>(field->vCpu());
		cFloat	      	 *m2In = static_cast<cFloat *>(field->m2Cpu());

		// pointers for padding ...
		char *mTf  = static_cast<char *>(static_cast<void*>(mIn));
		char *vTf  = static_cast<char *>(static_cast<void*>(vIn));
		char *m2Tf = static_cast<char *>(static_cast<void*>(m2In));

		auto &myPlanP = AxionFFT::fetchPlan("WKB p");
		size_t	zBase = Lz*commRank();

		LogOut("test suite \n") ;
		LogOut("------------\n") ;
		LogOut("aMass2zIni2 %f\n",aMass2zIni2) ;
		LogOut("aMass2zEnd2 %f\n",aMass2zEnd2) ;
		LogOut("aMass2zIni1 %f\n",aMass2zEnd2) ;
		LogOut("zBase1 %f\n",zBase1) ;
		LogOut("phiBase1 %f\n",phiBase1) ;
		LogOut("nn1 %f\n",nn1) ;
		LogOut("nn2 %f\n",nn2) ;
		LogOut("------------\n") ;
		LogOut("nModes %d\n",nModes) ;
		LogOut("Ly %d\n",Ly) ;
		LogOut("rLx %d\n",rLx) ;
		LogOut("Tz %d\n",Tz) ;
		LogOut("zBase %d\n",zBase) ;
		LogOut("hLy %d\n",hLy) ;
		LogOut("hTz %d\n",hTz) ;


    LogOut ("start mode calculation! \n");
		//#pragma omp parallel for schedule(static)
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
			size_t mom = kx*kx + ky*ky + kz*kz;
			double k2  = mom;
			k2 *= (4.*M_PI*M_PI)/(sizeL*sizeL);

			// frequencies
			double w1 = sqrt(k2 + aMass2zIni2);
			double w2 = sqrt(k2 + aMass2zEnd2);
			// adiabatic parameters
			double zeta1 = zBase1/(w1*w1*w1);
			double zeta2 = zBase2/(w2*w2*w2);

			// useful variables?
			double ooI = sqrt(w1/w2);

			double phi ;
			// WKB phase
			if (mom == 0)
				phi = phiBase2*w2 - phiBase1*w1; // I think this was wrong...
			else {
      // old Javi Alex implementation
			//	phi =  phiBase2*w2*(1.+n2p1*h2F1(1., nn1, nn2, -aMass2zEnd2/k2))
			//	      -phiBase1*w1*(1.+n2p1*h2F1(1., nn1, nn2, -aMass2zIni2/k2));
      // new Javi implementation to avoid negative arguments
      // note that the last argument is v2, but the 2F1 funciton shoul have the 1-v2
				phi =  phiBase2*w2*(1.+n2p1*v2h2F1(0.5, 1., nn2, k2/(w2*w2)))
				      -phiBase1*w1*(1.+n2p1*v2h2F1(0.5, 1., nn2, k2/(w1*w1)));
			}

			// phasor
			complex<double> pha = exp(im*phi);

			// initial conditions of the mode
			// in principle this could be done only once...
			std::complex<cFloat> Maux, Daux ;
			Maux = mAux[idx];
			Daux = vAux[idx];

			std::complex<double> M0, D0, ap, am;
			double ra, ia ;
			// M0 = (complex<double>) mAux[idx];
			// D0 = (complex<double>) vAux[idx]/(im*w1);
			//M0 = (real(Maux),imag(Maux)) ;
			ra = (double) real(Maux) ;
			ia = (double) imag(Maux) ;
			M0 = ra + im*ia	;
			ra = (double) real(Daux) ;
			ia = (double) imag(Daux) ;
			D0 = (ra + im*ia)/(im*w1)	;
			//D0 = Daux.real() + ( im * Daux.imag() ) ;

			// we could save these
			// ap = 0.5*(M0*(1.,-zeta1)+D0);
			// am = 0.5*(M0*(1., zeta1)-D0);
			ap = 0.5*(M0*(1.0 - im*zeta1) + D0);
			am = 0.5*(M0*(1.0 + im*zeta1) - D0);

			if ( idx%(Sm+5) == 0 )
			{
				LogOut("idx %d ",idx) ;
				LogOut("(%d,%d,%d) ",kx, ky, kz) ;
				LogOut("%d %.2f \n",mom, k2) ;
				LogOut("w1 %.2f z1 %.2e ooI %.2f Rpha %.2f M0[%.2f,%.2f] D0[%.2f,%.2f] ap[%.2f,%.2f] am[%.2f,%.2f]\n",
								w1,     zeta1,   ooI,   real(pha),real(M0),imag(M0),real(D0),imag(D0),real(ap),imag(ap),real(am),imag(am)) ;
			}

			// propagate
			ap *= ooI*pha;
			am *= ooI*conj(pha);
			M0 = ap + am;
			D0 = ap - am + im*zeta2*M0;

			if ( idx%(Sm+5) == 0 )
			{
				LogOut("w2 %.2f z2 %.2e phi %.2f Ipha %.2f M0[%.2f,%.2f] D0[%.2f,%.2f] ap[%.2f,%.2f] am[%.2f,%.2f]\n",
								w2,   zeta2,   phi,   imag(pha),real(M0),imag(M0),real(D0),imag(D0),real(ap),imag(ap),real(am),imag(am)) ;
			}

			D0 *= im*w2	;

			cFloat rere, imim ;
			rere = (cFloat) real(M0)	;
			imim = (cFloat) imag(M0)	;
			// save in axion1 m2
			m2IC[idx] = (rere, imim);

			rere = (cFloat) real(D0)	;
			imim = (cFloat) imag(D0)	;
			// save in axion1 v
			//Daux = (rere , imim	);
			vInC[idx] = rere +imim*(0.f,1.f);

			if ( idx%(Sm+5) == 0 )
			{
				//LogOut("compare[%.2f,%.2f]-[%.2f,%.2f]\n", Daux.real(), Daux.imag(), vInC[idx].real(), vInC[idx].imag() ) ;
				LogOut("compare[%.2f,%.2f]-[%.2f,%.2f]\n", real(Daux), imag(Daux), real(vInC[idx]), imag(vInC[idx]) ) ;
				LogOut("fumpare[%.2f,%.2f]-[%.2f,%.2f]\n", real(D0), imag(D0), rere, imim) ;
			}

			// save in axion1 m2
			//m2IC[idx] = (complex<cFloat>) (M0);
			// save in axion1 v
			//vInC[idx] = (complex<cFloat>) (im*w2*D0);


		}

    LogOut (" invFFTWing AXION m2 inplace ... ");
		// FFT in place in m2 of axion1
		myPlanP.run(FFT_BCK);
    LogOut ("done!!\n ");

		const size_t	dataLine = field->DataSize()*Ly;
		const size_t	padLine  = field->DataSize()*(Ly+2);

		// TODO Define dataLine y dataLineC
		LogOut ("copying psi m2 unpadded -> m padded ");
		#pragma omp parallel for schedule(static)
		for (size_t sl=0; sl<Sm; sl++) {
			auto	oOff = sl*dataLine;
			auto	fOff = sl*padLine;
			memcpy	(mTf+oOff ,  m2Tf+fOff, dataLine);
		}
		LogOut ("and FT(psi_z) v->m2 ");
		memcpy	(m2Tf, vTf, field->eSize()*field->DataSize());
		LogOut ("done!\n");

    LogOut (" invFFTWing AXION m2 inplace ... ");
		// FFT in place in m2 of axion1
		myPlanP.run(FFT_BCK);
    LogOut ("done!!\n ");

		// transfer m2 into v
		LogOut ("copying psi_z m2 padded -> v unpadded ");
		//Copy m,v -> m2,v2 with padding
		#pragma omp parallel for schedule(static)
		for (size_t sl=0; sl<Sm; sl++) {
			auto	oOff = sl*dataLine;
			auto	fOff = sl*padLine;
			memcpy	(vTf+oOff ,  m2Tf+fOff, dataLine);
		}
		LogOut ("done!\n");

    *field->zV() = zEnd ;
    LogOut ("set z=%f done\n", (*field->zV()) );


    double toton = 1/((double) field->TotalSize()) ;
    LogOut ("scale x%2.2e ",toton);
    #pragma omp parallel for schedule(static)
    for (size_t idx=0; idx<field->Size(); idx++)
    {
      mIn[idx] *= toton   ;
      vIn[idx] *= toton   ;
		}
    LogOut ("done!\n ");

	}
}

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
