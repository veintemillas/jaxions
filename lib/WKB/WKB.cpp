#include <cmath>
#include <complex>
#include <cstring>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>

#include "WKB/WKB.h"
#include "scalar/folder.h"
#include "scalar/scalar.h"

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
		// } else {
			return z*gsl_sf_hyperg_2F1(a, b, c, 1.0-z) ;
		} else {
			return 0. ;
		}
	}

	WKB::WKB(Scalar *field, Scalar *tmp): field(field), tmp(tmp), Ly(field->Length()), Lz(field->Depth()), zIni((*field->zV())), fPrec(field->Precision()),
					      Tz(field->TotalDepth()), nModes(field->eSize()/2), hLy(field->Length()/2), hLz(field->Depth()/2), hTz(field->TotalDepth()/2),
					      rLx(field->Length()/2 + 1), Sm(field->Length()*field->Depth())
	{
		if (field->Field() == FIELD_SAXION) {
			LogError("Error: WKB only available for axion/WKB fields. Ignoring request");
			return;
		 }
		LogMsg (VERB_NORMAL, "\n");
		LogMsg (VERB_NORMAL, "[WKB] Constructor");

		bool	wasFolded = field->Folded();

		Folder	*munge;

		if (wasFolded)
		{
			LogMsg (VERB_HIGH, "[WKB] Unfolding configuration!");
			munge	= new Folder(field);
			(*munge)(UNFOLD_ALL);
		}



		if (field == tmp)
			{
			// THIS 1-field CONSTRUCTOR does:
			 	// copies v to m2/1 padded
				// FFT in place, copy to m2/2
				// plans FB in v r2C
				// copies m to m2/1 padded
				// FFT in place
				// plans FB in v r2C
				// -- FT of m and v in m2
				// computes WKB in m2 to m and v
				// FFT m2/1 to m
				// FFT m2/2 to v

				LogMsg(VERB_NORMAL, "[WKB] WKB in m2 saving mode!");

				char *mOr = static_cast<char *>(field->mStart());
				char *vOr = static_cast<char *>(field->vCpu());

				char *mDt = static_cast<char *>(field->mCpu());
				char *vDt = static_cast<char *>(field->vCpu());     /*note that it is the same*/

				char *m2 = static_cast<char *>(field->m2Cpu());
				char *m2h = static_cast<char *>(field->m2half());

				const size_t	dataLine = field->DataSize()*Ly;
				const size_t	padLine  = field->DataSize()*(Ly+2);

				size_t dataTotalSize2 = (field->Precision())*(field->eSize());

				auto &myPlan = AxionFFT::fetchPlan("pSpecAx");

				LogMsg(VERB_NORMAL, "copying v -> m2/1 ");
				//Copy m -> m2 with padding
				#pragma omp parallel for schedule(static)
				for (int sl=0; sl<Sm; sl++) {
					auto	oOff = sl*field->DataSize()*Ly;
					auto	fOff = sl*field->DataSize()*(Ly+2);
					memcpy	(m2+fOff, vOr+oOff, dataLine);
				}

					LogMsg(VERB_NORMAL, "Planning in v ... ");
					AxionFFT::initPlan (field, FFT_RtoC_VtoV_WKB,  FFT_FWDBCK, "WKB v");
					LogMsg(VERB_NORMAL, "done!");

						LogMsg(VERB_NORMAL," FFTWing v in m2 inplace ... ");
						myPlan.run(FFT_FWD);
						LogMsg(VERB_NORMAL,"done!!");

						  LogMsg(VERB_HIGH," chech precision %d and datasize %d %ld",field->Precision(),field->DataSize(), dataTotalSize2);
							LogMsg(VERB_NORMAL, "copying m2/1 -> m2/2 ");
							memmove	(m2h, m2, dataTotalSize2);
							LogMsg(VERB_NORMAL,"done!!");

				LogMsg(VERB_NORMAL, "copying m -> m2 ");
				//Copy m -> m2 with padding
				#pragma omp parallel for schedule(static)
				for (int sl=0; sl<Sm; sl++) {
					auto	oOff = sl*field->DataSize()*Ly;
					auto	fOff = sl*field->DataSize()*(Ly+2);
					memcpy	(m2+fOff, mOr+oOff, dataLine);
				}
				LogMsg(VERB_NORMAL, "done!");

					LogMsg(VERB_NORMAL, "Planning in m ... ");
					AxionFFT::initPlan (field, FFT_RtoC_MtoM_WKB,  FFT_FWDBCK, "WKB m");
					LogMsg(VERB_NORMAL, "done!");

						LogMsg(VERB_NORMAL," FFTWing m in m2 inplace ... ");
						myPlan.run(FFT_FWD);
						LogMsg(VERB_NORMAL,"done!!");

						LogMsg(VERB_NORMAL, "FTs of m and v (time=%f) set in m2 ",zIni);
						LogMsg(VERB_NORMAL, "      - - ->   ready to WKBonce! \n");
				LogOut ("      - - ->   ready to WKBonce! \n");

			}
			else
			{
				// THIS CONSTRUCTOR COPIES M1, V1 INTO M2, V2 OF AN AXION 2 AND COMPUTES FFT INPLACE TRASPOSED_OUT
				// PREPARES THE FFT IN AXION2 TO BUILD THE FIELD AT ANY OTHER TIME
				// THIS IS DONE WITH THE FUNCITONS DEFINED IN WKB.h

				LogMsg(VERB_NORMAL, "Planning in axion2 ... ");
				AxionFFT::initPlan (tmp, FFT_RtoC_MtoM_WKB,  FFT_FWD, "WKB m");
				AxionFFT::initPlan (tmp, FFT_RtoC_VtoV_WKB,  FFT_FWD, "WKB v");
				LogMsg(VERB_NORMAL, "done!");

				// pointers for copying 1->2
				char *mOr = static_cast<char *>(field->mCpu()) + field->Surf()*field->DataSize();
				char *vOr = static_cast<char *>(field->vCpu());

				char *mDt = static_cast<char *>(tmp->mCpu());
				char *vDt = static_cast<char *>(tmp->vCpu());

							// // input=output check
							// float	*mIno  = static_cast<float *>(field->mCpu()) + field->Surf();
							// float	*vIno  = static_cast<float *>(field->vCpu()) ;
							// LogOut ("\n  --> points %e %e %e !\n ", mIno[0],mIno[1],mIno[2]);
							// LogOut ("  --> points %e %e %e !\n\n", mIno[field->Size()-1],mIno[field->Size()-2],mIno[field->Size()-3]);
							// LogOut ("  --> voints %e %e %e !\n ",  vIno[0],vIno[1],vIno[2]);
							// LogOut ("  --> voints %e %e %e !\n\n", vIno[field->Size()-1],vIno[field->Size()-2],vIno[field->Size()-3]);

				// note Lz can be different from n1 if running MPI
				size_t dataLine = field->DataSize()*Ly;

				LogMsg(VERB_NORMAL, "copying 1->2 ");
				//Copy m,v -> m2,v2 with padding
				#pragma omp parallel for schedule(static)
				for (int sl=0; sl<Sm; sl++) {
					auto	oOff = sl*field->DataSize()*Ly;
					auto	fOff = sl*field->DataSize()*(Ly+2);
					memcpy	(mDt+fOff, mOr+oOff, dataLine);
					memcpy	(vDt+fOff, vOr+oOff, dataLine);
				}
				LogMsg(VERB_NORMAL, "done!");

				auto &myPlanM = AxionFFT::fetchPlan("WKB m");
				auto &myPlanV = AxionFFT::fetchPlan("WKB v");

				LogMsg(VERB_NORMAL," FFTWing AXION2 m inplace ... ");
				myPlanM.run(FFT_FWD);
				LogMsg(VERB_NORMAL,"done!!");

				LogMsg(VERB_NORMAL," FFTWing AXION2 v inplace ... ");
				myPlanV.run(FFT_FWD);
				LogMsg(VERB_NORMAL,"done!! ");

				LogMsg(VERB_NORMAL,"Planning IN axion1 m2 ");
				AxionFFT::initPlan (field, FFT_RtoC_M2toM2_WKB, FFT_BCK, "WKB p");	// Momenta coefficients
				LogMsg(VERB_NORMAL,"done!!");

				LogOut ("      - - ->   ready to WKB! \n");

			}


	};

	// THIS FUNCTION COMPUTES THE FFT COEFFICIENTS AT A TIME newz > zini
	void WKB::operator()(double zEnd) {
		if (field->Field() == FIELD_SAXION) {
			LogError("Error: WKB only available for axion/WKB fields. Ignoring request");
			return;
		}

		switch (fPrec) {
			case FIELD_SINGLE:
			if (field == tmp) {
				doWKBinplace<float> (zEnd);
				}
			else{
				doWKB<float> (zEnd);
			}
			break;

			case FIELD_DOUBLE:
			if (field == tmp) {
				doWKBinplace<double> (zEnd);
				}
			else{
				doWKB<double> (zEnd);
			}
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
		double aMass2zIni2 = field->AxionMassSq(zIni)*zIni*zIni ;
		double aMass2zEnd2 = field->AxionMassSq(zEnd)*zEnd*zEnd ;
		double aMass2zIni1 = aMass2zIni2/zIni;
		double aMass2zEnd1 = aMass2zEnd2/zEnd;
		double nQcd	   = field->BckGnd()->QcdExp();
		double zBase1      = 0.25*(nQcd+2.)*aMass2zIni1;
		double zBase2      = 0.25*(nQcd+2.)*aMass2zEnd1;
		double phiBase1	   = 2.*zIni/(4.+nQcd);
		double phiBase2	   = 2.*zEnd/(4.+nQcd);
		double n2p1        = 1.+nQcd/2.;
		double nn1         = 1./(2.+nQcd)+0.5;
		double nn2         = 1./(2.+nQcd)+1.0;

		double lSize	   = field->BckGnd()->PhysSize();
		double minmom2	   = (4.*M_PI*M_PI)/(lSize*lSize);

									// if (firsttime)
									// {
									// 	// if (commRank() == 0)
									// 	// {
									// 		double amass2zEnd2 = axionmass2(zEnd, nQcd, zthres, zrestore)*zEnd*zEnd ;
									// 		double k2 = minmom2*(Ly*Ly) ;
									// 		double v2 = k2/(k2 + amass2zEnd2)	;
									// 		LogOut("firsttime check with am2 = %e, v2 max = %1.f\n", amass2zEnd2, 0., v2);
									// 		FILE *file_samp ;
									// 		file_samp = NULL;
									// 		file_samp = fopen("out/WKBgammatest.txt","w+");
									// 		int kmax = Ly ; // in earnest is (N/2-1)*sqrt(3) but more or less...
									// 		for (int i = 1 ; i< kmax ; i++ )
									// 		{
									// 			double k2 = minmom2*(i*i) ;
									// 			double v2 = k2/(k2 + amass2zEnd2)	;
									// 			double fs = v2h2F1(0.5, 1., nn2, v2 )	;
									// 			fprintf(file_samp,"%e %e\n", v2, fs);
									// 		}
									// 		fflush(file_samp);
									// 		fclose(file_samp);
									// 		firsttime = false ;
									// }


		// las FT estan en Axion2 [COMPLEX & TRANSPOSED_OUT], defino punteros
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

		// LogOut("test suite \n") ;
		// 		LogOut("------------\n") ;
		// 		LogOut("aMass2zIni2 %f\n",aMass2zIni2) ;
		// 		LogOut("aMass2zEnd2 %f\n",aMass2zEnd2) ;
		// 		LogOut("aMass2zIni1 %f\n",aMass2zIni1) ;
		// 		LogOut("zBase1 %f\n",zBase1) ;
		// 		LogOut("phiBase1 %f\n",phiBase1) ;
		// 		LogOut("nn1 %f\n",nn1) ;
		// 		LogOut("nn2 %f\n",nn2) ;
		// 		LogOut("------------\n") ;
		// 		LogOut("nModes %d\n",nModes) ;
		// 		LogOut("Ly %d\n",Ly) ;
		// 		LogOut("rLx %d\n",rLx) ;
		// 		LogOut("Tz %d\n",Tz) ;
		// 		LogOut("zBase %d\n",zBase) ;
		// 		LogOut("hLy %d\n",hLy) ;
		// 		LogOut("hTz %d\n",hTz) ;


		LogMsg(VERB_NORMAL,"[WKB] start mode calculation! \n");
		#pragma omp parallel for schedule(static)
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
			k2 *= minmom2;

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

			//output check
			// if ( idx%(Sm+5) == 0 )
			// {
			// 	LogOut("idx %d ",idx) ;
			// 	LogOut("(%d,%d,%d) ",kx, ky, kz) ;
			// 	LogOut("%d %.2f \n",mom, k2) ;
			// 	LogOut("w1 %.2f z1 %.2e ooI %.2f Rpha %.2f M0[%.2f,%.2f] D0[%.2f,%.2f] ap[%.2f,%.2f] am[%.2f,%.2f]\n",
			// 					w1,     zeta1,   ooI,   real(pha),real(M0),imag(M0),real(D0),imag(D0),real(ap),imag(ap),real(am),imag(am)) ;
			// }

			// propagate
			ap *= ooI*pha;
			am *= ooI*conj(pha);
			M0 = ap + am;
			D0 = ap - am + im*zeta2*M0;

			//output check
			// if ( idx%(Sm+5) == 0 )
			// {
			// 	LogOut("w2 %.2f z2 %.2e phi %.2f Ipha %.2f M0[%.2f,%.2f] D0[%.2f,%.2f] ap[%.2f,%.2f] am[%.2f,%.2f]\n",
			// 					w2,   zeta2,   phi,   imag(pha),real(M0),imag(M0),real(D0),imag(D0),real(ap),imag(ap),real(am),imag(am)) ;
			// }

			D0 *= im*w2	;

			m2IC[idx] = M0;
			vInC[idx] = D0;

			// // check if the modes are properly copied by zEnd=zIni and uncommenting this line
			// if ( idx%(Sm+5) == 0 )
			// {
			// 	//LogOut("compare[%.2f,%.2f]-[%.2f,%.2f]\n", Daux.real(), Daux.imag(), vInC[idx].real(), vInC[idx].imag() ) ;
			// 	LogOut("Mumpare[%.2f,%.2f]-[%.2f,%.2f]\n", real(Maux), imag(Maux), real(m2IC[idx]), imag(m2IC[idx])) ;
			// 	LogOut("Dompare[%.2f,%.2f]-[%.2f,%.2f]\n", real(Daux), imag(Daux), real(vInC[idx]), imag(vInC[idx]) ) ;
      //
			// }

		}

		LogMsg(VERB_NORMAL," modes evolved and copied into v and m2 ... ");
		LogMsg(VERB_NORMAL," invFFTWing AXION m2 inplace ... ");
		// FFT in place in m2 of axion1
		myPlanP.run(FFT_BCK);
		LogMsg(VERB_NORMAL,"done!!\n ");

		const size_t	dataLine = field->DataSize()*Ly;
		const size_t	padLine  = field->DataSize()*(Ly+2);

		LogMsg(VERB_NORMAL,"copying psi m2 padded -> m unpadded ");
		#pragma omp parallel for schedule(static)
		for (size_t sl=0; sl<Sm; sl++) {
			auto	oOff = sl*dataLine;
			auto	fOff = sl*padLine;
			memcpy	(mTf+oOff ,  m2Tf+fOff, dataLine);
		}
		LogMsg(VERB_NORMAL,"done!\n\n ");

		LogMsg(VERB_NORMAL,"and FT(psi_z) v->m2 ");
		memcpy	(m2Tf, vTf, field->eSize()*field->DataSize());
		LogMsg(VERB_NORMAL,"done!\n");

		LogMsg(VERB_NORMAL," invFFTWing AXION m2 inplace ... ");
		// FFT in place in m2 of axion1
		myPlanP.run(FFT_BCK);
		LogMsg(VERB_NORMAL,"done!!\n ");

		// transfer m2 into v
		LogMsg(VERB_NORMAL,"copying psi_z m2 padded -> v unpadded ");
		//Copy m,v -> m2,v2 with padding
		#pragma omp parallel for schedule(static)
		for (size_t sl=0; sl<Sm; sl++) {
			auto	oOff = sl*dataLine;
			auto	fOff = sl*padLine;
			memcpy	(vTf+oOff ,  m2Tf+fOff, dataLine);
		}
		LogMsg(VERB_NORMAL,"done!\n");

		cFloat toton = (cFloat) field->TotalSize();

		LogMsg(VERB_NORMAL,"scale x%2.2e ",toton);
		#pragma omp parallel for schedule(static)
		for (size_t idx=0; idx<field->Size(); idx++)
		{
			mIn[idx] /= toton   ;
			vIn[idx] /= toton   ;
		}
		LogMsg(VERB_NORMAL,"done!\n");


		// LogOut ("  --> points %e %e %e !\n ", mIn[0],mIn[1],mIn[2]);
		// LogOut ("  --> points %e %e %e !\n ", mIn[field->Size()-1],mIn[field->Size()-2],mIn[field->Size()-3]);
		// LogOut ("  --> voints %e %e %e !\n ", vIn[0],vIn[1],vIn[2]);
		// LogOut ("  --> voints %e %e %e !\n ", vIn[field->Size()-1],vIn[field->Size()-2],vIn[field->Size()-3]);

		*field->zV() = zEnd ;
		LogMsg(VERB_NORMAL,"[WKB] set z=%f done", (*field->zV()) );
		field->setFolded(false);
		LogMsg(VERB_NORMAL,"[WKB] m,v, set unfolded!");
		LogMsg(VERB_NORMAL,"[WKB] Complete!\n ");
	}














	template<typename cFloat>
	void	WKB::doWKBinplace(double zEnd) {

		// label 1 for ini, 2 for end
		double aMass2zIni2 = field->AxionMassSq(zIni)*zIni*zIni ;
		double aMass2zEnd2 = field->AxionMassSq(zEnd)*zEnd*zEnd ;
		double aMass2zIni1 = aMass2zIni2/zIni;
		double aMass2zEnd1 = aMass2zEnd2/zEnd;
		double nQcd	   = field->BckGnd()->QcdExp();
		double zBase1      = 0.25*(nQcd+2.)*aMass2zIni1;
		double zBase2      = 0.25*(nQcd+2.)*aMass2zEnd1;
		double phiBase1	   = 2.*zIni/(4.+nQcd);
		double phiBase2	   = 2.*zEnd/(4.+nQcd);
		double n2p1        = 1.+nQcd/2.;
		double nn1         = 1./(2.+nQcd)+0.5;
		double nn2         = 1./(2.+nQcd)+1.0;

		double lSize	   = field->BckGnd()->PhysSize();
		double minmom2 	   = (4.*M_PI*M_PI)/(lSize*lSize);

		// las FT estan en m2/1 y m2/2 [COMPLEX & TRANSPOSED_OUT], defino punteros
		std::complex<cFloat> *m2C1  = static_cast<std::complex<cFloat>*>(field->m2Cpu());
		std::complex<cFloat> *m2C2  = static_cast<std::complex<cFloat>*>(field->m2half());

		// las copiaré a m y v
		std::complex<cFloat> *mC  = static_cast<std::complex<cFloat>*>(field->mCpu());
		std::complex<cFloat> *vC  = static_cast<std::complex<cFloat>*>(field->vCpu());

		// tambien necesitare punteros float m y v de axion
		cFloat	      	 *mIn  = static_cast<cFloat *>(field->mStart());
		cFloat	      	 *vIn  = static_cast<cFloat *>(field->vCpu());
		cFloat	      	 *m2In = static_cast<cFloat *>(field->m2Cpu());


		size_t	zBase = Lz*commRank();

		LogMsg(VERB_NORMAL,"start mode calculation! \n");
		#pragma omp parallel for schedule(static)
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
			k2 *= minmom2;

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

				phi =  phiBase2*w2*(1.+n2p1*v2h2F1(0.5, 1., nn2, k2/(w2*w2)))
				      -phiBase1*w1*(1.+n2p1*v2h2F1(0.5, 1., nn2, k2/(w1*w1)));
			}

			// phasor
			complex<double> pha = exp(im*phi);

			// initial conditions of the mode
			// in principle this could be done only once...
			std::complex<cFloat> Maux, Daux ;
			Maux = m2C1[idx];
			Daux = m2C2[idx];

			std::complex<double> M0, D0, ap, am;
			double ra, ia ;

			ra = (double) real(Maux) ;
			ia = (double) imag(Maux) ;
			M0 = ra + im*ia	;
			ra = (double) real(Daux) ;
			ia = (double) imag(Daux) ;
			D0 = (ra + im*ia)/(im*w1)	;

			ap = 0.5*(M0*(1.0 - im*zeta1) + D0);
			am = 0.5*(M0*(1.0 + im*zeta1) - D0);

			// propagate
			ap *= ooI*pha;
			am *= ooI*conj(pha);
			M0 = ap + am;
			D0 = ap - am + im*zeta2*M0;

			D0 *= im*w2	;

			mC[idx] = M0;

			vC[idx] = D0;

		}

		auto &myPlanM = AxionFFT::fetchPlan("WKB m");
		auto &myPlanV = AxionFFT::fetchPlan("WKB v");

		LogMsg(VERB_NORMAL," FFTWing back AXION m inplace ... ");
		myPlanM.run(FFT_BCK);
		LogMsg(VERB_NORMAL,"done!!\n");

		LogMsg(VERB_NORMAL," FFTWing back AXION v inplace ... ");
		myPlanV.run(FFT_BCK);
		LogMsg(VERB_NORMAL,"done!!\n ");

		const size_t	dataLine = field->DataSize()*Ly;
		const size_t	padLine  = field->DataSize()*(Ly+2);

		// pointers for padding ...

		char *mTf  = static_cast<char *>(static_cast<void*>(mIn));
		char *m0Tf  = static_cast<char *>(static_cast<void*>(mC));
		char *vTf  = static_cast<char *>(static_cast<void*>(vIn));
		char *m2Tf = static_cast<char *>(static_cast<void*>(m2In));

			LogMsg(VERB_NORMAL," unpadding m in place ... ");
			LogMsg(VERB_NORMAL," unpadding (first line is not needed)");
					for (int sl=1; sl<Sm; sl++) {
						auto	oOff = sl*field->DataSize()*(Ly);
						auto	fOff = sl*field->DataSize()*(Ly+2);
						memcpy	(m0Tf+oOff, m0Tf+fOff, dataLine);
					}
			LogMsg(VERB_NORMAL," shifthing to host ghost");
			LogMsg(VERB_NORMAL," chech precision %f and datasize %f",field->Precision(),field->DataSize());
					size_t dataTotalSize = (field->Precision())*(field->Size());
					memcpy	(mTf, m0Tf, dataTotalSize);

			LogMsg(VERB_NORMAL," unpadding v in place ... ");
					for (int sl=0; sl<Sm; sl++) {
						auto	oOff = sl*field->DataSize()*(Ly);
						auto	fOff = sl*field->DataSize()*(Ly+2);
						memcpy	(vTf+oOff, vTf+fOff, dataLine);
					}

	  cFloat toton = (cFloat) field->TotalSize();

		LogMsg(VERB_NORMAL,"scale x%2.2e ",toton);
		#pragma omp parallel for schedule(static)
		for (size_t idx=0; idx<field->Size(); idx++)
		{
			mIn[idx] /= toton   ;
			vIn[idx] /= toton   ;
		}
		LogMsg(VERB_NORMAL,"done!\n");


		// LogOut ("  --> points %e %e %e !\n ", mIn[0],mIn[1],mIn[2]);
		// LogOut ("  --> points %e %e %e !\n ", mIn[field->Size()-1],mIn[field->Size()-2],mIn[field->Size()-3]);
		// LogOut ("  --> voints %e %e %e !\n ", vIn[0],vIn[1],vIn[2]);
		// LogOut ("  --> voints %e %e %e !\n ", vIn[field->Size()-1],vIn[field->Size()-2],vIn[field->Size()-3]);

    *field->zV() = zEnd ;
    LogMsg(VERB_NORMAL,"[WKB] scalar set z=%f done (m2 still in %f)", (*field->zV()), zIni);
		field->setFolded(false);
		LogMsg(VERB_NORMAL,"[WKB] m,v, set unfolded!");
		LogMsg(VERB_NORMAL,"[WKB] Complete!\n ");



	}
}
