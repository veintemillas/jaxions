#include <cmath>
#include <complex>
#include <cstring>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>

#include "WKB/WKB.h"
#include "scalar/folder.h"
#include "scalar/scalar.h"
#include <chrono>
#include <WKB/spline.h>
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

	double con2F1 (double a, double b, double c, double m2ow2) {
		if (abs(m2ow2) < 1) {
			// if (v2 < some interesting small value limit) {
			// 	return	simplified formula
		// } else {
			return gsl_sf_hyperg_2F1(a, b, c, m2ow2) ;
		} else {
			return 0. ;
		}
	}

	double WKB::calculatePhiexact(double zIni, double zEnd, double k, double nqcd, double indi3){
		// this calculates the integral
		// Integrate( sqrt(m^2 + k2) -m , z = (zIni, zEnd))
		// For m^2 = indi3**2 ct^(nqcd+2) which is the conformal mass
		// Requires modifications for massless axions
		// Requires generalisation for an arbitrary mass(R(t)) dependence

		// The integral is analytical for nqcd = 0.0
		// Can be approximated in Series in the NR and R limits
		// NR limit m^2Ini >> k2
		//  R limit m^2End << k2
		// We need to know the phase with accuracy better than 0.1 or so

		if (k == 0.0)
			return 0.0;

		if (indi3 == 0.0)
			return k*(zEnd-zIni);

		double k2, mIni, mEnd, m2Ini, m2End, wIni, wEnd;

		k2 = k*k;

		if (nqcd == 0.0){
				// log result assumes no precision problem
				mIni        = indi3*zIni ;
				mEnd        = indi3*zEnd ;
				m2Ini       = mIni*mIni ;
				m2End       = mEnd*mEnd ;
				wIni        = sqrt(k2 + m2Ini);
				wEnd        = sqrt(k2 + m2End);
				// LogOut("in %f, %f\n",log((mEnd+wEnd)/(mIni+wIni)),mEnd/(wEnd+mEnd) - mIni/(wIni+mIni));
				return 0.5*k*(log((mEnd+wEnd)/(mIni+wIni)) + mEnd/(wEnd+mEnd) - mIni/(wIni+mIni))/indi3;
				}

		mIni        = indi3*pow(zIni,nqcd/2+1) ;
		mEnd        = indi3*pow(zEnd,nqcd/2+1) ;
		m2Ini       = mIni*mIni ;
		m2End       = mEnd*mEnd ;
		wIni        = sqrt(k2 + m2Ini);
		wEnd        = sqrt(k2 + m2End);

		// double Rm = pow(k2,4/(nqcd+4));
		// int n = 1;
		// double eps  = 0.1;
		// double en   = 0.5*(nqcd+2)*(2*n+1) -1.0;
		// double RNR  = Rm * pow(Rm * eps * en,-1.0/en);
		//
		// if (RNR<zEnd)
		// 	{
		// 		// split into NR and
		// 	}
		// 	// coefficients of 1/(1+sqrt{1+alpha})
		// double coef = {	0.5,-0.125,	0.0625, -0.0390625, 0.0273438, -0.0205078, 0.0161133, -0.013092, 0.01091, -0.00927353, 0.00800896};

		/*for large WKBs it is advisable to find the moment when a simple cuadratic
		  approximaiton is quite good
			This introduces a zNR which depends on precision sought
			*/

		double v12         = k2/(k2+m2Ini);
		double v22         = k2/(k2+m2End);
		double phiBase1	   = 2.0*zIni/(4.0+nqcd);
		double phiBase2	   = 2.0*zEnd/(4.0+nqcd);
		double n2p1        = 1.0+nqcd/2.0;
		double nn2         = 1.0/(2.0+nqcd)+1.0;
		return phiBase2*wEnd*v22*( wEnd/(wEnd+mEnd) + n2p1*con2F1(0.5, 1.0, nn2, 1.0-v22) )
						-phiBase1*wIni*v12*( wIni/(wIni+mIni) + n2p1*con2F1(0.5, 1.0, nn2, 1.0-v12) );
	}



	void WKB::buildlookuptable(Scalar* axion, double zIni, double zEnd)
	{
		// momenta in this rank from nx-ny-nz > (0,n1/2+1),(zBase,zBase+n1/Ranks),(0,n1)
		// is only a bit more expensive to build it for the whole grid can be MPIed as well
		superTable.resize(powMax+1);
		superTable[0] = 0.0;
		k2Table.resize(powMax+1);
		k2Table[0] = 0.0;

		double lSize	   = axion->BckGnd()->PhysSize();
		double k0 	     = (2.0*M_PI)/(lSize);


		double zCri = axion->BckGnd()->ZThRes();
		double nqcd = axion->BckGnd()->QcdExp();
		/* Indi3 is used for adjusting above*/
		double indi3 		= axion->BckGnd()->Indi3();
		double indi3aux = indi3;

		LogMsg(VERB_NORMAL,"Buildlookuptable zIni %f zCri %f zEnd %f nqcd %f indi3 %f",zIni,zCri,zEnd,nqcd,indi3);
		// LogOut("Buildlookuptable zIni %f zCri %f zEnd %f nqcd %f indi3 %f\n ",zIni,zCri,zEnd,nqcd,indi3);

		if (zIni <= zCri && zCri < zEnd){
			indi3aux *= pow(zCri,nqcd/2);
			LogMsg(VERB_NORMAL,"transition! indi3aux %f ",indi3aux);

			#pragma omp parallel for schedule(static)
			for (size_t ik=1; ik<powMax+1; ik++)
			{
				double k = k0*(ik);
				k2Table[ik] = k*k;
				superTable[ik] = calculatePhiexact(zIni, zCri, k, nqcd,indi3) + calculatePhiexact(zCri, zEnd, k, 0.0,indi3aux);
			}
		}
		else {
			if (zCri <= zIni){
				indi3aux *= pow(zCri,nqcd/2);
				nqcd = 0.0;
			}

			LogMsg(VERB_NORMAL,"no transition! indi3aux %f nqcd %f ",indi3aux,nqcd);
			#pragma omp parallel for schedule(static)
			for (size_t ik=1; ik<powMax+1; ik++){
				double k = k0*(ik);
				k2Table[ik] = k*k;
				superTable[ik] = calculatePhiexact(zIni, zEnd, k, nqcd, indi3aux);
			}
		}

	}

	double WKB::interpolatephi(double dk, double k2){
	// linear interpolation in k2 (good for small k2 in NR limit... ) to be improved
	//dk is simply the double-version of the k-integer
	size_t in = floor(dk);
	return superTable[in] + (k2-k2Table[in])*(superTable[in+1]-superTable[in])/(k2Table[in+1]-k2Table[in]);
	}

	WKB::WKB(Scalar *field, Scalar *tmp): field(field), tmp(tmp), rLx(field->Length()/2 + 1), Ly(field->Length()), Lz(field->Depth()), Tz(field->TotalDepth()), hLy(field->Length()/2),
					      hLz(field->Depth()/2), hTz(field->TotalDepth()/2), nModes(field->eSize()/2), Sm(field->Length()*field->Depth()),
					      zIni((*field->zV())), fPrec(field->Precision())
	{
		powMax = floor(sqrt(2.*(Ly>>1)*(Ly>>1) + (Tz>>1)*(Tz>>1)))+1;

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

				//char *mDt = static_cast<char *>(field->mCpu());
				//char *vDt = static_cast<char *>(field->vCpu());     /*note that it is the same*/

				char *m2 = static_cast<char *>(field->m2Cpu());
				char *m2h = static_cast<char *>(field->m2half());

				const size_t	dataLine = field->DataSize()*Ly;
				//const size_t	padLine  = field->DataSize()*(Ly+2);

				size_t dataTotalSize2 = (field->Precision())*(field->eSize());

				auto &myPlan = AxionFFT::fetchPlan("pSpecAx");

				LogMsg(VERB_NORMAL, "copying v -> m2/1 ");
				//Copy m -> m2 with padding
				#pragma omp parallel for schedule(static)
				for (uint sl=0; sl<Sm; sl++) {
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
				for (uint sl=0; sl<Sm; sl++) {
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
				for (uint sl=0; sl<Sm; sl++) {
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
		double nQcd				 = field->BckGnd()->QcdExp();
		if (zEnd > field->BckGnd()->ZThRes() && zIni > field->BckGnd()->ZThRes())
			nQcd = 0.0;
		double zBase1      = 0.25*(nQcd+2.)*aMass2zIni1;
		double zBase2      = 0.25*(nQcd+2.)*aMass2zEnd1;
		double phiBase1	   = 2.*zIni/(4.+nQcd);
		double phiBase2	   = 2.*zEnd/(4.+nQcd);
		double n2p1        = 1.+nQcd/2.;
		//double nn1         = 1./(2.+nQcd)+0.5;
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
			if (ky > static_cast<int>(hLy)) ky -= static_cast<int>(Ly);
			if (kz > static_cast<int>(hTz)) kz -= static_cast<int>(Tz);

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
		field->updateR();
		LogMsg(VERB_NORMAL,"[WKB] set z=%f done", (*field->zV()) );
		field->setFolded(false);
		LogMsg(VERB_NORMAL,"[WKB] m,v, set unfolded!");
		LogMsg(VERB_NORMAL,"[WKB] Complete!\n ");
	}














	template<typename Float>
	void	WKB::doWKBinplace(double zEnd) {

		const auto ii = complex<Float>(1.0i);
		const auto hh = complex<Float>(0.5);
		double zC = field->BckGnd()->ZThRes();
		// builds the phase lookup table
		buildlookuptable(field, zIni, zEnd);
		LogMsg(VERB_NORMAL,"Lookup table built!");

if (commRank() == 0 ){
	FILE *file_wk ;
	file_wk = NULL;
	char base[256];
	sprintf(base, "out/lookup-%0.2f>%0.2f.txt", zIni,zEnd);
	file_wk = fopen(base,"w+");
	for (size_t i=0; i<powMax; i++)
		fprintf(file_wk,"%lf %lf\n",k2Table[i], superTable[i]);
	fclose(file_wk);
}
		tk::spline sss;
		sss.set_points(k2Table,superTable);

		LogMsg(VERB_NORMAL,"Spline built!");
		// use nQcd1 y 2
		double nQcdI				 = field->BckGnd()->QcdExp();
		double nQcdE				 = field->BckGnd()->QcdExp();

		if (zEnd > zC )
			nQcdE = 0.0;

		if (zIni >= zC )
			nQcdI = 0.0;

		Float mIni        = field->AxionMass(zIni)*zIni; //pow(zIni,nQcd/2+1) ;
		Float mEnd        = field->AxionMass(zEnd)*zEnd; //pow(zEnd,nQcd/2+1) ;
		Float m2Ini       = mIni*mIni ;
		Float m2End       = mEnd*mEnd ;

		Float aMass2zIni1 = m2Ini/zIni;
		Float aMass2zEnd1 = m2End/zEnd;

		Float zBase1      = 0.25*(nQcdI+2.)*aMass2zIni1;
		Float zBase2      = 0.25*(nQcdE+2.)*aMass2zEnd1;
		Float phiBase1	   = 2.0*zIni/(4.0+nQcdI);
		Float phiBase2	   = 2.0*zEnd/(4.0+nQcdE);

		// In normal situations nQcdE=nQcdI
		double massphaseE = 2.0*zEnd/(4.0+nQcdE)*field->AxionMass(zEnd)*zEnd;
		double massphaseI	=	-2.0*zIni/(4.0+nQcdI)*field->AxionMass(zIni)*zIni;
		double massphaseC = 0.0;
		if (zIni < zC && zC < zEnd )
		{
			massphaseC	= -2.0*zC/(4.0+nQcdE)*field->AxionMass(zC)*zC
										+2.0*zC/(4.0+nQcdI)*field->AxionMass(zC)*zC;
		}

		LogMsg(VERB_NORMAL,"Phases EIC %e %lf %lf (nqcdI %f nqcdE %f)",massphaseE,massphaseI,massphaseC,nQcdI, nQcdE);
		LogMsg(VERB_NORMAL,"Check mIni %f mEnd %f ",mIni, mEnd);

		// critical systematic are computed only once!
		complex<double> prephaD = exp(im*massphaseE);
		complex<Float> prepha = (complex<Float>) prephaD;

		prephaD = exp(im*massphaseI);
		prepha *= (complex<Float>) prephaD;

		prephaD = exp(im*massphaseC);
		prepha *= (complex<Float>) prephaD;

		double lSize	   = field->BckGnd()->PhysSize();
		double minmom2 	   = (4.*M_PI*M_PI)/(lSize*lSize);

		// las FT estan en m2/1 y m2/2 [COMPLEX & TRANSPOSED_OUT], defino punteros
		std::complex<Float> *m2C1  = static_cast<std::complex<Float>*>(field->m2Cpu());
		std::complex<Float> *m2C2  = static_cast<std::complex<Float>*>(field->m2half());

		// las copiaré a m y v
		std::complex<Float> *mC  = static_cast<std::complex<Float>*>(field->mCpu());
		std::complex<Float> *vC  = static_cast<std::complex<Float>*>(field->vCpu());

		// tambien necesitare punteros float m y v de axion
		Float	      	 *mIn  = static_cast<Float *>(field->mStart());
		Float	      	 *vIn  = static_cast<Float *>(field->vCpu());

		size_t	zBase = Lz*commRank();

double time1 = 0.0 ;
double time2 = 0.0 ;
// double time3 = 0.0 ;

// check precision interpolation
// 		double myarray[powMax] = {0.0};
// FILE *file_wk ;
// file_wk = NULL;
// if (commRank() == 0 ){
// 	char base[256];
// 	sprintf(base, "out/fullphase.txt", zIni,zEnd);
// 	file_wk = fopen(base,"w+");
// }

		LogMsg(VERB_NORMAL,"START MODE CALCULATION!");
		LogFlush();
// #pragma omp parallel for reduction(+:time1,time2,time3,myarray[:powMax]) schedule(static)
		#pragma omp parallel for reduction(+:time1,time2) schedule(static)
		for (size_t idx=0; idx<nModes; idx++)
		{
// auto start = std::chrono::steady_clock::now();

			int kz = idx/rLx;
			int kx = idx - kz*rLx;
			int ky = kz/Tz;

			kz -= ky*Tz;
			ky += zBase;	// For MPI, transposition makes the Y-dimension smaller

			// kx can never be that large
			//if (kx > hLx) kx -= static_cast<int>(Lx);
			if (ky > static_cast<int>(hLy)) ky -= static_cast<int>(Ly);
			if (kz > static_cast<int>(hTz)) kz -= static_cast<int>(Tz);

			// momentum2
			size_t mom = kx*kx + ky*ky + kz*kz;
			double k2  = mom;
			double dk  = sqrt(k2);
			k2 *= minmom2;

			// frequencies
			Float w1 = sqrt(k2 + m2Ini);
			Float w2 = sqrt(k2 + m2End);
			// adiabatic parameters
			Float zeta1 = zBase1/(w1*w1*w1);
			Float zeta2 = zBase2/(w2*w2*w2);
			// useful variables?
			Float ooI = sqrt(w1/w2);
			complex<Float> pha ;

// auto end = std::chrono::steady_clock::now();
// auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
// time1 += elapsed.count();
// start = std::chrono::steady_clock::now();

			// WKB phase


// auto start = std::chrono::steady_clock::now();
 			double phase = sss(k2);
// auto end = std::chrono::steady_clock::now();
// auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
// time1 += elapsed.count();

// start = std::chrono::steady_clock::now();
// 			double phasa = interpolatephi(dk,k2);
// end = std::chrono::steady_clock::now();
// elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
// time2 += elapsed.count();


// check precision interpolation
// double phasa = interpolatephi(dk,k2);
// if (commRank()==0)
// 	fprintf(file_wk,"%.14lf %.14lf %.14lf %e\n",k2, phase, phasa, phasa-phase);
			pha = exp(im*phase);
			pha *= prepha;



// end = std::chrono::steady_clock::now();
// elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
// time2 += elapsed.count();
// size_t mo = floor(sqrt( (double)(mom)));
// myarray[mo] += elapsed.count();

// start = std::chrono::steady_clock::now();

			// Float version
			{
auto start = std::chrono::steady_clock::now();
			// initial conditions of the mode
			std::complex<Float> Maux = m2C1[idx];
			std::complex<Float> Daux = m2C2[idx]/(ii*w1);

			std::complex<Float> ap = (Maux - Maux*ii*zeta1 + Daux)*hh;
			std::complex<Float> am = (Maux + Maux*ii*zeta1 - Daux)*hh;

			// propagate
			ap *= ooI*pha;
			am *= ooI*conj(pha);
			Maux = ap + am;
			Daux = ap - am + ii*zeta2*Maux;
			Daux *= ii*w2	;

			mC[idx] = Maux;
			vC[idx] = Daux;

auto end = std::chrono::steady_clock::now();
auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
time1 += elapsed.count();
			}
			// double version
			{
			double w2d = (double) w2;
			double w1d = (double) w1;
			double ooId = (double) ooI;
			double zeta1d = (double) zeta1;
			double zeta2d = (double) zeta2;
auto start = std::chrono::steady_clock::now();
			std::complex<Float> Maux = m2C1[idx];
			std::complex<Float> Daux = m2C2[idx];
			std::complex<double> M0, D0, ap, am;
			double ra, ia ;
			ra = (double) real(Maux) ;
			ia = (double) imag(Maux) ;
			M0 = ra + im*ia	;
			ra = (double) real(Daux) ;
			ia = (double) imag(Daux) ;
			D0 = (ra + im*ia)/(im*w1d)	;
			ap = 0.5*(M0*(1.0 - im*zeta1d) + D0);
			am = 0.5*(M0*(1.0 + im*zeta1d) - D0);
			ap *= ooI*pha;
			am *= ooI*conj(pha);
			M0 = ap + am;
			D0 = ap - am + im*zeta2d*M0;
			D0 *= im*w2d	;

			mC[idx] = M0;
			vC[idx] = D0;

auto end = std::chrono::steady_clock::now();
auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
time2 += elapsed.count();
			}

// end = std::chrono::steady_clock::now();
// elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
// time3 += elapsed.count();
		}
LogMsg(VERB_NORMAL,"WKB float/double %f/%f\n",time1, time2);
// LogMsg(VERB_NORMAL,"WKB spline/linear %f/%f\n",time1, time2);
// // check precision interpolation
// if (commRank()==0)
// 	fclose(file_wk);

// LogMsg(VERB_NORMAL,"WKB TIEMPO %f,%f,%f\n",time1, time2, time3);
// FILE *file_wkb ;
// file_wkb = NULL;
// file_wkb = fopen("out/wkbtime.txt","w+");
// for (size_t i=0; i<powMax; i++)
// 	fprintf(file_wkb,"%f\n",myarray[i]);
// fclose(file_wkb);

		auto &myPlanM = AxionFFT::fetchPlan("WKB m");
		auto &myPlanV = AxionFFT::fetchPlan("WKB v");

		LogMsg(VERB_NORMAL," FFTWing back AXION m inplace ... ");
		myPlanM.run(FFT_BCK);
		LogMsg(VERB_NORMAL,"done!!\n");

		LogMsg(VERB_NORMAL," FFTWing back AXION v inplace ... ");
		myPlanV.run(FFT_BCK);
		LogMsg(VERB_NORMAL,"done!!\n ");

		const size_t	dataLine = field->DataSize()*Ly;
		//const size_t	padLine  = field->DataSize()*(Ly+2);

		// pointers for padding ...

		char *mTf  = static_cast<char *>(static_cast<void*>(mIn));
		char *m0Tf  = static_cast<char *>(static_cast<void*>(mC));
		char *vTf  = static_cast<char *>(static_cast<void*>(vIn));
		//char *m2Tf = static_cast<char *>(static_cast<void*>(m2In));

			LogMsg(VERB_NORMAL," unpadding m in place ... ");
			LogMsg(VERB_NORMAL," unpadding (first line is not needed)");
					for (uint sl=1; sl<Sm; sl++) {
						auto	oOff = sl*field->DataSize()*(Ly);
						auto	fOff = sl*field->DataSize()*(Ly+2);
						memcpy	(m0Tf+oOff, m0Tf+fOff, dataLine);
					}
			LogMsg(VERB_NORMAL," shifthing to host ghost");
			LogMsg(VERB_NORMAL," chech precision %lu and datasize %lu",field->Precision(),field->DataSize());
					size_t dataTotalSize = (field->Precision())*(field->Size());
					memcpy	(mTf, m0Tf, dataTotalSize);

			LogMsg(VERB_NORMAL," unpadding v in place ... ");
					for (uint sl=0; sl<Sm; sl++) {
						auto	oOff = sl*field->DataSize()*(Ly);
						auto	fOff = sl*field->DataSize()*(Ly+2);
						memcpy	(vTf+oOff, vTf+fOff, dataLine);
					}

	  Float toton = (Float) field->TotalSize();

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
		field->updateR();
    LogMsg(VERB_NORMAL,"[WKB] scalar set z=%f done (m2 still in %f)", (*field->zV()), zIni);
		field->setFolded(false);
		LogMsg(VERB_NORMAL,"[WKB] m,v, set unfolded!");
		LogMsg(VERB_NORMAL,"[WKB] Complete!\n ");



	}
}


// Build phi tables
// build interpolation
// Idea is to split int w dt into
// int m dt (analitical)
// o int k dt (analitical)
// int dt w-k-m (numerical)
