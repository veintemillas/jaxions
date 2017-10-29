#include <cstdio>
#include <cstdlib>
#include <math.h>	/* pow */
#include <algorithm>	/* max */
#include "enum-field.h"
#include "utils/parse.h"

double	axionmass(double z, double nQcd, double zth, double zres)
{

	double morsa;
	//double expo = (8.+4.212*z*z)/(1.+0.5817*z*z);
	double expo = nQcd;
	if (z > zth &&  zth < zres )
	{
		morsa = indi3*pow(zth,expo/2.);
		if (z > zres)
		{
			morsa *= pow(z/zres,expo/2.);
		}
	}
	else
	{
		morsa = indi3*pow(z,expo/2.);
	}

	return morsa;
}

double	axionmass2(double z, double nQcd, double zth, double zres)
{
	double morsa;
	//double expo = (8.+4.212*z*z)/(1.+0.5817*z*z);
	double expo = nQcd;
	if (z > zth &&  zth < zres)
	{
		morsa = indi3*indi3*pow(zth,expo);
		if (z > zres)
		{
			morsa *= pow(z/zres,expo);
		}
	}
	else
	{
		morsa = indi3*indi3*pow(z,expo);
	}
	return morsa;
}

double rsvPQ2 (double a)
{
	double a2 = a*a;
	double a3 = a2*a;
	double a4 = a2*a2;
	// return (1 + 3.141589*alpha + 3.1645839*a2 + 1.1045853*a3 + 0.089438*a4)/
  //                (1 + 3.016589*alpha + 2.8578228*a2 + 0.89696*a3 + 0.0564027*a4);
	return (0.125*a + 0.30676113886283973*a2 + 0.20762392505082639*a3 + 0.03303541390146716*a4)/
   (1 + 3.0165891109027165*a + 2.857822775289389*a2 + 0.8969613324856603*a3 + 0.05640260585369341*a4);
}

double	saxionshift(double z, double nQcd, double zth, double zres, double LLL)
{
 	double alpha = axionmass2(z, nQcd, zth, zres)/LLL;
 	double discr = 4./3.-9.*alpha*alpha;

	return	((discr > 0.) ? ((2./sqrt(3.))*cos(atan2(sqrt(discr),3.0*alpha)/3.0)-1.) : ((2./sqrt(3.))*cosh(atanh(sqrt(-discr)/(3.0*alpha))/3.0)-1.));
}

double	saxionshift(double axmass, double LLL, VqcdType VqcdPQ)
{
		double alpha = 0.;
		double shift = 0.;
		double discr = 4./3.;

	switch	(VqcdPQ) {
		case	VQCD_1:
			alpha = axmass*axmass/(LLL);
			discr = 4./3.-9.*alpha*alpha;
			shift = ((discr > 0.) ? ((2./sqrt(3.))*cos(atan2(sqrt(discr),3.0*alpha)/3.0)-1.) : ((2./sqrt(3.))*cosh(atanh(sqrt(-discr)/(3.0*alpha))/3.0)-1.));
		 break;
		case	VQCD_1_PQ_2:
			alpha = axmass*axmass/(LLL);
	 		//shift = (sqrt(4.+9.*alpha)-2.)/18.;
			shift = rsvPQ2(alpha);
		 break;
		case	VQCD_2:
				shift = 0.;
			break;
		}

	return	shift;
}


double	dzSize	(double z, FieldType fType, LambdaType lType) {
	double oodl = ((double) sizeN)/sizeL;
	double mAx2 = axionmass2(z, nQcd, zthres, zrestore);
	double mAfq = 0.;

	if ((fType & FIELD_AXION) || (fType == FIELD_WKB))
		return	wDz/sqrt(mAx2*(z*z) + 12.*(oodl*oodl));
	 else
		mAfq = sqrt(mAx2*(z*z) + 12.*oodl*oodl);

	double mSfq = 0.;

	switch (lType) {
		case	LAMBDA_Z2:
			mSfq = sqrt(msa*msa + 12.)*oodl;
			break;

		case	LAMBDA_FIXED:
			mSfq = sqrt(2.*LL*(z*z)   + 12.*oodl);
			break;
	}

	return	wDz/std::max(mSfq,mAfq);
}
