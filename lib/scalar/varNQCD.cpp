#include <cstdio>
#include <cstdlib>
#include <math.h>       /* pow */

double	axionmass(double z, double nQcd, double zthreshold, double indi)
{
	double zthres= 10.;
	double morsa;
	//double expo = (8.+4.212*z*z)/(1.+0.5817*z*z);
	double expo = nQcd;
	if (z > zthres)
	{
		morsa = indi*pow(zthres,expo/2.);
	}
	else
	{
		morsa = indi*pow(z,expo/2.);
	}

	return morsa;
}

double	axionmass2(double z, double nQcd, double zthreshold, double indi)
{
	double zthres= 10.;
	double morsa;
	//double expo = (8.+4.212*z*z)/(1.+0.5817*z*z);
	double expo = nQcd;
	if (z > zthres)
	{
		morsa = indi*indi*pow(zthres,expo);
	}
	else
	{
		morsa = indi*indi*pow(z,expo);
	}
	return morsa;
}

double	saxionshift(double z, double nQcd, double zthreshold, double indi, double LL)
{
 	double alpha = axionmass2(z, nQcd, zthreshold, indi)/LL;
 	double morsa = ((2./sqrt(3.))*cos(atan2(sqrt(4./3.-alpha*alpha),3*alpha)/3.0)-1.);
 	return morsa;

}
