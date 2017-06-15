#include <cstdio>
#include <cstdlib>
#include <math.h>       /* pow */
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

double	saxionshift(double z, double nQcd, double zth, double zres, double LLL)
{
 	double alpha = axionmass2(z, nQcd, zth, zres)/LLL;
 	double morsa = ((2./sqrt(3.))*cos(atan2(sqrt(4./3.-9.*alpha*alpha),3.0*alpha)/3.0)-1.);
 	return morsa;

}
