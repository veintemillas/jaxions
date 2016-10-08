#include <cstdio>
#include <complex>

#include "index.h"
#include "scalarField.h"

using namespace std;

template<typename Float>
int	stringHand	(complex<Float> s1, complex<Float> s2, complex<Float> s3, complex<Float> s4)
{
	int	hand = 0;

	if ((s1.imag() > 0.) != (s2.imag() > 0.))
	{
		if ((s1*conj(s2)).imag() > 0.)
			hand++;
		else
			hand--;
	}
	
	if ((s2.imag() > 0.) != (s3.imag() > 0.))
	{
		if ((s2*conj(s3)).imag() > 0.)
			hand++;
		else
			hand--;
	}
	
	if ((s3.imag() > 0.) != (s4.imag() > 0.))
	{
		if ((s3*conj(s4)).imag() > 0.)
			hand++;
		else
			hand--;
	}
	
	if ((s4.imag() > 0.) != (s1.imag() > 0.))
	{
		if ((s4*conj(s1)).imag() > 0.)
			hand++;
		else
			hand--;
	}

	return hand;
}


/*	Termina el template	*/

void	analyzeStr	(Scalar *axion, int *window, const int index)
{
	//--------------------------------------------------
	//    STRINGS
	//--------------------------------------------------          

	const int nx = axion->Length();
	int hand;

	char stoStr[256];

	sprintf(stoStr, "out/str/str-%05d.dat", index);
	strWrite = NULL;

	if ((strWrite = fopen(stoStr, "w+")) == NULL)
	{
		printf("Error: Couldn't open file %s for writing\n", stoStr);
		return 1;
	}

	#pragma omp parallel for default(shared) schedule(static)
	for (int idx=0; idx<axion->Size(); idx++)
		window[idx] = 0;

	#pragma omp parallel for default(shared) private(L) schedule(static)
	for (int idx=0; idx<axion->Size(); idx++)
	{
		complex<double> s1, s2, s3, s4;

		int	n1, n2, n3, n4;
		int	X[3];

		idx2Vec (idx, X, nx);

		// PLAQUETTE IJ      11-12-22-21-11

		n1 = idx;

		X[1] = ((X[1]+1)%nx);
		n2 = vec2Idx(X, nx);

		X[2] = ((X[2]+1)%nx);
		n3 = vec2Idx(X, nx);

		X[1] = ((X[1]-1+nx)%nx);
		n4 = vec2Idx(X, nx);

		s1 = ((complex<double> *) axion->mCpu())[n1] ; s2 = ((complex<double> *) axion->mCpu())[n2] ;
		s3 = ((complex<double> *) axion->mCpu())[n3] ; s4 = ((complex<double> *) axion->mCpu())[n4] ;

		hand = stringHand(s1, s2, s3, s4);   

		if ((hand == 2) || (hand == -2))
		{
			#pragma omp critical
			fprintf(strWrite,  "%f %f %f %d\n", X[2]+0.5 , X[1]+0.5, X[0]+0., 1);

			window[n1] = 0;
			window[n2] = 0;
			window[n3] = 0;
			window[n4] = 0;
		}
		
		// PLAQUETTE IK      11-12-22-21-11
		X[0] = ((X[0]+1)%nx);
		n2 = vec2Idx(X, nx);

		X[2] = ((X[2]+1)%nx);
		n3 = vec2Idx(X, nx);

		X[0] = ((X[0]-1+nx)%nx);
		n4 = vec2Idx(X, nx);

		s1 = ((complex<double> *) axion->mCpu())[n1] ; s2 = ((complex<double> *) axion->mCpu())[n2] ;
		s3 = ((complex<double> *) axion->mCpu())[n3] ; s4 = ((complex<double> *) axion->mCpu())[n4] ;

		hand = stringHand(s1, s2, s3, s4);   

		if ((hand == 2) || (hand == -2))
		{
			#pragma omp critical
			fprintf(strWrite,  "%f %f %f %d\n", X[2]+0.5, X[1]+0., X[0]+0.5, 2);

			window[n1] = 0;
			window[n2] = 0;
			window[n3] = 0;
			window[n4] = 0;
		}

		// PLAQUETTE JK      11-12-22-21-11
		X[0] = ((X[0]+1)%nx);
		n2 = vec2Idx(X, nx);

		X[1] = ((X[1]+1)%nx);
		n3 = vec2Idx(X, nx);

		X[0] = ((X[0]-1+nx)%nx);
		n4 = vec2Idx(X, nx);

		s1 = ((complex<double> *) axion->mCpu())[n1] ; s2 = ((complex<double> *) axion->mCpu())[n2] ;
		s3 = ((complex<double> *) axion->mCpu())[n3] ; s4 = ((complex<double> *) axion->mCpu())[n4] ;

		hand = stringHand(s1, s2, s3, s4);   

		if ((hand == 2) || (hand == -2))
		{
			#pragma omp critical
			fprintf(strWrite,  "%f %f %f %d\n", X[2]+0., X[1]+0.5, X[0]+0.5, 3);

			window[n1] = 0;
			window[n2] = 0;
			window[n3] = 0;
			window[n4] = 0;
		}
	}

	fclose(strWrite);           
	printf("\nString printed\n");
}
