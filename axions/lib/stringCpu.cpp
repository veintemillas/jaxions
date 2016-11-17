#include <cstdio>
#include <complex>

#include "index.h"
//#include "comms.h"
#include "scalarField.h"
#include "parse.h"

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
	//printf("sh-called! hand=%d\n",hand);

	return hand;
}

//PROBLEMS TO COMPILE THIS FUNCTION

// /*	Termina el template	*/
//
// void	analyzeStr	(Scalar *axion, int *window, const int index)
// {
// 	//--------------------------------------------------
// 	//    STRINGS
// 	//--------------------------------------------------
//
// 	const int nx = axion->Length();
// 	const int myRank = commRank();
// 	int hand;
//
// 	char stoStr[256];
//
// 	sprintf(stoStr, "out/str/str-%05d.dat.%03d", index, myRank);
// 	strWrite = NULL;
//
// 	if ((strWrite = fopen(stoStr, "w+")) == NULL)
// 	{
// 		printf("Error: Couldn't open file %s for writing\n", stoStr);
// 		return 1;
// 	}
//
// 	#pragma omp parallel for default(shared) schedule(static)
// 	for (int idx=0; idx<axion->Size(); idx++)
// 		window[idx] = 0;
//
// 	#pragma omp parallel for default(shared) private(L) schedule(static)
// 	for (int idx=0; idx<axion->Size(); idx++)
// 	{
// 		complex<double> s1, s2, s3, s4;
//
// 		int	n1, n2, n3, n4;
// 		int	X[3];
//
// 		idx2Vec (idx, X, nx);
//
// 		// PLAQUETTE IJ      11-12-22-21-11
//
// 		n1 = idx;
//
// 		X[1] = ((X[1]+1)%nx);
// 		n2 = vec2Idx(X, nx);
//
// 		X[2] = ((X[2]+1)%nx);
// 		n3 = vec2Idx(X, nx);
//
// 		X[1] = ((X[1]-1+nx)%nx);
// 		n4 = vec2Idx(X, nx);
//
// 		s1 = ((complex<double> *) axion->mCpu())[n1] ; s2 = ((complex<double> *) axion->mCpu())[n2] ;
// 		s3 = ((complex<double> *) axion->mCpu())[n3] ; s4 = ((complex<double> *) axion->mCpu())[n4] ;
//
// 		hand = stringHand(s1, s2, s3, s4);
//
// 		if ((hand == 2) || (hand == -2))
// 		{
// 			#pragma omp critical
// 			fprintf(strWrite,  "%f %f %f %d\n", X[2]+0.5 , X[1]+0.5, X[0]+0., 1);
//
// 			window[n1] = 0;
// 			window[n2] = 0;
// 			window[n3] = 0;
// 			window[n4] = 0;
// 		}
//
// 		// PLAQUETTE IK      11-12-22-21-11
// 		X[0] = ((X[0]+1)%nx);
// 		n2 = vec2Idx(X, nx);
//
// 		X[2] = ((X[2]+1)%nx);
// 		n3 = vec2Idx(X, nx);
//
// 		X[0] = ((X[0]-1+nx)%nx);
// 		n4 = vec2Idx(X, nx);
//
// 		s1 = ((complex<double> *) axion->mCpu())[n1] ; s2 = ((complex<double> *) axion->mCpu())[n2] ;
// 		s3 = ((complex<double> *) axion->mCpu())[n3] ; s4 = ((complex<double> *) axion->mCpu())[n4] ;
//
// 		hand = stringHand(s1, s2, s3, s4);
//
// 		if ((hand == 2) || (hand == -2))
// 		{
// 			#pragma omp critical
// 			fprintf(strWrite,  "%f %f %f %d\n", X[2]+0.5, X[1]+0., X[0]+0.5, 2);
//
// 			window[n1] = 0;
// 			window[n2] = 0;
// 			window[n3] = 0;
// 			window[n4] = 0;
// 		}
//
// 		// PLAQUETTE JK      11-12-22-21-11
// 		X[0] = ((X[0]+1)%nx);
// 		n2 = vec2Idx(X, nx);
//
// 		X[1] = ((X[1]+1)%nx);
// 		n3 = vec2Idx(X, nx);
//
// 		X[0] = ((X[0]-1+nx)%nx);
// 		n4 = vec2Idx(X, nx);
//
// 		s1 = ((complex<double> *) axion->mCpu())[n1] ; s2 = ((complex<double> *) axion->mCpu())[n2] ;
// 		s3 = ((complex<double> *) axion->mCpu())[n3] ; s4 = ((complex<double> *) axion->mCpu())[n4] ;
//
// 		hand = stringHand(s1, s2, s3, s4);
//
// 		if ((hand == 2) || (hand == -2))
// 		{
// 			#pragma omp critical
// 			fprintf(strWrite,  "%f %f %f %d\n", X[2]+0., X[1]+0.5, X[0]+0.5, 3);
//
// 			window[n1] = 0;
// 			window[n2] = 0;
// 			window[n3] = 0;
// 			window[n4] = 0;
// 		}
// 	}
//
// 	fclose(strWrite);
// 	printf("\nString printed\n");
// }


//------------------------------------------------------------------------
//------------------------------------------------------------------------
//------------------------------------------------------------------------
//------------------------------------------------------------------------



void	analyzeStrFolded	(Scalar *axion, const int index)
{
	//--------------------------------------------------
	//    JAVI STRINGS FOLDED
	//--------------------------------------------------

	const size_t n1 = axion->Length();
	const size_t n2 = axion->Surf();
	const size_t shift = axion->shift();
	const size_t Lz = axion->Depth()	;
	size_t Nshift=n1/shift;
	const size_t fSize = axion->dataSize();

//	const int myRank = commRank();
	int hand;

	char stoStr[256];

	sprintf(stoStr, "out/str/str-%05d.txt", index);
	FILE *file_strings ;
	file_strings = NULL;
	file_strings = fopen(stoStr,"w+");
	fprintf(file_strings,  "# %d %f %f %f \n", sizeN, sizeL, sizeL/sizeN, (*axion->zV()) );
	//printf("TEST+ %d \n", stringHand( complex<double> (0.1,0.1), complex<double>(0.1,-0.1), complex<double>(-0.1,-0.1), complex<double>(-0.1,0.1)));
	//printf("TEST- %d \n", stringHand( complex<double> (0.1,0.1), complex<double>(0.1,-0.1), complex<double>(0.1,0.1), complex<double> (0.1,0.1)));

	//printf(" called (n1,n2,shift,Lz,Nshift)=(%d,%d,%d,%d,%d) \n",n1,n2,shift,Lz,Nshift);fflush (stdout);


	switch	(axion->Precision())
	{
		case	FIELD_DOUBLE:
		{
			complex<double> *mM = static_cast<complex<double>*> (axion->mCpu());

			#pragma omp parallel for default(shared) private(hand) schedule(static)
			for (size_t iz=0; iz<Lz; iz++)
			{

				complex<double> s1, s2, s3, s4;
				size_t sy, sy1, iys, iys1;
				size_t fIdx000,fIdx010 ;

				//printf("-%d-",iz);fflush (stdout);
				//DOES NOT TAKE THE LAST Iy=N-1 FOR SIMPLICITY
				for (size_t iy=0; iy<n1-1; iy++)
					{
						sy   =  iy/Nshift;
						iys  =  iy%Nshift;
						sy1  = (iy+1)/Nshift;
						iys1 = (iy+1)%Nshift;
						//printf("-(%d,%d,%d,%d))-",iy,iz,sy,iys);fflush (stdout);
					//DOES NOT TAKE THE LAST Ix=N-1 FOR SIMPLICITY
						for (size_t ix=0; ix<n1-1; ix++)
						{
							// PLAQUETTE XY      -------------------------------------------
							fIdx000 = n2 + iz*n2 + ((size_t) (iys*n1*shift + ix*shift + sy));
							fIdx010 = n2 + iz*n2 + ((size_t) (iys1*n1*shift + ix*shift + sy1));
							//s1 = static_cast<complex<double> *> (axion->mCpu())[fIdx000] ;
							//s2 = static_cast<complex<double> *> (axion->mCpu())[fIdx000 + shift] ;
							// s4 = static_cast<complex<double> *> (axion->mCpu())[fIdx010] ;
							// s3 = static_cast<complex<double> *> (axion->mCpu())[fIdx010 + shift] ;
							s1 = mM[fIdx000] ;
							s2 = mM[fIdx000 + shift] ;
							s4 = mM[fIdx010] ;
							s3 = mM[fIdx010 + shift] ;

							hand = stringHand(s1, s2, s3, s4);

							// if ((iz == 0) && (index == 1))
							// {
							// 	//printf("(fIdx000,fIdx100,fIdx110,fIdx010)= %lu %lu %lu %lu \n",fIdx000-n2,fIdx000+shift-n2,fIdx010+shift-n2,fIdx010-n2);
							// 	printf("(%d,%d) [%d,%d] [%d,%d] %lu %lu %lu %lu \n",ix, iy, sy, iys,sy1, iys1, fIdx000-n2,fIdx000+shift-n2,fIdx010+shift-n2,fIdx010-n2);
							// }
							if ((hand == 2) || (hand == -2))
							{
								// printf("(ix,iy,iz)=(%d,%d,%d) ",ix,iy,iz);
								// printf("s1= [%f,%f] (%lu) ",real(s1),imag(s1),fIdx000);
								// printf("s2= [%f,%f] (%lu) ",real(s2),imag(s2),fIdx000 + shift);
								// printf("s3= [%f,%f] (%lu) ",real(s3),imag(s3),fIdx010);
								// printf("s4= [%f,%f] (%lu) \n",real(s4),imag(s4),fIdx010 + shift);
								#pragma omp critical
								{
									fprintf(file_strings,  "%f %f %f \n", ix+0.5 , iy+0.5, iz+0.0);
									//window[n1] = 0;window[n2] = 0;window[n3] = 0;window[n4] = 0;
								}
							}
							//PLAQUETTE YZ      -------------------------------------------
							//s1 = ...fIdx000 ; //already got
							//s4 = ... 001 ; // already got
							//s2 = static_cast<complex<double> *> (axion->mCpu())[fIdx000 + n2] ;
							//s3 = static_cast<complex<double> *> (axion->mCpu())[fIdx010 + n2] ;
							s2 = mM[fIdx000 + n2] ;
							s3 = mM[fIdx010 + n2] ;

							hand = stringHand(s1, s4, s3, s2);

							if ((hand == 2) || (hand == -2))
							{
								#pragma omp critical
								{
									fprintf(file_strings,  "%f %f %f \n", ix+0.0 , iy+0.5, iz+0.5);
									//window[n1] = 0;window[n2] = 0;window[n3] = 0;window[n4] = 0;
								}
							}
							// PLAQUETTE XZ      -------------------------------------------
							//s1 = ...fIdx000 ; //already got
							//s2 = ... 001;			//already got
							// s4 = static_cast<complex<double> *> (axion->mCpu())[fIdx000 + shift] ;
							// s3 = static_cast<complex<double> *> (axion->mCpu())[fIdx000 + shift + n2];
							s4 = mM[fIdx000 + shift] ;
							s3 = mM[fIdx000 + shift + n2];

							hand = stringHand(s1, s2, s3, s4);

							if ((hand == 2) || (hand == -2))
							{
								#pragma omp critical
								{
									fprintf(file_strings,  "%f %f %f \n", ix+0.5 , iy+0.0, iz+0.5);
									//window[n1] = 0;window[n2] = 0;window[n3] = 0;window[n4] = 0;
								}
							}

						}	//end ix
					}	//end for iy
				}	//end for iz



		}
		break;

		case	FIELD_SINGLE:
		{
		complex<float> *mM = static_cast<complex<float>*> (axion->mCpu());

		#pragma omp parallel for default(shared) schedule(static)
		for (size_t iz=0; iz<Lz; iz++)
		{

			complex<double> s1, s2, s3, s4;
			size_t sy, sy1, iys, iys1;
			size_t fIdx000,fIdx010 ;

			//printf("-%d-",iz);fflush (stdout);
			//DOES NOT TAKE THE LAST Iy=N-1 FOR SIMPLICITY
			for (size_t iy=0; iy<n1-1; iy++)
				{
					sy   =  iy/Nshift;
					iys  =  iy%Nshift;
					sy1  = (iy+1)/Nshift;
					iys1 = (iy+1)%Nshift;
					//printf("-(%d,%d,%d,%d))-",iy,iz,sy,iys);fflush (stdout);
				//DOES NOT TAKE THE LAST Ix=N-1 FOR SIMPLICITY
					for (size_t ix=0; ix<n1-1; ix++)
					{
						// PLAQUETTE XY      -------------------------------------------
						fIdx000 = n2 + iz*n2 + ((size_t) (iys*n1*shift + ix*shift + sy));
						fIdx010 = n2 + iz*n2 + ((size_t) (iys1*n1*shift + ix*shift + sy1));
						//s1 = static_cast<complex<double> *> (axion->mCpu())[fIdx000] ;
						//s2 = static_cast<complex<double> *> (axion->mCpu())[fIdx000 + shift] ;
						// s4 = static_cast<complex<double> *> (axion->mCpu())[fIdx010] ;
						// s3 = static_cast<complex<double> *> (axion->mCpu())[fIdx010 + shift] ;
						s1 = mM[fIdx000] ;
						s2 = mM[fIdx000 + shift] ;
						s4 = mM[fIdx010] ;
						s3 = mM[fIdx010 + shift] ;

						hand = stringHand(s1, s2, s3, s4);

						// if ((iz == 0) && (index == 1))
						// {
						// 	//printf("(fIdx000,fIdx100,fIdx110,fIdx010)= %lu %lu %lu %lu \n",fIdx000-n2,fIdx000+shift-n2,fIdx010+shift-n2,fIdx010-n2);
						// 	printf("(%d,%d) [%d,%d] [%d,%d] %lu %lu %lu %lu \n",ix, iy, sy, iys,sy1, iys1, fIdx000-n2,fIdx000+shift-n2,fIdx010+shift-n2,fIdx010-n2);
						// }
						if ((hand == 2) || (hand == -2))
						{
							// printf("(ix,iy,iz)=(%d,%d,%d) ",ix,iy,iz);
							// printf("s1= [%f,%f] (%lu) ",real(s1),imag(s1),fIdx000);
							// printf("s2= [%f,%f] (%lu) ",real(s2),imag(s2),fIdx000 + shift);
							// printf("s3= [%f,%f] (%lu) ",real(s3),imag(s3),fIdx010);
							// printf("s4= [%f,%f] (%lu) \n",real(s4),imag(s4),fIdx010 + shift);
							#pragma omp critical
							{
								fprintf(file_strings,  "%f %f %f \n", ix+0.5 , iy+0.5, iz+0.0);
								//window[n1] = 0;window[n2] = 0;window[n3] = 0;window[n4] = 0;
							}
						}
						//PLAQUETTE YZ      -------------------------------------------
						//s1 = ...fIdx000 ; //already got
						//s4 = ... 001 ; // already got
						//s2 = static_cast<complex<double> *> (axion->mCpu())[fIdx000 + n2] ;
						//s3 = static_cast<complex<double> *> (axion->mCpu())[fIdx010 + n2] ;
						s2 = mM[fIdx000 + n2] ;
						s3 = mM[fIdx010 + n2] ;

						hand = stringHand(s1, s4, s3, s2);

						if ((hand == 2) || (hand == -2))
						{
							#pragma omp critical
							{
								fprintf(file_strings,  "%f %f %f \n", ix+0.0 , iy+0.5, iz+0.5);
								//window[n1] = 0;window[n2] = 0;window[n3] = 0;window[n4] = 0;
							}
						}
						// PLAQUETTE XZ      -------------------------------------------
						//s1 = ...fIdx000 ; //already got
						//s2 = ... 001;			//already got
						// s4 = static_cast<complex<double> *> (axion->mCpu())[fIdx000 + shift] ;
						// s3 = static_cast<complex<double> *> (axion->mCpu())[fIdx000 + shift + n2];
						s4 = mM[fIdx000 + shift] ;
						s3 = mM[fIdx000 + shift + n2];

						hand = stringHand(s1, s2, s3, s4);

						if ((hand == 2) || (hand == -2))
						{
							#pragma omp critical
							{
								fprintf(file_strings,  "%f %f %f \n", ix+0.5 , iy+0.0, iz+0.5);
								//window[n1] = 0;window[n2] = 0;window[n3] = 0;window[n4] = 0;
							}
						}

					}	//end ix
				}	//end for iy
			}	//end for iz



		}
		break;

		default:
		{
		printf("Unrecognized precision\n");
		exit(1);
		break;
		}
	}




	fclose(file_strings);
	printf(" ... String printed\n");
}

// void	analyzeStrFolded	(Scalar *axion, const int index)
// {
// 	//--------------------------------------------------
// 	//    JAVI STRINGS FOLDED
// 	//--------------------------------------------------
//
// 	const int n1 = axion->Length();
// 	const int n2 = axion->Surf();
// 	const int shift = axion->shift();
// 	const int Lz = axion->Depth()	;
// 	int Nshift=n1/shift;
//
// //	const int myRank = commRank();
// 	int hand;
//
// 	char stoStr[256];
//
// 	sprintf(stoStr, "out/str/str-%05d.txt", index);
// 	FILE *file_strings ;
// 	file_strings = NULL;
// 	file_strings = fopen(stoStr,"w+");
// 	fprintf(file_strings,  "# %d %f %f %f \n", sizeN, sizeL, sizeL/sizeN, (*axion->zV()) );
// 	//printf("TEST+ %d \n", stringHand( complex<double> (0.1,0.1), complex<double>(0.1,-0.1), complex<double>(-0.1,-0.1), complex<double>(-0.1,0.1)));
// 	//printf("TEST- %d \n", stringHand( complex<double> (0.1,0.1), complex<double>(0.1,-0.1), complex<double>(0.1,0.1), complex<double> (0.1,0.1)));
//
// 	//printf(" called (n1,n2,shift,Lz,Nshift)=(%d,%d,%d,%d,%d) ",n1,n2,shift,Lz,Nshift);fflush (stdout);
//
// 	//#pragma omp parallel for default(shared) schedule(static)
// 	for (int iz=0; iz<Lz; iz++)
// 	{
//
// 		complex<double> s1, s2, s3, s4;
// 		int sy, sy1, iys, iys1;
// 		size_t fIdx000,fIdx010 ;
//
// 		//printf("-%d-",iz);fflush (stdout);
// 		//DOES NOT TAKE THE LAST Iy=N-1 FOR SIMPLICITY
// 		for (int iy=0; iy<n1-1; iy++)
// 			{
// 				sy = iy/Nshift;
// 				iys=iy%Nshift;
// 				sy1 = (iy+1)/Nshift;
// 				iys=(iy+1)%Nshift;
// 				//printf("-(%d,%d,%d,%d))-",iy,iz,sy,iys);fflush (stdout);
// 			//DOES NOT TAKE THE LAST Ix=N-1 FOR SIMPLICITY
// 				for (int ix=0; ix<n1-1; ix++)
// 				{
// 					// PLAQUETTE XY      -------------------------------------------
// 					fIdx000 = n2 + iz*n2 + ((size_t) (iys*n1*shift + ix*shift + sy));
// 					s1 = static_cast<complex<double> *> (axion->mCpu())[fIdx000] ;
// 					if (abs(s1)/(*axion->zV())<0.5)
// 					{
// 						//#pragma omp critical
// 						fprintf(file_strings,  "%f %f %f \n", ix+0.0, iy+0.0, iz+0.0);
// 						//window[n1] = 0;window[n2] = 0;window[n3] = 0;window[n4] = 0;
// 					}
//
// 				}	//end ix
// 			}	//end for iy
// 		}	//end for iz
// 	fclose(file_strings);
// 	printf(" ... String printed\n");
// }

int	analyzeStrUNFolded	(Scalar *axion, const int index)
{
	//--------------------------------------------------
	//    JAVI STRINGS UNFOLDED
	//--------------------------------------------------

	const size_t n1 = axion->Length();
	const size_t n2 = axion->Surf();
	const size_t shift = axion->shift();
	const size_t Lz = axion->Depth()	;
	size_t Nshift=n1/shift;
	const size_t fSize = axion->dataSize();

//	const int myRank = commRank();
	int hand;
	int stlength = 0;

	char stoStr[256];

	sprintf(stoStr, "out/str/str-%05d.txt", index);
	FILE *file_strings ;
	file_strings = NULL;
	file_strings = fopen(stoStr,"w+");
	fprintf(file_strings,  "# %d %f %f %f \n", sizeN, sizeL, sizeL/sizeN, (*axion->zV()) );
	//printf("TEST+ %d \n", stringHand( complex<double> (0.1,0.1), complex<double>(0.1,-0.1), complex<double>(-0.1,-0.1), complex<double>(-0.1,0.1)));
	//printf("TEST- %d \n", stringHand( complex<double> (0.1,0.1), complex<double>(0.1,-0.1), complex<double>(0.1,0.1), complex<double> (0.1,0.1)));

	//printf(" called (n1,n2,shift,Lz,Nshift)=(%d,%d,%d,%d,%d) \n",n1,n2,shift,Lz,Nshift);fflush (stdout);


	switch	(axion->Precision())
	{
		case	FIELD_DOUBLE:
		{
			complex<double> *mM = static_cast<complex<double>*> (axion->mCpu());


			#pragma omp parallel for default(shared) private(hand) schedule(static) reduction(+:stlength)
			for (size_t iz=0; iz<Lz; iz++)
			{

				complex<double> s1, s2, s3, s4;
				size_t sy, sy1, iys, iys1;
				size_t fIdx000,fIdx010 ;

				//printf("-%d-",iz);fflush (stdout);
				//DOES NOT TAKE THE LAST Iy=N-1 FOR SIMPLICITY
				for (size_t iy=0; iy<n1-1; iy++)
					{
						for (size_t ix=0; ix<n1-1; ix++)
						{
							// PLAQUETTE XY      -------------------------------------------
							fIdx000 = n2 + iz*n2 + ((size_t) iy*n1 + ix);
							fIdx010 = n2 + iz*n2 + ((size_t) (iy+1)*n1 + ix);
							s1 = mM[fIdx000] ;
							s2 = mM[fIdx000 + 1] ;
							s4 = mM[fIdx010] ;
							s3 = mM[fIdx010 + 1] ;

							hand = stringHand(s1, s2, s3, s4);

							if ((hand == 2) || (hand == -2))
							{
								++stlength;
								#pragma omp critical
								{
									fprintf(file_strings,  "%f %f %f \n", ix+0.5 , iy+0.5, iz+0.0);
								}
							}
							//PLAQUETTE YZ      -------------------------------------------
							s2 = mM[fIdx000 + n2] ;
							s3 = mM[fIdx010 + n2] ;

							hand = stringHand(s1, s4, s3, s2);

							if ((hand == 2) || (hand == -2))
							{
								++stlength;
								#pragma omp critical
								{
									fprintf(file_strings,  "%f %f %f \n", ix+0.0 , iy+0.5, iz+0.5);
								}
							}
							// PLAQUETTE XZ      -------------------------------------------
							s4 = mM[fIdx000 + 1] ;
							s3 = mM[fIdx000 + 1 + n2];

							hand = stringHand(s1, s2, s3, s4);

							if ((hand == 2) || (hand == -2))
							{
								++stlength;
								#pragma omp critical
								{
									fprintf(file_strings,  "%f %f %f \n", ix+0.5 , iy+0.0, iz+0.5);
								}
							}

						}	//end ix
					}	//end for iy
				}	//end for iz



		}
		break;

		case	FIELD_SINGLE:
		{
		complex<float> *mM = static_cast<complex<float>*> (axion->mCpu());

		#pragma omp parallel for default(shared) schedule(static) reduction(+:stlength)
		for (size_t iz=0; iz<Lz; iz++)
		{

			complex<float> s1, s2, s3, s4;
			size_t sy, sy1, iys, iys1;
			size_t fIdx000,fIdx010 ;

			//DOES NOT TAKE THE LAST Iy=N-1 FOR SIMPLICITY
			for (size_t iy=0; iy<n1-1; iy++)
				{
					for (size_t ix=0; ix<n1-1; ix++)
					{
						// PLAQUETTE XY      -------------------------------------------
						fIdx000 = n2 + iz*n2 + ((size_t) iy*n1 + ix);
						fIdx010 = n2 + iz*n2 + ((size_t) (iy+1)*n1 + ix);
						s1 = mM[fIdx000] ;
						s2 = mM[fIdx000 + 1] ;
						s4 = mM[fIdx010] ;
						s3 = mM[fIdx010 + 1] ;

						hand = stringHand(s1, s2, s3, s4);

						if ((hand == 2) || (hand == -2))
						{
							++stlength;
							#pragma omp critical
							{
								fprintf(file_strings,  "%f %f %f \n", ix+0.5 , iy+0.5, iz+0.0);
							}
						}
						//PLAQUETTE YZ      -------------------------------------------
						s2 = mM[fIdx000 + n2] ;
						s3 = mM[fIdx010 + n2] ;

						hand = stringHand(s1, s4, s3, s2);

						if ((hand == 2) || (hand == -2))
						{
							++stlength;
							#pragma omp critical
							{
								fprintf(file_strings,  "%f %f %f \n", ix+0.0 , iy+0.5, iz+0.5);
							}
						}
						// PLAQUETTE XZ      -------------------------------------------
						s4 = mM[fIdx000 + 1] ;
						s3 = mM[fIdx000 + 1 + n2];

						hand = stringHand(s1, s2, s3, s4);

						if ((hand == 2) || (hand == -2))
						{
							++stlength;
							#pragma omp critical
							{
								fprintf(file_strings,  "%f %f %f \n", ix+0.5 , iy+0.0, iz+0.5);
							}
						}

					}	//end ix
				}	//end for iy
			}	//end for iz



		}
		break;

		default:
		{
		printf("Unrecognized precision\n");
		exit(1);
		break;
		}

	}

	fclose(file_strings);
	printf(" %d ... String printed\n", (int) stlength);

	return stlength ;
}
