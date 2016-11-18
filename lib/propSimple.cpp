#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include "scalarField.h"
#include "enum-field.h"
#include "RKParms.h"
#include "index.h"

using namespace std;

	void  shiftstencilboundary(const int Ng, const int lin, const size_t Lx, const size_t ix,
		                const size_t aYol[], const size_t aZol[], size_t Xol[], size_t Yol[], size_t Zol[])
	{
		int l; size_t auxi;
		for (int l = -Ng; l < Ng+1; l++)
		{
			//would be ix-Ng, ix-Ng+1 ... ix ... ix+Ng
			Xol[Ng+l] = (ix + l + Lx)%Lx ;
		}
			//  printf("Xol %lu %lu %lu %lu %lu\n", Xol[0] , Xol[1], Xol[2], Xol[3], Xol[4]);
			//shift to 3D cross
		for (int l = 0; l < lin; l++)
		{
			Zol[l] = aZol[l] + Xol[Ng];
			Yol[l] = aYol[l] + Xol[Ng] ;
		}
		// printf("-Zol %lu %lu %lu %lu %lu\n", Zol[0] , Zol[1], Zol[2], Zol[3], Zol[4]);
		for (int l = 0; l < lin; l++)
		{
			auxi = Xol[l] ;
			Xol[l] = auxi + aZol[Ng] ;
		}
	}
	void  shiftstencilbulk(const int Ng, const int lin, size_t Xol[], size_t Yol[], size_t Zol[])
	{
		int l;
		for (int l = 0; l < lin; l++)
		{
				Xol[l]++ ; Yol[l]++ ; Zol[l]++ ;
		}
	}

	//										(field->m2Cpu()), (field->vCpu()),(field->mCpu()), z2, zQ, dzc, dzd, ood2, LL, Lx, S, V+S, Ng);
	template<typename Float>
	void	propSimpleCoreN (const complex<Float> *m, complex<Float> *v, complex<Float> *m2, const Float z2, const Float zQ,
				const Float dzc, const Float dzd, const Float ood2, const Float LL, const size_t Lx, const size_t Sf, const size_t Vf,
				int Ng)
	{
		Float CO[4] = {0.0, 0.0, 0.0, 0.0} ;

		if (Ng == 2) {
			CO[0] = (Float) 16/12; CO[1] = (Float) -1/12;
		} else if (Ng == 3) {
			CO[0] = (Float) 3/2; CO[1] = (Float) -3/20; CO[2] = (Float) 1/90 ;
		} else if (Ng == 4) {
			CO[0] = (Float) 8/5; CO[1] = (Float) -1/5; CO[2] = (Float) 8/315 ; CO[3] = (Float) -1/560 ;
		} else if (Ng == 0) {
			CO[0] = (Float) 0 ;
		} else {
			CO[0] = (Float) 1 ;
		}

		// printf("CASE Ng=%d, %lf %lf %lf %lf \n", Ng, CO[0], CO[1], CO[2], CO[3]);

		// MULTIPLY BY dzc
		// RECALL THAT IN ACCELERATION
		// ACC = (lap*ood2 + zQ - tmp*(((Float) LL))*dzc
		// INTRODUCE NEW VARIABLES
		Float ood2dzc = ood2*dzc;
		Float zQdzc = zQ*dzc;
		Float LLdzc = ((Float) LL*dzc);
		Float z1 = sqrt(z2);
		Float modulo;

		int lin = 2*Ng+1;
		//

		//------------------------------------------------------
		// 					Z LOOP
		//------------------------------------------------------


		#pragma omp parallel for default(shared) schedule(static)
		for (size_t iz = 0; iz < Lx; iz++)
		{
			size_t auxi , iy, ix ;
			size_t Xol[lin], Yol[lin], Zol[lin];
			size_t aYol[lin], aZol[lin], bZol[lin];
			//size_t ix, iy, iz ;
			complex<Float> lap , acc, mupdate, vupdate ;
			int l ;

			//compute base arZ
			for (int l = -Ng; l < Ng+1; l++)
			{
				//would be iz-Ng, iz-Ng+1 ... iz ... iz+Ng
				bZol[Ng+l] = Sf + ((iz + l + Lx)%Lx)*Sf ;
			}
			  // printf("~Zol %lu %lu %lu %lu %lu\n", bZol[0] , bZol[1], bZol[2], bZol[3], bZol[4]);

			//------------------------------------------------------
			// 					Y LOOP
			//------------------------------------------------------

			for (int iy = 0; iy < Lx; iy++)
			{
				//compute base arY UNFOLDED
				for (int l = -Ng; l < Ng+1; l++)
				{
					//would be iy-Ng, iy-Ng+1 ... iy ... iy+Ng  l =2Ng+1
					Yol[Ng+l] = ((iy + l + Lx )%Lx)*Lx ;
				}
					// 	printf("~Yol %lu %lu %lu %lu %lu\n", Yol[0] , Yol[1], Yol[2], Yol[3], Yol[4]);
				//set the Y, Z, in position
				for (int l = 0; l<lin; l++)
				{
					auxi   = bZol[l]+Yol[Ng];
					aZol[l] = auxi;
					auxi   = bZol[Ng]+Yol[l];
					aYol[l] = auxi ;
				}
				  // printf("Zol %lu %lu %lu %lu %lu\n", aZol[0] , aZol[1], aZol[2], aZol[3], aZol[4]);
				  // printf("Yol %lu %lu %lu %lu %lu\n", aYol[0] , aYol[1], aYol[2], aYol[3], aYol[4]);

				for (int l = 0; l < lin; l++)
				{
					Zol[l] = aZol[l] ;
					Yol[l] = aYol[l] ;
				}

				//Zol, Yol increase by 1 at each iteration of ix
				//aZol, aYol are kept at ix=0

				//------------------------------------------------------
				// 					X BOUNDARY LOOP
				//------------------------------------------------------

				for (int ix = 0; ix < Lx; ix++)
				{
					for (int l = 0; l < lin; l++)
					{
						Xol[l] = aZol[Ng] + (ix + l-Ng + Lx)%Lx ;
					}

						// printf("-Zol %lu %lu %lu %lu %lu\n", Zol[0] , Zol[1], Zol[2], Zol[3], Zol[4]);

						// printf("iziyix %d %d %d \n", iz, iy, ix);
						// printf("-Xol %lu %lu %lu %lu %lu\n", Xol[0] , Xol[1], Xol[2], Xol[3], Xol[4]);
						// 	printf("-Yol %lu %lu %lu %lu %lu\n", Yol[0] , Yol[1], Yol[2], Yol[3], Yol[4]);
						// 	printf("-Zol %lu %lu %lu %lu %lu\n", Zol[0] , Zol[1], Zol[2]/Sf, Zol[3], Zol[4]);

						//---------------------------//
						//----------COMPUTE----------//
						//---------------------------//

					// printf("pretmp .. ");fflush(stdout);
					mupdate = ((Float) 6)*(m[Xol[Ng]]);
					lap = complex<Float> ( 0 , 0);
					// printf("prelap .. ");
					for (int l = 1; l < Ng+1; l++)
					{
						//printf("Ng %d l %d \n", Ng, l);
						//lap +=(m[Xol[l]] + m[Xol[lin-l-1]] + m[Yol[l]] + m[Yol[lin-l-1]] + m[Zol[l]] + m[Zol[lin-l-1]] - tmp)*CO[Ng-l-1];
						lap = lap + (m[Xol[Ng+l]] + m[Xol[Ng-l]] + m[Yol[Ng+l]] + m[Yol[Ng-l]] + m[Zol[Ng+l]] + m[Zol[Ng-l]] - mupdate)*CO[l-1] ;
					}
					mupdate = m[Xol[Ng]];
					modulo = sqrt(mupdate.real()*mupdate.real()+mupdate.imag()*mupdate.imag());
					//PQa
					//																					- mupdate*pow(modulo,a-2)*(a*LLdzc/2)*(modulo**a - z**a);
					//PQ2 VQCD1
					//acc = lap*ood2dzc + zQdzc - mupdate*LLdzc*( mupdate.real()*mupdate.real()+mupdate.imag()*mupdate.imag() - z2);

					//PQ2 VQCD2
					//acc = lap*ood2dzc - zQdzc*(mupdate-z1)/z1 -mupdate*LLdzc*( mupdate.real()*mupdate.real()+mupdate.imag()*mupdate.imag() - z2);

					//PQ1 VQCD2
					//acc = lap*ood2dzc - zQdzc*(mupdate-z1)/z1 - mupdate*((Float) LLdzc*z2*(1-z1/modulo)/2);

					//PQ4 VQCD2
					//acc = lap*ood2dzc - zQdzc*(mupdate-z1)/z1 - mupdate*((Float) LLdzc*2*(pow(modulo,6)-z2*z2*pow(modulo,2))/(z2*z2));
					//acc = lap*ood2dzc - zQdzc*(mupdate-z1)/z1 - mupdate*((Float) LLdzc*2*(powf(modulo,6)-z2*z2*powf(modulo,2))/(z2*z2));

					//acc = lap*ood2dzc - zQdzc*(mupdate-z1)/z1 -mupdate*LLdzc*( mupdate.real()*mupdate.real()+mupdate.imag()*mupdate.imag() - z2);
					//acc = lap*ood2dzc + zQdzc - mupdate*LLdzc*( mupdate.real()*mupdate.real()+mupdate.imag()*mupdate.imag() - z2);

					//harmonic oscillator centered in (0,1) with constant mass = 1;
					//saxion mass set to the usual m2 = 2 lambda
					// Use with modified initial conditions around vacuum for testing particles
					acc = lap*ood2dzc - I*LLdzc*(mupdate.real()-z1)/z1 - mupdate.imag()/z1;

					//acc already contains dzc
					v[Xol[Ng]-Sf] = v[Xol[Ng]-Sf] + acc ;
					m2[Xol[Ng]] = mupdate + v[Xol[Ng]-Sf]*dzd;

					//update Ysol, Zol
					for (int l = 0; l < lin; l++)
					{
						++Zol[l];
						++Yol[l];
					}

				}	//END X BOUNDARY LOOP
			}		//END Y LOOP
		}			//END Z LOOP
	}

	// //	----------------------------------------------------------------------------------------------------------------
	// //	THIS WORKS TOO (newer than version below)
	// //	----------------------------------------------------------------------------------------------------------------
	//

// //										(field->m2Cpu()), (field->vCpu()),(field->mCpu()), z2, zQ, dzc, dzd, ood2, LL, Lx, S, V+S, Ng);
// template<typename Float>
// void	propSimpleCoreN (const complex<Float> *m, complex<Float> *v, complex<Float> *m2, const Float z2, const Float zQ,
// 			const Float dzc, const Float dzd, const Float ood2, const Float LL, const size_t Lx, const size_t Sf, const size_t Vf,
// 			int Ng)
// {
// 	Float CO[4] = {0.0, 0.0, 0.0, 0.0} ;
//
// 	if (Ng == 2) {
// 		CO[0] = (Float) 16/12; CO[1] = (Float) -1/12;
// 	} else if (Ng == 3) {
// 		CO[0] = (Float) 3/2; CO[1] = (Float) -3/20; CO[2] = (Float) 1/90 ;
// 	} else if (Ng == 4) {
// 		CO[0] = (Float) 8/5; CO[1] = (Float) -1/5; CO[2] = (Float) 8/315 ; CO[3] = (Float) -1/560 ;
// 	} else if (Ng == 0) {
// 		CO[0] = (Float) 0 ;
// 	} else {
// 		CO[0] = (Float) 1 ;
// 	}
//
// 	// printf("CASE Ng=%d, %lf %lf %lf %lf \n", Ng, CO[0], CO[1], CO[2], CO[3]);
//
//
// 	int lin = 2*Ng+1;
//
// 	// z loop
// 	#pragma omp parallel for default(shared) schedule(static)
// 	for (size_t iz = 0; iz < Lx; iz++)
// 	{
// 		size_t auxi , iy, ix ;
// 		size_t Xol[lin], Yol[lin], Zol[lin];
// 		size_t aYol[lin], aZol[lin], bZol[lin];
// 		//size_t ix, iy, iz ;
// 		complex<Float> lap , tmp , acc ;
// 		int l ;
//
// 		//compute base arZ
// 		for (int l = -Ng; l < Ng+1; l++)
// 		{
// 			//would be iz-Ng, iz-Ng+1 ... iz ... iz+Ng
// 			bZol[Ng+l] = Sf + ((iz + l + Lx)%Lx)*Sf ;
// 		}
// 		  // printf("~Zol %lu %lu %lu %lu %lu\n", bZol[0] , bZol[1], bZol[2], bZol[3], bZol[4]);
// 		for (int iy = 0; iy < Lx; iy++)
// 		{
// 			//compute base arY UNFOLDED
// 			for (int l = -Ng; l < Ng+1; l++)
// 			{
// 				//would be iy-Ng, iy-Ng+1 ... iy ... iy+Ng  l =2Ng+1
// 				Yol[Ng+l] = ((iy + l + Lx )%Lx)*Lx ;
// 			}
// 				// 	printf("~Yol %lu %lu %lu %lu %lu\n", Yol[0] , Yol[1], Yol[2], Yol[3], Yol[4]);
// 			//set the Y, Z, in position
// 			for (int l = 0; l<lin; l++)
// 			{
// 				auxi   = bZol[l]+Yol[Ng];
// 				aZol[l] = auxi;
// 				auxi   = bZol[Ng]+Yol[l];
// 				aYol[l] = auxi ;
// 			}
// 			  // printf("Zol %lu %lu %lu %lu %lu\n", aZol[0] , aZol[1], aZol[2], aZol[3], aZol[4]);
// 			  // printf("Yol %lu %lu %lu %lu %lu\n", aYol[0] , aYol[1], aYol[2], aYol[3], aYol[4]);
// 			for (int ix = 0; ix < Lx; ix++)
// 			{
// 				for (int l = -Ng; l < Ng+1; l++)
// 				{
// 					//would be ix-Ng, ix-Ng+1 ... ix ... ix+Ng
// 					Xol[Ng+l] = (ix + l + Lx)%Lx ;
// 				}
// 					//  printf("Xol %lu %lu %lu %lu %lu\n", Xol[0] , Xol[1], Xol[2], Xol[3], Xol[4]);
// 					//shift to 3D cross
// 				for (int l = 0; l < lin; l++)
// 				{
// 					Zol[l] = aZol[l] + Xol[Ng];
// 					Yol[l] = aYol[l] + Xol[Ng] ;
// 				}
// 				// printf("-Zol %lu %lu %lu %lu %lu\n", Zol[0] , Zol[1], Zol[2], Zol[3], Zol[4]);
// 				for (int l = 0; l < lin; l++)
// 				{
// 					auxi = Xol[l] ;
// 					Xol[l] = auxi + aZol[Ng] ;
// 				}
// 					// printf("iziyix %d %d %d \n", iz, iy, ix);
// 					// printf("-Xol %lu %lu %lu %lu %lu\n", Xol[0] , Xol[1], Xol[2], Xol[3], Xol[4]);
// 					// 	printf("-Yol %lu %lu %lu %lu %lu\n", Yol[0] , Yol[1], Yol[2], Yol[3], Yol[4]);
// 					// 	printf("-Zol %lu %lu %lu %lu %lu\n", Zol[0] , Zol[1], Zol[2]/Sf, Zol[3], Zol[4]);
// 				//Compute
// 				// printf("pretmp .. ");fflush(stdout);
// 				tmp = ((Float) 6)*(m[Xol[Ng]]);
// 				lap = ((Float) 0 , (Float) 0);
// 				// printf("prelap .. ");
// 				for (int l = 1; l < Ng+1; l++)
// 				{
// 					//printf("Ng %d l %d \n", Ng, l);
// 					//lap +=(m[Xol[l]] + m[Xol[lin-l-1]] + m[Yol[l]] + m[Yol[lin-l-1]] + m[Zol[l]] + m[Zol[lin-l-1]] - tmp)*CO[Ng-l-1];
// 					acc = m[Xol[Ng+l]] + m[Xol[Ng-l]] + m[Yol[Ng+l]] + m[Yol[Ng-l]] + m[Zol[Ng+l]] + m[Zol[Ng-l]] - tmp ;
// 					acc *= CO[l-1];
// 					lap += acc;
// 				}
// 				tmp = m[Xol[Ng]];
// 				//lap += CO[Ng]*tmp;
// 				acc = lap*ood2 + zQ - tmp*(((Float) LL)*(tmp.real()*tmp.real() + tmp.imag()*tmp.imag() - z2));
// 				//lap is aux
// 				lap = v[Xol[Ng]-Sf];
// 				lap += acc*dzc;
// 				v[Xol[Ng]-Sf] = lap;
// 				lap *= dzd;
// 				tmp += lap;
// 				m2[Xol[Ng]] = tmp;
//
// 			}	//END X LOOP
// 		}		//END Y LOOP
// 	}			//END Z LOOP
// }



// //	----------------------------------------------------------------------------------------------------------------
// //	THIS WORKS
// //	----------------------------------------------------------------------------------------------------------------
//
//
// //										(field->m2Cpu()), (field->vCpu()),(field->mCpu()), z2, zQ, dzc, dzd, ood2, LL, Lx, S, V+S, Ng);
// template<typename Float>
// void	propSimpleCoreN (const complex<Float> *m, complex<Float> *v, complex<Float> *m2, const Float z2, const Float zQ,
// 			const Float dzc, const Float dzd, const Float ood2, const Float LL, const size_t Lx, const size_t Sf, const size_t Vf,
// 			int Ng)
// {
// 	Float CO[4] = {0.0, 0.0, 0.0, 0.0} ;
//
// 	if (Ng == 2) {
// 		CO[0] = (Float) 16/12; CO[1] = (Float) -1/12;
// 	} else if (Ng == 3) {
// 		CO[0] = (Float) 3/2; CO[1] = (Float) -3/20; CO[2] = (Float) 1/90 ;
// 	} else if (Ng == 4) {
// 		CO[0] = (Float) 8/5; CO[1] = (Float) -1/5; CO[2] = (Float) 8/315 ; CO[3] = (Float) -1/560 ;
// 	} else if (Ng == 0) {
// 		CO[0] = (Float) 0 ;
// 	} else {
// 		CO[0] = (Float) 1 ;
// 	}
//
// 	// printf("CASE Ng=%d, %lf %lf %lf %lf \n", Ng, CO[0], CO[1], CO[2], CO[3]);
//
//
// 	int lin = 2*Ng+1;
//
// 	// z loop
// 	#pragma omp parallel for default(shared) schedule(static)
// 	for (size_t iz = 0; iz < Lx; iz++)
// 	{
// 		size_t auxi , iy, ix ;
// 		size_t Xol[lin], Yol[lin], Zol[lin];
// 		size_t aYol[lin], aZol[lin], bZol[lin];
// 		//size_t ix, iy, iz ;
// 		complex<Float> lap , tmp , acc ;
// 		int l ;
//
// 		//compute base arZ
// 		for (int l = -Ng; l < Ng+1; l++)
// 		{
// 			//would be iz-Ng, iz-Ng+1 ... iz ... iz+Ng
// 			bZol[Ng+l] = Sf + ((iz + l + Lx)%Lx)*Sf ;
// 		}
// 		  // printf("~Zol %lu %lu %lu %lu %lu\n", bZol[0] , bZol[1], bZol[2], bZol[3], bZol[4]);
// 		for (int iy = 0; iy < Lx; iy++)
// 		{
// 			//compute base arY UNFOLDED
// 			for (int l = -Ng; l < Ng+1; l++)
// 			{
// 				//would be iy-Ng, iy-Ng+1 ... iy ... iy+Ng  l =2Ng+1
// 				Yol[Ng+l] = ((iy + l + Lx )%Lx)*Lx ;
// 			}
// 				// 	printf("~Yol %lu %lu %lu %lu %lu\n", Yol[0] , Yol[1], Yol[2], Yol[3], Yol[4]);
// 			//set the Y, Z, in position
// 			for (int l = 0; l<lin; l++)
// 			{
// 				auxi   = bZol[l]+Yol[Ng];
// 				aZol[l] = auxi;
// 				auxi   = bZol[Ng]+Yol[l];
// 				aYol[l] = auxi ;
// 			}
// 			  // printf("Zol %lu %lu %lu %lu %lu\n", aZol[0] , aZol[1], aZol[2], aZol[3], aZol[4]);
// 			  // printf("Yol %lu %lu %lu %lu %lu\n", aYol[0] , aYol[1], aYol[2], aYol[3], aYol[4]);
// 			for (int ix = 0; ix < Lx; ix++)
// 			{
// 				for (int l = -Ng; l < Ng+1; l++)
// 				{
// 					//would be ix-Ng, ix-Ng+1 ... ix ... ix+Ng
// 					Xol[Ng+l] = (ix + l + Lx)%Lx ;
// 				}
// 					//  printf("Xol %lu %lu %lu %lu %lu\n", Xol[0] , Xol[1], Xol[2], Xol[3], Xol[4]);
// 					//shift to 3D cross
// 				for (int l = 0; l < lin; l++)
// 				{
// 					Zol[l] = aZol[l] + Xol[Ng];
// 					Yol[l] = aYol[l] + Xol[Ng] ;
// 				}
// 				// printf("-Zol %lu %lu %lu %lu %lu\n", Zol[0] , Zol[1], Zol[2], Zol[3], Zol[4]);
// 				for (int l = 0; l < lin; l++)
// 				{
// 					auxi = Xol[l] ;
// 					Xol[l] = auxi + aZol[Ng] ;
// 				}
// 					// printf("iziyix %d %d %d \n", iz, iy, ix);
// 					// printf("-Xol %lu %lu %lu %lu %lu\n", Xol[0] , Xol[1], Xol[2], Xol[3], Xol[4]);
// 					// 	printf("-Yol %lu %lu %lu %lu %lu\n", Yol[0] , Yol[1], Yol[2], Yol[3], Yol[4]);
// 					// 	printf("-Zol %lu %lu %lu %lu %lu\n", Zol[0] , Zol[1], Zol[2]/Sf, Zol[3], Zol[4]);
// 				//Compute
// 				// printf("pretmp .. ");fflush(stdout);
// 				tmp = ((Float) 6)*(m[Xol[Ng]]);
// 				lap = ((Float) 0 , (Float) 0);
// 				// printf("prelap .. ");
// 				for (int l = 1; l < Ng+1; l++)
// 				{
// 					//printf("Ng %d l %d \n", Ng, l);
// 					//lap +=(m[Xol[l]] + m[Xol[lin-l-1]] + m[Yol[l]] + m[Yol[lin-l-1]] + m[Zol[l]] + m[Zol[lin-l-1]] - tmp)*CO[Ng-l-1];
// 					acc = m[Xol[Ng+l]] + m[Xol[Ng-l]] + m[Yol[Ng+l]] + m[Yol[Ng-l]] + m[Zol[Ng+l]] + m[Zol[Ng-l]] - tmp ;
// 					acc *= CO[l-1];
// 					lap += acc;
// 				}
// 				tmp = m[Xol[Ng]];
// 				//lap += CO[Ng]*tmp;
// 				acc = lap*ood2 + zQ - tmp*(((Float) LL)*(tmp.real()*tmp.real() + tmp.imag()*tmp.imag() - z2));
// 				//lap is aux
// 				lap = v[Xol[Ng]-Sf];
// 				lap += acc*dzc;
// 				v[Xol[Ng]-Sf] = lap;
// 				lap *= dzd;
// 				tmp += lap;
// 				m2[Xol[Ng]] = tmp;
//
// 			}	//END X LOOP
// 		}		//END Y LOOP
// 	}			//END Z LOOP
// }














//	----------------------------------------------------------------------------------------------------------------
//	NOT CHECKED
//	----------------------------------------------------------------------------------------------------------------



// //									(field->m2Cpu()), (field->vCpu()),(field->mCpu()), z2, zQ, dzc, dzd, ood2, LL, Lx, S, V+S, Ng);
// template<typename Float>
// void	propSimpleCore (const complex<Float> *m, complex<Float> *v, complex<Float> *m2, const Float z2, const Float zQ,
// 			const Float dzc, const Float dzd, const Float ood2, const Float LL, const size_t Lx, const size_t Sf, const size_t Vf)
// {
//
// 	#pragma omp parallel for default(shared) schedule(static)
// 	for (size_t idx=Sf; idx<Vf+Sf; idx++)
// 	{
// 		size_t X[3], idxPx, idxPy, idxMx, idxMy;
//
// 		complex<Float> mel, a, tmp;
//
// 		indexXeon::idx2Vec(idx, X, Lx);
//
// 		if (X[0] == Lx-1)
// 			idxPx = idx - Lx+1;
// 		else
// 			idxPx = idx+1;
//
// 		if (X[0] == 0)
// 			idxMx = idx + Lx-1;
// 		else
// 			idxMx = idx-1;
//
// 		if (X[1] == Lx-1)
// 			idxPy = idx - Sf + Lx;
// 		else
// 			idxPy = idx + Lx;
//
// 		if (X[1] == 0)
// 			idxMy = idx + Sf - Lx;
// 		else
// 			idxMy = idx - Lx;
//
// 		mel = m[idxMx] + m[idxPx] + m[idxPy] + m[idxMy] + m[idx+Sf] + m[idx-Sf];
// 		tmp = m[idx];
//
// 		a = (mel-((Float) 6.)*tmp)*ood2 + zQ - tmp*(((Float) LL)*(tmp.real()*tmp.real() + tmp.imag()*tmp.imag() - z2));
//
// 		mel = v[idx-Sf];
// 		mel += a*dzc;
// 		v[idx-Sf] = mel;
// 		mel *= dzd;
// 		tmp += mel;
// 		m2[idx] = tmp;
//
// 	}
// }

void	propSimpleKernel(Scalar *field, const double LL, const double nQcd, const double delta,
	const double dz, const double c, const double d, bool st, const int Ng)
{
	const size_t Lx = field->Length();
	const size_t S = field->Surf();
	const size_t V = field->Size();

	switch(field->Precision())
	{
		case FIELD_DOUBLE:
		{
			double *z = field->zV();
			const double zR = *z;
			const double z2 =zR*zR;
			const double zQ = 9.*pow(zR, nQcd+3.);
			const double dzc = dz*c;
			const double dzd = dz*d;
			const double ood2 = 1/(delta*delta);

			if (st == true)
			{
				field->exchangeGhosts(FIELD_M2);
				propSimpleCoreN<double>(static_cast<const complex<double>*>(field->m2Cpu()), static_cast<complex<double>*>(field->vCpu()),
						       static_cast<complex<double>*>(field->mCpu()), z2, zQ, dzc, dzd, ood2, LL, Lx, S, V+S, Ng);
			} else {
				field->exchangeGhosts(FIELD_M);
				propSimpleCoreN<double>(static_cast<const complex<double>*>(field->mCpu()), static_cast<complex<double>*>(field->vCpu()),
						       static_cast<complex<double>*>(field->m2Cpu()), z2, zQ, dzc, dzd, ood2, LL, Lx, S, V+S, Ng);
			}

			*z += dzd;

			break;
		}

		case FIELD_SINGLE:
		{
			double *z = field->zV();
			const double zR = *z;
			const float  z2 = zR*zR;
			const float  zQ = 9.f*powf(((float) zR), 3.f+((float) nQcd));
			const float  dzc = dz*c;
			const double dzd = dz*d;
			const float  ood2 = 1.f/(delta*delta);

			if (st == true)
			{
				field->exchangeGhosts(FIELD_M2);
				propSimpleCoreN<float>(static_cast<const complex<float>*>(field->m2Cpu()), static_cast<complex<float>*>(field->vCpu()),
						      static_cast<complex<float>*>(field->mCpu()), z2, zQ, dzc, dzd, ood2, LL, Lx, S, V+S, Ng);
			} else {
				field->exchangeGhosts(FIELD_M);
				propSimpleCoreN<float>(static_cast<const complex<float>*>(field->mCpu()), static_cast<complex<float>*>(field->vCpu()),
						      static_cast<complex<float>*>(field->m2Cpu()), z2, zQ, dzc, dzd, ood2, LL, Lx, S, V+S, Ng);
			}

			*z += dzd;

			break;
		}

		default:
		printf ("Not a valid precision.\n");

		break;
	}
}

void	propagateSimple(Scalar *field, const double dz, const double LL, const double nQcd, const double delta, const int Ng)
{
	propSimpleKernel(field, LL, nQcd, delta, dz, C1, D1, 0, Ng);
	propSimpleKernel(field, LL, nQcd, delta, dz, C2, D2, 1, Ng);
	propSimpleKernel(field, LL, nQcd, delta, dz, C3, D3, 0, Ng);
	propSimpleKernel(field, LL, nQcd, delta, dz, C4, D4, 1, Ng);
}
