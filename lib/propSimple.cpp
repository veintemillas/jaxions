#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include "scalarField.h"
#include "enum-field.h"
#include "RKParms.h"
#include "index.h"

using namespace std;

//										(field->m2Cpu()), (field->vCpu()),(field->mCpu()), z2, zQ, dzc, dzd, ood2, LL, Lx, S, V+S, Ng);
template<typename Float>
void	propSimpleCoreN (const complex<Float> *m, complex<Float> *v, complex<Float> *m2, const Float z2, const Float zQ,
			const Float dzc, const Float dzd, const Float ood2, const Float LL, const size_t Lx, const size_t Sf, const size_t Vf,
			int Ng)
{
	Float CO[4]  ;

	if (Ng == 2)
	{
		CO[0] = -7.50000000; CO[1] = 1.333333333; CO[2] = -0.083333333 ;
	}
	else if (Ng == 3)
	{
		CO[0] = -8.16666666; CO[1] = 1.500000000; CO[2] = -0.150000000 ; CO[3] = 0.01111111 ;
	}
	else
	{
		CO[0] = -6.00000000; CO[1] = 1.000000000;
		if (Ng != 1)
		{
			printf("Unknown gradient!\n");
		}
	}
	//printf("%f %f %f %f \n", CO[0], CO[1], CO[2], CO[3]);
	// switch (Ng)
	// {
	// 	case 1:	Float CO[2]; CO[0] = -6.00000; CO[1] = 1.000000; break;
	// 	case 2: Float CO[3]; CO[0] = -7.50000; CO[1] = 1.333333; CO[2] = -0.083333 ; break;
	// 	case 3: Float CO[4]; CO[0] = -8.16666; CO[1] = 1.500000; CO[2] = -0.150000 ; CO[3]; = 0.01111 ; break;
	// 	default: printf("parameters not defined - switch to 1 neighbour\n");
	// 					Float CO[2]; CO[0] = -6.00000; CO[1] = 1.000000; break;
	// }
	int lin = 2*Ng+1;
	//	Float  arX[lin], arY[lin], arZ[lin];

	// z loop
	#pragma omp parallel for default(shared) schedule(static)
	for (size_t iz = 0; iz < Lx; iz++)
	{
		size_t auxi , iy, ix ;
		size_t Xol[lin], Yol[lin], Zol[lin];
		size_t aYol[lin], aZol[lin];
		//size_t ix, iy, iz ;
		complex<Float> lap , tmp , acc ;
		int l ;

		//compute base arZ
		for (int l = -Ng; l < Ng+1; l++)
		{
			//would be iz-Ng, iz-Ng+1 ... iz ... iz+Ng
			aZol[Ng+l] = Sf + ((iz + l + Lx)%Lx)*Sf ;
		}
		//printf("aZol %lu %lu %lu %lu %lu\n", aZol[0] , aZol[1], aZol[2], aZol[3], aZol[4]);
		for (int iy = 0; iy < Lx; iy++)
		{
			//compute base arY UNFOLDED
			for (int l = -Ng; l < Ng+1; l++)
			{
				//would be iy-Ng, iy-Ng+1 ... iy ... iy+Ng  l =2Ng+1
				aYol[Ng+l] = ((iy + l + Lx )%Lx)*Lx ;
			}
			//printf("aYol %lu %lu %lu %lu %lu\n", aYol[0] , aYol[1], aYol[2], aYol[3], aYol[4]);
			//set the Y, Z, in position
			for (int l = 0; l<lin; l++)
			{
				//Xol is aux here
				auxi   = aZol[l]+aYol[Ng];
				Zol[l] = auxi;
				auxi   = aZol[Ng]+aYol[l];
				Yol[l] = auxi ;
			}
			//printf("Zol %lu %lu %lu %lu %lu\n", Zol[0] , Zol[1], Zol[2], Zol[3], Zol[4]);
			for (int ix = 0; ix < Lx; ix++)
			{
				for (int l = -Ng; l < Ng+1; l++)
				{
					//would be ix-Ng, ix-Ng+1 ... ix ... ix+Ng
					Xol[Ng+l] = (ix + l + Lx)%Lx ;
				}
					//printf("Xol %lu %lu %lu %lu %lu\n", Xol[0] , Xol[1], Xol[2], Xol[3], Xol[4]);
					//shift to 3D cross
				for (int l = 0; l < lin; l++)
				{
					auxi = Zol[l] + Xol[Ng];
					Zol[l] = auxi ;
					auxi = Yol[l] + Xol[Ng] ;
					Yol[l] = auxi ;
				}
				//printf("Zol %lu %lu %lu %lu %lu\n", Zol[0] , Zol[1], Zol[2], Zol[3], Zol[4]);
				for (int l = 0; l < lin; l++)
				{
					auxi = Xol[l] ;
					Xol[l] = auxi + aZol[Ng] + aYol[Ng] ;
				}
				//printf("Xol %lu %lu %lu %lu %lu\n", Zol[0] , Zol[1], Zol[2], Zol[3], Zol[4]);
				//printf("Yol %lu %lu %lu %lu %lu\n", Yol[0] , Yol[1], Yol[2], Yol[3], Yol[4]);
				//printf("Zol %lu %lu %lu %lu %lu\n", Zol[0] , Zol[1], Zol[2], Zol[3], Zol[4]);
				//Compute
					lap = 0;
				for (int l = 0; l < Ng; l++)
				{
					lap += CO[Ng-l]*(m[Xol[l]] + m[Xol[lin-l-1]] + m[Yol[l]] + m[Yol[lin-l-1]] + m[Zol[l]] + m[Zol[lin-l-1]]);
				}
				tmp = m[Xol[Ng]];
				lap += CO[Ng]*tmp;

				acc = lap*ood2 + zQ - tmp*(((Float) LL)*(tmp.real()*tmp.real() + tmp.imag()*tmp.imag() - z2));
				//lap is aux
				lap = v[Xol[Ng]-Sf];
				lap += acc*dzc;
				v[Xol[Ng]-Sf] = lap;
				lap *= dzd;
				tmp += lap;
				m2[Xol[Ng]] = tmp;

			}	//END X LOOP
		}		//END Y LOOP
	}			//END Z LOOP
}

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
