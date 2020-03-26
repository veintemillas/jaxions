#include <complex>
#include <random>
#include <omp.h>

#include "scalar/scalarField.h"
#include "enum-field.h"
#include "utils/memAlloc.h"
#include "utils/parse.h"

#include "enum-field.h"
#include "comms/comms.h"

template<typename Float, ConfsubType SMVT>
void	randXeon (std::complex<Float> * __restrict__ m, Scalar *field, IcData ic)
{
	int	maxThreads = omp_get_max_threads();
	int	*sd;

	trackAlloc((void **) &sd, sizeof(int)*maxThreads);

	std::random_device seed;		// Totally random seed coming from memory garbage

	for (int i=0; i<maxThreads; i++)
		sd[i] = seed()*(1 + commRank());

	const size_t Lx = field->Length();
	const size_t Sf = field->Surf();
	const size_t V  = field->Size();
	const double L  = field->BckGnd()->PhysSize();

	/* used from ic */
	double mod0  = ic.mode0;
	double kCri  = ic.kcr;
	/* Interpreted as inverse sigma in conf-minicluster in ADM Units*/
	double kCri2 = ic.kcr*ic.kcr*L*L/(2.0*Sf);
	size_t kMa   = ic.kMax;

	double kMx   = (double) ic.kMax;
	double kMy   = 0.;
	double kMz   = 0.;
	double kBase = 2.0*M_PI/Lx;

	/* this is useless in many applications but harms not much */
	FILE *cacheFile = nullptr;
	if (((cacheFile  = fopen("./kkk.dat", "r")) == nullptr)){
		LogMsg(VERB_NORMAL,"No kkk.dat file use defaults k = (kMax,1,0)");
	} else {
		fscanf (cacheFile ,"%lf ", &kMx);
		fscanf (cacheFile ,"%lf ", &kMy);
		fscanf (cacheFile ,"%lf ", &kMz);
		LogMsg(VERB_NORMAL,"[rand] kkk.dat file used k = (%.2f,%.2f,%.2f)",kMx,kMy,kMz);
	}

	#pragma omp parallel default(shared)
	{
		int nThread = omp_get_thread_num();
		int rank = commRank();
		size_t Lz = Lx/commSize();
		size_t local_z_start = rank*Lz;
		//printf("rank %d (t %d)-> N=%d Lz %d lzs = %d \n", rank, nThread, Lx, Lz, local_z_start);

		std::mt19937_64 mt64(sd[nThread]);		// Mersenne-Twister 64 bits, independent per thread
		std::uniform_real_distribution<Float> uni(-1.0, 1.0);

		#pragma omp for schedule(static)	// This is NON-REPRODUCIBLE, unless one thread is used. Alternatively one can fix the seeds
		for (size_t idx=0; idx<V; idx++)
		{
			size_t ix, iy, iz, rho2;
			int     x,  y,  z;

			switch (SMVT)
			{
				case CONF_RAND:
				//RANDOM INITIAL CONDITIONS
				{
					m[idx]   = std::complex<Float>(uni(mt64), uni(mt64));
					break;
				}

				//RANDOM AXIONS AROUND CP CONSERVING MINIMUM
				case CONF_AXNOISE:
				{
					Float theta  = mod0 + uni(mt64)*kCri ;
					m[idx] = std::complex<Float>(cos(theta), sin(theta));
					break;
				}

				case CONF_SAXNOISE:
				{
					Float theta  = uni(mt64)*kCri + 1.;
					m[idx] = std::complex<Float>(theta*cos(mod0), theta*sin(mod0));
					break;
				}

				//	ONE MODE
				case CONF_AX1MODE:
				{
					// pidx = idx-Sf;
					iz = idx/Sf + local_z_start;
					iy = (idx%Sf)/Lx ;
					ix = (idx%Sf)%Lx ;
					Float theta = ((Float) mod0*cos(6.2831853*(ix*kMa + iy)/Lx));
					m[idx] = std::complex<Float>(cos(theta), sin(theta));
					break;
				}

				case CONF_MINICLUSTER:
				//	MINICLUSTER CENTERED AT GRID
				{
					// pidx = idx-Sf;
					iz = idx/Sf + local_z_start;
					iy = (idx%Sf)/Lx ;
					ix = (idx%Sf)%Lx ;
					z = iz;
					y = iy;
					x = ix;
					if (iz>Lx/2) { z = z-Lx; }
					Float theta = ((Float) ((x-Lx/2)*(x-Lx/2)+(y-Lx/2)*(y-Lx/2)+z*z));
					theta = exp(-theta*kCri2)*mod0;
					m[idx] = std::complex<Float>(cos(theta), sin(theta));
				break;
				}

				//	MINICLUSTER CENTERED AT ZERO
				case CONF_MINICLUSTER0:
				{
					// pidx = idx-Sf;
					iz = idx/Sf + local_z_start;
					iy = (idx%Sf)/Lx ;
					ix = (idx%Sf)%Lx ;
					z = iz;
					y = iy;
					x = ix;
					if (iz>Lx/2) { z = z-Lx; }
					if (iy>Lx/2) { y = y-Lx; }
					if (ix>Lx/2) { x = x-Lx; }
					Float theta = ((Float) (x*x + y*y + z*z));
					theta = exp(-theta*kCri2)*mod0;
					m[idx] = std::complex<Float>(cos(theta), sin(theta));
					break;
				}

				case CONF_AXITON:
				{
					// pidx = idx-Sf;
					iz = idx/Sf + local_z_start;
					iy = (idx%Sf)/Lx ;
					ix = (idx%Sf)%Lx ;
					z = iz;
					y = iy;
					x = ix;
					if (iz>Lx/2) { z = z-Lx; }
					if (iy>Lx/2) { y = y-Lx; }
					if (ix>Lx/2) { x = x-Lx; }
					Float theta = ((Float) (x*x + y*y + z*z));
					theta = mod0/(theta*kCri2 + 1.0);
					m[idx] = std::complex<Float>(cos(theta), sin(theta));
					break;
				}

				//// if(ix<2)
				//// {
				//// 	printf("MINICLUSTER data! %d %d (%d,%d,%d) %f %f \n",idx, (idx%Sf)%Lx, ix,iy,iz,m[idx].real(),m[idx].imag());
				//// }
				//	STRING XY
				case CONF_STRINGXY:
				{
					iz = idx/Sf + local_z_start;
					iy = (idx%Sf)/Lx ;
					ix = (idx%Sf)%Lx ;
					z = iz;
					y = iy;
					x = ix;
					//CENTERED AT GRID, z=0
					if (iz>Lx/2) { z = z-Lx; }
					Float aL = ((Float) Lx)/4.01;	//RADIUS
					rho2 = (x-Lx/2)*(x-Lx/2)+(y-Lx/2)*(y-Lx/2);
					Float rho = sqrt((Float) rho2)	;
					Float z2 = ((Float) z*z) ;
					Float d12 = (rho + aL)*(rho + aL) + z2 ;
					Float d22 = (rho - aL)*(rho - aL) + z2 ;
					// d12 /= ((Float) Sf) ;
					// d22 /= ((Float) Sf) ;
					Float zis = (Float) z ;
					Float theta = 3.14159265*(0.5 + (4.f*aL*aL - d12 - d22)/(4.f*sqrt(d12*d22)))*(-0.5 + zis)/abs(-0.5 + zis)	;
					m[idx] = std::complex<Float>(cos(theta), sin(theta));
					break;
				}

				//	STRING YZ
				case CONF_STRINGYZ:
				{
					iz = idx/Sf + local_z_start;
					iy = (idx%Sf)/Lx ;
					ix = (idx%Sf)%Lx ;
					z = iz;
					y = iy;
					x = ix;
					//CENTERED AT GRID, z=0
					if (iz>Lx/2) { z = z-Lx; }
					Float aL = ((Float) Lx)/4.01;	//RADIUS
					rho2 = (z)*(z)+(y-Lx/2)*(y-Lx/2);
					Float rho = sqrt((Float) rho2)	;
					Float z2 = ((Float) ((x-Lx/2)*(x-Lx/2))) ;
					Float d12 = (rho + aL)*(rho + aL) + z2 ;
					Float d22 = (rho - aL)*(rho - aL) + z2 ;
					// d12 /= ((Float) Sf) ;
					// d22 /= ((Float) Sf) ;
					//Float zis = (Float) x ;
					Float theta = (0.5 + (4.f*aL*aL - d12 - d22)/(4.f*sqrt(d12*d22)))	;
					theta = 3.14159265*theta*theta	;
					if (ix>Lx/2)
						theta *= -1 ;

					m[idx] = std::complex<Float>(cos(theta), sin(theta));
					break;
				}
				//	ONE MODE
				case CONF_PARRES:
				{
					// pidx = idx-Sf;
					iz = idx/Sf + local_z_start;
					iy = (idx%Sf)/Lx ;
					ix = (idx%Sf)%Lx ;
					Float theta = ((Float) mod0*cos(6.2831853*(ix*kMx + iy*kMy + iz*kMz)/Lx));
					m[idx] = std::complex<Float>(kCri*cos(theta), kCri*sin(theta));
					break;
				}

				// case CONF_STRWAVE:
				// {
				// 	iz = idx/Sf + local_z_start;
				// 	iy = (idx%Sf)/Lx ;
				// 	ix = (idx%Sf)%Lx ;
				// 	z = iz;
				// 	y = iy;
				// 	x = ix;
				// 	Float aL = ((Float) Lx)/4.01;	//RADIUS
				// 	rho2 = (y-Lx/2)*(y-Lx/2);
				// 	Float rho = sqrt((Float) rho2)	;
				// 	Float z2 = ((Float) ((x-Lx/2)*(x-Lx/2))) ;
				// 	Float d12 = (rho + aL)*(rho + aL) + z2 ;
				// 	Float d22 = (rho - aL)*(rho - aL) + z2 ;
				// 	Float theta = (0.5 + (4.f*aL*aL - d12 - d22)/(4.f*sqrt(d12*d22)))	;
				// 	theta = 3.14159265*theta*theta	;
				// 	if (ix>Lx/2)
				// 		theta *= -1 ;
				//
				// 	// theta += ((Float) mod0*cos(6.2831853*(ix*kMx + iy*kMy + iz*kMz)/Lx));
				// 	m[idx] = std::complex<Float>(cos(theta), sin(theta));
				// 	break;
				// }

				case CONF_STRWAVE:
				{
					iz = idx/Sf + local_z_start;
					iy = (idx%Sf)/Lx ;
					ix = (idx%Sf)%Lx ;
					z = iz;
					y = iy;
					x = ix;
					Float L1 = ((Float) Lx)/4.01;
					Float L3 = ((Float) Lx)*3.01/4.01;
					Float LL = ((Float) Lx)/2.;
					Float theta = 0.;
					for (int nx = -2 ; nx < 4; nx++){
						for (int ny = -2 ; ny < 4; ny++){
							theta += pow(-1,nx+ny)*std::atan2(y-((Float) ny + 0.5)*LL,x-((Float) nx + 0.5)*LL);
						}
					}
					m[idx] = std::complex<Float>(cos(theta), sin(theta));
					break;
				}


				case CONF_THETAVEL:
				{
					// pidx = idx-Sf;
					iz = idx/Sf + local_z_start;
					iy = (idx%Sf)/Lx ;
					ix = (idx%Sf)%Lx ;
					Float thetap = ((Float) mod0*sin(kBase*(ix*kMx + iy*kMy + iz*kMz)));
					m[idx] = std::complex<Float>(0., thetap);
					break;
				}

			}
		}
	}

	trackFree((void *) sd);
}

void	randConf (Scalar *field, IcData ic)
{
	switch (field->Precision())
	{
		case FIELD_DOUBLE:
		{
		std::complex<double>* ma;
		if (ic.fieldindex == FIELD_M)
		 	ma = static_cast<std::complex<double>*> (field->mStart());
		else if (ic.fieldindex == FIELD_V)
			ma = static_cast<std::complex<double>*> (field->vCpu());
		else if (ic.fieldindex == FIELD_M2)
			ma = static_cast<std::complex<double>*> (field->m2Cpu());

		switch (ic.smvarType)
		{
			case CONF_RAND:
				randXeon<double,CONF_RAND>(ma, field, ic);
				break;
			case CONF_STRINGXY:
				randXeon<double,CONF_STRINGXY>(ma, field, ic);
				break;
			case CONF_STRINGYZ:
				randXeon<double,CONF_STRINGYZ> (ma, field, ic);
				break;
			case CONF_MINICLUSTER0:
				randXeon<double,CONF_MINICLUSTER0> (ma, field, ic);
				break;
			case CONF_MINICLUSTER:
				randXeon<double,CONF_MINICLUSTER> (ma, field, ic);
				break;
			case CONF_AXNOISE:
				randXeon<double,CONF_AXNOISE> (ma, field, ic);
				break;
			case CONF_SAXNOISE:
				randXeon<double,CONF_SAXNOISE> (ma, field, ic);
				break;
			case CONF_AX1MODE:
				randXeon<double,CONF_AX1MODE> (ma, field, ic);
				break;
			case CONF_PARRES:
				randXeon<double,CONF_PARRES> (ma, field, ic);
				break;
			case CONF_AXITON:
				randXeon<double,CONF_PARRES> (ma, field, ic);
				break;
			case CONF_STRWAVE:
				randXeon<double,CONF_STRWAVE> (ma, field, ic);
				break;
			case CONF_THETAVEL:
				randXeon<double,CONF_THETAVEL> (ma, field, ic);
				break;
		}
		}
		break;

		case FIELD_SINGLE:
		{
		std::complex<float>* ma;
		if (ic.fieldindex == FIELD_M)
			ma = static_cast<std::complex<float>*> (field->mStart());
		else if (ic.fieldindex == FIELD_V)
			ma = static_cast<std::complex<float>*> (field->vCpu());
		else if (ic.fieldindex == FIELD_M2)
			ma = static_cast<std::complex<float>*> (field->m2Cpu());


		switch (ic.smvarType)
		{
			case CONF_RAND:
				randXeon<float,CONF_RAND> (ma, field, ic);
				break;
			case CONF_STRINGXY:
				randXeon<float,CONF_STRINGXY> (ma, field, ic);
				break;
			case CONF_STRINGYZ:
				randXeon<float,CONF_STRINGYZ> (ma, field, ic);
				break;
			case CONF_MINICLUSTER0:
				randXeon<float,CONF_MINICLUSTER0> (ma, field, ic);
				break;
			case CONF_MINICLUSTER:
				randXeon<float,CONF_MINICLUSTER> (ma, field, ic);
				break;
			case CONF_AXNOISE:
				randXeon<float,CONF_AXNOISE> (ma, field, ic);
				break;
			case CONF_SAXNOISE:
				randXeon<float,CONF_SAXNOISE> (ma, field, ic);
				break;
			case CONF_AX1MODE:
				randXeon<float,CONF_AX1MODE> (ma, field, ic);
				break;
			case CONF_PARRES:
				randXeon<float,CONF_PARRES> (ma, field, ic);
				break;
			case CONF_AXITON:
				randXeon<float,CONF_AXITON> (ma, field, ic);
				break;
			case CONF_STRWAVE:
				randXeon<float,CONF_STRWAVE> (ma, field, ic);
				break;
			case CONF_THETAVEL:
				randXeon<float,CONF_THETAVEL> (ma, field, ic);
				break;
			}
		}
		break;

		default:
		break;
	}
}
