#include <complex>
#include <random>
#include <omp.h>

#include "scalar/scalarField.h"
#include "enum-field.h"
#include "utils/index.h"
#include "utils/memAlloc.h"
#include "utils/parse.h"

#include "enum-field.h"
#include "comms/comms.h"

template<typename Float, ConfsubType SMVT>
void	randXeon (std::complex<Float> * __restrict__ m, Scalar *field, IcData ic)
{
	LogMsg(VERB_NORMAL,"[rX] Random configuration %d",SMVT);
	int	maxThreads = omp_get_max_threads();
	int	*sd;

	trackAlloc((void **) &sd, sizeof(int)*maxThreads);

	std::random_device seed;		// Totally random seed coming from memory garbage

	for (int i=0; i<maxThreads; i++)
		sd[i] = seed()*(1 + commRank());

	const size_t Nx = field->NX();
	const size_t Ny = field->NY();
	const size_t Nz = field->NZ();
	const size_t Sf = field->Surf();
	const size_t V  = field->Size();
	const double L  = field->BckGnd()->PhysSize();
	int rank = commRank();
	size_t Tz = field->TZ();
	size_t local_z_start = rank*Nz;

	/* used from ic */
	double mod0  = ic.mode0 ;
	double kCri  = ic.kcr;
	/* for string wave */
	int div = ic.kMax; // number of strings in one dimension
	/* kCri2 Interpreted as sigma in conf-minicluster in ADM Units*/
	double kCri2 = L*L/(2.0*Sf*ic.kcr*ic.kcr);
	size_t kMa   = ic.kMax;

	double kMx   = (double) ic.kMax;
	double kMy   = 0.;
	double kMz   = 0.;
	double kBase = 2.0*M_PI/Nx;
	/* for string radius  */
	double strR  = ic.mode0 > 0.0 ? ic.mode0 : 0.2501;
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

	const Float invNx = 1.0 / (Float) Nx;
	const Float invNy = 1.0 / (Float) Ny;
	const Float invNz = 1.0 / (Float) Tz;
	const Float twopi = 2.0 * M_PI;

	switch(SMVT)
	{
		case CONF_RAND:
			LogMsg(VERB_NORMAL,"[RX] >>>>> Random configuration ");
		break;
		case CONF_AXNOISE:
			LogMsg(VERB_NORMAL,"[RX] >>>>> Axnoise configuration (mod0 %.e, KCri %.e)",mod0, kCri);
		break;
		case CONF_SAXNOISE:
			LogMsg(VERB_NORMAL,"[RX] >>>>> Saxnoise configuration (mod0 %.e, KCri %.e)",mod0, kCri);
		break;
		case CONF_AX1MODE:
			LogMsg(VERB_NORMAL,"[RX] >>>>> Axion 1 mode (mod0 %.e, kx %d ky 1)",mod0, kMa);
		break;
		case CONF_MINICLUSTER:
			LogMsg(VERB_NORMAL,"[RX] >>>>> Minicluster (mod0 %.e, kCri2 %.e)",mod0, kCri2);
		break;
		case CONF_MINICLUSTER0:
			LogMsg(VERB_NORMAL,"[RX] >>>>> Minicluster (mod0 %.e, kCri2 %.e)",mod0, kCri2);
		break;
		case CONF_AXITON:
			LogMsg(VERB_NORMAL,"[RX] >>>>> Minicluster (mod0 %.e, kCri2 %.e)",mod0, kCri2);
		break;
		case CONF_STRINGXY:
		case CONF_STRINGYZ:
			LogMsg(VERB_NORMAL,"[RX] >>>>> String ");
		break;
		case CONF_PARRES:
			LogMsg(VERB_NORMAL,"[RX] >>>>> ParRes (mod0 %.e, kxyz %.e %.e %.e kCri %.e)",mod0, kMx, kMy, kMz, kCri);
		break;
		case CONF_STRWAVE:
			LogMsg(VERB_NORMAL,"[RX] >>>>> StrWav ");
		break;
		case CONF_THETAVEL:
			LogMsg(VERB_NORMAL,"[RX] >>>>> ThetVel (mod0 %.e, kxyz %.e %.e %.e kBase %.e)",mod0, kMx, kMy, kMz, kBase);
		break;
		case CONF_VELRAND:
			LogMsg(VERB_NORMAL,"[RX] >>>>> VelRand (mod0 %.e)",mod0);
		break;
		case CONF_FLAT:
			LogMsg(VERB_NORMAL,"[RX] >>>>> FLAT (mod0 %.e)",mod0);
		break;
	}

	if (SMVT != CONF_STRWAVE)
	{
		#pragma omp parallel default(shared)
		{
			int nThread = omp_get_thread_num();
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

					case CONF_AX1MODE:
					{
					    size_t X[3];
					    indexXeon::idx2Vec(idx, X, Nx, Ny);

					    const size_t ix = X[0];
					    const size_t iy = X[1];
					    const size_t iz = X[2] + local_z_start;

							const Float theta = mod0 * sin(
							    twopi * (
							        kMx * ix * invNx +
							        kMy * iy * invNy +
							        kMz * iz * invNz
							    )
							);

					    m[idx] = std::complex<Float>(cos(theta), sin(theta));
					    break;
					}

					case CONF_MINICLUSTER:
					{
					    size_t X[3];
					    indexXeon::idx2Vec(idx, X, Nx, Ny);

					    ix = X[0];
					    iy = X[1];
					    iz = X[2] + local_z_start;

					    x = static_cast<Float>(ix);
					    y = static_cast<Float>(iy);
					    z = static_cast<Float>(iz);

					    if (iz > Tz/2) { z -= static_cast<Float>(Tz); }

					    Float theta = (x - static_cast<Float>(Nx)/2.0)*(x - static_cast<Float>(Nx)/2.0)
					                + (y - static_cast<Float>(Ny)/2.0)*(y - static_cast<Float>(Ny)/2.0)
					                + z*z;

					    theta = exp(-theta*kCri2)*mod0;
					    m[idx] = std::complex<Float>(cos(theta), sin(theta));
					    break;
					}

					case CONF_MINICLUSTER0:
					{
					    size_t X[3];
					    indexXeon::idx2Vec(idx, X, Nx, Ny);

					    ix = X[0];
					    iy = X[1];
					    iz = X[2] + local_z_start;

					    x = static_cast<Float>(ix);
					    y = static_cast<Float>(iy);
					    z = static_cast<Float>(iz);

					    if (iz > Tz/2) { z -= static_cast<Float>(Tz); }
					    if (iy > Ny/2) { y -= static_cast<Float>(Ny); }
					    if (ix > Nx/2) { x -= static_cast<Float>(Nx); }

					    Float theta = x*x + y*y + z*z;
					    theta = exp(-theta*kCri2)*mod0;

					    m[idx] = std::complex<Float>(cos(theta), sin(theta));
					    break;
					}

					case CONF_AXITON:
					{
					    size_t X[3];
					    indexXeon::idx2Vec(idx, X, Nx, Ny);

					    ix = X[0];
					    iy = X[1];
					    iz = X[2] + local_z_start;

					    x = static_cast<Float>(ix);
					    y = static_cast<Float>(iy);
					    z = static_cast<Float>(iz);

					    if (iz > Tz/2) { z -= static_cast<Float>(Tz); }
					    if (iy > Ny/2) { y -= static_cast<Float>(Ny); }
					    if (ix > Nx/2) { x -= static_cast<Float>(Nx); }

					    Float theta = x*x + y*y + z*z;
					    theta = mod0/(theta*kCri2 + 1.0);

					    m[idx] = std::complex<Float>(cos(theta), sin(theta));
					    break;
					}

					case CONF_STRINGXY:
					{
					    size_t X[3];
					    indexXeon::idx2Vec(idx, X, Nx, Ny);

					    ix = X[0];
					    iy = X[1];
					    iz = X[2] + local_z_start;

					    x = static_cast<Float>(ix);
					    y = static_cast<Float>(iy);
					    z = static_cast<Float>(iz);

					    // symmetry with respect to z = kCrit
					    Float zis = z - static_cast<Float>(kCrit);
					    if ( zis >  static_cast<Float>(Tz)/2.0) zis -= static_cast<Float>(Tz);
					    if (-zis >  static_cast<Float>(Tz)/2.0) zis += static_cast<Float>(Tz);

					    Float aL = static_cast<Float>(Nx) * static_cast<Float>(strR); // radius
					    rho2 = (x - static_cast<Float>(Nx)/2.0)*(x - static_cast<Float>(Nx)/2.0)
					         + (y - static_cast<Float>(Ny)/2.0)*(y - static_cast<Float>(Ny)/2.0);

					    Float rho = sqrt(static_cast<Float>(rho2));
					    Float z2  = zis*zis;
					    Float d12 = (rho + aL)*(rho + aL) + z2;
					    Float d22 = (rho - aL)*(rho - aL) + z2;

					    Float theta = 3.14159265*(0.5 + (4.f*aL*aL - d12 - d22)/(4.f*sqrt(d12*d22)))
					                * (-0.01 + zis)/abs(-0.01 + zis);

					    m[idx] = std::complex<Float>(cos(theta), sin(theta));
					    break;
					}

					case CONF_STRINGYZ:
					{
					    size_t X[3];
					    indexXeon::idx2Vec(idx, X, Nx, Ny);

					    ix = X[0];
					    iy = X[1];
					    iz = X[2] + local_z_start;

					    x = static_cast<Float>(ix);
					    y = static_cast<Float>(iy);
					    z = static_cast<Float>(iz);

					    if (iz > Tz/2) { z -= static_cast<Float>(Tz); }

					    Float aL = static_cast<Float>(Ny) * static_cast<Float>(strR); // choose transverse scale
					    rho2 = z*z + (y - static_cast<Float>(Ny)/2.0)*(y - static_cast<Float>(Ny)/2.0);

					    Float rho = sqrt(static_cast<Float>(rho2));
					    Float x2  = (x - static_cast<Float>(Nx)/2.0)*(x - static_cast<Float>(Nx)/2.0);
					    Float d12 = (rho + aL)*(rho + aL) + x2;
					    Float d22 = (rho - aL)*(rho - aL) + x2;

					    Float theta = 3.14159265*(0.5 + (4.f*aL*aL - d12 - d22)/(4.f*sqrt(d12*d22)));

					    if (ix > Nx/2)
					        theta *= -1;

					    m[idx] = std::complex<Float>(cos(theta), sin(theta));
					    break;
					}
\
					//	ONE MODE
					case CONF_PARRES:
					{
					    size_t X[3];
					    indexXeon::idx2Vec(idx, X, Nx, Ny);

					    ix = X[0];
					    iy = X[1];
					    iz = X[2] + local_z_start;

					    // Float theta = static_cast<Float>(
					    //     mod0*cos(6.2831853*(ix*kMx + iy*kMy + iz*kMz)/static_cast<Float>(Nx))
							const Float theta = mod0 * sin(
							    twopi * (
							        kMx * ix * invNx +
							        kMy * iy * invNy +
							        kMz * iz * invNz
							    )
							);

					    m[idx] = std::complex<Float>(kCri*cos(theta), kCri*sin(theta));
					    break;
					}

					case CONF_THETAVEL:
					{
					    size_t X[3];
					    indexXeon::idx2Vec(idx, X, Nx, Ny);

					    ix = X[0];
					    iy = X[1];
					    iz = X[2] + local_z_start;

					    x = static_cast<Float>(ix);
					    y = static_cast<Float>(iy);
					    z = static_cast<Float>(iz);

							const Float thetap = mod0 * sin(
							    twopi * (
							        kMx * ix * invNx +
							        kMy * iy * invNy +
							        kMz * iz * invNz
							    )
							);
					    m[idx] = std::complex<Float>(0, thetap);
					    break;
					}

					case CONF_VELRAND:
					{
						m[idx] = std::complex<Float>(0, mod0*uni(mt64));
						break;
					}

					case CONF_FLAT:
					{
						m[idx] = std::complex<Float>(1, 0);
						break;
					}
				}
			}
		}
	}

	if (SMVT == CONF_STRWAVE)
	{
		LogMsg(VERB_NORMAL,"[RX] CONF_STRWAVE! ");

		#pragma omp parallel default(shared)
		{
		    const Float LLx = static_cast<Float>(Nx) / static_cast<Float>(div);
		    const Float LLy = static_cast<Float>(Ny) / static_cast<Float>(div);

		    #pragma omp for schedule(static)
		    for (size_t idx = 0; idx < Sf; idx++)
		    {
		        size_t ix = idx % Nx;
		        size_t iy = idx / Nx;

		        const Float x = static_cast<Float>(ix);
		        const Float y = static_cast<Float>(iy);

		        Float theta = 0.;

		        for (int nx = -div; nx < div + 2; nx++) {
		            for (int ny = -div; ny < div + 2; ny++) {
		                const Float xc = (static_cast<Float>(nx) + 0.5) * LLx;
		                const Float yc = (static_cast<Float>(ny) + 0.5) * LLy;

		                theta += static_cast<Float>(((nx + ny) & 1) ? -1.0 : 1.0)
		                       * std::atan2(y - yc, x - xc);
		            }
		        }

		        const std::complex<Float> eee(std::cos(theta), std::sin(theta));

		        for (size_t iz = 0; iz < Nz; iz++)
		            m[idx + iz*Sf] = eee;
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
		if (ic.fieldindex == FIELD_M){
		 	ma = static_cast<std::complex<double>*> (field->mStart());
			LogMsg(VERB_NORMAL,"[RC] Generating double conf in mS! ");
		}
		else if (ic.fieldindex == FIELD_V){
			ma = static_cast<std::complex<double>*> (field->vCpu());
			LogMsg(VERB_NORMAL,"[RC] Generating double conf in v! ");
		}
		else if (ic.fieldindex == FIELD_M2){
			ma = static_cast<std::complex<double>*> (field->m2Cpu());
			LogMsg(VERB_NORMAL,"[RC] Generating double conf in m2! ");
		}

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
			case CONF_VELRAND:
				randXeon<double,CONF_VELRAND> (ma, field, ic);
				break;
			case CONF_FLAT:
				randXeon<double,CONF_FLAT> (ma, field, ic);
				break;
		}
		}
		break;

		case FIELD_SINGLE:
		{
		std::complex<float>* ma;
		if (ic.fieldindex == FIELD_M){
			ma = static_cast<std::complex<float>*> (field->mStart());
			LogMsg(VERB_NORMAL,"[RC] Generating single conf in mS! ");
		}
		else if (ic.fieldindex == FIELD_V){
			ma = static_cast<std::complex<float>*> (field->vCpu());
			LogMsg(VERB_NORMAL,"[RC] Generating single conf in v! type %d",ic.smvarType);
		}
		else if (ic.fieldindex == FIELD_M2){
			ma = static_cast<std::complex<float>*> (field->m2Cpu());
			LogMsg(VERB_NORMAL,"[RC] Generating single conf in m2! ");
		}



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
			case CONF_VELRAND:
				randXeon<float,CONF_VELRAND> (ma, field, ic);
				break;
			case CONF_FLAT:
				randXeon<float,CONF_FLAT> (ma, field, ic);
				break;
			}
		}
		break;

		default:
		break;
	}
}
