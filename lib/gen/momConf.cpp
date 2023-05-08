#include<cstdio>
// #include<cstdlib>
#include<cstdlib>
#include<math.h>	/* pow */
#include<cstring>
#include<complex>
#include<random>

#include <omp.h>

#include "scalar/scalarField.h"
#include "enum-field.h"
#include "utils/memAlloc.h"
#include "comms/comms.h"
#include "utils/parse.h"
#include "gsl/gsl_sf_bessel.h"

using namespace std;

template<typename Float, MomConfType Moco>
void	momXeon (complex<Float> * __restrict__ fM, complex<Float> * __restrict__ fV, const MomParms mopa, const size_t Lx, const size_t Lz, const size_t Tz, const size_t S, const size_t V)
{
	size_t        kMax  = mopa.kMax;
	double        kCrat  = mopa.kCrt;
	double        mass2 = mopa.mass2;

	LogMsg(VERB_NORMAL,"[momXeon] Called with kMax %zu kCrit %f (kCrit es %f)", kMax, kCrat, kCrit);
	if (mopa.randommom)
		LogMsg(VERB_NORMAL,"[momXeon] random momenta");
	else
		LogMsg(VERB_NORMAL,"[momXeon] non-random momenta (fixed with --norandommom or others...)");

	std::vector<double> 	mm = mopa.mfttab;
	std::vector<double> 	ii;
	tk::spline mf;

	if (Moco == MOM_SPAX){
		for (int il = 0; il < mm.size();il++){
			ii.push_back((double) il);
		}
		mf.set_points(ii,mm);
		LogMsg(VERB_NORMAL,"[momXeon] Called SPAX, mm.size -= %d",mm.size());
	}

	/* used only for MOM_STRING*/
	std::vector<double> 	xx,yy,zz,x0;
	bool circular_loop   = true;
	Float circ_loop_pref, circ_loop_rad, k0N ;
	k0N            = 2*M_PI/Lx;
	if (Moco == MOM_STRING){
		xx = mopa.xx;
		yy = mopa.yy;
		zz = mopa.zz;
		x0 = mopa.x0;

		if (xx.size()>0){
			/* generic case */
			circular_loop = false;
			circ_loop_pref = (Float) (8*M_PI);
		} else {
			/* circular loop case */

			circ_loop_pref = (Float) (-4*M_PI*M_PI)/(mopa.k0*mopa.k0);
			circ_loop_rad  = (Float) mopa.kCrt*Lx;
			LogMsg(VERB_NORMAL,"[momXeon] CIRCULAR LOOP R=%f",circ_loop_rad);
		}
	}

	long long kmax;
	int adp = 0;
	if (kMax > Lx/2 - 1)
	{
		kmax = Lx/2 - 1;
		adp = 1;
	}
	else {
		kmax = kMax;
	}
	size_t kmax2 = kmax*kmax;
	// printf("kmax2 %d\n",kmax2);
	constexpr Float Twop = 2.0*M_PI;
	complex<Float> II = complex<Float>{0,1} ;
	Float kcrit = (Float) kCrat;
	Float bee = (Float) 4*kcrit*kcrit/(Lx*Lx);

	/* prefactors for thermal ICs */
	Float m2 = mopa.mass2;   // re,im effective mass
	Float k0 = mopa.k0;      // 2pi/L
	Float ik02 = (Float) (1/(mopa.k0*mopa.k0));
	int	maxThreads = omp_get_max_threads();
	int	*sd;

	trackAlloc((void **) &sd, sizeof(int)*maxThreads);

	std::random_device seed;		// Totally random seed coming from memory garbage

	for (int i=0; i<maxThreads; i++){
		sd[i] = seed()*(1 + commRank());
		// printf("seed %d %d\n",i,sd[i]);
	}


	const size_t LLy = Lx/commSize();
	const size_t zBase = LLy*commRank();
	const int hLx = Lx>>1;
	const int hTz = Tz>>1;
	uint   maxLx = (mopa.cmplx) ? Lx : (Lx>>1)+1;
	size_t maxSf = maxLx*Tz;
	LogMsg(VERB_NORMAL,"Size of the FT box Ny(local) %d Nz %d Nx %d\n",LLy,Tz,maxLx);
	#pragma omp parallel default(shared)
	{
		int nThread = omp_get_thread_num();


		std::mt19937_64 mt64(sd[nThread]);		// Mersenne-Twister 64 bits, independent per thread
		std::uniform_real_distribution<Float> uni(0.0, 1.0);
		std::normal_distribution<Float> distri(0.0,1.0);

		#pragma omp parallel for collapse(3) schedule(static) default(shared)
		for (uint oy = 0; oy < LLy; oy++)	// As Javier pointed out, the transposition makes y the slowest coordinate
			for (uint oz = 0; oz < Tz; oz++)
				for (uint ox = 0; ox < maxLx; ox++)
				{
					size_t idx = ox + oy*maxSf + oz*maxLx;
					// printf("maa idx %lu y %lu z %lu x %lu\n",idx,oy,ox,oz);
					int px = ox;
					int py = oy + zBase;
					int pz = oz ;

					if (px > hLx)
						px -= Lx;

					if (py > hLx)
						py -= Lx;

					if (pz > hTz)
						pz -= Tz;

					size_t modP = pz*pz + py*py + px*px;

					/* this monitors the speed of the loop */
					if (nThread ==0 && commRank() ==0 && !(py%10) && px == 0 && pz == 0)
						LogMsg (VERB_PARANOID, "[momXeon] px py pz %d %d %d", px, py, pz);

					switch (Moco)
					{
						/* Configurations that use random momenta */
						default:
							if (modP <= 3*kmax2 )
							{
								Float vl = Twop*(uni(mt64));
								Float al = distri(mt64);
								// complex<Float> marsa = exp( complex<Float>(0,vl + px*Twop/2.0+py*Twop/2.0) )*al;
								//complex<Float> marsa = exp( complex<Float>(0,vl) )*al;
								complex<Float> marsa = exp(complex<Float>(0,vl));
								if (mopa.randommom)
									marsa *= al;

								switch (Moco)
								{
									case(MOM_MFLAT):
										// fM[idx] = complex<Float>(cos(vl), sin(vl))*al;
										fM[idx] = marsa;
									break;
									case(MOM_SPAX):
										{
											//Float sc = mf((double) sqrt( (Float) modP));
											double sc = (Float) sqrt(modP);
											int b    = (int) sc;
											Float c0 = mm[b];
											Float c1 = mm[b+1];
											fM[idx]  = marsa*((Float) (c0+(c1-c0)*(sc-b)));
										}
									break;
									case(MOM_MSIN):
										{
											Float mP = sqrt(((Float) modP))/(kcrit);
											Float sc = (modP == 0) ? 1.0 : sin(mP)/mP;
										// fM[idx] = complex<Float>(cos(vl), sin(vl))*al*sc;
										fM[idx] = marsa*sc;
										}
									break;

									case(MOM_MVSINCOS):
										{
											Float mP = sqrt(((Float) modP))/(kcrit);
											// v to m2 m to v
											Float sc = (modP == 0) ? 1.0 : sin(mP)/mP;
											fM[idx] = marsa*sc;
											sc = (modP == 0) ? 0.0 : (cos(mP) - sc) ;
											fV[idx] = marsa*sc;
										}
									break;

									case(MOM_MVTHERMAL):
										{
											// needs mass!
											Float mP = sqrt(sqrt(((Float) modP)*k0*k0 + m2));
											Float mE = sqrt(1./(exp(mP*mP/kcrit)-1.));
											// field (goes to V array)
											// the zero mode has infinite thermal expectation value in the continuum
											// discrete version not ... 0? adjusted to VEV? ...
											fM[idx] = (modP == 0) ? 0 : marsa*mE/mP ;
											// velocity (goes into M)
											// the zero mode is finite mE/mP -> kcrit
											vl = Twop*(uni(mt64));
											al = distri(mt64);
											marsa   = exp( complex<Float>(0,vl) )*al;
											fV[idx] = (modP == 0) ? marsa*sqrt(kcrit) : marsa*mE*mP ;
										}
									break;

									case(MOM_MEXP):
										{
											Float mP = sqrt(((Float) modP))/(kcrit);
											Float sc = (modP == 0) ? 1.0 : exp(-mP);
											fM[idx] = marsa*sc;
										}
									break;

									default:
									case(MOM_MEXP2):
										{
											Float mP = ((Float) modP)/(kcrit*kcrit);
											Float sc = (modP == 0) ? 1.0 : exp(-mP);
											fM[idx] = marsa*sc;
										}
									break;
									case(MOM_COLE):
										{
											Float mP = ((Float) modP)/(kcrit*kcrit);
											Float sc = (modP == 0) ? 1.0 : exp(-bee*mP)/pow(mP+1,1.5);
											fM[idx] = marsa*sc;
										}
									break;
									case(MOM_KCOLE):
										{
											Float mP = ((Float) modP)/(kcrit*kcrit);
											Float sc = (modP == 0) ? 1.0 : exp(-mP); //*sqrt(mP)
											fM[idx] = marsa*sc;
										}
									break;
								}

							} // END if
							else {
								fM[idx] = complex<Float>(0,0);
							}
						break;

						case(MOM_STRING):
						{
								/* Theory in pieces/duality.pdf
								We compute the sum over string elements for each momentum
								and add the required prefactors.
								if no string coordinates are given use the
								circular loop */
								Float p  = (Float) std::sqrt( (Float) modP);
								if (circular_loop)
								{
									Float p  = (Float) std::sqrt( (Float) (px*px+py*py));
									Float we = p == 0 ? circ_loop_rad*k0N/2 : ((Float) gsl_sf_bessel_J1 (k0N*p*circ_loop_rad))/ ((Float) p) ;
									fM[idx] = complex<Float>(0,circ_loop_pref*pz*we/modP) * exp(complex<Float>(0,-k0N*(px*x0[0]+py*x0[1]+pz*x0[2])));
									printf("rank (%d %d) br px py pz %d %d %d -> %f %f pref %f k0N %f we %f\n",commRank(),zBase,px,py,pz,real(fM[idx]),imag(fM[idx]),circ_loop_pref, k0N, circ_loop_pref*pz*we);
										// * exp(complex<Float>(0,-k0N*(px*x0[0]+py*x0[1]+pz*x0[2])));
								//	printf("br px py pz %d %d %d -> %f %f pref %f \n",px,py,pz,real(fM[idx]),imag(fM[idx]), circ_loop_pref*pz/std::pow(p,3));
								// fM[idx] = complex<Float>(0.,0.);
								// 	if (idx == 10) {
								// 		fM[idx] = complex<Float>(1.0,0);
								// 		printf("br px py pz %d %d %d -> %f %f pref %f \n",px,py,pz,real(fM[idx]),imag(fM[idx]),
								// 		circ_loop_pref*pz/std::pow(p,3));
								// 		}
								}
								else // read points and connect them to calculate
								{
									complex<Float> su = (0,0);

									/* most of the grid */
									if (px != 0){
										for (int il = 0; il < xx.size()-1;il++)
											{
												Float dx = xx[il+1]-xx[il];
												Float dy = yy[il+1]-yy[il];
												Float dz = zz[il+1]-zz[il];
												Float pdx = px*dx+py*dy+pz*dz;
												complex<Float> aux = (1.0,0.);
												if (pdx != 0.0){
													// aux = (exp(complex<Float>(0,-k0N*pdx))-complex<Float>((1.0,0)))/complex<Float>((0.,-k0N*pdx));
													aux = complex<Float>(sin(k0N*pdx),cos(k0N*pdx)-1)/(k0N*pdx);}
												// printf("aux idx %lu %f (%f,%f)\n",idx,k0N*pdx,aux.real(),aux.imag());
												// py Wz - pzWy
												su += (py*dz-pz*dy)/(px*modP) * exp(complex<Float>(0,-k0N*(px*xx[il]+py*yy[il]+pz*zz[il])))*aux;
											}
										}
									else if (py != 0)
										{
											for (int il = 0; il < xx.size();il++){
												Float dx = xx[il+1]-xx[il];
												Float dy = yy[il+1]-yy[il];
												Float dz = zz[il+1]-zz[il];
												Float pdx = px*dx+py*dy+pz*dz;
												complex<Float> aux = (1.0,0.);
												if (pdx != 0.0){
													// aux = (exp(complex<Float>(0,-k0N*pdx))-complex<Float>((1.0,0)))/complex<Float>((0.,-k0N*pdx));
													aux = complex<Float>(sin(k0N*pdx),cos(k0N*pdx)-1)/(k0N*pdx);}
												// pz Wx - pxWz
												su += (pz*dx-px*dz)/(py*modP) * exp(complex<Float>(0,-k0N*(px*xx[il]+py*yy[il]+pz*zz[il])))*aux;
											}
										}
									else if (pz != 0)
										{ // px=py=0 case
												for (int il = 0; il < xx.size();il++){
													Float dx = xx[il+1]-xx[il];
													Float dy = yy[il+1]-yy[il];
													Float dz = zz[il+1]-zz[il];
													Float pdx = px*dx+py*dy+pz*dz;
													complex<Float> aux = (1.0,0.);
													if (pdx != 0.0){
														// aux = (exp(complex<Float>(0,-k0N*pdx))-complex<Float>((1.0,0)))/complex<Float>((0.,-k0N*pdx));
														aux = complex<Float>(sin(k0N*pdx),cos(k0N*pdx)-1)/(k0N*pdx);}
													// px Wy - pyWx
													su += (px*dy-py*dx)/(pz*modP) * exp(complex<Float>(0,-k0N*(px*xx[il]+py*yy[il]+pz*zz[il])))*aux;
												}
										}
									// else p=0, keep fM=0

									fM[idx] = su * ik02 ; //* exp(-k0N*k0N*modP/((Float) 8.))
								} // end case generic loop

						}
						break;
					} //END 1st switch-moco
		} // END triple loop
	}

	// In some cases we want to adjust the zero mode by hand
	// not in SPAX!
	switch (Moco){
		case(MOM_SPAX):
		break;

		case(MOM_STRING):
		if (commRank() == 0){
			fM[0] = complex<Float>(0,0);
			LogMsg (VERB_NORMAL, "mode0 set to %f %f in rank %d", real(fM[0]), imag(fM[0]), commRank());
		}
		break;

		default:
		if (commRank() == 0)
			{
				if ( mode0 < 3.141597 )
				{
					LogMsg (VERB_NORMAL, "mode0 set to %f in rank %d", mode0, commRank());
					// fM[0] = complex<Float>(cos(mode0), sin(mode0));
					if (Moco == MOM_MVSINCOS || Moco == MOM_MVTHERMAL)
						fV[0] = exp( complex<Float>(0,mode0) );
						else
						fM[0] = exp( complex<Float>(0,mode0) );
				}
				else
				{
					if (Moco == MOM_MVSINCOS || Moco == MOM_MVTHERMAL)
						mode0 = atan2(fV[0].imag(),fV[0].real());
						else
						mode0 = atan2(fM[0].imag(),fM[0].real());

					LogMsg (VERB_NORMAL, "mode0 is been randomly set to %f by rank %d", mode0, commRank());
				}
			}
		break;
	}

	trackFree((void *) sd);
}





void	momConf (Scalar *field, MomParms mopa)
{
	LogMsg (VERB_HIGH, "[MoCo] Called momConf Moco = %d",mopa.mocoty);
	LogFlush();

	const size_t n1 = field->Length();
	const size_t n2 = field->Surf();
	const size_t n3 = field->Size();
	const size_t Lz = field->Depth();
	const size_t Tz = field->TotalDepth();

	MomConfType   Moco  = mopa.mocoty;

	// size_t        kMax  = mopa.kMax;
	// double        kCrt  = mopa.kCrt;
	// double        mass2 = mopa.mass2;
	// FieldType     ftype = mopa.mass2;


	const size_t offset = field->DataSize()*n2;

	switch (field->Precision())
	{
		case FIELD_DOUBLE:
		{
			complex<double>* ma = static_cast<complex<double>*> (mopa.mp);
			complex<double>* va = static_cast<complex<double>*> (field->vCpu());
			// Wild modification
			//if (field->LowMem())
			//ma = static_cast<complex<double>*> (field->mStart());
			//else
			//	ma = static_cast<complex<double>*> (field->m2Cpu());

			switch(Moco)
			{
				case MOM_MFLAT:
				momXeon<double, MOM_MFLAT> (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
				case MOM_MSIN:
				momXeon<double, MOM_MSIN> (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
				case MOM_MVSINCOS:
				momXeon<double, MOM_MVSINCOS> (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
				case MOM_MVTHERMAL:
				momXeon<double, MOM_MVTHERMAL> (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
				default:
				case MOM_MEXP:
				momXeon<double, MOM_MEXP> (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
				case MOM_MEXP2:
				momXeon<double, MOM_MEXP2> (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
				case MOM_COLE:
				momXeon<double, MOM_COLE>  (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
				case MOM_SPAX:
				momXeon<double, MOM_SPAX>  (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
				case MOM_STRING:
				momXeon<double, MOM_STRING>  (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
			}

		// if (field->LowMem())
		// 	momXeon<double, Moco> (static_cast<complex<double>*> (field->mStart()), static_cast<complex<double>*> (field->vCpu()), mopa, n1, Lz, Tz, n2, n3);
		// else
		// 	momXeon<double, Moco> (static_cast<complex<double>*> (field->m2Cpu()), static_cast<complex<double>*> (field->vCpu()), mopa, n1, Lz, Tz, n2, n3);
		}
		break;

		case FIELD_SINGLE:
		{
			complex<float>* ma = static_cast<complex<float>*> (mopa.mp);
			complex<float>* va = static_cast<complex<float>*> (field->vCpu());
			//WILD modification
			//if (field->LowMem())
			//ma = static_cast<complex<float>*> (field->mStart());
			//else
			//	ma = static_cast<complex<float>*> (field->m2Cpu());

			switch(Moco)
			{
				case MOM_MFLAT:
				momXeon<float, MOM_MFLAT> (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
				case MOM_MSIN:
				momXeon<float, MOM_MSIN> (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
				case MOM_MVSINCOS:
				momXeon<float, MOM_MVSINCOS> (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
				case MOM_MVTHERMAL:
				momXeon<float, MOM_MVTHERMAL> (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
				default:
				case MOM_MEXP:
				momXeon<float, MOM_MEXP> (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
				case MOM_MEXP2:
				momXeon<float, MOM_MEXP2> (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
				case MOM_COLE:
				momXeon<float, MOM_COLE>  (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
				case MOM_SPAX:
				momXeon<float, MOM_SPAX>  (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;
				case MOM_STRING:
				momXeon<float, MOM_STRING>  (ma, va, mopa, n1, Lz, Tz, n2, n3);
				break;

			}
		// if (field->LowMem())
		// 	momXeon<float, Moco> (static_cast<complex<float> *> (field->mStart()), static_cast<complex<double>*> (field->vCpu()), kMax, static_cast<float>(kCrt), n1, Lz, Tz, n2, n3);
		// else
		// 	momXeon<float, Moco> (static_cast<complex<float> *> (field->m2Cpu()), static_cast<complex<double>*> (field->vCpu()), kMax, static_cast<float>(kCrt), n1, Lz, Tz, n2, n3);
		// break;
		}
		break;

		default:
		break;
	}
}
