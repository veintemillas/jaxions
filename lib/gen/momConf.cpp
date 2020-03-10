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

using namespace std;

template<typename Float, MomConfType Moco>
void	momXeon (complex<Float> * __restrict__ fM, complex<Float> * __restrict__ fV, const MomParms mopa, const size_t Lx, const size_t Lz, const size_t Tz, const size_t S, const size_t V)
{
	size_t        kMax  = mopa.kMax;
	double        kCrat  = mopa.kCrt;
	double        mass2 = mopa.mass2;

	LogMsg(VERB_NORMAL,"[momXeon] Called with kMax %zu kCrit %f (kCrit es %f)", kMax, kCrat, kCrit);
	std::vector<double> 	mm = mopa.mfttab;
	std::vector<double> 	ii;
	tk::spline mf;

	if (Moco == MOM_SPAX){
		for (int il = 0; il < mm.size();il++){
			ii.push_back((double) il);
		}
		mf.set_points(ii,mm);
		LogMsg(VERB_NORMAL,"[momXeon] Called SPAX");
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
// printf("purrum y %d z %d x %d\n",LLy,Tz,maxLx);
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

					if (modP <= 3*kmax2 )
					{
						Float vl = Twop*(uni(mt64));
						Float al = distri(mt64);
						// complex<Float> marsa = exp( complex<Float>(0,vl + px*Twop/2.0+py*Twop/2.0) )*al;
						complex<Float> marsa = exp( complex<Float>(0,vl) )*al;

						switch (Moco)
						{
							case(MOM_MFLAT):
								// fM[idx] = complex<Float>(cos(vl), sin(vl))*al;
								fM[idx] = marsa;
							break;
							case(MOM_SPAX):
								{
									Float sc = mf((double) sqrt( (Float) modP));
									fM[idx] = marsa*sc;
									// if (modP < 10)
									// 	printf("mom %f sc %f\n",(double) sqrt( (Float) modP),sc);

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
									fV[idx] = marsa*sc;
									sc = (modP == 0) ? 0.0 : (cos(mP) - sc) ;
									fM[idx] = marsa*sc;
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
		} // END triple loop
	}


	if (commRank() == 0)
		{
			if ( mode0 < 3.141597 )
			{
				LogMsg (VERB_NORMAL, "mode0 set to %f in rank %d", mode0, commRank());
				// fM[0] = complex<Float>(cos(mode0), sin(mode0));
				if (Moco == MOM_MVSINCOS)
					fV[0] = exp( complex<Float>(0,mode0) );
					else
					fM[0] = exp( complex<Float>(0,mode0) );
			}
			else
			{
				if (Moco == MOM_MVSINCOS)
					mode0 = atan2(fV[0].imag(),fV[0].real());
					else
					mode0 = atan2(fM[0].imag(),fM[0].real());

				LogMsg (VERB_NORMAL, "mode0 is been randomly set to %f by rank %d", mode0, commRank());
			}
		}

	trackFree((void *) sd);
}





// /* This function to create real fields */
//
// template<typename Float, MomConfType Moco>
// void	momXeonRe (complex<Float> * __restrict__ fM, complex<Float> * __restrict__ fV, const MomParms mopa, const size_t Lx, const size_t Lz, const size_t Tz, const size_t S, const size_t V)
// {
// 	size_t        kMax  = mopa.kMax;
// 	double        kCrat  = mopa.kCrt;
// 	double        mass2 = mopa.mass2;
//
// 	std::vector 	mm = mopa.mfttab;
// 	std::vector 	ii;
// 	tk::spline mf;
// 	if (mopa.mocoty == CONF_SPAX){
// 		for (int i=0;i<mm.size();i++)
// 			ii.push_back(i);
//
// 		mf.set_points(ii,mm);
// 	}
//
// 	LogMsg(VERB_NORMAL,"[momXeonRe] Called with kMax %zu kCrit %f (kCrit es %f)", kMax, kCrat, kCrit);
//
// 	long long kmax;
// 	int adp = 0;
// 	if (kMax > Lx/2 - 1)
// 	{
// 		kmax = Lx/2 - 1;
// 		adp = 1;
// 	}
// 	else {
// 		kmax = kMax;
// 	}
// 	size_t kmax2 = kmax*kmax;
//
// 	constexpr Float Twop = 2.0*M_PI;
// 	complex<Float> II = complex<Float>{0,1} ;
// 	Float kcrit = (Float) kCrat;
//
// 	Float bee = (Float) 4*kcrit*kcrit/(Lx*Lx);
//
// 	int	maxThreads = omp_get_max_threads();
// 	int	*sd;
//
// 	trackAlloc((void **) &sd, sizeof(int)*maxThreads);
//
// 	std::random_device seed;		// Totally random seed coming from memory garbage
//
// 	for (int i=0; i<maxThreads; i++)
// 		sd[i] = seed()*(1 + commRank());
//
//
// 	const size_t LLy = Lx/commSize();
// 	const size_t zBase = LLy*commRank();
// 	const uint   maxLx = (Lx >> 1)+1;
// 	const size_t maxSf = maxLx*Tz;
//
//
// 	#pragma omp parallel default(shared)
// 	{
// 		int nThread = omp_get_thread_num();
//
//
// 		std::mt19937_64 mt64(sd[nThread]);		// Mersenne-Twister 64 bits, independent per thread
// 		std::uniform_real_distribution<Float> uni(0.0, 1.0);
// 		std::normal_distribution<Float> distri(0.0,1.0);
//
// 		#pragma omp parallel for collapse(3) schedule(static) default(shared)
// 		for (uint oy = 0; oy < LLy; oy++)	// As Javier pointed out, the transposition makes y the slowest coordinate
// 			for (uint oz = 0; oz < Tz; oz++)
// 				for (uint ox = 0; ox < maxLx; ox++)
// 				{
// 					size_t idx = ox + oy*maxSf + oz*maxLx;
//
// 					int px = ox;
// 					int py = oy + zBase;
// 					int pz = oz ;
//
// 					if (px > hLx)
// 						px -= Lx;
//
// 					if (py > hLx)
// 						py -= Lx;
//
// 					if (pz > hTz)
// 						pz -= Tz;
//
// 					size_t modP = pz*pz + py*py + px*px;
//
// 					if (modP <= 3*kmax2 )
// 					{
// 						Float vl = Twop*(uni(mt64));
// 						Float al = distri(mt64);
// 						complex<Float> marsa = exp( complex<Float>(0,vl) )*al;
//
// 						switch (Moco)
// 						{
// 							case(MOM_MFLAT):
// 								// fM[idx] = complex<Float>(cos(vl), sin(vl))*al;
// 								fM[idx] = marsa;
// 							break;
//
// 							case(MOM_MSIN):
// 								{
// 									Float mP = sqrt(((Float) modP))/(kcrit);
// 									Float sc = (modP == 0) ? 1.0 : sin(mP)/mP;
// 								// fM[idx] = complex<Float>(cos(vl), sin(vl))*al*sc;
// 								fM[idx] = marsa*sc;
// 								}
// 							break;
//
// 							case(MOM_MVSINCOS):
// 								{
// 									Float mP = sqrt(((Float) modP))/(kcrit);
// 									// v to m2 m to v
// 									Float sc = (modP == 0) ? 1.0 : sin(mP)/mP;
// 									fV[idx] = marsa*sc;
// 									sc = (modP == 0) ? 0.0 : (cos(mP) - sc) ;
// 									fM[idx] = marsa*sc;
// 								}
// 							break;
//
// 							case(MOM_MEXP):
// 								{
// 									Float mP = sqrt(((Float) modP))/(kcrit);
// 									Float sc = (modP == 0) ? 1.0 : exp(-mP);
// 									fM[idx] = marsa*sc;
// 								}
// 							break;
//
// 							default:
// 							case(MOM_MEXP2):
// 								{
// 									Float mP = ((Float) modP)/(kcrit*kcrit);
// 									Float sc = (modP == 0) ? 1.0 : exp(-mP);
// 									fM[idx] = marsa*sc;
// 								}
// 							break;
// 							case(MOM_COLE):
// 								{
// 									Float mP = ((Float) modP)/(kcrit*kcrit);
// 									Float sc = (modP == 0) ? 1.0 : exp(-bee*mP)/pow(mP+1,1.5);
// 									fM[idx] = marsa*sc;
// 								}
// 							break;
// 							case(MOM_KCOLE):
// 								{
// 									Float mP = ((Float) modP)/(kcrit*kcrit);
// 									Float sc = (modP == 0) ? 1.0 : exp(-mP); //*sqrt(mP)
// 									fM[idx] = marsa*sc;
// 								}
// 							break;
// 						}
//
// 					} // END if
// 		} // END triple loop
// 	}
//
// 	// zero mode
// 	if (commRank() == 0)
// 	{
// 		if ( mode0 < 3.141597 )
// 		{
// 			LogMsg (VERB_NORMAL, "mode0 set to %f in rank %d", mode0, commRank());
// 			// fM[0] = complex<Float>(cos(mode0), sin(mode0));
// 			if (Moco == MOM_MVSINCOS)
// 				fV[0] = exp( complex<Float>(0,mode0) );
// 				else
// 				fM[0] = exp( complex<Float>(0,mode0) );
// 		}
// 		else
// 		{
// 			if (Moco == MOM_MVSINCOS)
// 				mode0 = atan2(fV[0].imag(),fV[0].real());
// 				else
// 				mode0 = atan2(fM[0].imag(),fM[0].real());
//
// 			LogMsg (VERB_NORMAL, "mode0 is been randomly set to %f by rank %d", mode0, commRank());
// 		}
// 	}
//
// 	trackFree((void *) sd);
// }




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
			complex<double>* ma;
			complex<double>* va = static_cast<complex<double>*> (field->vCpu());
			if (field->LowMem())
				ma = static_cast<complex<double>*> (field->mStart());
			else
				ma = static_cast<complex<double>*> (field->m2Cpu());

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
			}

		// if (field->LowMem())
		// 	momXeon<double, Moco> (static_cast<complex<double>*> (field->mStart()), static_cast<complex<double>*> (field->vCpu()), mopa, n1, Lz, Tz, n2, n3);
		// else
		// 	momXeon<double, Moco> (static_cast<complex<double>*> (field->m2Cpu()), static_cast<complex<double>*> (field->vCpu()), mopa, n1, Lz, Tz, n2, n3);
		}
		break;

		case FIELD_SINGLE:
		{
			complex<float>* ma;
			complex<float>* va = static_cast<complex<float>*> (field->vCpu());
			if (field->LowMem())
				ma = static_cast<complex<float>*> (field->mStart());
			else
				ma = static_cast<complex<float>*> (field->m2Cpu());

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
