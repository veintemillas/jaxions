#include<cstdlib>
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

template<typename Float>
void	momXeon (complex<Float> * __restrict__ fM, const long long kMax, const Float kCrit, const size_t Lx, const size_t Lz, const size_t Tz, const size_t S, const size_t V)
{
	long long kmax ;
	int adp = 0;
	if (kMax > Lx/2 - 1)
	{
		kmax = Lx/2 -1 ;
		adp = 1 ;
	}
	else {
		kmax = kMax ;
	}
	size_t kmax2 = kmax*kmax;

	const Float Twop = 2.0*M_PI;

	int	maxThreads = omp_get_max_threads();
	int	*sd;



	trackAlloc((void **) &sd, sizeof(int)*maxThreads);

	std::random_device seed;		// Totally random seed coming from memory garbage

	for (int i=0; i<maxThreads; i++)
		sd[i] = seed()*(1 + commRank());

	#pragma omp parallel default(shared)
	{
		int nThread = omp_get_thread_num();


		std::mt19937_64 mt64(sd[nThread]);		// Mersenne-Twister 64 bits, independent per thread
		std::uniform_real_distribution<Float> uni(0.0, 1.0);

		#pragma omp for schedule(static)
		for (size_t oz = 0; oz < Tz; oz++)
		{
			if (oz/Lz != commRank())
				continue;

			long long pz = oz - (oz/(Tz >> 1))*Tz;

			for(long long py = -kmax  ; py <= kmax + adp; py++)
			{
				for(long long px = -kmax ; px <= kmax + adp; px++)
				{
					size_t idx  = ((px + Lx)%Lx) + ((py+Lx)%Lx)*Lx + ((pz+Tz)%Tz)*S - commRank()*V;
					size_t modP = px*px + py*py + pz*pz;

					if (modP <= 3*(kmax2 + adp*(1+Lx)))
					{
						Float vl = Twop*(uni(mt64));

						// Float mP = sqrt(((Float) modP))/((Float) (kCrit));
						// Float sc = (modP == 0) ? 1.0 : sin(mP)/mP;
						 Float mP = ((Float) modP)/((Float) (kCrit*kCrit));
						 Float sc = (modP == 0) ? 1.0 : exp(-mP*mP);

						fM[idx] = complex<Float>(cos(vl), sin(vl))*sc;
						//printf("mom (%d,%d,%d) = %f %f*I\n",pz,py,px,fM[idx].real(),fM[idx].imag());
					}
				} // END  px loop
			} // END  py loop
		} // END oz FOR
	}

	// zero mode

	if (mode0 < 3.141597)
	fM[0] = complex<Float>(cos(mode0), sin(mode0));

	trackFree((void **) &sd, ALLOC_TRACK);
}

void	momConf (Scalar *field, const size_t kMax, const double kCrt)
{
	const size_t n1 = field->Length();
	const size_t n2 = field->Surf();
	const size_t n3 = field->Size();
	const size_t Lz = field->Depth();
	const size_t Tz = field->TotalDepth();

	const size_t offset = field->DataSize()*n2;

	switch (field->Precision())
	{
		case FIELD_DOUBLE:

		if (field->LowMem())
			momXeon (static_cast<complex<double>*> (static_cast<void*>(static_cast<char*>(field->mCpu())  + offset)), kMax,                    kCrt,  n1, Lz, Tz, n2, n3);
		else
			momXeon (static_cast<complex<double>*> (field->m2Cpu()), kMax, kCrt,  n1, Lz, Tz, n2, n3);
		break;

		case FIELD_SINGLE:

		if (field->LowMem())
			momXeon (static_cast<complex<float> *> (static_cast<void*>(static_cast<char*>(field->mCpu())  + offset)), kMax, static_cast<float>(kCrt), n1, Lz, Tz, n2, n3);
		else
			momXeon (static_cast<complex<float> *> (field->m2Cpu()), kMax, static_cast<float>(kCrt), n1, Lz, Tz, n2, n3);
		break;

		default:
		break;
	}
}
