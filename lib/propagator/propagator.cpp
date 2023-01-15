#include <cstdio>
#include <cstdlib>
#include <memory>
#include <chrono>
#include <string>
#include <vector>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"
#include "propagator/propClass.h"
#include "utils/utils.h"
#include "gravity/potential.h"
#include "propagator/prop-def-mac.h"

#include <omp.h>

std::unique_ptr<PropBase> prop;

template<VqcdType pot>
class	PropLeap : public PropClass<2, PROP_FIRST, pot> {

	public:
		PropLeap(Scalar *field, const PropcType propclass) :
		PropClass<2, PROP_FIRST, pot>(field, propclass) {
		//	Set up Leapfrog parameters

		double nC[2] = { 0.5, 0.5 };
		double nD[3] = { 0.25, 0.5, 0.25 };

		this->setCoeff(nC, nD);

		switch(propclass)
		{
			case PROPC_SPEC:
				if (field->Device() == DEV_CPU) {
					this->setBaseName("Leapfrog spectral ");
				} else
					this->setBaseName("Leapfrog ");
			break;
			case PROPC_FSPEC:
				if (field->Device() == DEV_CPU) {
					this->setBaseName("Leapfrog full spectral ");
				} else
					this->setBaseName("Leapfrog ");
			break;
			default:
			case PROPC_BASE:
				if (field->LowMem())
					this->setBaseName("Lowmem Leapfrog ");
				else
					this->setBaseName("Leapfrog ");
			break;
		}
	}
};

template<VqcdType pot>
class	PropMLeap : public PropClass<4, PROP_FIRST, pot> {

	public:
		PropMLeap(Scalar *field, const PropcType propclass) :
		PropClass<4, PROP_FIRST, pot>(field, propclass) {
		//	Set up Leapfrog parameters

		double nC[4] = { 0.25,  0.25, 0.25, 0.25 };
		double nD[5] = { 0.125, 0.25, 0.25, 0.25, 0.125 };

		this->setCoeff(nC, nD);

		switch(propclass)
		{
			case PROPC_SPEC:
				if (field->Device() == DEV_CPU) {
					this->setBaseName("Multi-Leapfrog spectral ");
				} else
					this->setBaseName("Multi-Leapfrog ");
			break;
			case PROPC_FSPEC:
				if (field->Device() == DEV_CPU) {
					this->setBaseName("Multi-Leapfrog full spectral ");
				} else
					this->setBaseName("Multi-Leapfrog ");
			break;
			default:
			case PROPC_BASE:
				if (field->LowMem())
					this->setBaseName("Lowmem Multi-Leapfrog ");
				else
					this->setBaseName("Multi-Leapfrog ");
			break;
		}
	}
};


template<VqcdType pot>
class	PropOmelyan2 : public PropClass<2, PROP_FIRST, pot> {

	public:
		PropOmelyan2(Scalar *field, const PropcType propclass) :
		PropClass<2, PROP_FIRST, pot>(field, propclass) {
		constexpr double chi = +0.19318332750378360;

		//	Set up Omelyan parameters for BABAB

		double nC[2] = { 0.5, 0.5 };
		double nD[3] = { chi, 1.-2.*chi, chi };

		this->setCoeff(nC, nD);

		switch(propclass)
		{
			case PROPC_SPEC:
				if (field->Device() == DEV_CPU)
					this->setBaseName("Omelyan2 spectral ");
				else
					this->setBaseName("Omelyan2 ");
			break;
			case PROPC_FSPEC:
				if (field->Device() == DEV_CPU)
					this->setBaseName("Omelyan2 full spectral ");
				else
					this->setBaseName("Omelyan2 ");
			break;
			default:
			case PROPC_BASE:
				if (field->LowMem())
					this->setBaseName("Lowmem Omelyan2 ");
				else
					this->setBaseName("Omnelyan2 ");
			break;
		}
	}
};

template<VqcdType pot>
class	PropOmelyan4 : public PropClass<4, PROP_FIRST, pot> {
//class	PropOmelyan4 : public PropClass<4, PROP_LAST, pot> {

	public:
		PropOmelyan4(Scalar *field, const PropcType propclass) :
		PropClass<4, PROP_FIRST, pot>(field, propclass) {
//		PropClass<4, PROP_LAST, pot>(field, propclass) {
		constexpr double xi  = +0.1786178958448091;
		constexpr double lb  = -0.2123418310626054;
		constexpr double chi = -0.06626458266981849;
//		constexpr double xi  = +0.16449865155757600;
//		constexpr double lb  = -0.02094333910398989;
//		constexpr double chi = +1.23569265113891700;

		//	Set up Omelyan parameters for ABABABABA
		////	Set up Omelyan parameters for BABABABAB

		double nC[4] = { 0.5*(1.-2.*lb), lb, lb, 0.5*(1.-2.*lb) };
		double nD[5] = { xi, chi, 1.-2.*(xi+chi), chi, xi };
		//double nC[5] = { xi, chi, 1.-2.*(xi+chi), chi, xi };
		//double nD[4] = { 0.5*(1.-2.*lb), lb, lb, 0.5*(1.-2.*lb) };

		this->setCoeff(nC, nD);

		// if (spec && field->Device() == DEV_CPU) {
		// 	this->setBaseName("Omelyan4 spectral ");
		// } else {
		// 	if (field->LowMem())
		// 		this->setBaseName("Lowmem Omelyan4 ");
		// 	else
		// 		this->setBaseName("Omelyan4 ");
		// }
		switch(propclass)
		{
			case PROPC_SPEC:
				if (field->Device() == DEV_CPU) {
					this->setBaseName("Omelyan4 spectral ");
				} else
					this->setBaseName("Omelyan4 ");
			break;
			case PROPC_FSPEC:
				if (field->Device() == DEV_CPU) {
					this->setBaseName("Omelyan4 full spectral ");
				} else
					this->setBaseName("Omelyan4 ");
			break;
			default:
			case PROPC_BASE:
				if (field->LowMem())
					this->setBaseName("Lowmem Omelyan4 ");
				else
					this->setBaseName("Omelyan4 ");
			break;
		}
	}
};

template<VqcdType pot>
class	PropRKN4 : public PropClass<4, PROP_NORMAL, pot> {

	public:
		PropRKN4(Scalar *field, const PropcType propclass) :
		PropClass<4, PROP_NORMAL, pot>(field, propclass) {
		//	Set up RKN parameters for BABABABA

		const double nC[4] = { +0.1344961992774310892, -0.2248198030794208058, +0.7563200005156682911, +0.3340036032863214255 };
		const double nD[4] = { +0.5153528374311229364, -0.085782019412973646,  +0.4415830236164665242, +0.1288461583653841854 };

		this->setCoeff(nC, nD);

		// if (spec && field->Device() == DEV_CPU) {
		// 	this->setBaseName("RKN4 spectral ");
		// } else {
		// 	if (field->LowMem())
		// 		this->setBaseName("Lowmem RKN4 ");
		// 	else
		// 		this->setBaseName("RKN4 ");
		// }
		switch(propclass)
		{
			case PROPC_SPEC:
				if (field->Device() == DEV_CPU) {
					this->setBaseName("RKN4 spectral ");
				} else
					this->setBaseName("RKN4 ");
			break;
			case PROPC_FSPEC:
				if (field->Device() == DEV_CPU) {
					this->setBaseName("RKN4 full spectral ");
				} else
					this->setBaseName("RKN4 ");
			break;
			default:
			case PROPC_BASE:
				if (field->LowMem())
					this->setBaseName("Lowmem RKN4 ");
				else
				{
					if (field->LowMemGPU())
						this->setBaseName("Lowmem G RKN4 ");
					else 
						this->setBaseName("RKN4 ");
				}
			break;
		}
	}
};


void	initPropagator	(PropType pType, Scalar *field, VqcdType pot, int Ng=-1) {

	LogMsg	(VERB_NORMAL, "[ip] Initializing propagator");
	LogMsg	(VERB_NORMAL, "[ip] pType is %d pot is %d ",pType, pot);
	// bool	spec = (pType & PROP_SPEC) ? true : false, wasTuned = false;
	bool wasTuned = false;

	PropcType propclass = PROPC_BASE;

	if 	( (pType & PROPC_BASE) )
	{
		LogMsg	(VERB_NORMAL, "[ip] propagator BASE with Ng=%d selected (%d)",Ng,pType);LogFlush();
		LogMsg	(VERB_HIGH, "[ip] field->Ng is set to %d",field->getNg());LogFlush();
	}
	if 	(pType & PROP_FSPEC)
	{
 		LogMsg	(VERB_NORMAL, "[ip] propagator Full Spectral selected %d",pType);LogFlush();
		propclass = PROPC_FSPEC;
	}
	if 	(pType & PROP_SPEC) // overwritting
	{
 		LogMsg	(VERB_NORMAL, "[ip] propagator Spectral selected",pType);LogFlush();
		propclass = PROPC_SPEC;
	}
	LogMsg	(VERB_NORMAL, "[ip] propclass set to %d", propclass);

	unsigned int xBlock, yBlock, zBlock;

	if (prop != nullptr)
		if (prop->IsTuned()) {
			wasTuned = true;
			xBlock = prop->TunedBlockX();
			yBlock = prop->TunedBlockY();
			zBlock = prop->TunedBlockZ();
		}

	LogMsg(VERB_NORMAL,"\n");
 	LogMsg(VERB_NORMAL,"[ip] Init propagator %d\n",pType & PROP_MASK);


	switch (pType & PROP_MASK) {

#ifdef	USE_PROP_OM2
		case PROP_OMELYAN2:
			DEFALLPROPTEM(PropOmelyan2);
		break;
#endif

#ifdef	USE_PROP_OM2
		case PROP_OMELYAN4:
			DEFALLPROPTEM(PropOmelyan4);
		break;
#endif

#ifdef	USE_PROP_LEAP
		case PROP_LEAP:
			DEFALLPROPTEM(PropLeap);
		break;
#endif

#ifdef	USE_PROP_MLEAP
		case PROP_MLEAP:
			DEFALLPROPTEM(PropMLeap);
		break;
#endif

#ifdef	USE_PROP_RKN4
		case PROP_RKN4:
			DEFALLPROPTEM(PropRKN4);
		break;
#endif

		default:
			LogError ("Error: unrecognized propagator PROP_ %d (RKN4/MLEAP/LEAP/OM4/OM2 %d/%d/%d/%d/%d) ",
				pType, PROP_RKN4,PROP_MLEAP,PROP_LEAP,PROP_OMELYAN4,PROP_OMELYAN2);
			exit(1);
		break;
	}

	LogMsg(VERB_HIGH,"[ip] getBaseName");
	prop->getBaseName();

	LogMsg(VERB_HIGH,"[ip] set blocks");
	if (wasTuned) {
		prop->SetBlockX(xBlock);
		prop->SetBlockY(yBlock);
		prop->SetBlockZ(zBlock);
		prop->UpdateBestBlock();
	}

	LogMsg	(VERB_NORMAL, "Propagator %s successfully initialized", prop->Name().c_str());
 	LogFlush();

}





using	namespace profiler;

void	propagate	(Scalar *field, const double dz)
{
	LogMsg	(VERB_HIGH, "[pate] Called propagator");
LogFlush();
	Profiler &prof = getProfiler(PROF_PROP);

LogFlush();
 	if	( (pType & PROP_BASE) && !field->Folded() )
	{
		Folder	munge(field);
		munge(FOLD_ALL);
	}

	prop->getBaseName();

	prof.start();

	char loli[2048];

	switch (field->Field()) {
		case FIELD_SAXION:
			if (pType & PROP_BASE){
				sprintf (loli, "N %01d Ng %01d Saxion", Nng, field->getNg());
				prop->appendName(loli);
			} else {
				prop->appendName("Saxion");
			}
			(prop->propSaxion)(dz);
			break;

		case FIELD_AXION:
		if (pType & PROP_BASE){
				sprintf (loli, "N %01d Ng %01d Axion", Nng, field->getNg());
				prop->appendName(loli);
			} else {
				prop->appendName("Axion");
			}
			(prop->propAxion)(dz);
			break;

		case FIELD_AXION_MOD:
			if (pType & PROP_BASE){
				sprintf (loli, "N %01d Ng %01d Axion Mod", Nng, field->getNg());
				prop->appendName(loli);
			} else {
				prop->appendName("Axion Mod");
			}
			(prop->propAxion)(dz);
			break;

		case FIELD_NAXION:
		if (pType & PROP_BASE){
				sprintf (loli, "N %01d Ng %01d NAxion", Nng, field->getNg());
				prop->appendName(loli);
			} else {
				prop->appendName("Naxion");
			}
			(prop->propNaxion)(dz);
			break;

		case FIELD_PAXION:
			if (pType & PROP_BASE){
					sprintf (loli, "N %01d Ng %01d Paxion", Nng, field->getNg());
					prop->appendName(loli);
				} else {
					prop->appendName("Paxion");
				}
				(prop->propPaxion)(dz);
				break;

		default:
			LogError ("Error: invalid field type");
			prof.stop();
			return;
	}

	auto mFlops = prop->cFlops((pType & PROP_SPEC) ? PROPC_SPEC : PROPC_BASE);
	auto mBytes = prop->cBytes((pType & PROP_SPEC) ? PROPC_SPEC : PROPC_BASE);

	prop->add(mFlops, mBytes);

	prof.stop();

	prof.add(prop->Name(), prop->GFlops(), prop->GBytes());

	prop->reset();

	LogMsg	(VERB_HIGH, "Propagator %s reporting %lf GFlops %lf GBytes", prop->Name().c_str(), prof.Prof()[prop->Name()].GFlops(), prof.Prof()[prop->Name()].GBytes());

	return;
}

/*
std::vector<int>	calculateAllowedBlockSize(int length) {

	int 			maxSize = (int) (floor(sqrt(length)));
	std::vector<int>	lowDiv, highDiv;

	for (dLow=2; dLow<=maxSize; dLow++) {
		if ((length % dLow) != 0)
			continue;

		dHigh = length/d;
		lowDiv.push_back(dLow);
		highDiv.push_back(dHigh);
	}

	std::reverse(highDiv.begin(), highDiv.end());
	lowDiv.insert(lowDiv.end(), highDiv.begin(), highDiv.end());

	for (i=0; i<lowDiv.size; i++)
		printf("%d\t\t%d\n", i, lowDiv[i]);

	return lowDiv;
}
*/

void	resetPropagator(Scalar *field) {
	/*	Default block size gives just one block	*/

	if (prop == nullptr){
		LogMsg(VERB_NORMAL,"[tp] reset propagator called but not initialised: EXIT!");
		return ;
	}

	LogMsg(VERB_NORMAL,"[tp] reseting!");
	if (pType & PROP_SPEC)
		return;
	if (pType & PROP_FSPEC)
		return;

	int tmp   = field->DataAlign()/field->DataSize();
	int shift = 0;

	while (tmp != 1) {
		LogMsg(VERB_HIGH,"[tp] tmp %d!",tmp);
		shift++;
		tmp >>= 1;
	}

	prop->SetBlockX(field->Length() << shift);
	prop->SetBlockY(field->Length() >> shift);
	prop->SetBlockZ(field->Depth ());
	prop->UpdateBestBlock();
	prop->UnTune();
	LogMsg(VERB_HIGH,"[tp] done!",tmp);
}




void	tunePropagator (Scalar *field) {
	// Hash CPU model so we don't mix different cache files
	LogMsg(VERB_NORMAL,"\n");
 	LogMsg(VERB_NORMAL,"[tp] Tune propagator!\n");
	if (pType & PROP_SPEC)
		return;
	if (pType & PROP_FSPEC)
		return;

	int  myRank   = commRank();
	//int  nThreads = 1;
	bool newFile  = false, found = false;

	if (prop == nullptr) {
		LogError("Error: propagator not initialized, can't be tuned.");
		return;
	}

	Profiler &prof = getProfiler(PROF_TUNER);

	std::chrono::high_resolution_clock::time_point start, end;
	size_t bestTime, lastTime, cTime;

	LogMsg (VERB_HIGH, "[tp] Started tuner");
	prof.start();

	if (field->Device() == DEV_CPU)
		prop->InitBlockSize(field->Length(), field->Depth(), field->DataSize(), field->DataAlign());
	else
		prop->InitBlockSize(field->Length(), field->Depth(), field->DataSize(), field->DataAlign(), true);

	/*	Check for a cache file	*/

	if (myRank == 0) {
		FILE *cacheFile;
		char tuneName[2048];
		// if (pType == PROP_BASE)
		// 	sprintf (tuneName, "%s/tuneCache.dat", wisDir);
		if (pType & PROP_BASE)
			sprintf (tuneName, "%s/tuneCache.dat", wisDir);

		if ((cacheFile = fopen(tuneName, "r")) == nullptr) {
LogMsg(VERB_HIGH,"[tp] new cache!!");
LogMsg (VERB_NORMAL, "Missing tuning cache file %s, will create a new one", tuneName);
			newFile = true;
		} else {
			int	     rMpi, rThreads;
			size_t       rLx, rLz, Nghost;
			unsigned int rBx, rBy, rBz, fType, myField ;
			if      (field->Field() == FIELD_SAXION)
				myField = 0;
			else if (field->Field() == FIELD_AXION)
				myField = 1;
			else if (field->Field() == FIELD_NAXION)
				myField = 2;
			else if (field->Field() == FIELD_PAXION)
				myField = 3;

			char	     mDev[8];

			std::string tDev(field->Device() == DEV_GPU ? "Gpu" : "Cpu");
LogMsg(VERB_HIGH,"[tp] Reading cache file %s",tuneName);
			do {
				fscanf (cacheFile, "%s %d %d %lu %lu %u %u %u %u %lu\n", reinterpret_cast<char*>(&mDev), &rMpi, &rThreads, &rLx, &rLz, &fType, &rBx, &rBy, &rBz, &Nghost);
				std::string fDev(mDev);
LogMsg(VERB_HIGH,"[tp] Read: MPI %d, threads %d, Ng %d, Lx,Lz (%d,%d) rBx,y,z (%d,%d,%d)  ",rMpi, rThreads, Nghost, rLx, rLz, rBx, rBy, rBz);
				if (rMpi == commSize() && rThreads == omp_get_max_threads() && rLx == field->Length() && rLz == field->Depth() && fType == myField && fDev == tDev && Nghost == field->getNg()) {
					if ((field->Device() == DEV_CPU && (rBx <= prop->BlockX() && rBy <= field->Surf()/prop->BlockX() && rBz <= field->Depth())) ||
					    (field->Device() == DEV_GPU	&& (rBx <= prop->MaxBlockX() && rBy <= prop->MaxBlockY() && rBz <= prop->MaxBlockZ()))) {
						found = true;
LogMsg(VERB_HIGH,"[tp] X!!");
						prop->SetBlockX(rBx);
LogMsg(VERB_HIGH,"[tp] Y!!");
						prop->SetBlockY(rBy);
LogMsg(VERB_HIGH,"[tp] Z!!");
						prop->SetBlockZ(rBz);
LogMsg(VERB_HIGH,"[tp] update best block!!");
						prop->UpdateBestBlock();
					}
				}
			}	while(!feof(cacheFile) && !found);

			fclose (cacheFile);
LogMsg(VERB_HIGH,"[tp] cache file closed!!");
		}
	}
LogMsg(VERB_HIGH,"[tp] BCAST!");

	MPI_Bcast (&found, sizeof(found), MPI_BYTE, 0, MPI_COMM_WORLD);

	commSync();

	// If a cache file was found, we broadcast the best block and exit
	if (found) {
LogMsg(VERB_HIGH,"[tp] optimum found!");
		unsigned int block[3];

		if (myRank == 0) {
			block[0] = prop->TunedBlockX();
			block[1] = prop->TunedBlockY();
			block[2] = prop->TunedBlockZ();
		}

		MPI_Bcast (&block, sizeof(int)*3, MPI_BYTE, 0, MPI_COMM_WORLD);
		commSync();

		if (myRank != 0) {
			prop->SetBlockX(block[0]);
			prop->SetBlockY(block[1]);
			prop->SetBlockZ(block[2]);
			prop->UpdateBestBlock();
		}

LogMsg (VERB_NORMAL, "Tuned values read from cache file. Best block %u x %u x %u", prop->TunedBlockX(), prop->TunedBlockY(), prop->TunedBlockZ());
LogMsg (VERB_HIGH,   "Chosen block %u x %u x %u\n", prop->BlockX(), prop->BlockY(), prop->BlockZ());
		prop->Tune();
		prof.stop();
		prof.add(prop->Name(), 0., 0.);
		return;
	}

	// Otherwise we start tuning

LogMsg (VERB_HIGH,   "[tp] Start tuning ... ");

	start = std::chrono::high_resolution_clock::now();
	propagate(field, 0.);
	end   = std::chrono::high_resolution_clock::now();

	cTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();

	// If there is an error in GPU propagation, we set the time to an absurd value
	#ifdef USE_GPU
	if (field->Device() == DEV_GPU) {
		auto gErr = cudaGetLastError();

		if (gErr != cudaSuccess)
			cTime = std::numeric_limits<std::size_t>::max();
	}
	#endif

	MPI_Allreduce(&cTime, &bestTime, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

	if (field->Device() == DEV_GPU && cTime == std::numeric_limits<std::size_t>::max())
		LogMsg (VERB_HIGH, "Block %u x %u x %u gave an error and couldn't run on the GPU", prop->BlockX(), prop->BlockY(), prop->BlockZ());
	else
		LogMsg (VERB_HIGH, "Block %u x %u x %u done in %lu ns", prop->BlockX(), prop->BlockY(), prop->BlockZ(), bestTime);

	prop->AdvanceBlockSize();

	while (!prop->IsTuned()) {

		start = std::chrono::high_resolution_clock::now();
		propagate(field, 0.);
		end   = std::chrono::high_resolution_clock::now();

		cTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();

		#ifdef USE_GPU
		if (field->Device() == DEV_GPU) {
			auto gErr = cudaGetLastError();

			if (gErr != cudaSuccess)
				cTime = std::numeric_limits<std::size_t>::max();
		}
		#endif

    MPI_Allreduce(&cTime, &lastTime, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

		if (field->Device() == DEV_GPU && cTime == std::numeric_limits<std::size_t>::max())
			LogMsg (VERB_HIGH, "Block %u x %u x %u gave an error and couldn't run on the GPU", prop->BlockX(), prop->BlockY(), prop->BlockZ());
		else{
			LogMsg (VERB_HIGH, "Test Block %u x %u x %u done in %lu ns", prop->BlockX(), prop->BlockY(), prop->BlockZ(), lastTime);
			LogMsg (VERB_HIGH, "Best Block %u x %u x %u done in %lu ns", prop->TunedBlockX(), prop->TunedBlockY(), prop->TunedBlockZ(), bestTime);
		}

		if (lastTime < bestTime) {
			bestTime = lastTime;
			prop->UpdateBestBlock();
			LogMsg (VERB_HIGH, "Best block updated");
		}

		prop->AdvanceBlockSize();
	}

	prop->getBaseName();

	char loli[2048];

	switch (field->Field()) {
		case FIELD_SAXION:
			if (pType & PROP_BASE){
				sprintf (loli, "N %01d Ng %01d Saxion", Nng, field->getNg());
				prop->appendName(loli);
			} else {
				prop->appendName("Saxion");
			}
			break;

		case FIELD_AXION:
		if (pType & PROP_BASE){
				sprintf (loli, "N %01d Ng %01d Axion", Nng, field->getNg());
				prop->appendName(loli);
			} else {
				prop->appendName("Axion");
			}
			break;

		case FIELD_AXION_MOD:
			if (pType & PROP_BASE){
				sprintf (loli, "N %01d Ng %01d Axion Mod", Nng, field->getNg());
				prop->appendName(loli);
			} else {
				prop->appendName("Axion Mod");
			}
			break;

		case FIELD_NAXION:
			if (pType & PROP_BASE){
					sprintf (loli, "N %01d Ng %01d Naxion", Nng, field->getNg());
					prop->appendName(loli);
				} else {
					prop->appendName("Naxion");
				}
				break;

		case FIELD_PAXION:
			if (pType & PROP_BASE){
					sprintf (loli, "N %01d Ng %01d Paxion", Nng, field->getNg());
					prop->appendName(loli);
				} else {
					prop->appendName("Paxion");
				}
				break;
		default:
			LogError ("Error: invalid field type");
			prof.stop();
			return;
	}

	Profiler &propProf = getProfiler(PROF_PROP);
	propProf.reset(prop->Name());

	prop->SetBestBlock();
	LogMsg (VERB_NORMAL, "Propagator tuned! Best block %u x %u x %u in %lu ns", prop->TunedBlockX(), prop->TunedBlockY(), prop->TunedBlockZ(), bestTime);

	/*	Write cache file if necessary, block of rank 0 prevails		*/

	if (myRank == 0) {
		FILE *cacheFile;
		char tuneName[2048];
		// sprintf (tuneName, "%s/tuneCache.dat", wisDir);
		if (pType & PROP_BASE){
			sprintf (tuneName, "%s/tuneCache.dat", wisDir);
			LogMsg(VERB_HIGH,"[tp] tuneName = %s",tuneName);
		}
		// We distinguish between opening and appending a new line
		if (!newFile) {
			if ((cacheFile = fopen(tuneName, "a")) == nullptr) {
				LogError ("Error: can't open cache file, can't save tuning results");
				commSync();
				prof.stop();
				prof.add(prop->Name(), 0., 0.);
			}
		} else {
			if ((cacheFile = fopen(tuneName, "w")) == nullptr) {
				LogError ("Error: can't create cache file, can't save tuning results");
				commSync();
				prof.stop();
				prof.add(prop->Name(), 0., 0.);
			}
		}

		unsigned int fType ;
		if      (field->Field() == FIELD_SAXION)
			fType = 0;
		else if (field->Field() == FIELD_AXION)
			fType = 1;
		else if (field->Field() == FIELD_NAXION)
			fType = 2;
		else if (field->Field() == FIELD_PAXION)
			fType = 3;

		std::string myDev(field->Device() == DEV_GPU ? "Gpu" : "Cpu");
		fprintf (cacheFile, "%s %d %d %lu %lu %u %u %u %u %lu\n", myDev.c_str(), commSize(), omp_get_max_threads(), field->Length(), field->Depth(),
									fType, prop->TunedBlockX(), prop->TunedBlockY(), prop->TunedBlockZ(), field->getNg());
		fclose  (cacheFile);
	}
LogMsg (VERB_NORMAL, "\n");

	commSync();
	prof.stop();
	prof.add(prop->Name(), 0., 0.);

}

void	initGravity	(Scalar *field){

	if (field->Field() != FIELD_PAXION)
		if (field->Field() != FIELD_AXION){
		LogError("Gravity only available in PAXION/AXION mode; exit!");
		exit(1);
	}
		InitGravity(field);
		tuneGravity(prop->TunedBlockX(), prop->TunedBlockY(), prop->TunedBlockZ());
		prop->SetGravity(true);
		normaliseFields	();
}
