#include <cstdio>
#include <cstdlib>
#include <memory>
#include <chrono>
#include <string>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"
#include "propagator/propClass.h"
#include "utils/utils.h"

#include <omp.h>

std::unique_ptr<PropBase> prop;

template<VqcdType pot>
class	PropLeap : public PropClass<2, true, pot> {

	public:
		PropLeap(Scalar *field, const PropcType propclass) :
		PropClass<2, true, pot>(field, propclass) {
		//	Set up Leapfrog parameters

		double nC[3] = { 0.5, 0.5, 0.0 };
		double nD[2] = { 1.0, 0.0 };

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
			case PROPC_NNEIG:
				LogError("Error: propagator N-neighbour not supported with leapfrog yet");
				exit(1);
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
class	PropMLeap : public PropClass<4, true, pot> {

	public:
		PropMLeap(Scalar *field, const PropcType propclass) :
		PropClass<4, true, pot>(field, propclass) {
		//	Set up Leapfrog parameters

		double nC[5] = { 0.125, 0.25, 0.25, 0.25, 0.125 };
		double nD[4] = { 0.25,  0.25, 0.25, 0.25 };

		this->setCoeff(nC, nD);

		// if (spec && field->Device() == DEV_CPU) {
		// 	this->setBaseName("Multi-Leapfrog spectral ");
		// } else {
		// 	if (field->LowMem())
		// 		this->setBaseName("Lowmem Multi-Leapfrog ");
		// 	else
		// 		this->setBaseName("Multi-Leapfrog ");
		// }
		//

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
			case PROPC_NNEIG:
				LogError("Error: propagator N-neighbour not supported with Multi-leapfrog yet");
				exit(1);
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
class	PropOmelyan2 : public PropClass<2, true, pot> {

	public:
		PropOmelyan2(Scalar *field, const PropcType propclass) :
		PropClass<2, true, pot>(field, propclass) {
		constexpr double chi = +0.19318332750378360;

		//	Set up Omelyan parameters for BABAB

		double nC[3] = { chi, 1.-2.*chi, chi };
		double nD[2] = { 0.5, 0.5 };

		this->setCoeff(nC, nD);

		if (propclass && field->Device() == DEV_CPU) {
			this->setBaseName("Omelyan2 spectral ");
		} else {
			if (field->LowMem())
				this->setBaseName("Lowmem Omelyan2 ");
			else
				this->setBaseName("Omelyan2 ");
		}
	}
};

template<VqcdType pot>
class	PropOmelyan4 : public PropClass<4, true, pot> {

	public:
		PropOmelyan4(Scalar *field, const PropcType propclass) :
		PropClass<4, true, pot>(field, propclass) {
		constexpr double xi  = +0.16449865155757600;
		constexpr double lb  = -0.02094333910398989;
		constexpr double chi = +1.23569265113891700;

		//	Set up Omelyan parameters for BABABABAB

		double nC[5] = { xi, chi, 1.-2.*(xi+chi), chi, xi };
		double nD[4] = { 0.5*(1.-2.*lb), lb, lb, 0.5*(1.-2.*lb) };

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
			case PROPC_NNEIG:
				LogError("Error: propagator N-neighbour not supported with Omelyan yet (only even step coded)");
				exit(1);
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
class	PropRKN4 : public PropClass<4, false, pot> {

	public:
		PropRKN4(Scalar *field, const PropcType propclass) :
		PropClass<4, false, pot>(field, propclass) {
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
			case PROPC_NNEIG:
				if (field->LowMem()){
					LogError("Error: propagator N-neighbour not supported with lowmem yet (only coded with m2!)");
					exit(1);
				} else
					this->setBaseName("Ng RKN4 ");
			break;
			default:
			case PROPC_BASE:
				if (field->LowMem())
					this->setBaseName("Lowmem RKN4 ");
				else
					this->setBaseName("RKN4 ");
			break;
		}
	}
};


void	initPropagator	(PropType pType, Scalar *field, VqcdType pot, int Ng=-1) {

	LogMsg	(VERB_NORMAL, "[ip] Initializing propagator");
	LogMsg	(VERB_NORMAL, "[ip] pType is %d",pType);
	// bool	spec = (pType & PROP_SPEC) ? true : false, wasTuned = false;
	bool wasTuned = false;

	PropcType propclass = PROPC_BASE;

	if 	( (pType & PROPC_BASE) )
	{
		LogMsg	(VERB_NORMAL, "[ip] propagator BASE with Ng=%d selected (%d)",Ng,pType);LogFlush();
		LogMsg	(VERB_HIGH, "[ip] field->Ng is set to %d",field->getNg());LogFlush();
	}
	if 	( (pType & PROPC_NNEIG) && (Ng > -1))
	{
		LogMsg	(VERB_NORMAL, "[ip] propagator Ng=%d selected (%d)",Ng,pType);LogFlush();
		propclass = PROPC_NNEIG;
		LogMsg	(VERB_HIGH, "[ip] External variable Nng is %d",Nng);LogFlush();
		if (Nng != Ng)
		{
			LogMsg	(VERB_HIGH, "[ip] WARNING: Nng was %d but changed to %d!",Nng,Ng);
			Nng = Ng;
		}
		// we send ghosts to initialise COMM definitions once; minimum overhead, perhaps a waste of time?
		field->sendGhosts(FIELD_M, COMM_SDRV);
		field->sendGhosts(FIELD_M, COMM_WAIT);
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

	//auto pot  = field->BckGnd()->QcdPot();
	//auto gm   = field->BckGnd()->Gamma ();
	LogMsg(VERB_NORMAL,"\n");
 	LogMsg(VERB_NORMAL,"[ip] Init propagator\n");
	switch (pType & PROP_MASK) {
		case PROP_OMELYAN2:
			switch (pot) {
				case VQCD_1:
					prop = std::make_unique<PropOmelyan2<VQCD_1>>		(field, propclass);
					break;
				case VQCD_1_PQ_2:
					prop = std::make_unique<PropOmelyan2<VQCD_1_PQ_2>>	(field, propclass);
					break;
				case VQCD_1_PQ_2_RHO:
					prop = std::make_unique<PropOmelyan2<VQCD_1_PQ_2_RHO>>	(field, propclass);
					break;
				case VQCD_2:
					prop = std::make_unique<PropOmelyan2<VQCD_2>>		(field, propclass);
					break;
				case VQCD_0:
						prop = std::make_unique<PropOmelyan2<VQCD_0>>		(field, propclass);
						break;

				default:
				case VQCD_NONE:
					prop = std::make_unique<PropOmelyan2<VQCD_NONE>>	(field, propclass);
					break;
			}
			break;

		case PROP_OMELYAN4:
			switch (pot) {
				case VQCD_1:
					prop = std::make_unique<PropOmelyan4<VQCD_1>>		(field, propclass);
					break;
				case VQCD_1_PQ_2:
					prop = std::make_unique<PropOmelyan4<VQCD_1_PQ_2>>	(field, propclass);
					break;
				case VQCD_1_PQ_2_RHO:
					prop = std::make_unique<PropOmelyan4<VQCD_1_PQ_2_RHO>>	(field, propclass);
					break;
				case VQCD_2:
					prop = std::make_unique<PropOmelyan4<VQCD_2>>		(field, propclass);
					break;
				case VQCD_0:
					prop = std::make_unique<PropOmelyan4<VQCD_0>>		(field, propclass);
					break;
				default:
				case VQCD_NONE:
					prop = std::make_unique<PropOmelyan4<VQCD_NONE>>	(field, propclass);
					break;
			}
			break;

		case PROP_LEAP:
			switch (pot) {
				case VQCD_1:
					prop = std::make_unique<PropLeap<VQCD_1>>		(field, propclass);
					break;
				case VQCD_1_PQ_2:
					prop = std::make_unique<PropLeap<VQCD_1_PQ_2>>		(field, propclass);
					break;
				case VQCD_1_PQ_2_RHO:
					prop = std::make_unique<PropLeap<VQCD_1_PQ_2_RHO>>	(field, propclass);
					break;
				case VQCD_2:
					prop = std::make_unique<PropLeap<VQCD_2>>		(field, propclass);
					break;
				case VQCD_0:
					prop = std::make_unique<PropLeap<VQCD_0>>		(field, propclass);
					break;
				default:
				case VQCD_NONE:
					prop = std::make_unique<PropLeap<VQCD_NONE>>		(field, propclass);
					break;
			}
			break;

			case PROP_MLEAP:
				switch (pot) {
					case VQCD_1:
						prop = std::make_unique<PropMLeap<VQCD_1>>		(field, propclass);
						break;
					case VQCD_1_RHO:
						prop = std::make_unique<PropMLeap<VQCD_1_RHO>>		(field, propclass);
						break;
					case VQCD_1_DRHO:
						prop = std::make_unique<PropMLeap<VQCD_1_DRHO>>		(field, propclass);
						break;
					case VQCD_1_PQ_2:
						prop = std::make_unique<PropMLeap<VQCD_1_PQ_2>>		(field, propclass);
						break;
					case VQCD_1_PQ_2_RHO:
						prop = std::make_unique<PropMLeap<VQCD_1_PQ_2_RHO>>	(field, propclass);
						break;
					case VQCD_1_PQ_2_DRHO:
						prop = std::make_unique<PropMLeap<VQCD_1_PQ_2_DRHO>>	(field, propclass);
						break;

					case VQCD_2:
						prop = std::make_unique<PropMLeap<VQCD_2>>		(field, propclass);
						break;
					case VQCD_2_RHO:
						prop = std::make_unique<PropMLeap<VQCD_2_RHO>>		(field, propclass);
						break;
					case VQCD_2_DRHO:
						prop = std::make_unique<PropMLeap<VQCD_2_DRHO>>		(field, propclass);
						break;

					case VQCD_0:
						prop = std::make_unique<PropMLeap<VQCD_0>>		(field, propclass);
						break;
					case VQCD_0_RHO:
						prop = std::make_unique<PropMLeap<VQCD_0_RHO>>		(field, propclass);
						break;
					case VQCD_0_DRHO:
						prop = std::make_unique<PropMLeap<VQCD_0_DRHO>>		(field, propclass);
						break;

					case VQCD_1N2:
						prop = std::make_unique<PropMLeap<VQCD_1N2>>		(field, propclass);
						break;
					case VQCD_1N2_RHO:
						prop = std::make_unique<PropMLeap<VQCD_1N2_RHO>>	(field, propclass);
						break;
					case VQCD_1N2_DRHO:
						prop = std::make_unique<PropMLeap<VQCD_1N2_DRHO>>	(field, propclass);
						break;

					case VQCD_NONE:
					default:
						prop = std::make_unique<PropMLeap<VQCD_NONE>>		(field, propclass);
						break;
				break;
			}
		case PROP_RKN4:
			switch (pot) {
				case VQCD_1:
					prop = std::make_unique<PropRKN4<VQCD_1>>		(field, propclass);
					break;
				case VQCD_1_RHO:
					prop = std::make_unique<PropRKN4<VQCD_1_RHO>>		(field, propclass);
					break;
				case VQCD_1_DRHO:
					prop = std::make_unique<PropRKN4<VQCD_1_DRHO>>		(field, propclass);
					break;

				case VQCD_1_PQ_2:
					prop = std::make_unique<PropRKN4<VQCD_1_PQ_2>>		(field, propclass);
					break;
				case VQCD_1_PQ_2_RHO:
					prop = std::make_unique<PropRKN4<VQCD_1_PQ_2_RHO>>	(field, propclass);
					break;
				case VQCD_1_PQ_2_DRHO:
					prop = std::make_unique<PropRKN4<VQCD_1_PQ_2_DRHO>>	(field, propclass);
					break;

				case VQCD_2:
					prop = std::make_unique<PropRKN4<VQCD_2>>		(field, propclass);
					break;
				case VQCD_2_RHO:
					prop = std::make_unique<PropRKN4<VQCD_2_RHO>>		(field, propclass);
					break;
				case VQCD_2_DRHO:
					prop = std::make_unique<PropRKN4<VQCD_2_DRHO>>		(field, propclass);
					break;

				case VQCD_0:
					prop = std::make_unique<PropRKN4<VQCD_0>>		(field, propclass);
					break;
				case VQCD_0_RHO:
					prop = std::make_unique<PropRKN4<VQCD_0_RHO>>		(field, propclass);
					break;
				case VQCD_0_DRHO:
					prop = std::make_unique<PropRKN4<VQCD_0_DRHO>>		(field, propclass);
					break;

				case VQCD_PQ_ONLY:
					prop = std::make_unique<PropRKN4<VQCD_PQ_ONLY>>		(field, propclass);
					break;
				case VQCD_PQ_ONLY_RHO:
					prop = std::make_unique<PropRKN4<VQCD_PQ_ONLY_RHO>>		(field, propclass);
					break;
				case VQCD_PQ_ONLY_DRHO:
					prop = std::make_unique<PropRKN4<VQCD_PQ_ONLY_DRHO>>		(field, propclass);
					break;

				case VQCD_1N2:
					prop = std::make_unique<PropRKN4<VQCD_1N2>>		(field, propclass);
					break;
				case VQCD_1N2_RHO:
					prop = std::make_unique<PropRKN4<VQCD_1N2_RHO>>		(field, propclass);
					break;
				case VQCD_1N2_DRHO:
					prop = std::make_unique<PropRKN4<VQCD_1N2_DRHO>>		(field, propclass);
					break;

				case VQCD_NONE:
				default:
					prop = std::make_unique<PropRKN4<VQCD_NONE>>		(field, propclass);
					break;
			}

			break;

		default:
			LogError ("Error: unrecognized propagator %d", pType);
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
	LogMsg	(VERB_NORMAL, "Propagator %ssuccessfully initialized", prop->Name().c_str());
 	LogFlush();
}

using	namespace profiler;

void	propagate	(Scalar *field, const double dz)
{
	LogMsg	(VERB_HIGH, "[pate] Called propagator");
LogFlush();
	Profiler &prof = getProfiler(PROF_PROP);

LogFlush();
 	if	( (pType & (PROP_BASE | PROP_NNEIG)) && !field->Folded() )
	{
		Folder	munge(field);
		munge(FOLD_ALL);
	}

	prop->getBaseName();

	prof.start();

	char loli[2048];

	switch (field->Field()) {
		case FIELD_AXION:
		if (pType & (PROP_BASE | PROP_NNEIG)){
				sprintf (loli, "N %01d Ng %01d Axion", Nng, field->getNg());
				prop->appendName(loli);
			} else {
				prop->appendName("Axion");
			}
			(prop->propAxion)(dz);
			break;

		case FIELD_AXION_MOD:
			if (pType & (PROP_BASE | PROP_NNEIG)){
				sprintf (loli, "N %01d Ng %01d Axion Mod", Nng, field->getNg());
				prop->appendName(loli);
			} else {
				prop->appendName("Axion Mod");
			}
			(prop->propAxion)(dz);
			break;

		case FIELD_SAXION:
			if (pType & (PROP_BASE | PROP_NNEIG)){
				sprintf (loli, "N %01d Ng %01d Saxion", Nng, field->getNg());
				prop->appendName(loli);
			} else {
				prop->appendName("Saxion");
			}
			(prop->propSaxion)(dz);
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





void	resetPropagator(Scalar *field) {
	/*	Default block size gives just one block	*/

	LogMsg(VERB_NORMAL,"[tp] reseting!");
	if (pType & PROP_SPEC)
		return;
	if (pType & PROP_FSPEC)
		return;
	if (pType & PROP_NNEIG)
		return;

	int tmp   = field->DataAlign()/field->DataSize();
	int shift = 0;

	while (tmp != 1) {
		shift++;
		tmp >>= 1;
	}

	prop->SetBlockX(field->Length() << shift);
	prop->SetBlockY(field->Length() >> shift);
	prop->SetBlockZ(field->Depth ());
	prop->UpdateBestBlock();

	prop->UnTune();
}




void	tunePropagator (Scalar *field) {
	// Hash CPU model so we don't mix different cache files
	LogMsg(VERB_NORMAL,"\n");
 	LogMsg(VERB_NORMAL,"[tp] Tune propagator!\n");
	if (pType & PROP_SPEC)
		return;
	if (pType & PROP_FSPEC)
		return;
	// if (pType & PROP_NNEIG)
	// 	return;

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
		// else if (pType == PROP_NNEIG)
		// 	sprintf (tuneName, "%s/tuneCache%d.dat", wisDir,field->getNg());
		if (pType & PROP_BASE)
			sprintf (tuneName, "%s/tuneCache.dat", wisDir);
		else if (pType & PROP_NNEIG){
			LogMsg(VERB_HIGH,"%01d",field->getNg());
			sprintf (tuneName, "%s/tuneCache%01d.dat", wisDir,field->getNg());
			LogMsg(VERB_HIGH,"%s",tuneName);
			}
		if ((cacheFile = fopen(tuneName, "r")) == nullptr) {
LogMsg(VERB_HIGH,"[tp] new cache!!");
LogMsg (VERB_NORMAL, "Missing tuning cache file %s, will create a new one", tuneName);
			newFile = true;
		} else {
			int	     rMpi, rThreads;
			size_t       rLx, rLz, Nghost;
			unsigned int rBx, rBy, rBz, fType, myField = (field->Field() == FIELD_SAXION) ? 0 : 1;
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
		case FIELD_AXION:
		if (pType & (PROP_BASE | PROP_NNEIG)){
				sprintf (loli, "N %01d Ng %01d Axion", Nng, field->getNg());
				prop->appendName(loli);
			} else {
				prop->appendName("Axion");
			}
			break;

		case FIELD_AXION_MOD:
			if (pType & (PROP_BASE | PROP_NNEIG)){
				sprintf (loli, "N %01d Ng %01d Axion Mod", Nng, field->getNg());
				prop->appendName(loli);
			} else {
				prop->appendName("Axion Mod");
			}
			break;

		case FIELD_SAXION:
			if (pType & (PROP_BASE | PROP_NNEIG)){
				sprintf (loli, "N %01d Ng %01d Saxion", Nng, field->getNg());
				prop->appendName(loli);
			} else {
				prop->appendName("Saxion");
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
		if (pType & PROP_BASE)
			sprintf (tuneName, "%s/tuneCache.dat", wisDir);
		else if (pType & PROP_NNEIG){
			LogMsg(VERB_HIGH,"%01d",field->getNg());
			sprintf (tuneName, "%s/tuneCache%01d.dat", wisDir,field->getNg());
			LogMsg(VERB_HIGH,"%s",tuneName);
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

		unsigned int fType = (field->Field() == FIELD_SAXION) ? 0 : 1;
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
