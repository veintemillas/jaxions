#include<string>
#include<vector>
#include<map>
#include"utils/flopCounter.h"
#include"utils/logger.h"
#include"enum-field.h"

#include"utils/profiler.h"


namespace profiler {

	static	std::map<ProfType,Profiler>	profs;

	void	Profiler::printStats	() {
        	for (auto data = prof.cbegin(); data != prof.cend(); data++)
	        {
			std::string	name   = data->first;
		        FlopCounter	fCount = data->second;

			if (fCount.Started() == true)
			        LogMsg (VERB_NORMAL, "\tFunction %s\tGFlops %lf\tGBytes %lf", name.c_str(), fCount.GFlops(), fCount.GBytes());
        	}
	}

	void	initProfilers() {

		Profiler	scalarProfiler("Scalar class");

		FlopCounter		initScalarfCounter;
		scalarProfiler.insert(std::move(std::string("Init")), initScalarfCounter);

		FlopCounter		normCoreScalarfCounter;
		scalarProfiler.insert(std::move(std::string("Normalise Core")), normCoreScalarfCounter);

		FlopCounter		normFieldScalarfCounter;
		scalarProfiler.insert(std::move(std::string("Normalise")), normFieldScalarfCounter);

		profs.insert(std::make_pair(PROF_SCALAR, scalarProfiler));


		Profiler	genConfProfiler("Genconf");

		FlopCounter		smootherfCounter;
		genConfProfiler.insert(std::string("Smoother"), smootherfCounter);

		FlopCounter		randfCounter;
		genConfProfiler.insert(std::string("Random"), randfCounter);

		profs.insert(std::make_pair(PROF_GENCONF, genConfProfiler));


		Profiler	propProfiler("Propagator");

		FlopCounter		omelyan3SaxionfCounter;
		propProfiler.insert(std::string("Omelyan3 Saxion"), omelyan3SaxionfCounter);

		FlopCounter		omelyan3SaxionLMfCounter;
		propProfiler.insert(std::string("Omelyan3 Saxion Lowmem"), omelyan3SaxionLMfCounter);

		FlopCounter		omelyan3AxionfCounter;
		propProfiler.insert(std::string("Omelyan3 Axion"),  omelyan3AxionfCounter);

		FlopCounter		rkn4SaxionfCounter;
		propProfiler.insert(std::string("RKN4 Saxion"), rkn4SaxionfCounter);

		FlopCounter		rkn4sSaxionfCounter;
		propProfiler.insert(std::string("RKN4 Spectral Saxion"), rkn4sSaxionfCounter);

		FlopCounter		rkn4SaxionLMfCounter;
		propProfiler.insert(std::string("RKN4 Saxion Lowmem"), rkn4SaxionLMfCounter);

		FlopCounter		rkn4AxionfCounter;
		propProfiler.insert(std::string("RKN4 Axion"),  rkn4AxionfCounter);

		FlopCounter		rkn4sAxionfCounter;
		propProfiler.insert(std::string("RKN4 Spectral Axion"),  rkn4sAxionfCounter);

		profs.insert(std::make_pair(PROF_PROP, propProfiler));


		Profiler	stringsProfiler("Strings");

		FlopCounter	stringsfCounter;
		stringsProfiler.insert(std::string("Strings and walls"), stringsfCounter);

		profs.insert(std::make_pair(PROF_STRING, stringsProfiler));


		Profiler	energyProfiler("Energy");

		FlopCounter	energySaxionfCounter;
		energyProfiler.insert(std::string("Energy Saxion"), energySaxionfCounter);

		FlopCounter	energyAxionfCounter;
		energyProfiler.insert(std::string("Energy Axion"),  energyAxionfCounter);

		FlopCounter	energySaxionMapfCounter;
		energyProfiler.insert(std::string("EnergyMap Saxion"), energySaxionMapfCounter);

		FlopCounter	energyAxionMapfCounter;
		energyProfiler.insert(std::string("EnergyMap Axion"),  energyAxionMapfCounter);

		profs.insert(std::make_pair(PROF_ENERGY, energyProfiler));


		Profiler	folderProfiler("Folder");

		FlopCounter	foldfCounter;
		folderProfiler.insert(std::string("Fold"), foldfCounter);

		FlopCounter	unfoldfCounter;
		folderProfiler.insert(std::string("Unfold"), unfoldfCounter);

		FlopCounter	unfoldSlicefCounter;
		folderProfiler.insert(std::string("Unfold slice"), unfoldSlicefCounter);

		profs.insert(std::make_pair(PROF_FOLD, folderProfiler));

		Profiler	hdf5Profiler("Hdf5 I/O");

		FlopCounter	readfCounter;
		folderProfiler.insert(std::string("Read configuration"),  readfCounter);

		FlopCounter	writefCounter;
		folderProfiler.insert(std::string("Write configuration"), writefCounter);

		FlopCounter	wrStrfCounter;
		folderProfiler.insert(std::string("Write strings"), wrStrfCounter);

		FlopCounter	wEngyfCounter;
		folderProfiler.insert(std::string("Write energy"), wEngyfCounter);

		FlopCounter	wEmapfCounter;
		folderProfiler.insert(std::string("Write energy map"), wEmapfCounter);

		FlopCounter	wArrayfCounter;
		folderProfiler.insert(std::string("Write array"), wArrayfCounter);

		profs.insert(std::make_pair(PROF_HDF5, hdf5Profiler));
	}

	void	destroyProfilers() {
		for (auto &pf : profs) {
			auto &cProf = pf.second;
			for (auto &x : cProf.Prof()) {
				auto name = x.first;
				cProf.Prof().erase(name);
			}

			auto name = pf.first;
			profs.erase(name);
		}
	}

	void	printMiniStats(double z, StringData strDen, ProfType prof, std::string counter) {
		Profiler& myProf = getProfiler(prof);
		auto gFlops = myProf.Prof()[counter].GFlops();
		auto gBytes = myProf.Prof()[counter].GBytes();
		LogOut ("z %lf\tnStrings %lu\tnWalls %lu\tGFlops %.3f\tGBytes %.3lf\n", z, strDen.strDen, strDen.wallDn, gFlops, gBytes);
	}

	void	printProfStats() {
		for (auto &data : profs) {
			auto &cProf = data.second;
			LogMsg(VERB_NORMAL, "Profiler %s:", cProf.name().c_str());
			cProf.printStats();
			LogMsg(VERB_NORMAL, "");
		}
	}

	Profiler&	getProfiler(ProfType pType) {
		return	profs[pType];
	}
};
