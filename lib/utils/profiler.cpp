#include<string>
#include<vector>
#include<map>
#include"utils/utils.h"
#include"enum-field.h"


namespace profiler {

	static	std::map<ProfType,Profiler>	profs;

	void	Profiler::printStats	() {
        	for (auto data = prof.cbegin(); data != prof.cend(); data++)
	        {
			std::string	name   = data->first;
		        FlopCounter	fCount = data->second;

//			if (fCount.Started() == true)
			        LogMsg (VERB_NORMAL, "\tFunction %-20s GFlops %lf\tGBytes %lf", name.c_str(), fCount.GFlops(), fCount.GBytes());
        	}
	}

	void	initProfilers() {

		Profiler	scalarProfiler("Scalar class");
		profs.insert(std::make_pair(PROF_SCALAR, scalarProfiler));


		Profiler	genConfProfiler("Genconf");
		profs.insert(std::make_pair(PROF_GENCONF, genConfProfiler));


		Profiler	propProfiler("Propagator");
		profs.insert(std::make_pair(PROF_PROP, propProfiler));


		Profiler	stringsProfiler("Strings");
		profs.insert(std::make_pair(PROF_STRING, stringsProfiler));


		Profiler	energyProfiler("Energy");
		profs.insert(std::make_pair(PROF_ENERGY, energyProfiler));


		Profiler	folderProfiler("Folder");
		profs.insert(std::make_pair(PROF_FOLD, folderProfiler));

		Profiler	hdf5Profiler("Hdf5 I/O");
		profs.insert(std::make_pair(PROF_HDF5, hdf5Profiler));
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
