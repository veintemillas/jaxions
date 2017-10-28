#include<string>
#include<vector>
#include<map>
#include"utils/utils.h"
#include"enum-field.h"
#include"comms/comms.h"


namespace profiler {

	static	std::map<ProfType,Profiler>			profs;
	static	std::chrono::high_resolution_clock::time_point	stPt;

	static	double tTime = 0.;

	double	Profiler::printStats	() {
		double	aTime = 0.;

		for (auto data = prof.cbegin(); data != prof.cend(); data++)
	        {
			std::string	name   = data->first;
		        FlopCounter	fCount = data->second;

			aTime += fCount.DTime();

			LogMsg (VERB_SILENT, "\tFunction %-20s GFlops %.4lf\tGBytes %.4lf\tTotal time %.2lfs (%.2lf\%)", name.c_str(), fCount.GFlops(), fCount.GBytes(), fCount.DTime(), 100.*fCount.DTime()/tTime);
        	}

		return	aTime;
	}

	void	initProfilers() {

		stPt = std::chrono::high_resolution_clock::now();

		Profiler	scalarProfiler("Scalar");
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
		if (commRank() != 0)
			return;

		double	aTime = 0.;

		tTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - stPt).count()*1e-6;

		for (auto &data : profs) {
			auto &cProf = data.second;
			LogMsg(VERB_SILENT, "Profiler %s:", cProf.name().c_str());
			auto cTime = cProf.printStats();
			aTime += cTime;
			LogMsg(VERB_SILENT, "Total %s: %.2lf", cProf.name().c_str(), cTime);
			LogMsg(VERB_SILENT, "");
		}
		LogMsg (VERB_SILENT, "Unaccounted time %.2lfs of %.2lfs (%.2lf\%)", tTime - aTime, tTime, 100.*(1. - aTime/tTime));
	}

	Profiler&	getProfiler(ProfType pType) {
		return	profs[pType];
	}
};
