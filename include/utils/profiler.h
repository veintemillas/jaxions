#ifndef	__PROFILER__
	#define	__PROFILER__

	#include<chrono>
	#include<string>
	#include<map>

	#include"utils/flopCounter.h"

	namespace profiler {

        	class   Profiler {

                	private:

                        	std::string nameProf;
	                        std::map<std::string,FlopCounter> prof;
				std::chrono::high_resolution_clock::time_point sTime;
				double	dTime;


        	        public:

						Profiler() {};
						Profiler(const char * pName) : nameProf(std::string(pName)) {};
						Profiler(const Profiler &p) : nameProf(p.nameProf), prof(p.prof) {};
						Profiler(Profiler &&p) : nameProf(std::move(p.nameProf)), prof(std::move(p.prof)) {};

						~Profiler() {};

				Profiler&	operator=(const Profiler &p) = default;

				void		start() { sTime = std::chrono::high_resolution_clock::now(); }
				void		stop()  { dTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - sTime).count()*1e-6; }
				void		add(std::string str, double gFlops, double gBytes) { prof[str].addTime(dTime); prof[str].addFlops(gFlops, gBytes); }

				void		insert(std::string mName, FlopCounter fCount) { prof.insert(std::make_pair(mName, fCount)); };

				std::string	name() { return nameProf; };
				void		name(const char *pName) { nameProf.assign(pName); };

				void    	printStats();

				std::map<std::string,FlopCounter>& Prof() { return prof; }
		};

		void	initProfilers();
		void	destroyProfilers();
		void	printProfStats();
		void	printMiniStats(double z, StringData strDen, ProfType prof, std::string counter);

		Profiler&	getProfiler(ProfType pType);
	};
#endif
