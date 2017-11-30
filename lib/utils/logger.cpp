#include<memory>
#include"utils/logger.h"

namespace AxionsLog {
	std::shared_ptr<Logger> myLog;
	const char	levelTable[3][16] = { " Msg ", "Debug", "Error" };
}

void	createLogger(const int index, const LogMpi logMpi, const VerbosityLevel verbosity) {
	static bool	initLog = false;

	if	(initLog == false) {
		AxionsLog::myLog = std::make_shared<AxionsLog::Logger>(index, logMpi, verbosity);
	}	else	{
		LogMsg(VERB_HIGH, "Logger already initialized");
	}
}


