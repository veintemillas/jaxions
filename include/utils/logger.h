#ifndef	__LOGGER__
	#define	__LOGGER__

	#include<string>
	#include<sstream>
	#include<fstream>
	#include<vector>
	#include<chrono>
	#include<memory>

	#include<cstdarg>
	#include<sys/stat.h>
	#include<unistd.h>
	#include<omp.h>

	#include"enum-field.h"
	#include"comms/comms.h"

	namespace AxionsLog {

		constexpr	long long int	logFreq	= 500;

		extern	const char	levelTable[3][16];

		class	Msg {
			private:
				std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;	// Timestamp

				int		tIdx;							// OMP thread
				int		size;							// Message size
				std::string	data;							// Log message
				LogLevel	logLevel;						// Level of logging (info, debug or error)

			public:
					 Msg(LogLevel logLevel, const int tIdx, const char * format, ...) noexcept : logLevel(logLevel), tIdx(tIdx) {
						char buffer[256];
						va_list args;
						va_start (args, format);
						size = vsnprintf (buffer, 255, format, args) + 1;
						va_end (args);

						data.assign(buffer, size);

						timestamp = std::chrono::high_resolution_clock::now();
					}

					 Msg(const Msg &myMsg) noexcept : tIdx(myMsg.tIdx), size(myMsg.size), data(myMsg.data), logLevel(myMsg.logLevel), timestamp(myMsg.timestamp) {};
					 Msg(Msg &&myMsg)      noexcept : tIdx(std::move(myMsg.tIdx)), size(std::move(myMsg.size)), data(std::move(myMsg.data)),
									  logLevel(std::move(myMsg.logLevel)), timestamp(myMsg.timestamp) {};

					~Msg() noexcept {};

				Msg&	operator=(const AxionsLog::Msg& msg) {
					tIdx = msg.tIdx;
					size = msg.size;
					data = msg.data;

					logLevel  = msg.logLevel;
					timestamp = msg.timestamp;
				}

				inline	long long int	time(std::chrono::time_point<std::chrono::high_resolution_clock> start) const {
					return std::chrono::duration_cast<std::chrono::milliseconds> (timestamp - start).count();
				}

				inline	int		thread() const { return tIdx; }
				inline	std::string	msg()    const { return data; }
				inline	const LogLevel	level()  const { return logLevel; }
		};

		class	Logger {
			private:
				std::chrono::time_point<std::chrono::high_resolution_clock> logStart;
				std::ofstream		oFile;

				std::vector<Msg>	msgStack;
				LogMpi			mpiType;
				const VerbosityLevel	verbose;

				void	printMsg	(const Msg &myMsg) noexcept {
					oFile << myMsg.time(logStart) << "ms: Logger level[" << levelTable[myMsg.level()] << "] Rank " << commRank() << "/" << commSize()
					      << " - Thread " << myMsg.thread() << "/" << omp_get_num_threads() << " ==> " << myMsg.msg() << std::endl;
				}

				void	flushMsg	() noexcept {
					auto it = msgStack.cbegin();
					printMsg(*it);
					msgStack.erase(it);
				}

				void	flushStack	() noexcept {
					for (auto it = msgStack.cbegin(); it != msgStack.cend(); it++)
						printMsg(*it);

					msgStack.clear();
				}

				void	flushDisk	() {
					oFile.flush();
				}

			public:
				 Logger(const int index, const LogMpi mpiType, const VerbosityLevel verbosity) : mpiType(mpiType), verbose(verbosity) {

					bool			test;
					struct stat		buffer;
					std::string		base;
					std::stringstream	ss;

					int			idx = index - 1;

					switch	(mpiType) {
						case	ALL_RANKS:
							ss << "axion.log.MpiR" << commRank() << ".";
							base = ss.str();
							break;

						case	ZERO_RANK:
							base.assign("axion.log.");
							break;
					}

					do {
						idx++;
						ss.str("");
						ss << base << idx;
						test = (stat (ss.str().c_str(), &buffer) == 0) ? true : false;
					}	while (test);

					oFile.open(ss.str().c_str(), std::ofstream::out);

					logStart = std::chrono::high_resolution_clock::now();

					banner();
				}

				~Logger() { flushStack(); flushDisk(); oFile.close(); }

				template<typename... Fargs>
				void	operator()(LogLevel level, const char * file, const int line, const char * format, Fargs... vars)
				{
					switch	(level) {

						case	LOG_MSG:
						{
							// We push the messages in the stack and we flush them later
							msgStack.push_back(std::move(Msg(level, omp_get_thread_num(), format, vars...)));

							if	(std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::high_resolution_clock::now() - logStart).count() > logFreq) {
								flushStack();
								flushDisk();
							}
							break;
						}

						case	LOG_DEBUG:
						{
							Msg	msg(level, omp_get_thread_num(), format, vars...);
							printMsg(msg);										// We directly write to disk
							flushDisk();
							break;
						}

						case	LOG_ERROR:
						{
							// We add a message to the stack and immediately flush the whole message stack
							msgStack.push_back(std::move(Msg(level, omp_get_thread_num(), "Error in file %s line %d", file, line)));
							msgStack.push_back(std::move(Msg(level, omp_get_thread_num(), format, vars...)));
							flushStack();
							flushDisk();
							break;
						}
					}
				}

				void	banner		() {
					(*this)(LOG_MSG, nullptr, 0, "JAxions logger started");
				}

				const VerbosityLevel	Verbosity	() const { return	verbose; }
		};

		extern std::shared_ptr<Logger> myLog;
	};

	void	createLogger(const int index, const LogMpi logMpi, const VerbosityLevel verb);

	#define	LogAll(logType, ...)	((*(AxionsLog::myLog))(logType,   __FILE__, __LINE__, __VA_ARGS__))
	#define	LogDebug(...)		((*(AxionsLog::myLog))(LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__))
	#define	LogError(...)		((*(AxionsLog::myLog))(LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__))
	#define	LogMsg(verb, ...)	do { if (AxionsLog::myLog->Verbosity() >= verb) { ((*(AxionsLog::myLog))(LOG_MSG, __FILE__, __LINE__, __VA_ARGS__)); } } while(0)
	#define LogOut(...) 		do { if (!commRank()) { printf(__VA_ARGS__); fflush(stdout); } } while(0)
#endif
