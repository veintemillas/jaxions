#ifndef	__LOGGER__
	#define	__LOGGER__

	#include<string>
	#include<cstring>
	#include<sstream>
	#include<fstream>
	#include<iomanip>
	#include<vector>
	#include<chrono>
	#include<memory>
	#include<cstddef>
	#include<algorithm>

	#include<cstdarg>
	#include<cstring>
	#include<sys/stat.h>
	#include<unistd.h>

	#include<omp.h>
	#include<mpi.h>

	#include"enum-field.h"
	#include"comms/comms.h"

	namespace AxionsLog {

		constexpr long long int	logFreq	= 5000000;
		constexpr size_t 	basePack = sizeof(ptrdiff_t)*5;

		extern	const char	levelTable[3][16];

		class	Msg {
			private:
				std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;	// Timestamp

				int		tIdx;							// OMP thread
				int		mRnk;							// MPI rank
				int		size;							// Message size
				std::string	data;							// Log message
				LogLevel	logLevel;						// Level of logging (info, debug or error)

				char		packed[1024];						// For serialization and MPI

				mutable MPI_Request req;

			public:
					Msg(LogLevel logLevel, const int tIdx, const char * format, ...) noexcept : logLevel(logLevel), tIdx(tIdx) {
					char buffer[1024 - basePack];
					va_list args;
					va_start (args, format);
					size = vsnprintf (buffer, 1023 - basePack, format, args);
					va_end (args);

					data.assign(buffer, size);

					timestamp = std::chrono::high_resolution_clock::now();
					mRnk = commRank();
				}

					 Msg(const Msg &myMsg) noexcept : tIdx(myMsg.tIdx), size(myMsg.size), data(myMsg.data), logLevel(myMsg.logLevel),
									  mRnk(myMsg.mRnk), timestamp(myMsg.timestamp), req(MPI_REQUEST_NULL) {};
					 Msg(Msg &&myMsg)      noexcept : tIdx(std::move(myMsg.tIdx)), size(std::move(myMsg.size)), data(std::move(myMsg.data)), mRnk(std::move(myMsg.mRnk)),
									  logLevel(std::move(myMsg.logLevel)), timestamp(myMsg.timestamp), req(MPI_REQUEST_NULL) {};
					 Msg(void *vPack)      noexcept : tIdx(static_cast<ptrdiff_t*>(vPack)[1]), size(static_cast<ptrdiff_t*>(vPack)[3]),
									  logLevel((LogLevel) static_cast<ptrdiff_t*>(vPack)[2]), mRnk(static_cast<ptrdiff_t*>(vPack)[4]) {
					auto mTime = static_cast<ptrdiff_t*>(vPack)[0];
					timestamp  = std::chrono::time_point<std::chrono::high_resolution_clock>(std::chrono::microseconds(mTime));

					req = MPI_REQUEST_NULL;
					char *msgData = static_cast<char*>(vPack) + basePack;
					data.assign(msgData, size);
				};

					~Msg() noexcept {};

				Msg&	operator=(const AxionsLog::Msg& msg) {
					tIdx = msg.tIdx;
					size = msg.size;
					data = msg.data;
					mRnk = msg.mRnk;

					logLevel  = msg.logLevel;
					timestamp = msg.timestamp;
				}

				inline	long long int	time(std::chrono::time_point<std::chrono::high_resolution_clock> start) const {
					return std::chrono::duration_cast<std::chrono::microseconds> (timestamp - start).count();
				}

				inline	int		thread() const { return tIdx; }
				inline	int		rank()   const { return mRnk; }
				inline	std::string	msg()    const { return data; }
				inline	const LogLevel	level()  const { return logLevel; }

				inline	char*		pack() {
					ptrdiff_t *sPack = static_cast<ptrdiff_t*>(static_cast<void*>(packed));
					sPack[0] = std::chrono::time_point_cast<std::chrono::microseconds> (timestamp).time_since_epoch().count();
					sPack[1] = tIdx;
					sPack[2] = (ptrdiff_t) logLevel;
					sPack[3] = data.length();
					sPack[4] = mRnk;

					memcpy (packed+basePack, data.data(), data.length());

					return	packed;
				}

				inline	MPI_Request&	mpiReq() { return req; }
		};

		class	Logger {
			private:
				std::chrono::time_point<std::chrono::high_resolution_clock> logStart;
				std::ofstream		oFile;
				const LogMpi		mpiType;
				std::vector<Msg>	msgStack;
				const VerbosityLevel	verbose;

				void	printMsg	(const Msg &myMsg) noexcept {
					oFile << std::setw(11) << myMsg.time(logStart)/1000 << "ms: Logger level[" << std::right << std::setw(5) << levelTable[myMsg.level()>>21] << "]"
					      << " Rank " << std::setw(4)  << myMsg.rank()+1 << "/" << commSize() << " - Thread " << std::setw(3) << myMsg.thread()+1 << "/"
					      << omp_get_num_threads() << " ==> " << myMsg.msg() << std::endl;
				}

				/* We only allow thread 0 to write to disk, but any other thread can put messages in the stack		*/
				/* The stack is flushed if there is an error on any thread because the variable mustFlush is shared	*/
				void	flushMsg	() noexcept {
					if (omp_get_thread_num() != 0)
						return;
					auto it = msgStack.cbegin();
					printMsg(*it);
					msgStack.erase(it);
				}

				void	flushStack	() noexcept {
					if (omp_get_thread_num() != 0)
						return;

					std::sort(msgStack.begin(), msgStack.end(), [logStart = logStart](Msg a, Msg b) { return (a.time(logStart) < b.time(logStart)); } );

					for (auto it = msgStack.cbegin(); it != msgStack.cend(); it++)
						printMsg(*it);

					msgStack.clear();
				}

				void	flushDisk	() {
					if (omp_get_thread_num() != 0)
						return;
					oFile.flush();
				}

				bool	getMpiMsg	(const LogLevel level) {
					if (omp_get_thread_num() != 0)
						return	false;

					bool msgPending = false;

					int flag = 0;
					MPI_Status status;

					// Get all the messages of a particular log level
					do {
						MPI_Iprobe(MPI_ANY_SOURCE, level, MPI_COMM_WORLD, &flag, &status);

						if (flag) {
							char packed[1024];
							int  mSize;
							auto srcRank = status.MPI_SOURCE;

							// Get message
							MPI_Get_count(&status, MPI_CHAR, &mSize);
							MPI_Recv(packed, mSize, MPI_CHAR, srcRank, level, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
							Msg msg(static_cast<void*>(packed));
							// Put the message in the stack
							msgStack.push_back(std::move(msg));
							msgPending = true;
						}
					}	while (flag);

					return	msgPending;
				}

			public:
				 Logger(const int index, const LogMpi mpiType, const VerbosityLevel verbosity) : mpiType(mpiType), verbose(verbosity) {

					bool			test;
					struct stat		buffer;
					std::string		base("axion.log.");
					std::stringstream	ss;

					int			idx = index - 1;

					// Let's force a sync before recording the starting time, so all the ranks more or less agree on this
					commSync();
					logStart = std::chrono::high_resolution_clock::now();

					if (commRank() == 0) {
						do {
							idx++;
							ss.str("");
							ss << base << idx;
							test = (stat (ss.str().c_str(), &buffer) == 0) ? true : false;
						}	while (test);

						oFile.open(ss.str().c_str(), std::ofstream::out);
						banner();
					}

				}

				// Receives pending MPI messages and flushes them to disk
				void	flushLog	() {
					if (omp_get_thread_num() != 0)
						return;

					// Get all the messages
					if (mpiType == ALL_RANKS) {
						getMpiMsg (LOG_MSG);
						getMpiMsg (LOG_ERROR);
					}

					flushStack();
					flushDisk();
				}

				~Logger() { int noMpi; MPI_Finalized(&noMpi); if (noMpi == 0) flushLog(); if (commRank()==0) { oFile.close(); } }

				template<typename... Fargs>
				void	operator()(LogLevel level, const char * file, const int line, const char * format, Fargs... vars)
				{
					static bool       mustFlush = false;
					static const int  myRank    = commRank();
					static const int  nSplit    = commSize();
					static const bool mpiLogger = ((nSplit > 1) && (mpiType == ALL_RANKS));

					if (mpiType == ZERO_RANK && myRank != 0)
						return;

					switch	(level) {

						case	LOG_MSG:
						{
							// We push the messages in the stack and we flush them later
							msgStack.push_back(std::move(Msg(level, omp_get_thread_num(), format, vars...)));

							if	(std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::high_resolution_clock::now() - logStart).count() > logFreq)
								mustFlush = true;
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
							auto thisThread = omp_get_thread_num();
							// We add a message to the stack and immediately flush the whole message stack
							msgStack.push_back(std::move(Msg(level, thisThread, "Error in file %s line %d", file, line)));
							msgStack.push_back(std::move(Msg(level, thisThread, format, vars...)));
							mustFlush = true;
							break;
						}
					}

					if (mpiLogger) {	// If we are using MPI
						if (myRank == 0) {
							// Get all the pending error messages. If found any, prepare to flush the stack
							if (getMpiMsg (LOG_ERROR))	// We use this if not to overwrite mustFlush
								mustFlush = true;
						} else {
							/* If loglevel is ERROR, we block comms until all the pending messages are sent and  we clear the stack */
							if (level == LOG_ERROR) {
								for (auto it = msgStack.begin(); it != msgStack.end();) {
									MPI_Request& req = it->mpiReq();
									if (req == MPI_REQUEST_NULL)
										MPI_Isend(it->pack(), it->msg().length() + basePack, MPI_CHAR, 0, level, MPI_COMM_WORLD, &req);
									MPI_Wait(&req, MPI_STATUS_IGNORE);
									it = msgStack.erase(it);
								}
								msgStack.clear();
							} else {
								/* As the messages are sent, we remove them from the stack */
								for (auto it = msgStack.begin(); it != msgStack.end();) {
									int flag = 0;
									MPI_Request& req = it->mpiReq();
									if (req == MPI_REQUEST_NULL) {
										MPI_Isend(it->pack(), it->msg().length() + basePack, MPI_CHAR, 0, level, MPI_COMM_WORLD, &req);
										it++;
									} else {
										MPI_Test(&req, &flag, MPI_STATUS_IGNORE);

										if (flag)
											it = msgStack.erase(it);
										else
											it++;
									}
								}
							}

						}
					}

					if (mustFlush && myRank == 0) {
						if (mpiLogger) {	// If we are using MPI
							// Get all the standard messages
							getMpiMsg(LOG_MSG);
						}

						flushStack();
						flushDisk();
						mustFlush = false;
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
	#define	LogFlush()		(AxionsLog::myLog->flushLog())
#endif
