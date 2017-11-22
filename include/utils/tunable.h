#ifndef	_TUNABLE_
	#define	_TUNABLE_
	#include <string>

	#ifdef	__AVX512F__
		#define	VecSize	64
	#elif	__AVX__
		#define	VecSize	32
	#else
		#define	VecSize	16
	#endif

	class	Tunable {
		protected:

		std::string	name;

		double		gFlops;
		double		gBytes;

		unsigned int	xBlock;
		unsigned int	yBlock;
		unsigned int	zBlock;

		unsigned int	xSize;
		unsigned int	ySize;
		unsigned int	zSize;

		bool		isTuned;

		public:

				Tunable() noexcept : gFlops(0.), gBytes(0.), xBlock(16), yBlock(4), zBlock(1), xSize(16), ySize(4), zSize(1), isTuned(false), name("") {}
				Tunable(unsigned int Lx, unsigned int Lz) noexcept : gFlops(0.), gBytes(0.), xBlock(Lx), yBlock(Lx), zBlock(Lz), isTuned(false), name("") {}

		double		GFlops () const noexcept { return gFlops; }
		double		GBytes () const noexcept { return gBytes; }

		unsigned int	BlockX () const noexcept { return xBlock; }
		unsigned int	BlockY () const noexcept { return yBlock; }
		unsigned int	BlockZ () const noexcept { return zBlock; }

		std::string	Name   () const noexcept { return name; }

		void		reset  ()                     { gFlops = 0.; gBytes = 0.; }
		void		add    (double GF, double GB) { gFlops += GF; gBytes += GB; }

		void		setName   (const char * newName) { name.assign(newName); }
		void		appendName(const char * appName) { name += std::string(appName); }

		void		changeBlockSize(size_t dataSize) {
			static bool init = false;

			if (!init) {
				xBlock = xSize*(VecSize/dataSize);
				auto memAvail = getCache()/(
			
		}
	};
#endif
