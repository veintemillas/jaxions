#ifndef	_TUNABLE_
	#define	_TUNABLE_
	#include <string>

	class	Tunable {
		protected:

		std::string	name;

		double		gFlops;
		double		gBytes;

		unsigned int	xBlock;
		unsigned int	yBlock;
		unsigned int	zBlock;

		unsigned int	xBest;
		unsigned int	yBest;
		unsigned int	zBest;

		unsigned int	ySize;
		unsigned int	zSize;

		bool		isTuned;

		public:

				Tunable() noexcept : gFlops(0.), gBytes(0.), xBlock(0), yBlock(0), zBlock(0), xBest(0), yBest(0), zBest(0),
						     ySize(0), zSize(0), isTuned(false), name("") {}

		double		GFlops () const noexcept { return gFlops; }
		double		GBytes () const noexcept { return gBytes; }

		unsigned int	BlockX () const noexcept { return xBlock; }
		unsigned int	BlockY () const noexcept { return yBlock; }
		unsigned int	BlockZ () const noexcept { return zBlock; }

		bool		IsTuned() const noexcept { return isTuned;  }
		void		UnTune ()       noexcept { isTuned = false; }

		unsigned int	TunedBlockX () const noexcept { return xBest; }
		unsigned int	TunedBlockY () const noexcept { return yBest; }
		unsigned int	TunedBlockZ () const noexcept { return zBest; }

		unsigned int	SetBlockX (unsigned int bSize) noexcept { xBlock = bSize; }
		unsigned int	SetBlockY (unsigned int bSize) noexcept { yBlock = bSize; }
		unsigned int	SetBlockZ (unsigned int bSize) noexcept { zBlock = bSize; }

		void		UpdateBestBlock() noexcept { yBest  = yBlock; zBest  = zBlock; }
		void		SetBestBlock()    noexcept { yBlock = yBest;  zBlock = zBest;  }

		void		AdvanceBlockSize() noexcept {

			if (yBlock < ySize) {
				do {
					yBlock++;
				}	while ((ySize % yBlock) != 0);
			} else {
				yBlock = 4;

				if (zBlock < zSize) {
					do {
						zBlock++;
					}	while ((zSize % zBlock) != 0);
				} else {
					isTuned = true;
				}
			}
		}

		std::string	Name   () const noexcept { return name; }

		void		reset  ()                     { gFlops = 0.; gBytes = 0.; }
		void		add    (double GF, double GB) { gFlops += GF; gBytes += GB; }

		void		setName   (const char * newName) { name.assign(newName); }
		void		appendName(const char * appName) { name += std::string(appName); }

		void		InitBlockSize(unsigned int Lx, unsigned int Lz, size_t dataSize, size_t alignSize) {
			int tmp   = alignSize/dataSize;
			int shift = 0;

			while (tmp != 1) {
				shift++;
				tmp >>= 1;
			}

			ySize = (Lx >> shift);
			zSize = Lz;

			xBest = xBlock = (Lx << shift);
			yBest = yBlock = 4;
			zBest = zBlock = 1;

			isTuned = false;
		}
	};

	
#endif
