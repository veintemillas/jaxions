#ifndef	_TUNABLE_
	#define	_TUNABLE_
	#include <comms/comms.h>
	#include <utils/logger.h>
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

		unsigned int	xMax;
		unsigned int	yMax;
		unsigned int	zMax;

		unsigned int	xSize;
		unsigned int	ySize;
		unsigned int	zSize;

		bool		isTuned;
		bool		isGpu;

		public:

				Tunable() noexcept : name(""), gFlops(0.), gBytes(0.), xBlock(0), yBlock(0), zBlock(0), xBest(0), yBest(0), zBest(0),
						     ySize(0), zSize(0), isTuned(false), isGpu(false) {}

		double		GFlops () const noexcept { return gFlops; }
		double		GBytes () const noexcept { return gBytes; }

		unsigned int	BlockX () const noexcept { return xBlock; }
		unsigned int	BlockY () const noexcept { return yBlock; }
		unsigned int	BlockZ () const noexcept { return zBlock; }

		bool		IsTuned() const noexcept { return isTuned;  }
		void		UnTune ()       noexcept { isTuned = false; }
		void		Tune   ()       noexcept { isTuned = true;  }

		unsigned int	TunedBlockX () const noexcept { return xBest; }
		unsigned int	TunedBlockY () const noexcept { return yBest; }
		unsigned int	TunedBlockZ () const noexcept { return zBest; }

		unsigned int	MaxBlockX () const noexcept { return xMax; }
		unsigned int	MaxBlockY () const noexcept { return yMax; }
		unsigned int	MaxBlockZ () const noexcept { return zMax; }

		size_t		TotalThreads() const noexcept { return xBlock*yBlock*zBlock; }

		void		SetBlockX (unsigned int bSize) noexcept { xBlock = bSize; }
		void		SetBlockY (unsigned int bSize) noexcept { yBlock = bSize; }
		void		SetBlockZ (unsigned int bSize) noexcept { zBlock = bSize; }

		void		UpdateBestBlock() noexcept { xBest  = xBlock; yBest  = yBlock; zBest  = zBlock; }
		void		SetBestBlock()    noexcept { xBlock = xBest;  yBlock = yBest;  zBlock = zBest;  }

		void		AdvanceBlockSize() noexcept {

			if (isGpu) {
				do {
					if (xBlock < xMax) {
						do {
							xBlock++;
						}	while ((xSize % xBlock) != 0);
					} else {
						xBlock = 8;

						if (yBlock < yMax) {
							do {
								yBlock++;
							}	while ((ySize % yBlock) != 0);
						} else {
							isTuned = true;
						}
					}
				}	while (!isTuned && TotalThreads() > ((size_t) maxThreadsPerBlock()));
			} else {
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
		}

		std::string	Name   () const noexcept { return name; }

		void		reset  ()                     { gFlops = 0.; gBytes = 0.; }
		void		add    (double GF, double GB) { gFlops += GF; gBytes += GB; }

		void		setName   (const char * newName) { name.assign(newName); }
		void		appendName(const char * appName) { name += std::string(appName); }

		void		InitBlockSize(unsigned int Lx, unsigned int Lz, size_t dataSize, size_t alignSize, bool gpu = false) {
			int tmp   = alignSize/dataSize;
			int shift = 0;
			//size_t lV = SIZE_MAX;

			isGpu = gpu;

			if (!isGpu) {

				while (tmp != 1) {
					shift++;
					tmp >>= 1;
				}

				xSize = (Lx << shift);
				xMax  = xSize;
				ySize = (Lx >> shift);
				yMax  = ySize;
				zSize = Lz;
				zMax  = Lz;

				xBest = xBlock = xMax;
				yBest = yBlock = 4;
				zBest = zBlock = 1;
			} else {
				size_t xTmp = maxThreadsPerDim(0); 
				size_t yTmp = maxThreadsPerDim(1);
				//auto zTmp = 1;
				//lV = maxThreadsPerBlock();

				xMax = (Lx*Lx > xTmp) ? xTmp : Lx*Lx;
				yMax = (Lz > yTmp) ? yTmp : Lz;
				zMax = 1;

				xSize = Lx*Lx;
				ySize = Lz;
				zSize = 1;

				if (yTmp*maxGridSize(2) < Lz)
					LogError("Error: not enough threads on gpu to accomodate z-dimension");

				xBest = xBlock = 8;
				yBest = yBlock = 1;
				zBest = zBlock = 1;
			}

			isTuned = false;
		}
	};


#endif
