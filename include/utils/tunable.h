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

		bool legacySquareLayout;
		unsigned int alignBlock;

		bool		isTuned;
		bool		isGpu;

		bool adaptiveMode;
		unsigned int tunePhase;   // 0 = coarse X, 1 = coarse Y, 2 = coarse Z, 3 = refine X, ...

		public:

				Tunable() noexcept : name(""), gFlops(0.), gBytes(0.), xBlock(0), yBlock(0), zBlock(0), xBest(0), yBest(0), zBest(0),
						     ySize(0), zSize(0), isTuned(false), isGpu(false), legacySquareLayout(false), alignBlock(1) {}

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

		void AdvanceBlockSize() noexcept {
			if (legacySquareLayout) {
				// old square code unchanged
				if (isGpu) {
					do {
						if (xBlock < xMax) {
							do {
								xBlock++;
							} while ((xSize % xBlock) != 0);
						} else {
							xBlock = 7;
							do {
								xBlock++;
							} while ((xSize % xBlock) != 0);

							if (yBlock < yMax) {
								do {
									yBlock++;
								} while ((ySize % yBlock) != 0);
							} else {
								isTuned = true;
							}
						}
					} while (!isTuned && TotalThreads() > ((size_t) maxThreadsPerBlock()));
				} else {
					if (yBlock < ySize) {
						do {
							yBlock++;
						} while ((ySize % yBlock) != 0);
					} else {
						yBlock = 3;
						do {
							yBlock++;
						} while ((ySize % yBlock) != 0);

						if (zBlock < zSize) {
							do {
								zBlock++;
							} while ((zSize % zBlock) != 0);
						} else {
							isTuned = true;
						}
					}
				}
				return;
			}

			// generic path: no exact divisibility required
			if (!isGpu) {
					unsigned int step = (alignBlock == 0) ? 1 : alignBlock;
					unsigned int yStep = (ySize == 1) ? 1 : step;

					if (xBlock + step <= xMax) {
						xBlock += step;
					} else {
						xBlock = step;

						if (yBlock + yStep <= yMax) {
							yBlock += yStep;
						} else {
							yBlock = (ySize == 1) ? 1 : yStep;

							if (zBlock < zMax) {
								zBlock++;
							} else {
								isTuned = true;
							}
						}
					}

					return;
				}

				// keep GPU generic for now
				do {
					if (xBlock < xMax) {
						xBlock++;
					} else {
						xBlock = 1;

						if (yBlock < yMax) {
							yBlock++;
						} else {
							yBlock = 1;

							if (zBlock < zMax) {
								zBlock++;
							} else {
								isTuned = true;
							}
						}
					}
				} while (!isTuned && TotalThreads() > ((size_t) maxThreadsPerBlock()));
		}

		std::string	Name   () const noexcept { return name; }

		void		reset  ()                     { gFlops = 0.; gBytes = 0.; }
		void		add    (double GF, double GB) { gFlops += GF; gBytes += GB; }

		void		setName   (const char * newName) { name.assign(newName); }
		void		appendName(const char * appName) { name += std::string(appName); }

		void InitBlockSize(unsigned int Nx, unsigned int Ny, unsigned int Nz,
                   size_t dataSize, size_t alignSize, bool gpu) {

		unsigned int tmp = (dataSize == 0) ? 1 : alignSize / dataSize;
		if (tmp == 0) tmp = 1;
		alignBlock = tmp;

		bool validPackedShape =
			(Nx % tmp == 0) &&
			((Ny == 1) || (Ny % tmp == 0));

		if (!validPackedShape)
			LogError("Error: lattice shape incompatible with packed alignment requirements");

		// bool useLegacySquare = validPackedShape && (Nx == Ny) && (Ny > 1);
		// bool usePacked1D     = validPackedShape && (Ny == 1);
		// bool useAdaptiveRect = validPackedShape && (Nx != Ny) && (Ny > 1);

		int shift = 0;

		isGpu = gpu;
		legacySquareLayout = (Nx == Ny);

		if (legacySquareLayout) {
			// Keep old behaviour exactly
			if (!isGpu) {
				while (tmp != 1) {
					shift++;
					tmp >>= 1;
				}

				xSize = (Nx << shift);
				xMax  = xSize;
				ySize = (Nx >> shift);
				yMax  = ySize;
				zSize = Nz;
				zMax  = Nz;

				xBest = xBlock = xMax;
				yBest = yBlock = 2;
				zBest = zBlock = 1;
			} else {
				size_t xTmp = maxThreadsPerDim(0);
				size_t yTmp = maxThreadsPerDim(1);

				xMax = (Nx*Ny > xTmp) ? xTmp : Nx*Ny;
				yMax = (Nz > yTmp) ? yTmp : Nz;
				zMax = 1;

				xSize = Nx*Ny;
				ySize = Nz;
				zSize = 1;

				if (yTmp*maxGridSize(2) < Nz)
					LogError("Error: not enough threads on gpu to accomodate z-dimension");

				xBest = xBlock = 4;
				yBest = yBlock = 1;
				zBest = zBlock = 1;
			}
		} else {
			// Generic path for arbitrary Nx,Ny,Nz
			xSize = Nx;
			ySize = Ny;
			zSize = Nz;

			if (!isGpu) {

				xSize = Nx; ySize = Ny; zSize = Nz;
				xMax = Nx; yMax = Ny; zMax = Nz;
				xBest = xBlock = tmp; zBest = zBlock = 1;
				yBest = yBlock = Ny == 1? 1 : tmp;

			} else {
				size_t xTmp = maxThreadsPerDim(0);
				size_t yTmp = maxThreadsPerDim(1);
				size_t zTmp = maxThreadsPerDim(2);

				xMax = (Nx > xTmp) ? xTmp : Nx;
				yMax = (Ny > yTmp) ? yTmp : Ny;
				zMax = (Nz > zTmp) ? zTmp : Nz;

				xBest = xBlock = (xMax >= 4) ? 4 : xMax;
				yBest = yBlock = 1;
				zBest = zBlock = 1;
			}
		}

		isTuned = false;
	}

	void InitBlockSize(unsigned int Lx, unsigned int Lz, size_t dataSize, size_t alignSize, bool gpu = false) {
		InitBlockSize(Lx, Lx, Lz, dataSize, alignSize, gpu);
	}

	};


#endif
