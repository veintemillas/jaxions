#ifndef _TUNABLE_
#define _TUNABLE_

#include <comms/comms.h>
#include <utils/logger.h>
#include <string>
#include <vector>
#include <tuple>
#include <set>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <cstdio>

class Tunable {
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
	unsigned int tunePhase;

	// Adaptive-grid info
	unsigned int xMin;
	unsigned int yMin;
	unsigned int zMin;

	unsigned int xStepMin;
	unsigned int yStepMin;
	unsigned int zStepMin;

	double adaptiveStopRelImprove;
	unsigned int adaptiveCoarsePoints;
	unsigned int adaptiveMaxEvals;

	public:

	struct BlockCandidate {
		unsigned int bx = 1;
		unsigned int by = 1;
		unsigned int bz = 1;
		size_t       timeNs = std::numeric_limits<size_t>::max();
		bool         valid = false;
		bool         predicted = false;
		int          level = 0;
		int          regionId = 0;
	};

	struct SearchStats {
		size_t nValid   = 0;
		size_t nInvalid = 0;
		double minTime    = 0.0;
		double maxTime    = 0.0;
		double meanTime   = 0.0;
		double medianTime = 0.0;
		double stdTime    = 0.0;
		BlockCandidate best;
	};

	struct SearchRegion {
		unsigned int xLo = 1, xHi = 1;
		unsigned int yLo = 1, yHi = 1;
		unsigned int zLo = 1, zHi = 1;
		unsigned int dx  = 1, dy  = 1, dz  = 1;
		int level = 0;
		int regionId = 0;
	};

	Tunable() noexcept :
		name(""), gFlops(0.), gBytes(0.),
		xBlock(0), yBlock(0), zBlock(0),
		xBest(0), yBest(0), zBest(0),
		xMax(0), yMax(0), zMax(0),
		xSize(0), ySize(0), zSize(0),
		legacySquareLayout(false), alignBlock(1),
		isTuned(false), isGpu(false),
		adaptiveMode(false), tunePhase(0),
		xMin(1), yMin(1), zMin(1),
		xStepMin(1), yStepMin(1), zStepMin(1),
		adaptiveStopRelImprove(0.01),
		adaptiveCoarsePoints(5),
		adaptiveMaxEvals(160)
	{}

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

	unsigned int	MinBlockX () const noexcept { return xMin; }
	unsigned int	MinBlockY () const noexcept { return yMin; }
	unsigned int	MinBlockZ () const noexcept { return zMin; }

	unsigned int	StepBlockX () const noexcept { return xStepMin; }
	unsigned int	StepBlockY () const noexcept { return yStepMin; }
	unsigned int	StepBlockZ () const noexcept { return zStepMin; }

	bool		UsesLegacySquareLayout() const noexcept { return legacySquareLayout; }
	bool		AdaptiveMode() const noexcept { return adaptiveMode; }

	double          AdaptiveStopRelImprove() const noexcept { return adaptiveStopRelImprove; }
	unsigned int    AdaptiveCoarsePoints() const noexcept { return adaptiveCoarsePoints; }
	unsigned int    AdaptiveMaxEvals() const noexcept { return adaptiveMaxEvals; }

	void SetAdaptiveStopRelImprove(double x) noexcept {
		adaptiveStopRelImprove = (x > 0.0) ? x : 0.01;
	}
	void SetAdaptiveCoarsePoints(unsigned int n) noexcept {
		adaptiveCoarsePoints = (n < 3) ? 3 : n;
	}
	void SetAdaptiveMaxEvals(unsigned int n) noexcept {
		adaptiveMaxEvals = (n == 0) ? 1 : n;
	}

	size_t		TotalThreads() const noexcept { return xBlock*yBlock*zBlock; }

	void		SetBlockX (unsigned int bSize) noexcept { xBlock = bSize; }
	void		SetBlockY (unsigned int bSize) noexcept { yBlock = bSize; }
	void		SetBlockZ (unsigned int bSize) noexcept { zBlock = bSize; }

	void		UpdateBestBlock() noexcept { xBest  = xBlock; yBest  = yBlock; zBest  = zBlock; }
	void		SetBestBlock()    noexcept { xBlock = xBest;  yBlock = yBest;  zBlock = zBest;  }

	// Keep old iterator for fallback / legacy use
	void AdvanceBlockSize() noexcept {
		if (legacySquareLayout) {
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
	                   size_t dataSize, size_t alignSize, bool gpu)
	{
		unsigned int tmp = (dataSize == 0) ? 1 : alignSize / dataSize;
		if (tmp == 0) tmp = 1;
		alignBlock = tmp;

		bool validPackedShape =
			(Nx % tmp == 0) &&
			((Ny == 1) || (Ny % tmp == 0));

		if (!validPackedShape)
			LogError("Error: lattice shape incompatible with packed alignment requirements");

		int shift = 0;

		isGpu = gpu;
		legacySquareLayout = false; //(Nx == Ny);
		adaptiveMode = !legacySquareLayout;

		if (legacySquareLayout) {
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

				xMin = 1; yMin = 1; zMin = 1;
				xStepMin = 1; yStepMin = 1; zStepMin = 1;

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

				xMin = 1; yMin = 1; zMin = 1;
				xStepMin = 1; yStepMin = 1; zStepMin = 1;

				if (yTmp*maxGridSize(2) < Nz)
					LogError("Error: not enough threads on gpu to accomodate z-dimension");

				xBest = xBlock = 4;
				yBest = yBlock = 1;
				zBest = zBlock = 1;
			}
		} else {
			xSize = Nx;
			ySize = Ny;
			zSize = Nz;

			if (!isGpu) {
				xMax = Nx;
				yMax = Ny;
				zMax = Nz;

				xMin = tmp;
				yMin = (Ny == 1) ? 1 : tmp;
				zMin = 1;

				xStepMin = tmp;
				yStepMin = (Ny == 1) ? 1 : tmp;
				zStepMin = 1;

				xBest = xBlock = xMin;
				yBest = yBlock = yMin;
				zBest = zBlock = zMin;
			} else {
				size_t xTmp = maxThreadsPerDim(0);
				size_t yTmp = maxThreadsPerDim(1);
				size_t zTmp = maxThreadsPerDim(2);

				xMax = (Nx > xTmp) ? xTmp : Nx;
				yMax = (Ny > yTmp) ? yTmp : Ny;
				zMax = (Nz > zTmp) ? zTmp : Nz;

				xMin = 1;
				yMin = 1;
				zMin = 1;

				xStepMin = 1;
				yStepMin = 1;
				zStepMin = 1;

				xBest = xBlock = (xMax >= 4) ? 4 : xMax;
				yBest = yBlock = 1;
				zBest = zBlock = 1;

				#ifdef USE_2DCYL
        			ySize = 1;
        			yMin  = 1;
        			yMax  = 1;
        			yStepMin = 1;
        			yBest = yBlock = 1;
				#endif

			}
		}

		isTuned = false;
		tunePhase = 0;
	}

	void InitBlockSize(unsigned int Lx, unsigned int Lz, size_t dataSize, size_t alignSize, bool gpu = false)
	{
		InitBlockSize(Lx, Lx, Lz, dataSize, alignSize, gpu);
	}

	unsigned int SnapToGrid(unsigned int v, unsigned int lo, unsigned int hi, unsigned int step) const noexcept
	{
		if (step == 0) step = 1;
		if (v <= lo) return lo;

		unsigned int last = lo + ((hi - lo) / step) * step;
		if (v >= hi) return last;

		unsigned int k = (v - lo + step/2) / step;
		unsigned int s = lo + k * step;
		if (s > hi) s = last;
		return s;
	}

	bool IsValidBlock(unsigned int bx, unsigned int by, unsigned int bz) const noexcept
	{
		if (bx < xMin || bx > xMax) return false;
		if (by < yMin || by > yMax) return false;
		if (bz < zMin || bz > zMax) return false;

		if (((bx - xMin) % xStepMin) != 0) return false;
		if (((by - yMin) % yStepMin) != 0) return false;
		if (((bz - zMin) % zStepMin) != 0) return false;

		if (isGpu && ((size_t) bx * (size_t) by * (size_t) bz > (size_t) maxThreadsPerBlock()))
			return false;

		return true;
	}

	std::vector<unsigned int> SampleDim(unsigned int lo, unsigned int hi,
	                                    unsigned int step, unsigned int nPts) const
	{
		std::vector<unsigned int> vals;
		if (nPts < 2) nPts = 2;
		if (step == 0) step = 1;

		if (hi <= lo) {
			vals.push_back(lo);
			return vals;
		}

		unsigned int nFine = 1 + (hi - lo) / step;
		if (nFine <= nPts) {
			for (unsigned int i = 0; i < nFine; i++)
				vals.push_back(lo + i * step);
			return vals;
		}

		for (unsigned int i = 0; i < nPts; i++) {
			double alpha = (double) i / (double) (nPts - 1);
			double raw = (double) lo + alpha * (double) (hi - lo);
			unsigned int s = SnapToGrid((unsigned int) std::llround(raw), lo, hi, step);
			vals.push_back(s);
		}

		std::sort(vals.begin(), vals.end());
		vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
		return vals;
	}

	std::vector<unsigned int> SampleDimHighToLow(unsigned int lo, unsigned int hi,
	                                             unsigned int step, unsigned int nPts) const
	{
		std::vector<unsigned int> vals;
		if (nPts < 2) nPts = 2;
		if (step == 0) step = 1;

		if (hi <= lo) {
			vals.push_back(lo);
			return vals;
		}

		unsigned int nFine = 1 + (hi - lo) / step;
		if (nFine <= nPts) {
			for (unsigned int i = 0; i < nFine; i++)
				vals.push_back(hi - i * step);
		} else {
			for (unsigned int i = 0; i < nPts; i++) {
				double alpha = (double) i / (double) (nPts - 1);
				double raw = (double) hi - alpha * (double) (hi - lo);
				unsigned int s = SnapToGrid((unsigned int) std::llround(raw), lo, hi, step);
				vals.push_back(s);
			}
		}

		std::sort(vals.begin(), vals.end(), std::greater<unsigned int>());
		vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
		return vals;
	}

	std::vector<BlockCandidate> SampleRegion(const SearchRegion &R, unsigned int nPts,
	                                         bool predicted = false) const
	{
		std::vector<BlockCandidate> out;

		// auto xs = SampleDim(R.xLo, R.xHi, R.dx, nPts);
		auto xs = SampleDimHighToLow(R.xLo, R.xHi, R.dx, nPts);
		auto ys = SampleDim(R.yLo, R.yHi, R.dy, nPts);
		auto zs = SampleDim(R.zLo, R.zHi, R.dz, nPts);

		std::set<std::tuple<unsigned int,unsigned int,unsigned int>> seen;

		for (auto bx : xs) {
			for (auto by : ys) {
				for (auto bz : zs) {
					if (!IsValidBlock(bx, by, bz))
						continue;

					auto key = std::make_tuple(bx, by, bz);
					if (!seen.insert(key).second)
						continue;

					BlockCandidate c;
					c.bx = bx;
					c.by = by;
					c.bz = bz;
					c.predicted = predicted;
					c.level = R.level;
					c.regionId = R.regionId;
					out.push_back(c);
				}
			}
		}

		return out;
	}

	SearchStats ComputeStats(const std::vector<BlockCandidate> &cand) const
	{
		SearchStats S;
		std::vector<double> t;
		t.reserve(cand.size());

		for (const auto &c : cand) {
			if (c.valid) {
				S.nValid++;
				t.push_back((double) c.timeNs);
				if (S.nValid == 1 || c.timeNs < S.best.timeNs)
					S.best = c;
			} else {
				S.nInvalid++;
			}
		}

		if (t.empty())
			return S;

		std::sort(t.begin(), t.end());

		S.minTime  = t.front();
		S.maxTime  = t.back();
		S.meanTime = std::accumulate(t.begin(), t.end(), 0.0) / (double) t.size();

		if (t.size() % 2)
			S.medianTime = t[t.size()/2];
		else
			S.medianTime = 0.5 * (t[t.size()/2 - 1] + t[t.size()/2]);

		double var = 0.0;
		for (auto x : t) {
			double d = x - S.meanTime;
			var += d*d;
		}
		var /= (double) t.size();
		S.stdTime = std::sqrt(var);

		return S;
	}

	BlockCandidate ProposeFromTopK(const std::vector<BlockCandidate> &cand, int k,
	                               const SearchRegion &R) const
	{
		std::vector<BlockCandidate> valid;
		for (const auto &c : cand)
			if (c.valid)
				valid.push_back(c);

		if (valid.empty()) {
			BlockCandidate out;
			out.bx = xBest;
			out.by = yBest;
			out.bz = zBest;
			out.predicted = true;
			out.level = R.level;
			out.regionId = R.regionId;
			return out;
		}

		std::sort(valid.begin(), valid.end(),
		          [](const BlockCandidate &a, const BlockCandidate &b) {
			          return a.timeNs < b.timeNs;
		          });

		if ((int) valid.size() > k)
			valid.resize((size_t) k);

		double t0 = (double) valid.front().timeNs;
		double eps = 1.0;

		double sx = 0.0, sy = 0.0, sz = 0.0, sw = 0.0;
		for (const auto &c : valid) {
			double w = 1.0 / (((double) c.timeNs - t0) + eps);
			sx += w * (double) c.bx;
			sy += w * (double) c.by;
			sz += w * (double) c.bz;
			sw += w;
		}

		BlockCandidate out;
		out.bx = SnapToGrid((unsigned int) std::llround(sx / sw), R.xLo, R.xHi, R.dx);
		out.by = SnapToGrid((unsigned int) std::llround(sy / sw), R.yLo, R.yHi, R.dy);
		out.bz = SnapToGrid((unsigned int) std::llround(sz / sw), R.zLo, R.zHi, R.dz);
		out.predicted = true;
		out.level = R.level;
		out.regionId = R.regionId;

		if (!IsValidBlock(out.bx, out.by, out.bz))
			out = valid.front();

		out.predicted = true;
		out.level = R.level;
		out.regionId = R.regionId;
		return out;
	}

	SearchRegion RefineAroundBest(const SearchRegion &R, const BlockCandidate &best, int newRegionId) const
	{
		auto shrink = [](unsigned int lo, unsigned int hi, unsigned int step) -> unsigned int {
			if (hi <= lo) return step;
			unsigned int width = hi - lo;
			unsigned int h = width / 4;
			if (h < step) h = step;
			return h;
		};

		unsigned int hx = shrink(R.xLo, R.xHi, R.dx);
		unsigned int hy = shrink(R.yLo, R.yHi, R.dy);
		unsigned int hz = shrink(R.zLo, R.zHi, R.dz);

		SearchRegion out;
		out.xLo = SnapToGrid((best.bx > hx) ? best.bx - hx : xMin, xMin, xMax, xStepMin);
		out.xHi = SnapToGrid(best.bx + hx, xMin, xMax, xStepMin);
		out.yLo = SnapToGrid((best.by > hy) ? best.by - hy : yMin, yMin, yMax, yStepMin);
		out.yHi = SnapToGrid(best.by + hy, yMin, yMax, yStepMin);
		out.zLo = SnapToGrid((best.bz > hz) ? best.bz - hz : zMin, zMin, zMax, zStepMin);
		out.zHi = SnapToGrid(best.bz + hz, zMin, zMax, zStepMin);

		out.dx = R.dx;
		out.dy = R.dy;
		out.dz = R.dz;
		out.level = R.level + 1;
		out.regionId = newRegionId;
		return out;
	}

	bool RegionAtMinResolution(const SearchRegion &R) const noexcept
	{
		return ((R.xHi <= R.xLo + xStepMin) &&
		        (R.yHi <= R.yLo + yStepMin) &&
		        (R.zHi <= R.zLo + zStepMin));
	}

	void AppendTuneLogHeader(FILE *f, const char *fieldName, const char *devName,
	                         unsigned int Nx, unsigned int Ny, unsigned int Nz) const
	{
		if (f == nullptr) return;

		fprintf(f, "# adaptive tune\n");
		fprintf(f, "# field %s\n", fieldName);
		fprintf(f, "# device %s\n", devName);
		fprintf(f, "# lattice %u %u %u\n", Nx, Ny, Nz);
		fprintf(f, "# xrange [%u,%u] step=%u\n", xMin, xMax, xStepMin);
		fprintf(f, "# yrange [%u,%u] step=%u\n", yMin, yMax, yStepMin);
		fprintf(f, "# zrange [%u,%u] step=%u\n", zMin, zMax, zStepMin);
		fprintf(f, "# stop_rel_improve %.8f\n", adaptiveStopRelImprove);
		fprintf(f, "# coarse_points %u\n", adaptiveCoarsePoints);
		fprintf(f, "# max_evals %u\n", adaptiveMaxEvals);
		fprintf(f, "# columns: eval level region source bx by bz time_ns valid predicted\n");
		fflush(f);
	}

	void AppendBatchLog(FILE *f, const char *source, const std::vector<BlockCandidate> &cand, const SearchStats &S) const
	{
		if (f == nullptr) return;

		fprintf(f,
		        "# batch level=%d region=%d source=%s n_valid=%zu n_invalid=%zu min=%.3f max=%.3f mean=%.3f median=%.3f std=%.3f best=%u %u %u\n",
		        cand.empty() ? -1 : cand.front().level,
		        cand.empty() ? -1 : cand.front().regionId,
		        source,
		        S.nValid, S.nInvalid,
		        S.minTime, S.maxTime, S.meanTime, S.medianTime, S.stdTime,
		        S.best.bx, S.best.by, S.best.bz);

		static size_t evalCounter = 0;
		for (const auto &c : cand) {
			fprintf(f, "%zu %d %d %s %u %u %u %zu %d %d\n",
			        evalCounter++, c.level, c.regionId, source,
			        c.bx, c.by, c.bz, c.timeNs, c.valid ? 1 : 0, c.predicted ? 1 : 0);
		}

		fflush(f);
	}
};

#endif
