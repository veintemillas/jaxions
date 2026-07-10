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
#include <functional>

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

	std::vector<unsigned int> xCandidates;
	std::vector<unsigned int> yCandidates;
	std::vector<unsigned int> zCandidates;

	double adaptiveStopRelImprove;
	unsigned int adaptiveCoarsePoints;
	unsigned int adaptiveMaxEvals;

	bool FitsAxisGrid(unsigned int b, unsigned int lo, unsigned int hi, unsigned int step) const noexcept
	{
		if (step == 0) step = 1;
		if (b < lo || b > hi) return false;
		return (((b - lo) % step) == 0);
	}

	bool DividesAxis(unsigned int extent, unsigned int b) const noexcept
	{
		return (b != 0) && (extent != 0) && ((extent % b) == 0);
	}

	std::vector<unsigned int> AxisDivisors(unsigned int lo, unsigned int hi,
	                                      unsigned int step, unsigned int extent,
	                                      bool highToLow = false) const
	{
		std::vector<unsigned int> vals;
		if (step == 0) step = 1;

		for (unsigned int v = lo; v <= hi; v++) {
			if (!FitsAxisGrid(v, lo, hi, step))
				continue;
			if (!DividesAxis(extent, v))
				continue;
			vals.push_back(v);
		}

		if (highToLow)
			std::sort(vals.begin(), vals.end(), std::greater<unsigned int>());
		else
			std::sort(vals.begin(), vals.end());

		return vals;
	}

	unsigned int SnapToAxisDivisor(unsigned int v, unsigned int lo, unsigned int hi,
	                               unsigned int step, unsigned int extent) const
	{
		auto vals = AxisDivisors(lo, hi, step, extent, false);
		if (vals.empty())
			return SnapToGrid(v, lo, hi, step);

		return *std::min_element(vals.begin(), vals.end(),
			[v](unsigned int a, unsigned int b) {
				unsigned int da = (a > v) ? (a - v) : (v - a);
				unsigned int db = (b > v) ? (b - v) : (v - b);
				if (da != db) return da < db;
				return a > b;
			});
	}

	void NormalizeInitialBlock()
	{
		xBlock = SnapToCandidate(xBlock, xCandidates);
		yBlock = SnapToCandidate(yBlock, yCandidates);
		zBlock = SnapToCandidate(zBlock, zCandidates);

		if (!IsValidBlock(xBlock, yBlock, zBlock)) {
			auto xs = AxisDivisors(xMin, xMax, xStepMin, xSize, true);
			auto ys = AxisDivisors(yMin, yMax, yStepMin, ySize, false);
			auto zs = AxisDivisors(zMin, zMax, zStepMin, zSize, false);

			for (auto bx : xs) {
				for (auto by : ys) {
					for (auto bz : zs) {
						if (!IsValidBlock(bx, by, bz))
							continue;

						xBlock = bx;
						yBlock = by;
						zBlock = bz;
						xBest = xBlock;
						yBest = yBlock;
						zBest = zBlock;
						return;
					}
				}
			}
		}

		xBest = xBlock;
		yBest = yBlock;
		zBest = zBlock;
	}

	void BuildCandidateLists()
	{
		xCandidates = AxisDivisors(xMin, xMax, xStepMin, xSize, true);
		yCandidates = AxisDivisors(yMin, yMax, yStepMin, ySize, false);
		zCandidates = AxisDivisors(zMin, zMax, zStepMin, zSize, false);
	}

	unsigned int SnapToCandidate(unsigned int v, const std::vector<unsigned int> &vals) const
	{
		if (vals.empty())
			return v;

		return *std::min_element(vals.begin(), vals.end(),
			[v](unsigned int a, unsigned int b) {
				unsigned int da = (a > v) ? (a - v) : (v - a);
				unsigned int db = (b > v) ? (b - v) : (v - b);
				if (da != db) return da < db;
				return a > b;
			});
	}

	size_t CandidateIndex(const std::vector<unsigned int> &vals, unsigned int v) const
	{
		auto it = std::find(vals.begin(), vals.end(), v);
		if (it != vals.end())
			return (size_t) std::distance(vals.begin(), it);

		if (vals.empty())
			return 0;

		auto best = std::min_element(vals.begin(), vals.end(),
			[v](unsigned int a, unsigned int b) {
				unsigned int da = (a > v) ? (a - v) : (v - a);
				unsigned int db = (b > v) ? (b - v) : (v - b);
				if (da != db) return da < db;
				return a > b;
			});

		return (size_t) std::distance(vals.begin(), best);
	}

	std::vector<size_t> SampleIndexDim(size_t lo, size_t hi, unsigned int nPts) const
	{
		std::vector<size_t> vals;
		if (lo > hi) std::swap(lo, hi);
		if (nPts < 2) nPts = 2;

		size_t nFine = hi - lo + 1;
		if (nFine <= nPts) {
			for (size_t i = lo; i <= hi; i++)
				vals.push_back(i);
			return vals;
		}

		for (unsigned int i = 0; i < nPts; i++) {
			double alpha = (double) i / (double) (nPts - 1);
			size_t idx = lo + (size_t) std::llround(alpha * (double) (hi - lo));
			if (idx > hi) idx = hi;
			vals.push_back(idx);
		}

		vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
		return vals;
	}

	void PrintCandidateList(FILE *f, const char *name, const std::vector<unsigned int> &vals) const
	{
		if (f == nullptr) return;

		fprintf(f, "# %s_candidates count=%zu values=", name, vals.size());
		for (size_t i = 0; i < vals.size(); i++) {
			fprintf(f, "%s%u", (i == 0) ? "" : ",", vals[i]);
		}
		fprintf(f, "\n");
	}

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
		// Bounds are indices into the per-axis candidate arrays.
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

		auto xs = AxisDivisors(xMin, xMax, xStepMin, xSize, false);
		auto ys = AxisDivisors(yMin, yMax, yStepMin, ySize, false);
		auto zs = AxisDivisors(zMin, zMax, zStepMin, zSize, false);

		bool takeNext = false;
		for (auto bz : zs) {
			for (auto by : ys) {
				for (auto bx : xs) {
					if (!IsValidBlock(bx, by, bz))
						continue;

					if (takeNext) {
						xBlock = bx;
						yBlock = by;
						zBlock = bz;
						return;
					}

					if (bx == xBlock && by == yBlock && bz == zBlock)
						takeNext = true;
				}
			}
		}

		isTuned = true;
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

				xBlock = xMin;
				yBlock = yMin;
				zBlock = zMin;
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

				xBlock = (xMax >= 4) ? 4 : xMax;
				yBlock = 1;
				zBlock = 1;

				#ifdef USE_2DCYL
				ySize = 1;
				yMin  = 1;
				yMax  = 1;
				yStepMin = 1;
				yBlock = 1;
				#endif

			}
		}

		BuildCandidateLists();
		NormalizeInitialBlock();

		isTuned = false;
		tunePhase = 0;
	}

	SearchRegion InitialSearchRegion() const noexcept
	{
		SearchRegion R;
		R.xLo = 0;
		R.yLo = 0;
		R.zLo = 0;
		R.xHi = xCandidates.empty() ? 0 : (unsigned int) (xCandidates.size() - 1);
		R.yHi = yCandidates.empty() ? 0 : (unsigned int) (yCandidates.size() - 1);
		R.zHi = zCandidates.empty() ? 0 : (unsigned int) (zCandidates.size() - 1);
		R.dx = 1;
		R.dy = 1;
		R.dz = 1;
		R.level = 0;
		R.regionId = 0;
		return R;
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
		if (!FitsAxisGrid(bx, xMin, xMax, xStepMin)) return false;
		if (!FitsAxisGrid(by, yMin, yMax, yStepMin)) return false;
		if (!FitsAxisGrid(bz, zMin, zMax, zStepMin)) return false;

		if (!DividesAxis(xSize, bx)) return false;
		if (!DividesAxis(ySize, by)) return false;
		if (!DividesAxis(zSize, bz)) return false;

		if (isGpu && ((size_t) bx * (size_t) by * (size_t) bz > (size_t) maxThreadsPerBlock()))
			return false;

		return true;
	}

	bool IsValidRegion(const SearchRegion &R) const noexcept
	{
		return (!xCandidates.empty() && !yCandidates.empty() && !zCandidates.empty() &&
		        R.xLo <= R.xHi && R.yLo <= R.yHi && R.zLo <= R.zHi &&
		        R.xHi < xCandidates.size() &&
		        R.yHi < yCandidates.size() &&
		        R.zHi < zCandidates.size());
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

	std::vector<unsigned int> SampleDivisorDim(unsigned int lo, unsigned int hi,
	                                           unsigned int step, unsigned int extent,
	                                           unsigned int nPts, bool highToLow = false) const
	{
		auto vals = AxisDivisors(lo, hi, step, extent, highToLow);
		if (vals.empty())
			return vals;

		if (nPts < 2) nPts = 2;
		if (vals.size() <= nPts)
			return vals;

		std::vector<unsigned int> out;
		for (unsigned int i = 0; i < nPts; i++) {
			double alpha = (double) i / (double) (nPts - 1);
			size_t idx = (size_t) std::llround(alpha * (double) (vals.size() - 1));
			if (idx >= vals.size()) idx = vals.size() - 1;
			out.push_back(vals[idx]);
		}

		out.erase(std::unique(out.begin(), out.end()), out.end());
		return out;
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

		if (!IsValidRegion(R))
			return out;

		auto xs = SampleIndexDim(R.xLo, R.xHi, nPts);
		auto ys = SampleIndexDim(R.yLo, R.yHi, nPts);
		auto zs = SampleIndexDim(R.zLo, R.zHi, nPts);

		std::set<std::tuple<unsigned int,unsigned int,unsigned int>> seen;

		for (auto ix : xs) {
			for (auto iy : ys) {
				for (auto iz : zs) {
					unsigned int bx = xCandidates[ix];
					unsigned int by = yCandidates[iy];
					unsigned int bz = zCandidates[iz];

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
			sx += w * (double) CandidateIndex(xCandidates, c.bx);
			sy += w * (double) CandidateIndex(yCandidates, c.by);
			sz += w * (double) CandidateIndex(zCandidates, c.bz);
			sw += w;
		}

		BlockCandidate out;
		auto clampIdx = [](long long idx, unsigned int lo, unsigned int hi) -> unsigned int {
			if (idx < (long long) lo) return lo;
			if (idx > (long long) hi) return hi;
			return (unsigned int) idx;
		};

		unsigned int ix = clampIdx((long long) std::llround(sx / sw), R.xLo, R.xHi);
		unsigned int iy = clampIdx((long long) std::llround(sy / sw), R.yLo, R.yHi);
		unsigned int iz = clampIdx((long long) std::llround(sz / sw), R.zLo, R.zHi);

		out.bx = xCandidates[ix];
		out.by = yCandidates[iy];
		out.bz = zCandidates[iz];
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

	std::vector<BlockCandidate> NeighborCandidates(const BlockCandidate &center,
	                                               const SearchRegion &R) const
	{
		std::vector<BlockCandidate> out;
		if (!IsValidRegion(R))
			return out;

		unsigned int ix = (unsigned int) CandidateIndex(xCandidates, center.bx);
		unsigned int iy = (unsigned int) CandidateIndex(yCandidates, center.by);
		unsigned int iz = (unsigned int) CandidateIndex(zCandidates, center.bz);

		auto push = [&](long long x, long long y, long long z) {
			if (x < (long long) R.xLo || x > (long long) R.xHi) return;
			if (y < (long long) R.yLo || y > (long long) R.yHi) return;
			if (z < (long long) R.zLo || z > (long long) R.zHi) return;

			BlockCandidate c;
			c.bx = xCandidates[(size_t) x];
			c.by = yCandidates[(size_t) y];
			c.bz = zCandidates[(size_t) z];
			c.predicted = true;
			c.level = R.level;
			c.regionId = R.regionId;

			if (!IsValidBlock(c.bx, c.by, c.bz))
				return;

			out.push_back(c);
		};

		push((long long) ix + 1, iy, iz);
		push((long long) ix - 1, iy, iz);
		push(ix, (long long) iy + 1, iz);
		push(ix, (long long) iy - 1, iz);
		push(ix, iy, (long long) iz + 1);
		push(ix, iy, (long long) iz - 1);

		return out;
	}

	SearchRegion RefineAroundBest(const SearchRegion &R, const BlockCandidate &best, int newRegionId) const
	{
		if (!IsValidRegion(R))
			return R;

		auto shrink = [](unsigned int lo, unsigned int hi) -> unsigned int {
			if (hi <= lo) return 1;
			unsigned int width = hi - lo;
			unsigned int h = width / 4;
			if (h < 1) h = 1;
			return h;
		};

		unsigned int ix = (unsigned int) CandidateIndex(xCandidates, best.bx);
		unsigned int iy = (unsigned int) CandidateIndex(yCandidates, best.by);
		unsigned int iz = (unsigned int) CandidateIndex(zCandidates, best.bz);

		unsigned int hx = shrink(R.xLo, R.xHi);
		unsigned int hy = shrink(R.yLo, R.yHi);
		unsigned int hz = shrink(R.zLo, R.zHi);

		SearchRegion out;
		out.xLo = (ix > hx) ? ix - hx : R.xLo;
		out.xHi = std::min<unsigned int>(ix + hx, R.xHi);
		out.yLo = (iy > hy) ? iy - hy : R.yLo;
		out.yHi = std::min<unsigned int>(iy + hy, R.yHi);
		out.zLo = (iz > hz) ? iz - hz : R.zLo;
		out.zHi = std::min<unsigned int>(iz + hz, R.zHi);

		out.dx = 1;
		out.dy = 1;
		out.dz = 1;
		out.level = R.level + 1;
		out.regionId = newRegionId;
		return out;
	}

	bool RegionAtMinResolution(const SearchRegion &R) const noexcept
	{
		return ((R.xHi <= R.xLo + 1) &&
		        (R.yHi <= R.yLo + 1) &&
		        (R.zHi <= R.zLo + 1));
	}

	void AppendTuneLogHeader(FILE *f, const char *fieldName, const char *devName,
	                         unsigned int Nx, unsigned int Ny, unsigned int Nz,
	                         unsigned int propType, unsigned int precision,
	                         unsigned int lowMem, unsigned int lowMemGpu,
	                         int mpiRanks, int ompThreads, size_t nGhost) const
	{
		if (f == nullptr) return;

		fprintf(f, "# adaptive tune\n");
		fprintf(f, "# field %s\n", fieldName);
		fprintf(f, "# device %s\n", devName);
		fprintf(f, "# prop_type %u\n", propType);
		fprintf(f, "# precision %u\n", precision);
		fprintf(f, "# lowmem %u\n", lowMem);
		fprintf(f, "# lowmemgpu %u\n", lowMemGpu);
		fprintf(f, "# mpi_ranks %d\n", mpiRanks);
		fprintf(f, "# omp_threads %d\n", ompThreads);
		fprintf(f, "# Ng %zu\n", nGhost);
		fprintf(f, "# lattice %u %u %u\n", Nx, Ny, Nz);
		fprintf(f, "# tuner_lattice %u %u %u\n", xSize, ySize, zSize);
		fprintf(f, "# xrange [%u,%u] step=%u\n", xMin, xMax, xStepMin);
		fprintf(f, "# yrange [%u,%u] step=%u\n", yMin, yMax, yStepMin);
		fprintf(f, "# zrange [%u,%u] step=%u\n", zMin, zMax, zStepMin);
		fprintf(f, "# block_order x=descending y=ascending z=ascending\n");
		PrintCandidateList(f, "x", xCandidates);
		PrintCandidateList(f, "y", yCandidates);
		PrintCandidateList(f, "z", zCandidates);
		if (isGpu)
			fprintf(f, "# gpu_max_threads_per_block %zu\n", (size_t) maxThreadsPerBlock());
		fprintf(f, "# stop_rel_improve %.8f\n", adaptiveStopRelImprove);
		fprintf(f, "# coarse_points %u\n", adaptiveCoarsePoints);
		fprintf(f, "# max_evals %u\n", adaptiveMaxEvals);
		fprintf(f, "# columns: eval level region source ix iy iz bx by bz threads time_ns valid predicted\n");
		fflush(f);
	}

	void AppendTuneConclusion(FILE *f, const BlockCandidate &best, unsigned int nEvals,
	                          const char *stopReason, bool cacheWritten,
	                          const char *cachePath) const
	{
		if (f == nullptr) return;

		size_t ix = CandidateIndex(xCandidates, best.bx);
		size_t iy = CandidateIndex(yCandidates, best.by);
		size_t iz = CandidateIndex(zCandidates, best.bz);
		size_t nThreads = (size_t) best.bx * (size_t) best.by * (size_t) best.bz;

		fprintf(f, "# conclusion\n");
		fprintf(f, "# conclusion_stop %s\n", stopReason ? stopReason : "unknown");
		fprintf(f, "# conclusion_evals %u\n", nEvals);
		fprintf(f, "# conclusion_best_index %zu %zu %zu\n", ix, iy, iz);
		fprintf(f, "# conclusion_best_block %u %u %u threads=%zu\n",
		        best.bx, best.by, best.bz, nThreads);
		fprintf(f, "# conclusion_best_time_ns %zu\n", best.timeNs);
		fprintf(f, "# conclusion_cache_written %d path=%s\n",
		        cacheWritten ? 1 : 0, cachePath ? cachePath : "");
		fprintf(f, "# end adaptive tune\n");
		fflush(f);
	}

	void AppendEvalStart(FILE *f, size_t evalCounter, const char *source,
	                     const BlockCandidate &c) const
	{
		if (f == nullptr) return;

		size_t ix = CandidateIndex(xCandidates, c.bx);
		size_t iy = CandidateIndex(yCandidates, c.by);
		size_t iz = CandidateIndex(zCandidates, c.bz);
		size_t nThreads = (size_t) c.bx * (size_t) c.by * (size_t) c.bz;

		fprintf(f,
		        "# eval_start %zu level=%d region=%d source=%s "
		        "idx=%zu %zu %zu block=%u %u %u threads=%zu\n",
		        evalCounter, c.level, c.regionId, source ? source : "unknown",
		        ix, iy, iz, c.bx, c.by, c.bz, nThreads);
		fflush(f);
	}

	void AppendEvalResult(FILE *f, size_t evalCounter, const char *source,
	                      const BlockCandidate &c) const
	{
		if (f == nullptr) return;

		size_t ix = CandidateIndex(xCandidates, c.bx);
		size_t iy = CandidateIndex(yCandidates, c.by);
		size_t iz = CandidateIndex(zCandidates, c.bz);
		size_t nThreads = (size_t) c.bx * (size_t) c.by * (size_t) c.bz;

		fprintf(f, "%zu %d %d %s %zu %zu %zu %u %u %u %zu %zu %d %d\n",
		        evalCounter, c.level, c.regionId, source ? source : "unknown",
		        ix, iy, iz,
		        c.bx, c.by, c.bz, nThreads,
		        c.timeNs, c.valid ? 1 : 0, c.predicted ? 1 : 0);
		fflush(f);
	}

	void AppendBatchLog(FILE *f, const char *source, const std::vector<BlockCandidate> &cand,
	                    const SearchStats &S, const SearchRegion &R, bool writeRows = true) const
	{
		if (f == nullptr) return;

		fprintf(f,
		        "# batch level=%d region=%d source=%s "
		        "idx_x=[%u,%u] idx_y=[%u,%u] idx_z=[%u,%u] "
		        "n_valid=%zu n_invalid=%zu min=%.3f max=%.3f mean=%.3f median=%.3f std=%.3f best=%u %u %u\n",
		        cand.empty() ? -1 : cand.front().level,
		        cand.empty() ? -1 : cand.front().regionId,
		        source,
		        R.xLo, R.xHi, R.yLo, R.yHi, R.zLo, R.zHi,
		        S.nValid, S.nInvalid,
		        S.minTime, S.maxTime, S.meanTime, S.medianTime, S.stdTime,
		        S.best.bx, S.best.by, S.best.bz);

		if (!writeRows) {
			fflush(f);
			return;
		}

		static size_t evalCounter = 0;
		for (const auto &c : cand) {
			size_t ix = CandidateIndex(xCandidates, c.bx);
			size_t iy = CandidateIndex(yCandidates, c.by);
			size_t iz = CandidateIndex(zCandidates, c.bz);
			size_t nThreads = (size_t) c.bx * (size_t) c.by * (size_t) c.bz;

			fprintf(f, "%zu %d %d %s %zu %zu %zu %u %u %u %zu %zu %d %d\n",
			        evalCounter++, c.level, c.regionId, source,
			        ix, iy, iz,
			        c.bx, c.by, c.bz, nThreads,
			        c.timeNs, c.valid ? 1 : 0, c.predicted ? 1 : 0);
		}

		fflush(f);
	}
};

#endif
