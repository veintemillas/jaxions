#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <limits>
#include <mpi.h>
#include "strings/loopradius.h"
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "utils/logger.h"

namespace {

	// Constants for the weighted masking method (must match Python post-processing)
	const double SIGMA   = 30.0;        // w = exp(-phi^2 * sigma)
	const double D_ALPHA = M_PI / 8.0;  // angular half-width of each wedge

	/**
	 * @brief Center of mass with Gaussian weight w = exp(-phi^2 * sigma).
	 */
	template<typename Float>
	void computeCenterOfMass(const Float *rephi, size_t Lx, size_t Ly,
	                         double &x0, double &y0) {
		double sum_w  = 0.0;
		double sum_wx = 0.0;
		double sum_wy = 0.0;

		for (size_t j = 0; j < Ly; j++) {
			for (size_t i = 0; i < Lx; i++) {
				Float phi = rephi[j * Lx + i];
				double w  = std::exp(-phi * phi * SIGMA);
				sum_w  += w;
				sum_wx += w * i;
				sum_wy += w * j;
			}
		}

		x0 = sum_wx / sum_w;
		y0 = sum_wy / sum_w;
	}

	/**
	 * @brief Weighted radius using a single weighted mean over the *union*
	 *        of angular wedges around the supplied angles.
	 *
	 * Matches the Python convention:
	 *     mask = OR_k {|angle_pixel - angle_k| < dα}
	 *     r    = sum(w * R) / sum(w)   over the union mask
	 *
	 * (NOT the unweighted mean of per-direction means used previously.)
	 */
	template<typename Float>
	double computeWeightedRadius(const Float *rephi, size_t Lx, size_t Ly,
	                             double x0, double y0,
	                             const double *angles, int n_angles) {
		double sum_wr = 0.0;
		double sum_w  = 0.0;

		for (size_t j = 0; j < Ly; j++) {
			for (size_t i = 0; i < Lx; i++) {
				double dx = i - x0;
				double dy = j - y0;
				double r  = std::sqrt(dx * dx + dy * dy);
				if (r < 1e-10) continue;

				double phi_pixel = std::atan2(dy, dx);
				if (phi_pixel < 0) phi_pixel += 2.0 * M_PI;

				bool in_mask = false;
				for (int ia = 0; ia < n_angles; ia++) {
					double dangle = std::abs(phi_pixel - angles[ia]);
					if (dangle > M_PI) dangle = 2.0 * M_PI - dangle;
					if (dangle < D_ALPHA) { in_mask = true; break; }
				}
				if (!in_mask) continue;

				Float  phi = rephi[j * Lx + i];
				double w   = std::exp(-phi * phi * SIGMA);
				sum_wr += w * r;
				sum_w  += w;
			}
		}

		if (sum_w <= 0.0) return std::numeric_limits<double>::quiet_NaN();
		return sum_wr / sum_w;
	}

	/**
	 * @brief Half-width along the horizontal cut through (x0, y0).
	 *
	 * Scans the full row j = round(y0) for all sign changes of rephi (with
	 * linear interpolation), then returns 0.5 * (x_rightmost - x_leftmost).
	 * Returns NaN if fewer than two zeros were found.
	 */
	template<typename Float>
	double findHorizontalHalfWidth(const Float *rephi, size_t Lx, size_t Ly,
	                               double x0, double y0) {
		int y_idx = static_cast<int>(std::round(y0));
		if (y_idx < 0 || y_idx >= static_cast<int>(Ly))
			return std::numeric_limits<double>::quiet_NaN();

		std::vector<double> zeros;

		for (size_t i = 0; i + 1 < Lx; i++) {
			Float v1 = rephi[y_idx * Lx + i];
			Float v2 = rephi[y_idx * Lx + (i + 1)];
			if (v1 == static_cast<Float>(0)) {
				zeros.push_back(static_cast<double>(i));
			} else if (v1 * v2 < 0) {
				double frac = -static_cast<double>(v1) /
				              (static_cast<double>(v2) - static_cast<double>(v1));
				zeros.push_back(static_cast<double>(i) + frac);
			}
		}
		if (rephi[y_idx * Lx + (Lx - 1)] == static_cast<Float>(0))
			zeros.push_back(static_cast<double>(Lx - 1));

		if (zeros.size() < 2)
			return std::numeric_limits<double>::quiet_NaN();
		return 0.5 * (zeros.back() - zeros.front());
	}

	/**
	 * @brief Bilinear interpolation; returns NaN if (x, y) is outside [0, L-1].
	 */
	template<typename Float>
	double bilinearInterp(const Float *rephi, size_t Lx, size_t Ly,
	                      double x, double y) {
		if (x < 0.0 || x > static_cast<double>(Lx) - 1.0 ||
		    y < 0.0 || y > static_cast<double>(Ly) - 1.0)
			return std::numeric_limits<double>::quiet_NaN();

		int i0 = static_cast<int>(std::floor(x));
		int j0 = static_cast<int>(std::floor(y));
		int i1 = std::min(i0 + 1, static_cast<int>(Lx) - 1);
		int j1 = std::min(j0 + 1, static_cast<int>(Ly) - 1);
		double tx = x - i0;
		double ty = y - j0;

		double v00 = rephi[j0 * Lx + i0];
		double v10 = rephi[j0 * Lx + i1];
		double v01 = rephi[j1 * Lx + i0];
		double v11 = rephi[j1 * Lx + i1];

		return (1.0 - tx) * (1.0 - ty) * v00
		     +        tx  * (1.0 - ty) * v10
		     + (1.0 - tx) *        ty  * v01
		     +        tx  *        ty  * v11;
	}

	/**
	 * @brief Half-width along the π/4 diagonal through (x0, y0).
	 *
	 * Parameterise the line as (x, y) = (x0, y0) + (s/√2)(1, 1). Sample with
	 * Δs = 1, using bilinear interpolation, over the full extent that stays
	 * inside the box. Find all zero crossings (linear interp in s), then
	 * return 0.5 * (s_rightmost - s_leftmost). NaN if fewer than two zeros.
	 */
	template<typename Float>
	double findDiagonalHalfWidth(const Float *rephi, size_t Lx, size_t Ly,
	                             double x0, double y0) {
		const double sqrt2     = std::sqrt(2.0);
		const double inv_sqrt2 = 1.0 / sqrt2;

		double smin = std::max(-x0, -y0) * sqrt2;
		double smax = std::min(static_cast<double>(Lx) - 1.0 - x0,
		                       static_cast<double>(Ly) - 1.0 - y0) * sqrt2;

		std::vector<double> s_vals;
		std::vector<double> f_vals;
		s_vals.reserve(static_cast<size_t>(std::max(0.0, smax - smin)) + 1);
		f_vals.reserve(s_vals.capacity());

		for (double s = smin; s < smax; s += 1.0) {
			double x = x0 + s * inv_sqrt2;
			double y = y0 + s * inv_sqrt2;
			double f = bilinearInterp(rephi, Lx, Ly, x, y);
			if (std::isfinite(f)) {
				s_vals.push_back(s);
				f_vals.push_back(f);
			}
		}

		std::vector<double> zeros;
		for (size_t i = 0; i + 1 < f_vals.size(); i++) {
			double f1 = f_vals[i];
			double f2 = f_vals[i + 1];
			if (f1 == 0.0) {
				zeros.push_back(s_vals[i]);
			} else if (f1 * f2 < 0.0) {
				double sz = s_vals[i] -
				            f1 * (s_vals[i + 1] - s_vals[i]) / (f2 - f1);
				zeros.push_back(sz);
			}
		}

		if (zeros.size() < 2)
			return std::numeric_limits<double>::quiet_NaN();
		return 0.5 * (zeros.back() - zeros.front());
	}

}  // anonymous namespace

// -----------------------------------------------------------------------------
// Public API: unchanged signatures.
// -----------------------------------------------------------------------------

double loopRadiusAxes(const float *rephi, lrpar p) {
	const double angles[4] = {0.0, M_PI/2.0, M_PI, 3.0*M_PI/2.0};
	double R = computeWeightedRadius(rephi, p.Lx, p.Ly, p.x0, p.y0, angles, 4);
	return R * p.delta;
}

double loopRadiusDiag(const float *rephi, lrpar p) {
	const double angles[4] = {M_PI/4.0, 3.0*M_PI/4.0, 5.0*M_PI/4.0, 7.0*M_PI/4.0};
	double R = computeWeightedRadius(rephi, p.Lx, p.Ly, p.x0, p.y0, angles, 4);
	return R * p.delta;
}

double loopRadiusAxesInterp(const float *rephi, lrpar p) {
	double R = findHorizontalHalfWidth(rephi, p.Lx, p.Ly, p.x0, p.y0);
	return R * p.delta;
}

double loopRadiusDiagInterp(const float *rephi, lrpar p) {
	double R = findDiagonalHalfWidth(rephi, p.Lx, p.Ly, p.x0, p.y0);
	return R * p.delta;
}

double loopRadiusAxes(const double *rephi, lrpar p) {
	const double angles[4] = {0.0, M_PI/2.0, M_PI, 3.0*M_PI/2.0};
	double R = computeWeightedRadius(rephi, p.Lx, p.Ly, p.x0, p.y0, angles, 4);
	return R * p.delta;
}

double loopRadiusDiag(const double *rephi, lrpar p) {
	const double angles[4] = {M_PI/4.0, 3.0*M_PI/4.0, 5.0*M_PI/4.0, 7.0*M_PI/4.0};
	double R = computeWeightedRadius(rephi, p.Lx, p.Ly, p.x0, p.y0, angles, 4);
	return R * p.delta;
}

double loopRadiusAxesInterp(const double *rephi, lrpar p) {
	double R = findHorizontalHalfWidth(rephi, p.Lx, p.Ly, p.x0, p.y0);
	return R * p.delta;
}

double loopRadiusDiagInterp(const double *rephi, lrpar p) {
	double R = findDiagonalHalfWidth(rephi, p.Lx, p.Ly, p.x0, p.y0);
	return R * p.delta;
}

// -----------------------------------------------------------------------------
// Driver: extracts the real part of the slice and dispatches all 4 measures.
// -----------------------------------------------------------------------------

LoopRadiusData computeLoopRadius(Scalar *axion, int slice) {
	LoopRadiusData result = {0.0, 0.0, 0.0, 0.0};

	LogMsg(VERB_NORMAL, "[LoopRadius] Compute loop radius (field %s)",
	       axion->Folded() ? "Folded" : "Unfolded");

	if (axion->Field() != FIELD_SAXION) {
		LogMsg(VERB_NORMAL, "[LoopRadius] Only SAXION fields supported");
		return result;
	}

	size_t Lx = axion->Length();
	size_t Ly = axion->Length();   // square slice
	size_t Lz = axion->Depth();    // local depth on this rank
	double delta = axion->Delta();

	int myRank        = commRank();
	int slicesPerRank = Lz;
	int prank         = slice / slicesPerRank;
	int localSlice    = slice % slicesPerRank;

	if (myRank == prank) {
		Folder munge(axion);
		LogMsg(VERB_NORMAL, "[LoopRadius] If configuration folded, unfold 2D slice");
		munge(UNFOLD_SLICE, localSlice);

		// The unfolded slice is now in mFrontGhost as interleaved complex:
		//   [re(0), im(0), re(1), im(1), ...]
		// Compact the real parts in place into positions [0, sliceSize):
		//   re_phi[k] = re_phi[2k] for k = 1, ..., sliceSize-1.
		// (Position 0 already holds re(0); reads at 2k > k are always
		// from un-yet-written cells.)
		size_t sliceSize = Lx * Ly;
		double x0 = -1.0, y0 = -1.0;

		if (axion->Precision() == FIELD_DOUBLE) {
			double *re_phi = static_cast<double*>(axion->mFrontGhost());
			for (size_t co = 1; co < sliceSize; co++)
				re_phi[co] = re_phi[co * 2];

			computeCenterOfMass(re_phi, Lx, Ly, x0, y0);
			lrpar para;
			para.Lx = Lx; para.Ly = Ly;
			para.x0 = x0; para.y0 = y0;
			para.delta = delta;

			result.R_axes        = loopRadiusAxes      (re_phi, para);
			result.R_diag        = loopRadiusDiag      (re_phi, para);
			result.R_axes_interp = loopRadiusAxesInterp(re_phi, para);
			result.R_diag_interp = loopRadiusDiagInterp(re_phi, para);
		} else {
			float *re_phi = static_cast<float*>(axion->mFrontGhost());
			for (size_t co = 1; co < sliceSize; co++)
				re_phi[co] = re_phi[co * 2];

			computeCenterOfMass(re_phi, Lx, Ly, x0, y0);
			lrpar para;
			para.Lx = Lx; para.Ly = Ly;
			para.x0 = x0; para.y0 = y0;
			para.delta = delta;

			result.R_axes        = loopRadiusAxes      (re_phi, para);
			result.R_diag        = loopRadiusDiag      (re_phi, para);
			result.R_axes_interp = loopRadiusAxesInterp(re_phi, para);
			result.R_diag_interp = loopRadiusDiagInterp(re_phi, para);
		}
	} else {
		LogMsg(VERB_HIGH, "[LoopRadius] Slice %d not on rank %d (owned by rank %d)",
		       slice, myRank, prank);
	}

	LogMsg(VERB_HIGH, "[LoopRadius] Local: R_axes=%.3e R_diag=%.3e R_axes_interp=%.3e R_diag_interp=%.3e",
	       result.R_axes, result.R_diag, result.R_axes_interp, result.R_diag_interp);

	// Only the owning rank holds non-zero values; SUM gathers them globally.
	// NaN from the owning rank propagates correctly (0 + NaN = NaN).
	LoopRadiusData result_global = {0.0, 0.0, 0.0, 0.0};
	MPI_Allreduce(&result.R_axes,        &result_global.R_axes,        1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&result.R_diag,        &result_global.R_diag,        1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&result.R_axes_interp, &result_global.R_axes_interp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&result.R_diag_interp, &result_global.R_diag_interp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	LogMsg(VERB_HIGH, "[LoopRadius] Global: R_axes=%.3e R_diag=%.3e R_axes_interp=%.3e R_diag_interp=%.3e",
	       result_global.R_axes, result_global.R_diag, result_global.R_axes_interp, result_global.R_diag_interp);

	return result_global;
}
