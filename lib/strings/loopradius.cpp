#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <mpi.h>
#include "strings/loopradius.h"
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "utils/logger.h"

namespace {
	// Constants for the weighted masking method
	const double SIGMA = 30.0;        // Weighting parameter: w = exp(-phi²*σ)
	const double D_ALPHA = M_PI / 8.0; // Angular width for masking

	/**
	 * @brief Compute center of mass of the loop from rephi data
	 */
	template<typename Float>
	void computeCenterOfMass(const Float *rephi, size_t Lx, size_t Ly, double &x0, double &y0) {
		double sum_w = 0.0;
		double sum_wx = 0.0;
		double sum_wy = 0.0;
		
		for (size_t j = 0; j < Ly; j++) {
			for (size_t i = 0; i < Lx; i++) {
				Float phi = rephi[j * Lx + i];
				double w = std::exp(-phi * phi * SIGMA);
				sum_w += w;
				sum_wx += w * i;
				sum_wy += w * j;
			}
		}
		
		x0 = sum_wx / sum_w;
		y0 = sum_wy / sum_w;
	}

	/**
	 * @brief Compute weighted average radius for given angles
	 * 
	 * @param rephi Field data
	 * @param Lx Grid size x
	 * @param Ly Grid size y
	 * @param x0 Center of mass x
	 * @param y0 Center of mass y
	 * @param angles List of angles to average over
	 * @param n_angles Number of angles
	 * @return Average radius in grid units
	 */
	template<typename Float>
	double computeWeightedRadius(const Float *rephi, size_t Lx, size_t Ly, 
	                             double x0, double y0, const double *angles, int n_angles) {
		double R_sum = 0.0;
		
		for (int ia = 0; ia < n_angles; ia++) {
			double angle = angles[ia];
			double sum_wr = 0.0;
			double sum_w = 0.0;
			
			// Process all pixels
			for (size_t j = 0; j < Ly; j++) {
				for (size_t i = 0; i < Lx; i++) {
					double dx = i - x0;
					double dy = j - y0;
					double r = std::sqrt(dx * dx + dy * dy);
					
					if (r < 1e-10) continue; // Skip center
					
					double phi_pixel = std::atan2(dy, dx);
					if (phi_pixel < 0) phi_pixel += 2.0 * M_PI;
					
					// Angular distance to target angle
					double dangle = std::abs(phi_pixel - angle);
					if (dangle > M_PI) dangle = 2.0 * M_PI - dangle;
					
					// Check if within angular window
					if (dangle <= D_ALPHA) {
						Float phi = rephi[j * Lx + i];
						double w = std::exp(-phi * phi * SIGMA);
						sum_wr += w * r;
						sum_w += w;
					}
				}
			}
			
			if (sum_w > 0) {
				R_sum += sum_wr / sum_w;
			}
		}
		
		return R_sum / n_angles;
	}

	/**
	 * @brief Find zero-crossings along a horizontal line with interpolation
	 */
	template<typename Float>
	double findHorizontalZeroCrossings(const Float *rephi, size_t Lx, size_t Ly, 
	                                   double x0, double y0) {
		int y_idx = static_cast<int>(std::round(y0));
		if (y_idx < 0 || y_idx >= static_cast<int>(Ly)) {
			return 0.0; // Invalid
		}
		
		std::vector<double> zero_positions;
		
		// Scan from center right
		for (size_t i = static_cast<size_t>(x0); i < Lx - 1; i++) {
			Float v1 = rephi[y_idx * Lx + i];
			Float v2 = rephi[y_idx * Lx + (i + 1)];
			
			// Check for sign change
			if (v1 * v2 < 0) {
				// Linear interpolation to find exact zero position
				double frac = -v1 / (v2 - v1);
				double x_zero = i + frac;
				double r = x_zero - x0;
				if (r > 0) zero_positions.push_back(r);
			}
		}
		
		// Scan from center left
		for (int i = static_cast<int>(x0); i > 0; i--) {
			Float v1 = rephi[y_idx * Lx + i];
			Float v2 = rephi[y_idx * Lx + (i - 1)];
			
			// Check for sign change
			if (v1 * v2 < 0) {
				// Linear interpolation to find exact zero position
				double frac = -v1 / (v2 - v1);
				double x_zero = i - frac;
				double r = std::abs(x_zero - x0);
				if (r > 0) zero_positions.push_back(r);
			}
		}
		
		// Return average of zero positions (typically should be ~2 for a loop)
		if (zero_positions.empty()) return 0.0;
		
		double sum = 0.0;
		for (double r : zero_positions) sum += r;
		return sum / zero_positions.size();
	}

	/**
	 * @brief Bilinear interpolation for sub-pixel sampling
	 */
	template<typename Float>
	Float bilinearInterp(const Float *rephi, size_t Lx, size_t Ly, double x, double y) {
		int ix = static_cast<int>(std::floor(x));
		int iy = static_cast<int>(std::floor(y));
		
		// Boundary checks
		if (ix < 0 || ix >= static_cast<int>(Lx) - 1 ||
		    iy < 0 || iy >= static_cast<int>(Ly) - 1) {
			return 0.0;
		}
		
		double fx = x - ix;
		double fy = y - iy;
		
		Float v00 = rephi[iy * Lx + ix];
		Float v10 = rephi[iy * Lx + (ix + 1)];
		Float v01 = rephi[(iy + 1) * Lx + ix];
		Float v11 = rephi[(iy + 1) * Lx + (ix + 1)];
		
		Float v0 = v00 * (1 - fx) + v10 * fx;
		Float v1 = v01 * (1 - fx) + v11 * fx;
		
		return v0 * (1 - fy) + v1 * fy;
	}

	/**
	 * @brief Find zero-crossings along a diagonal line with interpolation
	 */
	template<typename Float>
	double findDiagonalZeroCrossings(const Float *rephi, size_t Lx, size_t Ly, 
	                                 double x0, double y0) {
		std::vector<double> zero_positions;
		
		// Diagonal at angle π/4: dx = dy
		double cos45 = std::cos(M_PI / 4.0);
		double sin45 = std::sin(M_PI / 4.0);
		
		// Sample along the diagonal with fine resolution
		double max_r = std::min(Lx, Ly) / 2.0;
		double dr = 0.1; // Sub-pixel sampling
		
		// Scan from center outward in positive direction
		for (double r = 0; r < max_r; r += dr) {
			double x = x0 + r * cos45;
			double y = y0 + r * sin45;
			
			if (x < 0 || x >= Lx - 1 || y < 0 || y >= Ly - 1) break;
			
			Float v1 = bilinearInterp(rephi, Lx, Ly, x, y);
			
			double x_next = x0 + (r + dr) * cos45;
			double y_next = y0 + (r + dr) * sin45;
			
			if (x_next < 0 || x_next >= Lx - 1 || y_next < 0 || y_next >= Ly - 1) break;
			
			Float v2 = bilinearInterp(rephi, Lx, Ly, x_next, y_next);
			
			// Check for sign change
			if (v1 * v2 < 0 && r > 0.5) { // Avoid very small r
				// Linear interpolation for zero position
				double frac = -v1 / (v2 - v1);
				double r_zero = r + frac * dr;
				zero_positions.push_back(r_zero);
			}
		}
		
		// Scan from center outward in negative direction
		for (double r = 0; r < max_r; r += dr) {
			double x = x0 - r * cos45;
			double y = y0 - r * sin45;
			
			if (x < 0 || x >= Lx - 1 || y < 0 || y >= Ly - 1) break;
			
			Float v1 = bilinearInterp(rephi, Lx, Ly, x, y);
			
			double x_next = x0 - (r + dr) * cos45;
			double y_next = y0 - (r + dr) * sin45;
			
			if (x_next < 0 || x_next >= Lx - 1 || y_next < 0 || y_next >= Ly - 1) break;
			
			Float v2 = bilinearInterp(rephi, Lx, Ly, x_next, y_next);
			
			// Check for sign change
			if (v1 * v2 < 0 && r > 0.5) { // Avoid very small r
				// Linear interpolation for zero position
				double frac = -v1 / (v2 - v1);
				double r_zero = r + frac * dr;
				zero_positions.push_back(r_zero);
			}
		}
		
		// Return average of zero positions
		if (zero_positions.empty()) return 0.0;
		
		double sum = 0.0;
		for (double r : zero_positions) sum += r;
		return sum / zero_positions.size();
	}
}

// Template implementations for float
double loopRadiusAxes(const float *rephi, size_t Lx, size_t Ly, double delta) {
	double x0, y0;
	computeCenterOfMass(rephi, Lx, Ly, x0, y0);
	
	// Cardinal axes angles: 0, π/2, π, 3π/2
	const double angles[4] = {0.0, M_PI/2.0, M_PI, 3.0*M_PI/2.0};
	
	double R = computeWeightedRadius(rephi, Lx, Ly, x0, y0, angles, 4);
	return R * delta;
}

double loopRadiusDiag(const float *rephi, size_t Lx, size_t Ly, double delta) {
	double x0, y0;
	computeCenterOfMass(rephi, Lx, Ly, x0, y0);
	
	// Diagonal angles: π/4, 3π/4, 5π/4, 7π/4
	const double angles[4] = {M_PI/4.0, 3.0*M_PI/4.0, 5.0*M_PI/4.0, 7.0*M_PI/4.0};
	
	double R = computeWeightedRadius(rephi, Lx, Ly, x0, y0, angles, 4);
	return R * delta;
}

double loopRadiusAxesInterp(const float *rephi, size_t Lx, size_t Ly, double delta) {
	double x0, y0;
	computeCenterOfMass(rephi, Lx, Ly, x0, y0);
	
	double R = findHorizontalZeroCrossings(rephi, Lx, Ly, x0, y0);
	return R * delta;
}

double loopRadiusDiagInterp(const float *rephi, size_t Lx, size_t Ly, double delta) {
	double x0, y0;
	computeCenterOfMass(rephi, Lx, Ly, x0, y0);
	
	double R = findDiagonalZeroCrossings(rephi, Lx, Ly, x0, y0);
	return R * delta;
}

// Template implementations for double
double loopRadiusAxes(const double *rephi, size_t Lx, size_t Ly, double delta) {
	double x0, y0;
	computeCenterOfMass(rephi, Lx, Ly, x0, y0);
	
	const double angles[4] = {0.0, M_PI/2.0, M_PI, 3.0*M_PI/2.0};
	
	double R = computeWeightedRadius(rephi, Lx, Ly, x0, y0, angles, 4);
	return R * delta;
}

double loopRadiusDiag(const double *rephi, size_t Lx, size_t Ly, double delta) {
	double x0, y0;
	computeCenterOfMass(rephi, Lx, Ly, x0, y0);
	
	const double angles[4] = {M_PI/4.0, 3.0*M_PI/4.0, 5.0*M_PI/4.0, 7.0*M_PI/4.0};
	
	double R = computeWeightedRadius(rephi, Lx, Ly, x0, y0, angles, 4);
	return R * delta;
}

double loopRadiusAxesInterp(const double *rephi, size_t Lx, size_t Ly, double delta) {
	double x0, y0;
	computeCenterOfMass(rephi, Lx, Ly, x0, y0);
	
	double R = findHorizontalZeroCrossings(rephi, Lx, Ly, x0, y0);
	return R * delta;
}

double loopRadiusDiagInterp(const double *rephi, size_t Lx, size_t Ly, double delta) {
	double x0, y0;
	computeCenterOfMass(rephi, Lx, Ly, x0, y0);
	
	double R = findDiagonalZeroCrossings(rephi, Lx, Ly, x0, y0);
	return R * delta;
}

// Main entry point that extracts rephi from Scalar field
LoopRadiusData computeLoopRadius(Scalar *axion, int slice) {
	LoopRadiusData result = {0.0, 0.0, 0.0, 0.0};
	
	if (axion->Field() != FIELD_SAXION) {
		LogMsg(VERB_HIGH, "[LoopRadius] Only SAXION fields supported");
		return result;
	}
	
	size_t Lx = axion->Length();
	size_t Ly = axion->Length(); // Assuming square grid
	size_t Lz = axion->Depth();   // Local depth (per rank)
	double delta = axion->Delta();
	
	// Determine if this slice is on this rank
	int myRank = commRank();
	int slicesPerRank = Lz;
	int prank = slice / slicesPerRank;  // Which rank owns this slice
	int localSlice = slice % slicesPerRank;  // Local index on that rank
	
	// Only compute if this rank owns the slice (others keep result={0,0,0,0})
	if (myRank == prank) {
		// Extract 2D slice real part from 3D complex field
		// Field layout: [z][y][x][complex] where complex = {real, imag}
		size_t sliceSize = Lx * Ly;
		
		if (axion->Precision() == FIELD_DOUBLE) {
			std::vector<double> rephi(sliceSize);
			
			// Access field data directly
			double *field = static_cast<double*>(axion->mCpu());
			size_t sliceOffset = localSlice * sliceSize * 2; // *2 for complex
			
			// Extract real part (every other value, starting at offset 0)
			for (size_t i = 0; i < sliceSize; i++) {
				rephi[i] = field[sliceOffset + 2*i];  // Real part at even indices
			}
			
			result.R_axes = loopRadiusAxes(rephi.data(), Lx, Ly, delta);
			result.R_diag = loopRadiusDiag(rephi.data(), Lx, Ly, delta);
			result.R_axes_interp = loopRadiusAxesInterp(rephi.data(), Lx, Ly, delta);
			result.R_diag_interp = loopRadiusDiagInterp(rephi.data(), Lx, Ly, delta);
		} else {
			std::vector<float> rephi(sliceSize);
			
			// Access field data directly
			float *field = static_cast<float*>(axion->mCpu());
			size_t sliceOffset = localSlice * sliceSize * 2; // *2 for complex
			
			// Extract real part (every other value, starting at offset 0)
			for (size_t i = 0; i < sliceSize; i++) {
				rephi[i] = field[sliceOffset + 2*i];  // Real part at even indices
			}
			
			result.R_axes = loopRadiusAxes(rephi.data(), Lx, Ly, delta);
			result.R_diag = loopRadiusDiag(rephi.data(), Lx, Ly, delta);
			result.R_axes_interp = loopRadiusAxesInterp(rephi.data(), Lx, Ly, delta);
			result.R_diag_interp = loopRadiusDiagInterp(rephi.data(), Lx, Ly, delta);
		}
	} else {
		LogMsg(VERB_HIGH, "[LoopRadius] Slice %d not on rank %d (owned by rank %d)", slice, myRank, prank);
	}
	
	LogMsg(VERB_HIGH, "[LoopRadius] Local: R_axes=%.3e R_diag=%.3e R_axes_interp=%.3e R_diag_interp=%.3e", 
	       result.R_axes, result.R_diag, result.R_axes_interp, result.R_diag_interp);
	
	// MPI reduction: only one rank has the slice data, others have zeros
	// Using SUM will effectively gather the result from the owning rank
	LoopRadiusData result_global = {0.0, 0.0, 0.0, 0.0};
	
	MPI_Allreduce(&result.R_axes, &result_global.R_axes, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&result.R_diag, &result_global.R_diag, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&result.R_axes_interp, &result_global.R_axes_interp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&result.R_diag_interp, &result_global.R_diag_interp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
	LogMsg(VERB_HIGH, "[LoopRadius] Global: R_axes=%.3e R_diag=%.3e R_axes_interp=%.3e R_diag_interp=%.3e", 
	       result_global.R_axes, result_global.R_diag, result_global.R_axes_interp, result_global.R_diag_interp);
	
	return result_global;
}
