#ifndef	_LOOPRADIUS_
#define	_LOOPRADIUS_

#include "scalar/scalarField.h"

/**
 * @brief Structure to hold loop radius measurement results
 */
struct LoopRadiusData {
	double R_axes;         // Loop radius along cardinal axes (0, π/2, π, 3π/2)
	double R_diag;         // Loop radius along diagonals (π/4, 3π/4, 5π/4, 7π/4)
	double R_axes_interp;  // Loop radius along axes using zero-crossing interpolation
	double R_diag_interp;  // Loop radius along diagonals using zero-crossing interpolation
};

/**
 * @brief Compute all loop radius observables from a 2D field slice
 * 
 * This function computes four loop radius measurements based on the real part
 * of the complex field (rephi) on a 2D slice. The measurements use different
 * methods to estimate the loop radius in physical units.
 * 
 * @param axion  Pointer to the Scalar field object
 * @param slice  Slice index to analyze
 * @return LoopRadiusData containing all four radius measurements
 */
LoopRadiusData computeLoopRadius(Scalar *axion, int slice);

/**
 * @brief Compute loop radius using weighted masking along cardinal axes
 * 
 * Uses angles [0, π/2, π, 3π/2] with angular width dα = π/8.
 * Weighting function: w = exp(-phi²*σ) with σ = 10.
 * 
 * @param rephi  2D array of real part of phi (linearized: row-major)
 * @param Lx     Grid size in x direction
 * @param Ly     Grid size in y direction
 * @param delta  Grid spacing in physical units
 * @return Radius in physical units
 */
double loopRadiusAxes(const float *rephi, size_t Lx, size_t Ly, double delta);
double loopRadiusAxes(const double *rephi, size_t Lx, size_t Ly, double delta);

/**
 * @brief Compute loop radius using weighted masking along diagonals
 * 
 * Uses angles [π/4, 3π/4, 5π/4, 7π/4] with angular width dα = π/8.
 * Weighting function: w = exp(-phi²*σ) with σ = 10.
 * 
 * @param rephi  2D array of real part of phi (linearized: row-major)
 * @param Lx     Grid size in x direction
 * @param Ly     Grid size in y direction
 * @param delta  Grid spacing in physical units
 * @return Radius in physical units
 */
double loopRadiusDiag(const float *rephi, size_t Lx, size_t Ly, double delta);
double loopRadiusDiag(const double *rephi, size_t Lx, size_t Ly, double delta);

/**
 * @brief Compute loop radius using zero-crossing interpolation along horizontal axis
 * 
 * Finds zero-crossings of rephi along the horizontal axis through center of mass,
 * uses linear interpolation to refine zero positions.
 * 
 * @param rephi  2D array of real part of phi (linearized: row-major)
 * @param Lx     Grid size in x direction
 * @param Ly     Grid size in y direction
 * @param delta  Grid spacing in physical units
 * @return Radius in physical units
 */
double loopRadiusAxesInterp(const float *rephi, size_t Lx, size_t Ly, double delta);
double loopRadiusAxesInterp(const double *rephi, size_t Lx, size_t Ly, double delta);

/**
 * @brief Compute loop radius using zero-crossing interpolation along π/4 diagonal
 * 
 * Finds zero-crossings of rephi along the diagonal at angle π/4 through center of mass,
 * uses bilinear interpolation for sub-pixel sampling.
 * 
 * @param rephi  2D array of real part of phi (linearized: row-major)
 * @param Lx     Grid size in x direction
 * @param Ly     Grid size in y direction
 * @param delta  Grid spacing in physical units
 * @return Radius in physical units
 */
double loopRadiusDiagInterp(const float *rephi, size_t Lx, size_t Ly, double delta);
double loopRadiusDiagInterp(const double *rephi, size_t Lx, size_t Ly, double delta);

#endif	// _LOOPRADIUS_
