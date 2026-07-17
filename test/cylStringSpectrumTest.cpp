#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <vector>

#include "comms/comms.h"
#include "io/readWrite.h"
#include "scalar/scalarField.h"
#include "spectrum/spectrum.h"
#include "utils/logger.h"
#include "utils/misc.h"
#include "utils/parse.h"

#ifndef USE_2DCYL
#error "cylStringSpectrumTest requires USE_2DCYL"
#endif

namespace {

template<typename Float>
void fillMovingLoop(Scalar *field, double radius, double boost,
                    double coreWidth)
{
	auto *m = static_cast<Float *>(field->mStart());
	auto *v = static_cast<Float *>(field->vCpu());
	const size_t Nz = field->NX();
	const size_t NrLocal = field->NZ();
	const double delta = field->Delta();

	unsigned long long localRows = NrLocal, rowOffset = 0;
	MPI_Exscan(&localRows, &rowOffset, 1, MPI_UNSIGNED_LONG_LONG,
		MPI_SUM, MPI_COMM_WORLD);
	if (commRank() == 0)
		rowOffset = 0;

	#pragma omp parallel for schedule(static)
	for (size_t irho = 0; irho < NrLocal; ++irho) {
		const double rho = double(rowOffset + irho)*delta;
		const double radialDistance = rho - radius;
		for (size_t iz = 0; iz < Nz; ++iz) {
			const size_t index = irho*Nz + iz;
			const double z = double(iz)*delta;
			const double distance2 = radialDistance*radialDistance + z*z;
			const double distance = std::sqrt(distance2);
			const double theta = std::atan2(z, radialDistance);
			const double modulus = std::tanh(distance/coreWidth);

			/* Translation R(t)=R+boost*t gives dtheta/dt = boost*z/d^2.
			 * The core-width regulator keeps the lattice value finite. */
			const double thetaDot = boost*z/(distance2 + coreWidth*coreWidth);
			const double cosine = std::cos(theta);
			const double sine = std::sin(theta);
			const double phiReal = modulus*cosine;
			const double phiImag = modulus*sine;

			m[2*index] = Float(phiReal);
			m[2*index + 1] = Float(phiImag);
			v[2*index] = Float(-thetaDot*phiImag);
			v[2*index + 1] = Float(thetaDot*phiReal);
		}
	}
}

struct FitResult {
	double slope;
	double intercept;
	size_t points;
};

FitResult fitPowerLaw(const SpecBin &spectrum, double binMultiplier,
                      double minimumK, double maximumK)
{
	double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
	size_t points = 0;
	for (size_t bin = 0; bin < spectrum.NBins(); ++bin) {
		const double k = (double(bin) + 0.5)/binMultiplier;
		const double power = spectrum(bin, SPECTRUM_KK);
		if (k < minimumK || k > maximumK || !(power > 0.0) ||
		    !std::isfinite(power))
			continue;
		const double x = std::log(k);
		const double y = std::log(power);
		sx += x;
		sy += y;
		sxx += x*x;
		sxy += x*y;
		++points;
	}
	const double denominator = double(points)*sxx - sx*sx;
	if (points < 2 || denominator == 0.0)
		return {std::numeric_limits<double>::quiet_NaN(), 0.0, points};
	const double slope = (double(points)*sxy - sx*sy)/denominator;
	return {slope, (sy - slope*sx)/double(points), points};
}

} // namespace

int main(int argc, char **argv)
{
	Cosmos cosmos = initAxions(argc, argv);
	commSync();

	auto *field = new Scalar(&cosmos, sizeN, sizeZ, sPrec, cDev, zInit, false,
		zGrid, FIELD_SAXION, lType, cosmos.ICData().Nghost);
	const double delta = field->Delta();
	const double loopRadius = 0.25*double(field->TZ())*delta;
	const double coreWidth = 2.0*delta;
	const double boost = 0.1;

	LogMsg(VERB_HIGH, "[cyl-string-spectrum] grid=%zux%zu radius=%.8e core=%.8e "
	       "boost=%.4f", field->TZ(), field->NX(), loopRadius, coreWidth,
	       boost);
	if (field->Precision() == FIELD_SINGLE)
		fillMovingLoop<float>(field, loopRadius, boost, coreWidth);
	else
		fillMovingLoop<double>(field, loopRadius, boost, coreWidth);

	if (cyl3DExportIndex >= 0)
		writeConf3DFrom2DCyl(field, cyl3DExportIndex);

	MeasInfo info = deninfa;
	info.nbinsspec = -1;
	info.mask = SPMASK_FLAT;
	info.nrt = NRUN_K;
	SpecBin spectrum(field, false, info);

	MPI_Barrier(MPI_COMM_WORLD);
	const double start = MPI_Wtime();
	spectrum.nRun(SPMASK_FLAT, NRUN_K);
	const double localElapsed = MPI_Wtime() - start;
	double elapsed = 0.0;
	MPI_Allreduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX,
		MPI_COMM_WORLD);
	spectrum.nmodRun();
	spectrum.avekRun();

	/* Avoid the box-scale infrared bins and the core-dominated ultraviolet. */
	const double fitMinimum = 4.0;
	/* N/16 is appropriate for production grids, but leaves no interval in the
	 * deliberately tiny N=32 exporter smoke test. */
	double fitMaximum = std::max(fitMinimum + 4.0,
		double(std::min(field->NX(), field->TZ()))/16.0);
	if (specKMax >= 0)
		fitMaximum = std::min(fitMaximum, 0.8*double(specKMax));
	const FitResult fit = fitPowerLaw(spectrum, 1.0, fitMinimum, fitMaximum);

	if (commRank() == 0) {
		std::ofstream output("out/cylStringSpectrum.dat");
		output << "# bin k_over_k0 power k_times_power nmodes averagek "
		          "rms_k_over_k0\n";
		for (size_t bin = 0; bin < spectrum.NBins(); ++bin) {
			const double k = double(bin) + 0.5;
			const double power = spectrum(bin, SPECTRUM_KK);
			const double modes = spectrum(bin, SPECTRUM_NN);
			const double averageK = spectrum(bin, SPECTRUM_AK);
			const double rmsK = modes > 0.0 ? std::sqrt(averageK/modes) : 0.0;
			output << bin << ' ' << k << ' ' << power << ' ' << k*power
			       << ' ' << modes << ' ' << averageK << ' ' << rmsK << '\n';
		}
		LogMsg(VERB_HIGH, "[cyl-string-spectrum] time=%.6e s fit-range=[%.2f,%.2f] "
		       "points=%zu slope=%.6f expected=-1 deviation=%.6f", elapsed,
		       fitMinimum, fitMaximum, fit.points, fit.slope, fit.slope + 1.0);
		LogMsg(VERB_HIGH, "[cyl-string-spectrum] wrote out/cylStringSpectrum.dat");
	}

	const bool valid = fit.points >= 4 && std::isfinite(fit.slope);
	int failure = valid ? 0 : 1, globalFailure = 0;
	MPI_Allreduce(&failure, &globalFailure, 1, MPI_INT, MPI_MAX,
		MPI_COMM_WORLD);
	delete field;
	endAxions();
	return globalFailure == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
