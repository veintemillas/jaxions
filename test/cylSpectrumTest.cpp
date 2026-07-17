#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <fftw3-mpi.h>

#include "comms/comms.h"
#include "scalar/scalarField.h"
#include "spectrum/spectrum.h"
#include "utils/misc.h"
#include "utils/parse.h"
#include "utils/logger.h"

#ifndef USE_2DCYL
#error "cylSpectrumTest requires USE_2DCYL"
#endif

namespace {

enum class RadialProfile { Constant, Bessel };

template<typename Float>
void fillMode(Scalar *field, size_t zMode, size_t rhoMode,
              double amplitude, RadialProfile radialProfile)
{
	auto *m = static_cast<Float *>(field->mStart());
	auto *v = static_cast<Float *>(field->vCpu());
	const size_t Nz = field->NX();
	const size_t NrLocal = field->NZ();
	const size_t NrGlobal = field->TZ();
	const size_t rhoOffset = size_t(commRank())*NrLocal;
	const double delta = field->Delta();
	const double kz = M_PI*double(zMode)/(double(Nz)*delta);
	const double krho = M_PI*double(rhoMode)/(double(NrGlobal)*delta);

	for (size_t irho = 0; irho < NrLocal; ++irho) {
		const double rho = double(rhoOffset + irho)*delta;
		const double radial = radialProfile == RadialProfile::Constant
			? 1.0 : ::j0(krho*rho);
		for (size_t iz = 0; iz < Nz; ++iz) {
			const size_t id = irho*Nz + iz;
			const double z = double(iz)*delta;
			const double thetaDot = amplitude*std::sin(kz*z)*radial;

			/* phi=1 and phidot=i*thetaDot give exactly the requested thetaDot. */
			m[2*id] = Float(1);
			m[2*id + 1] = Float(0);
			v[2*id] = Float(0);
			v[2*id + 1] = Float(thetaDot);
		}
	}
}

struct Peak {
	size_t bin;
	double value;
	double total;
};

struct Mode {
	size_t z;
	size_t rho;
	double amplitude;
};

double maximumRankTime(double localTime)
{
	double maximum = 0.0;
	MPI_Allreduce(&localTime, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	return maximum;
}

template<typename Float>
double benchmarkR2C(size_t NrGlobal, size_t Nz);

template<>
double benchmarkR2C<float>(size_t NrGlobal, size_t Nz)
{
	ptrdiff_t localRows = 0, rowOffset = 0;
	const ptrdiff_t complexColumns = ptrdiff_t(Nz/2 + 1);
	const ptrdiff_t allocation = fftwf_mpi_local_size_2d(
		ptrdiff_t(NrGlobal), complexColumns, MPI_COMM_WORLD,
		&localRows, &rowOffset);
	auto *data = fftwf_alloc_real(2*allocation);
	auto *plan = data == nullptr ? nullptr : fftwf_mpi_plan_dft_r2c_2d(
		ptrdiff_t(NrGlobal), ptrdiff_t(Nz), data,
		reinterpret_cast<fftwf_complex *>(data), MPI_COMM_WORLD, FFTW_ESTIMATE);
	if (plan == nullptr) {
		LogError("[cyl-spectrum-test] Could not create float MPI R2C benchmark plan");
		if (data != nullptr)
			fftwf_free(data);
		return -1.0;
	}
	std::fill(data, data + 2*allocation, 0.25f);
	MPI_Barrier(MPI_COMM_WORLD);
	const double start = MPI_Wtime();
	fftwf_execute(plan);
	const double elapsed = MPI_Wtime() - start;
	fftwf_destroy_plan(plan);
	fftwf_free(data);
	return maximumRankTime(elapsed);
}

template<>
double benchmarkR2C<double>(size_t NrGlobal, size_t Nz)
{
	ptrdiff_t localRows = 0, rowOffset = 0;
	const ptrdiff_t complexColumns = ptrdiff_t(Nz/2 + 1);
	const ptrdiff_t allocation = fftw_mpi_local_size_2d(
		ptrdiff_t(NrGlobal), complexColumns, MPI_COMM_WORLD,
		&localRows, &rowOffset);
	auto *data = fftw_alloc_real(2*allocation);
	auto *plan = data == nullptr ? nullptr : fftw_mpi_plan_dft_r2c_2d(
		ptrdiff_t(NrGlobal), ptrdiff_t(Nz), data,
		reinterpret_cast<fftw_complex *>(data), MPI_COMM_WORLD, FFTW_ESTIMATE);
	if (plan == nullptr) {
		LogError("[cyl-spectrum-test] Could not create double MPI R2C benchmark plan");
		if (data != nullptr)
			fftw_free(data);
		return -1.0;
	}
	std::fill(data, data + 2*allocation, 0.25);
	MPI_Barrier(MPI_COMM_WORLD);
	const double start = MPI_Wtime();
	fftw_execute(plan);
	const double elapsed = MPI_Wtime() - start;
	fftw_destroy_plan(plan);
	fftw_free(data);
	return maximumRankTime(elapsed);
}

std::vector<Mode> readModes()
{
	std::vector<Mode> modes;
	if (commRank() == 0) {
		std::ifstream input("k.dat");
		if (!input) {
			LogError("[cyl-spectrum-test] Could not open k.dat");
		} else {
			std::string line;
			while (std::getline(input, line)) {
				const size_t comment = line.find('#');
				if (comment != std::string::npos)
					line.erase(comment);
				std::istringstream values(line);
				size_t zMode = 0, rhoMode = 0;
				double amplitude = 1.0;
				if (values >> zMode >> rhoMode) {
					values >> amplitude; // optional; remains 1 when absent
					if (zMode == 0)
						LogError("[cyl-spectrum-test] zMode must be positive");
					else
						modes.push_back({zMode, rhoMode, amplitude});
				}
			}
		}
	}

	unsigned long long count = modes.size();
	MPI_Bcast(&count, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
	std::vector<unsigned long long> packed(2*size_t(count));
	std::vector<double> amplitudes(size_t(count), 0.0);
	if (commRank() == 0)
		for (size_t i = 0; i < modes.size(); ++i) {
			packed[2*i] = modes[i].z;
			packed[2*i + 1] = modes[i].rho;
			amplitudes[i] = modes[i].amplitude;
		}
	if (count > 0) {
		MPI_Bcast(packed.data(), int(packed.size()), MPI_UNSIGNED_LONG_LONG, 0,
			MPI_COMM_WORLD);
		MPI_Bcast(amplitudes.data(), int(amplitudes.size()), MPI_DOUBLE, 0,
			MPI_COMM_WORLD);
	}
	if (commRank() != 0)
		for (size_t i = 0; i < size_t(count); ++i)
			modes.push_back({size_t(packed[2*i]), size_t(packed[2*i + 1]),
				amplitudes[i]});

	return modes;
}

Peak findPeak(const SpecBin &spectrum)
{
	const double *bins = spectrum.data(SPECTRUM_KK);
	const size_t count = spectrum.NBins();
	const auto peak = std::max_element(bins, bins + count);
	return {size_t(peak - bins), *peak,
		std::accumulate(bins, bins + count, 0.0)};
}

template<typename Float>
bool runCase(Scalar *field, MeasInfo info, const char *name,
             size_t zMode, size_t rhoMode, double amplitude,
             RadialProfile radialProfile)
{
	fillMode<Float>(field, zMode, rhoMode, amplitude, radialProfile);
	SpecBin spectrum(field, false, info);
	MPI_Barrier(MPI_COMM_WORLD);
	const double spectrumStart = MPI_Wtime();
	spectrum.nRun(SPMASK_FLAT, NRUN_K);
	const double spectrumTime = maximumRankTime(MPI_Wtime() - spectrumStart);
	const double r2cTime = benchmarkR2C<Float>(field->TZ(), field->NX());
	const Peak peak = findPeak(spectrum);

	const double expectedKOverK0 = std::hypot(
		0.5*double(zMode),
		0.5*double(rhoMode)*double(field->NX())/double(field->TZ()));
	const size_t expectedBin = size_t(std::floor(expectedKOverK0));
	const double leakage = peak.total > 0.0 ? 1.0 - peak.value/peak.total : 1.0;

	if (commRank() == 0)
		LogMsg(VERB_HIGH, "[cyl-spectrum-test] %-12s expected-bin=%zu peak-bin=%zu "
		       "peak=%.8e total=%.8e leakage=%.6f",
		       name, expectedBin, peak.bin, peak.value, peak.total, leakage);
	if (commRank() == 0 && r2cTime > 0.0)
		LogMsg(VERB_HIGH, "[cyl-spectrum-test] timing grid=%zux%zu ranks=%d threads=%d: "
		       "cyl-spectrum=%.6e s fftw-mpi-r2c=%.6e s ratio=%.3f",
		       field->TZ(), field->NX(), commSize(), commThreads(), spectrumTime,
		       r2cTime, spectrumTime/r2cTime);

	/* Finite radial truncation spreads power; this is a debugging sanity check. */
	return std::isfinite(peak.total) && peak.total > 0.0 &&
		std::abs(long(peak.bin) - long(expectedBin)) <= 2;
}

} // namespace

int main(int argc, char **argv)
{
	Cosmos cosmos = initAxions(argc, argv);
	commSync();

	auto *field = new Scalar(&cosmos, sizeN, sizeZ, sPrec, cDev, zInit, false,
		zGrid, FIELD_SAXION, lType, cosmos.ICData().Nghost);
	MeasInfo info = deninfa;
	info.nbinsspec = -1;
	info.mask = SPMASK_FLAT;
	info.nrt = NRUN_K;
	const auto modes = readModes();

	bool ok = !modes.empty();
	for (const Mode &mode : modes) {
		const size_t zMode = mode.z;
		const size_t rhoMode = mode.rho;
		const std::string name = "z" + std::to_string(zMode) +
			"-rho" + std::to_string(rhoMode);
		const RadialProfile profile = rhoMode == 0
			? RadialProfile::Constant : RadialProfile::Bessel;
		const bool caseOk = field->Precision() == FIELD_SINGLE
			? runCase<float>(field, info, name.c_str(), zMode, rhoMode,
				mode.amplitude, profile)
			: runCase<double>(field, info, name.c_str(), zMode, rhoMode,
				mode.amplitude, profile);
		ok = caseOk && ok;
	}

	int localFailure = ok ? 0 : 1;
	int globalFailure = 0;
	MPI_Allreduce(&localFailure, &globalFailure, 1, MPI_INT, MPI_MAX,
		MPI_COMM_WORLD);
	delete field;
	endAxions();
	return globalFailure == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
