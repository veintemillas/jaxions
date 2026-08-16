#include "spectrum/cylindricalSpectrum.h"

#ifdef USE_2DCYL

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <fftw3-mpi.h>
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "comms/comms.h"
#include "fft/fftCode.h"
#include "io/readWrite.h"
#include "scalar/scalarField.h"
#include "spectrum/J0tabler.h"
#include "spectrum/spectrum.h"
#include "utils/logger.h"
#include "utils/parse.h"

namespace {

enum CylComponent {
	CYL_KINETIC = 0,
	CYL_GRADIENT_Z = 1,
	CYL_GRADIENT_RHO = 2
};

template<typename Float>
inline void accumulateZ(double *output, const Float *input, double coefficient,
                        size_t count)
{
	size_t iz = 0;
#if defined(__AVX512F__)
	const __m512d factor = _mm512_set1_pd(coefficient);
	for (; iz + 8 <= count; iz += 8) {
		__m512d values;
		if constexpr (std::is_same<Float, float>::value)
			values = _mm512_cvtps_pd(_mm256_loadu_ps(input + iz));
		else
			values = _mm512_loadu_pd(input + iz);
		const __m512d accumulated = _mm512_loadu_pd(output + iz);
		_mm512_storeu_pd(output + iz,
			_mm512_fmadd_pd(factor, values, accumulated));
	}
#elif defined(__AVX__)
	const __m256d factor = _mm256_set1_pd(coefficient);
	for (; iz + 4 <= count; iz += 4) {
		__m256d values;
		if constexpr (std::is_same<Float, float>::value)
			values = _mm256_cvtps_pd(_mm_loadu_ps(input + iz));
		else
			values = _mm256_loadu_pd(input + iz);
		const __m256d accumulated = _mm256_loadu_pd(output + iz);
		_mm256_storeu_pd(output + iz,
			_mm256_add_pd(accumulated, _mm256_mul_pd(factor, values)));
	}
#endif
#if defined(__AVX__) || defined(__AVX512F__)
	for (; iz < count; ++iz)
		output[iz] += coefficient*double(input[iz]);
#else
	#pragma omp simd
	for (size_t index = 0; index < count; ++index)
		output[index] += coefficient*double(input[index]);
#endif
}

} // namespace

template<typename Float>
std::vector<double> CylindricalSpectrum::runComponent(SpecBin &spectrum,
                                                       int component,
                                                       SpectrumMaskType mask)
{
	Scalar *field = spectrum.field;
	if (spectrum.fType != FIELD_SAXION) {
		LogError("[2Dcyl spectrum] Kinetic/gradient spectra currently support FIELD_SAXION only.");
		return {};
	}

	const bool axialGradient = component == CYL_GRADIENT_Z;
	const bool radialGradient = component == CYL_GRADIENT_RHO;
	auto &zPlan = AxionFFT::fetchPlan(axialGradient
		? "spec1DGradientZ" : "spec1Dm2");
	auto &transposePlan = AxionFFT::fetchPlan(axialGradient
		? "transposeGradientZ" : "transpose");

	auto *m  = static_cast<Float *>(field->mStart());
	auto *v  = static_cast<Float *>(field->vCpu());
	auto *m2 = static_cast<Float *>(field->m2Cpu());

	const size_t Nz = field->NX();
	const size_t NrLocal = field->NZ();
	const size_t NrGlobal = field->TZ();
	const double delta = field->Delta();
	const Float scaleFactor = Float(*field->RV());
	const size_t axialStride = axialGradient ? Nz : Nz - 1;
	LogMsg(VERB_HIGH, "[2Dcyl spectrum] begin Nz=%zu NrLocal=%zu NrGlobal=%zu",
		Nz, NrLocal, NrGlobal);

	if (Nz < 3 || NrGlobal == 0) {
		LogError("[2Dcyl spectrum] Invalid lattice dimensions Nz=%zu Nr=%zu.", Nz, NrGlobal);
		return {};
	}

	/* Build each component directly in its axial-transform row layout.  The
	 * complex-field quotient is the same branch-safe definition used by the
	 * ordinary Jaxions kinetic/gradient builders:
	 *
	 *        d_i theta = Im(conj(phi) d_i phi)/|phi|^2 .
	 *
	 * K and G_rho are odd in z (DST-I); G_z is even (DCT-I). */
	if (component != CYL_KINETIC)
		field->exchangeGhosts(FIELD_M);
	auto *mWithGhosts = reinterpret_cast<std::complex<Float> *>(field->mCpu());
	const size_t ghosts = field->getNg();
	const bool firstRadialRank = commRank() == 0;
	const bool lastRadialRank = commRank() == commSize() - 1;
	LogMsg(VERB_HIGH, "[2Dcyl spectrum] %s transforming %zu local rows",
		axialGradient ? "DCT-I" : "DST-I", NrLocal);
	#pragma omp parallel for schedule(static)
	for (size_t irho = 0; irho < NrLocal; ++irho) {
		Float *row = m2 + irho*axialStride;
		const size_t ghostRow = irho + ghosts;
		auto maskWeight = [mask, scaleFactor] (const std::complex<Float> &center) {
			/* Match the VIL/VIL2 definitions used by the Cartesian builders:
			 * VIL weights d(theta) by |phi|/R, and VIL2 by |phi|^2/R^2. */
			switch (mask) {
				case SPMASK_VIL:
					return std::abs(center)/scaleFactor;
				case SPMASK_VIL2:
					return std::norm(center)/(scaleFactor*scaleFactor);
				default:
					return Float(1);
			}
		};
		auto quotient = [] (const std::complex<Float> &center,
		                    const std::complex<Float> &difference) {
			const Float denominator = std::norm(center);
			const Float numerator = std::imag(std::conj(center)*difference);
			return denominator > Float(0) ? numerator/denominator : numerator;
		};

		if (axialGradient) {
				/* z=0 and z=Nz-1 are fixed points of conjugate reflection. */
				const auto phi0 = mWithGhosts[ghostRow*Nz];
				const auto phi1 = mWithGhosts[ghostRow*Nz + 1];
				const auto endpoint = mWithGhosts[ghostRow*Nz + Nz - 1];
				const auto inside = mWithGhosts[ghostRow*Nz + Nz - 2];
			row[0] = maskWeight(phi0)*scaleFactor*quotient(phi0, phi1 - std::conj(phi1))/
				Float(2*delta);
				for (size_t iz = 1; iz + 1 < Nz; ++iz) {
					const auto center = mWithGhosts[ghostRow*Nz + iz];
					const auto plus = mWithGhosts[ghostRow*Nz + iz + 1];
				const auto difference = plus -
					mWithGhosts[ghostRow*Nz + iz - 1];
				row[iz] = maskWeight(center)*scaleFactor*quotient(center, difference)/Float(2*delta);
			}
				row[Nz - 1] = maskWeight(endpoint)*scaleFactor*quotient(endpoint,
					std::conj(inside) - inside)/Float(2*delta);
			continue;
		}

			/* z=0 and z=Nz-1 vanish for odd channels and are omitted by DST-I. */
			for (size_t iz = 1; iz + 1 < Nz; ++iz) {
			const size_t id = irho*Nz + iz;
			if (radialGradient) {
				const auto center = mWithGhosts[ghostRow*Nz + iz];
				if (firstRadialRank && irho == 0) {
					/* Cylindrical regularity: d_rho theta vanishes at the axis. */
					row[iz - 1] = Float(0);
				} else {
					const auto inside = mWithGhosts[(ghostRow - 1)*Nz + iz];
					const auto outside = lastRadialRank && irho + 1 == NrLocal
						? center /* even reflection about rho=NrGlobal-1/2 */
						: mWithGhosts[(ghostRow + 1)*Nz + iz];
					row[iz - 1] = maskWeight(center)*scaleFactor*quotient(center, outside - inside)/
						Float(2*delta);
				}
			} else {
				const Float phir = m[2*id];
				const Float phii = m[2*id + 1];
				const Float vr = v[2*id];
				const Float vi = v[2*id + 1];
				const Float mod2 = phir*phir + phii*phii;
				const Float angularVelocity = vi*phir - vr*phii;
				const std::complex<Float> center(phir, phii);
				row[iz - 1] = maskWeight(center)*scaleFactor*(mod2 > Float(0)
					? angularVelocity/mod2 : angularVelocity);
			}
		}
			row[Nz - 2] = Float(0); // DST transpose padding column
	}

	/* Diagnostic map for the cylindrical radial-gradient spectrum.  Preserve
	 * the real-space d_rho theta values before the axial DST overwrites m2.
	 * The saved one-dimensional array is ordered (rho,z), including the two
	 * identically-zero odd-channel z endpoints, and can be reshaped using the
	 * measurement file's Depth and Size attributes. */
	if (radialGradient && debug) {
		const size_t localMapSize = NrLocal*Nz;
		std::vector<double> localMap(localMapSize, 0.0);
		#pragma omp parallel for schedule(static)
		for (size_t irho = 0; irho < NrLocal; ++irho)
			for (size_t iz = 1; iz + 1 < Nz; ++iz)
				localMap[irho*Nz + iz] =
					double(m2[irho*axialStride + iz - 1]);

		const int ranks = commSize();
		std::vector<unsigned long long> rowCounts(size_t(ranks), 0);
		const unsigned long long localRows = NrLocal;
		MPI_Allgather(&localRows, 1, MPI_UNSIGNED_LONG_LONG,
			rowCounts.data(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
		std::vector<int> receiveCounts(size_t(ranks), 0);
		std::vector<int> receiveDisplacements(size_t(ranks), 0);
		size_t mapSize = 0;
		for (int rank = 0; rank < ranks; ++rank) {
			const size_t count = size_t(rowCounts[size_t(rank)])*Nz;
			if (count > size_t(INT_MAX) || mapSize > size_t(INT_MAX)) {
				LogError("[2Dcyl spectrum] partialRhoTheta map exceeds MPI_Gatherv limits.");
				return {};
			}
			receiveCounts[size_t(rank)] = int(count);
			receiveDisplacements[size_t(rank)] = int(mapSize);
			mapSize += count;
		}
		std::vector<double> globalMap(commRank() == 0 ? mapSize : 0);
		MPI_Gatherv(localMap.data(), int(localMapSize), MPI_DOUBLE,
			globalMap.data(), receiveCounts.data(), receiveDisplacements.data(),
			MPI_DOUBLE, 0, MPI_COMM_WORLD);
		writeArray(globalMap.data(), mapSize, "/cylSpectrum", "partialRhoTheta");
	}
	zPlan.run(FFT_FWD);
	LogMsg(VERB_HIGH, "[2Dcyl spectrum] axial transforms completed");

	const size_t Nkrho = NrGlobal;
	const double dkz = M_PI/(double(Nz - 1)*delta);
	const double dkrho = M_PI/(double(NrGlobal)*delta);
	const size_t axialModeCount = axialGradient ? Nz : Nz - 2;
	const size_t axialModeOffset = axialGradient ? 0 : 1;
	const double kFundamental = spectrum.k0;
	const bool useCutoff = specKMax >= 0;
	const double kCut = useCutoff ? double(specKMax)*kFundamental : 0.0;
	auto *radialInput = static_cast<Float *>(field->m2half());
	std::vector<size_t> localKzModes;
	size_t localKzSize = 0;

	if (useCutoff) {
		const int ranks = commSize();
		const size_t maxRows = field->NXYZg()/NrGlobal;
		std::vector<std::vector<size_t>> modesByRank(static_cast<size_t>(ranks));
		std::vector<uint64_t> workByRank(size_t(ranks), 0);

		/* Largest-work-first assignment with a hard per-rank memory cap. */
		for (size_t mode = 0; mode < axialModeCount; ++mode) {
			const double kz = dkz*double(mode + axialModeOffset);
			if (kz > kCut)
				break;
			const double radialSquared = std::max(0.0, kCut*kCut - kz*kz);
			const size_t radialModes = std::min(Nkrho,
				size_t(std::floor(std::sqrt(radialSquared)/dkrho)) + 1);
			int owner = -1;
			for (int rank = 0; rank < ranks; ++rank) {
				if (modesByRank[size_t(rank)].size() >= maxRows)
					continue;
				if (owner < 0 || workByRank[size_t(rank)] < workByRank[size_t(owner)] ||
				    (workByRank[size_t(rank)] == workByRank[size_t(owner)] &&
				     modesByRank[size_t(rank)].size() < modesByRank[size_t(owner)].size()))
					owner = rank;
			}
			if (owner < 0) {
				LogError("[2Dcyl spectrum] Balanced kz distribution exceeds m2half capacity.");
				return {};
			}
			modesByRank[size_t(owner)].push_back(mode);
			workByRank[size_t(owner)] += radialModes;
		}
		for (auto &modes : modesByRank)
			std::sort(modes.begin(), modes.end());
		if (commRank() == 0)
			for (int rank = 0; rank < ranks; ++rank)
				LogMsg(VERB_HIGH, "[2Dcyl spectrum] cutoff owner %d: kz-rows=%zu "
				       "predicted-modes=%llu\n", rank,
				       modesByRank[size_t(rank)].size(),
				       static_cast<unsigned long long>(workByRank[size_t(rank)]));
		localKzModes = modesByRank[size_t(commRank())];
		localKzSize = localKzModes.size();
		if (localKzSize*NrGlobal > field->NXYZg()) {
			LogError("[2Dcyl spectrum] Balanced receive needs %zu values; m2half holds %zu.",
				localKzSize*NrGlobal, field->NXYZg());
			return {};
		}

		std::vector<unsigned long long> rhoCounts(static_cast<size_t>(ranks));
		const unsigned long long localRhoCount = NrLocal;
		MPI_Allgather(&localRhoCount, 1, MPI_UNSIGNED_LONG_LONG,
			rhoCounts.data(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
		std::vector<size_t> rhoOffsets(size_t(ranks), 0);
		for (int rank = 1; rank < ranks; ++rank)
			rhoOffsets[size_t(rank)] = rhoOffsets[size_t(rank - 1)] +
				size_t(rhoCounts[size_t(rank - 1)]);

		const MPI_Datatype baseType = std::is_same<Float, float>::value
			? MPI_FLOAT : MPI_DOUBLE;
		std::vector<int> sendCounts(size_t(ranks), 0), sendDisplacements(size_t(ranks), 0);
		std::vector<int> recvCounts(size_t(ranks), 0), recvDisplacements(size_t(ranks), 0);
		std::vector<MPI_Datatype> sendTypes(size_t(ranks), baseType);
		std::vector<MPI_Datatype> recvTypes(size_t(ranks), baseType);
		std::vector<MPI_Datatype> createdTypes;

		for (int destination = 0; destination < ranks; ++destination) {
			const auto &modes = modesByRank[size_t(destination)];
			if (modes.empty())
				continue;
			std::vector<MPI_Aint> displacements(modes.size());
			for (size_t index = 0; index < modes.size(); ++index)
				displacements[index] = MPI_Aint(modes[index]*sizeof(Float));
			MPI_Datatype selectedRow, selectedRows;
			MPI_Type_create_hindexed_block(int(modes.size()), 1,
				displacements.data(), baseType, &selectedRow);
			MPI_Type_create_hvector(int(NrLocal), 1,
				MPI_Aint(axialStride*sizeof(Float)),
				selectedRow, &selectedRows);
			MPI_Type_commit(&selectedRows);
			MPI_Type_free(&selectedRow);
			sendCounts[size_t(destination)] = 1;
			sendTypes[size_t(destination)] = selectedRows;
			createdTypes.push_back(selectedRows);
		}

		for (int source = 0; source < ranks; ++source) {
			const uint64_t count = uint64_t(localKzSize)*rhoCounts[size_t(source)];
			const uint64_t displacement = uint64_t(rhoOffsets[size_t(source)])*
				localKzSize*sizeof(Float);
			if (count > INT_MAX || displacement > INT_MAX) {
				LogError("[2Dcyl spectrum] MPI_Alltoallw count/displacement exceeds INT_MAX.");
				for (MPI_Datatype type : createdTypes) MPI_Type_free(&type);
				return {};
			}
			recvCounts[size_t(source)] = int(count);
			recvDisplacements[size_t(source)] = int(displacement);
		}

		LogMsg(VERB_HIGH, "[2Dcyl spectrum] balanced cutoff transpose: "
		       "spec-kmax=%d local-kz=%zu predicted-work=%llu",
		       specKMax, localKzSize,
		       static_cast<unsigned long long>(workByRank[size_t(commRank())]));
		MPI_Alltoallw(m2, sendCounts.data(), sendDisplacements.data(), sendTypes.data(),
			radialInput, recvCounts.data(), recvDisplacements.data(), recvTypes.data(),
			MPI_COMM_WORLD);
		for (MPI_Datatype type : createdTypes)
			MPI_Type_free(&type);
		LogMsg(VERB_HIGH, "[2Dcyl spectrum] balanced filtered MPI transpose completed");
	} else {
		/* Full-spectrum fallback through FFTW's equal-size transpose. */
		ptrdiff_t localNr = 0, rhoOffset = 0, localKz = 0, kzOffset = 0;
		const ptrdiff_t transposeElements = fftw_mpi_local_size_2d_transposed(
			static_cast<ptrdiff_t>(NrGlobal), static_cast<ptrdiff_t>(axialStride),
			MPI_COMM_WORLD, &localNr, &rhoOffset, &localKz, &kzOffset);
		LogMsg(VERB_HIGH, "[2Dcyl spectrum] transpose layout: required=%td "
		       "input=(%td@%td)x%zu output=(%td@%td)x%zu",
		       transposeElements, localNr, rhoOffset, axialStride, localKz, kzOffset,
		       NrGlobal);
		transposePlan.run(FFT_FWD);
		localKzSize = size_t(localKz);
		if (localKzSize*NrGlobal > field->NXYZg()) {
			LogError("[2Dcyl spectrum] Local transpose exceeds m2half capacity.");
			return {};
		}
		for (size_t local = 0; local < localKzSize; ++local)
			if (kzOffset + ptrdiff_t(local) < ptrdiff_t(axialModeCount))
				localKzModes.push_back(size_t(kzOffset) + local);

		constexpr size_t transposeTile = 32;
		#pragma omp parallel for collapse(2) schedule(static)
		for (size_t kzBase = 0; kzBase < localKzSize; kzBase += transposeTile) {
			for (size_t rhoBase = 0; rhoBase < NrGlobal; rhoBase += transposeTile) {
				const size_t kzEnd = std::min(kzBase + transposeTile, localKzSize);
				const size_t rhoEnd = std::min(rhoBase + transposeTile, NrGlobal);
				for (size_t ikz = kzBase; ikz < kzEnd; ++ikz)
					for (size_t irho = rhoBase; irho < rhoEnd; ++irho)
						radialInput[irho*localKzSize + ikz] =
							m2[ikz*NrGlobal + irho];
			}
		}
		LogMsg(VERB_HIGH, "[2Dcyl spectrum] FFTW and local transpose completed");
	}

	const int threadCount = commThreads();
	std::vector<double> threadBins(spectrum.nbins*threadCount, 0.0);
	const size_t workKrhoCount = useCutoff ? std::min(Nkrho,
		size_t(std::floor(kCut/dkrho)) + 1) : Nkrho;

#ifdef USE_CYL_J0_VECTOR_CACHE
	std::vector<Float> besselTable(NrGlobal*Nkrho);
	#pragma omp parallel for schedule(static)
	for (size_t ikrho = 0; ikrho < Nkrho; ++ikrho)
		for (size_t irho = 0; irho < NrGlobal; ++irho) {
			const double argument = dkrho*double(ikrho*irho)*delta;
			besselTable[ikrho*NrGlobal + irho] = Float(double(irho)*delta*
				(radialGradient ? ::j1(argument) : ::j0(argument)));
		}
	LogMsg(VERB_HIGH, "[2Dcyl spectrum] full J%d vector ready: %zu coefficients "
	       "dkz=%.8e dkrho=%.8e", radialGradient ? 1 : 0,
	       besselTable.size(), dkz, dkrho);
#else
	/* The field now lives in m2half, so m2 is free for the product cache. */
	Float *besselCache = m2;
	const size_t cacheCapacity = field->NXYZg();
	const size_t maxProduct = (NrGlobal - 1)*(workKrhoCount - 1);
	const size_t cacheSize = std::min(cacheCapacity, maxProduct + 1);
	const double asymptoticFrom = M_PI*double(cacheSize)/double(NrGlobal);
	char cacheName[256];
	std::snprintf(cacheName, sizeof(cacheName),
		"out/J%dproduct.N%zu.C%zu.F%zu.bin", radialGradient ? 1 : 0,
		NrGlobal, cacheSize, sizeof(Float));
	bool cacheLoaded = false;
	if (commRank() == 0) {
		cacheLoaded = loadJ0ProductCache(
			cacheName, besselCache, NrGlobal, cacheSize);
		if (!cacheLoaded) {
			#pragma omp parallel for schedule(static)
			for (size_t product = 0; product < cacheSize; ++product)
				besselCache[product] = Float(radialGradient
					? ::j1(M_PI*double(product)/double(NrGlobal))
					: ::j0(M_PI*double(product)/double(NrGlobal)));
			if (!saveJ0ProductCache(
				cacheName, besselCache, NrGlobal, cacheSize))
				LogError("[2Dcyl spectrum] Could not save J0 product cache %s.",
					cacheName);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (commRank() != 0) {
		cacheLoaded = loadJ0ProductCache(
			cacheName, besselCache, NrGlobal, cacheSize);
		if (!cacheLoaded) {
			#pragma omp parallel for schedule(static)
			for (size_t product = 0; product < cacheSize; ++product)
				besselCache[product] = Float(radialGradient
					? ::j1(M_PI*double(product)/double(NrGlobal))
					: ::j0(M_PI*double(product)/double(NrGlobal)));
		}
	}
	LogMsg(VERB_HIGH, "[2Dcyl spectrum] m2 J%d product cache ready: "
	       "%zu/%zu values (%s %s), asymptotic for x>=%.8e",
	       radialGradient ? 1 : 0, cacheSize, maxProduct + 1,
	       cacheLoaded ? "loaded from" :
	       "generated for", cacheName, asymptoticFrom);
#endif

	const size_t activeLocalKz = localKzModes.size();
#if defined(__AVX512F__)
	LogMsg(VERB_HIGH, "[2Dcyl spectrum] radial kernel AVX-512 (%zu kz values)",
		activeLocalKz);
#elif defined(__AVX2__)
	LogMsg(VERB_HIGH, "[2Dcyl spectrum] radial kernel AVX2 (%zu kz values)",
		activeLocalKz);
#elif defined(__AVX__)
	LogMsg(VERB_HIGH, "[2Dcyl spectrum] radial kernel AVX (%zu kz values)",
		activeLocalKz);
#else
	LogMsg(VERB_HIGH, "[2Dcyl spectrum] radial kernel compiler SIMD (%zu kz values)",
		activeLocalKz);
#endif
	#pragma omp parallel
	{
		const int thread = omp_get_thread_num();
		std::vector<double> transformed(activeLocalKz, 0.0);

		#pragma omp for schedule(dynamic, 1)
		for (size_t ikrho = 0; ikrho < workKrhoCount; ++ikrho) {
			std::fill(transformed.begin(), transformed.end(), 0.0);
			size_t activeForKrho = activeLocalKz;
			if (useCutoff) {
				const double krhoForCut = dkrho*double(ikrho);
				const double remaining = kCut*kCut - krhoForCut*krhoForCut;
				if (remaining <= 0.0) {
					activeForKrho = 0;
				} else {
					const size_t largestCoordinate =
						size_t(std::floor(std::sqrt(remaining)/dkz));
					if (largestCoordinate < axialModeOffset) {
						activeForKrho = 0;
					} else {
						activeForKrho = size_t(std::upper_bound(
							localKzModes.begin(), localKzModes.end(),
							largestCoordinate - axialModeOffset) -
							localKzModes.begin());
					}
				}
			}
			if (activeForKrho == 0)
				continue;
#ifdef USE_CYL_J0_VECTOR_CACHE
			const Float *coefficients = besselTable.data() + ikrho*NrGlobal;
#endif

			for (size_t irho = 0; irho < NrGlobal; ++irho) {
#ifdef USE_CYL_J0_VECTOR_CACHE
				const double coefficient = double(coefficients[irho]);
#else
				const size_t product = ikrho*irho;
				double bessel;
				if (product < cacheSize) {
					bessel = double(besselCache[product]);
				} else {
					const double x = M_PI*double(product)/double(NrGlobal);
					const double phase = x - (radialGradient ? 0.75 : 0.25)*M_PI;
					bessel = std::sqrt(2.0/(M_PI*x)) *
						(std::cos(phase) + (radialGradient ? -3.0 : 1.0)*
						 std::sin(phase)/(8.0*x));
				}
				const double coefficient = double(irho)*delta*bessel;
#endif
				accumulateZ<Float>(transformed.data(),
					radialInput + irho*localKzSize, coefficient, activeForKrho);
			}

			const double krho = dkrho*double(ikrho);
			for (size_t ikzLocal = 0; ikzLocal < activeForKrho; ++ikzLocal) {
				const double kz = dkz*double(
					localKzModes[ikzLocal] + axialModeOffset);
				const double k = std::hypot(kz, krho);
				const double value = transformed[ikzLocal]*2.0*M_PI*delta*delta;
				const size_t bin = size_t(std::floor(
					(k/kFundamental)*spectrum.nbinmul));
				if (bin < spectrum.nbins)
					threadBins[size_t(thread)*spectrum.nbins + bin] +=
						krho*value*value;
			}
		}
	}
	LogMsg(VERB_HIGH, "[2Dcyl spectrum] Fourier-Bessel binning completed");

	std::vector<double> componentBins(spectrum.nbins, 0.0);
	for (int thread = 0; thread < threadCount; ++thread)
		for (size_t bin = 0; bin < spectrum.nbins; ++bin)
			componentBins[bin] += threadBins[size_t(thread)*spectrum.nbins + bin];

	std::vector<double> localBins = componentBins;
	MPI_Allreduce(localBins.data(), componentBins.data(), int(spectrum.nbins),
		MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	LogMsg(VERB_HIGH, "[2Dcyl spectrum] bin reduction completed");

	/* Continuum normalization, exactly equivalent to the ordinary FFT factor
	 *
	 *                  V / (2*Nsites^2) = L^3/(2*N^6)
	 *
	 * for a cubic N^3 lattice.
	 *
	 * Here `value` already approximates the continuum Fourier integral
	 * F(k)=Delta^3 sum_x f(x)e^-ikx.  In
	 *
	 *   (1/(2V)) sum_k |F_k|^2
	 *     -> (1/(2(2*pi)^3)) integral d^3k |F(k)|^2,
	 *
	 * the azimuthal integral contributes 2*pi and the omitted negative-kz
	 * half contributes another factor 2.  The remaining measure is therefore
	 * krho*dkrho*dkz/(4*pi^2), with krho accumulated above. */
	const double momentumMeasure = dkrho*dkz/(4.0*M_PI*M_PI);
	for (double &bin : componentBins)
		bin *= momentumMeasure;

	field->setM2(M2_DIRTY);
	LogMsg(VERB_NORMAL, "[2Dcyl spectrum] completed Nz=%zu Nr=%zu%s",
		Nz, NrGlobal, useCutoff ? " with momentum cutoff" : "");
	return componentBins;
}

void CylindricalSpectrum::modeData(SpecBin &spectrum)
{
	/* nmodRun() and avekRun() are normally called consecutively. */
	if (spectrum.binNN.size() == spectrum.nbins &&
	    spectrum.binAK.size() == spectrum.nbins)
		return;

	Scalar *field = spectrum.field;
	const size_t Nz = field->NX();
	const size_t Nr = field->TZ();
	const double delta = field->Delta();
	const double dkz = M_PI/(double(Nz - 1)*delta);
	const double dkrho = M_PI/(double(Nr)*delta);
	const double k0 = spectrum.k0;
	const int threads = commThreads();

	/* The ordinary implementation stores sum(mode weight) in binNN and
	 * sum(mode weight*(k/k0)^2) in binAK.  A radial Fourier--Bessel cell
	 * represents an annulus: 2*pi*irho transverse modes, and each positive
	 * sine mode represents the +/-kz pair.  At irho=0 there is one axial
	 * transverse mode rather than an annulus of zero area. */
	std::vector<double> threadModes(size_t(threads)*spectrum.nbins, 0.0);
	std::vector<double> threadK2(size_t(threads)*spectrum.nbins, 0.0);
	#pragma omp parallel
	{
		const int thread = omp_get_thread_num();
		#pragma omp for schedule(static)
		for (size_t irho = size_t(commRank()); irho < Nr;
		     irho += size_t(commSize())) {
			const double transverseMultiplicity = irho == 0
				? 1.0 : 2.0*M_PI*double(irho);
			const double modeWeight = 2.0*transverseMultiplicity;
			const double krho = dkrho*double(irho);
				for (size_t iz = 1; iz + 1 < Nz; ++iz) {
				const double kz = dkz*double(iz);
				const double q2 = (krho*krho + kz*kz)/(k0*k0);
				const size_t bin = size_t(std::floor(
					spectrum.nbinmul*std::sqrt(q2)));
				if (bin >= spectrum.nbins)
					continue;
				threadModes[size_t(thread)*spectrum.nbins + bin] += modeWeight;
				threadK2[size_t(thread)*spectrum.nbins + bin] += modeWeight*q2;
			}
		}
	}

	spectrum.binNN.assign(spectrum.nbins, 0.0);
	spectrum.binAK.assign(spectrum.nbins, 0.0);
	for (int thread = 0; thread < threads; ++thread)
		for (size_t bin = 0; bin < spectrum.nbins; ++bin) {
			spectrum.binNN[bin] += threadModes[size_t(thread)*spectrum.nbins + bin];
			spectrum.binAK[bin] += threadK2[size_t(thread)*spectrum.nbins + bin];
		}
	std::vector<double> localModes = spectrum.binNN;
	std::vector<double> localK2 = spectrum.binAK;
	MPI_Allreduce(localModes.data(), spectrum.binNN.data(), int(spectrum.nbins),
		MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(localK2.data(), spectrum.binAK.data(), int(spectrum.nbins),
		MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void CylindricalSpectrum::nRun(SpecBin &spectrum, SpectrumMaskType mask, nRunType nrt)
{
	if (mask != SPMASK_FLAT && mask != SPMASK_VIL && mask != SPMASK_VIL2) {
		LogError("[2Dcyl spectrum] Requested mask is not implemented.");
		return;
	}
	if (!(nrt & (NRUN_K | NRUN_G)))
		return;
	if (nrt & (NRUN_V | NRUN_S))
		LogMsg(VERB_NORMAL,
			"[2Dcyl spectrum] NRUN_V/NRUN_S are not implemented and are skipped.");

	switch (spectrum.fPrec) {
		case FIELD_SINGLE:
			if (nrt & NRUN_K)
				spectrum.binK = runComponent<float>(spectrum, CYL_KINETIC, mask);
			if (nrt & NRUN_G) {
				spectrum.binG = runComponent<float>(spectrum, CYL_GRADIENT_Z, mask);
				const auto radial = runComponent<float>(spectrum, CYL_GRADIENT_RHO, mask);
				for (size_t bin = 0; bin < spectrum.binG.size(); ++bin)
					spectrum.binG[bin] += radial[bin];
			}
			break;
		case FIELD_DOUBLE:
			if (nrt & NRUN_K)
				spectrum.binK = runComponent<double>(spectrum, CYL_KINETIC, mask);
			if (nrt & NRUN_G) {
				spectrum.binG = runComponent<double>(spectrum, CYL_GRADIENT_Z, mask);
				const auto radial = runComponent<double>(spectrum, CYL_GRADIENT_RHO, mask);
				for (size_t bin = 0; bin < spectrum.binG.size(); ++bin)
					spectrum.binG[bin] += radial[bin];
			}
			break;
		default:
			LogError("[2Dcyl spectrum] Unsupported field precision.");
	}
}

#endif
