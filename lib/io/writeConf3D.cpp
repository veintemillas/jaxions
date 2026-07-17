#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include <hdf5.h>
#include <mpi.h>

#include "comms/comms.h"
#include "io/readWrite.h"
#include "utils/logger.h"
#include "utils/parse.h"

#ifdef USE_2DCYL
namespace {

template<typename Float>
void exportTyped(Scalar *axion, int index)
{
	const size_t nzHalf = axion->NX();
	const size_t nrLocal = axion->NZ();
	const size_t nrGlobal = axion->TZ();
	size_t transverseSide = 2*nrGlobal;
	if (cyl3DInscribed) {
		transverseSide = size_t(std::floor(std::sqrt(2.0)*double(nrGlobal)));
		transverseSide -= transverseSide % 2;
		/* The periodic Nyquist coordinate is exactly side/2.  Keep even the
		 * farthest corner strictly inside the available radial domain. */
		while (transverseSide > 2 &&
		       std::sqrt(2.0)*0.5*double(transverseSide) >= double(nrGlobal))
			transverseSide -= 2;
	}
	/* The inscribed diagnostic is deliberately a cube.  Besides avoiding the
	 * unavailable transverse corners, crop the reconstructed [-Nz,Nz) axial
	 * interval to the same physical length.  Keeping the same Delta makes its
	 * component-wise Nyquist momentum identical to the cylindrical lattice. */
	const size_t outputDepth = cyl3DInscribed ? transverseSide : 2*nzHalf;
	const int ranks = commSize();
	const int rank = commRank();

	if (outputDepth % size_t(ranks) != 0) {
		LogError("[cyl3d-export] Cartesian depth %zu is not divisible by %d ranks.",
			outputDepth, ranks);
		return;
	}
	if (nrLocal*nzHalf > size_t(std::numeric_limits<int>::max())) {
		LogError("[cyl3d-export] Cylindrical gather exceeds MPI's int count limit.");
		return;
	}

	/* Gather only O(N^2) theta-dot.  Each rank generates its own output slab. */
	std::vector<Float> local(nrLocal*nzHalf, Float(0));
	auto *m = static_cast<Float *>(axion->mStart());
	auto *v = static_cast<Float *>(axion->vCpu());
	#pragma omp parallel for schedule(static)
	for (size_t irho = 0; irho < nrLocal; ++irho) {
		for (size_t iz = 1; iz < nzHalf; ++iz) {
			const size_t id = irho*nzHalf + iz;
			const Float phir = m[2*id];
			const Float phii = m[2*id + 1];
			const Float vr = v[2*id];
			const Float vi = v[2*id + 1];
			const Float mod2 = phir*phir + phii*phii;
			const Float numerator = vi*phir - vr*phii;
			local[id] = mod2 > Float(0) ? numerator/mod2 : numerator;
		}
	}
	std::vector<Float> cylindrical(nrGlobal*nzHalf, Float(0));
	const MPI_Datatype mpiType = std::is_same<Float, float>::value
		? MPI_FLOAT : MPI_DOUBLE;
	MPI_Allgather(local.data(), int(local.size()), mpiType,
		cylindrical.data(), int(local.size()), mpiType, MPI_COMM_WORLD);

	/* Reuse writeConf for the complete, reader-compatible metadata. */
	writeConf(axion, index);
	commSync();

	char path[1152];
	sprintf(path, "%s/%s.%05d", outDir, outName, index);
	hid_t access = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_fapl_mpio(access, MPI_COMM_WORLD, MPI_INFO_NULL);
	hid_t file = H5Fopen(path, H5F_ACC_RDWR, access);
	H5Pclose(access);
	if (file < 0) {
		LogError("[cyl3d-export] Could not reopen %s.", path);
		return;
	}

	auto setUInt = [file](const char *name, unsigned int value) {
		hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
		if (attr >= 0) {
			H5Awrite(attr, H5T_NATIVE_UINT, &value);
			H5Aclose(attr);
		}
	};
	auto setDouble = [file](const char *name, double value) {
		hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
		if (attr >= 0) {
			H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
			H5Aclose(attr);
		}
	};
	const unsigned int outputSide = static_cast<unsigned int>(transverseSide);
	const unsigned int outputNz = static_cast<unsigned int>(outputDepth);
	setUInt("Nx", outputSide);
	setUInt("Ny", outputSide);
	setUInt("Nz", outputNz);
	setUInt("Size", outputSide);
	setUInt("Depth", outputNz);
	const double outputPhysicalSize = double(transverseSide)*axion->Delta();
	setDouble("Physical size", outputPhysicalSize);

	H5Ldelete(file, "/m", H5P_DEFAULT);
	H5Ldelete(file, "/v", H5P_DEFAULT);

	const hsize_t surface = hsize_t(transverseSide)*hsize_t(transverseSide);
	const hsize_t totalSites = surface*hsize_t(outputDepth);
	const hsize_t totalElements = 2*totalSites;
	const hsize_t localDepth = hsize_t(outputDepth/size_t(ranks));
	const hsize_t localSites = surface*localDepth;
	const hsize_t chunkSites = std::min<hsize_t>(localSites, hsize_t(1) << 20);
	const hsize_t chunkElements = std::max<hsize_t>(2, 2*chunkSites);
	const hid_t dataType = std::is_same<Float, float>::value
		? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
	if (rank == 0)
		LogMsg(VERB_HIGH, "[cyl3d-export] allocating %.3f GiB on disk for m+v",
			double(4*totalSites*sizeof(Float))/(1024.0*1024.0*1024.0));

	hid_t totalSpace = H5Screate_simple(1, &totalElements, nullptr);
	hid_t creation = H5Pcreate(H5P_DATASET_CREATE);
	H5Pset_chunk(creation, 1, &chunkElements);
	H5Pset_fill_time(creation, H5D_FILL_TIME_NEVER);
	hid_t mDataset = H5Dcreate(file, "/m", dataType, totalSpace,
		H5P_DEFAULT, creation, H5P_DEFAULT);
	hid_t vDataset = H5Dcreate(file, "/v", dataType, totalSpace,
		H5P_DEFAULT, creation, H5P_DEFAULT);
	H5Pclose(creation);
	H5Sclose(totalSpace);

	hid_t transfer = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE);
	hid_t mFileSpace = H5Dget_space(mDataset);
	hid_t vFileSpace = H5Dget_space(vDataset);
	std::vector<Float> mBuffer(size_t(2*chunkSites));
	std::vector<Float> vBuffer(size_t(2*chunkSites));

	LogMsg(VERB_HIGH, "[cyl3d-export] writing %zux%zux%zu%s to %s with %.1f MiB buffers",
		transverseSide, transverseSide, outputDepth,
		cyl3DInscribed ? " (inscribed)" : " (zero-padded disk)", path,
		double(4*chunkSites*sizeof(Float))/(1024.0*1024.0));
	for (hsize_t begin = 0; begin < localSites; begin += chunkSites) {
		const hsize_t countSites = std::min(chunkSites, localSites - begin);
		#pragma omp parallel for schedule(static)
		for (hsize_t item = 0; item < countSites; ++item) {
			const hsize_t localLinear = begin + item;
			const size_t localZ = size_t(localLinear/surface);
			const size_t inPlane = size_t(localLinear % surface);
			const size_t iy = inPlane/transverseSide;
			const size_t ix = inPlane - iy*transverseSide;
			const size_t globalZ = size_t(rank)*size_t(localDepth) + localZ;
			const size_t halfSide = transverseSide/2;
			const long sx = ix <= halfSide
				? long(ix) : long(ix) - long(transverseSide);
			const long sy = iy <= halfSide
				? long(iy) : long(iy) - long(transverseSide);
			/* FFT ordering for the cropped periodic interval: for an even M,
			 * retain z=0,...,M/2-1,-M/2,...,-1. */
			const bool negativeZ = globalZ >= (outputDepth + 1)/2;
			const size_t absZ = negativeZ
				? outputDepth - globalZ : globalZ;

			Float thetaDot = Float(0);
			const double radius = std::hypot(double(sx), double(sy));
			if (radius < double(nrGlobal) && absZ > 0 && absZ < nzHalf) {
				const size_t lower = size_t(std::floor(radius));
				const double fraction = radius - double(lower);
				const Float a = cylindrical[lower*nzHalf + absZ];
				const Float b = lower + 1 < nrGlobal
					? cylindrical[(lower + 1)*nzHalf + absZ] : Float(0);
				thetaDot = Float((1.0 - fraction)*double(a) + fraction*double(b));
				if (negativeZ)
					thetaDot = -thetaDot;
			}

			mBuffer[2*size_t(item)] = Float(1);
			mBuffer[2*size_t(item) + 1] = Float(0);
			vBuffer[2*size_t(item)] = Float(0);
			vBuffer[2*size_t(item) + 1] = thetaDot;
		}

		const hsize_t countElements = 2*countSites;
		const hsize_t fileOffset = 2*(hsize_t(rank)*localSites + begin);
		H5Sselect_hyperslab(mFileSpace, H5S_SELECT_SET, &fileOffset,
			nullptr, &countElements, nullptr);
		H5Sselect_hyperslab(vFileSpace, H5S_SELECT_SET, &fileOffset,
			nullptr, &countElements, nullptr);
		hid_t memorySpace = H5Screate_simple(1, &countElements, nullptr);
		const herr_t ms = H5Dwrite(mDataset, dataType, memorySpace,
			mFileSpace, transfer, mBuffer.data());
		const herr_t vs = H5Dwrite(vDataset, dataType, memorySpace,
			vFileSpace, transfer, vBuffer.data());
		H5Sclose(memorySpace);
		if (ms < 0 || vs < 0) {
			LogError("[cyl3d-export] Error writing Cartesian datasets.");
			break;
		}
	}

	H5Sclose(mFileSpace);
	H5Sclose(vFileSpace);
	H5Pclose(transfer);
	H5Dclose(mDataset);
	H5Dclose(vDataset);
	H5Fclose(file);
	commSync();
	if (rank == 0)
		LogMsg(VERB_HIGH, "[cyl3d-export] completed %s: grid=%zux%zux%zu Lxy=%.8e "
		       "Lz=%.8e corner-rho=%.8e Rmax=%.8e", path,
			transverseSide, transverseSide, outputDepth, outputPhysicalSize,
			double(outputDepth)*axion->Delta(),
			std::sqrt(2.0)*0.5*outputPhysicalSize,
			double(nrGlobal)*axion->Delta());
}

} // namespace
#endif

void writeConf3DFrom2DCyl(Scalar *axion, int index)
{
#ifdef USE_2DCYL
	if (axion->Field() != FIELD_SAXION) {
		LogError("[cyl3d-export] Only FIELD_SAXION is supported.");
		return;
	}
	if (axion->Device() == DEV_GPU)
		axion->transferCpu(FIELD_MV);
	if (axion->Precision() == FIELD_SINGLE)
		exportTyped<float>(axion, index);
	else if (axion->Precision() == FIELD_DOUBLE)
		exportTyped<double>(axion, index);
	else
		LogError("[cyl3d-export] Unsupported precision.");
#else
	(void) axion;
	(void) index;
	LogError("writeConf3DFrom2DCyl requires USE_2DCYL.");
#endif
}
