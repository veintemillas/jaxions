#include "io/loops_io.h"
#include <hdf5.h>
#include <mpi.h>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include "scalar/scalarField.h"
#include "io/readWrite.h"

struct GatherBlock {
    void*     ptr;    // rank 0: start of gathered block inside m2Cpu
    uint64_t  count;  // total elements across ranks
};

// Gathers 'local.size()' items into rank 0 at 'dst_bytes' and advances dst_bytes.
// Requires: on rank 0, 'dst_bytes' points into a large-enough buffer (m2Cpu).
template <typename T>
static inline GatherBlock gather_append_to_rank0(const std::vector<T>& local,
                                                 uint8_t*& dst_bytes,
                                                 MPI_Datatype mpi_type)
{
    int rank, nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    uint64_t localCount = (uint64_t)local.size();
    std::vector<uint64_t> counts(nRanks), displs(nRanks);

    MPI_Allgather(&localCount, 1, MPI_UINT64_T,
                  counts.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);

    displs[0] = 0;
    for (int r = 1; r < nRanks; ++r) displs[r] = displs[r-1] + counts[r-1];
    uint64_t totalCount = displs[nRanks-1] + counts[nRanks-1];

    void* recv_ptr = (rank == 0) ? (void*)dst_bytes : nullptr;

    // Safe cast: MPI_Gatherv counts/disp are int — use a temp int vec
    std::vector<int> counts_i(nRanks), displs_i(nRanks);
    for (int r=0; r<nRanks; ++r) {
        counts_i[r] = (int)counts[r];
        displs_i[r] = (int)displs[r];
    }

    MPI_Gatherv(local.data(), (int)localCount, mpi_type,
                recv_ptr, counts_i.data(), displs_i.data(), mpi_type,
                0, MPI_COMM_WORLD);

    GatherBlock blk{nullptr, totalCount};
    if (rank == 0) {
        blk.ptr = recv_ptr;
        dst_bytes += totalCount * sizeof(T);  // advance byte pointer
    }
    return blk;
}

static inline MPI_Datatype mpi_u32() { return MPI_UINT32_T; }
static inline MPI_Datatype mpi_u64() { return MPI_UINT64_T; }
static inline MPI_Datatype mpi_u8 () { return MPI_UINT8_T;  }
static inline MPI_Datatype mpi_i8 () { return MPI_INT8_T;   }
static inline MPI_Datatype mpi_f64() { return MPI_DOUBLE;   }

static inline void h5_write_1d(const char* group, const char* name,
                              hid_t h5type, const void* data, hsize_t N, int rango,
                               IOParms& iop)
{
    // Ensure group exists under meas_id
    if (!H5Lexists(iop.meas_id, group, H5P_DEFAULT)) {
        hid_t g = H5Gcreate2(iop.meas_id, group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gclose(g);
    }
    hid_t g = H5Gopen2(iop.meas_id, group, H5P_DEFAULT);

    // Dataset path = "<group>/<name>"
    std::string dpath = std::string(group) + "/" + name;

    hsize_t dims[1] = { N };
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dset  = H5Dcreate2(iop.meas_id, dpath.c_str(), h5type, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

		hid_t fspace = H5Dget_space(dset);
		if (commRank() == rango) {
			hsize_t offset = 0;
			H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &offset, NULL, dims, NULL);
		} else {
			H5Sselect_none(space);
			H5Sselect_none(fspace);
		}

    H5Dwrite(dset, h5type, space, fspace, H5P_DEFAULT, data);

    H5Dclose(dset);
    H5Sclose(space);
    H5Gclose(g);
}

static inline void h5_write_2d(const char* group, const char* name,
                               hid_t h5type, const void* data,
                                hsize_t rows, hsize_t cols, int rango,
                                 IOParms& iop)
{
    if (!H5Lexists(iop.meas_id, group, H5P_DEFAULT)) {
        hid_t g = H5Gcreate2(iop.meas_id, group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gclose(g);
    }
    hid_t g = H5Gopen2(iop.meas_id, group, H5P_DEFAULT);

    std::string dpath = std::string(group) + "/" + name;

    hsize_t dims[2] = { rows, cols };
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t dset  = H5Dcreate2(iop.meas_id, dpath.c_str(), h5type, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		hid_t fspace = H5Dget_space(dset);
		if (commRank() == rango) {
			hsize_t offset = 0;
			H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &offset, NULL, dims, NULL);
		} else {
			H5Sselect_none(space);
			H5Sselect_none(fspace);
		}
    H5Dwrite(dset, h5type, space, fspace, H5P_DEFAULT, data);

    H5Dclose(dset);
    H5Sclose(space);
    H5Gclose(g);
}

// Reorders loop tables by descending loop_len_com, in-place in m2Cpu,
// and rewires B_*.ptr to the sorted buffers. No large temporaries: only a
// small perm array lives in the m2 tail. Writing code remains unchanged.
static inline void sort_loops_inplace_rewire(
    Scalar* axion,
    GatherBlock& B_labels,
    GatherBlock& B_sizes,
    GatherBlock& B_offsets,   // must be global (N+1)
    GatherBlock& B_closed,
#ifdef USE_2DCYL
    GatherBlock& B_chiralities,
#endif
    GatherBlock& B_len_com,
    GatherBlock& B_com,
    GatherBlock& B_inertia,
    GatherBlock& B_eigs,
    GatherBlock& B_origin,
    GatherBlock& B_coords
) {
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return;

    const uint64_t N = B_sizes.count;                              // #loops
    const uint64_t M = static_cast<const uint64_t*>(B_offsets.ptr)[N]; // #vertices

    // Work in the free tail of m2Cpu, right after gathered coords
    auto* base = static_cast<uint8_t*>(axion->m2Cpu());
    uint8_t* wr = static_cast<uint8_t*>(B_coords.ptr) + 3*M*sizeof(double);
    auto a8 = [&](){ wr = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(wr)+7ULL) & ~7ULL); };
    auto resv = [&](uint64_t nbytes)->void* { a8(); void* p = wr; wr += nbytes; return p; };

    // (optional) capacity check
    const uint64_t cap = axion->eSize()*axion->DataSize();
    auto need = [&](uint64_t nb){ return (uint64_t)(wr - base) + nb <= cap; };

    // Small permutation buffer
    auto* perm = static_cast<uint32_t*>(resv(N*sizeof(uint32_t)));
    for (uint32_t i=0;i<N;++i) perm[i]=i;
    std::stable_sort(perm, perm+N, [&](uint32_t a, uint32_t b){
        return static_cast<const double*>(B_len_com.ptr)[a] >
               static_cast<const double*>(B_len_com.ptr)[b];
    });

    // Reserve sorted buffers in m2 tail
    auto* LAB = static_cast<uint32_t*>(resv(N   * sizeof(uint32_t)));
    auto* SIZ = static_cast<uint64_t*>(resv(N   * sizeof(uint64_t)));
    auto* OFF = static_cast<uint64_t*>(resv((N+1)*sizeof(uint64_t)));
    auto* CLO = static_cast<uint8_t *> (resv(N   * sizeof(uint8_t )));
#ifdef USE_2DCYL
    auto* CHI = static_cast<int8_t  *> (resv(N   * sizeof(int8_t  )));
#endif
    auto* LCM = static_cast<double*  > (resv(N   * sizeof(double   )));
    auto* COM = static_cast<double*  > (resv(3*N * sizeof(double   )));
    auto* INE = static_cast<double*  > (resv(6*N * sizeof(double   )));
    auto* EIG = static_cast<double*  > (resv(3*N * sizeof(double   )));
    auto* ORI = static_cast<double*  > (resv(3*N * sizeof(double   )));
    auto* CRD = static_cast<double*  > (resv(3*M * sizeof(double   )));

    if ((uint64_t)(wr - base) > cap) {
        LogError("[sort_loops] Not enough m2Cpu space for sorted view.");
        return;
    }

    // Source views
    const auto* L0 = static_cast<const uint32_t*>(B_labels.ptr);
    const auto* S0 = static_cast<const uint64_t*>(B_sizes.ptr);
    const auto* Of = static_cast<const uint64_t*>(B_offsets.ptr);
    const auto* C0 = static_cast<const uint8_t *> (B_closed.ptr);
#ifdef USE_2DCYL
    const auto* CH0= static_cast<const int8_t  *> (B_chiralities.ptr);
#endif
    const auto* LC = static_cast<const double*  > (B_len_com.ptr);
    const auto* CM = static_cast<const double*  > (B_com.ptr);
    const auto* IN = static_cast<const double*  > (B_inertia.ptr);
    const auto* EG = static_cast<const double*  > (B_eigs.ptr);
    const auto* OR = static_cast<const double*  > (B_origin.ptr);
    const auto* CF = static_cast<const double*  > (B_coords.ptr);

    // Fill sorted arrays + rebuild offsets
    OFF[0] = 0;
    for (uint64_t ii=0; ii<N; ++ii) {
        const uint32_t i = perm[ii];
        LAB[ii] = L0[i];
        SIZ[ii] = S0[i];
        CLO[ii] = C0[i];
#ifdef USE_2DCYL
        CHI[ii] = CH0[i];
#endif
        LCM[ii] = LC[i];
        std::memcpy(&COM[3*ii], &CM[3*i], 3*sizeof(double));
        std::memcpy(&INE[6*ii], &IN[6*i], 6*sizeof(double));
        std::memcpy(&EIG[3*ii], &EG[3*i], 3*sizeof(double));
        std::memcpy(&ORI[3*ii], &OR[3*i], 3*sizeof(double));
        OFF[ii+1] = OFF[ii] + SIZ[ii];
    }

    // Coords: stream chunks in perm order using original offsets
    uint64_t wv = 0;
    for (uint64_t ii=0; ii<N; ++ii) {
        const uint32_t i = perm[ii];
        const uint64_t a0 = Of[i], a1 = Of[i+1], K = a1 - a0;
        std::memcpy(&CRD[3*wv], &CF[3*a0], 3*K*sizeof(double));
        wv += K;
    }

		// OFF already sized to N+1; build prefix sum in the *sorted* order.
		OFF[0] = 0;
		for (uint64_t ii = 0; ii < N; ++ii)
		    OFF[ii+1] = OFF[ii] + SIZ[ii];

    // Rewire gathered blocks to point to the sorted buffers
    B_labels .ptr = LAB;
    B_sizes  .ptr = SIZ;
    B_offsets.ptr = OFF;            B_offsets.count = N+1; // important
    B_closed .ptr = CLO;
#ifdef USE_2DCYL
    B_chiralities.ptr = CHI;
#endif
    B_len_com.ptr = LCM;
    B_com    .ptr = COM;
    B_inertia.ptr = INE;
    B_eigs   .ptr = EIG;
    B_origin .ptr = ORI;
    B_coords .ptr = CRD;

}



void writeStringLoopObservables(Scalar *axion, StringLoopParms slp, int rango, IOParms& iop) {

	LogMsg (VERB_NORMAL, "[wSLO] String Loop Observables");LogFlush();
  const char* stringGroup = "/string";
  const char* loopsGroup  = "/string/loops";

  // Ensure /string group exists
  if (!H5Lexists(iop.meas_id, stringGroup, H5P_DEFAULT)) {
      hid_t str_grp = H5Gcreate2(iop.meas_id, stringGroup, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			H5Gclose(str_grp);
  }

  // Ensure /string/loops group exists
  if (!H5Lexists(iop.meas_id, loopsGroup, H5P_DEFAULT)) {
      hid_t str_grp = H5Gcreate2(iop.meas_id, loopsGroup, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			H5Gclose(str_grp);
  }

  // Write the individual datasets
  writeArray(slp.len.data(),    slp.len.size(),    loopsGroup, "lengths",    rango);
	writeArray(slp.vel.data(),    slp.vel.size(),    loopsGroup, "velocities",    rango);
	writeArray(slp.gam.data(),    slp.gam.size(),    loopsGroup, "gammas",    rango);
	writeArray(slp.cub.data(),    slp.cub.size(),    loopsGroup, "cubes",    rango);


	// ---------- NEW: gather all *loop-level* vectors to rank 0 and write ----------
	{
	const char* loopsGroup = "/string/loops";

	int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Work byte-pointer inside m2Cpu (big buffer you told me about)
	uint8_t* dst = (uint8_t*) axion->m2Cpu();

	// (Optional) Check capacity vs a rough upper bound if you like.


	if (0)
	printf("[pre-gather] rank %d: labels=%zu sizes=%zu len=%zu\n",
       commRank(),
       slp.loop_labels.size(),
       slp.loop_sizes.size(),
       slp.loop_len_com.size());
			 // make ofsets global again!

	LogMsg(VERB_NORMAL,"[wSLO] gathering");
	// 1) Gather/append each vector in a fixed order
	GatherBlock B_labels     = gather_append_to_rank0(slp.loop_labels,     dst, mpi_u32());
	GatherBlock B_sizes      = gather_append_to_rank0(slp.loop_sizes,      dst, mpi_u64());
	GatherBlock B_offsets    = gather_append_to_rank0(slp.loop_offsets,    dst, mpi_u64());
	GatherBlock B_closed     = gather_append_to_rank0(slp.loop_closed,     dst, mpi_u8 ());
#ifdef USE_2DCYL
	GatherBlock B_chiralities= gather_append_to_rank0(slp.loop_chiralities,dst, mpi_i8 ());
#endif
	GatherBlock B_len_com    = gather_append_to_rank0(slp.loop_len_com,    dst, mpi_f64());
	GatherBlock B_com        = gather_append_to_rank0(slp.loop_com,        dst, mpi_f64());
	GatherBlock B_inertia    = gather_append_to_rank0(slp.loop_inertia,    dst, mpi_f64());
	GatherBlock B_eigs       = gather_append_to_rank0(slp.loop_inertia_eigs,dst, mpi_f64());
	GatherBlock B_origin     = gather_append_to_rank0(slp.loop_origin,     dst, mpi_f64());
	GatherBlock B_coords     = gather_append_to_rank0(slp.loop_coords,     dst, mpi_f64());

	MPI_Barrier(MPI_COMM_WORLD);

	// make ofsets global again!
	if (rank==0){
	// std::vector<uint64_t> global_offsets(B_sizes.count + 1, 0);
	uint64_t* B_offsets_ = static_cast<uint64_t*>(B_offsets.ptr);
	uint64_t* B_sizes_   = static_cast<uint64_t*>(B_sizes.ptr);
	uint32_t* B_labels_  = static_cast<uint32_t*>(B_labels.ptr);
	double* B_len_       = static_cast<double*>(B_len_com.ptr);
	B_offsets_[0] = 0;
	for (uint64_t i = 0; i < B_sizes.count; ++i){
	    B_offsets_[i+1] = B_sizes_[i] + B_offsets_[i];
			if (0)
				printf("%d lab %hu len %lf\n",i, B_labels_[i], B_len_[i]);
		}
	}

	// Now sort & rewire; writing code stays untouched
	sort_loops_inplace_rewire(axion,
	                          B_labels, B_sizes, B_offsets, B_closed,
#ifdef USE_2DCYL
	                          B_chiralities,
#endif
	                          B_len_com, B_com, B_inertia, B_eigs, B_origin, B_coords);

	LogMsg(VERB_NORMAL,"[wSLO] writting");

	// 2) Write everything (flat names, as you chose)
	// Scalars
	h5_write_1d(loopsGroup, "labels",  H5T_NATIVE_UINT,  B_labels.ptr,   (hsize_t)B_labels.count,   0, iop);
	h5_write_1d(loopsGroup, "sizes",   H5T_NATIVE_ULLONG,B_sizes.ptr,    (hsize_t)B_sizes.count,    0, iop);
	h5_write_1d(loopsGroup, "offsets", H5T_NATIVE_ULLONG,B_offsets.ptr,  (hsize_t)(B_sizes.count+1),0, iop);
	h5_write_1d(loopsGroup, "closed",  H5T_NATIVE_UCHAR, B_closed.ptr,   (hsize_t)B_closed.count,   0, iop);
#ifdef USE_2DCYL
	h5_write_1d(loopsGroup, "chiralities", H5T_NATIVE_INT8, B_chiralities.ptr,
	            (hsize_t)B_chiralities.count, 0, iop);
#endif
	h5_write_1d(loopsGroup, "llengths", H5T_NATIVE_DOUBLE, B_len_com.ptr, (hsize_t)B_len_com.count, 0, iop);

	// 2D views for the flattened 3/6 columns
	const uint64_t N = B_labels.count;                   // #loops
	const uint64_t M = B_coords.count / 3;               // #vertices
	h5_write_2d(loopsGroup, "com_coords",   H5T_NATIVE_DOUBLE, B_com.ptr,     (hsize_t)N, 3, 0, iop);
	h5_write_2d(loopsGroup, "inertia",      H5T_NATIVE_DOUBLE, B_inertia.ptr, (hsize_t)N, 6, 0, iop);
	h5_write_2d(loopsGroup, "inertia_eigs", H5T_NATIVE_DOUBLE, B_eigs.ptr,    (hsize_t)N, 3, 0, iop);
	h5_write_2d(loopsGroup, "origin",       H5T_NATIVE_DOUBLE, B_origin.ptr,  (hsize_t)N, 3, 0, iop);
	h5_write_2d(loopsGroup, "coords",       H5T_NATIVE_DOUBLE, B_coords.ptr,  (hsize_t)M, 3, 0, iop);

	}

	/*	String metadata		*/

	hid_t group_id = H5Gopen2(iop.meas_id, "/string", H5P_DEFAULT);
	herr_t status = H5Aexists(group_id, "String number");

	if (status==0){
		writeAttribute(group_id, &(slp.stringdata.strDen),  "String number",    H5T_NATIVE_HSIZE);
		writeAttribute(group_id, &(slp.stringdata.strChr),  "String chirality", H5T_NATIVE_HSSIZE);
		writeAttribute(group_id, &(slp.stringdata.wallDn),  "Wall number",      H5T_NATIVE_HSIZE);
		writeAttribute(group_id, &(slp.stringdata.strLen),  "String length",    H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &(slp.stringdata.strDeng), "String number with gamma",    H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &(slp.stringdata.strVel),  "String velocity",  H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &(slp.stringdata.strVel2), "String velocity squared",    H5T_NATIVE_DOUBLE);
		writeAttribute(group_id, &(slp.stringdata.strGam),  "String gamma",     H5T_NATIVE_DOUBLE);
	}
		commSync();

		H5Gclose (group_id);
}

void	writeStringLabelMap (Scalar *axion, IOParms& iop)
{
	hid_t	eGrp_id, group_id, rset_id, tset_id, aset_id, plist_id, chunk_id;
	hid_t	rSpace, tSpace, aSpace, memSpace, dataType, totalSpace;
	hsize_t	total, slice, slab, offset, rOff;

	char	prec[16], fStr[16];
	int	length = 8;

	const hsize_t maxD[1] = { H5S_UNLIMITED };

	size_t	dataSize;

	int myRank = commRank();

	LogMsg (VERB_NORMAL, "[WSLM] Writing String Label Map from m2");
	LogMsg (VERB_NORMAL, "");

	if (axion->Field() != FIELD_SAXION || !(axion->m2Status() & M2_LABEL_MAP)){
			LogMsg(VERB_NORMAL,"[WSLM] Called without label map! (Field = %d, sDStatus= %d)\n",axion->Field(),axion->m2Status());
			return ;
		}

	/*      Start profiling         */

	Profiler &prof = getProfiler(PROF_HDF5);
	prof.start();

	if (iop.header == false || iop.opened == false)
	{
		LogError ("[wEd] Error: measurement file not opened. Ignoring write request.\n");
		return;
	}

	/* Label maps are unsigned short int */

	dataType = H5T_NATIVE_UINT;
	dataSize = sizeof(unsigned int);


	uint redlZ = axion->rTotalDepth();
	uint redlX = axion->rLength();

	total = ((hsize_t) redlX)*((hsize_t) redlX)*((hsize_t) redlZ);
	// uint furu = (axion->Depth())*commSize();
	// total = ((hsize_t) redlX)*((hsize_t) redlX)*((hsize_t) furu);
	slab  = ((hsize_t) redlX)*((hsize_t) redlX);

	/*	Create space for writing the raw data to disk with chunked access	*/
	if ((totalSpace = H5Screate_simple(1, &total, maxD)) < 0)	// Whole data
	{
		LogError ("Fatal error H5Screate_simple");
		prof.stop();
		exit (1);
	}

	/*	Set chunked access	*/
	if ((chunk_id = H5Pcreate (H5P_DATASET_CREATE)) < 0)
	{
		LogError ("Fatal error H5Pcreate");
		prof.stop();
		exit (1);
	}

	if (H5Pset_chunk (chunk_id, 1, &slab) < 0)
	{
		LogError ("Fatal error H5Pset_chunk");
		prof.stop();
		exit (1);
	}

	/*	Tell HDF5 not to try to write a 100Gb+ file full of zeroes with a single process	*/
	if (H5Pset_fill_time (chunk_id, H5D_FILL_TIME_NEVER) < 0)
	{
		LogError ("Fatal error H5Pset_alloc_time");
		prof.stop();
		exit (1);
	}

	/*	Create a group for string data if it doesn't exist	*/
	auto status = H5Lexists (iop.meas_id, "/string", H5P_DEFAULT);

	if (!status)
		eGrp_id = H5Gcreate2(iop.meas_id, "/string", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			eGrp_id = H5Gopen2(iop.meas_id, "/string", H5P_DEFAULT);		// Group exists
		} else {
			LogError ("Error: can't check whether group /energy exists");
			prof.stop();
			return;
		}
	}

	/*	Create a group for string labels if it doesn't exist
	we call it density for the full resolution and rdensity for the reduced resolution	*/
	char *gr_name;
	if (axion->Reduced())
		gr_name = "rlabels";
	else
		gr_name = "labels";

	status = H5Lexists (eGrp_id, gr_name, H5P_DEFAULT);

	if (!status)
		group_id = H5Gcreate2(eGrp_id, gr_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else {
		if (status > 0) {
			group_id = H5Gopen2(eGrp_id, gr_name, H5P_DEFAULT);		// Group exists
			LogMsg (VERB_HIGH, "Group /string/%s exists",gr_name);
		} else {
			LogError ("Error: can't check whether group /string/%s exists",gr_name);
			prof.stop();
			return;
		}
	}

	/*	Might be reduced	*/
	writeAttribute(group_id, &redlX, "Size",  H5T_NATIVE_UINT);
	writeAttribute(group_id, &redlZ, "Depth", H5T_NATIVE_UINT);

	/*	Create a dataset for the whole axion data	*/

	char auCh[24];
	sprintf (auCh, "/string/%s/data", gr_name);

	if (1) {
		aset_id = H5Dcreate (iop.meas_id, auCh, dataType, totalSpace, H5P_DEFAULT, chunk_id, H5P_DEFAULT);

		if (aset_id < 0)
		{
			LogError("Error creating aux dataset");
			prof.stop();
			exit (0);
		}

		aSpace = H5Dget_space (aset_id);
	}
	/*	We read 2D slabs as a workaround for the 2Gb data transaction limitation of MPIO	*/

	memSpace = H5Screate_simple(1, &slab, NULL);	// Slab

	commSync();

	LogMsg (VERB_HIGH, "Rank %d ready to write", myRank);

	const hsize_t Lz = axion->rDepth();

	if (1) {
		for (hsize_t zDim = 0; zDim < Lz; zDim++)
		{
			/*	Select the slab in the file	*/
			offset = (((hsize_t) (myRank*(Lz))) + zDim)*slab;
			H5Sselect_hyperslab(aSpace, H5S_SELECT_SET, &offset, NULL, &slab, NULL);

			/*	Write raw data	*/
			auto tErr = H5Dwrite (aset_id, dataType, memSpace, aSpace, iop.mlist_id, (static_cast<char *> (axion->m2Cpu())+slab*zDim*dataSize));

			if (tErr < 0)
			{
				LogError ("Error writing theta dataset");
				prof.stop();
				exit(0);
			}
		}

		commSync();
	}

	LogMsg (VERB_NORMAL, "[WSLM] Write label map successful ");

	size_t bytes = 0;

	/*	Close the dataset	*/

	if (1) {
		H5Dclose (aset_id);
		H5Sclose (aSpace);
		bytes += total*dataSize;
	}

	H5Sclose (memSpace);

	/*	Close the file		*/

	H5Sclose (totalSpace);
	H5Pclose (chunk_id);
	H5Gclose (group_id);
	H5Gclose (eGrp_id);

        prof.stop();
	prof.add(std::string("Write String Label map"), 0., ((double) bytes)*1e-9);

	LogMsg (VERB_NORMAL, "Written %lu bytes", bytes);
}
