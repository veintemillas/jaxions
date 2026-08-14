#include <memory>
#include <cstring>
#include <complex>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include "scalar/scalarField.h"
#include "scalar/folder.h"
#include "enum-field.h"

#include "strings/strings.h"
#include "strings/length.h"
#include "strings/labeling_tools.h"

#include <limits.h>

#include "utils/utils.h"

#include <vector>
#include "utils/index.h"

#include <omp.h>
#include <mpi.h>

#include "io/readWrite.h"

using namespace profiler;


// -----------------------------------------------------
// Function that calculates the length of strings
// -----------------------------------------------------

template<typename Float>
StringLoopParms	stringlength3	(Scalar *field, StringData strDen_in, StringMeasureType strmeas)
{
	LogMsg	(VERB_NORMAL, "[SL3] stringlength3");

	StringLoopParms slp;
	slp.stringdata = strDen_in;
	StringData	strDen = strDen_in;

	strDen.strLen = 0.;
	strDen.strDeng = 0.;
	strDen.strVel = 0.;
	strDen.strVel2 = 0.;
	strDen.strGam = 0.;
	strDen.strLen_local = 0.;
	strDen.strDeng_local = 0.;


//----------------------------------------------------------------------------
// Can the function run?
//----------------------------------------------------------------------------
	if (!(field->sDStatus() & SD_MAP)){
			LogMsg(VERB_NORMAL,"[SL3] Called without string map! (sDStatus= %d)\n",field->Field(),field->sDStatus());
			return slp;
		}

	if (strDen_in.strDen == 0) {
		LogMsg(VERB_NORMAL,"[SL3 called without strings: exit]");
		return slp;
	}

	if (!(field->Field() & FIELD_SAXION)) {
		LogMsg(VERB_NORMAL,"[SL3 ftype %d is not FIELD_SAXION  strings: exit]",field->Field());
		return slp;
	}

	//----------------------------------------------------------------------------
	// Global definitions to be used
	// strData stores 3 plaquette info (n3 char)
	// m2Cpu has at least (n3 + 2ng n2)x 2*sizeof(float) =  (8n3 + 16ng n2 char)
	// labelData will store a (ghosted) n3 map of unsigned int labels (n3+2Sf 4 char)
	// scOut will store n3 cube info (exit plaquettes + how many in/out) (n3 char)
	// scAll will store n3 cube info (in/out plaquettes + how many in/out) (n3 char)
	// idxCubeList stores a list of cubes with strings (2n3/8 size_t)
	//----------------------------------------------------------------------------

	size_t Lx = field->Length();
	size_t Lz = field->Depth();
	size_t Tz = field->TotalDepth();
	size_t n3 = field->Size();
	size_t Sf = Lx*Lx;
	int rank = commRank();
	int nMPI = commSize();

	char* strData = static_cast<char*>(field->sData());
  unsigned int*   labelData = reinterpret_cast<unsigned int*>(field->m2Cpu());
	typedef unsigned short int usi;
  usi* scOutData = reinterpret_cast<usi*>(reinterpret_cast<void*>(labelData + n3+2*Sf));
  usi* scInData =  reinterpret_cast<usi*>(reinterpret_cast<void*>(scOutData + n3));
  size_t* idxCubeList       = reinterpret_cast<size_t*>(reinterpret_cast<void*>(scOutData + 2*n3));
	size_t mBytes             = field->eSize()*field->DataSize();
	size_t idxCubeList_size   = (mBytes - 4*(n3+2*Sf)-2*n3)/sizeof(size_t);

	size_t nplaquettes_local = strDen.strDen_local;
	if (nplaquettes_local<idxCubeList_size)
		LogMsg	(VERB_NORMAL, "[SL3] Check Plaquettes < Space for idx, %zu < %zu ",nplaquettes_local,idxCubeList_size);
	else {
		LogMsg	(VERB_NORMAL, "[SL3] Check # Plaquettes < Space for idx, %zu > %zu!! Exit! ",nplaquettes_local,idxCubeList_size);
		LogError	("[SL3] # Plaquettes > Space for idx, %zu > %zu!! Exit! ");
		return slp;
	}

	//----------------------------------------------------------------------------
	// Program needs fields folded, Lz ghosts for m,v (for velocity) and strData
	//----------------------------------------------------------------------------

	if	(field->Folded())
	{
		Folder	munge(field);
		munge(UNFOLD_ALL);
	}

	LogMsg	(VERB_HIGH, "[SL3] Exchanging M ghosts");
	field->sendGhosts(FIELD_M,COMM_SDRV);
	field->sendGhosts(FIELD_M,COMM_WAIT);

	LogMsg	(VERB_NORMAL, "[SL3] recalling string data ghost");
	field->exchangeStringGhost();

	/* Check ghost FOR DEBUG */
	// for (size_t i=0;i<field->Surf();i++)
	// 	{
	// 		if (strdaa[i] != strdaa[field->Size()+i])
	// 			LogError("sD ghost exchange didn't work!");
	// 	}

	std::complex<Float> *ma  = static_cast<std::complex<Float>*>(field->mStart());

	/* clean m2 */
	LogMsg	(VERB_HIGH, "[SL3] Clear m2");
	memset (field->m2Cpu(), 0, mBytes);

	LogMsg	(VERB_HIGH, "[SL3]  Enter OMP");


	int nThreads = commThreads();

	//----------------------------------------------------------------------------
	// 1st we read strData, build cubes, gather idx
	// 2nd we label strings bidirectionally
	// 3rd we reduce labels, OMP, MPI
	// 4th we compute positions, lengths, velocities
	//----------------------------------------------------------------------------


	/* We read the whole grid and save STRCUB data */
	/* Threads will write indices atomically */
	std::atomic<size_t> globalIndex(0);

	#pragma omp parallel for collapse(2) schedule(static)
  for (size_t iz = 0; iz < Lz; ++iz) {
      for (size_t iy = 0; iy < Lx; ++iy) {
          for (size_t ix = 0; ix < Lx; ++ix) {
              size_t idx = ix + iy * Lx + iz * Sf;

              char st    = strData[idx];
              char stXY2 = strData[idx+Sf];
              char stYZ2 = strData[((ix + 1) % Lx) + iy * Lx + iz * Sf];
              char stZX2 = strData[ix + ((iy + 1) % Lx) * Lx + iz * Sf];

							int chi[6] = {0};
              int n_pla = 0;
              unsigned short exits = 0;
              unsigned short enter = 0;

              if (st & STRING_XY)    {
                n_pla++; chi[0] = (st & STRING_XY_POSITIVE) ?  1 : -1;  if (chi[0] > 0) exits |= STRCUB_XY; else enter |= STRCUB_XY;    };
              if (st & STRING_YZ)    {
                n_pla++; chi[1] = (st & STRING_YZ_POSITIVE) ?  1 : -1;  if (chi[1] > 0) exits |= STRCUB_YZ; else enter |= STRCUB_YZ;    };
              if (st & STRING_ZX)    {
                n_pla++; chi[2] = (st & STRING_ZX_POSITIVE) ?  1 : -1;  if (chi[2] > 0) exits |= STRCUB_ZX; else enter |= STRCUB_ZX;    };
              if (stXY2 & STRING_XY) {
                n_pla++; chi[3] = (stXY2 & STRING_XY_POSITIVE) ? -1 : 1; if (chi[3] > 0) exits |= STRCUB_XY2; else enter |= STRCUB_XY2; };
              if (stYZ2 & STRING_YZ) {
                n_pla++; chi[4] = (stYZ2 & STRING_YZ_POSITIVE) ? -1 : 1; if (chi[4] > 0) exits |= STRCUB_YZ2; else enter |= STRCUB_YZ2; };
              if (stZX2 & STRING_ZX) {
                n_pla++; chi[5] = (stZX2 & STRING_ZX_POSITIVE) ? -1 : 1; if (chi[5] > 0) exits |= STRCUB_ZX2; else enter |= STRCUB_ZX2; };

								if (n_pla == 0) continue;
                if (n_pla % 2 != 0) {
                    LogError("[SL3 ... identifyCubes] Odd number of plaquettes at (%zu,%zu,%zu)", ix, iy, iz);
                    continue;
                }
                if (n_pla ==2) {
                  exits |= STRCUB_1EX; enter |= STRCUB_1EX;};
                if (n_pla ==4) {
                  exits |= STRCUB_2EX; enter |= STRCUB_2EX;};
                if (n_pla ==6) {
                  exits |= STRCUB_3EX; enter |= STRCUB_3EX;};

                /* check INs = OUTs */
								int sum = 0 ;
								for (int i = 0; i < 6;i++)
									sum += chi[i];

								if (sum != 0){
									LogError("[SL3] chiralities mismatched!!");
// LogMsg(VERB_HIGH,"[SL3] (iz,iy,ix %d %d %d) %d xy2(%d).yz2(%d).zx2(%d) chi %d %d %d %d %d %d",iz,iy,ix,
// 	st&STRING_ONLY,stXY2&STRING_ONLY,stYZ2&STRING_ONLY,stZX2&STRING_ONLY,chi[0],chi[1],chi[2],chi[3],chi[4],chi[5]);
									}

									/* save cube data*/
									scOutData[idx] = exits;
									scInData[idx] = enter;

									size_t writeIndex = globalIndex++;
									if (writeIndex < idxCubeList_size) {
											idxCubeList[writeIndex] = idx;
									} else {
											LogError("[identifyCubes] Exceeded maximum cube capacity at index %zu (max %zu)", writeIndex,idxCubeList_size);
									}

			  } //end of ix loop
		  } //end of iy loop
	} // end of iz loop

	size_t ncubes_local = globalIndex;
	LogMsg(VERB_NORMAL,"[SL3] We found %zu cubes",ncubes_local);

// DEBUG
// {
// size_t count = 0;
// for (size_t i=0;i<n3;i++)
// 	if (scOutData[i])
// 		count++;
// LogOut("count %zu\n",count);
// count = 0;
// for (size_t i=0;i<n3;i++)
// 	if (scInData[i])
// 		count++;
// LogOut("count %zu\n",count);
// for (size_t i=0;i<n3+2*Sf;i++)
// 	if (labelData[i])
// 		count++;
// LogOut("count %zu\n",count);
// }

	//----------------------------------------------------------------------------
	// 2nd we label strings bidirectionally
	// When two labels meet, write an equivalence
	//----------------------------------------------------------------------------

	/* we update labels atomically, we start labeling with 1 (0=no_string)*/
	std::atomic<unsigned>nextLabel(1);
	// unsigned nextLabel = 1;
	std::vector<std::vector<std::pair<unsigned, unsigned>>> equiv_omp(nThreads);
	size_t counter = 0;

	#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		#pragma omp for
    for (size_t i = 0; i < ncubes_local; ++i) {

				size_t start_idx = Sf+idxCubeList[i]; // labelData has Sf ghosts!
        if (labelData[start_idx]) continue;   // if labelled exit

        unsigned label = nextLabel++; // returns old value atomically
				// alternative to prev. line
				// unsigned label = nextLabel.fetch_add(1, std::memory_order_relaxed); // unique

				size_t cur_idx = start_idx;           // save start_idx for backward
				labelData[cur_idx] = label;           // store thread safe

				counter++;

				char sc_out = scOutData[cur_idx-Sf];
				unsigned next_label;
        // Forward pass
        while (true) {

						if (cur_idx < Sf || cur_idx > n3+Sf) // we allow 1 label in the ghost region, then break
							break;

            size_t X[3];
            indexXeon::idx2Vec(cur_idx, X, Lx); // X includes ghost! 0-Lz+1
						bool isOut = false;
						// #pragma omp atomic read
            size_t next_idx = next_cube_idx(sc_out, X, Lx, Lz, Sf, &isOut,1);

						if (isOut) break;

						// #pragma omp atomic read
	            sc_out = scOutData[next_idx-Sf];
						// #pragma omp atomic read
						next_label = labelData[next_idx];
						bool msc = (sc_out & (STRCUB_2EX | STRCUB_3EX));

// LogOut("[SL3] F - loop %zu %u (%d %d %d) %zu/%u\n",start_idx,label,X[0],X[1],X[2],counter,ncubes_local);
// LogOut("        next %zu %u %d\n",next_idx,next_label,sc_out);

						/* if next cube is already labeled, we add equivalence and break
						(but for NEXT cubes with multiple exits) */
						if (next_label){
							if (next_label == label) {
							// no need to record an equivalence with ourselves
							break;
							}

							equiv_omp[tid].push_back({label,next_label});
							if (!msc)
								break;
						}

            cur_idx = next_idx;
						#pragma omp atomic write
							labelData[cur_idx] = label;
							counter++;
        } // end forward pass

        // Backward pass
				char sc_in = scInData[start_idx-Sf];
				unsigned prev_label;
        cur_idx = start_idx;
        while (true) {

					if (cur_idx < Sf || cur_idx > n3+Sf) // we allow 1 label in the ghost region, then break
						break;

            size_t X[3];
						indexXeon::idx2Vec(cur_idx, X, Lx);
						bool isOut = false;

            size_t prev_idx = prev_cube_idx(sc_in, X, Lx, Lz, Sf, &isOut,1);

						if (isOut) break;

						// #pragma omp atomic read
						sc_in = scInData[prev_idx-Sf];
						// #pragma omp atomic read
						prev_label = labelData[prev_idx];
						bool msc = (sc_in & (STRCUB_2EX | STRCUB_3EX));

// LogOut("[SL3] B - loop %zu %u (%d %d %d) %zu/%u\n",start_idx,label,X[0],X[1],X[2],counter,ncubes_local);
// LogOut("        prev %zu %u %d \n",prev_idx,prev_label,sc_in);

						if (prev_label){
							if (prev_label == label) break; // closed loop
							equiv_omp[tid].push_back({label,prev_label});
							if (!msc)
								break;
						}

						cur_idx = prev_idx;
						#pragma omp atomic write
            	labelData[cur_idx] = label;
							counter++;
        } // end backward pass

    } //end parallel
	}

		// size_t nlabels_local = nextLabel;

		// issued number of labels ()
		unsigned issued = nextLabel.load(std::memory_order_relaxed) - 1;

// LogMsg(VERB_NORMAL,"[SL3] We issued %u unique labels",nlabels_local);
// LogOut	("[SL3] We wrote %u unique labels\n",nlabels_local);

		LogMsg(VERB_NORMAL,"[SL3] We issued %u unique labels",issued);
LogFlush();
		//----------------------------------------------------------------------------
		// 3rd we build a global equivalence, and the canonical set
		// first OMP, then MPI
		//----------------------------------------------------------------------------

	std::vector<std::pair<unsigned, unsigned>> equiv;
	for (const auto& v : equiv_omp) {
      equiv.insert(equiv.end(), v.begin(), v.end());
  }
	// swap equiv_omp?

	LogMsg(VERB_HIGH,"[SL3] Assign dense labels local (OMP)]");LogFlush();
	// auto dense_map = assign_dense_labels(equiv, nlabels_local);
	auto dense_map = assign_dense_labels(equiv, issued);

//DEBUG LOCAL
// for (unsigned i=1;i<nlabels_local;i++){
// 	LogOut("label %u > denselabel %u\n",i,dense_map[i]);
// }

	unsigned dense = 0;
	for (unsigned i = 1; i <= issued; ++i)
		dense = std::max(dense, dense_map[i]);

	unsigned nlabels_local = dense;
	// old way
	// nlabels_local = dense_map[nlabels_local-1]; // number of non-zero labels

	LogMsg(VERB_HIGH,
	  "[SL3] local labels: issued=%u, dense=%u, dense_map[last=%u]=%u",
	  issued, dense, issued, issued ? dense_map[issued] : 0);LogFlush();

	// broadcast and get unique labels
	std::vector<int> nlabels_mpi(nMPI,0);
	std::vector<int> label_start_mpi(nMPI,0);

	commSync();

	int nlabels_local_i = (int)nlabels_local;
	MPI_Allgather(&nlabels_local_i, 1, MPI_INT,
                nlabels_mpi.data(), 1, MPI_INT, MPI_COMM_WORLD);

	for (int i=0; i < nMPI; i++){
		for (int j=0; j < i; j++){
			label_start_mpi[i] += nlabels_mpi[j];
		}
//DEBUG LOCAL
// LogOut("rank %d has %u labels and starts at %u\n",i,nlabels_mpi[i],label_start_mpi[i]+1);LogFlush();
	}

	// apply dense maps at each rank with MPI off-set
	#pragma omp parallel for
	for (size_t i = 0; i < ncubes_local; ++i) {
	size_t label_idx = Sf + idxCubeList[i];
	unsigned old_label = labelData[label_idx];
	unsigned new_label = label_start_mpi[rank] + dense_map[old_label]; // note that dense_map already starts at 1
	labelData[label_idx] = new_label;
	}

//DEBUG LOCAL
// {
// 	unsigned min_label = label_start_mpi[rank]+nlabels_mpi[rank], max_label =0;
// 	for (size_t idx = 0; idx < n3; ++idx) {
// 		unsigned labelo = labelData[idx+Sf];
// 		if (labelo == 0)
// 			continue;
// 		if (labelo > max_label)
// 			max_label = labelo;
// 		if (labelo < min_label)
// 			min_label = labelo;
// 	}
// 	printf("rank %d: min/max labels = %u,%u (expected %u,%u) \n",rank,
// 	min_label,max_label,label_start_mpi[rank]+1,label_start_mpi[rank]+nlabels_mpi[rank]);
// }

//MPI
	// ghost slices have now local dense labels

	commSync();


	const int sliceBytes = Sf*sizeof(unsigned int);
	void *sB = static_cast<void*>(labelData+Sf   );
	void *rF = static_cast<void*>(labelData+n3+Sf);
	void *sF = static_cast<void*>(labelData+n3   );
	void *rB = static_cast<void*>(labelData      );
	memset(rB,0,sliceBytes);
	memset(rF,0,sliceBytes);

	LogMsg(VERB_HIGH,"[SL3] send/wait ghosts") ;LogFlush();
	field->sendGeneral(COMM_SDRV, sliceBytes, MPI_BYTE, sB, rF, sF, rB);
	field->sendGeneral(COMM_WAIT, sliceBytes, MPI_BYTE, sB, rF, sF, rB);

	// build equivalences between labels in different ranks
	std::vector<std::vector<std::pair<unsigned, unsigned>>> equiv_omp_prev(nThreads);
	std::vector<std::vector<std::pair<unsigned, unsigned>>> equiv_omp_next(nThreads);

// if (commRank() == 0 ){
// 	for (size_t idx =0; idx<Sf; ++idx){
// 		if(labelData[idx+Sf])
// 		printf(">> >> rank %d has a label %u at %zu\n",rank,labelData[idx+Sf],idx+Sf);
// 		if(labelData[idx])
// 		printf(">> >> rank %d has a gabel %u at %zu\n",rank,labelData[idx],idx);
// }}
//


	commSync();
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int next_rank = (rank + 1)%nMPI;
		int prev_rank = (nMPI + rank - 1)%nMPI;
		int min_next_rank = label_start_mpi[next_rank]+1;
		int min_prev_rank = label_start_mpi[prev_rank]+1;
		int min_curr_rank = label_start_mpi[rank]+1;
		int max_next_rank = label_start_mpi[next_rank]+nlabels_mpi[next_rank];
		int max_prev_rank = label_start_mpi[prev_rank]+nlabels_mpi[prev_rank];
		int max_curr_rank = label_start_mpi[rank]+nlabels_mpi[rank];

	#pragma omp for
	for (size_t i = 0; i < ncubes_local; ++i) {
			size_t idx = idxCubeList[i]; // labelData has Sf ghosts!
			if ( (idx < Sf) ) // we look down at idx-Sf
			{
				if (labelData[idx] && (strData[idx] & STRING_XY)){ // ghost region = label from previous rank
					equiv_omp_prev[tid].push_back({labelData[idx],labelData[idx+Sf]});
//DEBUG
// if ( (labelData[idx] < min_prev_rank) ||
//   (labelData[idx] > max_prev_rank) ||
// 	 (labelData[idx+Sf] < min_curr_rank) ||
// 	 (labelData[idx+Sf] > max_curr_rank) )
// printf("rank %d (0): %u == %u (max min %u %u)\n",rank,labelData[idx],labelData[idx+Sf],min_label,max_label);

				}
			}
// 			if (idx > n3-Sf) // we look up at idx+Sf
// 			{
// 				// if (labelData[idx+2*Sf] && (strData[idx+Sf] & STRING_XY)){ // ghost region = label from next rank
// 				if (labelData[idx+2*Sf] ){ // ghost region = label from next rank
// 					equiv_omp_next[tid].push_back({labelData[idx+Sf],labelData[idx+2*Sf]});
// //DEBUG
// // if ( (labelData[idx+2*Sf] < min_next_rank) ||
// //   (labelData[idx+2*Sf] > max_next_rank) ||
// // 	 (labelData[idx+Sf] < min_curr_rank) ||
// // 	 (labelData[idx+Sf] > max_curr_rank) )
// printf("rank %d (Lz): %u == %u (max min %u %u)\n",rank,labelData[idx+2*Sf],labelData[idx+Sf],min_label,max_label);
// 				}
// 			}
	}
} // end parallel

	unsigned global_label_number = 0;
	for (int i = 0; i < nMPI;i++)
		global_label_number += nlabels_mpi[i];
LogMsg(VERB_NORMAL,"Global number of (redundant?) labels %u",global_label_number);


	commSync();

	// flatten OMP
	for (size_t i = 1; i < equiv_omp_prev.size(); ++i) {
	    equiv_omp_prev[0].insert(equiv_omp_prev[0].end(),
	                             equiv_omp_prev[i].begin(),
	                             equiv_omp_prev[i].end());
	}
	for (size_t i = 1; i < equiv_omp_next.size(); ++i) {
	    equiv_omp_next[0].insert(equiv_omp_next[0].end(),
	                             equiv_omp_next[i].begin(),
	                             equiv_omp_next[i].end());
	}
	// send the rank equivalence maps to root
	std::vector<std::pair<unsigned, unsigned>> global_equivs;
	if (commRank() == 0) {
	    global_equivs = gather_equivalences_to_root(equiv_omp_prev[0], equiv_omp_next[0], 0, MPI_COMM_WORLD);
	} else {
	    gather_equivalences_to_root(equiv_omp_prev[0], equiv_omp_next[0], 0, MPI_COMM_WORLD);
	}

	// root builds the global map
	std::vector<unsigned> global_map;


//DEBUG
// {
// 	unsigned seen_min = UINT_MAX, seen_max = 0;
// 	for (auto [a,b] : global_equivs) {
// 	    seen_min = std::min(seen_min, std::min(a,b));
// 	    seen_max = std::max(seen_max, std::max(a,b));
// 	}
// 	fprintf(stderr, "[rank %d] seen_min=%u seen_max=%u max_label=%u\n",
// 	        rank, seen_min, seen_max, max_label);
// }

	if (rank == 0) {
			global_map = assign_dense_labels(global_equivs, global_label_number);
	}

	// root distributes the global map
	global_map = broadcast_dense_label_vector_from_root(
		rank == 0 ? &global_map : nullptr, 0, MPI_COMM_WORLD);

	commSync();

// if (rank == 0){
// for (size_t label = 0; label < global_map.size(); ++label) {
//     LogOut("%zu → %u\n", label, global_map[label]);
// }
// }



	unsigned max_global_label = *std::max_element(global_map.begin(), global_map.end());

LogMsg(VERB_NORMAL,"[SL3] Total number of labels %d\n",max_global_label);
			// map is printed
	#pragma omp for
	for (size_t idx=0; idx<n3; ++idx){
		unsigned old_label = labelData[Sf+idx];
		if (old_label){
			unsigned new_label = global_map[old_label];
			labelData[Sf+idx] = new_label;
		}
	}

	commSync();

		//----------------------------------------------------------------------------
		// 4rd we calculate lengths, velocities, positions
		//----------------------------------------------------------------------------

		std::vector<double> string_len(max_global_label, 0.0);
		std::vector<double> string_vel(max_global_label, 0.0);
		std::vector<double> string_gam(max_global_label, 0.0);
		std::vector<double> string_cub(max_global_label, 0.0);

		const std::complex<Float>* m = static_cast<std::complex<Float>*>(field->mStart());
    const std::complex<Float>* v = static_cast<std::complex<Float>*>(field->vCpu());

		// we will need the backward v ghosts
		{
			const int sliceBytes = Sf*field->DataSize();
			void *sB = static_cast<void*>(field->vCpu());
			void *rF = static_cast<void*>(static_cast<char*>(field->vCpu())+n3*field->DataSize());
			void *sF = field->vBackGhost();
			void *rB = field->vBackGhost();
			memset(rF,0,sliceBytes);

			LogMsg(VERB_HIGH,"[SL3] send/wait v ghosts") ;
			field->sendGeneral(COMM_SDRV, sliceBytes, MPI_BYTE, sB, rF, sF, rB);
			field->sendGeneral(COMM_WAIT, sliceBytes, MPI_BYTE, sB, rF, sF, rB);
		}

		LogMsg(VERB_HIGH,"[SL3] Define constants and calculate lengths,velocities, etc...") ;
		Float c   = (Float) 0.41238;
		Float ms2 = (Float) field->SaxionMassSq();
		Float Rscale = (Float) *field->RV();
		Float Hc  = (Float) field->HubbleConformal();

		Float inv_a     = Float(1.0) / Rscale;
		Float inv_a2    = inv_a * inv_a;
		Float scale_phi = inv_a;
		Float scale_vel = inv_a2;

		/* segments */
		std::vector<SegRec> segs_all;
		segs_all.reserve(4 * ncubes_local);  // heuristic; ok to overshoot

		/* z baseline for absolute coordinates */
		Float local_z = (Float) (rank*Lz);
		#pragma omp parallel
    {
        std::vector<double> threadLengths(max_global_label, 0.0);
        std::vector<double> threadVelocities(max_global_label, 0.0);
				std::vector<double> threadGammas(max_global_label, 0.0);
				std::vector<double> threadCubes(max_global_label, 0.0);

				std::vector<SegRec> segs_thr;
				segs_thr.reserve(256);

        #pragma omp for
        for (size_t i = 0; i < ncubes_local; ++i) {
            size_t idx = idxCubeList[i];
            unsigned label = labelData[idx+Sf]; // labelData ghosted
            if (label == 0) continue; // should not happen

            size_t X[3];
            indexXeon::idx2Vec(idx, X, Lx);
            size_t ix = X[0], iy = X[1], iz = X[2];

            auto idx3D = [&](int dx, int dy, int dz) {
                return ((ix + dx + Lx) % Lx) + ((iy + dy + Lx) % Lx) * Lx + (iz + dz) * Sf;
            };

            size_t idx000 = idx3D(0, 0, 0);
            size_t idx001 = idx3D(0, 0, 1);
            size_t idx010 = idx3D(0, 1, 0);
            size_t idx011 = idx3D(0, 1, 1);
            size_t idx100 = idx3D(1, 0, 0);
            size_t idx101 = idx3D(1, 0, 1);
            size_t idx110 = idx3D(1, 1, 0);
            size_t idx111 = idx3D(1, 1, 1);

            std::complex<Float> m000 = m[idx000], m100 = m[idx100], m010 = m[idx010], m110 = m[idx110];
            std::complex<Float> m001 = m[idx001], m101 = m[idx101], m011 = m[idx011], m111 = m[idx111];

            std::complex<Float> v000 = v[idx000], v100 = v[idx100], v010 = v[idx010], v110 = v[idx110];
            std::complex<Float> v001 = v[idx001], v101 = v[idx101], v011 = v[idx011], v111 = v[idx111];

						// convert to non-conformal fields
						for (auto [mp, vp] : std::initializer_list<std::pair<std::complex<Float>*, std::complex<Float>*>>{
						        {&m000, &v000}, {&m001, &v001}, {&m010, &v010}, {&m011, &v011},
						        {&m100, &v100}, {&m101, &v101}, {&m110, &v110}, {&m111, &v111}
						    })
						{
						    *vp = scale_vel * (*v - Hc * (*m));
								*mp *= scale_phi;
						}

            Float total_vg2 = 0.0;
            int np = 0;
            Float pos_x[6] = {0}, pos_y[6] = {0}, pos_z[6] = {0};
						uint64_t pkey[6];
						uint8_t  isExit[6];

						// --- helper: register one point + metadata ---
						auto push_point = [&](Float px, Float py, Float pz, unsigned short faceFlag){
							pos_x[np] = px;
							pos_y[np] = py;
							pos_z[np] = pz;

							// unique plaquette id across ranks: maps *2 faces to neighbor cell
							pkey[np]  = canonical_plaq_key((usi)ix,(usi)iy,(usi)(local_z+iz),
							                               faceFlag, (usi)Lx, (usi)Tz);

							// classify OUT vs IN for THIS cube
							isExit[np] = ( (scOutData[idx] & faceFlag) ? 1 : 0 );

							++np;
						};

						auto collect = [&](std::complex<Float> m00, std::complex<Float> m10,
						                   std::complex<Float> m11, std::complex<Float> m01,
						                   std::complex<Float> v00, std::complex<Float> v10,
						                   std::complex<Float> v11, std::complex<Float> v01,
						                   Float origin_x, Float origin_y, Float origin_z,
						                   unsigned short faceflag, int orientation) {
						    if (!((scOutData[idx] | scInData[idx]) & faceflag)) return;

						    Float du[2], vg2 = 0.0;
						    set_cross_and_velocity<Float>(m00, m10, m11, m01,
						                           v00, v10, v11, v01,
						                           du, vg2, ms2, c);
								Float px, py, pz;
						    switch (orientation) {
						        case 0: // XY: u→x, v→y
						            px = origin_x + du[0];
						            py = origin_y + du[1];
						            pz = origin_z;
						            break;
						        case 1: // YZ: u→y, v→z
						            px = origin_x;
						            py = origin_y + du[0];
						            pz = origin_z + du[1];
						            break;
						        case 2: // ZX: u→z, v→x
						            px = origin_x + du[1];
						            py = origin_y;
						            pz = origin_z + du[0];
						            break;
						    }

						    total_vg2 += vg2;
								push_point(px, py, pz, faceflag);
						};
						// We do not wrap in the coordinates, or distances get unphysical factors of Lx,Lz
						// when collecting points for string positions, we need to avoid printing
						// points in the plaquettes belonging to other ranking to avoid redundancy
            collect(m000, m100, m110, m010, v000, v100, v110, v010, ix, iy, iz, STRCUB_XY,0);
            collect(m000, m010, m011, m001, v000, v010, v011, v001, ix, iy, iz, STRCUB_YZ,1);
            collect(m000, m001, m101, m100, v000, v001, v101, v100, ix, iy, iz, STRCUB_ZX,2);
            collect(m001, m101, m111, m011, v001, v101, v111, v011, ix, iy, iz+1, STRCUB_XY2,0);
            collect(m100, m110, m111, m101, v100, v110, v111, v101, ix+1, iy, iz, STRCUB_YZ2,1);
            collect(m010, m011, m111, m110, v010, v011, v111, v110, ix, iy+1, iz, STRCUB_ZX2,2);


            if (np >= 2) {

							int exits_idx[6], enters_idx[6], nE=0, nI=0;
							for (int k=0; k<np; ++k) {
								if (isExit[k]) exits_idx[nE++] = k;
								else           enters_idx[nI++] = k;
							}
							const int pairs = std::min(nE, nI);  // should be np/2

							for (int t=0; t<pairs; ++t) {
								const int a = exits_idx[t];
								const int b = enters_idx[t];

								SegRec s;
								s.label = label;
								s.a_key = pkey[a];
								s.b_key = pkey[b];
								s.ax = wrapf(pos_x[a], (double)Lx);
								s.ay = wrapf(pos_y[a], (double)Lx);
								s.az = wrapf(pos_z[a]+(Float)local_z, (double)Tz);
								s.bx = wrapf(pos_x[b], (double)Lx);
								s.by = wrapf(pos_y[b], (double)Lx);
								s.bz = wrapf(pos_z[b]+(Float)local_z, (double)Tz);
								segs_thr.push_back(s);
							}


							Float dl = dl_cal<Float>(pos_x, pos_y, pos_z, np);
              threadLengths[label - 1] += dl;
// printf("tl %d = %f (dl %f)\n",label,threadLengths[label - 1], dl);
							total_vg2 = total_vg2/np; //point average
							double vel   = std::sqrt(total_vg2/(1+total_vg2));
							double gamma = std::sqrt(1+total_vg2);
              threadVelocities[label - 1] += vel;
							threadGammas[label - 1]  += gamma;
							threadCubes[label-1]     += 1;
// printf("vel_out %f/%f/%f [%d %d %d] (label %d)\n",vel,threadVelocities[label - 1],threadCubes[label-1],ix,iy,iz,label);
            }
        }

				#pragma omp critical
				{
				for (size_t l = 0; l < max_global_label; ++l) {
					string_len[l] += threadLengths[l];
					string_vel[l] += threadVelocities[l];
					string_gam[l] += threadGammas[l];
					string_cub[l] += threadCubes[l];
				}
				segs_all.insert(segs_all.end(), segs_thr.begin(), segs_thr.end());
				}
    }
// for (size_t l = 0; l < max_global_label; ++l)
// 	printf("l,v,g,c %f %f %f %f (label %d)\n",string_len[l],string_vel[l],string_gam[l],string_cub[l],l);

		commSync();

		LogMsg(VERB_HIGH,"[SL3] Reductions") ;

// //DEBUG
// for (size_t i = 0; i < string_len.size(); ++i) {
//         std::cout << "rank " << rank << " Label " << std::setw(4) << i << ": " << string_len[i] << '\n';
//     }

		mpi_sum_vector_inplace(string_len);
		mpi_sum_vector_inplace(string_vel);
		mpi_sum_vector_inplace(string_gam);
		mpi_sum_vector_inplace(string_cub);
// for (size_t l = 0; l < max_global_label; ++l)
// 	printf("r%d - l,v,g,c %f %f %f %f (label %zu/%zu)\n",rank,
// 	string_len[l],string_vel[l],string_gam[l],string_cub[l],l,max_global_label);

		//------------------------------------------------------------
		// 4.1) Ownership
		//------------------------------------------------------------

		// compute local counts per label present on this rank
		std::vector<int> cnt_local(max_global_label, 0);
		for (const SegRec& s : segs_all) {
		    size_t L = s.label - 1;
		    if (L < cnt_local.size()) cnt_local[L] += 1;
		}

		// choose owner per label with MAXLOC
		struct { int val; int rank; } local_pair, owner_pair;
		std::vector<decltype(local_pair)> local_pairs(max_global_label), owner_pairs(max_global_label);


		for (size_t i=0;i<max_global_label;++i) { local_pairs[i].val = cnt_local[i]; local_pairs[i].rank = rank; }

		MPI_Allreduce(local_pairs.data(), owner_pairs.data(),
		              (int)max_global_label, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

		auto owner_of = [&](uint32_t label)->int { return owner_pairs[label-1].rank; };

		//------------------------------------------------------------
		// 4.1) Redistribute segments so each string label is owned by 1 rank
		//------------------------------------------------------------
		commSync();

		// Build send buffers grouped by destination rank
		std::vector<std::vector<SegRec>> sendBuf(nMPI);
		for (const SegRec &s : segs_all) {
		    int dst = owner_of(s.label);
		    sendBuf[dst].push_back(s);
		}

		// Prepare recv counts
		std::vector<int> sendCounts(nMPI), recvCounts(nMPI);
		for (int r = 0; r < nMPI; r++)
		    sendCounts[r] = (int)sendBuf[r].size();

		MPI_Alltoall(sendCounts.data(), 1, MPI_INT,
		             recvCounts.data(), 1, MPI_INT,
		             MPI_COMM_WORLD);

		// Compute displacements and total receive count
		std::vector<int> sendDisp(nMPI), recvDisp(nMPI);
		int totalSend = 0, totalRecv = 0;
		for (int r = 0; r < nMPI; r++) {
		    sendDisp[r] = totalSend;
		    recvDisp[r] = totalRecv;
		    totalSend += sendCounts[r];
		    totalRecv += recvCounts[r];
		}

		// Flatten
		std::vector<SegRec> sendFlat; sendFlat.reserve(totalSend);
		for (int r=0; r<nMPI; ++r) sendFlat.insert(sendFlat.end(), sendBuf[r].begin(), sendBuf[r].end());

		// Convert to bytes
		std::vector<int> sendCountsB(nMPI), recvCountsB(nMPI), sendDispB(nMPI), recvDispB(nMPI);
		for (int r=0; r<nMPI; ++r) {
		    sendCountsB[r] = sendCounts[r] * (int)sizeof(SegRec);
		    recvCountsB[r] = recvCounts[r] * (int)sizeof(SegRec);
		    sendDispB[r]   = sendDisp[r]   * (int)sizeof(SegRec);
		    recvDispB[r]   = recvDisp[r]   * (int)sizeof(SegRec);
		}

		std::vector<SegRec> recvFlat(totalRecv);
		MPI_Alltoallv(sendFlat.data(), sendCountsB.data(), sendDispB.data(), MPI_BYTE,
		              recvFlat.data(), recvCountsB.data(), recvDispB.data(), MPI_BYTE,
		              MPI_COMM_WORLD);

		static_assert(std::is_trivially_copyable<SegRec>::value, "SegRec must be POD for MPI_BYTE send.");

		// Now recvFlat contains **only** the segments whose label is owned by *this* rank
		segs_all.swap(recvFlat);   // replace local storage
		sendFlat.clear();
		sendBuf.clear();

		//=============================================================
		// 4.2) Stitch segments into ordered polylines per label
		// and compute stuff
		//=============================================================

		// ordered the segments by label
		std::unordered_map<usi, std::vector<SegRec>> segs_by_label;
		segs_by_label.reserve(segs_all.size());
		for (auto &s : segs_all) segs_by_label[s.label].push_back(s);

		// stable list of labels owned here
		std::vector<usi> labels_owned; labels_owned.reserve(segs_by_label.size());
		for (auto &kv : segs_by_label) labels_owned.push_back(kv.first);
		std::sort(labels_owned.begin(), labels_owned.end());

		const int N = (int)labels_owned.size();
		// --- sizes & offsets (prefix sum) ---
		std::vector<unsigned> sizes(N, 0);
		std::vector<unsigned> offsets(N+1, 0); // NOTE: N+1, offsets[0]=0

		for (int i = 0; i < N; ++i) {
		    const usi L = labels_owned[i];
		    sizes[i]     = (unsigned) segs_by_label[L].size(); // points == segments
		    offsets[i+1] = offsets[i] + sizes[i];
		}

		const unsigned M = offsets.back(); // total number of points
		// --- pre-size slp outputs (no push_backs later) ---
		slp.loop_labels.resize(N);
		slp.loop_sizes .resize(N);
		slp.loop_offsets.resize(N+1);
		slp.loop_closed.resize(N);
		slp.loop_origin.resize(3* (size_t)N);
		slp.loop_coords.resize(3* (size_t)M);

		slp.loop_com.resize(3*N);
		slp.loop_inertia.resize(6*N);
		slp.loop_inertia_eigs.resize(3*N);
		slp.loop_len_com.resize(N);

		// copy offsets
		for (int i=0;i<=N;++i) slp.loop_offsets[i] = offsets[i];

		// --- fill per loop in parallel ---
		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < N; ++i) {
			const usi L = labels_owned[i];
			auto &segv  = segs_by_label[L];

		// stitch into a polyline
		// result: vectors px, py, pz with length sizes[i], and a bool isClosed
			std::vector<double> px, py, pz;
			bool isClosed = false;
			{
			std::unordered_map<uint64_t,uint64_t> next_of;
			std::unordered_map<uint64_t,std::array<double,3>> coord_of;
			std::unordered_set<uint64_t> in_nodes;
			px.reserve(segv.size()); py.reserve(segv.size()); pz.reserve(segv.size());

			for (auto &s : segv) {
				next_of[s.a_key]  = s.b_key;
				coord_of[s.a_key] = {s.ax,s.ay,s.az};
				coord_of[s.b_key] = {s.bx,s.by,s.bz};
				in_nodes.insert(s.b_key);
			}
			uint64_t start = 0;
			for (auto &s : segv)
				if (!in_nodes.count(s.a_key)) { start = s.a_key; break; }
			if (start == 0 && !segv.empty()) start = segv[0].a_key;

			std::unordered_set<uint64_t> vis;
			uint64_t cur = start;
			while (coord_of.count(cur) && !vis.count(cur) && px.size() < segv.size())
				{
					vis.insert(cur);
					auto &p = coord_of[cur];
					px.push_back(p[0]); py.push_back(p[1]); pz.push_back(p[2]);
					auto it = next_of.find(cur);
					if (it == next_of.end())
						break;
					if (it->second == start) {
						isClosed = true; break;
					}
					cur = it->second;
				}
			}

		// write label, size, origin
		slp.loop_labels[i] = L;
		slp.loop_sizes[i]  = (unsigned)px.size();
		slp.loop_origin[3*i+0] = px.empty()?0.0:px[0];
		slp.loop_origin[3*i+1] = py.empty()?0.0:py[0];
		slp.loop_origin[3*i+2] = pz.empty()?0.0:pz[0];

		// unwrapped for COM/inertia + winding
		if (px.size() >= 2) {
			std::vector<V3> U(px.size());
			V3 Wprev{px[0],py[0],pz[0]}, uacc{0,0,0};
			U[0] = uacc;
			for (size_t k=1;k<px.size();++k) {
				V3 Wk{px[k],py[k],pz[k]};
				V3 du{
				mindelta(Wk.x-Wprev.x,(double)Lx),
				mindelta(Wk.y-Wprev.y,(double)Lx),
				mindelta(Wk.z-Wprev.z,(double)Tz)
				};
				uacc = {uacc.x+du.x, uacc.y+du.y, uacc.z+du.z};
				U[k] = uacc; Wprev = Wk;
			}

			// winding bitmask (0/1/2/4) from end-start
			auto rn = [](double t)->int { return (int)llround(t); };
			const int wx = rn((U.back().x - U.front().x)/ (double)Lx);
			const int wy = rn((U.back().y - U.front().y)/ (double)Lx);
			const int wz = rn((U.back().z - U.front().z)/ (double)Tz);
			uint8_t mask = 0; if (wx) mask|=1; if (wy) mask|=2; if (wz) mask|=4;
			slp.loop_closed[i] = mask;

			Acc A{};
			for (size_t k=0;k+1< U.size();++k)
				accum_seg(A, U[k], U[k+1]);
			if (isClosed) accum_seg(A, U.back(), U.front());
				V3 R; double I6[6];
			if (A.L>0)
				finalize(A, R, I6);
			else {
				R={0,0,0}; for(double&q:I6) q=0;
			}

			// slp.loop_len_com[i] = A.L;
			// slp.loop_com[i][0]=R.x;
			// 	slp.loop_com[i][1]=R.y;
			// 		slp.loop_com[i][2]=R.z;
			// slp.loop_inertia[i][0]=I6[0]; slp.loop_inertia[i][1]=I6[1]; slp.loop_inertia[i][2]=I6[2];
			// slp.loop_inertia[i][3]=I6[3]; slp.loop_inertia[i][4]=I6[4]; slp.loop_inertia[i][5]=I6[5];
			slp.loop_len_com[i] = A.L;
			slp.loop_com[3*i+0]=R.x;
				slp.loop_com[3*i+1]=R.y;
					slp.loop_com[3*i+2]=R.z;
			slp.loop_inertia[6*i+0]=I6[0]; slp.loop_inertia[6*i+1]=I6[1]; slp.loop_inertia[6*i+2]=I6[2];
			slp.loop_inertia[6*i+3]=I6[3]; slp.loop_inertia[6*i+4]=I6[4]; slp.loop_inertia[6*i+5]=I6[5];
			double evals[3];
			inertia_principal_eigs(I6[0],I6[1],I6[2],I6[3],I6[4],I6[5],evals);
			// slp.loop_inertia_eigs[i] = { std::sqrt(2*evals[0]/A.L),
			// 															std::sqrt(2*evals[1]/A.L),
			// 																std::sqrt(2*evals[2]/A.L)}; // ascending
			slp.loop_inertia_eigs[3*i  ] = std::sqrt(2*evals[0]/A.L);
			slp.loop_inertia_eigs[3*i+1] = std::sqrt(2*evals[1]/A.L);
			slp.loop_inertia_eigs[3*i+2] = std::sqrt(2*evals[2]/A.L);

			// write coordinates at their slice
			const unsigned base = offsets[i];
			double *C = slp.loop_coords.data() + 3*base;
			for (unsigned k=0; k<px.size(); ++k) {
				if ((int) mask == 0){
				C[3*k+0] = px[0]+U[k].x;
				C[3*k+1] = py[0]+U[k].y;
				C[3*k+2] = pz[0]+U[k].z;
			} else {
				C[3*k+0] = px[k];
				C[3*k+1] = py[k];
				C[3*k+2] = pz[k];
				}
			}

		}
		else  // px.size() 1or2
		{
		slp.loop_closed[i] = 0;
		slp.loop_len_com[i] = 0.0;
		slp.loop_com[3*i] = 0;
		slp.loop_com[3*i+1] = 0;
		slp.loop_com[3*i+2] = 0;
		slp.loop_inertia[6*i+0] = 0;
		slp.loop_inertia[6*i+1] = 0;
		slp.loop_inertia[6*i+2] = 0;
		slp.loop_inertia[6*i+3] = 0;
		slp.loop_inertia[6*i+4] = 0;
		slp.loop_inertia[6*i+5] = 0;
		}
		}

		//----------------------------------------------------------------------------
		// 5rd we print the label map, write Loop data
		//----------------------------------------------------------------------------

		commSync();

		memmove(field->m2Cpu(),static_cast<void*>(labelData+Sf),n3*sizeof(unsigned));
		field->setM2(M2_LABEL_MAP);

    commSync();

		// we calculate totals
		double total_len =0, total_vel =0, total_gam =0, total_cub =0;
		for (size_t i=0; i<max_global_label; ++i){
			total_len += string_len[i];
			total_vel += string_vel[i];
			total_gam += string_gam[i];
			total_cub += string_cub[i];
		}

	// printf("r%d - l,v,g,c %f %f %f %f (labels all)\n",rank,
	// total_len,total_vel,total_gam,total_cub,max_global_label);

		// save loop data
		slp.len = string_len;
		slp.vel = string_vel;
		slp.gam = string_gam;
		slp.cub = string_cub;

		// I try to match Kenichis definitions
// if (commRank()==0)
// {
// 	printf("total length %f\n",strDen.strLen);
// 	printf("av gamma %f\n",strDen.strDeng);
// 	printf("av vel %f\n",strDen.strVel);
// 	printf("strGamm %f\n",strDen.strGam);
// }
		strDen.strLen = total_len;
		strDen.strDeng = total_gam/total_cub;
		strDen.strVel = total_vel/total_cub;
		// we do not need these but can be calculated
		strDen.strVel2 = 0.;
		strDen.strGam = total_gam/total_cub;
		strDen.strLen_local = 0.;
		strDen.strDeng_local = 0.;

		slp.stringdata = strDen;

// if (commRank()==0)
// {
// 	printf("total length %f\n",slp.stringdata.strLen);
// 	printf("av gamma %f\n",slp.stringdata.strDeng);
// 	printf("av vel %f\n",slp.stringdata.strVel);
// 	printf("strGamm %f\n",slp.stringdata.strGam);
// }

	/*some debugging prints*/
	if (0)
	{
		if (rank==0)
		{
			printf("rank 0 prints %d loops \n",slp.len.size());
				for (int d=0; d<slp.len.size();d++){
					printf("global label %d len %lf vel %lf gam %lf cub %lf\n",1+d,slp.len[d],slp.vel[d]/slp.cub[d],slp.gam[d]/slp.cub[d],slp.cub[d]);
				}
		}
		commSync();
		for (int ran = 0 ; ran < nMPI; ran++)
		{
		if (rank == ran){
			const size_t K = slp.loop_labels.size();
			printf("rank %d printing ........................ \n", rank);
			for (size_t k = 0; k < K; ++k) {
			    unsigned short lab = slp.loop_labels[k];
			    unsigned off0 = slp.loop_offsets[k];
			    unsigned off1 = slp.loop_offsets[k+1];
			    unsigned sz   = off1 - off0;
			    printf("loop %zu label %hu\n", k, lab);
					printf("         size  %u\n", sz);
					printf("         len  %.1f estimate R %.1f\n", slp.loop_len_com[k],slp.loop_len_com[k]/(6.28));
			    printf("         offset  %u\n", off0);
			    printf("         com   %lf, %lf, %lf\n",
			           slp.loop_com[3*k], slp.loop_com[3*k+1], slp.loop_com[3*k+2]);
					printf("         inertia  %lf %lf %lf %lf %lf %lf \n",
					          slp.loop_inertia[6*k+0],slp.loop_inertia[6*k+1],slp.loop_inertia[6*k+2],
											slp.loop_inertia[6*k+3],slp.loop_inertia[6*k+4],slp.loop_inertia[6*k+5]);
					printf("         wrap  %d\n", slp.loop_closed[k]);
					printf("         eig(I)    λ1=%d λ2=%d λ3=%d\n",
			       (int)slp.loop_inertia_eigs[3*k], (int)slp.loop_inertia_eigs[3*k+1], (int)slp.loop_inertia_eigs[3*k+2]);
					if (0)
					for (uint64_t i = off0; i < off1; ++i) {
					    const double x = slp.loop_coords[3*i+0];
					    const double y = slp.loop_coords[3*i+1];
					    const double z = slp.loop_coords[3*i+2];
					    printf("           p[%lu] = (%.1f, %.1f, %.1f)\n", i, x, y, z);
					}


			}
		}
		}
	}

	commSync();
	//----------------------------------------------------------------------------
	// 6rd we store all data in rank0
	//----------------------------------------------------------------------------



	return	slp;
}







#ifdef USE_2DCYL
template<typename Float>
StringLoopParms stringlength3_2D(Scalar *field, StringData strDen)
{
	LogMsg(VERB_NORMAL, "[SL2] cylindrical string coordinates");

	StringLoopParms slp;
	slp.stringdata = strDen;

	if (!(field->sDStatus() & SD_MAP) || strDen.strDen == 0)
		return slp;

	const bool wasFolded = field->Folded();
	if (wasFolded) {
		Folder unfold(field);
		unfold(UNFOLD_ALL);
	}
	field->exchangeGhosts(FIELD_M);

	const size_t NzAxis = field->Length();
	const size_t Nrho   = field->Depth();
	const size_t Ng     = field->getNg();
	const size_t rho0   = static_cast<size_t>(commRank())*Nrho;
	const bool hasRhoNext = commRank() + 1 < commSize();
	const size_t rhoPlaquettes = Nrho - (hasRhoNext ? 0 : 1);
	const char *map     = static_cast<const char *>(field->sData());
	const auto *base    = static_cast<const std::complex<Float> *>(field->mCpu());
	const auto *m       = base + Ng*NzAxis;
	auto *v             = static_cast<std::complex<Float> *>(field->vCpu());

	/* SAXION v is stored without a front ghost.  Exchange the final radial
	 * row into the back-ghost slot, as in the 3D string-velocity path. */
	if (hasRhoNext) {
		const int rowBytes = static_cast<int>(NzAxis*field->DataSize());
		void *sendFirst = field->vCpu();
		void *recvNext  = static_cast<void *>(
			static_cast<char *>(field->vCpu()) + field->Size()*field->DataSize());
		void *sendLast  = static_cast<void *>(
			static_cast<char *>(field->vCpu()) + (field->Size()-NzAxis)*field->DataSize());
		field->sendGeneral(COMM_SDRV, rowBytes, MPI_BYTE,
		                   sendFirst, recvNext, sendLast, sendLast);
		field->sendGeneral(COMM_WAIT, rowBytes, MPI_BYTE,
		                   sendFirst, recvNext, sendLast, sendLast);
	}

	const Float c = Float(0.41238);
	const Float ms2 = static_cast<Float>(field->SaxionMassSq());
	const Float scalePhi = Float(1)/static_cast<Float>(*field->RV());
	const Float scaleVel = scalePhi*scalePhi;
	const Float Hc = static_cast<Float>(field->HubbleConformal());

	for (size_t ir = 0; ir < rhoPlaquettes; ++ir) {
		for (size_t iz = 0; iz + 1 < NzAxis; ++iz) {
			if (!(map[ir*NzAxis + iz] & STRING_ZX))
				continue;

			std::complex<Float> m00, m10, m11, m01;
			std::complex<Float> v00, v10, v11, v01;
			if (iz == 0) {
				m00 = std::conj(m[ ir   *NzAxis + 1]);
				m10 = std::conj(m[(ir+1)*NzAxis + 1]);
				m11 =           m[(ir+1)*NzAxis + 1];
				m01 =           m[ ir   *NzAxis + 1];
				v00 = std::conj(v[ ir   *NzAxis + 1]);
				v10 = std::conj(v[(ir+1)*NzAxis + 1]);
				v11 =           v[(ir+1)*NzAxis + 1];
				v01 =           v[ ir   *NzAxis + 1];
			} else {
				m00 = m[ ir   *NzAxis + iz    ];
				m10 = m[(ir+1)*NzAxis + iz    ];
				m11 = m[(ir+1)*NzAxis + iz + 1];
				m01 = m[ ir   *NzAxis + iz + 1];
				v00 = v[ ir   *NzAxis + iz    ];
				v10 = v[(ir+1)*NzAxis + iz    ];
				v11 = v[(ir+1)*NzAxis + iz + 1];
				v01 = v[ ir   *NzAxis + iz + 1];
			}

			for (auto pair : {std::pair{&m00,&v00}, std::pair{&m10,&v10},
			                  std::pair{&m11,&v11}, std::pair{&m01,&v01}}) {
				*pair.second = scaleVel*(*pair.second - Hc*(*pair.first));
				*pair.first *= scalePhi;
			}

			Float du[2], vgamma2 = 0;
			set_cross_and_velocity(m00, m10, m11, m01,
			                       v00, v10, v11, v01,
			                       du, vgamma2, ms2, c);
			const double velocity = std::sqrt(vgamma2/(Float(1) + vgamma2));
			const double gamma = std::sqrt(Float(1) + vgamma2);
			const double rho = static_cast<double>(rho0 + ir) + du[0];
			double z = (iz == 0) ? 2.0*du[1] - 1.0
			                         : static_cast<double>(iz) + du[1];
			if (std::abs(z) < 1.e-12)
				z = 0.;

			const double length = 2.0*std::acos(-1.0)*rho;
			slp.len.push_back(length);
			slp.vel.push_back(velocity);
			slp.gam.push_back(gamma);
			slp.cub.push_back(1.);
			slp.loop_sizes.push_back(1);
			slp.loop_closed.push_back(1);
			slp.loop_chiralities.push_back(
				(map[ir*NzAxis + iz] & STRING_ZX_POSITIVE) ? int8_t(1) : int8_t(-1));
			slp.loop_len_com.push_back(length);
			slp.loop_com.insert(slp.loop_com.end(), {z, 0., rho});
			slp.loop_origin.insert(slp.loop_origin.end(), {z, 0., rho});
			slp.loop_coords.insert(slp.loop_coords.end(), {z, 0., rho});
			slp.loop_inertia.insert(slp.loop_inertia.end(), 6, 0.);
			slp.loop_inertia_eigs.insert(slp.loop_inertia_eigs.end(), 3, 0.);
		}
	}

	const uint64_t nLocal = slp.loop_sizes.size();
	uint64_t labelOffset = 0;
	MPI_Exscan(&nLocal, &labelOffset, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
	if (commRank() == 0)
		labelOffset = 0;
	slp.loop_offsets.resize(nLocal + 1);
	for (uint64_t i = 0; i < nLocal; ++i) {
		slp.loop_labels.push_back(static_cast<uint32_t>(labelOffset + i + 1));
		slp.loop_offsets[i] = i;
	}
	slp.loop_offsets[nLocal] = nLocal;

	double localLength = 0.;
	double localWeightedVelocity = 0.;
	double localWeightedGamma = 0.;
	for (double length : slp.len)
		localLength += length;
	for (size_t i = 0; i < slp.len.size(); ++i) {
		localWeightedVelocity += slp.len[i]*slp.vel[i];
		localWeightedGamma += slp.len[i]*slp.gam[i];
	}
	double localTotals[3] = {localLength, localWeightedVelocity, localWeightedGamma};
	double globalTotals[3] = {};
	MPI_Allreduce(localTotals, globalTotals, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	slp.stringdata.strLen_local = localLength;
	slp.stringdata.strLen = globalTotals[0];
	if (globalTotals[0] > 0.) {
		slp.stringdata.strVel = globalTotals[1]/globalTotals[0];
		slp.stringdata.strGam = globalTotals[2]/globalTotals[0];
		slp.stringdata.strDeng = slp.stringdata.strGam;
	}

	if (wasFolded) {
		Folder refold(field);
		refold(FOLD_ALL);
	}

	return slp;
}
#endif

StringLoopParms stringlength3 (Scalar *field, StringData strDen_in, StringMeasureType strmeas)
{

	Profiler &prof = getProfiler(PROF_STRINGLENGTH3);
	prof.start();

	StringLoopParms slp;
#ifdef USE_2DCYL
	if (field->Precision() == FIELD_SINGLE)
		slp = stringlength3_2D<float>(field, strDen_in);
	else
		slp = stringlength3_2D<double>(field, strDen_in);
#else
	if (field->Precision() == FIELD_SINGLE)
	{
		slp = stringlength3<float> (field, strDen_in, strmeas);
	}
	else
	{
		slp = stringlength3<double>(field, strDen_in, strmeas);
	}
#endif

	prof.stop();
	prof.add("String Length 3",0,0);

	return slp;
}
