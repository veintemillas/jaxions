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
	size_t n3 = field->Size();
	size_t Sf = Lx*Lx;
	int rank = commRank();

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

        unsigned label = nextLabel;
				nextLabel++;         // get a unique label 1,2...

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

		size_t nlabels_local = nextLabel;
		LogMsg(VERB_NORMAL,"[SL3] We wrote %u unique labels",nlabels_local);
// LogOut	("[SL3] We wrote %u unique labels\n",nlabels_local);


		//----------------------------------------------------------------------------
		// 3rd we build a global equivalence, and the canonical set
		// first OMP, then MPI
		//----------------------------------------------------------------------------

	std::vector<std::pair<unsigned, unsigned>> equiv;
	for (const auto& v : equiv_omp) {
      equiv.insert(equiv.end(), v.begin(), v.end());
  }
	// swap equiv_omp?

	auto dense_map = assign_dense_labels(equiv, nlabels_local);

//DEBUG LOCAL
// for (unsigned i=1;i<nlabels_local;i++){
// 	LogOut("label %u > denselabel %u\n",i,dense_map[i]);
// }
	nlabels_local = dense_map[nlabels_local-1]; // number of non-zero labels

	// broadcast and get unique labels
	std::vector<int> nlabels_mpi(commSize(),0);
	std::vector<int> label_start_mpi(commSize(),0);

	commSync();

	MPI_Allgather(&nlabels_local, 1, MPI_INT,
                nlabels_mpi.data(), 1, MPI_INT, MPI_COMM_WORLD);


	for (int i=0; i < commSize(); i++){
		for (int j=0; j < i; j++){
			label_start_mpi[i] += nlabels_mpi[j];
		}
//DEBUG LOCAL
	// LogOut("rank %d has %u labels and starts at %u\n",i,nlabels_mpi[i],label_start_mpi[i]);
	}

	// apply dense maps at each rank with MPI off-set
	#pragma omp parallel for
	for (size_t i = 0; i < ncubes_local; ++i) {
	size_t label_idx = Sf + idxCubeList[i];
	unsigned old_label = labelData[label_idx];
	unsigned new_label = label_start_mpi[rank] + dense_map[old_label];
	labelData[label_idx] = new_label;
	}

//DEBUG LOCAL


	// unsigned min_label = label_start_mpi[rank]+nlabels_mpi[rank], max_label =0;
	// for (size_t idx = 0; idx < n3; ++idx) {
	// 	unsigned labelo = labelData[idx+Sf];
	// 	if (labelo == 0)
	// 		continue;
	// 	if (labelo > max_label)
	// 		max_label = labelo;
	// 	if (labelo < min_label)
	// 		min_label = labelo;
	// }
	// printf("rank %d: min/max labels = %u,%u (expected %u,%u) \n",rank,
	// min_label,max_label,label_start_mpi[rank]+1,label_start_mpi[rank]+nlabels_mpi[rank]);

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

	LogMsg(VERB_HIGH,"[SL3] send/wait ghosts") ;
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
		int next_rank = (rank + 1)%commSize();
		int prev_rank = (commSize() + rank - 1)%commSize();
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
	for (int i; i < commSize();i++)
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

		#pragma omp parallel
    {
        std::vector<double> threadLengths(max_global_label, 0.0);
        std::vector<double> threadVelocities(max_global_label, 0.0);
				std::vector<double> threadGammas(max_global_label, 0.0);
				std::vector<double> threadCubes(max_global_label, 0.0);

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
						for (auto [m, v] : std::initializer_list<std::pair<std::complex<Float>*, std::complex<Float>*>>{
						        {&m000, &v000}, {&m001, &v001}, {&m010, &v010}, {&m011, &v011},
						        {&m100, &v100}, {&m101, &v101}, {&m110, &v110}, {&m111, &v111}
						    })
						{
						    *v = scale_vel * (*v - Hc * (*m));
								*m *= scale_phi;
						}

            Float total_vg2 = 0.0;
            int np = 0;
            Float pos_x[6] = {0}, pos_y[6] = {0}, pos_z[6] = {0};

						auto collect = [&](std::complex<Float> m00, std::complex<Float> m10,
						                   std::complex<Float> m11, std::complex<Float> m01,
						                   std::complex<Float> v00, std::complex<Float> v10,
						                   std::complex<Float> v11, std::complex<Float> v01,
						                   Float origin_x, Float origin_y, Float origin_z,
						                   unsigned short flag, int orientation) {
						    if (!((scOutData[idx] | scInData[idx]) & flag)) return;

						    Float du[2], vg2 = 0.0;
						    set_cross_and_velocity<Float>(m00, m10, m11, m01,
						                           v00, v10, v11, v01,
						                           du, vg2, ms2, c);

						    switch (orientation) {
						        case 0: // XY: u→x, v→y
						            pos_x[np] = origin_x + du[0];
						            pos_y[np] = origin_y + du[1];
						            pos_z[np] = origin_z;
						            break;
						        case 1: // YZ: u→y, v→z
						            pos_x[np] = origin_x;
						            pos_y[np] = origin_y + du[0];
						            pos_z[np] = origin_z + du[1];
						            break;
						        case 2: // ZX: u→z, v→x
						            pos_x[np] = origin_x + du[1];
						            pos_y[np] = origin_y;
						            pos_z[np] = origin_z + du[0];
						            break;
						    }

						    total_vg2 += vg2;
						    np++;
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
        for (size_t l = 0; l < max_global_label; ++l) {
            string_len[l] += threadLengths[l];
            string_vel[l] += threadVelocities[l];
						string_gam[l] += threadGammas[l];
						string_cub[l] += threadCubes[l];
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

	return	slp;
}

StringLoopParms stringlength3 (Scalar *field, StringData strDen_in, StringMeasureType strmeas)
{

	Profiler &prof = getProfiler(PROF_STRINGLENGTH3);
	prof.start();

	StringLoopParms slp;
	if (field->Precision() == FIELD_SINGLE)
	{
		slp = stringlength3<float> (field, strDen_in, strmeas);
	}
	else
	{
		slp = stringlength3<double>(field, strDen_in, strmeas);
	}

	prof.stop();

	return slp;
}

//----------------------
// copia seguridad
//----------------------

//----------------------

// template<typename Float>
// StringData	stringlength3	(Scalar *field, StringData strDen_in, StringMeasureType strmeas)
// {
// 	LogMsg	(VERB_NORMAL, "[SL3] stringlength3");
//
//
// 	StringData	strDen;
//
// 	strDen.strDen = strDen_in.strDen;
// 	strDen.strChr = strDen_in.strChr;
// 	strDen.wallDn = strDen_in.wallDn;
// 	strDen.strDen_local = strDen_in.strDen_local;
// 	strDen.strChr_local = strDen_in.strChr_local;
// 	strDen.wallDn_local = strDen_in.wallDn_local;
//
// 	strDen.strLen = 0.;
// 	strDen.strDeng = 0.;
// 	strDen.strVel = 0.;
// 	strDen.strVel2 = 0.;
// 	strDen.strGam = 0.;
// 	strDen.strLen_local = 0.;
// 	strDen.strDeng_local = 0.;
//
// 	if (field->Field() != FIELD_SAXION || !(field->sDStatus() & SD_MAP)){
// 			LogMsg(VERB_NORMAL,"[SL3] Called without string map! (Field = %d, sDStatus= %d)\n",field->Field(),field->sDStatus());
// 			return strDen;
// 		}
//
// 	if (strDen_in.strDen == 0) {
// 		LogMsg(VERB_NORMAL,"[SL3 called without strings: exit]");
// 		return strDen;
// 	}
// 	if (!(field->Field() & FIELD_SAXION)) {
// 		LogMsg(VERB_NORMAL,"[SL3 ftype %d is not FIELD_SAXION  strings: exit]",field->Field());
// 		return strDen;
// 	}
//
// 	LogMsg	(VERB_NORMAL, "[SL3] recalling string data ghost");
// 	field->exchangeStringGhost();
//
// 	int rank = commRank();
//
// 	size_t carde = strDen.strDen_local;
// 	size_t Lx = field->Length();
// 	size_t Lz = field->Depth();
// 	size_t Sf = Lx*Lx;
//
// 	if	(field->Folded())
// 	{
// 		Folder	munge(field);
// 		munge(UNFOLD_ALL);
// 	}
//
// 	LogMsg	(VERB_HIGH, "[SL3] Exchanging M ghosts");
// 	field->sendGhosts(FIELD_M,COMM_SDRV);
// 	field->sendGhosts(FIELD_M,COMM_WAIT);
//
// 	char *strdaa                = static_cast<char *>(static_cast<void *>(field->sData()));
// 	std::complex<Float> *ma     = static_cast<std::complex<Float>*>(field->mStart());
//
//
//
// 	/* Check ghost FOR DEBUG */
// 	// for (size_t i=0;i<field->Surf();i++)
// 	// 	{
// 	// 		if (strdaa[i] != strdaa[field->Size()+i])
// 	// 			LogError("sD ghost exchange didn't work!");
// 	// 	}
//
//
// 	/* clean m2 */
// 	LogMsg	(VERB_HIGH, "[SL3] Clear m2");
// 	size_t mBytes = field->DataSize()*field->eSize();
// 	memset (field->m2Cpu(), 0, mBytes);
// 	char   *m2_c    = static_cast<char *>(field->m2Cpu());
//
// 	LogMsg	(VERB_HIGH, "[SL3]  Enter OMP");
//   if(1)
// 	{
//
// 		int nthreads = commThreads();
//
// 		size_t local_number_cubes[nthreads]    = {0}; // # of cubes found by thread
// 		size_t local_number_segments[nthreads] = {0}; // # of segments
// 		size_t local_start[nthreads]           = {0}; // to label threadwritting
//
//
// 		size_t global_number_cubes = 0;
// 		size_t global_number_segments = 0;
//
// 		/* We read the whole grid and save STRCUB data */
//
// 		size_t m2h_label_vol_perthread = mBytes/2/sizeof(size_t)/nthreads ;
// 		size_t chunk_size = Lz / nthreads;
//
// 		#pragma omp parallel
// 		{
// 			int tid = omp_get_thread_num();
// 			size_t Lzstart = tid * chunk_size;
// 			size_t Lzend = (tid == nthreads - 1) ? Lz : Lzstart + chunk_size;
//
// 			/* Each thread gets 1/nThreads of m2h m2h has mBytes/2 */
// 			size_t disp = Lzstart*(mBytes/4/Lz/sizeof(size_t));
//
// 			char st,stXY2,stYZ2,stZX2 = STRING_NOTHING;
// 			char sc_out = STRCUB_0;
// 			char sc_in  = STRCUB_0;
// 			int n_pla = 0; // number of plaquetes in cube
// 			int chi[6] = {0}; // for chiralities
//
// 			local_number_segments[tid] = 0;
// 			local_number_cubes[tid]    = 0;
// 			local_start[tid]           = 0;
//
// 			// LogMsg(VERB_HIGH,"[SL3] m2h has %d slices ",field->eSize()/Sf);
// 			// LogMsg(VERB_HIGH,"[SL3] Lz %d chunk %d nthreads %d",Lz,chunk_size,nthreads);
// 			// LogMsg(VERB_HIGH,"[SL3] thread %d will read from %d to %d ",tid,Lzstart,Lzend);
// 			// LogMsg(VERB_HIGH,"[SL3] Displacement m2h (thread %d) m2h disp %d*Sf ",tid,disp/Sf);
// 			// LogMsg(VERB_HIGH,"[SL3] Space for local_number_cubes in m2h per thread is %lu (slices)",m2h_label_vol_perthread/Sf);
// 			// LogMsg(VERB_HIGH,"[SL3] Initial local_number_cubes %lu ",local_number_cubes[tid]);
// 			// LogFlush();
//
// 			size_t *m2h     = static_cast<size_t *>(field->m2half()) + disp;
//
// 			for (size_t iz=Lzstart; iz < Lzend; iz++) {
// 				size_t zi = Lx*Lx*iz ;
// 				size_t zp = Lx*Lx*(iz+1) ;
// 				for (size_t iy=0; iy < Lx; iy++) {
// 					size_t yi = Lx*iy ;
// 					size_t yp = Lx*((iy+1)%Lx) ;
// 					for (size_t ix=0; ix < Lx; ix++) {
//
// 						sc_out= STRCUB_0;
// 						sc_in = STRCUB_0;
//
// 						st    = strdaa[ix + yi + zi];
// 						// to read plaquete YZ2
// 						stYZ2 = strdaa[((ix + 1) % Lx) + yi + zi];
// 						// to read plaquete ZX2
// 						stZX2 = strdaa[ix + yp + zi];
// 						// here I will read XY2
// 						stXY2 = strdaa[ix + yi + zp];
//
// 						n_pla = 0;
// 						memset(chi, 0, 6*sizeof(int));
//
// 						/* We search for plaquettes where string enters or exits
// 						1 enter, -1 exits
// 						we check sum is 0 */
// 						if (st & STRING_XY)
// 							{n_pla++; chi[0] = (st & STRING_XY_POSITIVE)? 1 : -1 ;}
// 						if (st & STRING_YZ)
// 							{n_pla++; chi[1] = (st & STRING_YZ_POSITIVE)? 1 : -1 ;}
// 						if (st & STRING_ZX)
// 							{n_pla++; chi[2] = (st & STRING_ZX_POSITIVE)? 1 : -1 ;}
// 						/* These belong to nearby cubes so in is out and viceversa */
// 						if (stXY2 & STRING_XY)
// 							{n_pla++; chi[3] = (stXY2 & STRING_XY_POSITIVE)? -1 : 1 ;}
// 						if (stYZ2 & STRING_YZ)
// 							{n_pla++; chi[4] = (stYZ2 & STRING_YZ_POSITIVE)? -1 : 1 ;}
// 						if (stZX2 & STRING_ZX)
// 							{n_pla++; chi[5] = (stZX2 & STRING_ZX_POSITIVE)? -1 : 1 ;}
//
// 						if (n_pla == 0)
// 							continue;
//
// 						if (n_pla%2 == 1)
// 							{LogError("[SL3] missing plaquete!!!");
// 								// LogMsg(VERB_HIGH,"[SL3] Cube with %d plaquette pierced! Something must be done! %d");LogFlush();
// 								// LogMsg(VERB_HIGH,"[SL3] (iz,iy,ix %d %d %d) %d xy2(%d).yz2(%d).zx2(%d) chi %d %d %d %d %d %d",iz,iy,ix,
// 									// st&STRING_ONLY,stXY2&STRING_ONLY,stYZ2&STRING_ONLY,stZX2&STRING_ONLY,chi[0],chi[1],chi[2],chi[3],chi[4],chi[5]);
// }
//
// 						if (n_pla%2 == 0)
// 						{
// 							/* record ONLY exits */
// 							if (chi[0]<0) sc_out |= STRCUB_XY;
// 							if (chi[1]<0) sc_out |= STRCUB_YZ;
// 							if (chi[2]<0) sc_out |= STRCUB_ZX;
// 							if (chi[3]<0) sc_out |= STRCUB_XY2;
// 							if (chi[4]<0) sc_out |= STRCUB_YZ2;
// 							if (chi[5]<0) sc_out |= STRCUB_ZX2;
// 							/* Record all */
// 							if (chi[0]!=0) sc_in |= STRCUB_XY;
// 							if (chi[1]!=0) sc_in |= STRCUB_YZ;
// 							if (chi[2]!=0) sc_in |= STRCUB_ZX;
// 							if (chi[3]!=0) sc_in |= STRCUB_XY2;
// 							if (chi[4]!=0) sc_in |= STRCUB_YZ2;
// 							if (chi[5]!=0) sc_in |= STRCUB_ZX2;
//
// 							/* check INs = OUTs */
// 							int sum = 0 ;
// 							for (int i = 0; i < 6;i++)
// 								sum += chi[i];
//
// 								LogFlush();
// 							if (sum != 0){
// 								LogError("[SL3] chiralities mismatched!!");
// 								// LogMsg(VERB_HIGH,"[SL3] (iz,iy,ix %d %d %d) %d xy2(%d).yz2(%d).zx2(%d) chi %d %d %d %d %d %d",iz,iy,ix,
// 								// 	st&STRING_ONLY,stXY2&STRING_ONLY,stYZ2&STRING_ONLY,stZX2&STRING_ONLY,chi[0],chi[1],chi[2],chi[3],chi[4],chi[5]);
// 							}
// 							/* record the number of exits */
// 							if (n_pla == 2)
// 								sc_out |= STRCUB_1EX; // 1 EXIT
// 							if (n_pla == 4)
// 								sc_out |= STRCUB_1EXEX; // 2 EXIT
// 							if (n_pla == 6)
// 								sc_out |= STRCUB_3EX; // 2 EXIT
// 								/* record the number of exits */
// 							if (n_pla == 2)
// 								sc_in |= STRCUB_1EX; // 1 EXIT
// 							if (n_pla == 4)
// 								sc_in |= STRCUB_1EXEX; // 2 EXIT
// 							if (n_pla == 6)
// 								sc_in |= STRCUB_3EX; // 2 EXIT
// 						/* we copy string_cube data in m2!*/
// 						m2_c[ix + yi + zi]               = sc_out;
// 						m2_c[field->Size()+ix + yi + zi] = sc_in;
//
// 						/* we keep cube idx in m2h, possible race condition if many cubes! */
// 						if (local_number_cubes[tid] > m2h_label_vol_perthread)
// 							LogError("[SL3] -- Too many cubes! Where did all these strings came from?");
// 						else
// 							{
// 								// LogMsg(VERB_HIGH,"-- (thread/iz/iy/ix %d/%d-%d-%d) %d ",tid,iz,iy,ix,local_number_cubes[tid]);LogFlush();
// 							m2h[local_number_cubes[tid]] = ix + yi + zi;
// 							local_number_cubes[tid]     += 1;
// 							local_number_segments[tid]  += n_pla/2;
// 						}
// 						}
// 				  } //end of ix loop
// 			  } //end of iy loop
// 		  } // end of iz loop
//
// 		} // end parallel section
//
//
// 			LogMsg(VERB_HIGH,"[SL3] continue serial ");
//
// 			for (int i = 0; i < nthreads; ++i) {
// 				global_number_cubes    += local_number_cubes[i];
// 				global_number_segments += local_number_segments[i];
// 			}
//
// 			for (unsigned int i = 0; i < nthreads; ++i)
// 				{
// 					for (unsigned int j = 0; j < i; ++j)
// 						local_start[i] += local_number_cubes[j];
// 				LogMsg(VERB_HIGH,"[SL3] Threah %d found %d cubes (%d segments) and will write starting at %d",i, local_number_cubes[i],local_number_segments[i],local_start[i]);
// 			}
// 			LogFlush();
//
// 			/* Compress cubes idx info to,
// 			each thread wrote at m2h+disp,
// 			it should now at m2h+local_start */
// 			if (global_number_cubes < (mBytes/sizeof(size_t)/4))
// 			{
// 				LogMsg(VERB_NORMAL,"Compressing idxs! global_number_cubes < m2/4 (%d < %d)",global_number_cubes,mBytes/sizeof(size_t)/4);
// 				for (int tid=1;tid<nthreads;tid++)
// 				{
// 					LogMsg(VERB_NORMAL,"[SL3] Threah %d copies %d from %d to %d",tid,
// 					local_number_cubes[tid],local_start[tid],local_start[tid]+local_number_cubes[tid]);
// 					size_t Lzstart = tid * chunk_size;
// 					size_t disp = Lzstart*(mBytes/4/Lz/sizeof(size_t));
// 					size_t *m2h     = static_cast<size_t *>(field->m2half()) + disp;
// 					size_t *m2_sizet = static_cast<size_t *>(field->m2half());
// 					void *origin = static_cast<void *>(m2h);
// 					void *dest   = static_cast<void *>(m2_sizet+local_start[tid]);
// 					memmove(dest, origin, local_number_cubes[tid]*sizeof(size_t));
// 					LogMsg(VERB_NORMAL,"[SL3] values %d - %d",m2h[0],m2_sizet[local_start[tid]]);
// 				}
// 			}
// 			else
// 			{
// 				LogMsg(VERB_NORMAL,"anda que no hay cuerdas ... me rindo!");
// 				return strDen;
// 			}
//
// 			LogMsg(VERB_HIGH,"[SL3] move strig-cube data to m234!");
// 			char *m234 = static_cast<char *>(field->m2half())+mBytes/4;
// // memcpy(field->sData(), field->m2Cpu(), field->Size()*sizeof(char));
// if (m234 < (char*)field->m2Cpu() + 2*field->Size() &&
//     m234 + 2*field->Size() > (char*)field->m2Cpu()) {
//     LogError("[SL3] Overlapping copy between m2Cpu and m234! Use memmove or fix offsets.");
// }
//
// 			memmove(static_cast<void *>(m234), field->m2Cpu(), 2*field->Size());
// 			LogMsg(VERB_HIGH,"[SL3] reset first half m2");
// 			memset (field->m2Cpu(), 0, mBytes/2);
//
// 			char *m2h_c = static_cast<char *>(field->m2half());
//
//
// 			/* We have all cubes tagged.
// 			Next we build the strings by a map of tags*/
//
//
// 			/* We read the whole grid and write size_t labels (IDs) in m2
// 			labels are (threadID,unsigned short int)
// 			when we find an equivalence we write it in a global local_equivs "map"*/
//
// 			typedef unsigned short int usi;
//
// 			std::vector<std::pair<std::pair<usi, usi>, std::pair<usi, usi>>> equivalences;
// 			usi local_used_labels[nthreads]    = {0}; // # of cubes found by thread
//
// 			usi *m2_usi = static_cast<usi *>(field->m2Cpu());
//
// 			#pragma omp parallel
// 			{
// 				int tid = omp_get_thread_num();
// 				usi tid_usi = (usi) tid;
// 				int Lzstart = tid * chunk_size;
// 				int Lzend = (tid == nthreads - 1) ? Lz : Lzstart + chunk_size;
// 				size_t disp = Lzstart*(mBytes/4/Lz/sizeof(size_t));
//
// 				char sc, next_sc, closed_sc = STRCUB_0;
// 				int npla = 0; // number of plaquetes in cube
// 				int chi[6] ; // for chiralities
// 				size_t X[3];
//
// // size_t *m2h     = static_cast<size_t *>(field->m2half()) + disp;
// 				size_t *m2h     = static_cast<size_t *>(field->m2half()) + local_start[tid];
//
// 				size_t idx, next_idx ;
// 				/* each thread has its same label counter
// 				grid is initialised to (0,0) */
// 				usi label = 0;
// 				usi next_label =0;
// 				usi next_tidusi = 0;
// 				int n_pla,next_n_pla;
// 				bool out_of_volume;
//
// 				/* we loop over identified cubes, label them and those connected
// 				when two threads get different labels for the same string we:
// 				- write a dictionary?
// 				- use the smallest and continue running?
// 				 */
//
// 				for (size_t i=0; i < local_number_cubes[tid]; i++)
// 				{
//
// 					/* read the idx of a cube with string */
// 					idx = m2h[i];
//
// 					/* if cube is unlabeled, label and track the string,
// 					otherwise it has already been followed and thus skip
// 					note:
// 						m2[2*idx]   = nthread
// 						m2[2*idx+1] = local_label  */
// 					if (m2_usi[2*idx+1] == 0){
// 						// if cube is untagged initialise a new label
// 						label = label+1;
// 						// no race condition, each thread has its own points
// 						m2_usi[2*idx+1] = label;
// 						m2_usi[2*idx]   = tid_usi;}
// 					else
// 						continue;
//
// 					/* We track cubes labeling with the same label
// 					If next_cube has another label, stop, write an equivalence in
// 					dictionary */
// 					/* load the strData to find next */
// // sc = strdaa[idx];
// 					sc = m234[idx];
// 					n_pla = how_many_plaquettes(sc);
// 					bool intersection = false;
// 					/* initialise stopping condition for the while loop that
// 					tracks the strings */
// 					next_label = 0;
//
// 					while ((next_label == 0) || intersection)
// 					{
// 						/* reset intersection */
// 						intersection = false;
//
// 						/* find next cube
// 						- IF OUTSIDE LOCAL MPI continue we deal with MPI later */
// 						indexXeon::idx2Vec (idx, X, Lx);
// 						next_idx = next_cube_idx(sc,X,Lx,Sf,&out_of_volume);
// 						if (out_of_volume)
// 							break;
//
// 						/* we modify strData to "close" one exit:
// 						if current cube has more than 2 plaquettes!
// 						(this way, the next time a thread-tracker passes by it takes
// 						a different exit) */
//
// 						if ( n_pla > 1){
// 							/* if n_pla > 1 means that we have arrived to a multi-string cube
// 							with more than 1 exit.
// 							These cubes might have been already labelled, but still have
// 							more exits to take, thus if they have a label but n_pla > 1
// 							WE will not stop the while loop,
// 							we keep the original label in the loop
// 							we write an equivalence (two loops with an intersection are the same)
// 							Note that by construction we will close the exit corresponding
// 							to next_idx (see length.h) */
// 							closed_sc = close_exit_cube(sc,X,Lx,Sf);
// 							/* we change strData with a slow locking to avoid race */
// 							#pragma omp critical
// 							{
// // strdaa[idx] = closed_sc;
// 							m234[idx] = closed_sc;
// 							}
// 						}
//
// 						/* read next cubes label:
// 						- if 0, label and continue
// 						- if cube has non-zero label, write an assotiation in the local map and continue
// 						*/
// 						next_label = m2_usi[2*next_idx+1];
// 						next_tidusi = m2_usi[2*next_idx];
// //next_sc    = strdaa[next_idx];
// 						next_sc    = m234[next_idx];
// 						next_n_pla = how_many_plaquettes(next_sc);
// 						if (next_label == 0){
// 							#pragma omp critical
// 								{
// 									m2_usi[2*next_idx]   = tid_usi;
// 									m2_usi[2*next_idx+1] = label;
// 								}
// 							}
// 						else {
//
// 							if ( (next_label == label) && ( next_tidusi == tid_usi)){
// 								// LogMsg(VERB_PARANOID, "loop %d closed! (thread %d)",label,tid);
// 								break;}
// 							else // track arrives to a (tracked) track from other thread
// 							{
// 								/* Two different IDs connected so we write an equivalence */
// 								#pragma omp critical
// 								equivalences.push_back({{tid_usi, label}, {next_tidusi, next_label}});
//
// 								/* Now, it could be that next cube IS an intersection,
// 								with one exit unexplored. Thus:
// 									- if next cube is an intersection we continue
// 									- if not we break.
// 								Note that if next had 2 or 3 exist it always has the 128 bit,
// 								so (next_sc & STRCUB_1EXEX) should be true for intersections and
// 								false for 1-exit cubes. */
//
// 								if (next_sc & STRCUB_1EXEX)
// 									intersection = true;
// 								else
// 									break ; //breaks the while loop
// 							}
// 							/* MPI !!*/
//
// 						}
// 						/* next cube is the cube */
// 						idx = next_idx;
// 						sc  = next_sc;
// 						n_pla = next_n_pla;
// 					}
// 					// end while tracking points connected to idx, consider next idx
//
// 				} //end cube loop
// 			local_used_labels[tid] = label;
// 			} //end parallel section
//
// 			/* At this point all our points are labelled,
// 			but are redundant, we want to relabel them
// 			with a unique size_t label */
//
// 			/* Calculate label number */
// 			size_t total_labels_with_redundancies = 0;
// 			size_t cumlabels[nthreads] = {0};
// 			for (int i=0; i < nthreads; i++){
// 				LogMsg(VERB_HIGH,"Thread %d used %d labels",i,local_used_labels[i]);
// 				total_labels_with_redundancies += local_used_labels[i];
// 				for (int j=0; j < i; j++)
// 					cumlabels[i] += local_used_labels[j];
// 				}
// 			LogMsg(VERB_HIGH,"Total labels with redundancies %zu",total_labels_with_redundancies);
// 			LogMsg(VERB_HIGH,"Equivalences %zu",equivalences.size());
//
// 			/* Dictionary list, asigns a unique label to each string
// 			taking equivalencies into account
// 			many of them would be solved by running the string backwards*/
// 			LogMsg(VERB_NORMAL,"Dics");
// 			std::vector<std::pair<size_t, size_t>> equiv2(equivalences.size());
// 			for (size_t i=0; i < equivalences.size(); i++)
// 			{
// 				auto e = equivalences[i];
// 				size_t id1 = label_map_index(cumlabels,e.first.first,e.first.second);
// 				size_t id2 = label_map_index(cumlabels,e.second.first,e.second.second);
// 				// equiv2.push_back({max(id1,id2)], min(id1,id2)});
// 				equiv2.push_back({id1,id2});
// 			}
//
// 			/* initialise label map */
// 			std::vector<size_t> labels_redundant(total_labels_with_redundancies);
// 			for (size_t i=0; i < total_labels_with_redundancies; i++)
// 				labels_redundant[i] = i;
// 			LogMsg(VERB_NORMAL,"LabelEquivalence");
// 			LabelEquivalence le;
//     	le.build(equiv2); // call le.get_min_label(size_t)
//
// 			LogMsg(VERB_NORMAL,"relabeling...");
// 			/* vector linking each size_t label with a dense label */
// 			auto relabeled = le.relabel_to_dense(labels_redundant);
//
// 			LogMsg(VERB_NORMAL,"max... %d",relabeled.size());LogFlush();
// 			// for (int i =0, i<relabeled.size(),i++)
//
// 			auto max_it = std::max_element(relabeled.begin(), relabeled.end());
// 			// if (max_it != vec.end()) {
//       //   std::cout << "Max value: " << *max_it << std::endl;
//     	// } else {
//       //   std::cout << "Vector is empty." << std::endl;
//     	// }
// 			size_t ncubes;
// 			if (max_it != relabeled.end()) {
// 				ncubes = *max_it+1;
// 				LogMsg(VERB_NORMAL,"# strings found = %d",ncubes);
// 				LogFlush();
//
// 			}
// 			else {
// 				ncubes = 0;
// 				LogMsg(VERB_NORMAL,"# strings found = NONE!");
// 				LogFlush();
// 				return strDen;
// 			}
//
// 			/* Apply dense labels
// 			calculate string length? */
//
// 			std::vector<std::vector<unsigned>> string_cubes(nthreads, std::vector<unsigned>(ncubes,0));
// 			unsigned int *m2_ui = static_cast<unsigned int *>(field->m2Cpu());
// 			size_t old_label, new_dense_label;
// 			#pragma omp parallel
// 			{
// 				int tid = omp_get_thread_num();
// 				usi tid_usi = (usi) tid;
// 				int chunk_size = Lz / nthreads;
// 				int Lzstart = tid * chunk_size;
// 				int Lzend = (tid == nthreads - 1) ? Lz : Lzstart + chunk_size;
// 				size_t disp = Lzstart*(mBytes/4/Lz/sizeof(size_t));
//
// // size_t *m2h     = static_cast<size_t *>(field->m2half()) + disp;
// 				size_t *m2h     = static_cast<size_t *>(field->m2half()) + local_start[tid];
//
// 				size_t idx,new_dense_label;
// 				usi l1,l2;
//
// 				for (size_t i=0; i < local_number_cubes[tid]; i++)
// 				{
// 					idx = m2h[i];
// 					l1 = m2_usi[2*idx];
// 					l2 = m2_usi[2*idx+1];
// 					old_label = label_map_index(cumlabels,l1,l2);
// 					new_dense_label = relabeled[le.get_min_label(old_label)];
// 					/* We label from 1 not from 0 to differentiate no-string in maps */
// 					m2_ui[idx] = (unsigned int) new_dense_label + 1;
// 					string_cubes[tid][new_dense_label] += 1;
// 				} //end cube loop
//
// 			} //end parallel section
//
// 			for (size_t is = 0; is<ncubes;is++)
// 			{	for (size_t tid=1;tid<nthreads;tid++)
// 					string_cubes[0][is] += string_cubes[tid][is];
// 			LogMsg(VERB_HIGH,"- string #%d length = %d cubes!",is,string_cubes[0][is]);
// 			}
//
// 			/* m2h contains list of cubes, run and fill lengths,
// 			assign  */
//
// 			std::vector<std::vector<double>> string_length(nthreads, std::vector<double>(ncubes,0));
//
// 			#pragma omp parallel
// 			{
// 				int tid = omp_get_thread_num();
// 				usi tid_usi = (usi) tid;
// 				int Lzstart = tid * chunk_size;
// 				int Lzend = (tid == nthreads - 1) ? Lz : Lzstart + chunk_size;
// 				// size_t disp = Lzstart*(mBytes/4/Lz/sizeof(size_t));
// 				size_t X[3];
//
// 				char sc, next_sc, closed_sc = STRCUB_0;
//
// 				// size_t *m2h     = static_cast<size_t *>(field->m2half()) + disp;
// 				size_t *m2h     = static_cast<size_t *>(field->m2half()) + local_start[tid];
//
// 				usi l1,l2;
// 				size_t idx,idx_in,idx_out,new_dense_label;
// 				/* we read the idx, read reduced label, compute length
// 				(coordenates?),etc... and save*/
//
// 				std::vector<double> pos_x;
// 				std::vector<double> pos_y;
// 				std::vector<double> pos_z;
//
// 				for (size_t i=0; i < local_number_cubes[tid]; i++)
// 				{
// 					idx = m2h[i];
// 					indexXeon::idx2Vec (idx, X, Lx);
// 					char sc_in = m234[field->Size()+idx];
// 					new_dense_label = m2_ui[idx];
//
// 					/* read sc_in to identify plaquettes */
// 						size_t ixM = ((X[0] + 1) % Lx) + X[1]*Lx + X[2]*Sf;
// 						size_t iyM = X[0] + ((X[1] + 1) % Lx)*Lx + X[2]*Sf;
// 						size_t izM = idx+Sf;
// 						size_t ixyM = ((X[0] + 1) % Lx) + ((X[1] + 1) % Lx)*Lx + X[2]*Sf;
// 						size_t iyzM = X[0] + ((X[1] + 1) % Lx)*Lx + (X[2]+1)*Sf;
// 						size_t izxM = ((X[0] + 1) % Lx) + X[1]*Lx + (X[2]+1)*Sf;
// 						size_t ixyzM = ((X[0] + 1) % Lx) + ((X[1] + 1) % Lx)*Lx + (X[2]+1)*Sf;
//
// 						pos_x.clear();
// 						pos_y.clear();
// 						pos_z.clear();
//
// 						double du[2];
// 						if (sc_in & STRCUB_XY) {
// 							setCross(ma[idx],ma[ixM],ma[iyM],ma[ixyM],du);
// 							pos_x.push_back(X[0] + du[0]);
// 							pos_y.push_back(X[1] + du[1]);
// 							pos_z.push_back(X[2]);
// 						}
// 						if (sc_in & STRCUB_YZ) {
// 							setCross(ma[idx],ma[iyM],ma[izM],ma[iyzM],du);
// 							pos_x.push_back(X[0]);
// 							pos_y.push_back(X[1] + du[0]);
// 							pos_z.push_back(X[2] + du[1]);
// 						}
// 						if (sc_in & STRCUB_ZX) {
// 							setCross(ma[idx],ma[izM],ma[ixM],ma[izxM],du);
// 							pos_x.push_back(X[0] + du[1]);
// 							pos_y.push_back(X[1]);
// 							pos_z.push_back(X[2] + du[0]);
// 						}
// 						if (sc_in & STRCUB_YZ2) {
// 							setCross(ma[ixM],ma[ixyM],ma[izxM],ma[ixyzM],du);
// 							pos_x.push_back(X[0] + 1.);
// 							pos_y.push_back(X[1] + du[0]);
// 							pos_z.push_back(X[2] + du[1]);
// 						}
// 						if (sc_in & STRCUB_ZX2) {
// 							setCross(ma[iyM],ma[iyzM],ma[ixyM],ma[ixyzM],du);
// 							pos_x.push_back(X[0] + du[1]);
// 							pos_y.push_back(X[1] + 1.);
// 							pos_z.push_back(X[2] + du[0]);
// 						}
// 						if (sc_in & STRCUB_XY2) {
// 							setCross(ma[izM],ma[izxM],ma[iyzM],ma[ixyzM],du);
// 							pos_x.push_back(X[0] + du[0]);
// 							pos_y.push_back(X[1] + du[1]);
// 							pos_z.push_back(X[2] + 1.);
// 						}
//
// 						double dl = dl_cal(pos_x,pos_y,pos_z);
//
// 						string_length[tid][new_dense_label] += dl;
// 				} //end cube loop
//
// 			} //end parallel section
//
// 			for (int is = 0; is<ncubes;is++)
// 			{	for (int tid=1;tid<nthreads;tid++)
// 					string_length[0][is] += string_length[tid][is];
// 			LogMsg(VERB_HIGH,"- string #%d length = %f !",is,string_length[0][is]);
// 			}
//
// 			field->setM2(M2_LABEL_MAP);
//
// 	} //if meas strinlength
//
//     commSync();
//
//
//
//
// 	return	strDen;
// }
