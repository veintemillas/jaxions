#ifndef	_LABEL_TOOLS_
	#define	_LABEL_TOOLS_

#include <mpi.h>
#include <vector>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <utility>

// Simple Union-Find with path compression
class UnionFind {
    std::vector<unsigned> parent;

public:
    UnionFind(unsigned size) : parent(size + 1) {
        for (unsigned i = 0; i <= size; ++i)
            parent[i] = i;
    }

    unsigned find(unsigned x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }

    void unite(unsigned x, unsigned y) {
        parent[find(x)] = find(y);
    }
};

std::vector<unsigned>
assign_dense_labels(const std::vector<std::pair<unsigned, unsigned>>& equivalences,
                    unsigned max_label) {

		LogMsg(VERB_HIGH,"[ADL] Start with %u max_label",max_label);LogFlush();
    // Step 1: Validate that all equivalence labels are in range
    // for (const auto& [a, b] : equivalences) {
    //     if (a > max_label || b > max_label) {
    //         throw std::invalid_argument("Equivalence contains label > max_label");
    //     }
    // }

		unsigned seen_max = 0;
    for (size_t i = 0; i < equivalences.size(); ++i) {
        auto [a,b] = equivalences[i];
				LogMsg(VERB_PARANOID,"[ADL] %d %d == %d",i,a,b);
        if (a < 1 || b < 1) {
						LogOut("assign_dense_labels: label < 1 at pair # %d (%d, %d) max_label= %u",i,a,b,max_label);
            std::ostringstream oss;
            oss << "assign_dense_labels: label < 1 at pair #" << i
                << " (" << a << "," << b << ") max_label=" << max_label;
            throw std::invalid_argument(oss.str());
        }
        seen_max = std::max(seen_max, std::max(a,b));
        if (a > max_label || b > max_label) {
						LogOut("assign_dense_labels: label > max_label at pair # %d (%d, %d) max_label= %u",i,a,b,max_label);
            std::ostringstream oss;
            oss << "assign_dense_labels: label > max_label at pair #" << i
                << " (" << a << "," << b << ") max_label=" << max_label
                << " seen_max=" << seen_max;
            throw std::invalid_argument(oss.str());
        }
    }

    // Step 2: Initialize and process Union-Find
    UnionFind uf(max_label);
    for (const auto& [a, b] : equivalences)
        uf.unite(a, b);

    // Step 3: Compute dense labels
    std::unordered_map<unsigned, unsigned> root_to_dense;
    std::vector<unsigned> dense_labels(max_label + 1);
    unsigned current_dense = 1;

    for (unsigned i = 1; i <= max_label; ++i) {
        unsigned root = uf.find(i);
        auto [it, inserted] = root_to_dense.emplace(root, current_dense);
        if (inserted) ++current_dense;
        dense_labels[i] = it->second;
    }

		LogMsg(VERB_HIGH,"[ADL] Exit");LogFlush();
    return dense_labels;  // dense_labels[i] is the dense label for original label i
}


std::vector<std::pair<unsigned, unsigned>> gather_global_equivalences(
    const std::vector<std::pair<unsigned, unsigned>>& local_equiv_1,
    const std::vector<std::pair<unsigned, unsigned>>& local_equiv_2,
    int root_rank,
    MPI_Comm comm = MPI_COMM_WORLD
) {
    int world_rank, world_size;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    // Step 1: Join and flatten
    std::vector<unsigned> flat_local;
    flat_local.reserve(2 * (local_equiv_1.size() + local_equiv_2.size()));

    for (const auto& p : local_equiv_1) {
        flat_local.push_back(p.first);
        flat_local.push_back(p.second);
    }
    for (const auto& p : local_equiv_2) {
        flat_local.push_back(p.first);
        flat_local.push_back(p.second);
    }

    int local_size = flat_local.size();
    std::vector<int> recv_counts, displs;
    std::vector<unsigned> all_data;

    // Step 2: Gather sizes
    if (world_rank == root_rank) {
        recv_counts.resize(world_size);
    }

    MPI_Gather(&local_size, 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT,
               root_rank, comm);

    // Step 3: Prepare displacements and receive buffer
    if (world_rank == root_rank) {
        displs.resize(world_size);
        displs[0] = 0;
        for (int i = 1; i < world_size; ++i) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        }

        int total_size = displs.back() + recv_counts.back();
        all_data.resize(total_size);
    }

    // Step 4: Gatherv the data
    MPI_Gatherv(flat_local.data(), local_size, MPI_UNSIGNED,
                all_data.data(), recv_counts.data(), displs.data(), MPI_UNSIGNED,
                root_rank, comm);

    // Step 5: Convert flat data into equivalence pairs (only on root)
    std::vector<std::pair<unsigned, unsigned>> global_equivs;

    if (world_rank == root_rank) {
        for (size_t i = 0; i < all_data.size(); i += 2) {
            global_equivs.emplace_back(all_data[i], all_data[i + 1]);
        }
    }

    return global_equivs;
}

// Gathers equivalence pairs to root rank (only root returns the result)
std::vector<std::pair<unsigned, unsigned>> gather_equivalences_to_root(
    const std::vector<std::pair<unsigned, unsigned>>& local_equiv_1,
    const std::vector<std::pair<unsigned, unsigned>>& local_equiv_2,
    int root_rank,
    MPI_Comm comm
) {
    int world_rank, world_size;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    // Flatten local pairs
    std::vector<unsigned> flat_local;
    flat_local.reserve(2 * (local_equiv_1.size() + local_equiv_2.size()));
    for (const auto& p : local_equiv_1) {
        flat_local.push_back(p.first);
        flat_local.push_back(p.second);
    }
    for (const auto& p : local_equiv_2) {
        flat_local.push_back(p.first);
        flat_local.push_back(p.second);
    }

    int local_size = flat_local.size();
    std::vector<int> recv_counts;
    std::vector<int> displs;
    std::vector<unsigned> all_data;

    // Step 1: Gather sizes
    if (world_rank == root_rank) {
        recv_counts.resize(world_size);
    }

    MPI_Gather(&local_size, 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT,
               root_rank, comm);

    // Step 2: Compute displacements and allocate receive buffer
    if (world_rank == root_rank) {
        displs.resize(world_size);
        displs[0] = 0;
        for (int i = 1; i < world_size; ++i)
            displs[i] = displs[i - 1] + recv_counts[i - 1];

        int total_size = displs.back() + recv_counts.back();
        all_data.resize(total_size);
    }

    // Step 3: Gather data
    MPI_Gatherv(flat_local.data(), local_size, MPI_UNSIGNED,
                all_data.data(), recv_counts.data(), displs.data(), MPI_UNSIGNED,
                root_rank, comm);

    // Step 4: Reconstruct global equivalences on root
    std::vector<std::pair<unsigned, unsigned>> global_equivs;
    if (world_rank == root_rank) {
        for (size_t i = 0; i < all_data.size(); i += 2) {
            global_equivs.emplace_back(all_data[i], all_data[i + 1]);
        }
    }

    return global_equivs;
}


std::vector<unsigned> broadcast_dense_label_vector_from_root(
    const std::vector<unsigned>* root_vec,
    int root_rank,
    MPI_Comm comm
) {
    int world_rank;
    MPI_Comm_rank(comm, &world_rank);

    int size = 0;
    if (world_rank == root_rank && root_vec != nullptr) {
        size = root_vec->size();
    }

    // Broadcast the size first
    MPI_Bcast(&size, 1, MPI_INT, root_rank, comm);

    std::vector<unsigned> result(size);

    // Broadcast the actual data
    if (world_rank == root_rank && root_vec != nullptr) {
        std::copy(root_vec->begin(), root_vec->end(), result.begin());
    }

    MPI_Bcast(result.data(), size, MPI_UNSIGNED, root_rank, comm);
    return result;
}



#endif
