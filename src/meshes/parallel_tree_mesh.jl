# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    partition!(mesh::ParallelTreeMesh, allow_coarsening=true)

Partition `mesh` using a static domain decomposition algorithm
based on leaf cell count and tree structure.
If `allow_coarsening` is `true`, the algorithm will keep leaf cells together
on one rank when needed for local coarsening (i.e. when all children of a cell are leaves).
"""
function partition!(mesh::ParallelTreeMesh; allow_coarsening = true)
    # Determine number of leaf cells per rank
    leaves = leaf_cells(mesh.tree)
    @assert length(leaves)>mpi_nranks() "Too many ranks to properly partition the mesh!"
    n_leaves_per_rank = OffsetArray(fill(div(length(leaves), mpi_nranks()),
                                         mpi_nranks()),
                                    0:(mpi_nranks() - 1))
    for d in 0:(rem(length(leaves), mpi_nranks()) - 1)
        n_leaves_per_rank[d] += 1
    end
    @assert sum(n_leaves_per_rank) == length(leaves)

    # Assign MPI ranks to all cells such that all ancestors of each cell - if not yet assigned to a
    # rank - belong to the same rank
    mesh.first_cell_by_rank = similar(n_leaves_per_rank)
    mesh.n_cells_by_rank = similar(n_leaves_per_rank)

    leaf_count = 0
    mesh.first_cell_by_rank[0] = 1 # Assign first cell to rank 0
    # Iterate over all ranks
    for d in 0:(mpi_nranks() - 1)
        leaf_count += n_leaves_per_rank[d]
        last_id = leaves[leaf_count]
        parent_id = mesh.tree.parent_ids[last_id]

        # Check if all children of the last parent are leaves
        if allow_coarsening &&
           all(id -> is_leaf(mesh.tree, id), @view mesh.tree.child_ids[:, parent_id]) &&
           d < length(n_leaves_per_rank) - 1

            # To keep children of parent together if they are all leaves,
            # all children are added to this rank
            additional_cells = (last_id + 1):mesh.tree.child_ids[end, parent_id]
            if length(additional_cells) > 0
                last_id = additional_cells[end]

                additional_leaves = count(id -> is_leaf(mesh.tree, id),
                                          additional_cells)
                leaf_count += additional_leaves
                # Add leaves to this rank, remove from next rank
                n_leaves_per_rank[d] += additional_leaves
                n_leaves_per_rank[d + 1] -= additional_leaves
            end
        end

        @assert all(n -> n > 0, n_leaves_per_rank) "Too many ranks to properly partition the mesh!"

        mesh.n_cells_by_rank[d] = last_id - mesh.first_cell_by_rank[d] + 1
        mesh.tree.mpi_ranks[mesh.first_cell_by_rank[d]:last_id] .= d

        # Set first cell of next rank
        if d < length(n_leaves_per_rank) - 1
            mesh.first_cell_by_rank[d + 1] = mesh.first_cell_by_rank[d] +
                                             mesh.n_cells_by_rank[d]
        end

        #println("Cells per rank $d: ", mesh.n_cells_by_rank[d])
    end

    @assert all(x -> x >= 0, mesh.tree.mpi_ranks[1:length(mesh.tree)])
    @assert sum(mesh.n_cells_by_rank) == length(mesh.tree)

    return nothing
end

#=
# NOTE: This does also not really solve the problem, as counting the effort works only for the 
# truly partitioned stages, but not for the shared ones.
# TODO: Better: Partition truly based on level selection
function partition!(mesh::ParallelTreeMesh, alg; allow_coarsening = true)
    leaves = leaf_cells(mesh.tree)
    num_leaves = length(leaves)
    @assert num_leaves>mpi_nranks() "Too many ranks to properly partition the mesh!"

    # Determine the computational effort per rank
    cost_per_leaf = Vector{Tuple{Int, Int}}(undef, num_leaves)
    total_cost = 0
    max_level = maximum_level(mesh.tree)
    for i in 1:num_leaves
        leaf = leaves[i]
        level_per_leaf = max_level + 1 - mesh.tree.levels[leaf]
        
        cost = alg.StageEvaluations[level_per_leaf]
        #cost = alg.StageEvaluations[level_per_leaf] - alg.NumStageEvalsMin
        cost = alg.StageEvaluations[level_per_leaf] - 3
        
        cost_per_leaf[i] = (leaf, cost)
        total_cost += cost
    end
    average_cost = total_cost / mpi_nranks()

    leaves_per_rank = Dict{Int, Vector{Int}}(i => [] for i in 0:mpi_nranks() - 1)
    leaf_pos = 1
    for rank in 0:(mpi_nranks() - 1)
        rank_cost = 0
        while rank_cost < average_cost && leaf_pos <= num_leaves
            leaf_id, cost = cost_per_leaf[leaf_pos]
            push!(leaves_per_rank[rank], leaf_id)
            rank_cost += cost
            leaf_pos += 1
        end
    end

    mesh.first_cell_by_rank[0] = 1 # Assign first cell to rank 0
    # Iterate over all ranks
    for d in 0:(mpi_nranks() - 1)
        last_id = leaves_per_rank[d][end] # end or maximum?
        parent_id = mesh.tree.parent_ids[last_id]

        # Check if all children of the last parent are leaves
        if allow_coarsening &&
           all(id -> is_leaf(mesh.tree, id), @view mesh.tree.child_ids[:, parent_id]) #&& d < length(n_leaves_per_rank) - 1

            # To keep children of parent together if they are all leaves,
            # all children are added to this rank
            additional_cells = (last_id + 1):mesh.tree.child_ids[end, parent_id]
            if length(additional_cells) > 0
                last_id = additional_cells[end]
            end
        end

        mesh.n_cells_by_rank[d] = last_id - mesh.first_cell_by_rank[d] + 1
        mesh.tree.mpi_ranks[mesh.first_cell_by_rank[d]:last_id] .= d

        # Set first cell of next rank
        if d < mpi_nranks() - 1
            mesh.first_cell_by_rank[d + 1] = mesh.first_cell_by_rank[d] +
                                             mesh.n_cells_by_rank[d]
        end
    end
    
    return nothing
end
=#

function partition_PERK!(mesh::ParallelTreeMesh; allow_coarsening = true)
    leaves = leaf_cells(mesh.tree)
    num_leaves = length(leaves)
    @assert num_leaves>mpi_nranks() "Too many ranks to properly partition the mesh!"

    min_level = minimum_level(mesh.tree)
    max_level = maximum_level(mesh.tree)
    n_levels = max_level - min_level + 1

    leaves_per_level = Dict{Int, Vector{Int}}(i => [] for i in 1:n_levels)
    for leaf in leaves
        level = max_level + 1 - mesh.tree.levels[leaf]
        
        # Append the leaf to the list corresponding to its level
        push!(leaves_per_level[level], leaf)
    end

    leaves_per_rank = Dict{Int, Vector{Int}}(i => [] for i in 0:mpi_nranks() - 1)

    for (_, leaves) in leaves_per_level
        num_leaves_at_level = length(leaves)
        leaves_per_rank_at_level = ceil(Int, num_leaves_at_level / mpi_nranks())
    
        for rank in 0:(mpi_nranks() - 1)
            start_idx = rank * leaves_per_rank_at_level + 1
            end_idx = min((rank + 1) * leaves_per_rank_at_level, num_leaves_at_level)
            
            if start_idx <= end_idx
                append!(leaves_per_rank[rank], leaves[start_idx:end_idx])
            end

            println("Leaves of $rank: ", length(leaves_per_rank[rank]))
        end
    end

    
    mesh.first_cell_by_rank[0] = 1 # Assign first cell to rank 0
    # Iterate over all ranks
    for d in 0:(mpi_nranks() - 1)
        
        sort!(leaves_per_rank[d])
        last_id = leaves_per_rank[d][end]

        
        parent_id = mesh.tree.parent_ids[last_id]
        # Check if all children of the last parent are leaves
        if allow_coarsening &&
           all(id -> is_leaf(mesh.tree, id), @view mesh.tree.child_ids[:, parent_id]) #&& d < length(n_leaves_per_rank) - 1

            # To keep children of parent together if they are all leaves,
            # all children are added to this rank
            additional_cells = (last_id + 1):mesh.tree.child_ids[end, parent_id]
            if length(additional_cells) > 0
                last_id = additional_cells[end]
            end
        end
        
        mesh.n_cells_by_rank[d] = last_id - mesh.first_cell_by_rank[d] + 1

        # TODO: Why is this causing such a major inequality?
        println("First cell of rank $d: ", mesh.first_cell_by_rank[d], " Last cell of rank $d: ", last_id)
        mesh.tree.mpi_ranks[mesh.first_cell_by_rank[d]:last_id] .= d

        # Set first cell of next rank
        if d < mpi_nranks() - 1
            mesh.first_cell_by_rank[d + 1] = mesh.first_cell_by_rank[d] +
                                             mesh.n_cells_by_rank[d]
        end

        #println("Cells per rank $d: ", mesh.n_cells_by_rank[d])
    end

    @assert all(x -> x >= 0, mesh.tree.mpi_ranks[1:length(mesh.tree)])
    @assert sum(mesh.n_cells_by_rank) == length(mesh.tree)
    
    return nothing
end

function get_restart_mesh_filename(restart_filename, mpi_parallel::True)
    # Get directory name
    dirname, _ = splitdir(restart_filename)

    if mpi_isroot()
        # Read mesh filename from restart file
        mesh_file = ""
        h5open(restart_filename, "r") do file
            mesh_file = read(attributes(file)["mesh_file"])
        end

        buffer = Vector{UInt8}(mesh_file)
        MPI.Bcast!(Ref(length(buffer)), mpi_root(), mpi_comm())
        MPI.Bcast!(buffer, mpi_root(), mpi_comm())
    else # non-root ranks
        count = MPI.Bcast!(Ref(0), mpi_root(), mpi_comm())
        buffer = Vector{UInt8}(undef, count[])
        MPI.Bcast!(buffer, mpi_root(), mpi_comm())
        mesh_file = String(buffer)
    end

    # Construct and return filename
    return joinpath(dirname, mesh_file)
end
end # @muladd
