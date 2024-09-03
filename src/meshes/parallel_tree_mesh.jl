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
    # Assign first cell to rank 0 (employ depth-first indexing)
    mesh.first_cell_by_rank[0] = 1
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
        # Assign cells to rank following depth-first indexing
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

function partition_PERK!(mesh::ParallelTreeMesh; allow_coarsening = true)
    leaves = leaf_cells(mesh.tree)
    num_leaves = length(leaves)
    num_cells = length(mesh.tree)
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

    leaves_per_rank = Dict{Int, Set{Int}}(i => Set{Int}() for i in 0:mpi_nranks() - 1)
    # Partition leaf-nodes level-wise equally among ranks
    for (_, leaves) in leaves_per_level
        num_leaves_at_level = length(leaves)
        leaves_per_rank_at_level = ceil(Int, num_leaves_at_level / mpi_nranks())
    
        for rank in 0:(mpi_nranks() - 1)
            start_idx = rank * leaves_per_rank_at_level + 1
            end_idx = min((rank + 1) * leaves_per_rank_at_level, num_leaves_at_level)
            
            if start_idx <= end_idx
                union!(leaves_per_rank[rank], leaves[start_idx:end_idx])
            end
        end
    end

    #non_leaves = setdiff(1:num_cells, leaves) # Not sure what is faster
    non_leaves = non_leaf_cells(mesh.tree)

    n_non_leaves = num_cells - num_leaves
    n_non_leaves_per_rank = ceil(Int, n_non_leaves / mpi_nranks())

    # Distribute non leaf cells evenly among ranks
    cells_per_rank = Dict{Int, Set{Int}}(i => Set{Int}() for i in 0:mpi_nranks() - 1)

    # Distribute non leaf cells among ranks
    for rank in 0:(mpi_nranks() - 1)
        start_idx = rank * n_non_leaves_per_rank + 1
        end_idx = min((rank + 1) * n_non_leaves_per_rank, n_non_leaves)

        cells = non_leaves[start_idx:end_idx]
        union!(cells_per_rank[rank], cells)
    end

    # In order to have AMR working we need to keep children of parent together when
    # all children are leaves.
    for rank in 0:(mpi_nranks() - 1)
        leaves = leaves_per_rank[rank]
        for leaf in leaves
            parent_id = mesh.tree.parent_ids[leaf]
            if allow_coarsening &&
                all(id -> is_leaf(mesh.tree, id), @view mesh.tree.child_ids[:, parent_id])

                additional_leaves = mesh.tree.child_ids[:, parent_id]
                union!(leaves, additional_leaves)

                # Add parent to same rank
                union!(cells_per_rank[rank], parent_id)

                # Remove these leaf cells from the other ranks
                for other_rank in 0:rank - 1
                    for leaf in additional_leaves
                        delete!(leaves_per_rank[other_rank], leaf)
                    end
                    delete!(cells_per_rank[other_rank], parent_id)
                end
                for other_rank in (rank + 1):(mpi_nranks() - 1)
                    for leaf in additional_leaves
                        delete!(leaves_per_rank[other_rank], leaf)
                    end
                    delete!(cells_per_rank[other_rank], parent_id)
                end
            end
        end
    end
    sum_assigned_leafs = sum(length(leaves_per_rank[rank]) for rank in 0:(mpi_nranks() - 1))
    @assert sum_assigned_leafs == num_leaves

    for rank in 0:(mpi_nranks() - 1)
        mesh.tree.mpi_ranks[collect(leaves_per_rank[rank])] .= rank
        mesh.tree.mpi_ranks[collect(cells_per_rank[rank])] .= rank
    end
    @assert sum(length(cells_per_rank[rank]) for rank in 0:(mpi_nranks() - 1)) + sum_assigned_leafs == num_cells

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
