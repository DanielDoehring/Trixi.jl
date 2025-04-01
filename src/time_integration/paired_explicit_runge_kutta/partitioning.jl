# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function get_n_levels(mesh::TreeMesh)
    # This assumes that the eigenvalues are of similar magnitude
    # and thus the mesh size is the main factor in the char. speed
    min_level = minimum_level(mesh.tree)
    max_level = maximum_level(mesh.tree)

    n_levels = max_level - min_level + 1

    return n_levels
end

@inline function get_n_levels(mesh::TreeMesh, alg)
    n_levels = get_n_levels(mesh)

    # CARE: This is for testcase with special assignment
    #n_levels = alg.num_methods
    #println("CARE: This is for testcase with special (random/round-robin) assignment!")

    return n_levels
end

@inline function get_n_levels(mesh::Union{P4estMesh, StructuredMesh}, alg)
    n_levels = alg.num_methods

    return n_levels
end

# TODO: Try out thread-parallelization of the assignment!

function partitioning_variables!(level_info_elements,
                                 level_info_elements_acc,
                                 level_info_interfaces_acc,
                                 level_info_boundaries_acc,
                                 level_info_boundaries_orientation_acc,
                                 level_info_mortars_acc,
                                 n_levels, n_dims, mesh::TreeMesh, dg, cache, alg)
    @unpack elements, interfaces, boundaries = cache

    max_level = maximum_level(mesh.tree)

    n_elements = length(elements.cell_ids)

    # CARE: This is for testcase with special assignment
    #element_id_level = Dict{Int, Int}()

    # Determine level for each element
    for element_id in 1:n_elements
        # Determine level
        # NOTE: For really different grid sizes
        level = mesh.tree.levels[elements.cell_ids[element_id]]

        # Convert to level id
        level_id = max_level + 1 - level

        # CARE: This is for testcase with special assignment
        #level_id = rand(1:n_levels)
        #level_id = mod(element_id - 1, n_levels) + 1 # Assign elements in round-robin fashion
        #element_id_level[element_id] = level_id

        # TODO: For case with locally changing mean speed of sound (Lin. Euler)
        #=
        c_max_el = 0.0
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element_id)

            c = u_node[end]
            if c > c_max_el
                c_max_el = c
            end
        end
        # Similar to procedure for P4est
        level_id = findfirst(x -> x < c_max_el, alg.dt_ratios)
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level_id === nothing
            level_id = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level_id = level_id - 1
        end
        =#

        push!(level_info_elements[level_id], element_id)
        # Add to accumulated container
        for l in level_id:n_levels
            push!(level_info_elements_acc[l], element_id)
        end
    end

    n_interfaces = length(interfaces.orientations)
    # Determine level for each interface
    for interface_id in 1:n_interfaces
        # Get element id: Interfaces only between elements of same size
        element_id = interfaces.neighbor_ids[1, interface_id]

        # Determine level
        level = mesh.tree.levels[elements.cell_ids[element_id]]

        level_id = max_level + 1 - level

        # CARE: This is for testcase with special assignment
        #level_id = element_id_level[element_id]

        # NOTE: For case with varying characteristic speeds
        #=
        el_id_1 = interfaces.neighbor_ids[1, interface_id]
        el_id_2 = interfaces.neighbor_ids[2, interface_id]

        level_1 = 0
        level_2 = 0

        for level in 1:n_levels
            if el_id_1 in level_info_elements[level]
                level_1 = level
                break
            end
        end

        for level in 1:n_levels
            if el_id_2 in level_info_elements[level]
                level_2 = level
                break
            end
        end
        level_id = min(level_1, level_2)
        =#

        for l in level_id:n_levels
            push!(level_info_interfaces_acc[l], interface_id)
        end
    end

    n_boundaries = length(boundaries.orientations)
    # Determine level for each boundary
    for boundary_id in 1:n_boundaries
        # Get element id (boundaries have only one unique associated element)
        element_id = boundaries.neighbor_ids[boundary_id]

        # Determine level
        level = mesh.tree.levels[elements.cell_ids[element_id]]

        # Convert to level id
        level_id = max_level + 1 - level

        # CARE: This is for testcase with special assignment
        #level_id = element_id_level[element_id]

        # Add to accumulated container
        for l in level_id:n_levels
            push!(level_info_boundaries_acc[l], boundary_id)
        end

        # For orientation-side wise specific treatment
        if boundaries.orientations[boundary_id] == 1 # x Boundary
            if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
                for l in level_id:n_levels
                    push!(level_info_boundaries_orientation_acc[l][2], boundary_id)
                end
            else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
                for l in level_id:n_levels
                    push!(level_info_boundaries_orientation_acc[l][1], boundary_id)
                end
            end
        elseif boundaries.orientations[boundary_id] == 2 # y Boundary
            if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
                for l in level_id:n_levels
                    push!(level_info_boundaries_orientation_acc[l][4], boundary_id)
                end
            else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
                for l in level_id:n_levels
                    push!(level_info_boundaries_orientation_acc[l][3], boundary_id)
                end
            end
        elseif boundaries.orientations[boundary_id] == 3 # z Boundary
            if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
                for l in level_id:n_levels
                    push!(level_info_boundaries_orientation_acc[l][6], boundary_id)
                end
            else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
                for l in level_id:n_levels
                    push!(level_info_boundaries_orientation_acc[l][5], boundary_id)
                end
            end
        end
    end

    if n_dims > 1
        @unpack mortars = cache
        n_mortars = length(mortars.orientations)

        for mortar_id in 1:n_mortars
            # This is by convention always one of the finer elements
            element_id = mortars.neighbor_ids[1, mortar_id]

            # Determine level
            level = mesh.tree.levels[elements.cell_ids[element_id]]

            # Higher element's level determines this mortars' level
            level_id = max_level + 1 - level

            # CARE: This is for testcase with special assignment
            #level_id = element_id_level[element_id]

            # Add to accumulated container
            for l in level_id:n_levels
                push!(level_info_mortars_acc[l], mortar_id)
            end
        end
    end
end

function partitioning_variables!(level_info_elements,
                                 level_info_elements_acc,
                                 level_info_interfaces_acc,
                                 level_info_boundaries_acc,
                                 level_info_boundaries_orientation_acc,
                                 level_info_mortars_acc,
                                 # MPI additions
                                 level_info_mpi_interfaces_acc,
                                 level_info_mpi_mortars_acc,
                                 n_levels, n_dims, mesh::ParallelTreeMesh{2}, dg, cache,
                                 alg)
    @unpack elements, interfaces, mpi_interfaces, boundaries = cache

    max_level = maximum_level(mesh.tree)

    n_elements = length(elements.cell_ids)
    # Determine level for each element
    for element_id in 1:n_elements
        # Determine level
        # NOTE: For really different grid sizes
        level = mesh.tree.levels[elements.cell_ids[element_id]]

        # Convert to level id
        level_id = max_level + 1 - level

        push!(level_info_elements[level_id], element_id)
        # Add to accumulated container
        for l in level_id:n_levels
            push!(level_info_elements_acc[l], element_id)
        end
    end

    n_interfaces = length(interfaces.orientations)
    # Determine level for each interface
    for interface_id in 1:n_interfaces
        # Get element id: Interfaces only between elements of same size
        element_id = interfaces.neighbor_ids[1, interface_id]

        # Determine level
        level = mesh.tree.levels[elements.cell_ids[element_id]]

        level_id = max_level + 1 - level

        for l in level_id:n_levels
            push!(level_info_interfaces_acc[l], interface_id)
        end
    end

    n_mpi_interfaces = length(mpi_interfaces.orientations)
    # Determine level for each interface
    for interface_id in 1:n_mpi_interfaces
        # Get element id: Interfaces only between elements of same size
        element_id = mpi_interfaces.local_neighbor_ids[interface_id]

        # Determine level
        level = mesh.tree.levels[elements.cell_ids[element_id]]

        level_id = max_level + 1 - level

        for l in level_id:n_levels
            push!(level_info_mpi_interfaces_acc[l], interface_id)
        end
    end

    n_boundaries = length(boundaries.orientations)
    # Determine level for each boundary
    for boundary_id in 1:n_boundaries
        # Get element id (boundaries have only one unique associated element)
        element_id = boundaries.neighbor_ids[boundary_id]

        # Determine level
        level = mesh.tree.levels[elements.cell_ids[element_id]]

        # Convert to level id
        level_id = max_level + 1 - level

        # Add to accumulated container
        for l in level_id:n_levels
            push!(level_info_boundaries_acc[l], boundary_id)
        end

        # For orientation-side wise specific treatment
        if boundaries.orientations[boundary_id] == 1 # x Boundary
            if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
                for l in level_id:n_levels
                    push!(level_info_boundaries_orientation_acc[l][2], boundary_id)
                end
            else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
                for l in level_id:n_levels
                    push!(level_info_boundaries_orientation_acc[l][1], boundary_id)
                end
            end
        elseif boundaries.orientations[boundary_id] == 2 # y Boundary
            if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
                for l in level_id:n_levels
                    push!(level_info_boundaries_orientation_acc[l][4], boundary_id)
                end
            else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
                for l in level_id:n_levels
                    push!(level_info_boundaries_orientation_acc[l][3], boundary_id)
                end
            end
        elseif boundaries.orientations[boundary_id] == 3 # z Boundary
            if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
                for l in level_id:n_levels
                    push!(level_info_boundaries_orientation_acc[l][6], boundary_id)
                end
            else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
                for l in level_id:n_levels
                    push!(level_info_boundaries_orientation_acc[l][5], boundary_id)
                end
            end
        end
    end

    if n_dims > 1
        @unpack mortars, mpi_mortars = cache
        n_mortars = length(mortars.orientations)

        for mortar_id in 1:n_mortars
            # This is by convention always one of the finer elements
            element_id = mortars.neighbor_ids[1, mortar_id]

            # Determine level
            level = mesh.tree.levels[elements.cell_ids[element_id]]

            # Higher element's level determines this mortars' level
            level_id = max_level + 1 - level
            # Add to accumulated container
            for l in level_id:n_levels
                push!(level_info_mortars_acc[l], mortar_id)
            end
        end

        n_mpi_mortars = length(mpi_mortars.orientations)
        for mortar_id in 1:n_mpi_mortars
            # This is by convention always one of the finer elements
            element_id = mpi_mortars.local_neighbor_ids[mortar_id][1]

            #=
            level = -1
            for element_id in mpi_mortars.local_neighbor_ids[mortar_id]

                # Determine level
                level_cand = mesh.tree.levels[elements.cell_ids[element_id]]

                if level_cand > level
                    level = level_cand
                end
            end
            =#

            # Determine level
            level = mesh.tree.levels[elements.cell_ids[element_id]]

            # Higher element's level determines this mortars' level
            level_id = max_level + 1 - level
            # Add to accumulated container
            for l in level_id:n_levels
                push!(level_info_mpi_mortars_acc[l], mortar_id)
            end
        end
    end
end

function partitioning_variables!(level_info_elements,
                                 level_info_elements_acc,
                                 level_info_interfaces_acc,
                                 level_info_boundaries_acc,
                                 level_info_boundaries_orientation_acc, # TODO: Not yet adapted for P4est!
                                 level_info_mortars_acc,
                                 n_levels, n_dims, mesh::P4estMesh, dg, cache, alg)
    @unpack elements, interfaces, boundaries, mortars = cache

    #nnodes = length(mesh.nodes)
    nnodes = length(dg.basis.nodes)
    n_elements = nelements(dg, cache)

    h_min_per_element, h_min, h_max = get_hmin_per_element(mesh, cache.elements,
                                                           n_elements,
                                                           nnodes,
                                                           eltype(dg.basis.nodes))

    for element_id in 1:n_elements
        h = h_min_per_element[element_id]

        # Beyond linear scaling of timestep
        level = findfirst(x -> x < h_min / h, alg.dt_ratios)
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level === nothing
            level = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level = level - 1
        end

        append!(level_info_elements[level], element_id)

        for l in level:n_levels
            push!(level_info_elements_acc[l], element_id)
        end
    end

    n_interfaces = last(size(interfaces.u))
    # Determine level for each interface
    for interface_id in 1:n_interfaces
        # For p4est: Cells on same level do not necessarily have same size
        element_id1 = interfaces.neighbor_ids[1, interface_id]
        element_id2 = interfaces.neighbor_ids[2, interface_id]
        h1 = h_min_per_element[element_id1]
        h2 = h_min_per_element[element_id2]
        h = min(h1, h2)

        # Beyond linear scaling of timestep
        level = findfirst(x -> x < h_min / h, alg.dt_ratios)
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level === nothing
            level = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level = level - 1
        end

        for l in level:n_levels
            push!(level_info_interfaces_acc[l], interface_id)
        end
    end

    n_boundaries = last(size(boundaries.u))
    # Determine level for each boundary
    for boundary_id in 1:n_boundaries
        # Get element id (boundaries have only one unique associated element)
        element_id = boundaries.neighbor_ids[boundary_id]
        h = h_min_per_element[element_id]

        # Beyond linear scaling of timestep
        level = findfirst(x -> x < h_min / h, alg.dt_ratios)
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level === nothing
            level = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level = level - 1
        end

        # Add to accumulated container
        for l in level:n_levels
            push!(level_info_boundaries_acc[l], boundary_id)
        end
    end

    # p4est is always dimension 2 or 3
    n_mortars = last(size(mortars.u))
    for mortar_id in 1:n_mortars
        # Get element ids
        element_id_lower = mortars.neighbor_ids[1, mortar_id]
        h_lower = h_min_per_element[element_id_lower]

        element_id_higher = mortars.neighbor_ids[2, mortar_id]
        h_higher = h_min_per_element[element_id_higher]

        h = min(h_lower, h_higher)

        # Beyond linear scaling of timestep
        level = findfirst(x -> x < h_min / h, alg.dt_ratios)
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level === nothing
            level = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level = level - 1
        end

        # Add to accumulated container
        for l in level:n_levels
            push!(level_info_mortars_acc[l], mortar_id)
        end
    end
end

function partitioning_variables!(level_info_elements,
                                 level_info_elements_acc,
                                 level_info_interfaces_acc,
                                 level_info_boundaries_acc,
                                 level_info_boundaries_orientation_acc, # TODO: Not yet adapted for P4est!
                                 level_info_mortars_acc,
                                 # MPI additions
                                 level_info_mpi_interfaces_acc,
                                 level_info_mpi_mortars_acc,
                                 n_levels, n_dims, mesh::ParallelP4estMesh, dg, cache,
                                 alg)
    @unpack elements, interfaces, boundaries, mortars = cache
    @unpack mpi_interfaces, mpi_mortars = cache

    #nnodes = length(mesh.nodes)
    nnodes = length(dg.basis.nodes)
    n_elements = nelements(dg, cache)

    h_min_per_element, h_min, h_max = get_hmin_per_element(mesh, cache.elements,
                                                           n_elements,
                                                           nnodes,
                                                           eltype(dg.basis.nodes))
    # Synchronize `h_min`, `h_max` to have consistent partitioning across ranks
    h_min = MPI.Allreduce!(Ref(h_min), Base.min, mpi_comm())[]
    h_max = MPI.Allreduce!(Ref(h_max), Base.max, mpi_comm())[]

    for element_id in 1:n_elements
        h = h_min_per_element[element_id]

        # Beyond linear scaling of timestep
        level = findfirst(x -> x < h_min / h, alg.dt_ratios)
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level === nothing
            level = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level = level - 1
        end

        append!(level_info_elements[level], element_id)

        for l in level:n_levels
            push!(level_info_elements_acc[l], element_id)
        end
    end

    n_interfaces = last(size(interfaces.u))
    # Determine level for each interface
    for interface_id in 1:n_interfaces
        # For p4est: Cells on same level do not necessarily have same size
        element_id1 = interfaces.neighbor_ids[1, interface_id]
        element_id2 = interfaces.neighbor_ids[2, interface_id]
        h1 = h_min_per_element[element_id1]
        h2 = h_min_per_element[element_id2]
        h = min(h1, h2)

        # Beyond linear scaling of timestep
        level = findfirst(x -> x < h_min / h, alg.dt_ratios)
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level === nothing
            level = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level = level - 1
        end

        for l in level:n_levels
            push!(level_info_interfaces_acc[l], interface_id)
        end
    end

    n_mpi_interfaces = last(size(mpi_interfaces.u))
    # Determine level for each interface
    for interface_id in 1:n_mpi_interfaces
        # For p4est: Cells on same level do not necessarily have same size
        element_id1 = interfaces.neighbor_ids[1, interface_id]
        element_id2 = interfaces.neighbor_ids[2, interface_id]
        h1 = h_min_per_element[element_id1]
        h2 = h_min_per_element[element_id2]
        h = min(h1, h2)

        # Beyond linear scaling of timestep
        level = findfirst(x -> x < h_min / h, alg.dt_ratios)
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level === nothing
            level = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level = level - 1
        end

        for l in level:n_levels
            push!(level_info_mpi_interfaces_acc[l], interface_id)
        end
    end

    n_boundaries = last(size(boundaries.u))
    # Determine level for each boundary
    for boundary_id in 1:n_boundaries
        # Get element id (boundaries have only one unique associated element)
        element_id = boundaries.neighbor_ids[boundary_id]
        h = h_min_per_element[element_id]

        # Beyond linear scaling of timestep
        level = findfirst(x -> x < h_min / h, alg.dt_ratios)
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level === nothing
            level = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level = level - 1
        end

        # Add to accumulated container
        for l in level:n_levels
            push!(level_info_boundaries_acc[l], boundary_id)
        end
    end

    # p4est is always dimension 2 or 3
    n_mortars = last(size(mortars.u))
    for mortar_id in 1:n_mortars
        # Get element ids
        element_id_lower = mortars.neighbor_ids[1, mortar_id]
        h_lower = h_min_per_element[element_id_lower]

        element_id_higher = mortars.neighbor_ids[2, mortar_id]
        h_higher = h_min_per_element[element_id_higher]

        h = min(h_lower, h_higher)

        # Beyond linear scaling of timestep
        level = findfirst(x -> x < h_min / h, alg.dt_ratios)
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level === nothing
            level = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level = level - 1
        end

        # Add to accumulated container
        for l in level:n_levels
            push!(level_info_mortars_acc[l], mortar_id)
        end
    end

    n_mpi_mortars = last(size(mpi_mortars.u))
    for mortar_id in 1:n_mortars
        # Get element ids
        element_id_lower = mortars.neighbor_ids[1, mortar_id]
        h_lower = h_min_per_element[element_id_lower]

        element_id_higher = mortars.neighbor_ids[2, mortar_id]
        h_higher = h_min_per_element[element_id_higher]

        h = min(h_lower, h_higher)

        # Beyond linear scaling of timestep
        level = findfirst(x -> x < h_min / h, alg.dt_ratios)
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level === nothing
            level = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level = level - 1
        end

        # Add to accumulated container
        for l in level:n_levels
            push!(level_info_mpi_mortars_acc[l], mortar_id)
        end
    end
end

function partitioning_variables!(level_info_elements,
                                 level_info_elements_acc,
                                 level_info_interfaces_acc,
                                 level_info_boundaries_acc,
                                 level_info_boundaries_orientation_acc, # TODO: Not yet adapted for StructuredMesh
                                 level_info_mortars_acc,
                                 n_levels, n_dims, mesh::StructuredMesh, dg, cache, alg)
    nnodes = length(dg.basis.nodes)
    n_elements = nelements(dg, cache)

    h_min_per_element, h_min, h_max = get_hmin_per_element(mesh, cache.elements,
                                                           n_elements,
                                                           nnodes,
                                                           eltype(dg.basis.nodes))

    # For "grid-based" partitioning approach
    #=
    S_min = alg.num_stage_evals_min
    S_max = alg.num_stages
    n_levels = Int((S_max - S_min) / 2) + 1 # Linearly increasing levels
    h_bins = LinRange(h_min, h_max, n_levels + 1) # These are the intervals
    println("h_bins:")
    display(h_bins)
    =#

    for element_id in 1:n_elements
        h = h_min_per_element[element_id]

        #=
        # This approach is "grid-based" in the sense that 
        # the entire grid range gets mapped linearly onto the available methods
        level = findfirst(x -> x >= h, h_bins) - 1
        # Catch case h = h_min
        if level == 0
            level = 1
        end
        =#

        # This approach is "method-based" in the sense that
        # the available methods get mapped linearly onto the grid, with cut-off for the too-coarse cells
        level = findfirst(x -> x < h_min / h, alg.dt_ratios)
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level === nothing
            level = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level = level - 1
        end

        append!(level_info_elements[level], element_id)

        for l in level:n_levels
            push!(level_info_elements_acc[l], element_id)
        end
    end

    # No interfaces, boundaries, mortars for structured meshes
end

function get_hmin_per_element(mesh::StructuredMesh{1}, elements, n_elements, nnodes,
                              RealT)
    h_min = floatmax(RealT)
    h_max = zero(RealT)

    hmin_per_element = zeros(n_elements)

    for element_id in 1:n_elements
        P0 = elements.node_coordinates[1, 1, element_id]
        P1 = elements.node_coordinates[1, nnodes, element_id]
        h = abs(P1 - P0) # Assumes P1 > P0

        hmin_per_element[element_id] = h
        if h > h_max
            h_max = h
        end
        if h < h_min
            h_min = h
        end
    end

    println("h_min: ", h_min, " h_max: ", h_max)
    println("h_max/h_min: ", h_max / h_min)
    println("\n")

    return hmin_per_element, h_min, h_max
end

function get_hmin_per_element(mesh::Union{P4estMesh{2}, StructuredMesh{2}}, elements,
                              n_elements, nnodes, RealT)
    h_min = floatmax(RealT)
    h_max = zero(RealT)

    hmin_per_element = zeros(n_elements)

    for element_id in 1:n_elements
        # pull the four corners numbered as

        #            <----
        #        3-----------2
        #        |           |
        #     |  |           |  ^
        #     |  |           |  |
        #     v  |           |  |
        #        |           |
        #        0-----------1
        #            ---->    
        #  ^ η
        #  |
        #  |----> ξ

        P0 = elements.node_coordinates[:, 1, 1, element_id]
        P1 = elements.node_coordinates[:, nnodes, 1, element_id]
        P2 = elements.node_coordinates[:, nnodes, nnodes, element_id]
        P3 = elements.node_coordinates[:, 1, nnodes, element_id]

        # In 2D four edges need to be checked
        L0 = norm(P1 - P0)
        L1 = norm(P2 - P1)
        L2 = norm(P3 - P2)
        L3 = norm(P0 - P3)

        h = min(L0, L1, L2, L3)
        # For square elements (RTI)
        #L0 = abs(P1[1] - P0[1])
        #h = L0
        hmin_per_element[element_id] = h

        # Set global `h_min` and `h_max`, i.e., for the entire mesh
        if h > h_max
            h_max = h
        end
        if h < h_min
            h_min = h
        end
    end

    println("h_min: ", h_min, " h_max: ", h_max)
    println("h_max/h_min: ", h_max / h_min, "\n")

    return hmin_per_element, h_min, h_max
end

function get_hmin_per_element(mesh::P4estMesh{3}, elements,
                              n_elements, nnodes, RealT)
    h_min = floatmax(RealT)
    h_max = zero(RealT)

    hmin_per_element = zeros(n_elements)

    for element_id in 1:n_elements
        # pull the eight corners numbered as

        #            7----------6
        #          / |         /|
        #         /  |        / | 
        #        3-----------2  |
        #        |   |       |  |
        #        |   |       |  |
        #        |   4-------|--5
        #        |  /        | /
        #        | /         |/
        #        0-----------1
        #           
        #  ^ η
        #  |   ζ
        #  |  / 
        #  | / 
        #  |----> ξ

        # "Front face"
        P0 = elements.node_coordinates[:, 1, 1, 1, element_id]
        P1 = elements.node_coordinates[:, nnodes, 1, 1, element_id]
        P2 = elements.node_coordinates[:, nnodes, nnodes, 1, element_id]
        P3 = elements.node_coordinates[:, 1, nnodes, 1, element_id]

        # "Back face"
        P4 = elements.node_coordinates[:, 1, 1, nnodes, element_id]
        P5 = elements.node_coordinates[:, nnodes, 1, nnodes, element_id]
        P6 = elements.node_coordinates[:, nnodes, nnodes, nnodes, element_id]
        P7 = elements.node_coordinates[:, 1, nnodes, nnodes, element_id]

        # In 3D, there are 12 edges that need to be checked

        # "Front" face
        L0 = norm(P1 - P0)
        L1 = norm(P2 - P1)
        L2 = norm(P3 - P2)
        L3 = norm(P0 - P3)

        # "Back" face
        L4 = norm(P5 - P4)
        L5 = norm(P6 - P5)
        L6 = norm(P7 - P6)
        L7 = norm(P4 - P7)

        # "Connecting" edges
        L8 = norm(P0 - P4)
        L9 = norm(P1 - P5)
        L10 = norm(P2 - P6)
        L11 = norm(P3 - P7)

        h = min(L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11)
        hmin_per_element[element_id] = h

        # Set global `h_min` and `h_max`, i.e., for the entire mesh
        if h > h_max
            h_max = h
        end
        if h < h_min
            h_min = h
        end
    end

    println("h_min: ", h_min, " h_max: ", h_max)
    println("h_max/h_min: ", h_max / h_min, "\n")

    return hmin_per_element, h_min, h_max
end

# TODO: 3D versions of "get_hmin_per_element"
# TODO: T8Code extensions

@inline function partitioning_u!(level_u_indices_elements, level_info_elements,
                                 n_levels, u_ode, mesh, equations, dg, cache)
    u = wrap_array(u_ode, mesh, equations, dg, cache)

    for level in 1:n_levels
        @views indices = collect(Iterators.flatten(LinearIndices(u)[..,
                                                                    level_info_elements[level]]))
        append!(level_u_indices_elements[level], indices)
        sort!(level_u_indices_elements[level])
    end
end

# Repartitioning of the DoF array of the gravity solver
@inline function partitioning_u_gravity!(integrator::AbstractPairedExplicitRKMultiIntegrator)
    @unpack level_info_elements, n_levels = integrator
    @unpack semi_gravity, cache = integrator.p

    u_gravity = wrap_array(cache.u_ode, semi_gravity)

    resize!(cache.level_u_gravity_indices_elements, n_levels)
    for level in 1:n_levels
        if isassigned(cache.level_u_gravity_indices_elements, level)
            empty!(cache.level_u_gravity_indices_elements[level])
        else
            cache.level_u_gravity_indices_elements[level] = []
        end
    end

    for level in 1:n_levels
        @views indices = collect(Iterators.flatten(LinearIndices(u_gravity)[..,
                                                                            level_info_elements[level]]))
        append!(cache.level_u_gravity_indices_elements[level], indices)
        sort!(cache.level_u_gravity_indices_elements[level])
    end
end
end # @muladd
