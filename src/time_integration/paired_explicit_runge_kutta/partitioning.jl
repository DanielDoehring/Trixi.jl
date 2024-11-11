function get_n_levels(mesh::TreeMesh, alg)
    # This assumes that the eigenvalues are of similar magnitude
    # and thus the mesh size is the main factor in the char. speed
    min_level = minimum_level(mesh.tree)
    max_level = maximum_level(mesh.tree)

    n_levels = max_level - min_level + 1

    return n_levels
end

function get_n_levels(mesh::Union{P4estMesh, StructuredMesh}, alg)
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
    # Determine level for each element
    for element_id in 1:n_elements
        # Determine level
        # NOTE: For really different grid sizes
        level = mesh.tree.levels[elements.cell_ids[element_id]]

        # Convert to level id
        level_id = max_level + 1 - level

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
                                 level_info_mpi_interfaces_acc,
                                 level_info_boundaries_acc,
                                 level_info_boundaries_orientation_acc,
                                 level_info_mortars_acc,
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
    @unpack elements, interfaces, boundaries = cache

    nnodes = length(mesh.nodes)
    n_elements = nelements(dg, cache)

    h_min_per_element, h_min, h_max = get_hmin_per_element(mesh, cache.elements, n_elements,
                                                           nnodes, eltype(dg.basis.nodes))

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

    if n_dims > 1
        @unpack mortars = cache # TODO: Could also make dimensionality check
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
end

function partitioning_variables!(level_info_elements,
                                 level_info_elements_acc,
                                 level_info_interfaces_acc,
                                 level_info_boundaries_acc,
                                 level_info_boundaries_orientation_acc, # TODO: Not yet adapted for P4est!
                                 level_info_mortars_acc,
                                 n_levels, n_dims, mesh::StructuredMesh, dg, cache, alg)
    nnodes = length(dg.basis.nodes)
    n_elements = nelements(dg, cache)

    h_min_per_element, h_min, h_max = get_hmin_per_element(mesh, cache.elements, n_elements,
                                                           nnodes, eltype(dg.basis.nodes))

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

function get_hmin_per_element(mesh::StructuredMesh{1}, elements, n_elements, nnodes, RealT)
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
        # pull the four corners numbered as right-handed
        P0 = elements.node_coordinates[:, 1, 1, element_id]
        P1 = elements.node_coordinates[:, nnodes, 1, element_id]
        P2 = elements.node_coordinates[:, nnodes, nnodes, element_id]
        P3 = elements.node_coordinates[:, 1, nnodes, element_id]
        # compute the four side lengths and get the smallest
        L0 = sqrt(sum((P1 - P0) .^ 2))
        L1 = sqrt(sum((P2 - P1) .^ 2))
        L2 = sqrt(sum((P3 - P2) .^ 2))
        L3 = sqrt(sum((P0 - P3) .^ 2))
        h = min(L0, L1, L2, L3)

        # For square elements (RTI)
        #L0 = abs(P1[1] - P0[1])
        #h = L0

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

# TODO: 3D versions of "get_hmin_per_element"
# TODO: T8Code extensions

function partitioning_u!(level_u_indices_elements,
                         n_levels, n_dims, level_info_elements, u_ode, mesh, equations, dg,
                         cache)
    u = wrap_array(u_ode, mesh, equations, dg, cache)

    if n_dims == 1
        for level in 1:n_levels
            for element_id in level_info_elements[level]
                # First dimension of u: nvariables, following: nnodes (per dim) last: nelements                                    
                indices = vec(transpose(LinearIndices(u)[:, :, element_id]))
                append!(level_u_indices_elements[level], indices)
            end
            sort!(level_u_indices_elements[level])
            @assert length(level_u_indices_elements[level]) ==
                    nvariables(equations) * Trixi.nnodes(dg)^ndims(mesh) *
                    length(level_info_elements[level])
        end
    elseif n_dims == 2
        for level in 1:n_levels
            for element_id in level_info_elements[level]
                # First dimension of u: nvariables, following: nnodes (per dim) last: nelements
                indices = collect(Iterators.flatten(LinearIndices(u)[:, :, :,
                                                                     element_id]))
                append!(level_u_indices_elements[level], indices)
            end
            sort!(level_u_indices_elements[level])
            @assert length(level_u_indices_elements[level]) ==
                    nvariables(equations) * Trixi.nnodes(dg)^ndims(mesh) *
                    length(level_info_elements[level])
        end
    elseif n_dims == 3
        for level in 1:n_levels
            for element_id in level_info_elements[level]
                # First dimension of u: nvariables, following: nnodes (per dim) last: nelements
                indices = collect(Iterators.flatten(LinearIndices(u)[:, :, :, :,
                                                                     element_id]))
                append!(level_u_indices_elements[level], indices)
            end
            sort!(level_u_indices_elements[level])
            @assert length(level_u_indices_elements[level]) ==
                    nvariables(equations) * Trixi.nnodes(dg)^ndims(mesh) *
                    length(level_info_elements[level])
        end
    end
end
