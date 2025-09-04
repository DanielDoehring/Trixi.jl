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

    # CARE: This is for testcases with special (random/round robin) assignment
    n_levels = alg.num_methods

    return n_levels
end

@inline function get_n_levels(mesh::Union{P4estMesh, StructuredMesh}, alg)
    n_levels = alg.num_methods

    return n_levels
end

# Version with DIFFERENT number of stages and partitioning for hyperbolic and parabolic part

@inline function get_n_levels(mesh::Union{P4estMesh, StructuredMesh},
                              alg::AbstractPairedExplicitRKSplitMulti)
    n_levels = alg.num_methods
    n_levels_para = alg.num_methods_para

    return n_levels, n_levels_para
end

# TODO: Try out (thread-)parallelization of the assignment!

function partition_variables!(level_info_elements,
                              level_info_elements_acc,
                              level_info_interfaces_acc,
                              level_info_boundaries_acc,
                              level_info_mortars_acc,
                              n_levels,
                              semi::AbstractSemidiscretization,
                              alg)
    mesh, _, dg, cache = mesh_equations_solver_cache(semi)

    partition_variables!(level_info_elements,
                         level_info_elements_acc,
                         level_info_interfaces_acc,
                         level_info_boundaries_acc,
                         level_info_mortars_acc,
                         n_levels, mesh, dg, cache, alg)
end

function partition_variables!(level_info_elements,
                              level_info_elements_acc,
                              level_info_interfaces_acc,
                              level_info_boundaries_acc,
                              level_info_mortars_acc,
                              n_levels,
                              semi::AbstractSemidiscretization,
                              alg, u_ode)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

    u = wrap_array(u_ode, semi)

    partition_variables!(level_info_elements,
                         level_info_elements_acc,
                         level_info_interfaces_acc,
                         level_info_boundaries_acc,
                         level_info_mortars_acc,
                         n_levels,
                         mesh, dg, cache,
                         alg,
                         equations, u)
end

function partition_variables!(level_info_elements,
                              level_info_elements_acc,
                              level_info_interfaces_acc,
                              level_info_boundaries_acc,
                              level_info_mortars_acc,
                              n_levels,
                              semi::SemidiscretizationHyperbolicParabolic,
                              alg;
                              quadratic_scaling = false)
    mesh, _, dg, cache = mesh_equations_solver_cache(semi)

    partition_variables!(level_info_elements,
                         level_info_elements_acc,
                         level_info_interfaces_acc,
                         level_info_boundaries_acc,
                         level_info_mortars_acc,
                         n_levels, mesh, dg, cache, alg,
                         quadratic_scaling = quadratic_scaling)
end

function partition_variables!(level_info_elements,
                              level_info_elements_acc,
                              level_info_interfaces_acc,
                              level_info_boundaries_acc,
                              level_info_mortars_acc,
                              # MPI additions
                              level_info_mpi_interfaces_acc,
                              level_info_mpi_mortars_acc,
                              n_levels,
                              semi::AbstractSemidiscretization,
                              alg)
    mesh, _, dg, cache = mesh_equations_solver_cache(semi)

    partition_variables!(level_info_elements,
                         level_info_elements_acc,
                         level_info_interfaces_acc,
                         level_info_boundaries_acc,
                         level_info_mortars_acc,
                         # MPI additions
                         level_info_mpi_interfaces_acc,
                         level_info_mpi_mortars_acc,
                         n_levels,
                         mesh, dg, cache, alg)
end

function partition_variables!(level_info_elements,
                              level_info_elements_acc,
                              level_info_interfaces_acc,
                              level_info_boundaries_acc,
                              level_info_mortars_acc,
                              n_levels, mesh::TreeMesh, dg, cache,
                              alg)
    @unpack elements, interfaces, boundaries = cache

    max_level = maximum_level(mesh.tree)

    n_elements = length(elements.cell_ids)

    # CARE: This is for testcase with special assignment
    element_id_level = Dict{Int, Int}()

    # Determine level for each element
    for element_id in 1:n_elements
        # Determine level
        # NOTE: For really different grid sizes
        level = mesh.tree.levels[elements.cell_ids[element_id]]

        # Convert to level id
        level_id = max_level + 1 - level

        # CARE: This is for testcase with special assignment
        #level_id = rand(1:n_levels)
        level_id = mod(element_id - 1, n_levels) + 1 # Assign elements in round-robin fashion
        element_id_level[element_id] = level_id

        # CARE: For case with locally changing mean speed of sound (Lin. Euler)
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
        # In 1D, interfaces are not necessarily between elements of same size
        element_id1 = interfaces.neighbor_ids[1, interface_id]
        element_id2 = interfaces.neighbor_ids[2, interface_id]

        # Determine level
        level1 = mesh.tree.levels[elements.cell_ids[element_id1]]
        level2 = mesh.tree.levels[elements.cell_ids[element_id2]]

        # Assign interface to the higher level of the two elements
        level_id = max_level + 1 - max(level1, level2)

        # CARE: This is for testcase with special assignment
        element_level1 = element_id_level[element_id1]
        element_level2 = element_id_level[element_id2]
        level_id = min(element_level1, element_level2)

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
    end

    if ndims(mesh) > 1
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

    return nothing
end

function partition_variables!(level_info_elements,
                              level_info_elements_acc,
                              level_info_interfaces_acc,
                              level_info_boundaries_acc,
                              level_info_mortars_acc,
                              n_levels, mesh::TreeMesh, dg, cache,
                              alg::AbstractPairedExplicitRKIMEXMulti)
    @unpack elements, interfaces, boundaries = cache

    max_level = maximum_level(mesh.tree)

    n_elements = length(elements.cell_ids)

    # CARE: This is for testcase with special assignment
    element_id_level = Dict{Int, Int}()

    # Determine level for each element
    for element_id in 1:n_elements
        # Determine level
        # NOTE: For really different grid sizes
        level = mesh.tree.levels[elements.cell_ids[element_id]]
        # Convert to level id
        level_id = max_level + 1 - level

        # CARE: This is for testcase with special assignment
        level_id = rand(1:n_levels)

        level_id = mod(element_id - 1, n_levels) + 1 # Assign elements in round-robin fashion
        #level_id = mod(element_id - 1, n_levels - 1) + 2 # Assign elements in round-robin fashion
        #level_id = 1

        element_id_level[element_id] = level_id

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
        # Exclude pushes to first level with is integrated implicitly
        if level_id == 1
            push!(level_info_elements_acc[1], element_id)
        else
            for l in level_id:n_levels
                push!(level_info_elements_acc[l], element_id)
            end
        end
    end

    # These are the finest cells, which should be integrated with the IMEX method,
    # which is due to historical reasons of the implementation the *LAST* method 
    # in the time integration algorithm.
    # Thus, we need to move it to the end and everything else to the front.
    first_entry = level_info_elements[1]
    for l in 1:(n_levels - 1)
        level_info_elements[l] = level_info_elements[l + 1]
    end
    level_info_elements[n_levels] = first_entry

    first_entry = level_info_elements_acc[1]
    for l in 1:(n_levels - 1)
        level_info_elements_acc[l] = level_info_elements_acc[l + 1]
    end
    level_info_elements_acc[n_levels] = first_entry

    n_interfaces = length(interfaces.orientations)
    # Determine level for each interface
    for interface_id in 1:n_interfaces
        # In 1D, interfaces are not necessarily between elements of same size
        element_id1 = interfaces.neighbor_ids[1, interface_id]
        element_id2 = interfaces.neighbor_ids[2, interface_id]

        # Determine level
        level1 = mesh.tree.levels[elements.cell_ids[element_id1]]
        level2 = mesh.tree.levels[elements.cell_ids[element_id2]]

        level_id = max_level + 1 - max(level1, level2)

        # CARE: This is for testcase with special assignment
        element_level1 = element_id_level[element_id1]
        element_level2 = element_id_level[element_id2]
        level_id = min(element_level1, element_level2)

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

        # For interfaces, we need a slightly different logic for the IMEX case.
        # This is due to the fact that the smaller element has actually less stage evaluations, but
        # is used for the finest cells.
        # As a consequence, we do not want to add every fine interface integrated with the
        # few-stage evaluation implicit method to the other, coarser levels.
        # Interfaces at partition borders, however, need to be added also to the next finest, i.e., 
        # second finest level and so.
        if level_id == 1 # On finest level
            push!(level_info_interfaces_acc[1], interface_id)

            #if level1 != level2 # At interface between differently sized cells
            # CARE: This is for testcase with special assignment
            if element_level1 != element_level2 # At interface between differently sized cells
                for l in 2:n_levels # Add to all other levels
                    push!(level_info_interfaces_acc[l], interface_id)
                end
            end
        else
            for l in level_id:n_levels
                push!(level_info_interfaces_acc[l], interface_id)
            end
        end
    end

    # These are the finest interfaces, which should be integrated with the IMEX method,
    # which is due to historical reasons of the implementation the *LAST* method 
    # in the time integration algorithm.
    # Thus, we need to move it to the end and everything else to the front.
    first_entry = level_info_interfaces_acc[1]
    for l in 1:(n_levels - 1)
        level_info_interfaces_acc[l] = level_info_interfaces_acc[l + 1]
    end
    level_info_interfaces_acc[n_levels] = first_entry

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
        level_id = element_id_level[element_id]

        # Add to accumulated container
        if level_id == 1
            push!(level_info_boundaries_acc[1], boundary_id)
        else
            for l in level_id:n_levels
                push!(level_info_boundaries_acc[l], boundary_id)
            end
        end
    end

    # These are the finest boundaries, which should be integrated with the IMEX method,
    # which is due to historical reasons of the implementation the *LAST* method 
    # in the time integration algorithm.
    # Thus, we need to move it to the end and everything else to the front.
    first_entry = level_info_boundaries_acc[1]
    for l in 1:(n_levels - 1)
        level_info_boundaries_acc[l] = level_info_boundaries_acc[l + 1]
    end
    level_info_boundaries_acc[n_levels] = first_entry

    if ndims(mesh) > 1
        # TODO: Mortars are not taken care of for IMEX!
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
            level_id = element_id_level[element_id]

            # Add to accumulated container
            if level_id == 1
                push!(level_info_mortars_acc[1], mortar_id)
            else
                for l in level_id:n_levels
                    push!(level_info_mortars_acc[l], mortar_id)
                end
            end
        end

        # These are the finest mortars, which should be integrated with the IMEX method,
        # which is due to historical reasons of the implementation the *LAST* method 
        # in the time integration algorithm.
        # Thus, we need to move it to the end and everything else to the front.
        first_entry = level_info_mortars_acc[1]
        for l in 1:(n_levels - 1)
            level_info_mortars_acc[l] = level_info_mortars_acc[l + 1]
        end
        level_info_mortars_acc[n_levels] = first_entry
    end

    return nothing
end

function partition_variables!(level_info_elements,
                              level_info_elements_acc,
                              level_info_interfaces_acc,
                              level_info_boundaries_acc,
                              level_info_mortars_acc,
                              # MPI additions
                              level_info_mpi_interfaces_acc,
                              level_info_mpi_mortars_acc,
                              n_levels, mesh::ParallelTreeMesh{2}, dg, cache,
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
        # Get element id: Interfaces only between elements of same size for 2D TreeMesh
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
    end

    # Dispatched for 2D TreeMesh => always check for mortars
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

    return nothing
end

function partition_variables!(level_info_elements,
                              level_info_elements_acc,
                              level_info_interfaces_acc,
                              level_info_boundaries_acc,
                              level_info_mortars_acc,
                              n_levels,
                              mesh::P4estMesh, dg, cache,
                              alg,
                              equations = nothing, u = nothing;
                              quadratic_scaling = false)
    @unpack elements, interfaces, boundaries, mortars = cache

    if quadratic_scaling
        dt_scaling_order = 2
        dt_ratios = alg.dt_ratios_para
    else
        dt_scaling_order = 1
        dt_ratios = alg.dt_ratios
    end

    nnodes = length(dg.basis.nodes)
    n_elements = nelements(dg, cache)

    if u !== nothing
        hmin_per_element_, hmin, hmax = dtmax_per_element(u, mesh, equations, dg, cache)
    else
        hmin_per_element_, hmin, hmax = hmin_per_element(mesh, cache.elements,
                                                         n_elements, nnodes)
    end

    println("hmin: ", hmin, " hmax: ", hmax)
    println("hmax/hmin: ", hmax / hmin, "\n")

    for element_id in 1:n_elements
        h = hmin_per_element_[element_id]

        # Partitioning strategy:
        # Use highest method for smallest elements, since these govern the stable timestep.
        # Cells that are "too coarse" are "cut off" and integrated with the lowest method.
        level = findfirst(x -> x < (hmin / h)^dt_scaling_order, dt_ratios)
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
        h1 = hmin_per_element_[element_id1]
        h2 = hmin_per_element_[element_id2]
        h = min(h1, h2)

        level = findfirst(x -> x < (hmin / h)^dt_scaling_order, dt_ratios)
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
        h = hmin_per_element_[element_id]

        level = findfirst(x -> x < (hmin / h)^dt_scaling_order, dt_ratios)
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
        h_lower = hmin_per_element_[element_id_lower]

        element_id_higher = mortars.neighbor_ids[2, mortar_id]
        h_higher = hmin_per_element_[element_id_higher]

        h = min(h_lower, h_higher)

        level = findfirst(x -> x < (hmin / h)^dt_scaling_order, dt_ratios)
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

    return nothing
end

function partition_variables!(level_info_elements,
                              level_info_elements_acc,
                              level_info_interfaces_acc,
                              level_info_boundaries_acc,
                              level_info_mortars_acc,
                              n_levels,
                              mesh::P4estMesh, dg, cache,
                              alg::AbstractPairedExplicitRKIMEXMulti,
                              equations = nothing, u = nothing;
                              quadratic_scaling = false)
    @unpack elements, interfaces, boundaries, mortars = cache

    if quadratic_scaling
        dt_scaling_order = 2
        dt_ratios = alg.dt_ratios_para
    else
        dt_scaling_order = 1
        dt_ratios = alg.dt_ratios
    end

    nnodes = length(dg.basis.nodes)
    n_elements = nelements(dg, cache)

    if u !== nothing
        hmin_per_element_, hmin, hmax = dtmax_per_element(u, mesh, equations, dg, cache)
    else
        hmin_per_element_, hmin, hmax = hmin_per_element(mesh, cache.elements,
                                                         n_elements, nnodes)
    end

    #println("hmin: ", hmin, " hmax: ", hmax)
    #println("hmax/hmin: ", hmax / hmin, "\n")

    for element_id in 1:n_elements
        h = hmin_per_element_[element_id]

        level = findfirst(x -> x < (hmin / h)^dt_scaling_order, dt_ratios)
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level === nothing
            level = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level = level - 1
        end

        append!(level_info_elements[level], element_id)

        # Add to accumulated container
        # Exclude pushes to first level with is integrated implicitly
        if level == 1
            push!(level_info_elements_acc[1], element_id)
        else
            for l in level:n_levels
                push!(level_info_elements_acc[l], element_id)
            end
        end
    end

    # These are the finest cells, which should be integrated with the IMEX method,
    # which is due to historical reasons of the implementation the *LAST* method 
    # in the time integration algorithm.
    # Thus, we need to move it to the end and everything else to the front.
    first_entry = level_info_elements[1]
    for l in 1:(n_levels - 1)
        level_info_elements[l] = level_info_elements[l + 1]
    end
    level_info_elements[n_levels] = first_entry

    first_entry = level_info_elements_acc[1]
    for l in 1:(n_levels - 1)
        level_info_elements_acc[l] = level_info_elements_acc[l + 1]
    end
    level_info_elements_acc[n_levels] = first_entry

    n_interfaces = last(size(interfaces.u))
    # Determine level for each interface
    for interface_id in 1:n_interfaces
        # For p4est: Cells on same level do not necessarily have same size
        element_id1 = interfaces.neighbor_ids[1, interface_id]
        element_id2 = interfaces.neighbor_ids[2, interface_id]
        h1 = hmin_per_element_[element_id1]
        h2 = hmin_per_element_[element_id2]

        level1 = findfirst(x -> x < (hmin / h1)^dt_scaling_order, dt_ratios)
        level2 = findfirst(x -> x < (hmin / h2)^dt_scaling_order, dt_ratios)

        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level1 === nothing
            level1 = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level1 = level1 - 1
        end
        if level2 === nothing
            level2 = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level2 = level2 - 1
        end

        level_id = min(level1, level2)

        # For interfaces, we need a slightly different logic for the IMEX case.
        # This is due to the fact that the smaller element has actually less stage evaluations, but
        # is used for the finest cells.
        # As a consequence, we do not want to add every fine interface integrated with the
        # few-stage evaluation implicit method to the other, coarser levels.
        # Interfaces at partition borders, however, need to be added also to the next finest, i.e., 
        # second finest level and so.
        if level_id == 1 # On finest level
            push!(level_info_interfaces_acc[1], interface_id)

            if level1 != level2 # At interface between differently sized cells
                for l in 2:n_levels # Add to all other levels
                    push!(level_info_interfaces_acc[l], interface_id)
                end
            end
        else
            for l in level_id:n_levels
                push!(level_info_interfaces_acc[l], interface_id)
            end
        end
    end

    # These are the finest interfaces, which should be integrated with the IMEX method,
    # which is due to historical reasons of the implementation the *LAST* method 
    # in the time integration algorithm.
    # Thus, we need to move it to the end and everything else to the front.
    first_entry = level_info_interfaces_acc[1]
    for l in 1:(n_levels - 1)
        level_info_interfaces_acc[l] = level_info_interfaces_acc[l + 1]
    end
    level_info_interfaces_acc[n_levels] = first_entry

    n_boundaries = last(size(boundaries.u))
    # Determine level for each boundary
    for boundary_id in 1:n_boundaries
        # Get element id (boundaries have only one unique associated element)
        element_id = boundaries.neighbor_ids[boundary_id]
        h = hmin_per_element_[element_id]

        level = findfirst(x -> x < (hmin / h)^dt_scaling_order, dt_ratios)
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level === nothing
            level = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level = level - 1
        end

        # Add to accumulated container
        if level == 1
            push!(level_info_boundaries_acc[1], boundary_id)
        else
            for l in level:n_levels
                push!(level_info_boundaries_acc[l], boundary_id)
            end
        end
    end

    # These are the finest boundaries, which should be integrated with the IMEX method,
    # which is due to historical reasons of the implementation the *LAST* method 
    # in the time integration algorithm.
    # Thus, we need to move it to the end and everything else to the front.
    first_entry = level_info_boundaries_acc[1]
    for l in 1:(n_levels - 1)
        level_info_boundaries_acc[l] = level_info_boundaries_acc[l + 1]
    end
    level_info_boundaries_acc[n_levels] = first_entry

    # TODO: Mortars are not taken care of for IMEX!
    # p4est is always dimension 2 or 3
    n_mortars = last(size(mortars.u))
    for mortar_id in 1:n_mortars
        # Get element ids
        element_id_lower = mortars.neighbor_ids[1, mortar_id]
        h_lower = hmin_per_element_[element_id_lower]

        element_id_higher = mortars.neighbor_ids[2, mortar_id]
        h_higher = hmin_per_element_[element_id_higher]

        h = min(h_lower, h_higher)

        level = findfirst(x -> x < (hmin / h)^dt_scaling_order, dt_ratios)
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

    return nothing
end

function partition_variables!(level_info_elements, n_levels,
                              semi::AbstractSemidiscretization, alg)
    mesh, _, dg, cache = mesh_equations_solver_cache(semi)

    partition_variables!(level_info_elements, n_levels,
                         mesh, dg, cache,
                         alg)
end

# Assign number of stage evaluations to elements for stage-evaluations weighted MPI load balancing.
function partition_variables!(level_info_elements, n_levels,
                              mesh::ParallelP4estMesh, dg, cache,
                              alg)
    @unpack elements, interfaces, boundaries, mortars = cache
    @unpack mpi_interfaces, mpi_mortars = cache

    nnodes = length(dg.basis.nodes)
    n_elements = nelements(dg, cache)

    hmin_per_element_, hmin, hmax = hmin_per_element(mesh, cache.elements,
                                                     n_elements, nnodes)

    # Synchronize `hmin`, `hmax` to have consistent partitioning across ranks
    hmin = MPI.Allreduce!(Ref(hmin), Base.min, mpi_comm())[]
    hmax = MPI.Allreduce!(Ref(hmax), Base.max, mpi_comm())[]

    println("hmin: ", hmin, " hmax: ", hmax)
    println("hmax/hmin: ", hmax / hmin, "\n")

    for element_id in 1:n_elements
        h = hmin_per_element_[element_id]

        level = findfirst(x -> x < hmin / h, alg.dt_ratios) # TODO: parabolic: (hmin / h)^2
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level === nothing
            level = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level = level - 1
        end

        append!(level_info_elements[level], element_id)
    end

    return nothing
end

function partition_variables!(level_info_elements,
                              level_info_elements_acc,
                              level_info_interfaces_acc,
                              level_info_boundaries_acc,
                              level_info_mortars_acc,
                              # MPI additions
                              level_info_mpi_interfaces_acc,
                              level_info_mpi_mortars_acc,
                              n_levels, mesh::ParallelP4estMesh, dg, cache,
                              alg)
    @unpack elements, interfaces, boundaries, mortars = cache
    @unpack mpi_interfaces, mpi_mortars = cache

    nnodes = length(dg.basis.nodes)
    n_elements = nelements(dg, cache)

    # `hmin_per_element_` needs to be recomputed after balancing as 
    # the number of elements per rank may have changed
    hmin_per_element_, hmin, hmax = hmin_per_element(mesh, cache.elements,
                                                     n_elements, nnodes)

    # Synchronize `hmin`, `hmax` to have consistent partitioning across ranks
    hmin = MPI.Allreduce!(Ref(hmin), Base.min, mpi_comm())[]
    hmax = MPI.Allreduce!(Ref(hmax), Base.max, mpi_comm())[]

    #println("hmin: ", hmin, " hmax: ", hmax)
    #println("hmax/hmin: ", hmax / hmin, "\n")

    for element_id in 1:n_elements
        h = hmin_per_element_[element_id]

        level = findfirst(x -> x < hmin / h, alg.dt_ratios) # TODO: parabolic: (hmin / h)^2
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
        h1 = hmin_per_element_[element_id1]
        h2 = hmin_per_element_[element_id2]
        h = min(h1, h2)

        level = findfirst(x -> x < hmin / h, alg.dt_ratios) # TODO: parabolic: (hmin / h)^2
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
        h1 = hmin_per_element_[element_id1]
        h2 = hmin_per_element_[element_id2]
        h = min(h1, h2)

        level = findfirst(x -> x < hmin / h, alg.dt_ratios) # TODO: parabolic: (hmin / h)^2
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
        h = hmin_per_element_[element_id]

        level = findfirst(x -> x < hmin / h, alg.dt_ratios)
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
        h_lower = hmin_per_element_[element_id_lower]

        element_id_higher = mortars.neighbor_ids[2, mortar_id]
        h_higher = hmin_per_element_[element_id_higher]

        h = min(h_lower, h_higher)

        level = findfirst(x -> x < hmin / h, alg.dt_ratios) # TODO: parabolic: (hmin / h)^2
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
    for mortar_id in 1:n_mpi_mortars
        # Get element ids
        element_id_lower = mortars.neighbor_ids[1, mortar_id]
        h_lower = hmin_per_element_[element_id_lower]

        element_id_higher = mortars.neighbor_ids[2, mortar_id]
        h_higher = hmin_per_element_[element_id_higher]

        h = min(h_lower, h_higher)

        level = findfirst(x -> x < hmin / h, alg.dt_ratios) # TODO: parabolic: (hmin / h)^2
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

    return nothing
end

function partition_variables!(level_info_elements,
                              level_info_elements_acc,
                              level_info_interfaces_acc,
                              level_info_boundaries_acc,
                              level_info_mortars_acc,
                              n_levels, mesh::StructuredMesh, dg, cache,
                              alg)
    nnodes = length(dg.basis.nodes)
    n_elements = nelements(dg, cache)

    hmin_per_element_, hmin, hmax = hmin_per_element(mesh, cache.elements,
                                                     n_elements, nnodes)

    println("hmin: ", hmin, " hmax: ", hmax)
    println("hmax/hmin: ", hmax / hmin, "\n")

    for element_id in 1:n_elements
        h = hmin_per_element_[element_id]

        level = findfirst(x -> x < hmin / h, alg.dt_ratios) # Parabolic terms are not supported on `StructuredMesh`
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level === nothing
            level = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level = level - 1
        end

        # CARE: This is for testcase with special assignment
        #level = rand(1:n_levels)
        #level = mod(element_id - 1, n_levels) + 1 # Assign elements in round-robin fashion

        append!(level_info_elements[level], element_id)

        for l in level:n_levels
            push!(level_info_elements_acc[l], element_id)
        end
    end

    # No interfaces, boundaries, mortars for structured meshes

    return nothing
end

function partition_variables!(level_info_elements,
                              level_info_elements_acc,
                              level_info_interfaces_acc,
                              level_info_boundaries_acc,
                              level_info_mortars_acc,
                              n_levels, mesh::StructuredMesh, dg, cache,
                              alg::AbstractPairedExplicitRKIMEXMulti)
    nnodes = length(dg.basis.nodes)
    n_elements = nelements(dg, cache)

    hmin_per_element_, hmin, hmax = hmin_per_element(mesh, cache.elements,
                                                     n_elements, nnodes)

    println("hmin: ", hmin, " hmax: ", hmax)
    println("hmax/hmin: ", hmax / hmin, "\n")

    for element_id in 1:n_elements
        h = hmin_per_element_[element_id]

        level = findfirst(x -> x < hmin / h, alg.dt_ratios) # Parabolic terms are not supported on `StructuredMesh`
        # Catch case that cell is "too coarse" for method with fewest stage evals
        if level === nothing
            level = n_levels
        else # Avoid reduction in timestep: Use next higher level
            level = level - 1
        end

        # CARE: This is for testcase with special assignment
        #level = rand(1:n_levels)
        #level = mod(element_id - 1, n_levels) + 1 # Assign elements in round-robin fashion

        append!(level_info_elements[level], element_id)

        # Add to accumulated container
        # Exclude pushes to first level with is integrated implicitly
        if level == 1
            push!(level_info_elements_acc[1], element_id)
        else
            for l in level:n_levels
                push!(level_info_elements_acc[l], element_id)
            end
        end
    end

    # These are the finest cells, which should be integrated with the IMEX method,
    # which is due to historical reasons of the implementation the *LAST* method 
    # in the time integration algorithm.
    # Thus, we need to move it to the end and everything else to the front.
    first_entry = level_info_elements[1]
    for l in 1:(n_levels - 1)
        level_info_elements[l] = level_info_elements[l + 1]
    end
    level_info_elements[n_levels] = first_entry

    first_entry = level_info_elements_acc[1]
    for l in 1:(n_levels - 1)
        level_info_elements_acc[l] = level_info_elements_acc[l + 1]
    end
    level_info_elements_acc[n_levels] = first_entry

    # No interfaces, boundaries, mortars for structured meshes

    return nothing
end

function hmin_per_element(mesh::StructuredMesh{1}, elements,
                          n_elements, nnodes)
    RealT = real(mesh)
    hmin = floatmax(RealT)
    hmax = zero(RealT)

    hmin_per_element_ = zeros(n_elements)

    for element_id in 1:n_elements
        P0 = elements.node_coordinates[1, 1, element_id]
        P1 = elements.node_coordinates[1, nnodes, element_id]
        h = abs(P1 - P0) # Assumes P1 > P0

        hmin_per_element_[element_id] = h
        if h > hmax
            hmax = h
        end
        if h < hmin
            hmin = h
        end
    end

    return hmin_per_element_, hmin, hmax
end

function hmin_per_element(mesh::Union{P4estMesh{2}, StructuredMesh{2}}, elements,
                          n_elements, nnodes)
    hmin_per_element_ = zeros(real(mesh), n_elements)

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
        hmin_per_element_[element_id] = h
    end

    # Set global `hmin` and `hmax`, i.e., for the entire mesh
    hmin = minimum(hmin_per_element_)
    hmax = maximum(hmin_per_element_)

    return hmin_per_element_, hmin, hmax
end

function hmin_per_element(mesh::P4estMesh{3}, elements,
                          n_elements, nnodes)
    hmin_per_element_ = zeros(real(mesh), n_elements)

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
        hmin_per_element_[element_id] = h
    end

    # Set global `hmin` and `hmax`, i.e., for the entire mesh
    hmin = minimum(hmin_per_element_)
    hmax = maximum(hmin_per_element_)

    return hmin_per_element_, hmin, hmax
end

function dtmax_per_element(u, mesh::P4estMesh{3}, equations, dg, cache)
    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)

    dtmax_per_element_ = zeros(n_elements)

    @unpack contravariant_vectors, inverse_jacobian = cache.elements
    @unpack weights = dg.basis

    for element in eachelement(dg, cache)
        max_a1 = max_a2 = max_a3 = zero(eltype(u))
        #max_a_total = zero(eltype(u))
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)

            # Speeds
            a1, a2, a3 = max_abs_speeds(u_node, equations)

            # Mesh data
            Ja11, Ja12, Ja13 = get_contravariant_vector(1, contravariant_vectors,
                                                        i, j, k, element)
            Ja21, Ja22, Ja23 = get_contravariant_vector(2, contravariant_vectors,
                                                        i, j, k, element)
            Ja31, Ja32, Ja33 = get_contravariant_vector(3, contravariant_vectors,
                                                        i, j, k, element)

            inv_jacobian = abs(inverse_jacobian[i, j, k, element])

            # Transform
            a1_transformed = abs(Ja11 * a1 + Ja12 * a2 + Ja13 * a3) * inv_jacobian
            a2_transformed = abs(Ja21 * a1 + Ja22 * a2 + Ja23 * a3) * inv_jacobian
            a3_transformed = abs(Ja31 * a1 + Ja32 * a2 + Ja33 * a3) * inv_jacobian

            max_a1 = max(max_a1, a1_transformed)
            max_a2 = max(max_a2, a2_transformed)
            max_a3 = max(max_a3, a3_transformed)

            #a_total = a1_transformed + a2_transformed + a3_transformed
            #max_a_total = max(max_a_total, a_total)
        end
        dtmax_per_element_[element] = 1 / (max_a1 + max_a2 + max_a3)
        #dtmax_per_element_[element] = 1 / max_a_total

        # Try integration-alike
        #=
        u_mean = compute_u_mean(u, element, mesh, equations, dg, cache)
        # Speeds
        a1, a2, a3 = max_abs_speeds(u_mean, equations)
        a_total = a1 + a2 + a3
        dtmax_per_element_[element] = 1 / a_total
        =#

        #=
        volume = zero(real(mesh))
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            # Speeds
            a1, a2, a3 = max_abs_speeds(u_node, equations)
            a_total = a1 + a2 + a3

            volume_jacobian = abs(inv(inverse_jacobian[i, j, k, element]))
            #dtmax_per_element_[element] += volume_jacobian * weights[i] * weights[j] * weights[k] / a_total
            volume += volume_jacobian * weights[i] * weights[j] * weights[k]
        end
        dtmax_per_element_[element] *= volume
        =#
    end

    dtmax_per_element_ .*= 2 / nnodes(dg) # Becomes only relevant for both advection/diffusion

    dtmin = minimum(dtmax_per_element_)
    dtmax = maximum(dtmax_per_element_)

    return dtmax_per_element_, dtmin, dtmax
end

@inline function partition_u!(level_info_u,
                              level_info_elements, n_levels,
                              u_ode, semi::AbstractSemidiscretization)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    partition_u!(level_info_u,
                 level_info_elements, n_levels,
                 u_ode, mesh, equations, dg, cache)
end

# Partitioning function for approach: Each level stores its indices
@inline function partition_u!(level_info_u,
                              level_info_elements, n_levels,
                              u_ode, mesh, equations, dg, cache)
    u = wrap_array(u_ode, mesh, equations, dg, cache)

    for level in 1:n_levels
        @views indices = collect(Iterators.flatten(LinearIndices(u)[..,
                                                                    level_info_elements[level]]))
        level_info_u[level] = indices
        sort!(level_info_u[level])
    end

    return nothing
end

# Optimized version with accumulated indices
@inline function partition_u!(level_info_u, level_info_u_acc,
                              level_info_elements, n_levels,
                              u_ode, semi::SemidiscretizationHyperbolicParabolic)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    partition_u!(level_info_u, level_info_u_acc,
                 level_info_elements, n_levels,
                 u_ode, mesh, equations, dg, cache)
end

@inline function partition_u!(level_info_u, level_info_u_acc,
                              level_info_elements, n_levels,
                              u_ode, mesh, equations, dg, cache)
    u = wrap_array(u_ode, mesh, equations, dg, cache)

    for level in 1:n_levels
        @views indices = collect(Iterators.flatten(LinearIndices(u)[..,
                                                                    level_info_elements[level]]))
        level_info_u[level] = indices
        sort!(level_info_u[level])

        # Add to accumulated container
        for l in 1:level
            append!(level_info_u_acc[level], level_info_u[l])
        end
        sort!(level_info_u_acc[level])
    end

    return nothing
end

#=
# Partitioning function for approach: Each index stores its level
@inline function partition_u!(level_info_u, level_info_elements,
                              u_to_level,
                              n_levels, u_ode,
                              mesh, equations, dg, cache)
    u = wrap_array(u_ode, mesh, equations, dg, cache)

    for level in 1:n_levels
        @views indices = collect(Iterators.flatten(LinearIndices(u)[..,
                                                                    level_info_elements[level]]))
        level_info_u[level] = indices
        sort!(level_info_u[level])

        @views u_to_level[indices] .= level
    end

    return nothing
end
=#

# Repartitioning of the DoF array of the gravity solver
@inline function partition_u_gravity!(integrator::AbstractPairedExplicitRKMultiIntegrator)
    @unpack level_info_elements, n_levels = integrator
    @unpack semi_gravity, cache = integrator.p

    u_gravity = wrap_array(cache.u_ode, semi_gravity)

    resize!(cache.level_info_u_gravity, n_levels)
    for level in 1:n_levels
        if isassigned(cache.level_info_u_gravity, level)
            empty!(cache.level_info_u_gravity[level])
        else
            cache.level_info_u_gravity[level] = []
        end
    end

    for level in 1:n_levels
        @views indices = collect(Iterators.flatten(LinearIndices(u_gravity)[..,
                                                                            level_info_elements[level]]))
        append!(cache.level_info_u_gravity[level], indices)
        sort!(cache.level_info_u_gravity[level])
    end

    return nothing
end

# Functions for PERK-MPI load balancing
function save_rhs_evals_iter_volume(info, user_data)
    info_pw = PointerWrapper(info)

    # Load tree from global trees array, one-based indexing
    tree_pw = load_pointerwrapper_tree(info_pw.p4est, info_pw.treeid[] + 1)
    # Quadrant numbering offset of this quadrant
    offset = tree_pw.quadrants_offset[]
    # Global quad ID
    quad_id = offset + info_pw.quadid[]

    # Access user_data = `rhs_per_element`
    user_data_pw = PointerWrapper(Int, user_data)
    # Load `rhs_evals = rhs_per_element[quad_id + 1]`
    rhs_evals = user_data_pw[quad_id + 1]

    # Access quadrant's user data (`[rhs_evals]`)
    quad_data_pw = PointerWrapper(Int, info_pw.quad.p.user_data[])
    # Save number of rhs evaluations to quadrant's user data.
    quad_data_pw[1] = rhs_evals

    return nothing
end

# 2D
function cfunction(::typeof(save_rhs_evals_iter_volume), ::Val{2})
    @cfunction(save_rhs_evals_iter_volume, Cvoid,
               (Ptr{p4est_iter_volume_info_t}, Ptr{Cvoid}))
end
# 3D
function cfunction(::typeof(save_rhs_evals_iter_volume), ::Val{3})
    @cfunction(save_rhs_evals_iter_volume, Cvoid,
               (Ptr{p8est_iter_volume_info_t}, Ptr{Cvoid}))
end

function weight_fn_perk(p4est, which_tree, quadrant)
    # Number of RHS evaluations has been copied to the quadrant's user data storage before.
    # Unpack quadrant's user data ([rhs_evals]).
    # Use `unsafe_load` here since `quadrant.p.user_data isa Ptr{Ptr{Nothing}}`
    # and we only need the first entry
    pw = PointerWrapper(Int, unsafe_load(quadrant.p.user_data))
    weight = pw[1] # rhs_evals

    return Cint(weight)
end

# 2D
function cfunction(::typeof(weight_fn_perk), ::Val{2})
    @cfunction(weight_fn_perk, Cint,
               (Ptr{p4est_t}, Ptr{p4est_topidx_t}, Ptr{p4est_quadrant_t}))
end

# 3D
function cfunction(::typeof(weight_fn_perk), ::Val{3})
    @cfunction(weight_fn_perk, Cint,
               (Ptr{p8est_t}, Ptr{p4est_topidx_t}, Ptr{p8est_quadrant_t}))
end

function get_rhs_per_element(dg, cache,
                             level_info_elements, stages)
    rhs_per_element = zeros(Int, nelements(dg, cache))

    for level in eachindex(level_info_elements)
        rhs_evals = stages[level]
        for element in level_info_elements[level]
            rhs_per_element[element] = rhs_evals
        end
    end

    return rhs_per_element
end

function balance_p4est_perk!(mesh::ParallelP4estMesh, dg, cache,
                             level_info_elements, stages)
    rhs_per_element = get_rhs_per_element(dg, cache, level_info_elements, stages)

    # The pointer to rhs_per_element will be interpreted as Ptr{Int} below
    #@assert rhs_per_element isa Vector{Int}
    @boundscheck begin
        @assert axes(rhs_per_element) == (Base.OneTo(ncells(mesh)),)
    end

    iter_volume_c = cfunction(save_rhs_evals_iter_volume, Val(ndims(mesh)))
    iterate_p4est(mesh.p4est, rhs_per_element; iter_volume_c = iter_volume_c)

    weight_fn_c = cfunction(weight_fn_perk, Val(ndims(mesh)))
    partition!(mesh; weight_fn = weight_fn_c)
end
end # @muladd
