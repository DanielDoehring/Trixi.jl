# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function save_solution_file(u, time, dt, timestep,
                            mesh::Union{SerialTreeMesh, StructuredMesh,
                                        StructuredMeshView,
                                        UnstructuredMesh2D,
                                        SerialP4estMesh, P4estMeshView,
                                        SerialT8codeMesh},
                            equations, dg::DG, cache,
                            solution_callback,
                            element_variables = Dict{Symbol, Any}(),
                            node_variables = Dict{Symbol, Any}();
                            system = "")
    @unpack output_directory, solution_variables = solution_callback

    # Filename based on current time step
    if isempty(system)
        filename = joinpath(output_directory, @sprintf("solution_%09d.h5", timestep))
    else
        filename = joinpath(output_directory,
                            @sprintf("solution_%s_%09d.h5", system, timestep))
    end

    # Convert to different set of variables if requested
    if solution_variables === cons2cons
        data = u
        n_vars = nvariables(equations)
    else
        # Reinterpret the solution array as an array of conservative variables,
        # compute the solution variables via broadcasting, and reinterpret the
        # result as a plain array of floating point numbers
        data = Array(reinterpret(eltype(u),
                                 solution_variables.(reinterpret(SVector{nvariables(equations),
                                                                         eltype(u)}, u),
                                                     Ref(equations))))

        # Find out variable count by looking at output from `solution_variables` function
        n_vars = size(data, 1)
    end

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["equations"] = get_name(equations)
        attributes(file)["polydeg"] = polydeg(dg)
        attributes(file)["n_vars"] = n_vars
        attributes(file)["n_elements"] = nelements(dg, cache)
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
        attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
        attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
        attributes(file)["timestep"] = timestep

        # Store each variable of the solution data
        for v in 1:n_vars
            # Convert to 1D array
            file["variables_$v"] = vec(data[v, .., :])

            # Add variable name as attribute
            var = file["variables_$v"]
            attributes(var)["name"] = varnames(solution_variables, equations)[v]
        end

        # Store element variables
        for (v, (key, element_variable)) in enumerate(element_variables)
            # Add to file
            file["element_variables_$v"] = element_variable

            # Add variable name as attribute
            var = file["element_variables_$v"]
            attributes(var)["name"] = string(key)
        end

        # Store node variables
        for (v, (key, node_variable)) in enumerate(node_variables)
            # Add to file
            file["node_variables_$v"] = node_variable

            # Add variable name as attribute
            var = file["node_variables_$v"]
            attributes(var)["name"] = string(key)
        end
    end

    return filename
end

function save_solution_file(u, time, dt, timestep,
                            mesh::SerialDGMultiMesh,
                            equations, dg::DG, cache,
                            solution_callback,
                            element_variables = Dict{Symbol, Any}(),
                            node_variables = Dict{Symbol, Any}();
                            system = "")
    @unpack output_directory, solution_variables = solution_callback

    # Filename based on current time step.
    if isempty(system)
        filename = joinpath(output_directory, @sprintf("solution_%09d.h5", timestep))
    else
        filename = joinpath(output_directory,
                            @sprintf("solution_%s_%09d.h5", system, timestep))
    end

    # Convert to different set of variables if requested.
    if solution_variables === cons2cons
        data = u
        n_vars = nvariables(equations)
    else
        data = map(u_node -> solution_variables(u_node, equations), u)
        # Find out variable count by looking at output from `solution_variables` function.
        n_vars = length(data[1])
    end

    # Open file (clobber existing content).
    h5open(filename, "w") do file
        # Add context information as attributes.
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["equations"] = get_name(equations)

        if dg.basis.approximation_type isa TensorProductWedge
            attributes(file)["polydeg_tri"] = dg.basis.N[2]
            attributes(file)["polydeg_line"] = dg.basis.N[1]
        else
            attributes(file)["polydeg"] = dg.basis.N
        end

        attributes(file)["element_type"] = dg.basis.element_type |> typeof |> nameof |>
                                           string
        attributes(file)["n_vars"] = n_vars
        attributes(file)["n_elements"] = nelements(dg, cache)
        attributes(file)["dof_per_elem"] = length(dg.basis.r)
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
        attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar.
        attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar.
        attributes(file)["timestep"] = timestep

        # Store each variable of the solution data.
        for v in 1:n_vars
            temp = zeros(size(u.u))
            n_nodes, n_elems = size(u.u)
            for i_elem in 1:n_elems
                for i_node in 1:n_nodes
                    temp[i_node, i_elem] = data[i_node, i_elem][v]
                end
            end

            file["variables_$v"] = temp

            # Add variable name as attribute.
            var = file["variables_$v"]
            attributes(var)["name"] = varnames(solution_variables, equations)[v]
        end

        # Store element variables.
        for (v, (key, element_variable)) in enumerate(element_variables)
            # Add to file.
            file["element_variables_$v"] = element_variable

            # Add variable name as attribute.
            var = file["element_variables_$v"]
            attributes(var)["name"] = string(key)
        end

        # Store node variables.
        for (v, (key, node_variable)) in enumerate(node_variables)
            # Add to file
            file["node_variables_$v"] = node_variable

            # Add variable name as attribute.
            var = file["node_variables_$v"]
            attributes(var)["name"] = string(key)
        end
    end

    return filename
end

function save_solution_file(u, time, dt, timestep,
                            mesh::Union{ParallelTreeMesh, ParallelP4estMesh,
                                        ParallelT8codeMesh}, equations,
                            dg::DG, cache,
                            solution_callback,
                            element_variables = Dict{Symbol, Any}(),
                            node_variables = Dict{Symbol, Any}();
                            system = "")
    @unpack output_directory, solution_variables = solution_callback

    # Filename based on current time step
    if isempty(system)
        filename = joinpath(output_directory, @sprintf("solution_%09d.h5", timestep))
    else
        filename = joinpath(output_directory,
                            @sprintf("solution_%s_%09d.h5", system, timestep))
    end

    # Convert to different set of variables if requested
    if solution_variables === cons2cons
        data = u
        n_vars = nvariables(equations)
    else
        # Reinterpret the solution array as an array of conservative variables,
        # compute the solution variables via broadcasting, and reinterpret the
        # result as a plain array of floating point numbers
        data = Array(reinterpret(eltype(u),
                                 solution_variables.(reinterpret(SVector{nvariables(equations),
                                                                         eltype(u)}, u),
                                                     Ref(equations))))

        # Find out variable count by looking at output from `solution_variables` function
        n_vars = size(data, 1)
    end

    if HDF5.has_parallel()
        save_solution_file_parallel(data, time, dt, timestep, n_vars, mesh, equations,
                                    dg, cache, solution_variables, filename,
                                    element_variables, node_variables)
    else
        save_solution_file_on_root(data, time, dt, timestep, n_vars, mesh, equations,
                                   dg, cache, solution_variables, filename,
                                   element_variables, node_variables)
    end
end

function save_solution_file_parallel(data, time, dt, timestep, n_vars,
                                     mesh::Union{ParallelTreeMesh, ParallelP4estMesh,
                                                 ParallelT8codeMesh},
                                     equations, dg::DG, cache,
                                     solution_variables, filename,
                                     element_variables = Dict{Symbol, Any}(),
                                     node_variables = Dict{Symbol, Any}())

    # Calculate element and node counts by MPI rank
    element_size = nnodes(dg)^ndims(mesh)
    element_counts = cache.mpi_cache.n_elements_by_rank
    node_counts = element_counts * element_size
    # Cumulative sum of elements per rank starting with an additional 0
    cum_element_counts = append!(zeros(eltype(element_counts), 1),
                                 cumsum(element_counts))
    # Cumulative sum of nodes per rank starting with an additional 0
    cum_node_counts = append!(zeros(eltype(node_counts), 1), cumsum(node_counts))

    # Open file using parallel HDF5 (clobber existing content)
    h5open(filename, "w", mpi_comm()) do file
        # Add context information as attributes
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["equations"] = get_name(equations)
        attributes(file)["polydeg"] = polydeg(dg)
        attributes(file)["n_vars"] = n_vars
        attributes(file)["n_elements"] = nelementsglobal(mesh, dg, cache)
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
        attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
        attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
        attributes(file)["timestep"] = timestep

        # Store each variable of the solution data
        for v in 1:n_vars
            # Need to create dataset explicitly in parallel case
            var = create_dataset(file, "/variables_$v", datatype(eltype(data)),
                                 dataspace((ndofsglobal(mesh, dg, cache),)))
            # Write data of each process in slices (ranks start with 0)
            slice = (cum_node_counts[mpi_rank() + 1] + 1):cum_node_counts[mpi_rank() + 2]
            # Convert to 1D array
            var[slice] = vec(data[v, .., :])
            # Add variable name as attribute
            attributes(var)["name"] = varnames(solution_variables, equations)[v]
        end

        # Store element variables
        for (v, (key, element_variable)) in enumerate(element_variables)
            # Need to create dataset explicitly in parallel case
            var = create_dataset(file, "/element_variables_$v",
                                 datatype(eltype(element_variable)),
                                 dataspace((nelementsglobal(mesh, dg, cache),)))

            # Write data of each process in slices (ranks start with 0)
            slice = (cum_element_counts[mpi_rank() + 1] + 1):cum_element_counts[mpi_rank() + 2]
            # Add to file
            var[slice] = element_variable
            # Add variable name as attribute
            attributes(var)["name"] = string(key)
        end

        # Store node variables
        for (v, (key, node_variable)) in enumerate(node_variables)
            # Need to create dataset explicitly in parallel case
            var = create_dataset(file, "/node_variables_$v",
                                 datatype(eltype(node_variable)),
                                 dataspace((nelementsglobal(mesh, dg, cache) *
                                            element_size,)))

            # Write data of each process in slices (ranks start with 0)
            slice = (cum_node_counts[mpi_rank() + 1] + 1):cum_node_counts[mpi_rank() + 2]
            # Add to file
            var[slice] = node_variable
            # Add variable name as attribute
            attributes(var)["name"] = string(key)
        end
    end

    return filename
end

function save_solution_file_on_root(data, time, dt, timestep, n_vars,
                                    mesh::Union{ParallelTreeMesh, ParallelP4estMesh,
                                                ParallelT8codeMesh},
                                    equations, dg::DG, cache,
                                    solution_variables, filename,
                                    element_variables = Dict{Symbol, Any}(),
                                    node_variables = Dict{Symbol, Any}())

    # Calculate element and node counts by MPI rank
    element_size = nnodes(dg)^ndims(mesh)
    element_counts = convert(Vector{Cint}, collect(cache.mpi_cache.n_elements_by_rank))
    node_counts = element_counts * Cint(element_size)

    # non-root ranks only send data
    if !mpi_isroot()
        # Send nodal data to root
        for v in 1:n_vars
            MPI.Gatherv!(vec(data[v, .., :]), nothing, mpi_root(), mpi_comm())
        end

        # Send element data to root
        for (key, element_variable) in element_variables
            MPI.Gatherv!(element_variable, nothing, mpi_root(), mpi_comm())
        end

        # Send additional/extra node variables to root
        for (key, node_variable) in node_variables
            MPI.Gatherv!(node_variable, nothing, mpi_root(), mpi_comm())
        end

        return filename
    end

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["equations"] = get_name(equations)
        attributes(file)["polydeg"] = polydeg(dg)
        attributes(file)["n_vars"] = n_vars
        attributes(file)["n_elements"] = nelementsglobal(mesh, dg, cache)
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
        attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
        attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
        attributes(file)["timestep"] = timestep

        # Store each variable of the solution data
        for v in 1:n_vars
            # Convert to 1D array
            recv = Vector{eltype(data)}(undef, sum(node_counts))
            MPI.Gatherv!(vec(data[v, .., :]), MPI.VBuffer(recv, node_counts),
                         mpi_root(), mpi_comm())
            file["variables_$v"] = recv

            # Add variable name as attribute
            var = file["variables_$v"]
            attributes(var)["name"] = varnames(solution_variables, equations)[v]
        end

        # Store element variables
        for (v, (key, element_variable)) in enumerate(element_variables)
            # Add to file
            recv = Vector{eltype(data)}(undef, sum(element_counts))
            MPI.Gatherv!(element_variable, MPI.VBuffer(recv, element_counts),
                         mpi_root(), mpi_comm())
            file["element_variables_$v"] = recv

            # Add variable name as attribute
            var = file["element_variables_$v"]
            attributes(var)["name"] = string(key)
        end

        # Store node variables
        for (v, (key, node_variable)) in enumerate(node_variables)
            # Add to file
            recv = Vector{eltype(data)}(undef, sum(node_counts))
            MPI.Gatherv!(node_variable, MPI.VBuffer(recv, node_counts),
                         mpi_root(), mpi_comm())
            file["node_variables_$v"] = recv

            # Add variable name as attribute
            var = file["node_variables_$v"]
            attributes(var)["name"] = string(key)
        end
    end

    return filename
end

function average_interface_values!(data, cache, u,
                                   mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                   equations, dg::DG)

    @unpack interfaces = cache
    index_range = eachnode(dg)

    data_ = copy(data)

    @threaded for interface in eachinterface(dg, cache)
        # Copy solution data from the primary element using "delayed indexing" with
        # a start value and two step sizes to get the correct face and orientation.
        # Note that in the current implementation, the interface will be
        # "aligned at the primary element", i.e., the indices of the primary side
        # will always run forwards.
        primary_element = interfaces.neighbor_ids[1, interface]
        primary_indices = interfaces.node_indices[1, interface]

        i_primary_start, i_primary_step_i, i_primary_step_j = index_to_start_step_3d(primary_indices[1],
                                                                                     index_range)
        j_primary_start, j_primary_step_i, j_primary_step_j = index_to_start_step_3d(primary_indices[2],
                                                                                     index_range)
        k_primary_start, k_primary_step_i, k_primary_step_j = index_to_start_step_3d(primary_indices[3],
                                                                                     index_range)

        i_primary = i_primary_start
        j_primary = j_primary_start
        k_primary = k_primary_start
        for j in eachnode(dg)
            for i in eachnode(dg)
                for v in eachvariable(equations)
                    #=
                    interfaces.u[1, v, i, j, interface] = data_[v, i_primary, j_primary,
                                                            k_primary, primary_element]
                    =#
                end
                i_primary += i_primary_step_i
                j_primary += j_primary_step_i
                k_primary += k_primary_step_i
            end
            i_primary += i_primary_step_j
            j_primary += j_primary_step_j
            k_primary += k_primary_step_j
        end

        # Copy solution data from the secondary element using "delayed indexing" with
        # a start value and two step sizes to get the correct face and orientation.
        secondary_element = interfaces.neighbor_ids[2, interface]
        secondary_indices = interfaces.node_indices[2, interface]

        i_secondary_start, i_secondary_step_i, i_secondary_step_j = index_to_start_step_3d(secondary_indices[1],
                                                                                           index_range)
        j_secondary_start, j_secondary_step_i, j_secondary_step_j = index_to_start_step_3d(secondary_indices[2],
                                                                                           index_range)
        k_secondary_start, k_secondary_step_i, k_secondary_step_j = index_to_start_step_3d(secondary_indices[3],
                                                                                           index_range)

        i_secondary = i_secondary_start
        j_secondary = j_secondary_start
        k_secondary = k_secondary_start
        for j in eachnode(dg)
            for i in eachnode(dg)
                for v in eachvariable(equations)
                    interfaces.u[2, v, i, j, interface] = u[v, i_secondary, j_secondary,
                                                            k_secondary,
                                                            secondary_element]
                end
                i_secondary += i_secondary_step_i
                j_secondary += j_secondary_step_i
                k_secondary += k_secondary_step_i
            end
            i_secondary += i_secondary_step_j
            j_secondary += j_secondary_step_j
            k_secondary += k_secondary_step_j
        end
    end

    return nothing
end

end # @muladd
