# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Compute and save to disk a space-dependent `surface_variable`.
# For the purely hyperbolic, i.e., non-parabolic case, this is for instance 
# the pressure coefficient `SurfacePressureCoefficient`.
# The boundary/boundaries along which this quantity is to be integrated is determined by
# `boundary_symbols`, which is retrieved from `surface_variable`.
function analyze(surface_variable::AnalysisSurfacePointwise, du, u, t,
                 mesh::P4estMesh{3},
                 equations, dg::DGSEM, cache, semi, iter)
    @unpack boundaries = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis

    @unpack variable, boundary_symbols = surface_variable
    @unpack boundary_symbol_indices = semi.boundary_conditions
    indices = get_boundary_indices(boundary_symbols, boundary_symbol_indices)

    dim = 3
    n_nodes = nnodes(dg)
    n_elements = length(indices)

    coordinates = Matrix{real(dg)}(undef, n_elements * n_nodes, dim) # physical coordinates of indices
    values = Vector{real(dg)}(undef, n_elements * n_nodes) # variable values at indices

    index_range = eachnode(dg)

    global_node_counter = 1 # Keeps track of solution point number on the surface
    for boundary in indices
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]

        i_node_start, i_node_step = index_to_start_step_3d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_3d(node_indices[2], index_range)
        k_node_start, k_node_step = index_to_start_step_3d(node_indices[3], index_range)

        # In 3D, boundaries are surfaces => `node_index1`, `node_index2`
        for node_index1 in index_range
            # Reset node indices
            i_node = i_node_start
            j_node = j_node_start
            k_node = k_node_start
            for node_index2 in index_range
                u_node = Trixi.get_node_vars(cache.boundaries.u, equations, dg,
                                             node_index1, node_index2, boundary)

                x = get_node_coords(node_coordinates, equations, dg,
                                    i_node, j_node, k_node, element)
                value = variable(u_node, equations)

                coordinates[global_node_counter, 1] = x[1]
                coordinates[global_node_counter, 2] = x[2]
                coordinates[global_node_counter, 3] = x[3]
                values[global_node_counter] = value

                i_node += i_node_step
                j_node += j_node_step
                k_node += k_node_step
                global_node_counter += 1
            end
        end
    end

    # Save to disk
    save_pointwise_file(surface_variable.output_directory, varname(variable),
                        coordinates, values, t, iter)
end
end # muladd
