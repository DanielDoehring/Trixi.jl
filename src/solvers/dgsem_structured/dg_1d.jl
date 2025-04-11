# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function rhs!(du, u, t,
              mesh::StructuredMesh{1}, equations,
              boundary_conditions, source_terms::Source,
              dg::DG, cache,
              element_indices = eachelement(dg, cache),
              interface_indices = nothing,
              boundary_indices = nothing,
              mortar_indices = nothing) where {Source}
    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache, element_indices)

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, u, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.volume_integral, dg, cache, element_indices)
    end

    # Calculate interface and boundary fluxes
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache, u, mesh, equations, dg.surface_integral, dg,
                             element_indices)
    end

    # TODO: Boundary Orientations?
    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux!(cache, u, t, boundary_conditions, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(du, u, mesh, equations,
                               dg.surface_integral, dg, cache, element_indices)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" apply_jacobian!(du, mesh, equations, dg, cache,
                                                     element_indices)

    # Calculate source terms
    @trixi_timeit timer() "source terms" begin
        calc_sources!(du, u, t, source_terms, equations, dg, cache, element_indices)
    end

    return nothing
end

function calc_volume_integral!(du, u,
                               mesh::StructuredMesh,
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralShockCapturingHG,
                               dg::DGSEM, cache,
                               element_indices = eachelement(dg, cache))
    @unpack volume_flux_dg, volume_flux_fv, indicator = volume_integral

    # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
    alpha = @trixi_timeit timer() "blending factors" indicator(u, mesh, equations, dg,
                                                               cache, element_indices)

    # For `Float64`, this gives 1.8189894035458565e-12
    # For `Float32`, this gives 1.1920929f-5
    RealT = eltype(alpha)
    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))
    @threaded for element in element_indices
        alpha_element = alpha[element]
        # Clip blending factor for values close to zero (-> pure DG)
        dg_only = isapprox(alpha_element, 0, atol = atol)

        if dg_only
            flux_differencing_kernel!(du, u, element, mesh,
                                      nonconservative_terms, equations,
                                      volume_flux_dg, dg, cache)
        else
            # Calculate DG volume integral contribution
            flux_differencing_kernel!(du, u, element, mesh,
                                      nonconservative_terms, equations,
                                      volume_flux_dg, dg, cache, 1 - alpha_element)

            # Calculate FV volume integral contribution
            fv_kernel!(du, u, mesh, nonconservative_terms, equations, volume_flux_fv,
                       dg, cache, element, alpha_element)
        end
    end

    return nothing
end

function calc_interface_flux!(cache, u, mesh::StructuredMesh{1},
                              equations, surface_integral, dg::DG,
                              element_indices = eachelement(dg, cache))
    @unpack surface_flux = surface_integral

    @threaded for element in element_indices
        left_element = cache.elements.left_neighbors[1, element]

        if left_element > 0 # left_element = 0 at boundaries
            u_ll = get_node_vars(u, equations, dg, nnodes(dg), left_element)
            u_rr = get_node_vars(u, equations, dg, 1, element)

            f1 = surface_flux(u_ll, u_rr, 1, equations)

            for v in eachvariable(equations)
                cache.elements.surface_flux_values[v, 2, left_element] = f1[v]
                cache.elements.surface_flux_values[v, 1, element] = f1[v]
            end
        end
    end

    return nothing
end

# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, u, t, boundary_condition::BoundaryConditionPeriodic,
                             mesh::StructuredMesh{1}, equations, surface_integral,
                             dg::DG)
    @assert isperiodic(mesh)
end

function calc_boundary_flux!(cache, u, t, boundary_conditions::NamedTuple,
                             mesh::StructuredMesh{1}, equations, surface_integral,
                             dg::DG)
    @unpack surface_flux = surface_integral
    @unpack surface_flux_values, node_coordinates = cache.elements

    orientation = 1

    # Negative x-direction
    direction = 1

    u_rr = get_node_vars(u, equations, dg, 1, 1)
    x = get_node_coords(node_coordinates, equations, dg, 1, 1)

    flux = boundary_conditions[direction](u_rr, orientation, direction, x, t,
                                          surface_flux, equations)

    for v in eachvariable(equations)
        surface_flux_values[v, direction, 1] = flux[v]
    end

    # Positive x-direction
    direction = 2

    u_rr = get_node_vars(u, equations, dg, nnodes(dg), nelements(dg, cache))
    x = get_node_coords(node_coordinates, equations, dg, nnodes(dg),
                        nelements(dg, cache))

    flux = boundary_conditions[direction](u_rr, orientation, direction, x, t,
                                          surface_flux, equations)

    # Copy flux to left and right element storage
    for v in eachvariable(equations)
        surface_flux_values[v, direction, nelements(dg, cache)] = flux[v]
    end
end
end # @muladd
