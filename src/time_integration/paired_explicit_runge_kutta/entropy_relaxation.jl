# NOTE: This could actually live in a more general location,
# as it is not PERK-specific.

@inline function int_w_dot_stage(stage, u_i,
                                 mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                 equations, dg::DG, cache)
    @trixi_timeit timer() "Integrate w ⋅ k" begin
        # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
        integrate_via_indices(u_i, mesh, equations, dg, cache,
                              stage) do u_i, i, element, equations, dg, stage
            w_node = cons2entropy(get_node_vars(u_i, equations, dg, i, element),
                                  equations)
            stage_node = get_node_vars(stage, equations, dg, i, element)
            dot(w_node, stage_node)
        end
    end
end

@inline function int_w_dot_stage(stage, u_i,
                                 mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                             UnstructuredMesh2D, P4estMesh{2},
                                             T8codeMesh{2}},
                                 equations, dg::DG, cache)
    @trixi_timeit timer() "Integrate w ⋅ k" begin
        # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
        integrate_via_indices(u_i, mesh, equations, dg, cache,
                              stage) do u_i, i, j, element, equations, dg, stage
            w_node = cons2entropy(get_node_vars(u_i, equations, dg, i, j, element),
                                  equations)
            stage_node = get_node_vars(stage, equations, dg, i, j, element)
            dot(w_node, stage_node)
        end
    end
end

@inline function int_w_dot_stage(stage, u_i,
                                 mesh::Union{TreeMesh{3}, StructuredMesh{3}, P4estMesh{3},
                                             T8codeMesh{3}},
                                 equations, dg::DG, cache)
    @trixi_timeit timer() "Integrate w ⋅ k" begin
        # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
        integrate_via_indices(u_i, mesh, equations, dg, cache,
                              stage) do u_i, i, j, k, element, equations, dg, stage
            w_node = cons2entropy(get_node_vars(u_i, equations, dg, i, j, k, element),
                                  equations)
            stage_node = get_node_vars(stage, equations, dg, i, j, k, element)
            dot(w_node, stage_node)
        end
    end
end

@inline function entropy_difference(gamma, S_old, dS, u_gamma_dir, mesh,
                                    equations, dg, cache)
    return integrate(entropy_math, u_gamma_dir, mesh, equations, dg, cache) -
           S_old - gamma * dS
end

function gamma_bisection!(integrator, u_tmp_wrap, u_wrap, dir_wrap, S_old, dS,
                          mesh, equations, dg, cache)
    gamma_min = 0.1 # Not clear what value to choose here!
    gamma_max = 1.2 # For diffusive schemes, gamma > 1 is required to ensure EC

    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                            gamma_max *
                                            dir_wrap[.., element]
    end
    r_max = entropy_difference(gamma_max, S_old, dS, u_tmp_wrap,
                               mesh, equations, dg, cache)

    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                            gamma_min *
                                            dir_wrap[.., element]
    end
    r_min = entropy_difference(gamma_min, S_old, dS, u_tmp_wrap,
                               mesh, equations, dg, cache)

    # Check if there exists a root for `r` in the interval [gamma_min, gamma_max]
    if r_max > 0 && r_min < 0
        gamma_tol = 100 * eps(typeof(integrator.gamma))
        #bisection_its_max = 100
        #bisect_its = 0
        while gamma_max - gamma_min > gamma_tol #&& bisect_its < bisection_its_max
            integrator.gamma = (gamma_max + gamma_min) / 2

            @threaded for element in eachelement(dg, cache)
                @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                                    integrator.gamma *
                                                    dir_wrap[.., element]
            end
            r_gamma = entropy_difference(integrator.gamma, S_old, dS, u_tmp_wrap,
                                         mesh, equations, dg, cache)

            if r_gamma < 0
                gamma_min = integrator.gamma
            else
                gamma_max = integrator.gamma
            end
            #bisect_its += 1
        end
    else
        integrator.gamma = 1
    end
end

function gamma_newton!(integrator, u_tmp_wrap, u_wrap, dir_wrap, S_old, dS,
                       mesh, equations, dg, cache)
    # Custom Newton: Probably required for demonstrating the method
    step_scaling = 1.0 # > 1: Accelerated Newton, < 1: Damped Newton

    r_tol = 1e-14 # Similar to e.g. conservation error: 1e-14
    r_gamma = floatmax(typeof(integrator.gamma)) # Initialize with large value

    n_its_max = 5
    n_its = 0
    while abs(r_gamma) > r_tol && n_its < n_its_max
        @threaded for element in eachelement(dg, cache)
            @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                                integrator.gamma *
                                                dir_wrap[.., element]
        end
        r_gamma = entropy_difference(integrator.gamma, S_old, dS, u_tmp_wrap, mesh,
                                     equations,
                                     dg, cache)
        dr = int_w_dot_stage(dir_wrap, u_tmp_wrap, mesh, equations, dg, cache) - dS

        integrator.gamma -= step_scaling * r_gamma / dr

        n_its += 1
    end

    # Catch Newton failures
    if integrator.gamma < 0
        integrator.gamma = 1
    end
end
