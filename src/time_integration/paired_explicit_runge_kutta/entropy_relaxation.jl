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
                                    equations, dg::DG, cache)
    return integrate(entropy_math, u_gamma_dir, mesh, equations, dg, cache) -
           S_old - gamma * dS
end

abstract type RelaxationSolver end

struct EntropyRelaxationBisection <: RelaxationSolver
    gamma_min::Real
    gamma_max::Real
    gamma_tol::Real
    max_iterations::Int
end

function EntropyRelaxationBisection(; gamma_min = 0.1, gamma_max = 1.2,
                                    gamma_tol = 100 * eps(typeof(gamma_min)),
                                    max_iterations = 100)
    return EntropyRelaxationBisection(gamma_min, gamma_max, gamma_tol, max_iterations)
end

struct EntropyRelaxationNewton <: RelaxationSolver
    step_scaling::Real
    root_tol::Real
    max_iterations::Int
end

function EntropyRelaxationNewton(; step_scaling = 1.0, root_tol = 1e-14,
                                 max_iterations = 5)
    return EntropyRelaxationNewton(step_scaling, root_tol, max_iterations)
end

function relaxation_solver!(integrator::AbstractPairedExplicitRelaxationRKIntegrator,
                            u_tmp_wrap, u_wrap, dir_wrap,
                            S_old, dS,
                            mesh, equations, dg::DG, cache,
                            relaxation_solver::EntropyRelaxationBisection)
    @unpack gamma_min, gamma_max, gamma_tol, max_iterations = relaxation_solver

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
        iterations = 0
        while gamma_max - gamma_min > gamma_tol && iterations < max_iterations
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
            iterations += 1
        end
    else
        integrator.gamma = 1
    end
end

function relaxation_solver!(integrator::AbstractPairedExplicitRelaxationRKIntegrator,
                            u_tmp_wrap, u_wrap, dir_wrap,
                            S_old, dS,
                            mesh, equations, dg::DG, cache,
                            relaxation_solver::EntropyRelaxationNewton)
    @unpack step_scaling, root_tol, max_iterations = relaxation_solver

    r_gamma = floatmax(typeof(integrator.gamma)) # Initialize with large value
    iterations = 0
    while abs(r_gamma) > root_tol && iterations < max_iterations
        @threaded for element in eachelement(dg, cache)
            @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                                integrator.gamma *
                                                dir_wrap[.., element]
        end
        r_gamma = entropy_difference(integrator.gamma, S_old, dS, u_tmp_wrap,
                                     mesh, equations, dg, cache)
        dr = int_w_dot_stage(dir_wrap, u_tmp_wrap, mesh, equations, dg, cache) - dS

        integrator.gamma -= step_scaling * r_gamma / dr
        iterations += 1
    end

    # Catch Newton failures
    if integrator.gamma < 0 || isnan(integrator.gamma) || isinf(integrator.gamma)
        integrator.gamma = 1
    end
end
