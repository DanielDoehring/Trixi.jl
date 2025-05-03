# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# NOTE: This could actually live in a more general location,
# as it is not PERK-specific.

@inline function int_w_dot_stage(stage, u_stage,
                                 mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                 equations, dg::DG, cache)
    @trixi_timeit timer() "Integrate w ⋅ k" begin
        # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
        integrate_via_indices(u_stage, mesh, equations, dg, cache,
                              stage) do u_stage, i, element, equations, dg, stage
            u_local = get_node_vars(u_stage, equations, dg, i, element)
            w_node = cons2entropy(u_local, equations)
            stage_node = get_node_vars(stage, equations, dg, i, element)
            dot(w_node, stage_node)
        end
    end
end

@inline function int_w_dot_stage(stage, u_stage,
                                 mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                             UnstructuredMesh2D, P4estMesh{2},
                                             T8codeMesh{2}},
                                 equations, dg::DG, cache)
    @trixi_timeit timer() "Integrate w ⋅ k" begin
        # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
        integrate_via_indices(u_stage, mesh, equations, dg, cache,
                              stage) do u_stage, i, j, element, equations, dg, stage
            u_local = get_node_vars(u_stage, equations, dg, i, j, element)
            w_node = cons2entropy(u_local, equations)
            stage_node = get_node_vars(stage, equations, dg, i, j, element)
            dot(w_node, stage_node)
        end
    end
end

@inline function int_w_dot_stage(stage, u_stage,
                                 mesh::Union{TreeMesh{3}, StructuredMesh{3},
                                             P4estMesh{3}, T8codeMesh{3}},
                                 equations, dg::DG, cache)
    @trixi_timeit timer() "Integrate w ⋅ k" begin
        # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
        integrate_via_indices(u_stage, mesh, equations, dg, cache,
                              stage) do u_stage, i, j, k, element, equations, dg, stage
            u_local = get_node_vars(u_stage, equations, dg, i, j, k, element)
            w_node = cons2entropy(u_local, equations)
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

# TODO: Field dS_accept which is checked whether relaxation should be performed at all
struct RelaxationSolverBisection{RealT <: Real} <: RelaxationSolver
    gamma_min::RealT    # Lower bound of the initial bracketing interval
    gamma_max::RealT    # Upper bound of the initial bracketing interval
    gamma_tol::RealT    # Absolute tolerance for the bracketing interval length
    max_iterations::Int # Maximum number of bisection iterations
end

function RelaxationSolverBisection(; gamma_min = 0.1, gamma_max = 1.2,
                                   gamma_tol = 1e-14, max_iterations = 25)
    return RelaxationSolverBisection(gamma_min, gamma_max, gamma_tol, max_iterations)
end

struct RelaxationSolverSecant{RealT <: Real} <: RelaxationSolver
    gamma_min::RealT    # Lower bound of the initial bracketing interval
    gamma_max::RealT    # Upper bound of the initial bracketing interval
    gamma_tol::RealT    # Absolute tolerance for the bracketing interval length
    max_iterations::Int # Maximum number of bisection iterations
end

function RelaxationSolverSecant(; gamma_min = 0.1, gamma_max = 1.2,
                                gamma_tol = 1e-14, max_iterations = 15)
    return RelaxationSolverSecant(gamma_min, gamma_max, gamma_tol, max_iterations)
end

function Base.show(io::IO,
                   relaxation_solver::Union{RelaxationSolverBisection,
                                            RelaxationSolverSecant})
    if typeof(relaxation_solver) <: RelaxationSolverBisection
        solver_type = "RelaxationSolverBisection"
    elseif typeof(relaxation_solver) <: RelaxationSolverSecant
        solver_type = "RelaxationSolverSecant"
    end
    print(io, "$solver_type(gamma_min=", relaxation_solver.gamma_min,
          ", gamma_max=", relaxation_solver.gamma_max,
          ", gamma_tol=", relaxation_solver.gamma_tol,
          ", max_iterations=", relaxation_solver.max_iterations, ")")
end
function Base.show(io::IO, ::MIME"text/plain",
                   relaxation_solver::Union{RelaxationSolverBisection,
                                            RelaxationSolverSecant})
    if get(io, :compact, false)
        show(io, relaxation_solver)
    else
        setup = [
            "gamma_min" => relaxation_solver.gamma_min,
            "gamma_max" => relaxation_solver.gamma_max,
            "gamma_tol" => relaxation_solver.gamma_tol,
            "max_iterations" => relaxation_solver.max_iterations
        ]
        if typeof(relaxation_solver) <: RelaxationSolverBisection
            solver_type = "RelaxationSolverBisection"
        elseif typeof(relaxation_solver) <: RelaxationSolverSecant
            solver_type = "RelaxationSolverSecant"
        end
        summary_box(io, solver_type, setup)
    end
end

struct RelaxationSolverNewton{RealT <: Real} <: RelaxationSolver
    step_scaling::RealT # Scaling factor for the Newton step
    root_tol::RealT     # Function-tolerance for the "relaxation equation" 
    # Minimum relaxation parameter. If the Newton iteration computes a value smaller than this, 
    # the relaxation parameter is set to 1.
    gamma_min::RealT
    max_iterations::Int # Maximum number of Newton iterations
end

function RelaxationSolverNewton(; step_scaling = 1.0, root_tol = 1e-14,
                                gamma_min = 1e-13, max_iterations = 5)
    return RelaxationSolverNewton(step_scaling, root_tol, gamma_min, max_iterations)
end

function Base.show(io::IO, relaxation_solver::RelaxationSolverNewton)
    print(io, "RelaxationSolverNewton(step_scaling=", relaxation_solver.step_scaling,
          ", root_tol=", relaxation_solver.root_tol,
          ", max_iterations=", relaxation_solver.max_iterations,
          ", gamma_min=", relaxation_solver.gamma_min, ")")
end
function Base.show(io::IO, ::MIME"text/plain",
                   relaxation_solver::RelaxationSolverNewton)
    if get(io, :compact, false)
        show(io, relaxation_solver)
    else
        setup = [
            "step_scaling" => relaxation_solver.step_scaling,
            "root_tol" => relaxation_solver.root_tol,
            "max_iterations" => relaxation_solver.max_iterations,
            "gamma_min" => relaxation_solver.gamma_min
        ]
        summary_box(io, "RelaxationSolverNewton", setup)
    end
end

function relaxation_solver!(integrator,
                            u_tmp_wrap, u_wrap, dir_wrap,
                            S_old, dS,
                            mesh, equations, dg::DG, cache,
                            relaxation_solver::RelaxationSolverBisection)
    @unpack gamma_min, gamma_max, gamma_tol, max_iterations = relaxation_solver

    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                            gamma_max * dir_wrap[.., element]
    end
    @trixi_timeit timer() "Δη" r_max=entropy_difference(gamma_max, S_old, dS,
                                                        u_tmp_wrap, mesh,
                                                        equations, dg, cache)

    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                            gamma_min * dir_wrap[.., element]
    end
    @trixi_timeit timer() "Δη" r_min=entropy_difference(gamma_min, S_old, dS,
                                                        u_tmp_wrap, mesh,
                                                        equations, dg, cache)

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
            @trixi_timeit timer() "Δη" r_gamma=entropy_difference(integrator.gamma,
                                                                  S_old, dS,
                                                                  u_tmp_wrap, mesh,
                                                                  equations, dg, cache)

            if r_gamma < 0
                gamma_min = integrator.gamma
            else
                gamma_max = integrator.gamma
            end
            iterations += 1
        end
    else
        integrator.gamma = 1
        # CARE: This is an experimental strategy: 
        # Set gamma to smallest value s.t. convergence is still assured
        #integrator.gamma = 1 - integrator.dt^(ORDER - 1)
    end

    return nothing
end

function relaxation_solver!(integrator,
                            u_tmp_wrap, u_wrap, dir_wrap,
                            S_old, dS,
                            mesh, equations, dg::DG, cache,
                            relaxation_solver::RelaxationSolverSecant)
    @unpack gamma_min, gamma_max, gamma_tol, max_iterations = relaxation_solver

    # Naming aliases to avoid confusion
    gamma_0, gamma_1 = gamma_min, gamma_max

    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                            gamma_1 * dir_wrap[.., element]
    end
    @trixi_timeit timer() "Δη" r_1=entropy_difference(gamma_1, S_old, dS,
                                                      u_tmp_wrap, mesh,
                                                      equations, dg, cache)

    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                            gamma_0 * dir_wrap[.., element]
    end
    @trixi_timeit timer() "Δη" r_0=entropy_difference(gamma_0, S_old, dS,
                                                      u_tmp_wrap, mesh,
                                                      equations, dg, cache)

    # Check if there exists a root for `r` in the interval [gamma_0, gamma_1] = [gamma_min, gamma_max]
    if r_1 > 0 && r_0 < 0
        # Perform first step which does not require extra evaluation of the `entropy_difference` function
        # We consider `gamma_1 = gamma_max` as the better initial guess, as this is for the default values most likely closer to the root
        gamma_0 = gamma_1 - r_1 * (gamma_1 - gamma_0) / (r_1 - r_0)
        # Switch order of 0, 1:
        gamma_0, gamma_1 = gamma_1, gamma_0

        iterations = 1
        integrator.gamma = gamma_1 # Write result back to integrator
        while abs(gamma_1 - gamma_0) > gamma_tol && iterations < max_iterations
            @threaded for element in eachelement(dg, cache)
                @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                                    gamma_1 * dir_wrap[.., element]
            end
            r_0 = r_1
            @trixi_timeit timer() "Δη" r_1=entropy_difference(gamma_1, S_old, dS,
                                                              u_tmp_wrap, mesh,
                                                              equations, dg, cache)

            gamma_0 = gamma_1 - r_1 * (gamma_1 - gamma_0) / (r_1 - r_0)
            # Switch order of 0, 1:
            gamma_0, gamma_1 = gamma_1, gamma_0

            iterations += 1
            integrator.gamma = gamma_1 # Write result back to integrator

            # Catch failure
            if integrator.gamma < gamma_min
                break
                integrator.gamma = 1
            end
        end
    else
        integrator.gamma = 1
        # CARE: This is an experimental strategy: 
        # Set gamma to smallest value s.t. convergence is still assured
        #integrator.gamma = 1 - integrator.dt^(ORDER - 1)
    end

    return nothing
end

#=
function relaxation_solver!(integrator::Union{AbstractPairedExplicitRelaxationRKIntegrator{ORDER},
                                              AbstractPairedExplicitRelaxationRKMultiParabolicIntegrator{ORDER}},
                            u_tmp_wrap, u_wrap, dir_wrap,
                            S_old, dS,
                            mesh, equations, dg::DG, cache,
                            relaxation_solver::RelaxationSolverNewton) where {ORDER}
=#
function relaxation_solver!(integrator,
                            u_tmp_wrap, u_wrap, dir_wrap,
                            S_old, dS,
                            mesh, equations, dg::DG, cache,
                            relaxation_solver::RelaxationSolverNewton)
    @unpack step_scaling, root_tol, max_iterations, gamma_min = relaxation_solver

    r_gamma = floatmax(typeof(integrator.gamma)) # Initialize with large value
    iterations = 0
    while abs(r_gamma) > root_tol && iterations < max_iterations
        @threaded for element in eachelement(dg, cache)
            @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                                integrator.gamma *
                                                dir_wrap[.., element]
        end
        @trixi_timeit timer() "Δη" r_gamma=entropy_difference(integrator.gamma,
                                                              S_old, dS,
                                                              u_tmp_wrap, mesh,
                                                              equations, dg, cache)
        dr = int_w_dot_stage(dir_wrap, u_tmp_wrap, mesh, equations, dg, cache) - dS

        integrator.gamma -= step_scaling * r_gamma / dr
        iterations += 1
    end

    # Catch Newton failures
    if integrator.gamma < gamma_min || isnan(integrator.gamma) ||
       isinf(integrator.gamma)
        integrator.gamma = 1
        # CARE: This is an experimental strategy: 
        # Set gamma to smallest value s.t. convergence is still assured
        #integrator.gamma = 1 - integrator.dt^(ORDER - 1)
    end

    return nothing
end
end # @muladd
