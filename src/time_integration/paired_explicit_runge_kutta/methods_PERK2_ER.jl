# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
using DelimitedFiles: readdlm
using LinearAlgebra: eigvals

@muladd begin
#! format: noindent

@doc raw"""
    PairedExplicitRK2_ER(num_stages, base_path_monomial_coeffs::AbstractString, dt_opt,
                      bS = 1.0, cS = 0.5)
    PairedExplicitRK2_ER(num_stages, tspan, semi::AbstractSemidiscretization;
                      verbose = false, bS = 1.0, cS = 0.5)
    PairedExplicitRK2_ER(num_stages, tspan, eig_vals::Vector{ComplexF64};
                      verbose = false, bS = 1.0, cS = 0.5)
    Parameters:
    - `num_stages` (`Int`): Number of stages in the PERK method.
    - `base_path_monomial_coeffs` (`AbstractString`): Path to a file containing 
      monomial coefficients of the stability polynomial of PERK method.
      The coefficients should be stored in a text file at `joinpath(base_path_monomial_coeffs, "gamma_$(num_stages).txt")` and separated by line breaks.
    - `dt_opt` (`Float64`): Optimal time step size for the simulation setup.
    - `tspan`: Time span of the simulation.
    - `semi` (`AbstractSemidiscretization`): Semidiscretization setup.
    -  `eig_vals` (`Vector{ComplexF64}`): Eigenvalues of the Jacobian of the right-hand side (rhs) of the ODEProblem after the
      equation has been semidiscretized.
    - `verbose` (`Bool`, optional): Verbosity flag, default is false.
    - `bS` (`Float64`, optional): Value of b in the Butcher tableau at b_s, when 
      s is the number of stages, default is 1.0.
    - `cS` (`Float64`, optional): Value of c in the Butcher tableau at c_s, when
      s is the number of stages, default is 0.5.

The following structures and methods provide a minimal implementation of
the second-order paired explicit Runge-Kutta (PERK) method
optimized for a certain simulation setup (PDE, IC & BC, Riemann Solver, DG Solver).

- Brian Vermeire (2019).
  Paired explicit Runge-Kutta schemes for stiff systems of equations
  [DOI: 10.1016/j.jcp.2019.05.014](https://doi.org/10.1016/j.jcp.2019.05.014)
"""
mutable struct PairedExplicitRK2_ER <: AbstractPairedExplicitRKSingle
    const num_stages::Int

    a_matrix::Matrix{Float64}
    c::Vector{Float64}
    b1::Float64
    bS::Float64
    cS::Float64
    dt_opt::Float64
end # struct PairedExplicitRK2_ER

# Constructor that reads the coefficients from a file
function PairedExplicitRK2_ER(num_stages, base_path_monomial_coeffs::AbstractString,
                              dt_opt,
                              bS = 1.0, cS = 0.5)
    # If the user has the monomial coefficients, they also must have the optimal time step
    a_matrix, c = compute_PairedExplicitRK2_butcher_tableau(num_stages,
                                                            base_path_monomial_coeffs,
                                                            bS, cS)

    return PairedExplicitRK2_ER(num_stages, a_matrix, c, 1 - bS, bS, cS, dt_opt)
end

# Constructor that calculates the coefficients with polynomial optimizer from a
# semidiscretization
function PairedExplicitRK2_ER(num_stages, tspan, semi::AbstractSemidiscretization;
                              verbose = false,
                              bS = 1.0, cS = 0.5)
    eig_vals = eigvals(jacobian_ad_forward(semi))

    return PairedExplicitRK2_ER(num_stages, tspan, eig_vals; verbose, bS, cS)
end

# Constructor that calculates the coefficients with polynomial optimizer from a
# list of eigenvalues
function PairedExplicitRK2_ER(num_stages, tspan, eig_vals::Vector{ComplexF64};
                              verbose = false,
                              bS = 1.0, cS = 0.5)
    a_matrix, c, dt_opt = compute_PairedExplicitRK2_butcher_tableau(num_stages,
                                                                    eig_vals, tspan,
                                                                    bS, cS;
                                                                    verbose)

    return PairedExplicitRK2_ER(num_stages, a_matrix, c, 1 - bS, bS, cS, dt_opt)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRK2_ERIntegrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                              PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKSingleIntegrator
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    tdir::RealT
    dt::RealT # current time step
    dtcache::RealT # manually set time step
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F
    alg::Alg # This is our own class written above; Abbreviation for ALGorithm
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # PairedExplicitRK2_ER stages:
    k1::uType
    k_higher::uType
    # Naive implementation for entropy relaxation:
    direction::uType
    num_timestep_relaxations::Int
end

"""
    calculate_cfl(ode_algorithm::AbstractPairedExplicitRKSingle, ode)

This function computes the CFL number once using the initial condition of the problem and the optimal timestep (`dt_opt`) from the ODE algorithm.
"""
function calculate_cfl(ode_algorithm::AbstractPairedExplicitRKSingle, ode)
    t0 = first(ode.tspan)
    u_ode = ode.u0
    semi = ode.p
    dt_opt = ode_algorithm.dt_opt

    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    cfl_number = dt_opt / max_dt(u, t0, mesh,
                        have_constant_speed(equations), equations,
                        solver, cache)
    return cfl_number
end

"""
    add_tstop!(integrator::PairedExplicitRK2_ERIntegrator, t)
Add a time stop during the time integration process.
This function is called after the periodic SaveSolutionCallback to specify the next stop to save the solution.
"""
function add_tstop!(integrator::PairedExplicitRK2_ERIntegrator, t)
    integrator.tdir * (t - integrator.t) < zero(integrator.t) &&
        error("Tried to add a tstop that is behind the current time. This is strictly forbidden")
    # We need to remove the first entry of tstops when a new entry is added.
    # Otherwise, the simulation gets stuck at the previous tstop and dt is adjusted to zero.
    if length(integrator.opts.tstops) > 1
        pop!(integrator.opts.tstops)
    end
    push!(integrator.opts.tstops, integrator.tdir * t)
end

has_tstop(integrator::PairedExplicitRK2_ERIntegrator) = !isempty(integrator.opts.tstops)
first_tstop(integrator::PairedExplicitRK2_ERIntegrator) = first(integrator.opts.tstops)

function init(ode::ODEProblem, alg::PairedExplicitRK2_ER;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    # PairedExplicitRK2_ER stages
    k1 = zero(u0)
    k_higher = zero(u0)

    # Naive implementation for entropy relaxation
    direction = zero(u0)

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    integrator = PairedExplicitRK2_ERIntegrator(u0, du, u_tmp, t0, tdir, dt, dt, iter,
                                                ode.p,
                                                (prob = ode,), ode.f, alg,
                                                PairedExplicitRKOptions(callback,
                                                                        ode.tspan;
                                                                        kwargs...),
                                                false, true, false,
                                                k1, k_higher,
                                                direction, 0)

    # initialize callbacks
    if callback isa CallbackSet
        for cb in callback.continuous_callbacks
            throw(ArgumentError("Continuous callbacks are unsupported with paired explicit Runge-Kutta methods."))
        end
        for cb in callback.discrete_callbacks
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    end

    return integrator
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::PairedExplicitRK2_ER;
               dt, callback = nothing, kwargs...)
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    solve!(integrator)
end

function solve!(integrator::PairedExplicitRK2_ERIntegrator)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
        step!(integrator)
    end # "main loop" timer

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

function int_w_dot_stage(stage, u_i,
                         mesh::Union{TreeMesh{1}, StructuredMesh{1}}, equations, dg::DG,
                         cache)
    # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
    integrate_via_indices(u_i, mesh, equations, dg, cache,
                          stage) do u_i, i, element, equations, dg, stage
        w_node = cons2entropy(get_node_vars(u_i, equations, dg, i, element),
                              equations)
        stage_node = get_node_vars(stage, equations, dg, i, element)
        dot(w_node, stage_node)
    end
end

function int_w_dot_stage(stage, u_i,
                         mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                     StructuredMeshView{2},
                                     UnstructuredMesh2D, P4estMesh{2}, T8codeMesh{2}},
                         equations, dg::DG, cache)
    # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
    integrate_via_indices(u_i, mesh, equations, dg, cache,
                          stage) do u_i, i, j, element, equations, dg, stage
        w_node = cons2entropy(get_node_vars(u_i, equations, dg, i, j, element),
                              equations)
        stage_node = get_node_vars(stage, equations, dg, i, j, element)
        dot(w_node, stage_node)
    end
end

function int_w_dot_stage(stage, u_i,
                         mesh::Union{TreeMesh{3}, StructuredMesh{3}, P4estMesh{3},
                                     T8codeMesh{3}},
                         equations, dg::DG, cache)
    # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
    integrate_via_indices(u_i, mesh, equations, dg, cache,
                          stage) do u_i, i, j, k, element, equations, dg, stage
        w_node = cons2entropy(get_node_vars(u_i, equations, dg, i, j, k, element),
                              equations)
        stage_node = get_node_vars(stage, equations, dg, i, j, k, element)
        dot(w_node, stage_node)
    end
end

function entropy_diff(gamma, S_old, dS, u_gamma_dir, mesh, equations, dg, cache)
    return integrate(entropy_math, u_gamma_dir, mesh, equations, dg, cache) -
           S_old - gamma * dS
end

function step!(integrator::PairedExplicitRK2_ERIntegrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    mesh, equations, dg, cache = mesh_equations_solver_cache(integrator.p)

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    modify_dt_for_tstops!(integrator)

    # if the next iteration would push the simulation beyond the end time, 
    # set dt accordingly to avoid overshoots
    if integrator.t + integrator.dt > t_end ||
       isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
    end

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
        # k1
        integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
        @threaded for i in eachindex(integrator.du)
            integrator.k1[i] = integrator.du[i] * integrator.dt
        end

        # Construct current state
        @threaded for i in eachindex(integrator.u)
            integrator.u_tmp[i] = integrator.u[i] + alg.c[2] * integrator.k1[i]
        end
        # k2
        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[2] * integrator.dt)

        @threaded for i in eachindex(integrator.du)
            integrator.k_higher[i] = integrator.du[i] * integrator.dt
        end

        # Higher stages
        for stage in 3:(alg.num_stages)
            # Construct current state
            @threaded for i in eachindex(integrator.u)
                integrator.u_tmp[i] = integrator.u[i] +
                                      alg.a_matrix[stage - 2, 1] *
                                      integrator.k1[i] +
                                      alg.a_matrix[stage - 2, 2] *
                                      integrator.k_higher[i]
            end

            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                         integrator.t + alg.c[stage] * integrator.dt)

            @threaded for i in eachindex(integrator.du)
                integrator.k_higher[i] = integrator.du[i] * integrator.dt
            end
        end

        @threaded for i in eachindex(integrator.u)
            integrator.direction[i] = alg.b1 * integrator.k1[i] +
                                      alg.bS * integrator.k_higher[i]
        end

        @trixi_timeit timer() "Entropy Relaxation" begin
            @trixi_timeit timer() "ER: Initial check" begin
                u_wrap = wrap_array(integrator.u, integrator.p)
                u_tmp_wrap = wrap_array(integrator.u_tmp, integrator.p)
                k1_wrap = wrap_array(integrator.k1, integrator.p)
                k_higher_wrap = wrap_array(integrator.k_higher, integrator.p)
                dir_wrap = wrap_array(integrator.direction, integrator.p)
                # Re-use du as helper data structure (not needed anymore)
                u_gamma_dir_wrap = wrap_array(integrator.du, integrator.p)

                S_old = integrate(entropy_math, u_wrap, mesh, equations, dg, cache)
                dS = (alg.b1 *
                      int_w_dot_stage(k1_wrap, u_wrap, mesh, equations, dg, cache) +
                      # u_tmp corresponds to input leading to last k_higher
                      alg.bS *
                      int_w_dot_stage(k_higher_wrap, u_tmp_wrap, mesh, equations, dg,
                                      cache))

                gamma = 1.0 # Default value if entropy relaxation methodology not applicable

                # TODO: If we do not want to sacrifice order, we would need to restrict this lower bound to 1 - O(dt)
                gamma_min = 0.5
                gamma_max = 1.0
                bisection_its_max = 100

                @threaded for element in eachelement(dg, cache)
                    @views @. u_gamma_dir_wrap[.., element] = u_wrap[.., element] +
                                                              gamma_max *
                                                              dir_wrap[.., element]
                end
                r_max = entropy_diff(gamma_max, S_old, dS, u_gamma_dir_wrap, mesh,
                                     equations, dg, cache)

                @threaded for element in eachelement(dg, cache)
                    @views @. u_gamma_dir_wrap[.., element] = u_wrap[.., element] +
                                                              gamma_min *
                                                              dir_wrap[.., element]
                end
                r_min = entropy_diff(gamma_min, S_old, dS, u_gamma_dir_wrap,
                                     mesh, equations, dg, cache)
            end

            # Check if there exists a root for `r` in the interval [gamma_min, gamma_max]
            if r_max > 0 && r_min < 0 # && 
                # integrator.finalstep == false # Avoid last-step shenanigans for now

                integrator.num_timestep_relaxations += 1
                # Init with gamma_0
                gamma_eps = 1e-13

                bisect_its = 0
                @trixi_timeit timer() "ER: Bisection" while gamma_max - gamma_min >
                                                            gamma_eps &&
                                                            bisect_its <
                                                            bisection_its_max
                    gamma = 0.5 * (gamma_max + gamma_min)

                    @threaded for element in eachelement(dg, cache)
                        @views @. u_gamma_dir_wrap[.., element] = u_wrap[.., element] +
                                                                  gamma *
                                                                  dir_wrap[.., element]
                    end
                    r_gamma = entropy_diff(gamma, S_old, dS, u_gamma_dir_wrap,
                                           mesh, equations, dg, cache)

                    if r_gamma < 0
                        gamma_min = gamma
                    else
                        gamma_max = gamma
                    end
                    bisect_its += 1
                end
            end

            #=
            # Sanity check: Condition for desired order of convergence
            if integrator.finalstep == false
                # Condition for convergence: gamma = 1 + O[dt^(p-1)] = 1 + O(dt)
                gamma = 1.0 - integrator.dt
            else
                gamma = 1.0
            end
            =#

            # Last timestep shenanigans
            if integrator.t + gamma * integrator.dt > t_end ||
               isapprox(integrator.t + gamma * integrator.dt, t_end)
                integrator.t = t_end
                gamma = (t_end - integrator.t) / integrator.dt
                terminate!(integrator)
                println("# Relaxed timesteps: ", integrator.num_timestep_relaxations)
            else
                integrator.t += gamma * integrator.dt
            end
        end

        @threaded for i in eachindex(integrator.u)
            integrator.u[i] += gamma * integrator.direction[i]
        end
    end # PairedExplicitRK2_ER step

    integrator.iter += 1

    @trixi_timeit timer() "Step-Callbacks" begin
        # handle callbacks
        if callbacks isa CallbackSet
            foreach(callbacks.discrete_callbacks) do cb
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
                return nothing
            end
        end
    end

    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
        @warn "Interrupted. Larger maxiters is needed."
        terminate!(integrator)
    end
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PairedExplicitRK2_ERIntegrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)

    resize!(integrator.direction, new_size)
end
end # @muladd
