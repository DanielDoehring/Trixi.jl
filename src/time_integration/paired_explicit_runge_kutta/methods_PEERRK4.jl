# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.

@muladd begin
#! format: noindent

@doc raw"""
    PairedExplicitERRK4(num_stages, base_path_a_coeffs::AbstractString, dt_opt = nothing;
                      c_const = 1.0f0)

    Parameters:
    - `num_stages` (`Int`): Number of stages in the paired explicit Runge-Kutta (P-ERK) method.
    - `base_path_a_coeffs` (`AbstractString`): Path to a file containing some coefficients in the A-matrix in 
      the Butcher tableau of the Runge Kutta method.
      The matrix should be stored in a text file at `joinpath(base_path_a_coeffs, "a_$(num_stages).txt")` and separated by line breaks.
    - `dt_opt` (`Float64`, optional): Optimal time step size for the simulation setup. Can be `nothing` if it is unknown. 
       In this case the optimal CFL number cannot be computed and the [`StepsizeCallback`](@ref) cannot be used.
    - `c_const` (`Float64`, optional): Value of abscissae $c$ in the Butcher tableau for the optimized coefficients. 
       Default is 1.0.

The following structures and methods provide an implementation of
the fourth-order paired explicit Runge-Kutta (P-ERK) method
optimized for a certain simulation setup (PDE, IC & BC, Riemann Solver, DG Solver).
The method has been proposed in 
- D. Doehring, L. Christmann, M. Schlottke-Lakemper, G. J. Gassner and M. Torrilhon (2024).
  Fourth-Order Paired-Explicit Runge-Kutta Methods
  [DOI:10.48550/arXiv.2408.05470](https://doi.org/10.48550/arXiv.2408.05470)
"""
mutable struct PairedExplicitERRK4 <: AbstractPairedExplicitRKSingle
    const num_stages::Int # S

    a_matrix::Matrix{Float64}
    # This part of the Butcher array matrix A is constant for all PERK methods, i.e., 
    # regardless of the optimized coefficients.
    a_matrix_constant::Matrix{Float64}
    c::Vector{Float64}
    dt_opt::Union{Float64, Nothing}
end # struct PairedExplicitERRK4

# Constructor for previously computed A Coeffs
function PairedExplicitERRK4(num_stages, base_path_a_coeffs::AbstractString,
                             dt_opt = nothing;
                             c_const = 1.0f0)
    a_matrix, a_matrix_constant, c = compute_PairedExplicitRK4_butcher_tableau(num_stages,
                                                                               base_path_a_coeffs;
                                                                               c_const)

    return PairedExplicitERRK4(num_stages, a_matrix, a_matrix_constant, c, dt_opt)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct PairedExplicitERRK4Integrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                             PairedExplicitRKOptions} <:
               AbstractPairedExplicitERRKSingleIntegrator
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    tdir::Real
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
    # PairedExplicitRK stages:
    k1::uType
    k_higher::uType

    # Entropy Relaxation additions
    direction::uType
    gamma::RealT
end

function init(ode::ODEProblem, alg::PairedExplicitERRK4;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    # PairedExplicitRK stages
    k1 = zero(u0)
    k_higher = zero(u0)

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    # For entropy relaxation
    direction = zero(u0)
    gamma = one(eltype(u0))

    integrator = PairedExplicitERRK4Integrator(u0, du, u_tmp, t0, tdir, dt, dt, iter,
                                               ode.p,
                                               (prob = ode,), ode.f, alg,
                                               PairedExplicitRKOptions(callback,
                                                                       ode.tspan;
                                                                       kwargs...),
                                               false, true, false,
                                               k1, k_higher,
                                               direction, gamma)

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

@inline function last_three_stages!(integrator::AbstractPairedExplicitERRKIntegrator,
                                    alg, p)
    mesh, equations, dg, cache = mesh_equations_solver_cache(p)

    # S - 2
    @threaded for u_ind in eachindex(integrator.u)
        integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                  alg.a_matrix_constant[1, 1] *
                                  integrator.k1[u_ind] +
                                  alg.a_matrix_constant[1, 2] *
                                  integrator.k_higher[u_ind]
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[alg.num_stages - 2] * integrator.dt)

    @threaded for u_ind in eachindex(integrator.du)
        integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
    end

    # S - 1
    @threaded for u_ind in eachindex(integrator.u)
        integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                  alg.a_matrix_constant[2, 1] *
                                  integrator.k1[u_ind] +
                                  alg.a_matrix_constant[2, 2] *
                                  integrator.k_higher[u_ind]
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[alg.num_stages - 1] * integrator.dt)

    @threaded for u_ind in eachindex(integrator.du)
        integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
    end

    k_higher_wrap = wrap_array(integrator.k_higher, p)
    u_tmp_wrap = wrap_array(integrator.u_tmp, p)
    # 0.5 = b_{S-1}
    # TODO: Combine integration of i-1, i!
    # => Would need to store u_tmp_wrap in yet another register!
    dS = int_w_dot_stage(k_higher_wrap, u_tmp_wrap, mesh, equations, dg, cache) / 2

    # S
    @threaded for i in eachindex(integrator.du)
        integrator.u_tmp[i] = integrator.u[i] +
                              alg.a_matrix_constant[3, 1] *
                              integrator.k1[i] +
                              alg.a_matrix_constant[3, 2] *
                              integrator.k_higher[i]
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[alg.num_stages] * integrator.dt)

    @threaded for i in eachindex(integrator.du)
        integrator.direction[i] = (integrator.k_higher[i] +
                                   integrator.du[i] * integrator.dt) / 2
    end

    du_wrap = wrap_array(integrator.du, integrator.p)
    # 0.5 = b_{S}
    dS += integrator.dt *
          int_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache) / 2

    u_wrap = wrap_array(integrator.u, integrator.p)
    S_old = integrate(entropy_math, u_wrap, mesh, equations, dg, cache)

    dir_wrap = wrap_array(integrator.direction, p)

    
    # Bisection for gamma
    @trixi_timeit timer() "ER: Bisection" begin
        gamma_bisection!(integrator, u_tmp_wrap, u_wrap, dir_wrap, S_old, dS,
                         mesh, equations, dg, cache)
    end
    

    #=
    # Newton search for gamma
    @trixi_timeit timer() "ER: Newton" begin
        gamma_newton!(integrator, u_tmp_wrap, u_wrap, dir_wrap, S_old, dS,
                      mesh, equations, dg, cache)
    end
    =#

    t_end = last(integrator.sol.prob.tspan)
    integrator.iter += 1
    # Last timestep shenanigans
    if integrator.t + integrator.gamma * integrator.dt > t_end ||
       isapprox(integrator.t + integrator.gamma * integrator.dt, t_end)
        integrator.t = t_end
        integrator.gamma = (t_end - integrator.t) / integrator.dt
        terminate!(integrator)
    else
        integrator.t += integrator.gamma * integrator.dt
    end

    # Do relaxed update
    @threaded for i in eachindex(integrator.u)
        integrator.u[i] += integrator.gamma * integrator.direction[i]
    end
end

function step!(integrator::PairedExplicitERRK4Integrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
        k1!(integrator, prob.p, alg.c)

        # k2
        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[2] * integrator.dt)

        @threaded for i in eachindex(integrator.du)
            integrator.k_higher[i] = integrator.du[i] * integrator.dt
        end

        # Higher stages until "constant" stages
        for stage in 3:(alg.num_stages - 3)
            # Construct current state
            @threaded for i in eachindex(integrator.du)
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

        last_three_stages!(integrator, alg, prob.p)
    end # PairedExplicitRK step timer

    # handle callbacks
    if callbacks isa CallbackSet
        for cb in callbacks.discrete_callbacks
            if cb.condition(integrator.u, integrator.t, integrator)
                cb.affect!(integrator)
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
function Base.resize!(integrator::AbstractPairedExplicitERRKIntegrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)
    # Addition for entropy relaxation PERK methods
    resize!(integrator.direction, new_size)
end
end # @muladd
