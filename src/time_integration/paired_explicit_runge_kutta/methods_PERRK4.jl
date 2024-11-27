# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.

@muladd begin
#! format: noindent

@doc raw"""
    PairedExplicitRelaxationRK4(num_stages, base_path_a_coeffs::AbstractString, dt_opt = nothing;
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
mutable struct PairedExplicitRelaxationRK4 <: AbstractPairedExplicitRKSingle
    const num_stages::Int # S

    a_matrix::Matrix{Float64}
    # This part of the Butcher array matrix A is constant for all PERK methods, i.e., 
    # regardless of the optimized coefficients.
    a_matrix_constant::Matrix{Float64}
    c::Vector{Float64}
    dt_opt::Union{Float64, Nothing}
end # struct PairedExplicitRelaxationRK4

# Constructor for previously computed A Coeffs
function PairedExplicitRelaxationRK4(num_stages, base_path_a_coeffs::AbstractString,
                                     dt_opt = nothing;
                                     c_const = 1.0f0)
    a_matrix, a_matrix_constant, c = compute_PairedExplicitRK4_butcher_tableau(num_stages,
                                                                               base_path_a_coeffs;
                                                                               c_const)

    return PairedExplicitRelaxationRK4(num_stages, a_matrix, a_matrix_constant, c,
                                       dt_opt)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct PairedExplicitRelaxationRK4Integrator{RealT <: Real, uType, Params, Sol,
                                                     F, Alg,
                                                     PairedExplicitRKOptions} <:
               AbstractPairedExplicitRelaxationRKSingleIntegrator
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
    # Additional PERK stage
    k1::uType
    # Entropy Relaxation addition
    gamma::RealT
end

function init(ode::ODEProblem, alg::PairedExplicitRelaxationRK4;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    k1 = zero(u0) # Additional PERK stage

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    # For entropy relaxation
    gamma = one(eltype(u0))

    integrator = PairedExplicitRelaxationRK4Integrator(u0, du, u_tmp, t0, tdir, dt, dt,
                                                       iter,
                                                       ode.p,
                                                       (prob = ode,), ode.f, alg,
                                                       PairedExplicitRKOptions(callback,
                                                                               ode.tspan;
                                                                               kwargs...),
                                                       false, true, false,
                                                       k1,
                                                       gamma)

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

@inline function last_three_stages!(integrator::AbstractPairedExplicitRelaxationRKIntegrator,
                                    p, alg)
    mesh, equations, dg, cache = mesh_equations_solver_cache(p)

    for stage in 1:2
        @threaded for u_ind in eachindex(integrator.u)
            integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                      integrator.dt *
                                      (alg.a_matrix_constant[stage, 1] *
                                       integrator.k1[u_ind] +
                                       alg.a_matrix_constant[stage, 2] *
                                       integrator.du[u_ind])
        end

        integrator.f(integrator.du, integrator.u_tmp, p,
                     integrator.t +
                     alg.c[alg.num_stages - 3 + stage] * integrator.dt)
    end

    du_wrap = wrap_array(integrator.du, p)
    u_tmp_wrap = wrap_array(integrator.u_tmp, p)
    # 0.5 = b_{S-1}
    # IDEA: Combine integration of i-1, i?
    # => Would need to store u_tmp_wrap in yet another register!
    dS = 0.5 * integrator.dt *
         int_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

    # Last stage
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              integrator.dt *
                              (alg.a_matrix_constant[3, 1] * integrator.k1[i] +
                               alg.a_matrix_constant[3, 2] * integrator.du[i])
    end

    # Safe K_{S-1} in `k1`:
    @threaded for i in eachindex(integrator.u)
        integrator.k1[i] = integrator.du[i]
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[alg.num_stages] * integrator.dt)

    # Note: We re-use `k1` for the "direction"
    # Note: For efficiency, we multiply the direction already by dt here!
    @threaded for i in eachindex(integrator.u)
        integrator.k1[i] = 0.5 * integrator.dt * (integrator.k1[i] +
                                                  integrator.du[i])
    end

    # 0.5 = b_{S}
    dS += 0.5 * integrator.dt *
          int_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

    u_wrap = wrap_array(integrator.u, integrator.p)
    S_old = integrate(entropy_math, u_wrap, mesh, equations, dg, cache)

    # Note: We re-use `k1` for the "direction"
    dir_wrap = wrap_array(integrator.k1, p)

    #=
    # Bisection for gamma
    @trixi_timeit timer() "ER: Bisection" begin
        gamma_bisection!(integrator, u_tmp_wrap, u_wrap, dir_wrap, S_old, dS,
                         mesh, equations, dg, cache)
    end
    =#

    # Newton search for gamma
    @trixi_timeit timer() "ER: Newton" begin
        gamma_newton!(integrator, u_tmp_wrap, u_wrap, dir_wrap, S_old, dS,
                      mesh, equations, dg, cache)
    end

    integrator.iter += 1
    # Check if due to entropy relaxation the final step is not reached
    if integrator.finalstep == true && integrator.gamma != 1
        # If we would go beyond the final time, clip gamma at 1.0
        if integrator.gamma > 1.0
            integrator.gamma = 1.0
        else # If we are below the final time, reset finalstep flag
            integrator.finalstep = false
        end
    end
    integrator.t += integrator.gamma * integrator.dt

    # Do relaxed update
    @threaded for i in eachindex(integrator.u)
        # Note: We re-use `k1` for the "direction"
        integrator.u[i] += integrator.gamma * integrator.k1[i]
    end
end

function step!(integrator::PairedExplicitRelaxationRK4Integrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end ||
       isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
        terminate!(integrator)
    end

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
        PERK_k1!(integrator, prob.p)
        PERK_k2!(integrator, prob.p, alg)

        # Higher stages until "constant" stages
        for stage in 3:(alg.num_stages - 3)
            PERK_ki!(integrator, prob.p, alg, stage)
        end

        last_three_stages!(integrator, prob.p, alg)
    end

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
end # @muladd
