# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.

@muladd begin
#! format: noindent

# Compute the Butcher tableau for a paired explicit Runge-Kutta method order 3
# using provided values of coefficients a in A-matrix of Butcher tableau
function compute_EmbeddedPairedRK3_butcher_tableau(num_stages, num_stage_evals,
                                                   base_path_coeffs::AbstractString;
                                                   cS2)

    # Initialize array of c
    c = compute_c_coeffs(num_stages, cS2)

    # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
    coeffs_max = num_stage_evals - 2

    a_matrix = zeros(coeffs_max, 2)
    a_matrix[:, 1] = c[3:end]

    b = zeros(coeffs_max)

    path_a_coeffs = joinpath(base_path_coeffs,
                             "a_" * string(num_stages) * "_" * string(num_stage_evals) *
                             ".txt")

    path_b_coeffs = joinpath(base_path_coeffs,
                             "b_" * string(num_stages) * "_" * string(num_stage_evals) *
                             ".txt")

    @assert isfile(path_a_coeffs) "Couldn't find file $path_a_coeffs"
    a_coeffs = readdlm(path_a_coeffs, Float64)
    num_a_coeffs = size(a_coeffs, 1)

    @assert num_a_coeffs == coeffs_max
    # Fill A-matrix in P-ERK style
    a_matrix[:, 1] -= a_coeffs
    a_matrix[:, 2] = a_coeffs

    @assert isfile(path_b_coeffs) "Couldn't find file $path_b_coeffs"
    b = vec(readdlm(path_b_coeffs, Float64))
    num_b_coeffs = size(b, 1)
    @assert num_b_coeffs == coeffs_max + 2

    return a_matrix, b, c
end

@doc raw"""
    EmbeddedPairedRK3(num_stages, num_stage_evals, base_path_coeffs::AbstractString;
                      cS2 = 1.0f0)
"""
mutable struct EmbeddedPairedRK3 <: AbstractPairedExplicitRKSingle
    const num_stages::Int # S
    const num_stage_evals::Int # e

    a_matrix::Matrix{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
end # struct EmbeddedPairedRK3

# Constructor for previously computed A Coeffs
function EmbeddedPairedRK3(num_stages, num_stage_evals,
                           base_path_coeffs::AbstractString;
                           cS2 = 1.0f0)
    a_matrix, b, c = compute_EmbeddedPairedRK3_butcher_tableau(num_stages,
                                                               num_stage_evals,
                                                               base_path_coeffs;
                                                               cS2)
                                                             
    return EmbeddedPairedRK3(num_stages, num_stage_evals, a_matrix, b, c)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct EmbeddedPairedRK3Integrator{RealT <: Real, uType, Params, Sol, F, Alg,
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
    # PairedExplicitRK stages:
    k1::uType
    k_higher::uType
    # Extra register for saving u
    u_old::uType
end

function init(ode::ODEProblem, alg::EmbeddedPairedRK3;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    # PairedExplicitRK stages
    k1 = zero(u0)
    k_higher = zero(u0)

    # Required for embedded, i.e., populated PERK method
    u_old = zero(u0)

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    integrator = EmbeddedPairedRK3Integrator(u0, du, u_tmp, t0, tdir, dt, dt, iter,
                                             ode.p,
                                             (prob = ode,), ode.f, alg,
                                             PairedExplicitRKOptions(callback,
                                                                     ode.tspan;
                                                                     kwargs...),
                                             false, true, false,
                                             k1, k_higher,
                                             u_old)

    # initialize callbacks
    if callback isa CallbackSet
        for cb in callback.continuous_callbacks
            error("Continuous callbacks are unsupported with paired explicit Runge-Kutta methods.")
        end
        for cb in callback.discrete_callbacks
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    elseif !isnothing(callback)
        error("unsupported")
    end

    return integrator
end

function step!(integrator::EmbeddedPairedRK3Integrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    modify_dt_for_tstops!(integrator)

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end ||
       isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
        terminate!(integrator)
    end

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
        # Set `u_old` to incoming `u`    
        @threaded for i in eachindex(integrator.du)
          integrator.u_old[i] = integrator.u[i]
        end

        # k1 is in general required, as we use b_1 to satisfy the first-order cons. cond.
        integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
        @threaded for i in eachindex(integrator.du)
            integrator.k1[i] = integrator.du[i] * integrator.dt
            # Add the contribution of the first stage (b_1 in general non-zero)
            integrator.u[i] += alg.b[1] * integrator.k1[i]
        end

        # This is the first stage after stage 1 for which we need to evaluate `k_higher`
        stage = alg.num_stages - alg.num_stage_evals + 2
        # Construct current state
        @threaded for i in eachindex(integrator.du)
            integrator.u_tmp[i] = integrator.u_old[i] + alg.c[stage] * integrator.k1[i]
        end

        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[stage] * integrator.dt)

        @threaded for i in eachindex(integrator.du)
            integrator.k_higher[i] = integrator.du[i] * integrator.dt
            integrator.u[i] += alg.b[stage] * integrator.k_higher[i]
        end

        # Non-reducible stages
        for stage in (alg.num_stages - alg.num_stage_evals + 3):alg.num_stages
            # Construct current state
            @threaded for i in eachindex(integrator.du)
              integrator.u_tmp[i] = integrator.u_old[i] +
                                    alg.a_matrix[stage - 2, 1] *
                                    integrator.k1[i] +
                                    alg.a_matrix[stage - 2, 2] *
                                    integrator.k_higher[i]
            end

            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                         integrator.t + alg.c[stage] * integrator.dt)

            @threaded for i in eachindex(integrator.du)
                integrator.k_higher[i] = integrator.du[i] * integrator.dt
                integrator.u[i] += alg.b[stage] * integrator.k_higher[i]
            end
        end
        
    end # PairedExplicitRK step timer

    integrator.iter += 1
    integrator.t += integrator.dt

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
function Base.resize!(integrator::EmbeddedPairedRK3Integrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)

    resize!(integrator.u_old, new_size)
end
end # @muladd