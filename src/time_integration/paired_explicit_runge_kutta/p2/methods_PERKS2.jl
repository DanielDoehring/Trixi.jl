# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    PairedExplicitRK2Split(num_stages,
                          base_path_monomial_coeffs::AbstractString,
                          base_path_monomial_coeffs_para::AbstractString;
                          dt_opt = nothing, bS = 1.0, cS = 0.5)
"""
struct PairedExplicitRK2Split <:
       AbstractPairedExplicitRKSplitSingle{2}
    num_stages::Int

    a_matrix::Matrix{Float64}
    a_matrix_para::Matrix{Float64}
    c::Vector{Float64}
    b1::Float64
    bS::Float64
    cS::Float64

    dt_opt::Union{Float64, Nothing}
end

# Constructor that reads the coefficients from a file
function PairedExplicitRK2Split(num_stages,
                                base_path_monomial_coeffs::AbstractString,
                                base_path_monomial_coeffs_para::AbstractString;
                                dt_opt = nothing,
                                bS = 1.0, cS = 0.5)
    @assert num_stages>=2 "PERK2 requires at least two stages"
    # If the user has the monomial coefficients, they also must have the optimal time step
    a_matrix, c = compute_PairedExplicitRK2_butcher_tableau(num_stages,
                                                            base_path_monomial_coeffs,
                                                            bS, cS)

    a_matrix_para, _ = compute_PairedExplicitRK2_butcher_tableau(num_stages,
                                                                 base_path_monomial_coeffs_para,
                                                                 bS, cS)

    return PairedExplicitRK2Split(num_stages, a_matrix, a_matrix_para, c, 1 - bS, bS,
                                  cS, dt_opt)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRK2SplitIntegrator{RealT <: Real, uType,
                                                Params, Sol, F,
                                                PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKSplitSingleIntegrator{2}
    u::uType
    du::uType # In-place output of `f`
    u_tmp::uType # Used for building the argument to `f`
    t::RealT
    tdir::RealT # DIRection of time integration, i.e., if one marches forward or backward in time
    dt::RealT # current time step
    dtcache::RealT # manually set time step
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F # `rhs!` of the semidiscretization
    alg::PairedExplicitRK2Split
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # Additional PERK register
    k1::uType
    # For split (hyperbolic-parabolic) problems
    du_para::uType # Stores the parabolic part of the overall rhs!
    k1_para::uType # Additional PERK register for the parabolic part
end

function init(ode::ODEProblem, alg::PairedExplicitRK2Split;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    k1 = zero(u0) # Additional PERK register
    du_para = zero(u0) # Stores the parabolic part of the overall rhs!
    k1_para = zero(u0) # Additional PERK register for the parabolic part

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    integrator = PairedExplicitRK2SplitIntegrator(u0, du, u_tmp,
                                                  t0, tdir, dt, zero(dt),
                                                  iter, ode.p,
                                                  (prob = ode,), ode.f,
                                                  alg,
                                                  PairedExplicitRKOptions(callback,
                                                                          ode.tspan;
                                                                          kwargs...),
                                                  false, true, false,
                                                  k1, du_para, k1_para)

    initialize_callbacks!(callback, integrator)

    return integrator
end

function step!(integrator::AbstractPairedExplicitRKSplitIntegrator{2})
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    #modify_dt_for_tstops!(integrator)

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end ||
       isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
        terminate!(integrator)
    end

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
        # First and second stage are identical across all single/standalone PERK methods
        PERK_k1!(integrator, prob.p)
        PERK_k2!(integrator, prob.p, alg)

        # Higher stages
        for stage in 3:(alg.num_stages)
            PERK_ki!(integrator, prob.p, alg, stage)
        end

        @threaded for i in eachindex(integrator.u)
            #=
            integrator.u[i] += integrator.dt *
                               (alg.b1 * (integrator.k1[i] + integrator.k1_para[i]) +
                                alg.bS * (integrator.du[i] + integrator.du_para[i]))
            =#

            # More performant version for b1 = 0, bS = 1
            # Try optimize for `@muladd`: avoid `+=`
            integrator.u[i] = integrator.u[i] +
                              integrator.dt * (integrator.du[i] + integrator.du_para[i])
        end
    end

    integrator.iter += 1
    integrator.t += integrator.dt

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

    return nothing
end
end # @muladd
