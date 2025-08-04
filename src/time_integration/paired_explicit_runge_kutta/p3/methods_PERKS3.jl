# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    PairedExplicitRK3Split(num_stages, 
                           base_path_a_coeffs::AbstractString,
                           base_path_a_coeffs_para::AbstractString,
                           dt_opt = nothing;
                           cS2 = 1.0f0)
"""
struct PairedExplicitRK3Split <:
       AbstractPairedExplicitRKSplitSingle{3}
    num_stages::Int # S

    a_matrix::Matrix{Float64}
    a_matrix_para::Matrix{Float64}
    c::Vector{Float64}

    dt_opt::Union{Float64, Nothing}
end

# Constructor for previously computed A Coeffs
function PairedExplicitRK3Split(num_stages,
                                base_path_a_coeffs::AbstractString,
                                base_path_a_coeffs_para::AbstractString,
                                dt_opt = nothing;
                                cS2 = 1.0f0)
    @assert num_stages>=3 "PERK3 requires at least three stages"
    a_matrix, c = compute_PairedExplicitRK3_butcher_tableau(num_stages,
                                                            base_path_a_coeffs;
                                                            cS2)

    a_matrix_para, _ = compute_PairedExplicitRK3_butcher_tableau(num_stages,
                                                                 base_path_a_coeffs_para;
                                                                 cS2)

    return PairedExplicitRK3Split(num_stages, a_matrix, a_matrix_para, c, dt_opt)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct PairedExplicitRK3SplitIntegrator{RealT <: Real, uType,
                                                Params, Sol, F,
                                                PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKSplitSingleIntegrator{3}
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    tdir::RealT # DIRection of time integration, i.e., if one marches forward or backward in time
    dt::RealT # current time step
    dtcache::RealT # manually set time step
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F # `rhs!` of the semidiscretization
    alg::PairedExplicitRK3Split
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # Additional PERK3 registers
    k1::uType
    kS1::uType
    # For split (hyperbolic-parabolic) problems
    du_para::uType
    k1_para::uType
end

function init(ode::ODEProblem, alg::PairedExplicitRK3Split;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    # Additional PERK3 registers
    k1 = zero(u0)
    kS1 = zero(u0)

    # For split (hyperbolic-parabolic) problems
    du_para = zero(u0)
    k1_para = zero(u0)

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    integrator = PairedExplicitRK3SplitIntegrator(u0, du, u_tmp,
                                                  t0, tdir, dt, zero(dt),
                                                  iter, ode.p,
                                                  (prob = ode,), ode.f,
                                                  alg,
                                                  PairedExplicitRKOptions(callback,
                                                                          ode.tspan;
                                                                          kwargs...),
                                                  false, true, false,
                                                  k1, kS1,
                                                  du_para, k1_para)

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

function step!(integrator::AbstractPairedExplicitRKSplitIntegrator{3})
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

        for stage in 3:(alg.num_stages - 1)
            PERK_ki!(integrator, prob.p, alg, stage)
        end

        # We need to store `du` of the S-1 stage in `kS1` for the final update:
        @threaded for i in eachindex(integrator.u)
            integrator.kS1[i] = integrator.du[i] + integrator.du_para[i] # Faster than broadcasted version (with .=)
        end

        PERK_ki!(integrator, prob.p, alg, alg.num_stages)

        @threaded for i in eachindex(integrator.u)
            # "Own" PairedExplicitRK based on SSPRK33.
            # Note that 'kS1' carries the values of K_{S-1}
            # and that we construct 'K_S' "in-place" from 'integrator.du'
            #=
            integrator.u[i] += integrator.dt *
                               (integrator.k1[i] + integrator.kS1[i] +
                                4 * integrator.du[i]) / 6
            =#
            # Try to optimize for `@muladd`: avoid `+=`
            integrator.u[i] = integrator.u[i] +
                              integrator.dt *
                              (integrator.k1[i] + integrator.k1_para[i] +
                               integrator.kS1[i] +
                               4 * (integrator.du[i] + integrator.du_para[i])) / 6
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
end

function Base.resize!(integrator::AbstractPairedExplicitRKSplitSingleIntegrator{3},
                      new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.kS1, new_size)

    resize!(integrator.du_para, new_size)
    resize!(integrator.k1_para, new_size)
end
end # @muladd
