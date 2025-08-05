# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    PairedExplicitRK4Split(num_stages,
                           base_path_a_coeffs::AbstractString,
                           base_path_a_coeffs_para::AbstractString,
                           dt_opt = nothing;
                           cS3 = 1.0f0)
"""
struct PairedExplicitRK4Split <:
       AbstractPairedExplicitRKSingle{4}
    num_stages::Int # S

    # Optimized coefficients, i.e., flexible part of the Butcher array matrix A.
    a_matrix::Union{Matrix{Float64}, Nothing}
    a_matrix_para::Union{Matrix{Float64}, Nothing}

    # This part of the Butcher array matrix A is constant for all PERK methods, i.e., 
    # regardless of the optimized coefficients.
    #a_matrix_constant::SMatrix{2, 3, Float64}
    # NOTE: Somehow SMatrix allocates for Single PERK4, but not Multirate...
    a_matrix_constant::Matrix{Float64}

    c::Vector{Float64}

    dt_opt::Union{Float64, Nothing}
end

# Constructor for previously computed A Coeffs
function PairedExplicitRK4Split(num_stages,
                                base_path_a_coeffs::AbstractString,
                                base_path_a_coeffs_para::AbstractString,
                                dt_opt = nothing;
                                cS3 = 1.0f0)  # Default value for best internal stability
    @assert num_stages>=5 "PERK4 requires at least five stages"
    a_matrix, a_matrix_constant, c = compute_PairedExplicitRK4_butcher_tableau(num_stages,
                                                                               base_path_a_coeffs,
                                                                               cS3)

    a_matrix_para, _, _ = compute_PairedExplicitRK4_butcher_tableau(num_stages,
                                                                    base_path_a_coeffs_para,
                                                                    cS3)

    return PairedExplicitRK4Split(num_stages, a_matrix, a_matrix_para,
                                  a_matrix_constant, c, dt_opt)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct PairedExplicitRK4SplitIntegrator{RealT <: Real, uType,
                                                Params, Sol, F,
                                                PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKSplitSingleIntegrator{4}
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
    alg::PairedExplicitRK4Split
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

function init(ode::ODEProblem, alg::PairedExplicitRK4Split;
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

    integrator = PairedExplicitRK4SplitIntegrator(u0, du, u_tmp,
                                                  t0, tdir, dt, zero(dt),
                                                  iter, ode.p,
                                                  (prob = ode,), ode.f,
                                                  alg,
                                                  PairedExplicitRKOptions(callback,
                                                                          ode.tspan;
                                                                          kwargs...),
                                                  false, true, false,
                                                  k1, du_para, k1_para)

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

function step!(integrator::AbstractPairedExplicitRKSplitSingleIntegrator{4})
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
        PERK_k1!(integrator, prob.p)
        PERK_k2!(integrator, prob.p, alg)

        # Higher stages until "constant" stages
        for stage in 3:(alg.num_stages - 3)
            PERK_ki!(integrator, prob.p, alg, stage)
        end

        # Accumulate hyperbolic and parabolic contributions into `du` and `k1`
        # to reuse `PERK4_kS2_to_kS!`
        @threaded for i in eachindex(integrator.du)
            # Try to optimize for `@muladd`: avoid `+=`
            integrator.du[i] = integrator.du[i] + integrator.du_para[i]
        end
        # Access registers after another to allow for optimized memory access
        @threaded for i in eachindex(integrator.du)
            # Try to optimize for `@muladd`: avoid `+=`
            integrator.k1[i] = integrator.k1[i] + integrator.k1_para[i]
        end

        PERK4_kS2_to_kS!(integrator, prob.p, alg)
    end

    integrator.iter += 1
    integrator.t += integrator.dt

    @trixi_timeit timer() "Step-Callbacks" begin
        # handle callbacks
        if callbacks isa CallbackSet
            for cb in callbacks.discrete_callbacks
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
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
