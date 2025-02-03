# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

abstract type vanderHouwenAlgorithm end

"""
    RCKL54()

Relaxation version of the Carpenter-Kennedy-Lewis 5-stage, 4th-order Runge-Kutta method.
"""
struct RCKL54 <: vanderHouwenAlgorithm
    a::SVector{5, Float64}
    b::SVector{5, Float64}
    c::SVector{5, Float64}

    function RCKL54()
        a = SVector(0.0,
                    970286171893 / 4311952581923,
                    6584761158862 / 12103376702013,
                    2251764453980 / 15575788980749,
                    26877169314380 / 34165994151039)

        b = SVector(1153189308089 / 22510343858157,
                    1772645290293 / 4653164025191,
                    -1672844663538 / 4480602732383,
                    2114624349019 / 3568978502595,
                    5198255086312 / 14908931495163)
        c = SVector(0.0,
                    a[2],
                    b[1] + a[3],
                    b[1] + b[2] + a[4],
                    b[1] + b[2] + b[3] + a[5])

        new(a, b, c)
    end
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct vanderHouwenIntegrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                      SimpleIntegrator2NOptions}
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi.jl
    sol::Sol # faked
    f::F # `rhs` of the semidiscretization
    alg::Alg
    opts::SimpleIntegrator2NOptions
    finalstep::Bool # added for convenience
    # Addition for Relaxation methodology/efficient implementation
    direction::uType
    k_prev::uType
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::vanderHouwenIntegrator, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

function init(ode::ODEProblem, alg::vanderHouwenAlgorithm;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u = copy(ode.u0)
    du = similar(u)
    u_tmp = copy(u)
    direction = similar(u)
    k_prev = similar(u)

    t = first(ode.tspan)
    iter = 0
    integrator = vanderHouwenIntegrator(u, du, u_tmp, t, dt, zero(dt), iter, ode.p,
                                        (prob = ode,), ode.f, alg,
                                        SimpleIntegrator2NOptions(callback, ode.tspan;
                                                                  kwargs...), false,
                                        direction, k_prev)

    # initialize callbacks
    if callback isa CallbackSet
        foreach(callback.continuous_callbacks) do cb
            throw(ArgumentError("Continuous callbacks are unsupported with the 2N storage time integration methods."))
        end
        foreach(callback.discrete_callbacks) do cb
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    end

    return integrator
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::vanderHouwenAlgorithm;
               dt, callback = nothing, kwargs...)
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    solve!(integrator)
end

function solve!(integrator::vanderHouwenIntegrator)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
        step!(integrator)
    end # "main loop" timer

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

function step!(integrator::vanderHouwenIntegrator)
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

    num_stages = length(alg.c)

    # First stage
    integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
    @threaded for i in eachindex(integrator.u)
        integrator.direction[i] = alg.b[1] * integrator.du[i]
    end
    integrator.k_prev .= integrator.du

    # Second to last stage
    for stage in 2:num_stages
        @threaded for i in eachindex(integrator.u)
            integrator.u_tmp[i] = integrator.u[i] +
                                  alg.a[stage] * integrator.dt * integrator.du[i]
        end

        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[stage] * integrator.dt)

        @threaded for i in eachindex(integrator.u)
            integrator.direction[i] = alg.b[stage] * integrator.du[i]

            integrator.u[i] += alg.b[stage - 1] * integrator.dt * integrator.k_prev[i]
        end
        integrator.k_prev .= integrator.du
    end

    # Update solution
    @threaded for i in eachindex(integrator.u)
        integrator.u[i] += integrator.dt * alg.b[num_stages] * integrator.du[i]
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

# get a cache where the RHS can be stored
get_du(integrator::vanderHouwenIntegrator) = integrator.du
get_tmp_cache(integrator::vanderHouwenIntegrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::vanderHouwenIntegrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::vanderHouwenIntegrator, dt)
    integrator.dt = dt
end

# Required e.g. for `glm_speed_callback` 
function get_proposed_dt(integrator::vanderHouwenIntegrator)
    return integrator.dt
end

# stop the time integration
function terminate!(integrator::vanderHouwenIntegrator)
    integrator.finalstep = true
    empty!(integrator.opts.tstops)
end

# used for AMR
function Base.resize!(integrator::vanderHouwenIntegrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    resize!(integrator.direction, new_size)
    resize!(integrator.k_prev, new_size)
end
end # @muladd
