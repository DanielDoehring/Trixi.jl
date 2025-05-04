# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

abstract type SubDiagonalAlgorithm end
abstract type SubDiagonalRelaxationAlgorithm end

"""
    RK33()

Relaxation version of Ralston's third-order Runge-Kutta method.
This method has minimum local error bound among the S=p=3 methods.
"""
struct RK33 <: SubDiagonalAlgorithm
    b::SVector{3, Float64}
    c::SVector{3, Float64}
end
function RK33()
    b = SVector(2 / 9, 1 / 3, 4 / 9)
    c = SVector(0.0, 0.5, 0.75)

    return RK33(b, c)
end

struct RelaxationRK33{RelaxationSolver} <: SubDiagonalRelaxationAlgorithm
    sub_diag_alg::RK33
    relaxation_solver::RelaxationSolver
end
function RelaxationRK33(; relaxation_solver = RelaxationSolverNewton())
    return RelaxationRK33{typeof(relaxation_solver)}(RK33(), relaxation_solver)
end

"""
    RK44()

Relaxation version of the classical fourth-order Runge-Kutta method.
"""
struct RK44 <: SubDiagonalAlgorithm
    b::SVector{4, Float64}
    c::SVector{4, Float64}
end
function RK44()
    b = SVector(1 / 6, 1 / 3, 1 / 3, 1 / 6)
    c = SVector(0.0, 0.5, 0.5, 1.0)

    return RK44(b, c)
end

struct RelaxationRK44{RelaxationSolver} <: SubDiagonalRelaxationAlgorithm
    sub_diag_alg::RK44
    relaxation_solver::RelaxationSolver
end
function RelaxationRK44(; relaxation_solver = RelaxationSolverNewton())
    return RelaxationRK44{typeof(relaxation_solver)}(RK44(), relaxation_solver)
end

"""
    TS64()

Relaxation version of the Low-Dissipation, Low-Dispersion Runge-Kutta method by
Tselios and Simos (2005).
DOI: 10.1016/j.cam.2004.06.012
"""
struct TS64 <: SubDiagonalAlgorithm
    b::SVector{6, Float64}
    c::SVector{6, Float64}
end

function TS64()
    b = SVector(-3.94810815871644627868730966001274,
                6.15933360719925137209615595259797,
                -8.74466100703228369513719502355456,
                4.07387757397683429863757134989527,
                0.0,
                3.45955798457264430309077738107406)
    c = SVector(0.0,
                0.14656005951358278141218736059705,
                0.27191031708348360233615451628133,
                0.06936819398523233741339353210366,
                0.25897940086636139111948386831759,
                0.48921096998463659243576995327396)

    return TS64(b, c)
end

struct RelaxationTS64{RelaxationSolver} <: SubDiagonalRelaxationAlgorithm
    sub_diag_alg::TS64
    relaxation_solver::RelaxationSolver
end
function RelaxationTS64(; relaxation_solver = RelaxationSolverNewton())
    return RelaxationTS64{typeof(relaxation_solver)}(TS64(), relaxation_solver)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct SubDiagIntegrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                 SimpleIntegrator2NOptions} <: AbstractTimeIntegrator
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
end

mutable struct SubDiagRelaxationIntegrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                           SimpleIntegrator2NOptions,
                                           RelaxationSolver} <: AbstractTimeIntegrator
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
    # Entropy Relaxation additions
    gamma::RealT
    relaxation_solver::RelaxationSolver
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::Union{SubDiagIntegrator,
                                            SubDiagRelaxationIntegrator}, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

function init(ode::ODEProblem, alg::SubDiagonalAlgorithm;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u = copy(ode.u0)
    du = zero(u)
    u_tmp = zero(u)
    direction = zero(u)

    t = first(ode.tspan)
    iter = 0

    integrator = SubDiagIntegrator(u, du, u_tmp, t, dt, zero(dt), iter, ode.p,
                                   (prob = ode,), ode.f, alg,
                                   SimpleIntegrator2NOptions(callback, ode.tspan;
                                                             kwargs...), false,
                                   direction)

    # initialize callbacks
    if callback isa CallbackSet
        foreach(callback.continuous_callbacks) do cb
            throw(ArgumentError("Continuous callbacks are unsupported with sub-diagonal time integration methods."))
        end
        foreach(callback.discrete_callbacks) do cb
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    end

    return integrator
end

function init(ode::ODEProblem, alg::SubDiagonalRelaxationAlgorithm;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u = copy(ode.u0)
    du = zero(u)
    u_tmp = zero(u)
    direction = zero(u)

    t = first(ode.tspan)
    iter = 0

    # For entropy relaxation
    gamma = one(eltype(u))

    integrator = SubDiagRelaxationIntegrator(u, du, u_tmp, t, dt, zero(dt), iter, ode.p,
                                             (prob = ode,), ode.f, alg.sub_diag_alg,
                                             SimpleIntegrator2NOptions(callback,
                                                                       ode.tspan;
                                                                       kwargs...),
                                             false,
                                             direction,
                                             gamma, alg.relaxation_solver)

    # initialize callbacks
    if callback isa CallbackSet
        foreach(callback.continuous_callbacks) do cb
            throw(ArgumentError("Continuous callbacks are unsupported with sub-diagonal time integration methods."))
        end
        foreach(callback.discrete_callbacks) do cb
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    end

    return integrator
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem,
               alg::Union{SubDiagonalAlgorithm, SubDiagonalRelaxationAlgorithm};
               dt, callback = nothing, kwargs...)
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    solve!(integrator)
end

function solve!(integrator)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
        step!(integrator)
    end # "main loop" timer

    finalize_callbacks(integrator)

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

function step!(integrator::SubDiagIntegrator)
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

    @trixi_timeit timer() "Sub-diagonal RK integration step" begin
        # First stage
        integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
        @threaded for i in eachindex(integrator.u)
            integrator.direction[i] = alg.b[1] * integrator.du[i] * integrator.dt
        end

        # Second to last stage
        for stage in 2:length(alg.c)
            @threaded for i in eachindex(integrator.u)
                integrator.u_tmp[i] = integrator.u[i] +
                                      alg.c[stage] * integrator.dt * integrator.du[i]
            end
            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                         integrator.t + alg.c[stage] * integrator.dt)
            @threaded for i in eachindex(integrator.u)
                integrator.direction[i] += alg.b[stage] * integrator.du[i] *
                                           integrator.dt
            end
        end

        # Update solution
        @threaded for i in eachindex(integrator.u)
            integrator.u[i] += integrator.direction[i]
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

function step!(integrator::SubDiagRelaxationIntegrator)
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

    @trixi_timeit timer() "Relaxation sub-diagonal RK integration step" begin
        mesh, equations, dg, cache = mesh_equations_solver_cache(prob.p)

        u_wrap = wrap_array(integrator.u, prob.p)
        u_tmp_wrap = wrap_array(integrator.u_tmp, prob.p)
        S_old = integrate(entropy_math, u_wrap, mesh, equations, dg, cache)

        # First stage
        integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
        @threaded for i in eachindex(integrator.u)
            integrator.direction[i] = alg.b[1] * integrator.du[i] * integrator.dt
        end

        du_wrap = wrap_array(integrator.du, prob.p)
        # Entropy change due to first stage
        dS = alg.b[1] * integrator.dt *
             int_w_dot_stage(du_wrap, u_wrap, mesh, equations, dg, cache)

        # Second to last stage
        for stage in 2:length(alg.c)
            @threaded for i in eachindex(integrator.u)
                integrator.u_tmp[i] = integrator.u[i] +
                                      alg.c[stage] * integrator.dt * integrator.du[i]
            end
            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                         integrator.t + alg.c[stage] * integrator.dt)
            @threaded for i in eachindex(integrator.u)
                integrator.direction[i] += alg.b[stage] * integrator.du[i] *
                                           integrator.dt
            end

            dS += alg.b[stage] * integrator.dt *
                  int_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)
        end

        direction_wrap = wrap_array(integrator.direction, prob.p)

        @trixi_timeit timer() "Relaxation solver" relaxation_solver!(integrator,
                                                                     u_tmp_wrap, u_wrap,
                                                                     direction_wrap,
                                                                     S_old, dS,
                                                                     mesh, equations,
                                                                     dg, cache,
                                                                     integrator.relaxation_solver)

        integrator.iter += 1
        update_t_relaxation!(integrator)

        # Do relaxed update
        @threaded for i in eachindex(integrator.u)
            integrator.u[i] += integrator.gamma * integrator.direction[i]
        end
    end

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
get_du(integrator) = integrator.du
get_tmp_cache(integrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator, dt)
    integrator.dt = dt
end

# Required e.g. for `glm_speed_callback` 
function get_proposed_dt(integrator)
    return integrator.dt
end

# stop the time integration
function terminate!(integrator)
    integrator.finalstep = true
    empty!(integrator.opts.tstops)
end

# used for AMR
function Base.resize!(integrator::SubDiagIntegrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    resize!(integrator.direction, new_size)
end
function Base.resize!(integrator::SubDiagRelaxationIntegrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    resize!(integrator.direction, new_size)

    resize!(integrator.direction, new_size)
end
end # @muladd
