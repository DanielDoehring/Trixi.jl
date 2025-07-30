# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

using NLsolve
using NonlinearSolve

# Abstract base type for time integration schemes of storage class `2N`
abstract type AbstractLobattoRKAlgorithm <: AbstractTimeIntegrationAlgorithm end

"""
    LobattoIIIA_p2()

See https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Lobatto_IIIA_methods
"""
struct LobattoIIIA_p2 <: AbstractLobattoRKAlgorithm
    A::SMatrix{1, 2, Float64}
    b::SVector{2, Float64}
    c::SVector{2, Float64}

    function LobattoIIIA_p2()
        a = SMatrix{1, 2}(0.5, 0.5)
        b = SVector(0.5, 0.5)
        c = SVector(0, 1)

        new(a, b, c)
    end
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct LobattoIII3AIntegrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                      SimpleIntegratorOptions} <: AbstractTimeIntegrator
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi.jl
    sol::Sol # faked
    f::F # `rhs!` of the semidiscretization
    alg::Alg # AbstractLobattoRKAlgorithm
    opts::SimpleIntegratorOptions
    finalstep::Bool # added for convenience
end

function init(ode::ODEProblem, alg::AbstractLobattoRKAlgorithm;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u = copy(ode.u0)
    du = zero(u)
    u_tmp = zero(u)

    t = first(ode.tspan)
    iter = 0
    integrator = LobattoIII3AIntegrator(u, du, u_tmp,
                                        t, dt, zero(dt), iter,
                                        ode.p, (prob = ode,), ode.f, alg,
                                        SimpleIntegratorOptions(callback, ode.tspan;
                                                                kwargs...), false)

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

function stage_residual!(residual, implicit_stage, integrator::LobattoIII3AIntegrator,
                         prob, alg::LobattoIIIA_p2)
    a_dt = alg.A[1, 2] * integrator.dt
    @threaded for i in eachindex(integrator.u_tmp)
        integrator.u_tmp[i] = integrator.u[i] + a_dt * implicit_stage[i]
    end
    integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t + alg.c[2] * integrator.dt)

    @threaded for i in eachindex(residual)
        residual[i] = implicit_stage[i] - integrator.du[i]
    end

    return nothing
end

#=
function stage_residual!(residual, implicit_stage, p)
    @unpack prob, alg, dt, t, u, u_tmp, du = p

    a_dt = alg.A[1, 2] * dt
    @threaded for i in eachindex(u_tmp)
        u_tmp[i] = u[i] + a_dt * implicit_stage[i]
    end
    integrator.f(du, u_tmp, prob.p, t + alg.c[2] * dt)

    @threaded for i in eachindex(residual)
        residual[i] = implicit_stage[i] - du[i]
    end

    return nothing
end
=#

function step!(integrator::LobattoIII3AIntegrator)
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

    @trixi_timeit timer() "LobattoIII3AIntegrator ODE integration step" begin
        # First without splitting of f:
        integrator.f(integrator.du, integrator.u, prob.p, integrator.t)

        b_dt = alg.b[1] * integrator.dt
        @threaded for i in eachindex(integrator.du)
            integrator.u[i] = integrator.u[i] + b_dt * integrator.du[i]
        end

        #=
        # First stage for Lobatto type IIIA methods:
        integrator.f2(integrator.du_tmp, integrator.u_ode, prob.p, integrator.t) # Later to be done by explicit methods
        integrator.f1(integrator.du, integrator.u_ode, prob.p, integrator.t) # Parabolic part

        b_dt = alg.b[1] * integrator.dt
        @threaded for i in eachindex(integrator.du)
            integrator.du[i] = integrator.du[i] + integrator.du_tmp[i]
            integrator.u_tmp[i] = integrator.u_tmp[i] + b_dt * integrator.du[i]
        end
        =#

        objective_function! = (residual, implicit_stage) -> stage_residual!(residual, implicit_stage, integrator, prob, alg)

        @trixi_timeit timer() "nlsolve + update" begin
          sol = nlsolve(objective_function!, integrator.u; inplace = true) # TODO: Advanced initialization instead of u?

          #=
          #nonlinear_eq = NonlinearProblem{true, SciMLBase.FullSpecialize}(stage_residual!, integrator.u, integrator)
          p = (prob = prob, alg = alg, dt = integrator.dt,
              t = integrator.t, u = integrator.u, du = integrator.du,
              u_tmp = integrator.u_tmp)

          nonlinear_eq = NonlinearProblem(stage_residual!, integrator.u, p)
          print(nonlinear_eq)

          sol = solve(nonlinear_eq, NewtonRaphson());
          =#

          @threaded for i in eachindex(integrator.du)
              integrator.u[i] = integrator.u[i] + b_dt * sol.zero[i]
          end
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

# get a cache where the RHS can be stored
get_tmp_cache(integrator::LobattoIII3AIntegrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::LobattoIII3AIntegrator, ::Bool) = false

# stop the time integration
function terminate!(integrator::LobattoIII3AIntegrator)
    integrator.finalstep = true
    empty!(integrator.opts.tstops)
end

# used for AMR
function Base.resize!(integrator::LobattoIII3AIntegrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    # TODO
end
end # @muladd
