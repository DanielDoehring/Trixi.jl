# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

using NonlinearSolve

# Advanced packages
#using SparseConnectivityTracer
#using LinearSolve # for KrylovJL_GMRES

abstract type AbstractLobattoRKIMEXAlgorithm <: AbstractTimeIntegrationAlgorithm end

"""
    LobattoIIIA_p2_Heun()

Two-stage, second-order Implicit-Explicit (IMEX) Runge-Kutta method.
Composed of the Lobatto IIIA (diagonally-)implicit Runge-Kutta method and Heun's method.

# TODO: Link to IMEX papers by Ascher
"""
struct LobattoIIIA_p2_Heun <: AbstractLobattoRKIMEXAlgorithm
    # Reduced matrices: Do not store first row full of zeros
    A_im::SMatrix{1, 2, Float64} # Implicit (Lobatto IIIA) part
    A_ex::SMatrix{1, 2, Float64} # Explicit (Heun) part
    b::SVector{2, Float64}
    c::SVector{2, Float64}

    function LobattoIIIA_p2_Heun()
        A_im = SMatrix{1, 2}(0.5, 0.5)
        A_ex = SMatrix{1, 2}(1.0, 0.0)
        b = SVector(0.5, 0.5)
        c = SVector(0, 1)

        new(A_im, A_ex, b, c)
    end
end

abstract type AbstractLobatto3AIMEXTimeIntegrator <: AbstractTimeIntegrator end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct LobattoIII3Ap2HeunIntegrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                            SimpleIntegratorOptions} <:
               AbstractLobatto3AIMEXTimeIntegrator
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi.jl
    sol::Sol # faked
    f::F # `rhs!` of the semidiscretization, expected to be a SplitFunction
    alg::Alg # AbstractLobattoRKAlgorithm
    opts::SimpleIntegratorOptions
    finalstep::Bool # added for convenience

    # For nonlinear solve
    k_nonlinear::uType

    # TODO: Try using NonlinearFunction as integrator field with iip & specialize, see 
    # https://docs.sciml.ai/NonlinearSolve/stable/basics/nonlinear_functions/#SciMLBase.NonlinearFunction
    #
    # then, one can also supply sparsity structure and coloring vector of the Jacobian
    # or try to get the sparsity detector from "SparseConnectivityTracer.jl" to work

    # For split problems
    du_tmp::uType # Additional storage for the split-part of the rhs! function
end

function init(ode::ODEProblem, alg::LobattoIIIA_p2_Heun;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u = copy(ode.u0)
    du = zero(u)
    u_tmp = zero(u)
    du_tmp = zero(u)

    k_nonlinear = zero(u)

    t = first(ode.tspan)
    iter = 0

    integrator = LobattoIII3Ap2HeunIntegrator(u, du, u_tmp,
                                              t, dt, zero(dt), iter,
                                              ode.p, (prob = ode,), ode.f, alg,
                                              SimpleIntegratorOptions(callback,
                                                                      ode.tspan;
                                                                      kwargs...), false,
                                              k_nonlinear, du_tmp)

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

function stage_residual_imex!(residual, implicit_stage, p)
    @unpack alg, dt, t, u_tmp, u, du_tmp, semi, f1 = p

    a_dt = alg.A_im[1, 2] * dt
    @threaded for i in eachindex(u_tmp)
        u_tmp[i] = u[i] + a_dt * implicit_stage[i]
    end
    f1(du_tmp, u_tmp, semi, t + alg.c[2] * dt)

    @threaded for i in eachindex(residual)
        residual[i] = implicit_stage[i] - du_tmp[i]
    end

    return nothing
end

function step!(integrator::LobattoIII3Ap2HeunIntegrator)
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

    @trixi_timeit timer() "LobattoIII3Ap2HeunIntegrator ODE integration step" begin
        # First stage for Lobatto type IIIA methods:
        integrator.f.f1(integrator.du, integrator.u, prob.p, integrator.t) # Parabolic part
        integrator.f.f2(integrator.du_tmp, integrator.u, prob.p, integrator.t) # Hyperbolic part

        b_dt = alg.b[1] * integrator.dt
        @threaded for i in eachindex(integrator.du)
            # Build overall update for first stage
            integrator.du[i] = integrator.du[i] + integrator.du_tmp[i]
            # Update u and start building intermediate solution (utilize that A_im[1, 2] = b[1])
            integrator.u[i] = integrator.u[i] + b_dt * integrator.du[i]
            integrator.k_nonlinear[i] = integrator.u[i] # Set initial guess for nonlinear solve
        end

        # Hyperbolic part is done explicitly

        # Build intermediate stage for Heun's method
        #a_dt = (alg.A_ex[1, 1] - alg.b[1]) * integrator.dt
        a_dt = 0.5 * integrator.dt
        @threaded for i in eachindex(integrator.u_tmp)
            integrator.u_tmp[i] = integrator.u[i] + a_dt * integrator.du[i]
        end
        # Compute the explicit part of the second stage
        integrator.f.f2(integrator.du, integrator.u_tmp, prob.p,
                        integrator.t + alg.c[2] * integrator.dt)

        # Second stage: Split into implicit and explicit solves
        @trixi_timeit timer() "nonlinear solve" begin
            p = (alg = alg, dt = integrator.dt, t = integrator.t,
                 u_tmp = integrator.u_tmp, u = integrator.u, du_tmp = integrator.du_tmp,
                 semi = prob.p, f1 = integrator.f.f1)

            nonlinear_eq = NonlinearProblem{true}(stage_residual_imex!,
                                                  integrator.k_nonlinear, p)

            SciMLBase.solve(nonlinear_eq,
                            NewtonRaphson(autodiff = AutoFiniteDiff()),
                            #NewtonRaphson(autodiff = AutoFiniteDiff(), linsolve = KrylovJL_GMRES()),
                            alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = true))
        end

        # Do overall update
        @threaded for i in eachindex(integrator.du)
            integrator.u[i] = integrator.u[i] +
                              b_dt * (integrator.k_nonlinear[i] + integrator.du[i])
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
get_tmp_cache(integrator::LobattoIII3Ap2HeunIntegrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::LobattoIII3Ap2HeunIntegrator, ::Bool) = false

# stop the time integration
function terminate!(integrator::LobattoIII3Ap2HeunIntegrator)
    integrator.finalstep = true
    empty!(integrator.opts.tstops)
end

# used for AMR
function Base.resize!(integrator::LobattoIII3Ap2HeunIntegrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    # For nonlinear solve
    resize!(integrator.k_nonlinear, new_size)
    # For split problems
    resize!(integrator.du_tmp, new_size)
end
end # @muladd
