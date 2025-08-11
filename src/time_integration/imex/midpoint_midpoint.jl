# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    IMEX_Midpoint_Midpoint()

Two-stage, second-order Implicit-Explicit (IMEX) Runge-Kutta method.
Composed of the implicit and explicit midpoint rules.
The implicit method is A-stable (belongs to family of Gauss-Legendre methods), but neither B-stable or L-stable.

For more details, see
- Uri M. Ascher, Steven J. Ruuth, Raymond J. Spiteri(1997)
  Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations.
  [DOI: 10.1016/S0168-9274(97)00056-1](https://doi.org/10.1016/S0168-9274(97)00056-1)
"""
struct IMEX_Midpoint_Midpoint <: AbstractIMEXAlgorithm
    # Reduced matrices: Do not store first row full of zeros
    A_im::SMatrix{1, 2, Float64} # Implicit midpoint part
    A_ex::SMatrix{1, 2, Float64} # Explicit midpoint part
    b::SVector{2, Float64}
    c::SVector{2, Float64}

    function IMEX_Midpoint_Midpoint()
        A_im = SMatrix{1, 2}(0, 0.5)
        A_ex = SMatrix{1, 2}(0.5, 0)
        b = SVector(0, 1.0)
        c = SVector(0, 0.5)

        new(A_im, A_ex, b, c)
    end
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct MidpointMidpointIntegrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                          SimpleIntegratorOptions,
                                          NonlinCache} <:
               AbstractIMEXTimeIntegrator
    u::uType
    du::uType # In-place output of `f`
    u_tmp::uType # Used for building the argument to `f`
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
    k_nonlin::uType

    # TODO: Try using NonlinearFunction as integrator field with iip & specialize, see 
    # https://docs.sciml.ai/NonlinearSolve/stable/basics/nonlinear_functions/#SciMLBase.NonlinearFunction
    #
    # then, one can also supply sparsity structure and coloring vector of the Jacobian
    # or try to get the sparsity detector from "SparseConnectivityTracer.jl" to work

    # For split problems solved with IMEX methods
    du_para::uType # Stores the parabolic part of the overall rhs!
    u_nonlin::uType # Stores the intermediate u approximation in nonlinear solver

    nonlin_cache::NonlinCache
end

function init(ode::ODEProblem, alg::IMEX_Midpoint_Midpoint;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u = copy(ode.u0)
    du = zero(u)
    u_tmp = zero(u)
    du_para = zero(u)

    t = first(ode.tspan)
    iter = 0

    k_nonlin = zero(u)
    u_nonlin = zero(u)

    # This creates references to the parameters
    p = (alg = alg, dt = dt, t = t,
         u_tmp = u_tmp, u_nonlin = u_nonlin,
         du_para = du_para,
         semi = ode.p, f1 = ode.f.f1)

    # Retrieve jac_prototype and colorvec from kwargs, fallback to nothing
    jac_prototype = get(kwargs, :jac_prototype, nothing)
    colorvec = get(kwargs, :colorvec, nothing)

    specialize = SciMLBase.FullSpecialize
    nonlin_func = NonlinearFunction{true, specialize}(stage_residual_midpoint!;
                                                      jac_prototype = jac_prototype,
                                                      colorvec = colorvec)

    nonlin_prob = NonlinearProblem(nonlin_func, k_nonlin, p)

    nonlin_solver = get(kwargs, :nonlin_solver, NewtonRaphson(autodiff = AutoFiniteDiff()))

    abstol = get(kwargs, :abstol, nothing)
    reltol = get(kwargs, :reltol, nothing)

    nonlin_cache = SciMLBase.init(nonlin_prob, nonlin_solver;
                                  alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = true),
                                  abstol = abstol, reltol = reltol)
                                  #show_trace = Val(true), trace_level = TraceAll())

    integrator = MidpointMidpointIntegrator(u, du, u_tmp,
                                            t, dt, zero(dt), iter,
                                            ode.p, (prob = ode,), ode.f, alg,
                                            SimpleIntegratorOptions(callback,
                                                                    ode.tspan;
                                                                    kwargs...), false,
                                            k_nonlin, du_para, u_nonlin,
                                            nonlin_cache)

    initialize_callbacks!(callback, integrator)

    return integrator
end

function stage_residual_midpoint!(residual, implicit_stage, p)
    @unpack alg, dt, t, u_tmp, u_nonlin, du_para, semi, f1 = p

    a_dt = alg.A_im[1, 2] * dt
    @threaded for i in eachindex(u_tmp)
        # Hard-coded for IMEX midpoint method
        u_nonlin[i] = u_tmp[i] + a_dt * implicit_stage[i]
    end
    f1(du_para, u_nonlin, semi, t + alg.c[2] * dt)

    @threaded for i in eachindex(residual)
        residual[i] = implicit_stage[i] - du_para[i]
    end

    return nothing
end

function step!(integrator::MidpointMidpointIntegrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    limit_dt!(integrator, t_end)

    @trixi_timeit timer() "MidpointMidpointIntegrator ODE integration step" begin
        ### First stage ###
        # f1(u) not needed, can skip computation
        integrator.f.f2(integrator.du, integrator.u, prob.p, integrator.t) # Hyperbolic part

        ### Second stage: Split into implicit and explicit solves ###

        # Build intermediate stage for implicit step: Explicit contributions
        a_dt = alg.A_ex[1, 1] * integrator.dt
        @threaded for i in eachindex(integrator.u_tmp)
            integrator.u_tmp[i] = integrator.u[i] + a_dt * integrator.du[i]
        end

        # Set initial guess for nonlinear solve
        @threaded for i in eachindex(integrator.u)
            integrator.k_nonlin[i] = integrator.u[i]
        end

        @trixi_timeit timer() "nonlinear solve" begin
            
            SciMLBase.reinit!(integrator.nonlin_cache, integrator.k_nonlin;
                              # Does not seem to have an effect
                              alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = true))
            
            #println("inplace: ", SciMLBase.isinplace(integrator.nonlin_cache)) # true
            #println("atol: ", NonlinearSolveBase.get_abstol(integrator.nonlin_cache))
            #println("rtol: ", NonlinearSolveBase.get_reltol(integrator.nonlin_cache))

            # These seem unfortunately not to work
            #SciMLBase.set_u!(integrator.nonlin_cache, integrator.k_nonlin)
            #SciMLBase.set_u!(integrator.nonlin_cache, integrator.u)
            
            # TODO: At some point use Polyester for copying data
            #sol = SciMLBase.solve!(integrator.nonlin_cache)
            #copyto!(integrator.k_nonlin, sol.u)

            SciMLBase.solve!(integrator.nonlin_cache)
            copyto!(integrator.k_nonlin, NonlinearSolveBase.get_u(integrator.nonlin_cache))
        end

        # Compute the intermediate approximation for the second explicit step: Take the implicit solution into account
        a_dt = alg.A_im[1, 2] * integrator.dt
        @threaded for i in eachindex(integrator.u_tmp)
            integrator.u_tmp[i] = integrator.u_tmp[i] + a_dt * integrator.k_nonlin[i]
        end

        # Compute the explicit part of the second stage
        integrator.f.f2(integrator.du, integrator.u_tmp, prob.p,
                        integrator.t + alg.c[2] * integrator.dt)

        ### Final update ###
        b_dt = alg.b[2] * integrator.dt
        @threaded for i in eachindex(integrator.du)
            integrator.u[i] = integrator.u[i] +
                              b_dt * (integrator.k_nonlin[i] + integrator.du[i])
        end
    end

    integrator.iter += 1
    integrator.t += integrator.dt

    @trixi_timeit timer() "Step-Callbacks" handle_callbacks!(callbacks, integrator)

    check_max_iter!(integrator)

    return nothing
end

# get a cache where the RHS can be stored
get_tmp_cache(integrator::MidpointMidpointIntegrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::MidpointMidpointIntegrator, ::Bool) = false

# stop the time integration
function terminate!(integrator::MidpointMidpointIntegrator)
    integrator.finalstep = true
    empty!(integrator.opts.tstops)

    return nothing
end

# used for AMR
function Base.resize!(integrator::MidpointMidpointIntegrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    # For nonlinear solve
    resize!(integrator.k_nonlin, new_size)
    # For split problems
    resize!(integrator.du_para, new_size)
    resize!(integrator.u_nonlin, new_size)

    return nothing
end
end # @muladd
