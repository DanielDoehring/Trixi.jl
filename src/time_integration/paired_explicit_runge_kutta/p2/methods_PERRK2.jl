# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct PairedExplicitRelaxationRK2{RelaxationSolver} <:
       AbstractPairedExplicitRKSingle{2}
    PERK2::PairedExplicitRK2
    relaxation_solver::RelaxationSolver
    recompute_entropy::Bool
end

# Constructor that reads the coefficients from a file
function PairedExplicitRelaxationRK2(num_stages,
                                     base_path_monomial_coeffs::AbstractString,
                                     dt_opt = nothing;
                                     bS = 1.0, cS = 0.5,
                                     relaxation_solver = RelaxationSolverNewton(),
                                     recompute_entropy = true)
    return PairedExplicitRelaxationRK2{typeof(relaxation_solver)}(PairedExplicitRK2(num_stages,
                                                                                    base_path_monomial_coeffs;
                                                                                    dt_opt = dt_opt,
                                                                                    bS = bS,
                                                                                    cS = cS),
                                                                  relaxation_solver,
                                                                  recompute_entropy)
end

# Constructor that calculates the coefficients with polynomial optimizer from a
# semidiscretization
function PairedExplicitRelaxationRK2(num_stages, tspan,
                                     semi::AbstractSemidiscretization;
                                     verbose = false,
                                     bS = 1.0, cS = 0.5,
                                     relaxation_solver = RelaxationSolverNewton(),
                                     recompute_entropy = true)
    return PairedExplicitRelaxationRK2{typeof(relaxation_solver)}(PairedExplicitRK2(num_stages,
                                                                                    tspan,
                                                                                    semi;
                                                                                    verbose = verbose,
                                                                                    bS = bS,
                                                                                    cS = cS),
                                                                  relaxation_solver,
                                                                  recompute_entropy)
end

# Constructor that calculates the coefficients with polynomial optimizer from a
# list of eigenvalues
function PairedExplicitRelaxationRK2(num_stages, tspan, eig_vals::Vector{ComplexF64};
                                     verbose = false,
                                     bS = 1.0, cS = 0.5,
                                     relaxation_solver = RelaxationSolverNewton(),
                                     recompute_entropy = true)
    return PairedExplicitRelaxationRK2{typeof(relaxation_solver)}(PairedExplicitRK2(num_stages,
                                                                                    tspan,
                                                                                    eig_vals;
                                                                                    verbose = verbose,
                                                                                    bS = bS,
                                                                                    cS = cS),
                                                                  relaxation_solver,
                                                                  recompute_entropy)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRelaxationRK2Integrator{RealT <: Real, uType,
                                                     Params, Sol, F,
                                                     PairedExplicitRKOptions,
                                                     RelaxationSolver} <:
               AbstractPairedExplicitRelaxationRKSingleIntegrator{2}
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
    alg::PairedExplicitRK2
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # Additional PERK register
    k1::uType
    # Entropy Relaxation additions
    gamma::RealT
    relaxation_solver::RelaxationSolver
    recompute_entropy::Bool
    S_old::RealT # Old entropy value, either last timestep or initial value
end

function init(ode::ODEProblem, alg::PairedExplicitRelaxationRK2;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    k1 = zero(u0) # Additional PERK register

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    # For entropy relaxation
    RealT = eltype(u0)
    gamma = one(RealT)

    integrator = PairedExplicitRelaxationRK2Integrator(u0, du, u_tmp,
                                                       t0, tdir, dt, zero(dt),
                                                       iter, ode.p,
                                                       (prob = ode,), ode.f,
                                                       # Note that here the `PERK4` algorithm is passed on as 
                                                       # `alg` of the integrator
                                                       alg.PERK2,
                                                       PairedExplicitRKOptions(callback,
                                                                               ode.tspan;
                                                                               kwargs...),
                                                       false, true, false,
                                                       k1,
                                                       gamma,
                                                       alg.relaxation_solver,
                                                       alg.recompute_entropy,
                                                       floatmin(RealT))

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

function step!(integrator::Union{AbstractPairedExplicitRelaxationRKIntegrator{2},
                                 AbstractPairedExplicitRelaxationRKMultiParabolicIntegrator{2}})
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

    mesh, equations, dg, cache = mesh_equations_solver_cache(prob.p)

    if !integrator.recompute_entropy && integrator.t == first(prob.tspan)
        u_wrap = wrap_array(integrator.u, prob.p)
        integrator.S_old = integrate(entropy_math, u_wrap, mesh, equations, dg, cache)
    end

    @trixi_timeit timer() "Paired Explicit Relaxation RK ODE integration step" begin
        u_wrap = wrap_array(integrator.u, prob.p)
        if integrator.recompute_entropy
            integrator.S_old = integrate(entropy_math, u_wrap, mesh, equations, dg,
                                         cache)
        end

        PERK_k1!(integrator, prob.p)

        k1_wrap = wrap_array(integrator.k1, prob.p)
        # Entropy change due to first stage
        dS = alg.b1 * integrator.dt *
             int_w_dot_stage(k1_wrap, u_wrap, mesh, equations, dg, cache)

        PERK_k2!(integrator, prob.p, alg)

        # Higher stages
        for stage in 3:(alg.num_stages)
            PERK_ki!(integrator, prob.p, alg, stage)
        end

        du_wrap = wrap_array(integrator.du, prob.p)
        u_tmp_wrap = wrap_array(integrator.u_tmp, prob.p)
        # Entropy change due to last (i = S) stage
        dS += alg.bS * integrator.dt *
              int_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

        # Note: We reuse `du` for the "direction"
        @threaded for i in eachindex(integrator.u)
            integrator.du[i] = integrator.dt *
                               (alg.b1 * integrator.k1[i] +
                                alg.bS * integrator.du[i])
        end

        @trixi_timeit timer() "Relaxation solver" relaxation_solver!(integrator,
                                                                     u_tmp_wrap, u_wrap,
                                                                     du_wrap,
                                                                     integrator.S_old,
                                                                     dS,
                                                                     mesh, equations,
                                                                     dg, cache,
                                                                     integrator.relaxation_solver)

        integrator.iter += 1
        update_t_relaxation!(integrator)

        # Do relaxed update
        @threaded for i in eachindex(integrator.u)
            # Note: We re-use `du` for the "direction"
            integrator.u[i] += integrator.gamma * integrator.du[i]
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
end # @muladd
