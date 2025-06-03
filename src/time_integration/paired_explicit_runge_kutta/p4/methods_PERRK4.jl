# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.

@muladd begin
#! format: noindent

struct PairedExplicitRelaxationRK4{RelaxationSolver} <:
       AbstractPairedExplicitRKSingle{4}
    PERK4::PairedExplicitRK4
    relaxation_solver::RelaxationSolver
    recompute_entropy::Bool
end

# Constructor for previously computed A Coeffs
function PairedExplicitRelaxationRK4(num_stages, base_path_a_coeffs::AbstractString,
                                     dt_opt = nothing;
                                     cS3 = 1.0f0,
                                     relaxation_solver = RelaxationSolverNewton(),
                                     recompute_entropy = true)
    return PairedExplicitRelaxationRK4{typeof(relaxation_solver)}(PairedExplicitRK4(num_stages,
                                                                                    base_path_a_coeffs,
                                                                                    dt_opt;
                                                                                    cS3 = cS3),
                                                                  relaxation_solver,
                                                                  recompute_entropy)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct PairedExplicitRelaxationRK4Integrator{RealT <: Real, uType,
                                                     Params, Sol, F,
                                                     PairedExplicitRKOptions,
                                                     RelaxationSolver} <:
               AbstractPairedExplicitRelaxationRKSingleIntegrator{4}
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
    alg::PairedExplicitRK4
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

function init(ode::ODEProblem, alg::PairedExplicitRelaxationRK4;
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

    integrator = PairedExplicitRelaxationRK4Integrator(u0, du, u_tmp,
                                                       t0, tdir, dt, zero(dt),
                                                       iter, ode.p,
                                                       (prob = ode,), ode.f,
                                                       # Note that here the `PERK4` algorithm is passed on as 
                                                       # `alg` of the integrator
                                                       alg.PERK4,
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

# Computes last three stages, i.e., i = S-2, S-1, S
@inline function PERK4_kS2_to_kS!(integrator::Union{AbstractPairedExplicitRelaxationRKIntegrator{4},
                                                    AbstractPairedExplicitRelaxationRKMultiParabolicIntegrator{4}},
                                  p, alg)
    mesh, equations, dg, cache = mesh_equations_solver_cache(p)

    for stage in 1:2
        @threaded for i in eachindex(integrator.u)
            integrator.u_tmp[i] = integrator.u[i] +
                                  integrator.dt *
                                  (alg.a_matrix_constant[1, stage] *
                                   integrator.k1[i] +
                                   alg.a_matrix_constant[2, stage] *
                                   integrator.du[i])
        end

        integrator.f(integrator.du, integrator.u_tmp, p,
                     integrator.t +
                     alg.c[alg.num_stages - 3 + stage] * integrator.dt,
                     integrator)
    end

    du_wrap = wrap_array(integrator.du, p)
    u_tmp_wrap = wrap_array(integrator.u_tmp, p)
    # Entropy change due to S-1 stage
    dS = 0.5 * integrator.dt * # 0.5 = b_{S-1}
         int_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

    # Last stage
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              integrator.dt *
                              (alg.a_matrix_constant[1, 3] * integrator.k1[i] +
                               alg.a_matrix_constant[2, 3] * integrator.du[i])

        # Store K_{S-1} in `k1`                               
        integrator.k1[i] = integrator.du[i] # Faster than broadcasted version (with .=)
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[alg.num_stages] * integrator.dt,
                 integrator)

    # Entropy change due to last (i = S) stage
    dS += 0.5 * integrator.dt * # 0.5 = b_{S}
          int_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

    # Note: We re-use `du` for the "direction"
    # Note: For efficiency, we multiply the direction with dt already here!
    @threaded for i in eachindex(integrator.u)
        integrator.du[i] = 0.5 * integrator.dt * (integrator.k1[i] +
                                                  integrator.du[i])
    end

    u_wrap = wrap_array(integrator.u, integrator.p)
    if integrator.recompute_entropy
        integrator.S_old = integrate(entropy_math, u_wrap, mesh, equations, dg, cache)
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

function step!(integrator::Union{AbstractPairedExplicitRelaxationRKIntegrator{4},
                                 AbstractPairedExplicitRelaxationRKMultiParabolicIntegrator{4}})
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

    if !integrator.recompute_entropy && integrator.t == first(prob.tspan)
        u_wrap = wrap_array(integrator.u, prob.p)
        mesh, equations, dg, cache = mesh_equations_solver_cache(prob.p)
        integrator.S_old = integrate(entropy_math, u_wrap, mesh, equations, dg, cache)
    end

    @trixi_timeit timer() "Paired Explicit Relaxation RK ODE integration step" begin
        PERK_k1!(integrator, prob.p)
        PERK_k2!(integrator, prob.p, alg)

        # Higher stages until "constant" stages
        for stage in 3:(alg.num_stages - 3)
            PERK_ki!(integrator, prob.p, alg, stage)
        end

        PERK4_kS2_to_kS!(integrator, prob.p, alg)
    end

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
