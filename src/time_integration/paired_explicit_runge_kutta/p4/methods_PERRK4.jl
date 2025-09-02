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
end

# Constructor for previously computed A Coeffs
function PairedExplicitRelaxationRK4(num_stages, base_path_a_coeffs::AbstractString,
                                     dt_opt = nothing;
                                     cS3 = 1.0f0,
                                     relaxation_solver = RelaxationSolverNewton())
    return PairedExplicitRelaxationRK4{typeof(relaxation_solver)}(PairedExplicitRK4(num_stages,
                                                                                    base_path_a_coeffs,
                                                                                    dt_opt;
                                                                                    cS3 = cS3),
                                                                  relaxation_solver)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct PairedExplicitRelaxationRK4Integrator{RealT <: Real, uType <: AbstractVector,
                                                     Params, Sol, F,
                                                     PairedExplicitRKOptions,
                                                     RelaxationSolver} <:
               AbstractPairedExplicitRelaxationRKSingleIntegrator{4}
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
    alg::PairedExplicitRK4
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # Additional PERK register
    k1::uType
    # Entropy Relaxation additions
    gamma::RealT # Relaxation parameter
    S_old::RealT # Entropy of previous iterate
    relaxation_solver::RelaxationSolver
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
    gamma = one(eltype(u0))
    semi = ode.p
    u_wrap = wrap_array(u0, semi)
    S_old = integrate(entropy, u_wrap, semi.mesh, semi.equations, semi.solver,
                      semi.cache)

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
                                                       gamma, S_old,
                                                       alg.relaxation_solver)

    initialize_callbacks!(callback, integrator)

    return integrator
end

# Computes last three stages, i.e., i = S-2, S-1, S
@inline function PERK4_kS2_to_kS!(integrator::Union{AbstractPairedExplicitRelaxationRKIntegrator{4},
                                                    AbstractPairedExplicitRelaxationRKMultiParabolicIntegrator{4}},
                                  p, alg)
    mesh, equations, dg, cache = mesh_equations_solver_cache(p)

    for stage in 1:2
        a1_dt = alg.a_matrix_constant[1, stage] * integrator.dt
        a2_dt = alg.a_matrix_constant[2, stage] * integrator.dt
        @threaded for i in eachindex(integrator.u)
            integrator.u_tmp[i] = integrator.u[i] +
                                  a1_dt * integrator.k1[i] + a2_dt * integrator.du[i]
        end

        integrator.f(integrator.du, integrator.u_tmp, p,
                     integrator.t +
                     alg.c[alg.num_stages - 3 + stage] * integrator.dt,
                     integrator)
    end

    du_wrap = wrap_array(integrator.du, p)
    u_tmp_wrap = wrap_array(integrator.u_tmp, p)
    # Entropy change due to S-1 stage
    b_dt = 0.5 * integrator.dt # 0.5 = b_{S-1}
    dS = b_dt *
         integrate_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

    # Last stage
    a1_dt = alg.a_matrix_constant[1, 3] * integrator.dt
    a2_dt = alg.a_matrix_constant[2, 3] * integrator.dt
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              a1_dt * integrator.k1[i] + a2_dt * integrator.du[i]
    end
    # Store K_{S-1} in `k1`
    @threaded for i in eachindex(integrator.k1)
        integrator.k1[i] = integrator.du[i] # Faster than broadcasted version (with .=)
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[alg.num_stages] * integrator.dt,
                 integrator)

    # Entropy change due to last (i = S) stage
    dS += b_dt * # 0.5 = b_{S}
          integrate_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

    # Note: We re-use `du` for the "direction"
    # Note: For efficiency, we multiply the direction with dt already here!
    @threaded for i in eachindex(integrator.u)
        integrator.du[i] = b_dt * (integrator.k1[i] + integrator.du[i])
    end

    u_wrap = wrap_array(integrator.u, integrator.p)
    @trixi_timeit timer() "Relaxation solver" relaxation_solver!(integrator,
                                                                 u_tmp_wrap, u_wrap,
                                                                 du_wrap, dS,
                                                                 mesh, equations,
                                                                 dg, cache,
                                                                 integrator.relaxation_solver)

    integrator.iter += 1
    update_t_relaxation!(integrator)

    # Do relaxed update
    @threaded for i in eachindex(integrator.u)
        # Note: We re-use `du` for the "direction"
        #integrator.u[i] += integrator.gamma * integrator.du[i]
        # Try optimize for `@muladd`: avoid `+=`
        integrator.u[i] = integrator.u[i] + integrator.gamma * integrator.du[i]
    end

    return nothing
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

    limit_dt!(integrator, t_end)

    @trixi_timeit timer() "Paired Explicit Relaxation RK ODE integration step" begin
        PERK_k1!(integrator, prob.p)
        PERK_k2!(integrator, prob.p, alg)

        # Higher stages until shared stages with constant Butcher tableau
        for stage in 3:(alg.num_stages - 3)
            PERK_ki!(integrator, prob.p, alg, stage)
        end

        PERK4_kS2_to_kS!(integrator, prob.p, alg)
    end

    @trixi_timeit timer() "Step-Callbacks" handle_callbacks!(callbacks, integrator)

    check_max_iter!(integrator)
end
end # @muladd
