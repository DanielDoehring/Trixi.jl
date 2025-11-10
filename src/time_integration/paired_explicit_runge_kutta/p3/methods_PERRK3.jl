# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct PairedExplicitRelaxationRK3{RelaxationSolver} <:
       AbstractPairedExplicitRKSingle{3}
    PERK3::PairedExplicitRK3
    relaxation_solver::RelaxationSolver
end

# Constructor for previously computed A Coeffs
function PairedExplicitRelaxationRK3(num_stages, base_path_a_coeffs::AbstractString,
                                     dt_opt = nothing;
                                     cS2 = 1.0f0,
                                     relaxation_solver = RelaxationSolverNewton())
    return PairedExplicitRelaxationRK3{typeof(relaxation_solver)}(PairedExplicitRK3(num_stages,
                                                                                    base_path_a_coeffs,
                                                                                    dt_opt;
                                                                                    cS2 = cS2),
                                                                  relaxation_solver)
end

# Constructor that computes Butcher matrix A coefficients from a semidiscretization
function PairedExplicitRelaxationRK3(num_stages, tspan,
                                     semi::AbstractSemidiscretization;
                                     verbose = false, cS2 = 1.0f0,
                                     relaxation_solver = RelaxationSolverNewton())
    return PairedExplicitRelaxationRK3{typeof(relaxation_solver)}(PairedExplicitRK3(num_stages,
                                                                                    tspan,
                                                                                    semi;
                                                                                    verbose = verbose,
                                                                                    cS2 = cS2),
                                                                  relaxation_solver)
end

# Constructor that calculates the coefficients with polynomial optimizer from a list of eigenvalues
function PairedExplicitRelaxationRK3(num_stages, tspan, eig_vals::Vector{ComplexF64};
                                     verbose = false, cS2 = 1.0f0,
                                     relaxation_solver = RelaxationSolverNewton())
    return PairedExplicitRelaxationRK3{typeof(relaxation_solver)}(PairedExplicitRK3(num_stages,
                                                                                    tspan,
                                                                                    eig_vals;
                                                                                    verbose = verbose,
                                                                                    cS2 = cS2),
                                                                  relaxation_solver)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct PairedExplicitRelaxationRK3Integrator{RealT <: Real,
                                                     uType <: AbstractVector,
                                                     Params, Sol, F,
                                                     PairedExplicitRKOptions,
                                                     RelaxationSolver} <:
               AbstractPairedExplicitRelaxationRKSingleIntegrator{3}
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
    const f::F # `rhs!` of the semidiscretization
    const alg::PairedExplicitRK3
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    const dtchangeable::Bool
    const force_stepfail::Bool
    # Additional PERK3 registers
    k1::uType
    kS1::uType
    # Entropy Relaxation additions
    gamma::RealT # Relaxation parameter
    S_old::RealT # Entropy of previous iterate
    relaxation_solver::RelaxationSolver
end

function init(ode::ODEProblem, alg::PairedExplicitRelaxationRK3;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    # Additional PERK3 registers
    k1 = zero(u0)
    kS1 = zero(u0)

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    # For entropy relaxation
    gamma = one(eltype(u0))
    semi = ode.p
    u_wrap = wrap_array(u0, semi)
    S_old = integrate(entropy, u_wrap, semi.mesh, semi.equations, semi.solver,
                      semi.cache)

    integrator = PairedExplicitRelaxationRK3Integrator(u0, du, u_tmp,
                                                       t0, tdir, dt, zero(dt),
                                                       iter, ode.p,
                                                       (prob = ode,), ode.f,
                                                       # Note that here the `PERK3` algorithm is passed on as 
                                                       # `alg` of the integrator
                                                       alg.PERK3,
                                                       PairedExplicitRKOptions(callback,
                                                                               ode.tspan;
                                                                               kwargs...),
                                                       false, true, false,
                                                       k1, kS1,
                                                       gamma, S_old,
                                                       alg.relaxation_solver)

    initialize_callbacks!(callback, integrator)

    return integrator
end

function step!(integrator::Union{AbstractPairedExplicitRelaxationRKIntegrator{3},
                                 AbstractPairedExplicitRelaxationRKMultiParabolicIntegrator{3}})
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

    mesh, equations, dg, cache = mesh_equations_solver_cache(prob.p)

    @trixi_timeit timer() "Paired Explicit Relaxation RK ODE integration step" begin
        PERK_k1!(integrator, prob.p)

        u_wrap = wrap_array(integrator.u, prob.p)
        k1_wrap = wrap_array(integrator.k1, prob.p)
        # Entropy change due to first stage
        dS = integrator.dt *
             integrate_w_dot_stage(k1_wrap, u_wrap, mesh, equations, dg, cache) / 6 # 1/6

        PERK_k2!(integrator, prob.p, alg)

        for stage in 3:(alg.num_stages - 1)
            PERK_ki!(integrator, prob.p, alg, stage)
        end

        du_wrap = wrap_array(integrator.du, prob.p)
        u_tmp_wrap = wrap_array(integrator.u_tmp, prob.p)
        # Entropy change due to S-1 stage
        dS += integrator.dt *
              integrate_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache) / 6 # b_{S-1} = 1/6

        # We need to store `du` of the S-1 stage in `kS1` for the final update:
        @threaded for i in eachindex(integrator.u)
            integrator.kS1[i] = integrator.du[i] # Faster than broadcasted version (with .=)
        end

        PERK_ki!(integrator, prob.p, alg, alg.num_stages)

        # Entropy change due to last (i = S) stage
        dS += 2 / 3 * integrator.dt * # b_{S} = 2/3
              integrate_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

        # Note: We reuse `du` for the "direction"
        @threaded for i in eachindex(integrator.u)
            integrator.du[i] = integrator.dt *
                               (integrator.k1[i] + integrator.kS1[i] +
                                4 * integrator.du[i]) / 6
        end

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
    end

    @trixi_timeit timer() "Step-Callbacks" handle_callbacks!(callbacks, integrator)

    check_max_iter!(integrator)

    return nothing
end
end # @muladd
