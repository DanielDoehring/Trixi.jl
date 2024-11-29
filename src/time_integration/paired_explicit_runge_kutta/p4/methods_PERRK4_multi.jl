# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct PairedExplicitRelaxationRK4Multi <: AbstractPairedExplicitRKMulti
    PERK4Multi::PairedExplicitRK4Multi
    relaxation_solver::RelaxationSolver
end

function PairedExplicitRelaxationRK4Multi(stages::Vector{Int64},
                                          base_path_a_coeffs::AbstractString,
                                          dt_ratios;
                                          c_const = 1.0f0,
                                          relaxation_solver = EntropyRelaxationNewton())
    return PairedExplicitRelaxationRK4Multi(PairedExplicitRK4Multi(stages,
                                                                   base_path_a_coeffs,
                                                                   dt_ratios;
                                                                   c_const = c_const),
                                            relaxation_solver)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRelaxationRK4MultiIntegrator{RealT <: Real, uType, Params,
                                                          Sol, F,
                                                          Alg,
                                                          PairedExplicitRKOptions} <:
               AbstractPairedExplicitRelaxationRKMultiIntegrator{4}
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    tdir::Real
    dt::RealT # current time step
    dtcache::RealT # Used for euler-acoustic coupling
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F
    alg::Alg # This is our own class written above; Abbreviation for ALGorithm
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # Additional PERK register
    k1::uType

    # Variables managing level-depending integration
    level_info_elements::Vector{Vector{Int64}}
    level_info_elements_acc::Vector{Vector{Int64}}

    level_info_interfaces_acc::Vector{Vector{Int64}}
    level_info_mpi_interfaces_acc::Vector{Vector{Int64}}

    level_info_boundaries_acc::Vector{Vector{Int64}}
    level_info_boundaries_orientation_acc::Vector{Vector{Vector{Int64}}}

    level_info_mortars_acc::Vector{Vector{Int64}}
    level_info_mpi_mortars_acc::Vector{Vector{Int64}}

    level_u_indices_elements::Vector{Vector{Int64}}

    coarsest_lvl::Int64
    n_levels::Int64

    # Entropy Relaxation additions
    gamma::RealT
    relaxation_solver::RelaxationSolver
end

mutable struct PairedExplicitRelaxationRK4MultiParabolicIntegrator{RealT <: Real, uType,
                                                                   Params,
                                                                   Sol, F,
                                                                   Alg,
                                                                   PairedExplicitRKOptions} <:
               AbstractPairedExplicitRelaxationRKMultiParabolicIntegrator{4}
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    tdir::Real
    dt::RealT # current time step
    dtcache::RealT # Used for euler-acoustic coupling
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F
    alg::Alg # This is our own class written above; Abbreviation for ALGorithm
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # Additional PERK register
    k1::uType

    # Variables managing level-depending integration
    level_info_elements::Vector{Vector{Int64}}
    level_info_elements_acc::Vector{Vector{Int64}}

    level_info_interfaces_acc::Vector{Vector{Int64}}
    level_info_mpi_interfaces_acc::Vector{Vector{Int64}}

    level_info_boundaries_acc::Vector{Vector{Int64}}
    level_info_boundaries_orientation_acc::Vector{Vector{Vector{Int64}}}

    level_info_mortars_acc::Vector{Vector{Int64}}
    level_info_mpi_mortars_acc::Vector{Vector{Int64}}

    level_u_indices_elements::Vector{Vector{Int64}}

    coarsest_lvl::Int64
    n_levels::Int64

    # Entropy Relaxation additions
    gamma::RealT
    relaxation_solver::RelaxationSolver

    # Addition for hyperbolic-parabolic problems:
    # We need another register to temporarily store the changes due to the hyperbolic part only.
    # The changes due to the parabolic part are stored in the usual `du` register.
    du_tmp::uType
end

function init(ode::ODEProblem, alg::PairedExplicitRelaxationRK4Multi;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    k1 = zero(u0) # Additional PERK register

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    ### Set datastructures for handling of level-dependent integration ###
    mesh, equations, dg, cache = mesh_equations_solver_cache(ode.p)

    n_levels = get_n_levels(mesh, alg)
    n_dims = ndims(mesh) # Spatial dimension

    level_info_elements = [Vector{Int64}() for _ in 1:n_levels]
    level_info_elements_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]
    level_info_mpi_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_boundaries_acc = [Vector{Int64}() for _ in 1:n_levels]
    level_info_boundaries_orientation_acc = [[Vector{Int64}()
                                              for _ in 1:(2 * n_dims)]
                                             for _ in 1:n_levels]

    level_info_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]
    level_info_mpi_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]

    # For entropy relaxation
    gamma = one(eltype(u0))

    partitioning_variables!(level_info_elements,
                            level_info_elements_acc,
                            level_info_interfaces_acc,
                            level_info_boundaries_acc,
                            level_info_boundaries_orientation_acc,
                            level_info_mortars_acc,
                            n_levels, n_dims, mesh, dg, cache, alg)

    for i in 1:n_levels
        println("#Number Elements integrated with level $i: ",
                length(level_info_elements[i]))
    end

    # Set initial distribution of DG Base function coefficients
    level_u_indices_elements = [Vector{Int64}() for _ in 1:n_levels]
    partitioning_u!(level_u_indices_elements, n_levels, n_dims, level_info_elements, u0,
                    mesh, equations, dg, cache)

    ### Done with setting up for handling of level-dependent integration ###

    if isa(ode.p, SemidiscretizationHyperbolicParabolic)
        du_tmp = zero(u0)
        integrator = PairedExplicitRelaxationRK4MultiParabolicIntegrator(u0, du, u_tmp,
                                                                         t0, tdir,
                                                                         dt, zero(dt),
                                                                         iter,
                                                                         ode.p,
                                                                         (prob = ode,),
                                                                         ode.f,
                                                                         # Note that here the `PERK4Multi` algorithm is passed on as 
                                                                         # `alg` of the integrator
                                                                         alg.PERK4Multi,
                                                                         PairedExplicitRKOptions(callback,
                                                                                                 ode.tspan;
                                                                                                 kwargs...),
                                                                         false, true,
                                                                         false,
                                                                         k1,
                                                                         level_info_elements,
                                                                         level_info_elements_acc,
                                                                         level_info_interfaces_acc,
                                                                         level_info_mpi_interfaces_acc,
                                                                         level_info_boundaries_acc,
                                                                         level_info_boundaries_orientation_acc,
                                                                         level_info_mortars_acc,
                                                                         level_info_mpi_mortars_acc,
                                                                         level_u_indices_elements,
                                                                         -1, n_levels,
                                                                         gamma,
                                                                         alg.relaxation_solver,
                                                                         du_tmp)
    else
        integrator = PairedExplicitRelaxationRK4MultiIntegrator(u0, du, u_tmp,
                                                                t0, tdir,
                                                                dt, zero(dt),
                                                                iter,
                                                                ode.p,
                                                                (prob = ode,),
                                                                ode.f,
                                                                # Note that here the `PERK4Multi` algorithm is passed on as 
                                                                # `alg` of the integrator
                                                                alg.PERK4Multi,
                                                                PairedExplicitRKOptions(callback,
                                                                                        ode.tspan;
                                                                                        kwargs...),
                                                                false, true,
                                                                false,
                                                                k1,
                                                                level_info_elements,
                                                                level_info_elements_acc,
                                                                level_info_interfaces_acc,
                                                                level_info_mpi_interfaces_acc,
                                                                level_info_boundaries_acc,
                                                                level_info_boundaries_orientation_acc,
                                                                level_info_mortars_acc,
                                                                level_info_mpi_mortars_acc,
                                                                level_u_indices_elements,
                                                                -1, n_levels,
                                                                gamma,
                                                                alg.relaxation_solver)
    end

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

@inline function last_three_stages!(integrator::PairedExplicitRelaxationRK4MultiParabolicIntegrator,
                                    p, alg)
    mesh, equations, dg, cache = mesh_equations_solver_cache(p)

    for stage in 1:2
        @threaded for u_ind in eachindex(integrator.u)
            integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                      integrator.dt *
                                      (alg.a_matrix_constant[1, stage] *
                                       integrator.k1[u_ind] +
                                       alg.a_matrix_constant[2, stage] *
                                       integrator.du[u_ind])
        end

        integrator.f(integrator.du, integrator.u_tmp, p,
                     integrator.t +
                     alg.c[alg.num_stages - 3 + stage] * integrator.dt,
                     integrator.du_tmp)
    end

    du_wrap = wrap_array(integrator.du, p)
    u_tmp_wrap = wrap_array(integrator.u_tmp, p)
    # 0.5 = b_{S-1}
    # IDEA: Combine integration of i-1, i?
    dS = 0.5 * integrator.dt *
         int_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

    # Last stage
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              integrator.dt *
                              (alg.a_matrix_constant[1, 3] * integrator.k1[i] +
                               alg.a_matrix_constant[2, 3] * integrator.du[i])
    end

    # Safe K_{S-1} in `k1`:
    @threaded for i in eachindex(integrator.u)
        integrator.k1[i] = integrator.du[i]
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[alg.num_stages] * integrator.dt,
                 integrator.du_tmp)

    # Note: We re-use `k1` for the "direction"
    # Note: For efficiency, we multiply the direction already by dt here!
    @threaded for i in eachindex(integrator.u)
        integrator.k1[i] = 0.5 * integrator.dt * (integrator.k1[i] +
                                                  integrator.du[i])
    end

    # 0.5 = b_{S}
    dS += 0.5 * integrator.dt *
          int_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

    u_wrap = wrap_array(integrator.u, integrator.p)
    S_old = integrate(entropy_math, u_wrap, mesh, equations, dg, cache)

    # Note: We re-use `k1` for the "direction"
    k1_wrap = wrap_array(integrator.k1, p)

    @trixi_timeit timer() "Relaxation solver" relaxation_solver!(integrator,
                                                                 u_tmp_wrap, u_wrap,
                                                                 k1_wrap,
                                                                 S_old, dS,
                                                                 mesh, equations,
                                                                 dg, cache,
                                                                 integrator.relaxation_solver)

    integrator.iter += 1
    # Check if due to entropy relaxation the final step is not reached
    if integrator.finalstep == true && integrator.gamma != 1
        # If we would go beyond the final time, clip gamma at 1.0
        if integrator.gamma > 1.0
            integrator.gamma = 1.0
        else # If we are below the final time, reset finalstep flag
            integrator.finalstep = false
        end
    end
    integrator.t += integrator.gamma * integrator.dt

    # Do relaxed update
    @threaded for i in eachindex(integrator.u)
        # Note: We re-use `k1` for the "direction"
        integrator.u[i] += integrator.gamma * integrator.k1[i]
    end
end

function step!(integrator::PairedExplicitRelaxationRK4MultiParabolicIntegrator)
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

    @trixi_timeit timer() "Paired Explicit Relaxation RK ODE integration step" begin
        PERK_k1!(integrator, prob.p)
        PERK_k2!(integrator, prob.p, alg)

        # Higher stages until "constant" stages
        for stage in 3:(alg.num_stages - 3)
            PERK_ki!(integrator, prob.p, alg, stage)
        end

        last_three_stages!(integrator, prob.p, alg)
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
