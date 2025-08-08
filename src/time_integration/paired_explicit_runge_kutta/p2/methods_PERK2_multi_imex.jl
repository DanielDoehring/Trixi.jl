# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function compute_PairedExplicitRK2IMEXMulti_butcher_tableau(stages::Vector{Int64},
                                                            num_stages::Int,
                                                            base_path_mon_coeffs::AbstractString,
                                                            bS, cS)
    c = PERK2_compute_c_coeffs(num_stages, cS)
    stage_scaling_factors = bS * reverse(c[2:(end - 1)])

    # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
    num_coeffs_max = num_stages - 2

    num_methods = length(stages) + 1 # + 1 for the implicit method
    a_matrices = zeros(num_methods, 2, num_coeffs_max)
    for i in 1:num_methods
        a_matrices[i, 1, :] = c[3:end]
    end
    # For second order: implicit method chosen as implicit midpoint, thus set
    a_matrices[num_methods, 1, num_stages - 2] = 0.0 # TODO: Currently this part of `a_matrices` is never used

    # Datastructure indicating at which stage which level is evaluated
    active_levels = [Vector{Int64}() for _ in 1:num_stages]
    # k1 is evaluated at all levels
    active_levels[1] = 1:num_methods

    # Datastructure indicating at which stage which level contributes to state
    add_levels = [Vector{Int64}() for _ in 1:num_stages]
    # k1 is used/added at all levels
    add_levels[1] = 1:num_methods
    # Second stage: Only finest method
    add_levels[2] = [1]

    for level in eachindex(stages)
        num_stage_evals = stages[level]
        num_free_coeffs = num_stage_evals - 2

        if num_free_coeffs > 0
            path_monomial_coeffs = joinpath(base_path_mon_coeffs,
                                            "gamma_" * string(num_stage_evals) * ".txt")

            @assert isfile(path_monomial_coeffs) "Couldn't find file"
            monomial_coeffs = readdlm(path_monomial_coeffs, Float64)
            num_monomial_coeffs = size(monomial_coeffs, 1)

            @assert num_monomial_coeffs == num_free_coeffs
            A = compute_a_coeffs(num_stage_evals, stage_scaling_factors,
                                 monomial_coeffs)

            a_matrices[level, 1, (num_coeffs_max - (num_stage_evals - 3)):end] -= A
            a_matrices[level, 2, (num_coeffs_max - (num_stage_evals - 3)):end] = A
        end

        # Add active levels to stages
        for stage in num_stages:-1:(num_stages - num_free_coeffs)
            push!(active_levels[stage], level)
        end

        # Push contributing (added) levels to stages
        for stage in num_stages:-1:(num_stages - num_free_coeffs + 1)
            push!(add_levels[stage], level)
        end
    end

    # Implicit midpoint method is evaluated in last stage.
    # Not strictly necessary, but added for consistency here
    push!(active_levels[end], num_methods)

    # First stage of implicit part is added at every level.
    # We do not reflect this in `add_levels`, though, as the implementation of the multirate PERK schemes
    # employs loops from `1 to max_add_levels[stage]`, which would then result in adding up all 
    # (especially all coarser) methods.
    # Thus, the implicit method is not present in the `add_levels` structure which only 
    # targets the explicit part.
    # The implicit contribution `k1` is added somewhat "manually" in every intermediate stage.

    max_active_levels = maximum.(active_levels)
    max_add_levels = maximum.(add_levels)

    return a_matrices, c, max_active_levels, max_add_levels
end

struct PairedExplicitRK2IMEXMulti <:
       AbstractPairedExplicitRKIMEXMulti{2}
    num_methods::Int64 # Number of optimized PERK family members, i.e., R + 1 for the implicit scheme
    num_stages::Int64 # = maximum number of stages
    stages::Vector{Int64} # For load-balancing of MPI-parallel p4est simulations

    # Δt of the different methods divided by Δt_max
    dt_ratios::Vector{Float64}

    # Butcher tableau variables
    a_matrices::Array{Float64, 3}
    c::Vector{Float64}
    b1::Float64
    bS::Float64

    # highest active/evaluated level; per stage
    max_active_levels::Vector{Int64}
    # highest level where the last stage `du` is added in the argument of the evaluated `rhs!`; per stage
    max_add_levels::Vector{Int64}
end

function PairedExplicitRK2IMEXMulti(stages::Vector{Int64},
                                    base_path_mon_coeffs::AbstractString,
                                    dt_ratios;
                                    bS = 1.0, cS = 0.5)
    num_stages = maximum(stages)

    a_matrices, c,
    max_active_levels,
    max_add_levels = compute_PairedExplicitRK2IMEXMulti_butcher_tableau(stages,
                                                                        num_stages,
                                                                        base_path_mon_coeffs,
                                                                        bS, cS)

    return PairedExplicitRK2IMEXMulti(length(stages) + 1, num_stages, stages,
                                      dt_ratios,
                                      a_matrices, c, 1 - bS, bS,
                                      max_active_levels, max_add_levels)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRK2IMEXMultiIntegrator{RealT <: Real, uType,
                                                    Params, Sol, F,
                                                    PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKIMEXMultiIntegrator{2}
    u::uType
    du::uType # In-place output of `f`
    u_tmp::uType # Used for building the argument to `f`
    t::RealT
    tdir::RealT # DIRection of time integration, i.e., if one marches forward or backward in time
    dt::RealT # current time step
    dtcache::RealT # Used for euler-acoustic coupling
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F # `rhs!` of the semidiscretization
    alg::PairedExplicitRK2IMEXMulti
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # Additional PERK register
    k1::uType

    # Variables managing level-dependent integration
    level_info_elements::Vector{Vector{Int64}}
    level_info_elements_acc::Vector{Vector{Int64}}

    level_info_interfaces_acc::Vector{Vector{Int64}}

    level_info_boundaries_acc::Vector{Vector{Int64}}

    level_info_mortars_acc::Vector{Vector{Int64}}

    level_u_indices_elements::Vector{Vector{Int64}}
    # TODO: `level_u_indices_elements_acc` makes sense for IMEX methods

    coarsest_lvl::Int64
    n_levels::Int64

    # For nonlinear solve
    k_nonlin::uType
    u_nonlin::uType # Stores the intermediate u approximation in nonlinear solver
    # TODO: Try to store cache or Nonlinearproblem itself here, see
    # https://docs.sciml.ai/NonlinearSolve/stable/basics/diagnostics_api/
end

function init(ode::ODEProblem, alg::PairedExplicitRK2IMEXMulti;
              dt, callback = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    k1 = zero(u0) # Additional PERK register

    k_nonlin = zero(u0)
    u_nonlin = zero(u0)

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    ### Set datastructures for handling of level-dependent integration ###
    semi = ode.p
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    n_dims = ndims(mesh) # Spatial dimension

    n_levels = get_n_levels(mesh, alg)

    level_info_elements = [Vector{Int64}() for _ in 1:n_levels]
    level_info_elements_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_boundaries_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]

    partition_variables_imex!(level_info_elements,
                              level_info_elements_acc,
                              level_info_interfaces_acc,
                              level_info_boundaries_acc,
                              level_info_mortars_acc,
                              n_levels, n_dims, mesh, dg, cache, alg.dt_ratios)

    for i in 1:n_levels
        println("Number Elements integrated with level $i: ",
                length(level_info_elements[i]))
    end

    println("level_info_elements: ", level_info_elements)
    println("level_info_elements_acc: ", level_info_elements_acc)
    println("level_info_interfaces_acc: ", level_info_interfaces_acc)
    #println("level_info_boundaries_acc: ", level_info_boundaries_acc)
    #println("level_info_mortars_acc: ", level_info_mortars_acc)

    # Set (initial) distribution of DG nodal values
    level_u_indices_elements = [Vector{Int64}() for _ in 1:n_levels]
    partition_u!(level_u_indices_elements, level_info_elements,
                 n_levels, u0, mesh, equations, dg, cache)

    println("level_u_indices_elements: ", level_u_indices_elements)

    ### Done with setting up for handling of level-dependent integration ###
    integrator = PairedExplicitRK2IMEXMultiIntegrator(u0, du, u_tmp,
                                                      t0, tdir,
                                                      dt, zero(dt),
                                                      iter, semi,
                                                      (prob = ode,),
                                                      ode.f,
                                                      alg,
                                                      PairedExplicitRKOptions(callback,
                                                                              ode.tspan;
                                                                              kwargs...),
                                                      false, true, false,
                                                      k1,
                                                      level_info_elements,
                                                      level_info_elements_acc,
                                                      level_info_interfaces_acc,
                                                      level_info_boundaries_acc,
                                                      level_info_mortars_acc,
                                                      level_u_indices_elements,
                                                      -1, n_levels,
                                                      k_nonlin, u_nonlin)

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

function stage_residual_PERK2IMEXMulti!(residual, implicit_stage, p)
    @unpack alg, dt, t, u_tmp, u_nonlin, u, du, semi, f,
    element_indices, interface_indices, boundary_indices, mortar_indices, u_indices_level = p

    # Set explicit contributions
    for level in 1:(alg.num_methods - 1)
        @threaded for i in u_indices_level[level]
            u_nonlin[i] = u_tmp[i]
        end
    end
    # Add implicit contribution
    a_dt = 0.5 * dt # Hard-coded for IMEX midpoint method
    @threaded for i in u_indices_level[alg.num_methods]
        u_nonlin[i] = u[i] + a_dt * implicit_stage[i]
    end
    f(du, u_nonlin, semi, t + alg.c[alg.num_stages] * dt,
      element_indices, interface_indices, boundary_indices, mortar_indices)

    @threaded for i in u_indices_level[alg.num_methods]
        residual[i] = implicit_stage[i] - du[i]
    end

    return nothing
end

function step!(integrator::PairedExplicitRK2IMEXMultiIntegrator) # TODO: Maybe generalize the integrator
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

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
        # First and second stage are identical across all single/standalone PERK methods
        PERK_k1!(integrator, prob.p)
        PERK_k2!(integrator, prob.p, alg)

        # Higher stages
        for stage in 3:(alg.num_stages - 1) # Stop at S - 1, last stage demands IMEX treatment
            PERK_ki!(integrator, prob.p, alg, stage)
        end

        # Build intermediate stage for implicit step: Explicit contributions
        # CARE: Currently only implemented for the case `alg.num_methods == integrator.n_levels`!
        for level in 1:(alg.num_methods - 1)
            a1_dt = alg.a_matrices[level, 1, alg.num_stages - 2] * integrator.dt
            a2_dt = alg.a_matrices[level, 2, alg.num_stages - 2] * integrator.dt
            @threaded for i in integrator.level_u_indices_elements[level]
                integrator.u_tmp[i] = integrator.u[i] +
                                      a1_dt * integrator.k1[i] +
                                      a2_dt * integrator.du[i]
            end
        end

        # Set initial guess for nonlinear solve
        @threaded for i in integrator.level_u_indices_elements[alg.num_methods]
            #integrator.k_nonlin[i] = integrator.u[i]
            integrator.k_nonlin[i] = integrator.du[i]
        end
        
        @trixi_timeit timer() "nonlinear solve" begin
            p = (alg = alg, dt = integrator.dt, t = integrator.t,
                 u_tmp = integrator.u_tmp, u_nonlin = integrator.u_nonlin,
                 u = integrator.u, du = integrator.du,
                 semi = prob.p, f = integrator.f,
                 # PERK-Multi additions
                 element_indices = integrator.level_info_elements_acc[alg.num_methods],
                 interface_indices = integrator.level_info_interfaces_acc[alg.num_methods],
                 boundary_indices = integrator.level_info_boundaries_acc[alg.num_methods],
                 mortar_indices = integrator.level_info_mortars_acc[alg.num_methods],
                 u_indices_level = integrator.level_u_indices_elements)

            nonlinear_eq = NonlinearProblem{true}(stage_residual_PERK2IMEXMulti!,
                                                  integrator.k_nonlin, p)

            SciMLBase.solve(nonlinear_eq,
                            #NewtonRaphson(autodiff = AutoFiniteDiff()), # Does not converge
                            NewtonRaphson(autodiff = AutoFiniteDiff(), linsolve = KrylovJL_GMRES()),
                            alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = true))
        end

        # Compute the intermediate approximation for the final explicit step: Take the implicit solution into account
        a_dt = 0.5 * integrator.dt # Hard-coded for IMEX midpoint method
        @threaded for i in integrator.level_u_indices_elements[alg.num_methods]
            integrator.u_tmp[i] = integrator.u[i] + a_dt * integrator.k_nonlin[i]
        end

        ### Final update ###
        b_dt = integrator.dt # Hard-coded for PERK2 IMEX midpoint method with bS = 1
        # Joint (with explicit part) re-evaluation of implicit stage
        # => Makes conservation much simpler
        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[alg.num_stages] * integrator.dt)

        @threaded for i in eachindex(integrator.u)
            integrator.u[i] = integrator.u[i] + b_dt * integrator.du[i]
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
end # @muladd
