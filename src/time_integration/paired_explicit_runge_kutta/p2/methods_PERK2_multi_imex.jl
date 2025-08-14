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
    # a_{S, 1} = 0 and a_{S, 2} = 0.5
    # These entries are, however, never queried
    a_matrices[num_methods, 1, num_stages - 2] = 0.0
    a_matrices[num_methods, 2, num_stages - 2] = 0.5

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
                                                    PairedExplicitRKOptions,
                                                    uImType, NonlinCache} <:
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
    # TODO: Store explicit indices in one datastructure for better threaded access?

    coarsest_lvl::Int64
    n_levels::Int64

    # For nonlinear solve
    u_nonlin::uType # Stores the intermediate u approximation in nonlinear solver
    k_nonlin::uImType
    nonlin_cache::NonlinCache # Contains Problem & Solver
end

# mutable struct to store references to data
mutable struct NonlinParams{RealT <: Real, uType,
                            Semi, F}
    t::RealT
    dt::RealT
    u::uType
    du::uType
    u_nonlin::uType
    semi::Semi
    const f::F
    # Do not support AMR for this: Keep indices constant
    const element_indices::Vector{Int64}
    const interface_indices::Vector{Int64}
    const boundary_indices::Vector{Int64}
    const mortar_indices::Vector{Int64}
    const u_indices::Vector{Int64}
end

function init(ode::ODEProblem, alg::PairedExplicitRK2IMEXMulti;
              dt, callback = nothing, kwargs...)
    u = copy(ode.u0)
    du = zero(u)
    u_tmp = zero(u)

    k1 = zero(u) # Additional PERK register

    t = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    ### Set datastructures for handling of level-dependent integration ###
    semi = ode.p
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

    n_levels = get_n_levels(mesh, alg)

    level_info_elements = [Vector{Int64}() for _ in 1:n_levels]
    level_info_elements_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_boundaries_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]

    partition_variables!(level_info_elements,
                         level_info_elements_acc,
                         level_info_interfaces_acc,
                         level_info_boundaries_acc,
                         level_info_mortars_acc,
                         n_levels, mesh, dg, cache, alg)

    for i in 1:n_levels
        println("Number Elements integrated with level $i: ",
                length(level_info_elements[i]))
    end

    #println("level_info_elements: ", level_info_elements)
    #println("level_info_elements_acc: ", level_info_elements_acc)
    #println("level_info_interfaces_acc: ", level_info_interfaces_acc)
    #println("level_info_boundaries_acc: ", level_info_boundaries_acc)
    #println("level_info_mortars_acc: ", level_info_mortars_acc)

    # Set (initial) distribution of DG nodal values
    level_u_indices_elements = [Vector{Int64}() for _ in 1:n_levels]
    partition_u!(level_u_indices_elements, level_info_elements,
                 n_levels, u, mesh, equations, dg, cache)

    #println("level_u_indices_elements: ", level_u_indices_elements)

    ### Nonlinear Solver ###

    # Full-size, argument to `rhs!`
    u_nonlin = zero(u)
    # For fixed meshes/no re-partitioning: Allocate only required storage
    k_nonlin = zeros(eltype(u), length(level_u_indices_elements[alg.num_methods]))

    # Initialize `k_nonlin` here with values from `u`, such that in the simulation we use the same value for init
    @threaded for i in 1:length(level_u_indices_elements[alg.num_methods])
        k_nonlin[i] = u[level_u_indices_elements[alg.num_methods][i]]
    end
    
    p = NonlinParams(t, dt,
                     u, du, u_nonlin,
                     semi, ode.f,
                     level_info_elements_acc[alg.num_methods],
                     level_info_interfaces_acc[alg.num_methods],
                     level_info_boundaries_acc[alg.num_methods],
                     level_info_mortars_acc[alg.num_methods],
                     level_u_indices_elements[alg.num_methods])
    
    # Retrieve jac_prototype and colorvec from kwargs, fallback to nothing
    jac_prototype = get(kwargs, :jac_prototype, nothing)
    colorvec = get(kwargs, :colorvec, nothing)

    specialize = SciMLBase.FullSpecialize
    nonlin_func = NonlinearFunction{true, specialize}(stage_residual_PERK2IMEXMulti!;
                                                      jac_prototype = jac_prototype,
                                                      colorvec = colorvec)

    nonlin_prob = NonlinearProblem(nonlin_func, k_nonlin, p)

    nonlin_solver = get(kwargs, :nonlin_solver,
                        # Fallback is plain Newton-Raphson
                        NewtonRaphson(autodiff = AutoFiniteDiff()))

    abstol = get(kwargs, :abstol, nothing)
    reltol = get(kwargs, :reltol, nothing)
    maxiters_nonlin = get(kwargs, :maxiters_nonlin, typemax(Int64))

    nonlin_cache = SciMLBase.init(nonlin_prob, nonlin_solver;
                                  alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = true),
                                  abstol = abstol, reltol = reltol,
                                  maxiters = maxiters_nonlin)
    #show_trace = Val(true), trace_level = TraceAll())

    ### Done with setting up for handling of level-dependent integration ###
    integrator = PairedExplicitRK2IMEXMultiIntegrator(u, du, u_tmp,
                                                      t, tdir,
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
                                                      u_nonlin, k_nonlin,
                                                      nonlin_cache)

    initialize_callbacks!(callback, integrator)

    return integrator
end

function stage_residual_PERK2IMEXMulti!(residual, implicit_stage, p)
    @unpack t, dt, u, du, u_nonlin, semi, f,
    element_indices, interface_indices, boundary_indices, mortar_indices, u_indices = p

    #a_dt = alg.a_matrices[alg.num_methods, 2, alg.num_stages - 2] * dt
    a_dt = 0.5 * dt # Hard-coded for IMEX midpoint method

    # Add implicit contribution
    @threaded for i in 1:length(u_indices)
        u_idx = u_indices[i] # Ensure thread safety
        u_nonlin[u_idx] = u[u_idx] + a_dt * implicit_stage[i]
    end

    # Evaluate implicit stage
    f(du, u_nonlin, semi,
      #t + alg.c[alg.num_stages] * dt,
      t + 0.5 * dt, # Hard-coded for IMEX midpoint method
      element_indices, interface_indices, boundary_indices, mortar_indices)

    # Compute residual
    @threaded for i in 1:length(u_indices)
        residual[i] = implicit_stage[i] - du[u_indices[i]]
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

    limit_dt!(integrator, t_end)

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

        # Copy data to `u_nonlin` before nonlinear solve
        @threaded for i in eachindex(integrator.u_tmp) # TODO: Could only use the explicit indices here
            integrator.u_nonlin[i] = integrator.u_tmp[i]
        end

        u_indices_implicit = integrator.level_u_indices_elements[alg.num_methods]

        # Set initial guess for nonlinear solve
        #=
        @threaded for i in 1:length(u_indices_implicit)
            # Trivial choices
            
            integrator.k_nonlin[i] = integrator.u[u_indices_implicit[i]]
            #integrator.k_nonlin[i] = integrator.du[u_indices_implicit[i]]
            
            # Try some extrapolation choices
            
            #integrator.k_nonlin[i] = 0.5 * integrator.du[u_indices_implicit[i]]
            #=
            integrator.k_nonlin[i] = 0.5 * (integrator.u[u_indices_implicit[i]] + 
                                            integrator.du[u_indices_implicit[i]])
            =#
        end
        =#

        @trixi_timeit timer() "nonlinear solve" begin
            SciMLBase.reinit!(integrator.nonlin_cache, integrator.k_nonlin;
                              # Does not seem to have an effect
                              alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = true))

            integrator.nonlin_cache.p.t = integrator.t
            integrator.nonlin_cache.p.dt = integrator.dt

            # These seem unfortunately not to work
            #SciMLBase.set_u!(integrator.nonlin_cache, integrator.k_nonlin)
            #SciMLBase.set_u!(integrator.nonlin_cache, integrator.u)

            # TODO: At some point use Polyester for copying data
            #sol = SciMLBase.solve!(integrator.nonlin_cache)
            #copyto!(integrator.k_nonlin, sol.u)

            SciMLBase.solve!(integrator.nonlin_cache)
            copyto!(integrator.k_nonlin,
                    NonlinearSolveBase.get_u(integrator.nonlin_cache))
        end

        # Compute the intermediate approximation for the final explicit step: Take the implicit solution into account
        a_dt = 0.5 * integrator.dt # Hard-coded for IMEX midpoint method
        @threaded for i in 1:length(u_indices_implicit)
            u_idx = u_indices_implicit[i]
            integrator.u_tmp[u_idx] = integrator.u[u_idx] +
                                      a_dt * integrator.k_nonlin[i]
        end

        ### Final update ###
        # Joint (with explicit part) re-evaluation of implicit stage
        # => Makes conservation much simpler
        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[alg.num_stages] * integrator.dt)

        b_dt = integrator.dt # Hard-coded for PERK2 IMEX midpoint method with bS = 1
        @threaded for i in eachindex(integrator.u)
            integrator.u[i] = integrator.u[i] + b_dt * integrator.du[i]
        end
    end

    integrator.iter += 1
    integrator.t += integrator.dt

    @trixi_timeit timer() "Step-Callbacks" handle_callbacks!(callbacks, integrator)

    check_max_iter!(integrator)

    return nothing
end
end # @muladd
