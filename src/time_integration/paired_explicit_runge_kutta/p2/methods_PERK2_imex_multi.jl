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
    # k1 is evaluated on all levels
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
    num_methods::Int64 # Number of optimized PERK family members, i.e., R + 1 with the implicit scheme
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
#=
mutable struct PairedExplicitRK2IMEXMultiIntegrator{RealT <: Real, uType <: AbstractVector,
                                                    Params, Sol, F,
                                                    PairedExplicitRKOptions,
                                                    uImplicitType,
                                                    SparseMat, JacCache, LinCache} <:
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

    level_info_u::Vector{Vector{Int64}}

    coarsest_lvl::Int64
    n_levels::Int64

    # For nonlinear solve
    k_nonlin::uImplicitType
    residual::uImplicitType

    jac_sparse::SparseMat # using `SparseMatrixCSC` leads to slightly more allocations
    jac_sparse_cache::JacCache # using `JacobianCache` leads to many more allocations

    abstol::RealT
    maxiters_nonlin::Int

    linear_cache::LinCache # using `LinearCache` leads to many more allocations
end
=#

mutable struct PairedExplicitRK2IMEXMultiIntegrator{RealT <: Real,
                                                    uType <: AbstractVector,
                                                    Params, Sol, F,
                                                    PairedExplicitRKOptions,
                                                    uImplicitType,
                                                    NonlinCache} <:
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

    level_info_u::Vector{Vector{Int64}}

    coarsest_lvl::Int64
    n_levels::Int64

    # For nonlinear solve
    k_nonlin::uImplicitType
    residual::uImplicitType

    nonlin_cache::NonlinCache
end

mutable struct NonlinParams{RealT <: Real, uType <: AbstractVector,
                            Semi, F}
    t::RealT
    dt::RealT
    u::uType
    du::uType
    u_tmp::uType
    semi::Semi
    const f::F
    # Do not support AMR for this: Keep indices constant
    const element_indices::Vector{Int64}
    const interface_indices::Vector{Int64}
    const boundary_indices::Vector{Int64}
    const mortar_indices::Vector{Int64}
    const u_indices::Vector{Int64}
end

# Somehow needed for NonlinearSolve
function Base.copy(p::NonlinParams)
    NonlinParams(p.t, p.dt,
                 p.u, p.du, p.u_tmp,
                 p.semi, p.f,
                 p.element_indices,
                 p.interface_indices,
                 p.boundary_indices,
                 p.mortar_indices,
                 p.u_indices)
end

#=
mutable struct PairedExplicitRK2IMEXMultiParabolicIntegrator{RealT <: Real, uType <: AbstractVector,
                                                             Params, Sol, F,
                                                             PairedExplicitRKOptions,
                                                             uImplicitType,
                                                             SparseMat, JacCache,
                                                             LinCache} <:
               AbstractPairedExplicitRKIMEXMultiParabolicIntegrator{2}
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

    level_info_u::Vector{Vector{Int64}}
    # For efficient addition of `du` to `du_ode` in `rhs_hyperbolic_parabolic`
    level_info_u_acc::Vector{Vector{Int64}}

    coarsest_lvl::Int64
    n_levels::Int64

    # For nonlinear solve
    k_nonlin::uImplicitType
    residual::uImplicitType

    jac_sparse::SparseMat # using `SparseMatrixCSC` leads to slightly more allocations
    jac_sparse_cache::JacCache # using `JacobianCache` leads to many more allocations

    abstol::RealT
    maxiters_nonlin::Int

    linear_cache::LinCache # using `LinearCache` leads to many more allocations

    # Addition for hyperbolic-parabolic problems:
    # We need another register to temporarily store the changes due to the parabolic part only.
    # The changes due to the hyperbolic part are stored in the usual `du` register.
    du_para::uType
end
=#

mutable struct PairedExplicitRK2IMEXMultiParabolicIntegrator{RealT <: Real,
                                                             uType <: AbstractVector,
                                                             Params, Sol, F,
                                                             PairedExplicitRKOptions,
                                                             uImplicitType,
                                                             NonlinCache} <:
               AbstractPairedExplicitRKIMEXMultiParabolicIntegrator{2}
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

    level_info_u::Vector{Vector{Int64}}
    # For efficient addition of `du` to `du_ode` in `rhs_hyperbolic_parabolic`
    level_info_u_acc::Vector{Vector{Int64}}

    coarsest_lvl::Int64
    n_levels::Int64

    # For nonlinear solve
    k_nonlin::uImplicitType
    residual::uImplicitType

    nonlin_cache::NonlinCache

    # Addition for hyperbolic-parabolic problems:
    # We need another register to temporarily store the changes due to the parabolic part only.
    # The changes due to the hyperbolic part are stored in the usual `du` register.
    du_para::uType
end

mutable struct NonlinParamsParabolic{RealT <: Real, uType <: AbstractVector,
                                     Semi, F}
    t::RealT
    dt::RealT
    u::uType
    du::uType
    u_tmp::uType
    du_para::uType
    semi::Semi
    const f::F
    # Do not support AMR for this: Keep indices constant
    const element_indices::Vector{Int64}
    const interface_indices::Vector{Int64}
    const boundary_indices::Vector{Int64}
    const mortar_indices::Vector{Int64}
    const u_indices::Vector{Int64}
end

function Base.copy(p::NonlinParamsParabolic)
    return NonlinParamsParabolic(p.t, p.dt,
                                 p.u, p.du, p.u_tmp,
                                 p.du_para,
                                 p.semi, p.f,
                                 p.element_indices,
                                 p.interface_indices,
                                 p.boundary_indices,
                                 p.mortar_indices,
                                 p.u_indices)
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
    mesh = semi.mesh

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
                         n_levels, semi, alg) # Mesh-only based part.
                         #n_levels, semi, alg, u) # Mesh+solution (CFL) based part.

    for i in 1:n_levels
        println("Number Elements integrated with level $i: ",
                length(level_info_elements[i]))
    end

    # Set (initial) distribution of DG nodal values
    level_info_u = [Vector{Int64}() for _ in 1:n_levels]

    if isa(semi, SemidiscretizationHyperbolicParabolic)
        # For efficient addition of `du` to `du_ode` in `rhs_hyperbolic_parabolic`
        level_info_u_acc = [Vector{Int64}() for _ in 1:n_levels]
        partition_u!(level_info_u, level_info_u_acc,
                     level_info_elements, n_levels,
                     u, semi)

        du_para = zero(u)
        # Initialize `k_nonlin` with values from `du`.
        # Use the full system since the partitioned system occasionally crashes
        ode.f(du, u, semi, t, du_para)

        p = NonlinParamsParabolic{typeof(t), typeof(u),
                                  typeof(semi), typeof(ode.f)}(t, dt,
                                                               u, du, u_tmp, du_para,
                                                               semi, ode.f,
                                                               level_info_elements_acc[alg.num_methods],
                                                               level_info_interfaces_acc[alg.num_methods],
                                                               level_info_boundaries_acc[alg.num_methods],
                                                               level_info_mortars_acc[alg.num_methods],
                                                               level_info_u[alg.num_methods])
    else
        partition_u!(level_info_u,
                     level_info_elements, n_levels,
                     u, semi)

        # Initialize `k_nonlin` with values from `du`.
        # Use the full system since the partitioned system occasionally crashes
        ode.f(du, u, semi, t)

        p = NonlinParams{typeof(t), typeof(u),
                         typeof(semi), typeof(ode.f)}(t, dt,
                                                      u, du, u_tmp,
                                                      semi, ode.f,
                                                      level_info_elements_acc[alg.num_methods],
                                                      level_info_interfaces_acc[alg.num_methods],
                                                      level_info_boundaries_acc[alg.num_methods],
                                                      level_info_mortars_acc[alg.num_methods],
                                                      level_info_u[alg.num_methods])
    end

    # For fixed meshes/no re-partitioning: Allocate only required storage
    u_implicit = level_info_u[alg.num_methods]
    n_nonlin = length(u_implicit)
    println("\nNonlinear system size: ", n_nonlin, " x ", n_nonlin)
    k_nonlin = zeros(eltype(u), n_nonlin)
    residual = copy(k_nonlin)

    # Initialize `k_nonlin` with values from `du`
    @threaded for i in eachindex(u_implicit)
        k_nonlin[i] = du[u_implicit[i]]
    end

    # Not needed for Jacobian-free Newton-Krylov(GMRES)
    jac_prototype = get(kwargs, :jac_prototype, nothing)
    colorvec = get(kwargs, :colorvec, nothing)

    nonlin_func = NonlinearFunction{true, SciMLBase.FullSpecialize}(residual_S_PERK2IMEXMulti!;
                                                                    jac_prototype = jac_prototype,
                                                                    colorvec = colorvec)

    nonlin_prob = NonlinearProblem(nonlin_func, k_nonlin, p)

    nonlin_solver = get(kwargs, :nonlin_solver,
                        # Fallback is plain Newton-Raphson
                        NewtonRaphson(autodiff = AutoFiniteDiff(),
                                      linsolve = KrylovJL_GMRES()))

    abstol = get(kwargs, :abstol, nothing)
    reltol = get(kwargs, :reltol, nothing)
    maxiters_nonlin = get(kwargs, :maxiters_nonlin, typemax(Int64))

    nonlin_cache = SciMLBase.init(nonlin_prob, nonlin_solver;
                                  alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = true),
                                  abstol = abstol, reltol = reltol,
                                  maxiters = maxiters_nonlin)

    #=
    jac_sparse_cache = JacobianCache(k_nonlin, colorvec = colorvec,
                                     sparsity = jac_prototype)
    jac_sparse = SparseMatrixCSC{eltype(u), Int}(jac_prototype) # Set up memory for sparse Jacobian

    linear_prob = LinearSolve.LinearProblem(jac_sparse, k_nonlin)
    linear_solver = get(kwargs, :linear_solver, KLUFactorization())
    linear_cache = init(linear_prob, linear_solver)
    =#

    ### Done with setting up for handling of level-dependent integration ###
    if isa(semi, SemidiscretizationHyperbolicParabolic)
        integrator = PairedExplicitRK2IMEXMultiParabolicIntegrator(u, du, u_tmp,
                                                                   t, tdir,
                                                                   dt, zero(dt), iter,
                                                                   semi, (prob = ode,),
                                                                   ode.f, alg,
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
                                                                   level_info_u,
                                                                   level_info_u_acc,
                                                                   -1, n_levels,
                                                                   k_nonlin, residual,
                                                                   #jac_sparse,
                                                                   #jac_sparse_cache,
                                                                   #abstol,
                                                                   #maxiters_nonlin,
                                                                   #linear_cache,
                                                                   nonlin_cache,
                                                                   du_para)
    else
        integrator = PairedExplicitRK2IMEXMultiIntegrator(u, du, u_tmp,
                                                          t, tdir,
                                                          dt, zero(dt), iter,
                                                          semi, (prob = ode,),
                                                          ode.f, alg,
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
                                                          level_info_u,
                                                          -1, n_levels,
                                                          k_nonlin, residual,
                                                          #jac_sparse, jac_sparse_cache,
                                                          #abstol, maxiters_nonlin,
                                                          #linear_cache
                                                          nonlin_cache)
    end

    initialize_callbacks!(callback, integrator)

    return integrator
end

# Version for manual Newton-Raphson, hyperbolic problems
function residual_S_PERK2IMEXMulti!(residual, k_nonlin,
                                    integrator::PairedExplicitRK2IMEXMultiIntegrator)
    #a_dt = alg.a_matrices[alg.num_methods, 2, alg.num_stages - 2] * dt
    a_dt = 0.5 * integrator.dt # Hard-coded for IMEX midpoint method

    R = integrator.alg.num_methods
    u_indicies_implict = integrator.level_info_u[R]

    # Add implicit contribution
    @threaded for i in eachindex(u_indicies_implict)
        u_idx = u_indicies_implict[i] # Ensure thread safety
        integrator.u_tmp[u_idx] = integrator.u[u_idx] + a_dt * k_nonlin[i]
    end

    # Evaluate implicit stage
    integrator.f(integrator.du, integrator.u_tmp, integrator.sol.prob.p,
                 #t + alg.c[alg.num_stages] * dt,
                 integrator.t + a_dt,
                 integrator.level_info_elements_acc[R],
                 integrator.level_info_interfaces_acc[R],
                 integrator.level_info_boundaries_acc[R],
                 integrator.level_info_mortars_acc[R])

    # Compute residual
    @threaded for i in eachindex(u_indicies_implict)
        residual[i] = k_nonlin[i] - integrator.du[u_indicies_implict[i]]
    end

    return nothing
end

# Version for manual Newton-Raphson, hyperbolic-parabolic problems
function residual_S_PERK2IMEXMulti!(residual, k_nonlin,
                                    integrator::PairedExplicitRK2IMEXMultiParabolicIntegrator)
    #a_dt = alg.a_matrices[alg.num_methods, 2, alg.num_stages - 2] * dt
    a_dt = 0.5 * integrator.dt # Hard-coded for IMEX midpoint method

    R = integrator.alg.num_methods
    u_indicies_implict = integrator.level_info_u[R]

    # Add implicit contribution
    @threaded for i in eachindex(u_indicies_implict)
        u_idx = u_indicies_implict[i] # Ensure thread safety
        integrator.u_tmp[u_idx] = integrator.u[u_idx] + a_dt * k_nonlin[i]
    end

    # Evaluate implicit stage
    integrator.f(integrator.du, integrator.u_tmp, integrator.sol.prob.p,
                 #t + alg.c[alg.num_stages] * dt,
                 integrator.t + a_dt,
                 integrator.du_para,
                 integrator.level_info_elements_acc[R],
                 integrator.level_info_interfaces_acc[R],
                 integrator.level_info_boundaries_acc[R],
                 integrator.level_info_mortars_acc[R],
                 u_indicies_implict)

    # Compute residual
    @threaded for i in eachindex(u_indicies_implict)
        residual[i] = k_nonlin[i] - integrator.du[u_indicies_implict[i]]
    end

    return nothing
end

# Version for handed-off nonlinear solve, hyperbolic problems
function residual_S_PERK2IMEXMulti!(residual, k_nonlin, p::NonlinParams)
    @unpack t, dt, u, du, u_tmp, semi, f,
    element_indices, interface_indices, boundary_indices, mortar_indices, u_indices = p

    #a_dt = alg.a_matrices[alg.num_methods, 2, alg.num_stages - 2] * dt
    a_dt = 0.5 * dt # Hard-coded for IMEX midpoint method

    # Add implicit contribution
    @threaded for i in eachindex(u_indices)
        u_idx = u_indices[i] # Ensure thread safety
        u_tmp[u_idx] = u[u_idx] + a_dt * k_nonlin[i]
    end

    # Evaluate implicit stage
    f(du, u_tmp, semi, t + a_dt,
      element_indices, interface_indices, boundary_indices, mortar_indices)

    # Compute residual
    @threaded for i in eachindex(u_indices)
        residual[i] = k_nonlin[i] - du[u_indices[i]]
    end

    return nothing
end

# Version for handed-off nonlinear solve, hyperbolic-parabolic problems
function residual_S_PERK2IMEXMulti!(residual, k_nonlin, p::NonlinParamsParabolic)
    @unpack t, dt, u, du, u_tmp, du_para, semi, f,
    element_indices, interface_indices, boundary_indices, mortar_indices, u_indices = p

    #a_dt = alg.a_matrices[alg.num_methods, 2, alg.num_stages - 2] * dt
    a_dt = 0.5 * dt # Hard-coded for IMEX midpoint method

    # Add implicit contribution
    @threaded for i in eachindex(u_indices)
        u_idx = u_indices[i] # Ensure thread safety
        u_tmp[u_idx] = u[u_idx] + a_dt * k_nonlin[i]
    end

    # Evaluate implicit stage
    f(du, u_tmp, semi, t + a_dt,
      du_para,
      element_indices, interface_indices, boundary_indices, mortar_indices, u_indices)

    # Compute residual
    @threaded for i in eachindex(u_indices)
        residual[i] = k_nonlin[i] - du[u_indices[i]]
    end

    return nothing
end

function newton_frozen_jac!(integrator::AbstractPairedExplicitRKIMEXMultiIntegrator{2})
    # Compute residual
    residual_S_PERK2IMEXMulti!(integrator.residual, integrator.k_nonlin, integrator)

    n = 0
    while norm(integrator.residual, Inf) > integrator.abstol &&
        n < integrator.maxiters_nonlin
        # Update RHS
        integrator.linear_cache.b = integrator.residual
        # Newton step
        LinearSolve.solve!(integrator.linear_cache) # Solve linear system for new RHS
        # Update unknown
        @threaded for i in eachindex(integrator.k_nonlin)
            integrator.k_nonlin[i] -= integrator.linear_cache.u[i]
        end

        # Compute residual
        residual_S_PERK2IMEXMulti!(integrator.residual, integrator.k_nonlin,
                                   integrator)
    end

    return nothing
end

function step!(integrator::AbstractPairedExplicitRKIMEXMultiIntegrator)
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
        # Currently only implemented for the case `alg.num_methods == integrator.n_levels`!
        for level in 1:(alg.num_methods - 1)
            a1_dt = alg.a_matrices[level, 1, alg.num_stages - 2] * integrator.dt
            a2_dt = alg.a_matrices[level, 2, alg.num_stages - 2] * integrator.dt
            @threaded for i in integrator.level_info_u[level]
                integrator.u_tmp[i] = integrator.u[i] +
                                      a1_dt * integrator.k1[i] +
                                      a2_dt * integrator.du[i]
            end
        end

        #=
        res_S_PERK2IMEXMulti!(residual, k_nonlin) = residual_S_PERK2IMEXMulti!(residual,
                                                                               k_nonlin,
                                                                               integrator)
        @trixi_timeit timer() "Jacobian computation" finite_difference_jacobian!(integrator.jac_sparse,
                                                                                 res_S_PERK2IMEXMulti!,
                                                                                 integrator.k_nonlin,
                                                                                 integrator.jac_sparse_cache)

        integrator.linear_cache.A = integrator.jac_sparse # Update system matrix

        @trixi_timeit timer() "Frozen Newton-Raphson" newton_frozen_jac!(integrator)
        =#

        @trixi_timeit timer() "nonlinear solve" begin
            SciMLBase.reinit!(integrator.nonlin_cache, integrator.k_nonlin;
                              # Does not seem to have an effect, need to re-copy data
                              alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = true))

            # Update scalar fields, which are not automatically constructed as references
            integrator.nonlin_cache.p.t = integrator.t
            integrator.nonlin_cache.p.dt = integrator.dt

            sol = SciMLBase.solve!(integrator.nonlin_cache) # Does not allocate more than `SciMLBase.solve!(integrator.nonlin_cache)`, ...
            # ... but allows to use polyester for copying data back instead of `copyto!`
            @threaded for i in eachindex(integrator.k_nonlin)
                integrator.k_nonlin[i] = sol.u[i]
            end

            #=
            SciMLBase.solve!(integrator.nonlin_cache)
            copyto!(integrator.k_nonlin,
                    NonlinearSolveBase.get_u(integrator.nonlin_cache))
            =#
        end

        ### Final update ###
        # Joint (with explicit part) re-evaluation of implicit stage
        # => Makes conservation much simpler
        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[alg.num_stages] * integrator.dt,
                     integrator)

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
