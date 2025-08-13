# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Version with SAME number of stages for hyperbolic and parabolic part
#=
struct PairedExplicitRK2SplitMulti <:
       AbstractPairedExplicitRKSplitMulti{2}
    num_methods::Int64 # Number of optimized PERK family members, i.e., R
    num_stages::Int64 # = maximum number of stages
    stages::Vector{Int64} # For load-balancing of MPI-parallel p4est simulations

    # Δt of the different methods divided by Δt_max
    dt_ratios::Vector{Float64}

    # Butcher tableau variables
    a_matrices::Array{Float64, 3}
    a_matrices_para::Array{Float64, 3}
    c::Vector{Float64}
    b1::Float64
    bS::Float64

    # highest active/evaluated level; per stage
    max_active_levels::Vector{Int64}
    # highest level where the last stage `du` is added in the argument of the evaluated `rhs!`; per stage
    max_add_levels::Vector{Int64}
end

function PairedExplicitRK2SplitMulti(stages::Vector{Int64},
                                     base_path_mon_coeffs::AbstractString,
                                     base_path_mon_coeffs_para::AbstractString,
                                     dt_ratios;
                                     bS = 1.0, cS = 0.5)
    num_stages = maximum(stages)

    a_matrices, c,
    max_active_levels,
    max_add_levels = compute_PairedExplicitRK2Multi_butcher_tableau(stages, num_stages,
                                                                    base_path_mon_coeffs,
                                                                    bS, cS)

    a_matrices_para, _, _, _, _ = compute_PairedExplicitRK2Multi_butcher_tableau(stages,
                                                                                 num_stages,
                                                                                 base_path_mon_coeffs_para,
                                                                                 bS, cS)

    return PairedExplicitRK2SplitMulti(length(stages), num_stages,
                                       stages, dt_ratios,
                                       a_matrices, a_matrices_para,
                                       c, 1 - bS, bS,
                                       max_active_levels, max_add_levels)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRK2SplitMultiIntegrator{RealT <: Real, uType,
                                                     Params, Sol, F,
                                                     PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKSplitMultiIntegrator{2}
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
    alg::PairedExplicitRK2SplitMulti
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # Additional PERK register
    k1::uType
    # For split (hyperbolic-parabolic) problems
    du_para::uType # Stores the parabolic part of the overall rhs!
    k1_para::uType # Additional PERK register for the parabolic part

    # Variables managing level-dependent integration
    level_info_elements::Vector{Vector{Int64}}
    level_info_elements_acc::Vector{Vector{Int64}}

    level_info_interfaces_acc::Vector{Vector{Int64}}
    level_info_mpi_interfaces_acc::Vector{Vector{Int64}}

    level_info_boundaries_acc::Vector{Vector{Int64}}

    level_info_mortars_acc::Vector{Vector{Int64}}
    level_info_mpi_mortars_acc::Vector{Vector{Int64}}

    level_u_indices_elements::Vector{Vector{Int64}}

    coarsest_lvl::Int64
    n_levels::Int64
end

function init(ode::ODEProblem, alg::PairedExplicitRK2SplitMulti;
              dt, callback = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    k1 = zero(u0) # Additional PERK register
    du_para = zero(u0) # Stores the parabolic part of the overall rhs!
    k1_para = zero(u0) # Additional PERK register for the parabolic part

    t0 = first(ode.tspan)
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

    # MPI additions
    level_info_mpi_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]
    level_info_mpi_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]

    if !mpi_isparallel()
        partition_variables!(level_info_elements,
                             level_info_elements_acc,
                             level_info_interfaces_acc,
                             level_info_boundaries_acc,
                             level_info_mortars_acc,
                             n_levels, mesh, dg, cache, alg)
    else
        if mesh isa ParallelP4estMesh
            # Get cell distribution for standard partitioning
            global_first_quadrant = unsafe_wrap(Array,
                                                unsafe_load(mesh.p4est).global_first_quadrant,
                                                mpi_nranks() + 1)
            # Need to copy `global_first_quadrant` to different variable as the former will change 
            # due to the call to `partition!`
            old_global_first_quadrant = copy(global_first_quadrant)

            # Get (global) element distribution to accordingly balance the solver
            partition_variables!(level_info_elements, n_levels,
                                 mesh, dg, cache, alg)

            # Balance such that each rank has the same number of RHS calls                                    
            balance_p4est_perk!(mesh, dg, cache, level_info_elements, alg.stages)
            # Actual move of elements across ranks
            rebalance_solver!(u0, mesh, equations, dg, cache, old_global_first_quadrant)
            reinitialize_boundaries!(semi.boundary_conditions, cache) # Needs to be called after `rebalance_solver!`

            # Reset `level_info_elements` after rebalancing
            level_info_elements = [Vector{Int64}() for _ in 1:n_levels]

            # Resize ODE vectors
            n_new = length(u0)
            resize!(du, n_new)
            resize!(u_tmp, n_new)
            resize!(k1, n_new)
        end
        partition_variables!(level_info_elements,
                             level_info_elements_acc,
                             level_info_interfaces_acc,
                             level_info_boundaries_acc,
                             level_info_mortars_acc,
                             # MPI additions
                             level_info_mpi_interfaces_acc,
                             level_info_mpi_mortars_acc,
                             n_levels, mesh, dg, cache, alg)
    end

    for i in 1:n_levels
        println("Number Elements integrated with level $i: ",
                length(level_info_elements[i]))
    end

    # Set (initial) distribution of DG nodal values
    level_u_indices_elements = [Vector{Int64}() for _ in 1:n_levels]
    partition_u!(level_u_indices_elements, level_info_elements,
                 n_levels, u0, mesh, equations, dg, cache)

    ### Done with setting up for handling of level-dependent integration ###
    du_para = zero(u0)
    integrator = PairedExplicitRK2SplitMultiIntegrator(u0, du, u_tmp,
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
                                                       k1, du_para, k1_para,
                                                       level_info_elements,
                                                       level_info_elements_acc,
                                                       level_info_interfaces_acc,
                                                       level_info_mpi_interfaces_acc,
                                                       level_info_boundaries_acc,
                                                       level_info_mortars_acc,
                                                       level_info_mpi_mortars_acc,
                                                       level_u_indices_elements,
                                                       -1, n_levels)

initialize_callbacks!(callback, integrator)

    return integrator
end
=#

# Version with DIFFERENT number of stages for hyperbolic and parabolic part
struct PairedExplicitRK2SplitMulti <:
       AbstractPairedExplicitRKSplitMulti{2}
    num_methods::Int64 # Number of optimized PERK family members, i.e., R
    num_stages::Int64 # = maximum number of stages
    stages::Vector{Int64} # For load-balancing of MPI-parallel p4est simulations

    num_methods_para::Int64 # Number of optimized PERK family members for the parabolic part, i.e., R
    num_stages_para::Int64 # = maximum number of stages for the parabolic part
    stages_para::Vector{Int64} # For load-balancing of MPI-parallel p4est simulations

    # Δt of the different methods divided by Δt_max
    dt_ratios::Vector{Float64} # hyperbolic timesteps
    dt_ratios_para::Vector{Float64} # parabolic timesteps

    # Butcher tableau variables
    a_matrices::Array{Float64, 3}
    a_matrices_para::Array{Float64, 3}
    c::Vector{Float64}
    b1::Float64
    bS::Float64

    # highest active/evaluated level; per stage
    max_active_levels::Vector{Int64}
    # highest level where the last stage `du` is added in the argument of the evaluated `rhs!`; per stage
    max_add_levels::Vector{Int64}

    # highest active level per stage for the parabolic part
    max_active_levels_para::Vector{Int64}
    # highest level where the last stage `du` is added in the argument of the evaluated `rhs!`; per stage for the parabolic part
    max_add_levels_para::Vector{Int64}
end

function PairedExplicitRK2SplitMulti(stages::Vector{Int64},
                                     stages_para::Vector{Int64},
                                     base_path_mon_coeffs::AbstractString,
                                     base_path_mon_coeffs_para::AbstractString,
                                     dt_ratios, dt_ratios_para;
                                     bS = 1.0, cS = 0.5)
    num_stages = maximum(stages)

    a_matrices, c,
    max_active_levels,
    max_add_levels = compute_PairedExplicitRK2Multi_butcher_tableau(stages, num_stages,
                                                                    base_path_mon_coeffs,
                                                                    bS, cS)

    num_stages_para = maximum(stages_para)

    a_matrices_para, _,
    max_active_levels_para_,
    max_add_levels_para_ = compute_PairedExplicitRK2Multi_butcher_tableau(stages_para,
                                                                          # Need to supply `num_stages` here to have matching Butcher tableaus
                                                                          num_stages,
                                                                          base_path_mon_coeffs_para,
                                                                          bS, cS)

    return PairedExplicitRK2SplitMulti(length(stages), num_stages, stages,
                                       length(stages_para), num_stages_para,
                                       stages_para,
                                       dt_ratios, dt_ratios_para,
                                       a_matrices, a_matrices_para,
                                       c, 1 - bS, bS,
                                       max_active_levels, max_add_levels,
                                       max_active_levels_para_, max_add_levels_para_)
end

# Version with DIFFERENT number of stages and partitioning(!) for hyperbolic and parabolic part
mutable struct PairedExplicitRK2SplitMultiIntegrator{RealT <: Real, uType,
                                                     Params, Sol, F,
                                                     PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKSplitMultiIntegrator{2}
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
    alg::PairedExplicitRK2SplitMulti
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # Additional PERK register
    k1::uType
    # For split (hyperbolic-parabolic) problems
    du_para::uType # Stores the parabolic part of the overall rhs!
    k1_para::uType # Additional PERK register for the parabolic part

    # Variables managing level-dependent integration
    level_info_elements::Vector{Vector{Int64}}
    level_info_elements_acc::Vector{Vector{Int64}}

    level_info_interfaces_acc::Vector{Vector{Int64}}
    level_info_mpi_interfaces_acc::Vector{Vector{Int64}}

    level_info_boundaries_acc::Vector{Vector{Int64}}

    level_info_mortars_acc::Vector{Vector{Int64}}
    level_info_mpi_mortars_acc::Vector{Vector{Int64}}

    level_u_indices_elements::Vector{Vector{Int64}}

    coarsest_lvl::Int64
    n_levels::Int64

    # For parabolic part
    level_info_elements_para::Vector{Vector{Int64}}
    level_info_elements_para_acc::Vector{Vector{Int64}}

    level_info_interfaces_para_acc::Vector{Vector{Int64}}
    level_info_mpi_interfaces_para_acc::Vector{Vector{Int64}}

    level_info_boundaries_para_acc::Vector{Vector{Int64}}

    level_info_mortars_para_acc::Vector{Vector{Int64}}
    level_info_mpi_mortars_para_acc::Vector{Vector{Int64}}

    level_u_indices_elements_para::Vector{Vector{Int64}}

    coarsest_lvl_para::Int64
    n_levels_para::Int64
end

function init(ode::ODEProblem, alg::PairedExplicitRK2SplitMulti;
              dt, callback = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    k1 = zero(u0) # Additional PERK register
    du_para = zero(u0) # Stores the parabolic part of the overall rhs!
    k1_para = zero(u0) # Additional PERK register for the parabolic part

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    ### Set datastructures for handling of level-dependent integration ###
    semi = ode.p
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

    n_levels, n_levels_para = get_n_levels(mesh, alg)

    level_info_elements = [Vector{Int64}() for _ in 1:n_levels]
    level_info_elements_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_boundaries_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]

    # MPI additions
    level_info_mpi_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]
    level_info_mpi_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]

    # For parabolic part
    level_info_elements_para = [Vector{Int64}() for _ in 1:n_levels_para]
    level_info_elements_para_acc = [Vector{Int64}() for _ in 1:n_levels_para]

    level_info_interfaces_para_acc = [Vector{Int64}() for _ in 1:n_levels_para]

    level_info_boundaries_para_acc = [Vector{Int64}() for _ in 1:n_levels_para]

    level_info_mortars_para_acc = [Vector{Int64}() for _ in 1:n_levels_para]

    level_info_mpi_interfaces_para_acc = [Vector{Int64}() for _ in 1:n_levels_para]
    level_info_mpi_mortars_para_acc = [Vector{Int64}() for _ in 1:n_levels_para]

    if !mpi_isparallel()
        partition_variables!(level_info_elements,
                             level_info_elements_acc,
                             level_info_interfaces_acc,
                             level_info_boundaries_acc,
                             level_info_mortars_acc,
                             n_levels, mesh, dg, cache, alg)

        # Partition parabolic helper variables
        partition_variables!(level_info_elements_para,
                             level_info_elements_para_acc,
                             level_info_interfaces_para_acc,
                             level_info_boundaries_para_acc,
                             level_info_mortars_para_acc,
                             n_levels_para, mesh, dg, cache, alg,
                             parabolic = true)
    else
        if mesh isa ParallelP4estMesh
            # Get cell distribution for standard partitioning
            global_first_quadrant = unsafe_wrap(Array,
                                                unsafe_load(mesh.p4est).global_first_quadrant,
                                                mpi_nranks() + 1)
            # Need to copy `global_first_quadrant` to different variable as the former will change 
            # due to the call to `partition!`
            old_global_first_quadrant = copy(global_first_quadrant)

            # Get (global) element distribution to accordingly balance the solver
            partition_variables!(level_info_elements, n_levels,
                                 mesh, dg, cache, alg)

            # Balance such that each rank has the same number of RHS calls                                    
            balance_p4est_perk!(mesh, dg, cache, level_info_elements, alg.stages)
            # Actual move of elements across ranks
            rebalance_solver!(u0, mesh, equations, dg, cache, old_global_first_quadrant)
            reinitialize_boundaries!(semi.boundary_conditions, cache) # Needs to be called after `rebalance_solver!`

            # Reset `level_info_elements` after rebalancing
            level_info_elements = [Vector{Int64}() for _ in 1:n_levels]

            # Resize ODE vectors
            n_new = length(u0)
            resize!(du, n_new)
            resize!(u_tmp, n_new)
            resize!(k1, n_new)
        end
        partition_variables!(level_info_elements,
                             level_info_elements_acc,
                             level_info_interfaces_acc,
                             level_info_boundaries_acc,
                             level_info_mortars_acc,
                             # MPI additions
                             level_info_mpi_interfaces_acc,
                             level_info_mpi_mortars_acc,
                             n_levels, mesh, dg, cache, alg)
    end

    for i in 1:n_levels
        println("Number Elements integrated with level $i: ",
                length(level_info_elements[i]))
    end
    for i in 1:n_levels_para
        println("Number Elements integrated with level $i (parabolic): ",
                length(level_info_elements_para[i]))
    end

    # Set (initial) distribution of DG nodal values
    level_u_indices_elements = [Vector{Int64}() for _ in 1:n_levels]
    partition_u!(level_u_indices_elements, level_info_elements,
                 n_levels, u0, mesh, equations, dg, cache)

    # For parabolic part
    level_u_indices_elements_para = [Vector{Int64}() for _ in 1:n_levels_para]
    partition_u!(level_u_indices_elements_para, level_info_elements_para,
                 n_levels_para, u0, mesh, equations, dg, cache)

    ### Done with setting up for handling of level-dependent integration ###
    du_para = zero(u0)
    integrator = PairedExplicitRK2SplitMultiIntegrator(u0, du, u_tmp,
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
                                                       k1, du_para, k1_para,
                                                       level_info_elements,
                                                       level_info_elements_acc,
                                                       level_info_interfaces_acc,
                                                       level_info_mpi_interfaces_acc,
                                                       level_info_boundaries_acc,
                                                       level_info_mortars_acc,
                                                       level_info_mpi_mortars_acc,
                                                       level_u_indices_elements,
                                                       -1, n_levels,
                                                       level_info_elements_para,
                                                       level_info_elements_para_acc,
                                                       level_info_interfaces_para_acc,
                                                       level_info_mpi_interfaces_para_acc,
                                                       level_info_boundaries_para_acc,
                                                       level_info_mortars_acc,
                                                       level_info_mpi_mortars_para_acc,
                                                       level_u_indices_elements_para,
                                                       -1, n_levels_para)

    initialize_callbacks!(callback, integrator)

    return integrator
end
end # @muladd
