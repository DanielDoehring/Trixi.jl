# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

#=
@doc raw"""
    PairedExplicitRK4SplitMulti(num_stages,
                                base_path_a_coeffs::AbstractString,
                                base_path_a_coeffs_para::AbstractString,
                                dt_opt = nothing;
                                cS3 = 1.0f0)
"""
struct PairedExplicitRK4SplitMulti <:
       AbstractPairedExplicitRKSplitMulti{4}
    num_methods::Int64 # Number of optimized PERK family members, i.e., R
    num_stages::Int64 # = maximum number of stages

    # Δt of the different methods divided by Δt_max
    dt_ratios::Vector{Float64}

    # Butcher tableau variables

    a_matrices::Array{Float64, 3}
    a_matrices_para::Array{Float64, 3}
    a_matrix_constant::Matrix{Float64}
    c::Vector{Float64}

    # highest active/evaluated level; per stage
    max_active_levels::Vector{Int64}
    # highest level where the last stage `du` is added in the argument of the evaluated `rhs!`; per stage
    max_add_levels::Vector{Int64}
end

function PairedExplicitRK4SplitMulti(stages::Vector{Int64},
                                     base_path_a_coeffs::AbstractString,
                                     base_path_a_coeffs_para::AbstractString,
                                     dt_ratios;
                                     cS3 = 1.0f0)
    num_stages = maximum(stages)

    a_matrices,
    a_matrix_constant, c,
    max_active_levels,
    max_add_levels = compute_PairedExplicitRK4Multi_butcher_tableau(stages,
                                                                    num_stages,
                                                                    base_path_a_coeffs,
                                                                    cS3)

    a_matrices_para, _, _, _, _, _ = compute_PairedExplicitRK4Multi_butcher_tableau(stages,
                                                                                    num_stages,
                                                                                    base_path_a_coeffs_para,
                                                                                    cS3)

    return PairedExplicitRK4SplitMulti(length(stages), num_stages,
                                       dt_ratios,
                                       a_matrices, a_matrices_para, a_matrix_constant,
                                       c,
                                       max_active_levels, max_add_levels)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct PairedExplicitRK4SplitMultiIntegrator{RealT <: Real, uType <: AbstractVector,
                                                     Params, Sol, F,
                                                     PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKSplitMultiIntegrator{4}
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
    const alg::PairedExplicitRK4SplitMulti
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

    level_info_boundaries_acc::Vector{Vector{Int64}}

    level_info_mortars_acc::Vector{Vector{Int64}}

    level_info_u::Vector{Vector{Int64}}

    coarsest_lvl::Int64
    n_levels::Int64
end

function init(ode::ODEProblem, alg::PairedExplicitRK4SplitMulti;
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

    partition_variables!(level_info_elements,
                         level_info_elements_acc,
                         level_info_interfaces_acc,
                         level_info_boundaries_acc,
                         level_info_mortars_acc,
                         n_levels, semi, alg)

    for i in 1:n_levels
        println("Number Elements integrated with level $i: ",
                length(level_info_elements[i]))
    end

    # Set (initial) distribution of DG nodal values
    level_info_u = [Vector{Int64}() for _ in 1:n_levels]
    partition_u!(level_info_u, level_info_elements,
                 n_levels, u0, semi)

    ### Done with setting up for handling of level-dependent integration ###
    du_para = zero(u0)
    integrator = PairedExplicitRK4SplitMultiIntegrator(u0, du, u_tmp,
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
                                                       level_info_boundaries_acc,
                                                       level_info_mortars_acc,
                                                       level_info_u,
                                                       -1, n_levels)

initialize_callbacks!(callback, integrator)

    return integrator
end
=#

# Version with DIFFERENT number of stages for hyperbolic and parabolic part
struct PairedExplicitRK4SplitMulti <:
       AbstractPairedExplicitRKSplitMulti{4}
    num_methods::Int64 # Number of optimized PERK family members, i.e., R
    num_stages::Int64 # = maximum number of stages

    num_methods_para::Int64 # Number of optimized PERK family members for the parabolic part, i.e., R
    num_stages_para::Int64 # = maximum number of stages for the parabolic part

    # Δt of the different methods divided by Δt_max
    dt_ratios::Vector{Float64} # hyperbolic timesteps
    dt_ratios_para::Vector{Float64} # parabolic timesteps

    # Butcher tableau variables
    a_matrices::Array{Float64, 3}
    a_matrices_para::Array{Float64, 3}
    a_matrix_constant::Matrix{Float64}
    c::Vector{Float64}

    # highest active/evaluated level; per stage
    max_active_levels::Vector{Int64}
    # highest level where the last stage `du` is added in the argument of the evaluated `rhs!`; per stage
    max_add_levels::Vector{Int64}

    # highest active/evaluated level; per stage for the parabolic part
    max_active_levels_para::Vector{Int64}
    # highest level where the last stage `du` is added in the argument of the evaluated `rhs!`; per stage for the parabolic part
    max_add_levels_para::Vector{Int64}
end

function PairedExplicitRK4SplitMulti(stages::Vector{Int64},
                                     stages_para::Vector{Int64},
                                     base_path_mon_coeffs::AbstractString,
                                     base_path_mon_coeffs_para::AbstractString,
                                     dt_ratios, dt_ratios_para;
                                     cS3 = 1.0f0)
    num_stages = maximum(stages)

    a_matrices,
    a_matrix_constant, c,
    max_active_levels,
    max_add_levels = compute_PairedExplicitRK4Multi_butcher_tableau(stages,
                                                                    num_stages,
                                                                    base_path_mon_coeffs,
                                                                    cS3)

    num_stages_para = maximum(stages_para)

    a_matrices_para,
    _, _,
    max_active_levels_para_,
    max_add_levels_para_ = compute_PairedExplicitRK4Multi_butcher_tableau(stages_para,
                                                                          # Need to supply `num_stages` here to have matching Butcher tableaus
                                                                          num_stages,
                                                                          base_path_mon_coeffs_para,
                                                                          cS3)

    return PairedExplicitRK4SplitMulti(length(stages), num_stages,
                                       length(stages_para), num_stages_para,
                                       dt_ratios, dt_ratios_para,
                                       a_matrices, a_matrices_para,
                                       a_matrix_constant, c,
                                       max_active_levels, max_add_levels,
                                       max_active_levels_para_, max_add_levels_para_)
end

# Version with DIFFERENT number of stages and partitioning(!) for hyperbolic and parabolic part
mutable struct PairedExplicitRK4SplitMultiIntegrator{RealT <: Real,
                                                     uType <: AbstractVector,
                                                     Params, Sol, F,
                                                     PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKSplitMultiIntegrator{4}
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
    const f::F # `rhs!` of the semidiscretization
    const alg::PairedExplicitRK4SplitMulti
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

    level_info_boundaries_acc::Vector{Vector{Int64}}

    level_info_mortars_acc::Vector{Vector{Int64}}

    level_info_u::Vector{Vector{Int64}}

    coarsest_lvl::Int64
    n_levels::Int64

    # For parabolic part
    level_info_elements_para::Vector{Vector{Int64}}
    level_info_elements_para_acc::Vector{Vector{Int64}}

    level_info_interfaces_para_acc::Vector{Vector{Int64}}

    level_info_boundaries_para_acc::Vector{Vector{Int64}}

    level_info_mortars_para_acc::Vector{Vector{Int64}}

    level_info_u_para::Vector{Vector{Int64}}

    coarsest_lvl_para::Int64
    n_levels_para::Int64
end

function init(ode::ODEProblem, alg::PairedExplicitRK4SplitMulti;
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
    mesh, _, _, _ = mesh_equations_solver_cache(semi)

    n_levels, n_levels_para = get_n_levels(mesh, alg)

    level_info_elements = [Vector{Int64}() for _ in 1:n_levels]
    level_info_elements_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_boundaries_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]

    # For parabolic part
    level_info_elements_para = [Vector{Int64}() for _ in 1:n_levels_para]
    level_info_elements_para_acc = [Vector{Int64}() for _ in 1:n_levels_para]

    level_info_interfaces_para_acc = [Vector{Int64}() for _ in 1:n_levels_para]

    level_info_boundaries_para_acc = [Vector{Int64}() for _ in 1:n_levels_para]

    level_info_mortars_para_acc = [Vector{Int64}() for _ in 1:n_levels_para]

    partition_variables!(level_info_elements,
                         level_info_elements_acc,
                         level_info_interfaces_acc,
                         level_info_boundaries_acc,
                         level_info_mortars_acc,
                         n_levels, semi, alg)

    # Partition parabolic helper variables
    partition_variables!(level_info_elements_para,
                         level_info_elements_para_acc,
                         level_info_interfaces_para_acc,
                         level_info_boundaries_para_acc,
                         level_info_mortars_para_acc,
                         n_levels_para, semi, alg;
                         quadratic_scaling = true)
                         #quadratic_scaling = false) # TODO: Could make this variable of the algorithm to avoid manual hacks

    for i in 1:n_levels
        println("Number Elements integrated with level $i: ",
                length(level_info_elements[i]))
    end
    for i in 1:n_levels_para
        println("Number Elements integrated with level $i (parabolic): ",
                length(level_info_elements_para[i]))
    end

    # Set (initial) distribution of DG nodal values
    level_info_u = [Vector{Int64}() for _ in 1:n_levels]
    partition_u!(level_info_u, level_info_elements,
                 n_levels, u0, semi)

    # For parabolic part
    level_info_u_para = [Vector{Int64}() for _ in 1:n_levels_para]
    partition_u!(level_info_u_para, level_info_elements_para,
                 n_levels_para, u0, semi)

    ### Done with setting up for handling of level-dependent integration ###
    integrator = PairedExplicitRK4SplitMultiIntegrator(u0, du, u_tmp,
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
                                                       level_info_boundaries_acc,
                                                       level_info_mortars_acc,
                                                       level_info_u,
                                                       -1, n_levels,
                                                       level_info_elements_para,
                                                       level_info_elements_para_acc,
                                                       level_info_interfaces_para_acc,
                                                       level_info_boundaries_para_acc,
                                                       level_info_mortars_acc,
                                                       level_info_u_para,
                                                       -1, n_levels_para)

    initialize_callbacks!(callback, integrator)

    return integrator
end
end # @muladd
