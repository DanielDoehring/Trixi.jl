# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct PairedExplicitCoupledRK2Multi <:
       AbstractPairedExplicitRKMulti{2}
    num_methods::Int64 # Number of optimized PERK family members, i.e., R
    num_stages::Int64 # = maximum number of stages

    # Δt of the different methods divided by Δt_max
    dt_ratios_1::Vector{Float64}
    dt_ratios_2::Vector{Float64}

    # Butcher tableau variables
    a_matrices_1::Array{Float64, 3}
    a_matrices_2::Array{Float64, 3}

    c::Vector{Float64}
    b1::Float64
    bS::Float64

    # highest active/evaluated level; per stage
    max_active_levels::Vector{Int64}
    # highest level where the last stage `du` is added in the argument of the evaluated `rhs!`; per stage
    max_add_levels::Vector{Int64}
end

function PairedExplicitCoupledRK2Multi(stages::Vector{Int64},
                                       base_path_mon_coeffs_1::AbstractString,
                                       base_path_mon_coeffs_2::AbstractString,
                                       dt_ratios_1, dt_ratios_2;
                                       bS = 1.0, cS = 0.5)
    num_stages = maximum(stages)

    a_matrices_1, c,
    max_active_levels,
    max_add_levels = compute_PairedExplicitRK2Multi_butcher_tableau(stages, num_stages,
                                                                    base_path_mon_coeffs_1,
                                                                    bS, cS)

    a_matrices_2, _,
    _, _ = compute_PairedExplicitRK2Multi_butcher_tableau(stages, num_stages,
                                                          base_path_mon_coeffs_2,
                                                          bS, cS)

    return PairedExplicitCoupledRK2Multi(length(stages), num_stages,
                                         dt_ratios_1, dt_ratios_2,
                                         a_matrices_1, a_matrices_2,
                                         c, 1 - bS, bS,
                                         max_active_levels, max_add_levels)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRK2CoupledMultiIntegrator{RealT <: Real,
                                                       uType <: AbstractVector,
                                                       Params, Sol, F,
                                                       PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKCoupledMultiIntegrator{2}
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
    const alg::PairedExplicitCoupledRK2Multi
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # Additional PERK register
    k1::uType

    # Variables managing level-dependent integration
    level_info_elements_1::Vector{Vector{Int64}}
    level_info_elements_acc_1::Vector{Vector{Int64}}
    level_info_interfaces_acc_1::Vector{Vector{Int64}}
    level_info_boundaries_acc_1::Vector{Vector{Int64}}
    level_info_mortars_acc_1::Vector{Vector{Int64}}
    level_info_u_1::Vector{Vector{Int64}}

    level_info_elements_2::Vector{Vector{Int64}}
    level_info_elements_acc_2::Vector{Vector{Int64}}
    level_info_interfaces_acc_2::Vector{Vector{Int64}}
    level_info_boundaries_acc_2::Vector{Vector{Int64}}
    level_info_mortars_acc_2::Vector{Vector{Int64}}
    level_info_u_2::Vector{Vector{Int64}}

    coarsest_lvl::Int64
    n_levels::Int64
end

function init(ode::ODEProblem, alg::PairedExplicitCoupledRK2Multi;
              dt, callback = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    k1 = zero(u0) # Additional PERK register

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    ### Set datastructures for handling of level-dependent integration ###
    semi = ode.p
    semi_1 = semi.semis[1]
    semi_2 = semi.semis[2]

    mesh_1 = semi_1.mesh
    n_levels = get_n_levels(mesh_1, alg)
    mesh_2 = semi_2.mesh
    @assert n_levels==get_n_levels(mesh_2, alg) "Number of levels must be the same for both semi-discretizations!"

    level_info_elements_1 = [Vector{Int64}() for _ in 1:n_levels]
    level_info_elements_acc_1 = [Vector{Int64}() for _ in 1:n_levels]
    level_info_interfaces_acc_1 = [Vector{Int64}() for _ in 1:n_levels]
    level_info_boundaries_acc_1 = [Vector{Int64}() for _ in 1:n_levels]
    level_info_mortars_acc_1 = [Vector{Int64}() for _ in 1:n_levels]

    partition_variables!(level_info_elements_1,
                         level_info_elements_acc_1,
                         level_info_interfaces_acc_1,
                         level_info_boundaries_acc_1,
                         level_info_mortars_acc_1,
                         n_levels, semi_1, alg.dt_ratios_1)

    for i in 1:n_levels
        println("Number Elements integrated with level $i: ",
                length(level_info_elements_1[i]))
    end

    # Set (initial) distribution of DG nodal values
    level_info_u_1 = [Vector{Int64}() for _ in 1:n_levels]

    u_1 = get_system_u_ode(u0, 1, semi)
    partition_u!(level_info_u_1,
                 level_info_elements_1, n_levels,
                 u_1, semi_1)

    level_info_elements_2 = [Vector{Int64}() for _ in 1:n_levels]
    level_info_elements_acc_2 = [Vector{Int64}() for _ in 1:n_levels]
    level_info_interfaces_acc_2 = [Vector{Int64}() for _ in 1:n_levels]
    level_info_boundaries_acc_2 = [Vector{Int64}() for _ in 1:n_levels]
    level_info_mortars_acc_2 = [Vector{Int64}() for _ in 1:n_levels]

    partition_variables!(level_info_elements_2,
                         level_info_elements_acc_2,
                         level_info_interfaces_acc_2,
                         level_info_boundaries_acc_2,
                         level_info_mortars_acc_2,
                         n_levels, semi_2, alg.dt_ratios_2)

    for i in 1:n_levels
        println("Number Elements integrated with level $i: ",
                length(level_info_elements_2[i]))
    end

    level_info_u_2 = [Vector{Int64}() for _ in 1:n_levels]

    u_2 = get_system_u_ode(u0, 2, semi)
    partition_u!(level_info_u_2,
                 level_info_elements_2, n_levels,
                 u_2, semi_2)

    # Convert local indices to global indices
    max_u_1 = maximum(level_info_u_1[1])
    for level in 1:n_levels
        level_info_u_2[level] .+= max_u_1 # TODO: Test .+= without loop over levels for more compact code!
    end

    integrator = PairedExplicitRK2CoupledMultiIntegrator(u0, du, u_tmp,
                                                         t0, tdir,
                                                         dt, zero(dt),
                                                         iter, semi,
                                                         (prob = ode,),
                                                         ode.f, alg,
                                                         PairedExplicitRKOptions(callback,
                                                                                 ode.tspan;
                                                                                 kwargs...),
                                                         false, true, false,
                                                         k1,
                                                         level_info_elements_1,
                                                         level_info_elements_acc_1,
                                                         level_info_interfaces_acc_1,
                                                         level_info_boundaries_acc_1,
                                                         level_info_mortars_acc_1,
                                                         level_info_u_1,
                                                         level_info_elements_2,
                                                         level_info_elements_acc_2,
                                                         level_info_interfaces_acc_2,
                                                         level_info_boundaries_acc_2,
                                                         level_info_mortars_acc_2,
                                                         level_info_u_2,
                                                         -1, n_levels)

    initialize_callbacks!(callback, integrator)

    return integrator
end
end # @muladd
