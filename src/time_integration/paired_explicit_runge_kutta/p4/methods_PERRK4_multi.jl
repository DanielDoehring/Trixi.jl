# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct PairedExplicitRelaxationRK4Multi{RelaxationSolver} <:
       AbstractPairedExplicitRKMulti{4}
    PERK4Multi::PairedExplicitRK4Multi
    relaxation_solver::RelaxationSolver
end

function PairedExplicitRelaxationRK4Multi(stages::Vector{Int64},
                                          base_path_a_coeffs::AbstractString,
                                          dt_ratios;
                                          cS3 = 1.0f0,
                                          relaxation_solver = RelaxationSolverNewton())
    return PairedExplicitRelaxationRK4Multi{typeof(relaxation_solver)}(PairedExplicitRK4Multi(stages,
                                                                                              base_path_a_coeffs,
                                                                                              dt_ratios;
                                                                                              cS3 = cS3),
                                                                       relaxation_solver)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRelaxationRK4MultiIntegrator{RealT <: Real, uType,
                                                          Params, Sol, F,
                                                          PairedExplicitRKOptions,
                                                          RelaxationSolver} <:
               AbstractPairedExplicitRelaxationRKMultiIntegrator{4}
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
    alg::PairedExplicitRK4Multi
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
    level_info_mpi_interfaces_acc::Vector{Vector{Int64}}

    level_info_boundaries_acc::Vector{Vector{Int64}}

    level_info_mortars_acc::Vector{Vector{Int64}}
    level_info_mpi_mortars_acc::Vector{Vector{Int64}}

    level_u_indices_elements::Vector{Vector{Int64}}

    coarsest_lvl::Int64
    n_levels::Int64

    # Entropy Relaxation additions
    gamma::RealT # Relaxation parameter
    S_old::RealT # Entropy of previous iterate
    relaxation_solver::RelaxationSolver

    # For AMR: Counting RHS evals
    #RHSCalls::Int64
end

mutable struct PairedExplicitRelaxationRK4MultiParabolicIntegrator{RealT <: Real, uType,
                                                                   Params, Sol, F,
                                                                   PairedExplicitRKOptions,
                                                                   RelaxationSolver} <:
               AbstractPairedExplicitRelaxationRKMultiParabolicIntegrator{4}
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
    alg::PairedExplicitRK4Multi
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
    level_info_mpi_interfaces_acc::Vector{Vector{Int64}}

    level_info_boundaries_acc::Vector{Vector{Int64}}

    level_info_mortars_acc::Vector{Vector{Int64}}
    level_info_mpi_mortars_acc::Vector{Vector{Int64}}

    level_u_indices_elements::Vector{Vector{Int64}}

    coarsest_lvl::Int64
    n_levels::Int64

    # Entropy Relaxation additions
    gamma::RealT # Relaxation parameter
    S_old::RealT # Entropy of previous iterate
    relaxation_solver::RelaxationSolver

    # Addition for hyperbolic-parabolic problems:
    # We need another register to temporarily store the changes due to the parabolic part only.
    # The changes due to the hyperbolic part are stored in the usual `du` register.
    du_para::uType
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
    semi = ode.p
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    n_dims = ndims(mesh) # Spatial dimension

    n_levels = get_n_levels(mesh, alg.PERK4Multi)

    level_info_elements = [Vector{Int64}() for _ in 1:n_levels]
    level_info_elements_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_boundaries_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]

    # MPI additions
    level_info_mpi_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]
    level_info_mpi_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]

    # For entropy relaxation
    gamma = one(eltype(u0))
    u_wrap = wrap_array(u0, semi)
    S_old = integrate(entropy, u_wrap, mesh, equations, dg, cache)

    # TODO: Call different function for mpi_isparallel() == true
    partition_variables!(level_info_elements,
                         level_info_elements_acc,
                         level_info_interfaces_acc,
                         level_info_boundaries_acc,
                         level_info_mortars_acc,
                         n_levels, n_dims, mesh, dg, cache, alg.PERK4Multi.dt_ratios)

    for i in 1:n_levels
        println("Number Elements integrated with level $i: ",
                length(level_info_elements[i]))
    end

    # Set (initial) distribution of DG nodal values
    level_u_indices_elements = [Vector{Int64}() for _ in 1:n_levels]
    partition_u!(level_u_indices_elements, level_info_elements,
                 n_levels, u0, mesh, equations, dg, cache)

    ### Done with setting up for handling of level-dependent integration ###

    if isa(semi, SemidiscretizationHyperbolicParabolic)
        du_para = zero(u0)
        integrator = PairedExplicitRelaxationRK4MultiParabolicIntegrator(u0, du, u_tmp,
                                                                         t0, tdir,
                                                                         dt, zero(dt),
                                                                         iter, semi,
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
                                                                         level_info_mortars_acc,
                                                                         level_info_mpi_mortars_acc,
                                                                         level_u_indices_elements,
                                                                         -1, n_levels,
                                                                         gamma, S_old,
                                                                         alg.relaxation_solver,
                                                                         du_para)
    else # Hyperbolic case
        integrator = PairedExplicitRelaxationRK4MultiIntegrator(u0, du, u_tmp,
                                                                t0, tdir,
                                                                dt, zero(dt),
                                                                iter, semi,
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
                                                                level_info_mortars_acc,
                                                                level_info_mpi_mortars_acc,
                                                                level_u_indices_elements,
                                                                -1, n_levels,
                                                                gamma, S_old,
                                                                alg.relaxation_solver)
        # For AMR: Counting RHS evals 
        # 0)
    end

    initialize_callbacks!(callback, integrator)

    return integrator
end
end # @muladd
