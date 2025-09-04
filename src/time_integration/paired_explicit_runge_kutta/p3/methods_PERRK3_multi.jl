# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct PairedExplicitRelaxationRK3Multi{RelaxationSolver} <:
       AbstractPairedExplicitRKMulti{3}
    PERK3Multi::PairedExplicitRK3Multi
    relaxation_solver::RelaxationSolver
end

function PairedExplicitRelaxationRK3Multi(stages::Vector{Int64},
                                          base_path_a_coeffs::AbstractString,
                                          dt_ratios;
                                          cS2::Float64 = 1.0,
                                          relaxation_solver = RelaxationSolverNewton())
    return PairedExplicitRelaxationRK3Multi{typeof(relaxation_solver)}(PairedExplicitRK3Multi(stages,
                                                                                              base_path_a_coeffs,
                                                                                              dt_ratios,
                                                                                              cS2 = cS2),
                                                                       relaxation_solver)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRelaxationRK3MultiIntegrator{RealT <: Real,
                                                          uType <: AbstractVector,
                                                          Params, Sol, F,
                                                          PairedExplicitRKOptions,
                                                          RelaxationSolver} <:
               AbstractPairedExplicitRelaxationRKMultiIntegrator{3}
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
    alg::PairedExplicitRK3Multi
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # Additional PERK3 registers
    k1::uType
    kS1::uType

    # Variables managing level-dependent integration
    level_info_elements::Vector{Vector{Int64}}
    level_info_elements_acc::Vector{Vector{Int64}}

    level_info_interfaces_acc::Vector{Vector{Int64}}
    level_info_mpi_interfaces_acc::Vector{Vector{Int64}}

    level_info_boundaries_acc::Vector{Vector{Int64}}

    level_info_mortars_acc::Vector{Vector{Int64}}
    level_info_mpi_mortars_acc::Vector{Vector{Int64}}

    level_info_u::Vector{Vector{Int64}}

    coarsest_lvl::Int64
    n_levels::Int64

    # Entropy Relaxation additions
    gamma::RealT # Relaxation parameter
    S_old::RealT # Entropy of previous iterate
    relaxation_solver::RelaxationSolver
end

mutable struct PairedExplicitRelaxationRK3MultiParabolicIntegrator{RealT <: Real,
                                                                   uType <:
                                                                   AbstractVector,
                                                                   Params, Sol, F,
                                                                   PairedExplicitRKOptions,
                                                                   RelaxationSolver} <:
               AbstractPairedExplicitRelaxationRKMultiParabolicIntegrator{3}
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
    alg::PairedExplicitRK3Multi
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # Additional PERK3 registers
    k1::uType
    kS1::uType

    # Variables managing level-dependent integration
    level_info_elements::Vector{Vector{Int64}}
    level_info_elements_acc::Vector{Vector{Int64}}

    level_info_interfaces_acc::Vector{Vector{Int64}}
    # Parabolic currently not MPI-parallelized

    level_info_boundaries_acc::Vector{Vector{Int64}}

    level_info_mortars_acc::Vector{Vector{Int64}}

    level_info_u::Vector{Vector{Int64}}
    # For efficient addition of `du` to `du_ode` in `rhs_hyperbolic_parabolic`
    level_info_u_acc::Vector{Vector{Int64}}

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

function init(ode::ODEProblem, alg::PairedExplicitRelaxationRK3Multi;
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

    ### Set datastructures for handling of level-dependent integration ###
    semi = ode.p
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

    n_levels = get_n_levels(mesh, alg.PERK3Multi)

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

    if !mpi_isparallel()
        partition_variables!(level_info_elements,
                             level_info_elements_acc,
                             level_info_interfaces_acc,
                             level_info_boundaries_acc,
                             level_info_mortars_acc,
                             n_levels, semi, alg.PERK3Multi)
    else # NOTE: Never tested
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
                                 semi, alg)

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
            resize!(kS1, n_new)
        end

        partition_variables!(level_info_elements,
                             level_info_elements_acc,
                             level_info_interfaces_acc,
                             level_info_boundaries_acc,
                             level_info_mortars_acc,
                             # MPI additions
                             level_info_mpi_interfaces_acc,
                             level_info_mpi_mortars_acc,
                             n_levels, semi, alg.PERK3Multi)
    end

    for i in 1:n_levels
        println("Number Elements integrated with level $i: ",
                length(level_info_elements[i]))
    end

    # Set (initial) distribution of DG nodal values
    level_info_u = [Vector{Int64}() for _ in 1:n_levels]

    ### Done with setting up for handling of level-dependent integration ###
    if isa(semi, SemidiscretizationHyperbolicParabolic)
        # For efficient addition of `du` to `du_ode` in `rhs_hyperbolic_parabolic`
        level_info_u_acc = [Vector{Int64}() for _ in 1:n_levels]
        partition_u!(level_info_u, level_info_u_acc,
                     level_info_elements, n_levels,
                     u0, semi)

        du_para = zero(u0)
        integrator = PairedExplicitRelaxationRK3MultiParabolicIntegrator(u0, du, u_tmp,
                                                                         t0, tdir,
                                                                         dt, zero(dt),
                                                                         iter, semi,
                                                                         (prob = ode,),
                                                                         ode.f,
                                                                         # Note that here the `PERK3Multi` algorithm is passed on as 
                                                                         # `alg` of the integrator
                                                                         alg.PERK3Multi,
                                                                         PairedExplicitRKOptions(callback,
                                                                                                 ode.tspan;
                                                                                                 kwargs...),
                                                                         false, true,
                                                                         false,
                                                                         k1, kS1,
                                                                         level_info_elements,
                                                                         level_info_elements_acc,
                                                                         level_info_interfaces_acc,
                                                                         level_info_boundaries_acc,
                                                                         level_info_mortars_acc,
                                                                         level_info_u,
                                                                         level_info_u_acc,
                                                                         -1, n_levels,
                                                                         gamma, S_old,
                                                                         alg.relaxation_solver,
                                                                         du_para)
    else # Hyperbolic case
        partition_u!(level_info_u,
                     level_info_elements, n_levels,
                     u0, semi)

        integrator = PairedExplicitRelaxationRK3MultiIntegrator(u0, du, u_tmp,
                                                                t0, tdir,
                                                                dt, zero(dt),
                                                                iter, semi,
                                                                (prob = ode,),
                                                                ode.f,
                                                                # Note that here the `PERK3Multi` algorithm is passed on as 
                                                                # `alg` of the integrator
                                                                alg.PERK3Multi,
                                                                PairedExplicitRKOptions(callback,
                                                                                        ode.tspan;
                                                                                        kwargs...),
                                                                false, true,
                                                                false,
                                                                k1, kS1,
                                                                level_info_elements,
                                                                level_info_elements_acc,
                                                                level_info_interfaces_acc,
                                                                level_info_mpi_interfaces_acc,
                                                                level_info_boundaries_acc,
                                                                level_info_mortars_acc,
                                                                level_info_mpi_mortars_acc,
                                                                level_info_u,
                                                                -1, n_levels,
                                                                gamma, S_old,
                                                                alg.relaxation_solver)
    end

    initialize_callbacks!(callback, integrator)

    return integrator
end
end # @muladd
