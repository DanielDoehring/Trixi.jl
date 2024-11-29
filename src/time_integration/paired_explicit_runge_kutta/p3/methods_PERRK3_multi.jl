# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct PairedExplicitRelaxationRK3Multi <: AbstractPairedExplicitRKMulti
    PERK3Multi::PairedExplicitRK3Multi
    relaxation_solver::RelaxationSolver
end

function PairedExplicitRelaxationRK3Multi(stages::Vector{Int64},
                                          base_path_a_coeffs::AbstractString,
                                          dt_ratios;
                                          cS2::Float64 = 1.0,
                                          relaxation_solver = EntropyRelaxationNewton())
    return PairedExplicitRelaxationRK3Multi(PairedExplicitRK3Multi(stages,
                                                                   base_path_a_coeffs,
                                                                   dt_ratios,
                                                                   cS2 = cS2),
                                            relaxation_solver)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRelaxationRK3MultiIntegrator{RealT <: Real, uType, Params,
                                                          Sol, F,
                                                          Alg,
                                                          PairedExplicitRKOptions} <:
               AbstractPairedExplicitRelaxationRKMultiIntegrator{3}
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
    # Additional PERK3 registers
    k1::uType
    kS1::uType

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

mutable struct PairedExplicitRelaxationRK3MultiParabolicIntegrator{RealT <: Real, uType,
                                                                   Params,
                                                                   Sol, F,
                                                                   Alg,
                                                                   PairedExplicitRKOptions} <:
               AbstractPairedExplicitRelaxationRKMultiParabolicIntegrator{3}
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
    # Additional PERK3 registers
    k1::uType
    kS1::uType

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
        integrator = PairedExplicitRelaxationRK3MultiParabolicIntegrator(u0, du, u_tmp,
                                                                         t0, tdir,
                                                                         dt, zero(dt),
                                                                         iter,
                                                                         ode.p,
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
                                                                         level_info_boundaries_orientation_acc,
                                                                         level_info_mortars_acc,
                                                                         level_info_mpi_mortars_acc,
                                                                         level_u_indices_elements,
                                                                         -1, n_levels,
                                                                         gamma,
                                                                         alg.relaxation_solver,
                                                                         du_tmp)
    else
        integrator = PairedExplicitRelaxationRK3MultiIntegrator(u0, du, u_tmp,
                                                                t0, tdir,
                                                                dt, zero(dt),
                                                                iter,
                                                                ode.p,
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

# TODO: step! for `PairedExplicitRelaxationRK3MultiParabolicIntegrator`

end # @muladd
