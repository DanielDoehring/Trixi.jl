# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function compute_PairedExplicitRK3Multi_butcher_tableau(stages::Vector{Int64},
                                                        num_stages::Int,
                                                        base_path_a_coeffs::AbstractString,
                                                        cS2::Float64)

    # c Vector form Butcher Tableau (defines timestep per stage)
    c = PERK3_compute_c_coeffs(num_stages, cS2)

    # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
    num_coeffs_max = num_stages - 2

    num_methods = length(stages)
    a_matrices = zeros(num_methods, 2, num_coeffs_max)
    for i in 1:num_methods
        a_matrices[i, 1, :] = c[3:end]
    end

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

        if num_stage_evals > 3
            path_a_coeffs = base_path_a_coeffs * "a_" * string(num_stage_evals) * "_" *
                            string(num_stages) * ".txt"

            @assert isfile(path_a_coeffs) "Couldn't find file $path_a_coeffs"
            A = readdlm(path_a_coeffs, Float64)
            num_a_coeffs = size(A, 1)
            @assert num_a_coeffs == num_stage_evals - 2

            a_matrices[level, 1, (num_coeffs_max - num_stage_evals + 3):end] -= A
            a_matrices[level, 2, (num_coeffs_max - num_stage_evals + 3):end] = A
        else
            num_a_coeffs = 0
        end

        # Add active levels to stages
        for stage in num_stages:-1:(num_stages - num_a_coeffs)
            push!(active_levels[stage], level)
        end

        # Push contributing (added) levels to stages
        for stage in num_stages:-1:(num_stages - num_a_coeffs + 1)
            push!(add_levels[stage], level)
        end
    end
    max_active_levels = maximum.(active_levels)
    max_add_levels = maximum.(add_levels)

    return a_matrices, c, active_levels, max_active_levels, max_add_levels
end

struct PairedExplicitRK3Multi <:
       AbstractPairedExplicitRKMulti{3}
    num_stage_evals_min::Int64
    num_methods::Int64
    num_stages::Int64
    stages::Vector{Int64}

    dt_ratios::Vector{Float64}

    a_matrices::Array{Float64, 3}
    c::Vector{Float64}

    active_levels::Vector{Vector{Int64}}
    max_active_levels::Vector{Int64}
    max_add_levels::Vector{Int64}
end

# Constructor for previously computed A Coeffs
function PairedExplicitRK3Multi(stages::Vector{Int64},
                                base_path_a_coeffs::AbstractString,
                                dt_ratios;
                                cS2::Float64 = 1.0)
    num_stages = maximum(stages)

    a_matrices, c,
    active_levels,
    max_active_levels,
    max_add_levels = compute_PairedExplicitRK3Multi_butcher_tableau(stages,
                                                                    num_stages,
                                                                    base_path_a_coeffs,
                                                                    cS2)

    return PairedExplicitRK3Multi(minimum(stages), length(stages), num_stages, stages,
                                  dt_ratios,
                                  a_matrices, c, active_levels,
                                  max_active_levels, max_add_levels)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRK3MultiIntegrator{RealT <: Real, uType,
                                                Params, Sol, F,
                                                PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKMultiIntegrator{3}
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    tdir::RealT
    dt::RealT # current time step
    dtcache::RealT # Used for euler-acoustic coupling
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F
    alg::PairedExplicitRK3Multi
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
end

mutable struct PairedExplicitRK3MultiParabolicIntegrator{RealT <: Real, uType,
                                                         Params, Sol, F,
                                                         PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKMultiParabolicIntegrator{3}
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    tdir::RealT
    dt::RealT # current time step
    dtcache::RealT # Used for euler-acoustic coupling
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F
    alg::PairedExplicitRK3Multi
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

    # Addition for hyperbolic-parabolic problems:
    # We need another register to temporarily store the changes due to the hyperbolic part only.
    # The changes due to the parabolic part are stored in the usual `du` register.
    du_tmp::uType
end

function init(ode::ODEProblem, alg::PairedExplicitRK3Multi;
              dt, callback = nothing, kwargs...)
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

    n_levels = get_n_levels(mesh, alg)
    n_dims = ndims(mesh) # Spatial dimension

    level_info_elements = [Vector{Int64}() for _ in 1:n_levels]
    level_info_elements_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_boundaries_acc = [Vector{Int64}() for _ in 1:n_levels]
    level_info_boundaries_orientation_acc = [[Vector{Int64}()
                                              for _ in 1:(2 * n_dims)]
                                             for _ in 1:n_levels]

    level_info_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]

    # MPI additions
    level_info_mpi_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]
    level_info_mpi_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]

    if !mpi_isparallel()
        partition_variables!(level_info_elements,
                             level_info_elements_acc,
                             level_info_interfaces_acc,
                             level_info_boundaries_acc,
                             level_info_boundaries_orientation_acc,
                             level_info_mortars_acc,
                             n_levels, n_dims, mesh, dg, cache, alg)
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
            partition_variables!(level_info_elements, n_levels, n_dims,
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
            resize!(kS1, n_new)
        end
        partition_variables!(level_info_elements,
                             level_info_elements_acc,
                             level_info_interfaces_acc,
                             level_info_boundaries_acc,
                             level_info_boundaries_orientation_acc,
                             level_info_mortars_acc,
                             # MPI additions
                             level_info_mpi_interfaces_acc,
                             level_info_mpi_mortars_acc,
                             n_levels, n_dims, mesh, dg, cache, alg)
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
    if isa(semi, SemidiscretizationHyperbolicParabolic)
        du_tmp = zero(u0)
        integrator = PairedExplicitRK3MultiParabolicIntegrator(u0, du, u_tmp,
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
                                                               du_tmp)
    else # Purely hyperbolic, Euler-Gravity, ...
        integrator = PairedExplicitRK3MultiIntegrator(u0, du, u_tmp,
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
                                                      -1, n_levels)

        if :semi_gravity in fieldnames(typeof(semi))
            partition_u_gravity!(integrator)
        end
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

function Base.resize!(integrator::AbstractPairedExplicitRKMultiParabolicIntegrator{3},
                      new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    # PERK stages
    resize!(integrator.k1, new_size)
    resize!(integrator.kS1, new_size)
    # Addition for multirate PERK methods for parabolic problems
    resize!(integrator.du_tmp, new_size)
end
end # @muladd
