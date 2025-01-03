# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function ComputePERK4_Multi_ButcherTableau(stages::Vector{Int64}, num_stages::Int,
                                           base_path_a_coeffs::AbstractString,
                                           cS3)
    c = PERK4_compute_c_coeffs(num_stages, cS3)

    # For the p = 4 method there are less free coefficients
    num_coeffs_max = num_stages - 5

    num_methods = length(stages)
    a_matrices = zeros(num_methods, 2, num_coeffs_max)
    for i in 1:num_methods
        a_matrices[i, 1, :] = c[3:(num_stages - 3)]
    end

    # Datastructure indicating at which stage which level is evaluated
    active_levels = [Vector{Int64}() for _ in 1:num_stages]
    # k1 is evaluated at all levels
    active_levels[1] = 1:num_methods

    # Datastructure indicating at which stage which level contributes to state
    eval_levels = [Vector{Int64}() for _ in 1:num_stages]
    # k1 is evaluated at all levels
    eval_levels[1] = 1:num_methods
    # Second stage: Only finest method
    eval_levels[2] = [1]

    for level in eachindex(stages)
        num_stage_evals = stages[level]

        #path_a_coeffs = base_path_a_coeffs * "a_" * string(num_stage_evals) * "_" * string(num_stages) * ".txt"
        # If all c = 1.0, the max number of stages does not matter
        path_a_coeffs = base_path_a_coeffs * "a_" * string(num_stage_evals) * ".txt"

        if num_stage_evals > 5
            @assert isfile(path_a_coeffs) "Couldn't find file $path_a_coeffs"
            A = readdlm(path_a_coeffs, Float64)
            num_a_coeffs = size(A, 1)
            @assert num_a_coeffs == num_stage_evals - 5
        else
            A = []
            num_a_coeffs = 0
        end

        if num_a_coeffs > 0
            a_matrices[level, 1, (num_coeffs_max - num_a_coeffs + 1):end] -= A
            a_matrices[level, 2, (num_coeffs_max - num_a_coeffs + 1):end] = A
        end

        # Add active levels to stages
        for stage in num_stages:-1:(num_stages - (3 + num_a_coeffs))
            push!(active_levels[stage], level)
        end

        # Add eval levels to stages
        for stage in num_stages:-1:(num_stages - (3 + num_a_coeffs) - 1)
            push!(eval_levels[stage], level)
        end
    end
    # Shared matrix
    a_matrix_constant = PERK4_a_matrix_constant(cS3)

    max_active_levels = maximum.(active_levels)
    max_eval_levels = maximum.(eval_levels)

    return a_matrices, a_matrix_constant, c, active_levels, max_active_levels,
           max_eval_levels
end

struct PairedExplicitRK4Multi <: AbstractPairedExplicitRKMulti
    num_stage_evals_min::Int64
    num_methods::Int64
    num_stages::Int64

    dt_ratios::Vector{Float64}

    a_matrices::Array{Float64, 3}
    a_matrix_constant::Matrix{Float64}
    c::Vector{Float64}

    active_levels::Vector{Vector{Int64}}
    max_active_levels::Vector{Int64}
    max_eval_levels::Vector{Int64}
end

function PairedExplicitRK4Multi(stages::Vector{Int64},
                                base_path_a_coeffs::AbstractString,
                                dt_ratios;
                                cS3 = 1.0f0)
    num_stages = maximum(stages)

    a_matrices,
    a_matrix_constant, c,
    active_levels,
    max_active_levels,
    max_eval_levels = ComputePERK4_Multi_ButcherTableau(stages, num_stages,
                                                        base_path_a_coeffs,
                                                        cS3)

    return PairedExplicitRK4Multi(minimum(stages), length(stages), num_stages,
                                  dt_ratios,
                                  a_matrices, a_matrix_constant, c, active_levels,
                                  max_active_levels, max_eval_levels)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRK4MultiIntegrator{RealT <: Real, uType, Params, Sol, F,
                                                Alg,
                                                PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKMultiIntegrator{4}
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
    # Additional PERK register
    k1::uType

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

mutable struct PairedExplicitRK4MultiParabolicIntegrator{RealT <: Real, uType, Params,
                                                         Sol, F,
                                                         Alg,
                                                         PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKMultiParabolicIntegrator{4}
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
    # Additional PERK register
    k1::uType

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

function init(ode::ODEProblem, alg::PairedExplicitRK4Multi;
              dt, callback = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    k1 = zero(u0) # Additional PERK register

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
        integrator = PairedExplicitRK4MultiParabolicIntegrator(u0, du, u_tmp, t0, tdir,
                                                               dt, zero(dt), iter,
                                                               ode.p, (prob = ode,),
                                                               ode.f, alg,
                                                               PairedExplicitRKOptions(callback,
                                                                                       ode.tspan;
                                                                                       kwargs...),
                                                               false, true, false,
                                                               k1,
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
    else
        integrator = PairedExplicitRK4MultiIntegrator(u0, du, u_tmp, t0, tdir,
                                                      dt, zero(dt), iter,
                                                      ode.p, (prob = ode,),
                                                      ode.f, alg,
                                                      PairedExplicitRKOptions(callback,
                                                                              ode.tspan;
                                                                              kwargs...),
                                                      false, true, false,
                                                      k1,
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

@inline function PERKMulti_intermediate_stage!(integrator::Union{AbstractPairedExplicitRKMultiIntegrator{4},
                                                                 AbstractPairedExplicitRelaxationRKMultiIntegrator{4}},
                                               alg, stage)
    #=                                               
    ### General implementation: Not own method for each grid level ###
    # Loop over different methods with own associated level
    for level in 1:min(alg.num_methods, integrator.n_levels)
        @threaded for i in integrator.level_u_indices_elements[level]
            integrator.u_tmp[i] = integrator.u[i] +
                                      integrator.dt *
                                      alg.a_matrices[level, 1, stage - 2] *
                                      integrator.k1[i]
        end
    end
    for level in 1:min(alg.max_eval_levels[stage], integrator.n_levels)
        @threaded for i in integrator.level_u_indices_elements[level]
            integrator.u_tmp[i] += integrator.dt *
                                       alg.a_matrices[level, 2, stage - 2] *
                                       integrator.du[i]
        end
    end

    # "Remainder": Non-efficiently integrated
    for level in (alg.num_methods + 1):(integrator.n_levels)
        @threaded for i in integrator.level_u_indices_elements[level]
            integrator.u_tmp[i] = integrator.u[i] +
                                      integrator.dt *
                                      alg.a_matrices[alg.num_methods, 1, stage - 2] *
                                      integrator.k1[i]
        end
    end
    if alg.max_eval_levels[stage] == alg.num_methods
        for level in (alg.max_eval_levels[stage] + 1):integrator.n_levels
            @threaded for i in integrator.level_u_indices_elements[level]
                integrator.u_tmp[i] += integrator.dt *
                                           alg.a_matrices[alg.num_methods, 2,
                                                          stage - 2] *
                                           integrator.du[i]
            end
        end
    end
    =#

    ### Optimized implementation for PERK4 case: Own method for each level with c[i] = 1.0, i = 2, S - 4 ###
    for level in 1:alg.max_eval_levels[stage]
        @threaded for i in integrator.level_u_indices_elements[level]
            integrator.u_tmp[i] = integrator.u[i] +
                                  integrator.dt *
                                  (alg.a_matrices[level, 1, stage - 2] *
                                   integrator.k1[i] +
                                   alg.a_matrices[level, 2, stage - 2] *
                                   integrator.du[i])
        end
    end
    for level in (alg.max_eval_levels[stage] + 1):(integrator.n_levels)
        @threaded for i in integrator.level_u_indices_elements[level]
            integrator.u_tmp[i] = integrator.u[i] +
                                  integrator.dt * integrator.k1[i] # * A[stage, 1, level] = c[level] = 1
        end
    end

    # For statically non-uniform meshes/characteristic speeds
    #integrator.coarsest_lvl = alg.max_active_levels[stage]

    # "coarsest_lvl" cannot be static for AMR, has to be checked with available levels
    integrator.coarsest_lvl = min(alg.max_active_levels[stage],
                                  integrator.n_levels)
end

# Computes last three stages, i.e., i = S-2, S-1, S
@inline function PERK4_kS2_to_kS!(integrator::PairedExplicitRK4MultiParabolicIntegrator,
                                  p, alg)
    for stage in 1:2
        @threaded for i in eachindex(integrator.u)
            integrator.u_tmp[i] = integrator.u[i] +
                                  integrator.dt *
                                  (alg.a_matrix_constant[1, stage] *
                                   integrator.k1[i] +
                                   alg.a_matrix_constant[2, stage] *
                                   integrator.du[i])
        end

        integrator.f(integrator.du, integrator.u_tmp, p,
                     integrator.t +
                     alg.c[alg.num_stages - 3 + stage] * integrator.dt,
                     integrator)
    end

    # Last stage
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              integrator.dt *
                              (alg.a_matrix_constant[1, 3] * integrator.k1[i] +
                               alg.a_matrix_constant[2, 3] * integrator.du[i])
    end

    # Safe K_{S-1} in `k1`:
    @threaded for i in eachindex(integrator.u)
        integrator.k1[i] = integrator.du[i]
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[alg.num_stages] * integrator.dt,
                 integrator)

    @threaded for i in eachindex(integrator.u)
        # Note that 'k1' carries the values of K_{S-1}
        # and that we construct 'K_S' "in-place" from 'integrator.du'
        integrator.u[i] += 0.5 * integrator.dt *
                           (integrator.k1[i] + integrator.du[i])
    end
end
end # @muladd
