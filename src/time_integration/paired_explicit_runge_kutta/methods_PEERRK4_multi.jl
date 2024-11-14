# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct PairedExplicitERRK4Multi <: AbstractPairedExplicitRKMulti
    const num_stage_evals_min::Int64
    const num_methods::Int64
    const num_stages::Int64
    const dt_ratios::Vector{Float64}

    a_matrices::Array{Float64, 3}
    a_matrix_constant::Matrix{Float64}
    c::Vector{Float64}
    active_levels::Vector{Vector{Int64}}
    max_active_levels::Vector{Int64}
    max_eval_levels::Vector{Int64}

    function PairedExplicitERRK4Multi(stages::Vector{Int64},
                                      base_path_a_coeffs::AbstractString,
                                      dt_ratios)
        newPERK4_Multi = new(minimum(stages),
                             length(stages),
                             maximum(stages),
                             dt_ratios)

        newPERK4_Multi.a_matrices, newPERK4_Multi.a_matrix_constant, newPERK4_Multi.c,
        newPERK4_Multi.active_levels, newPERK4_Multi.max_active_levels, newPERK4_Multi.max_eval_levels = ComputePERK4_Multi_ButcherTableau(stages,
                                                                                                                                           newPERK4_Multi.num_stages,
                                                                                                                                           base_path_a_coeffs)

        return newPERK4_Multi
    end
end # struct PairedExplicitERRK4Multi

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitERRK4MultiIntegrator{RealT <: Real, uType, Params, Sol, F,
                                                  Alg,
                                                  PairedExplicitRKOptions} <:
               AbstractPairedExplicitERRKMultiIntegrator
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # Used for euler-acoustic coupling
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F
    alg::Alg # This is our own class written above; Abbreviation for ALGorithm
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    # PairedExplicitERRK4Multi stages:
    k1::uType
    k_higher::uType

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
    direction::uType
    gamma::RealT
end

function init(ode::ODEProblem, alg::PairedExplicitERRK4Multi;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    # PairedExplicitERRK4Multi stages
    k1 = zero(u0)
    k_higher = zero(u0)

    t0 = first(ode.tspan)
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
    direction = zero(u0)
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

    integrator = PairedExplicitERRK4MultiIntegrator(u0, du, u_tmp, t0, dt, zero(dt),
                                                    iter,
                                                    ode.p,
                                                    (prob = ode,), ode.f, alg,
                                                    PairedExplicitRKOptions(callback,
                                                                            ode.tspan;
                                                                            kwargs...),
                                                    false,
                                                    k1, k_higher,
                                                    level_info_elements,
                                                    level_info_elements_acc,
                                                    level_info_interfaces_acc,
                                                    level_info_mpi_interfaces_acc,
                                                    level_info_boundaries_acc,
                                                    level_info_boundaries_orientation_acc,
                                                    level_info_mortars_acc,
                                                    level_info_mpi_mortars_acc,
                                                    level_u_indices_elements, -
                                                    1, n_levels,
                                                    direction, gamma)

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

function step!(integrator::PairedExplicitERRK4MultiIntegrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    callbacks = integrator.opts.callback

    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
        k1!(integrator, prob.p, alg.c)

        # k2: Only evaluated at finest level
        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[2] * integrator.dt,
                     integrator.level_info_elements_acc[1],
                     integrator.level_info_interfaces_acc[1],
                     integrator.level_info_boundaries_acc[1],
                     #integrator.level_info_boundaries_orientation_acc[1],
                     integrator.level_info_mortars_acc[1])

        @threaded for u_ind in integrator.level_u_indices_elements[1]
            integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
        end

        for stage in 3:(alg.num_stages - 3)

            ### General implementation: Not own method for each grid level ###
            # Loop over different methods with own associated level
            for level in 1:min(alg.num_methods, integrator.n_levels)
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                              alg.a_matrices[stage - 2, 1, level] *
                                              integrator.k1[u_ind]
                end
            end
            for level in 1:min(alg.max_eval_levels[stage], integrator.n_levels)
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] += alg.a_matrices[stage - 2, 2, level] *
                                               integrator.k_higher[u_ind]
                end
            end

            # "Remainder": Non-efficiently integrated
            for level in (alg.num_methods + 1):(integrator.n_levels)
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                              alg.a_matrices[stage - 2, 1,
                                                             alg.num_methods] *
                                              integrator.k1[u_ind]
                end
            end
            if alg.max_eval_levels[stage] == alg.num_methods
                for level in (alg.max_eval_levels[stage] + 1):(integrator.n_levels)
                    @threaded for u_ind in integrator.level_u_indices_elements[level]
                        integrator.u_tmp[u_ind] += alg.a_matrices[stage - 2, 2,
                                                                  alg.num_methods] *
                                                   integrator.k_higher[u_ind]
                    end
                end
            end

            ### Simplified implementation: Own method for each level ###
            #=
            for level in 1:integrator.n_levels
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] + alg.a_matrices[stage - 2, 1, level] *
                                               integrator.k1[u_ind]
                end
            end
            for level in 1:alg.max_eval_levels[stage]
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] += alg.a_matrices[stage - 2, 2, level] *
                                               integrator.k_higher[u_ind]
                end
            end
            =#

            #=
            ### Optimized implementation for case: Own method for each level with c[i] = 1.0, i = 2, S - 4
            for level in 1:alg.max_eval_levels[stage]
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] + alg.a_matrices[stage - 2, 1, level] *
                                               integrator.k1[u_ind] + 
                                               alg.a_matrices[stage - 2, 2, level] *
                                               integrator.k_higher[u_ind]
                end
            end
            for level in alg.max_eval_levels[stage]+1:integrator.n_levels
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] + integrator.k1[u_ind] # * A[stage, 1, level] = c[level] = 1
                end
            end
            =#

            # For statically non-uniform meshes/characteristic speeds
            #integrator.coarsest_lvl = alg.max_active_levels[stage]

            # "coarsest_lvl" cannot be static for AMR, has to be checked with available levels
            integrator.coarsest_lvl = min(alg.max_active_levels[stage],
                                          integrator.n_levels)

            # Check if there are fewer integrators than grid levels (non-optimal method)
            if integrator.coarsest_lvl == alg.num_methods
                # NOTE: This is supposedly more efficient than setting
                #integrator.coarsest_lvl = integrator.n_levels
                # and then using the level-dependent version

                integrator.f(integrator.du, integrator.u_tmp, prob.p,
                             integrator.t + alg.c[stage] * integrator.dt)

                @threaded for u_ind in eachindex(integrator.du)
                    integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
                end
            else
                integrator.f(integrator.du, integrator.u_tmp, prob.p,
                             integrator.t + alg.c[stage] * integrator.dt,
                             integrator.level_info_elements_acc[integrator.coarsest_lvl],
                             integrator.level_info_interfaces_acc[integrator.coarsest_lvl],
                             integrator.level_info_boundaries_acc[integrator.coarsest_lvl],
                             #integrator.level_info_boundaries_orientation_acc[integrator.coarsest_lvl],
                             integrator.level_info_mortars_acc[integrator.coarsest_lvl])

                # Update k_higher of relevant levels
                for level in 1:(integrator.coarsest_lvl)
                    @threaded for u_ind in integrator.level_u_indices_elements[level]
                        integrator.k_higher[u_ind] = integrator.du[u_ind] *
                                                     integrator.dt
                    end
                end
            end
        end # end loop over different stages

        last_three_stages!(integrator, alg, prob.p)
    end # PairedExplicitERRK4Multi step

    @trixi_timeit timer() "Step-Callbacks" begin
        # handle callbacks
        if callbacks isa CallbackSet
            foreach(callbacks.discrete_callbacks) do cb
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
                return nothing
            end
        end
    end

    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
        @warn "Interrupted. Larger maxiters is needed."
        terminate!(integrator)
    end
end
end # @muladd
