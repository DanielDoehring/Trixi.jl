# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function ComputePERK3_Multi_ButcherTableau(stages::Vector{Int64}, num_stages::Int,
                                           base_path_a_coeffs::AbstractString,
                                           cS2::Float64)

    # c Vector form Butcher Tableau (defines timestep per stage)
    c = zeros(num_stages)
    for k in 2:(num_stages - 2)
        c[k] = cS2 * (k - 1) / (num_stages - 3) # Equidistant timestep distribution (similar to PERK2)
    end

    # Own PERK based on SSPRK33
    c[num_stages - 1] = 1.0
    c[num_stages] = 0.5

    println("Timestep-split: ")
    display(c)
    println("\n")

    # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
    num_coeffs_max = num_stages - 2

    a_matrices = zeros(length(stages), num_coeffs_max, 2)
    for i in 1:length(stages)
        a_matrices[i, :, 1] = c[3:end]
    end

    # Datastructure indicating at which stage which level is evaluated
    active_levels = [Vector{Int64}() for _ in 1:num_stages]
    # k1 is evaluated at all levels
    active_levels[1] = 1:length(stages)

    # Datastructure indicating at which stage which level contributes to state
    eval_levels = [Vector{Int64}() for _ in 1:num_stages]
    # k1 is evaluated at all levels
    eval_levels[1] = 1:length(stages)
    # Second stage: Only finest method
    eval_levels[2] = [1]

    for level in eachindex(stages)
        num_stage_evals = stages[level]
        path_a_coeffs = base_path_a_coeffs * "a_" * string(num_stage_evals) * "_" *
                        string(num_stages) * ".txt"
        A = readdlm(path_a_coeffs, Float64)
        num_a_coeffs = size(A, 1)
        @assert num_a_coeffs == num_stage_evals - 2

        a_matrices[level, (num_coeffs_max - num_stage_evals + 3):end, 1] -= A
        a_matrices[level, (num_coeffs_max - num_stage_evals + 3):end, 2] = A

        # Add active levels to stages
        for stage in num_stages:-1:(num_stages - num_a_coeffs)
            push!(active_levels[stage], level)
        end

        # Add eval levels to stages
        for stage in num_stages:-1:(num_stages - num_a_coeffs + 1)
            push!(eval_levels[stage], level)
        end
    end
    max_active_levels = maximum.(active_levels)
    max_eval_levels = maximum.(eval_levels)

    for i in 1:length(stages)
        println("A-Matrix of Butcher tableau of level " * string(i))
        display(a_matrices[i, :, :])
        println()
    end

    println("Check violation of internal consistency")
    for i in 1:length(stages)
        for j in 1:i
            display(norm(a_matrices[i, :, 1] + a_matrices[i, :, 2] -
                         a_matrices[j, :, 1] -
                         a_matrices[j, :, 2], 1))
        end
    end

    println("\nActive Levels:")
    display(active_levels)
    println()
    println("\nmax_eval_levels:")
    display(max_eval_levels)
    println()

    return a_matrices, c, active_levels, max_active_levels, max_eval_levels
end

mutable struct PairedExplicitRK3Multi <: AbstractPairedExplicitRKMulti
    const num_stage_evals_min::Int64
    const num_methods::Int64
    const num_stages::Int64
    const dt_ratios::Vector{Float64}

    a_matrices::Array{Float64, 3}
    c::Vector{Float64}
    active_levels::Vector{Vector{Int64}}
    max_active_levels::Vector{Int64}
    max_eval_levels::Vector{Int64}

    function PairedExplicitRK3Multi(stages::Vector{Int64},
                                    path_a_coeffs::AbstractString,
                                    dt_ratios::Vector{Float64},
                                    cS2_::Float64 = 1.0)
        newPairedExplicitRK3Multi = new(minimum(stages),
                                        length(stages),
                                        maximum(stages),
                                        dt_ratios)

        newPairedExplicitRK3Multi.a_matrices, newPairedExplicitRK3Multi.c, newPairedExplicitRK3Multi.active_levels,
        newPairedExplicitRK3Multi.max_active_levels, newPairedExplicitRK3Multi.max_eval_levels = ComputePERK3_Multi_ButcherTableau(stages,
                                                                                                                                   newPairedExplicitRK3Multi.num_stages,
                                                                                                                                   path_a_coeffs,
                                                                                                                                   cS2_)

        return newPairedExplicitRK3Multi
    end
end # struct PairedExplicitRK3Multi

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRK3MultiIntegrator{RealT <: Real, uType, Params, Sol, F,
                                                Alg,
                                                PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKMultiIntegrator
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
    # PairedExplicitRK3Multi stages:
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
end

mutable struct PairedExplicitRK3MultiParabolicIntegrator{RealT <: Real, uType, Params,
                                                         Sol, F,
                                                         Alg,
                                                         PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKMultiParabolicIntegrator
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
    # PairedExplicitRK3Multi stages:
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

    # PairedExplicitRK3Multi stages
    k1 = zero(u0)
    k_higher = zero(u0)
    k_S1 = zero(u0)

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
        integrator = PairedExplicitRK3MultiParabolicIntegrator(u0, du, u_tmp, t0, dt,
                                                               zero(dt), iter,
                                                               ode.p,
                                                               (prob = ode,), ode.f,
                                                               alg,
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
                                                               level_u_indices_elements,
                                                               -1, n_levels,
                                                               du_tmp)
    else
        integrator = PairedExplicitRK3MultiIntegrator(u0, du, u_tmp, t0, dt, zero(dt),
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
                                                      level_u_indices_elements, -1,
                                                      n_levels)
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

function step!(integrator::PairedExplicitRK3MultiIntegrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end ||
       isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
        terminate!(integrator)
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

        for stage in 3:(alg.num_stages)
            # Loop over different methods with own associated level
            for level in 1:min(alg.num_methods, integrator.n_levels)
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                              alg.a_matrices[level, stage - 2, 1] *
                                              integrator.k1[u_ind]
                end
            end
            for level in 1:min(alg.max_eval_levels[stage], integrator.n_levels)
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] += alg.a_matrices[level, stage - 2, 2] *
                                               integrator.k_higher[u_ind]
                end
            end

            # "Remainder": Non-efficiently integrated
            for level in (alg.num_methods + 1):(integrator.n_levels)
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                              alg.a_matrices[alg.num_methods,
                                                             stage - 2, 1] *
                                              integrator.k1[u_ind]
                end
            end
            if alg.max_eval_levels[stage] == alg.num_methods
                for level in (alg.max_eval_levels[stage] + 1):(integrator.n_levels)
                    @threaded for u_ind in integrator.level_u_indices_elements[level]
                        integrator.u_tmp[u_ind] += alg.a_matrices[alg.num_methods,
                                                                  stage - 2, 2] *
                                                   integrator.k_higher[u_ind]
                    end
                end
            end

            # For statically non-uniform meshes/characteristic speeds:
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

                if stage != alg.num_stages
                    @threaded for u_ind in eachindex(integrator.du)
                        integrator.k_higher[u_ind] = integrator.du[u_ind] *
                                                     integrator.dt
                    end
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
        end

        @threaded for i in eachindex(integrator.u)
            # "Own" PairedExplicitRK based on SSPRK33.
            # Note that 'k_higher' carries the values of K_{S-1}
            # and that we construct 'K_S' "in-place" from 'integrator.du'
            integrator.u[i] += (integrator.k1[i] + integrator.k_higher[i] +
                                4.0 * integrator.du[i] * integrator.dt) / 6.0
        end
    end # PairedExplicitRK3Multi step timer

    integrator.iter += 1
    integrator.t += integrator.dt

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
