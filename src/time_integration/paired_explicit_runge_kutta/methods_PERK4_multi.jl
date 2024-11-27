# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function ComputePERK4_Multi_ButcherTableau(stages::Vector{Int64}, num_stages::Int,
                                           base_path_a_coeffs::AbstractString)

    # Current approach: Use ones (best internal stability properties)
    c = ones(num_stages)
    c[1] = 0.0

    c[num_stages - 3] = 1.0
    c[num_stages - 2] = 0.479274057836310
    c[num_stages - 1] = sqrt(3) / 6 + 0.5
    c[num_stages] = -sqrt(3) / 6 + 0.5

    println("Timestep-split: ")
    display(c)
    println("\n")

    # For the p = 4 method there are less free coefficients
    CoeffsMax = num_stages - 5

    a_matrices = zeros(CoeffsMax, 2, length(stages))
    for i in 1:length(stages)
        a_matrices[:, 1, i] = c[3:(num_stages - 3)]
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

        #path_a_coeffs = base_path_a_coeffs * "a_" * string(num_stage_evals) * "_" * string(num_stages) * ".txt"
        # If all c = 1.0, the max number of stages does not matter
        path_a_coeffs = base_path_a_coeffs * "a_" * string(num_stage_evals) * ".txt"

        if num_stage_evals > 5
            @assert isfile(path_a_coeffs) "Couldn't find file $path_a_coeffs"
            A = readdlm(path_a_coeffs, Float64)
            num_a_coeffs = size(A, 1)
        else
            A = []
            num_a_coeffs = 0
        end

        if num_a_coeffs > 0
            a_matrices[(CoeffsMax - num_a_coeffs + 1):end, 1, level] -= A
            a_matrices[(CoeffsMax - num_a_coeffs + 1):end, 2, level] = A
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
    a_matrix_constant = [0.364422246578869 0.114851811257441
                         0.1397682537005989 0.648906880894214
                         0.1830127018922191 0.028312163512968]

    max_active_levels = maximum.(active_levels)
    max_eval_levels = maximum.(eval_levels)

    for i in 1:length(stages)
        println("A-Matrix of Butcher tableau of level " * string(i))
        display(a_matrices[:, :, i])
        println()
    end

    println("\nActive Levels:")
    display(active_levels)
    println()
    println("\nmax_eval_levels:")
    display(max_eval_levels)
    println()

    return a_matrices, a_matrix_constant, c, active_levels, max_active_levels,
           max_eval_levels
end

mutable struct PairedExplicitRK4Multi <: AbstractPairedExplicitRKMulti
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

    function PairedExplicitRK4Multi(stages::Vector{Int64},
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
end # struct PairedExplicitRK4Multi

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRK4MultiIntegrator{RealT <: Real, uType, Params, Sol, F,
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
    # Additional PERK stage
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
    # Additional PERK stage
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

    k1 = zero(u0) # Additional PERK stage

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
        integrator = PairedExplicitRK4MultiParabolicIntegrator(u0, du, u_tmp, t0, dt,
                                                               zero(dt), iter,
                                                               ode.p,
                                                               (prob = ode,), ode.f,
                                                               alg,
                                                               PairedExplicitRKOptions(callback,
                                                                                       ode.tspan;
                                                                                       kwargs...),
                                                               false,
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
        integrator = PairedExplicitRK4MultiIntegrator(u0, du, u_tmp, t0, dt, zero(dt),
                                                      iter,
                                                      ode.p,
                                                      (prob = ode,), ode.f, alg,
                                                      PairedExplicitRKOptions(callback,
                                                                              ode.tspan;
                                                                              kwargs...),
                                                      false,
                                                      k1,
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

function step!(integrator::PairedExplicitRK4MultiIntegrator)
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
        PERK_k1!(integrator, prob.p)
        PERK_k2!(integrator, prob.p, alg)

        for stage in 3:(alg.num_stages - 3)
            PERK_ki!(integrator, prob.p, alg, stage)
        end

        last_three_stages!(integrator, prob.p, alg)
    end

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
