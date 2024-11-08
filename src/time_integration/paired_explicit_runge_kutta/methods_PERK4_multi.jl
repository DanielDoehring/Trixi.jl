# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function ComputePERK4_Multi_ButcherTableau(Stages::Vector{Int64}, num_stages::Int,
                                           BasePathMonCoeffs::AbstractString)

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

    AMatrices = zeros(CoeffsMax, 2, length(Stages))
    for i in 1:length(Stages)
        AMatrices[:, 1, i] = c[3:(num_stages - 3)]
    end

    # Datastructure indicating at which stage which level is evaluated
    ActiveLevels = [Vector{Int64}() for _ in 1:num_stages]
    # k1 is evaluated at all levels
    ActiveLevels[1] = 1:length(Stages)

    # Datastructure indicating at which stage which level contributes to state
    EvalLevels = [Vector{Int64}() for _ in 1:num_stages]
    # k1 is evaluated at all levels
    EvalLevels[1] = 1:length(Stages)
    # Second stage: Only finest method
    EvalLevels[2] = [1]

    for level in eachindex(Stages)
        NumStageEvals = Stages[level]

        #PathMonCoeffs = BasePathMonCoeffs * "a_" * string(NumStageEvals) * "_" * string(num_stages) * ".txt"
        # If all c = 1.0, the max number of stages does not matter
        PathMonCoeffs = BasePathMonCoeffs * "a_" * string(NumStageEvals) * ".txt"

        if NumStageEvals > 5
          @assert isfile(PathMonCoeffs) "Couldn't find file $path_a_coeffs"
          A = readdlm(PathMonCoeffs, Float64)
          NumMonCoeffs = size(A, 1)
        else
          A = []
          NumMonCoeffs = 0
        end

        if NumMonCoeffs > 0
            AMatrices[(CoeffsMax - NumMonCoeffs + 1):end, 1, level] -= A
            AMatrices[(CoeffsMax - NumMonCoeffs + 1):end, 2, level] = A
        end

        # Add active levels to stages
        for stage in num_stages:-1:(num_stages - (3 + NumMonCoeffs))
            push!(ActiveLevels[stage], level)
        end

        # Add eval levels to stages
        for stage in num_stages:-1:(num_stages - (3 + NumMonCoeffs) - 1)
            push!(EvalLevels[stage], level)
        end
    end
    # Shared matrix
    a_matrix_constant = [0.364422246578869 0.114851811257441
               0.1397682537005989 0.648906880894214
               0.1830127018922191 0.028312163512968]

    HighestActiveLevels = maximum.(ActiveLevels)
    HighestEvalLevels = maximum.(EvalLevels)

    for i in 1:length(Stages)
        println("A-Matrix of Butcher tableau of level " * string(i))
        display(AMatrices[:, :, i])
        println()
    end

    println("\nActive Levels:")
    display(ActiveLevels)
    println()
    println("\nHighestEvalLevels:")
    display(HighestEvalLevels)
    println()

    return AMatrices, a_matrix_constant, c, ActiveLevels, HighestActiveLevels, HighestEvalLevels
end

mutable struct PairedExplicitRK4Multi{StageCallbacks} <: AbstractPairedExplicitRKMulti
    const NumStageEvalsMin::Int64
    const NumMethods::Int64
    const num_stages::Int64
    const dtRatios::Vector{Float64}
    stage_callbacks::StageCallbacks

    AMatrices::Array{Float64, 3}
    a_matrix_constant::Matrix{Float64}
    c::Vector{Float64}
    ActiveLevels::Vector{Vector{Int64}}
    HighestActiveLevels::Vector{Int64}
    HighestEvalLevels::Vector{Int64}

    function PairedExplicitRK4Multi(Stages_::Vector{Int64},
                                    BasePathMonCoeffs_::AbstractString,
                                    dtRatios_,
                                    stage_callbacks = ())
        newPERK4_Multi = new{typeof(stage_callbacks)}(minimum(Stages_),
                                                      length(Stages_),
                                                      maximum(Stages_),
                                                      dtRatios_,
                                                      stage_callbacks)

        newPERK4_Multi.AMatrices, newPERK4_Multi.a_matrix_constant, newPERK4_Multi.c,
        newPERK4_Multi.ActiveLevels, newPERK4_Multi.HighestActiveLevels, newPERK4_Multi.HighestEvalLevels = ComputePERK4_Multi_ButcherTableau(Stages_,
                                                                                                                                              newPERK4_Multi.num_stages,
                                                                                                                                              BasePathMonCoeffs_)

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
    # PairedExplicitRK4Multi stages:
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

function init(ode::ODEProblem, alg::PairedExplicitRK4Multi;
              dt, callback = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    # PairedExplicitRK4Multi stages
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

    integrator = PairedExplicitRK4MultiIntegrator(u0, du, u_tmp, t0, dt, zero(dt), iter,
                                                  ode.p,
                                                  (prob = ode,), ode.f, alg,
                                                  PairedExplicitRKOptions(callback,
                                                                          ode.tspan;
                                                                          kwargs...),
                                                  false,
                                                  k1, k_higher, level_info_elements,
                                                  level_info_elements_acc,
                                                  level_info_interfaces_acc,
                                                  level_info_mpi_interfaces_acc,
                                                  level_info_boundaries_acc,
                                                  level_info_boundaries_orientation_acc,
                                                  level_info_mortars_acc,
                                                  level_info_mpi_mortars_acc,
                                                  level_u_indices_elements, -
                                                  1, n_levels)

    # initialize callbacks
    if callback isa CallbackSet
        for cb in callback.continuous_callbacks
            error("unsupported")
        end
        for cb in callback.discrete_callbacks
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    elseif !isnothing(callback)
        error("unsupported")
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
        k1!(integrator, prob.p, alg.c)

        # k2
        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[2] * integrator.dt,
                     integrator.level_info_elements_acc[1],
                     integrator.level_info_interfaces_acc[1],
                     integrator.level_info_boundaries_acc[1],
                     #integrator.level_info_boundaries_orientation_acc[1],
                     integrator.level_info_mortars_acc[1])

        # Update finest level only
        @threaded for u_ind in integrator.level_u_indices_elements[1]
            integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
        end

        for stage in 3:(alg.num_stages - 3)

            ### General implementation: Not own method for each grid level ###
            # Loop over different methods with own associated level
            for level in 1:min(alg.NumMethods, integrator.n_levels)
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                              alg.AMatrices[stage - 2, 1, level] *
                                              integrator.k1[u_ind]
                end
            end
            for level in 1:min(alg.HighestEvalLevels[stage], integrator.n_levels)
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] += alg.AMatrices[stage - 2, 2, level] *
                                               integrator.k_higher[u_ind]
                end
            end

            # "Remainder": Non-efficiently integrated
            for level in (alg.NumMethods + 1):(integrator.n_levels)
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                              alg.AMatrices[stage - 2, 1,
                                                            alg.NumMethods] *
                                              integrator.k1[u_ind]
                end
            end
            if alg.HighestEvalLevels[stage] == alg.NumMethods
                for level in (alg.HighestEvalLevels[stage] + 1):(integrator.n_levels)
                    @threaded for u_ind in integrator.level_u_indices_elements[level]
                        integrator.u_tmp[u_ind] += alg.AMatrices[stage - 2, 2,
                                                                 alg.NumMethods] *
                                                   integrator.k_higher[u_ind]
                    end
                end
            end

            ### Simplified implementation: Own method for each level ###
            #=
            for level in 1:integrator.n_levels
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] + alg.AMatrices[stage - 2, 1, level] *
                                               integrator.k1[u_ind]
                end
            end
            for level in 1:alg.HighestEvalLevels[stage]
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] += alg.AMatrices[stage - 2, 2, level] *
                                               integrator.k_higher[u_ind]
                end
            end
            =#

            #=
            ### Optimized implementation for case: Own method for each level with c[i] = 1.0, i = 2, S - 4
            for level in 1:alg.HighestEvalLevels[stage]
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] + alg.AMatrices[stage - 2, 1, level] *
                                               integrator.k1[u_ind] + 
                                               alg.AMatrices[stage - 2, 2, level] *
                                               integrator.k_higher[u_ind]
                end
            end
            for level in alg.HighestEvalLevels[stage]+1:integrator.n_levels
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] + integrator.k1[u_ind] # * A[stage, 1, level] = c[level] = 1
                end
            end
            =#

            # For statically non-uniform meshes/characteristic speeds
            #integrator.coarsest_lvl = alg.HighestActiveLevels[stage]

            # "coarsest_lvl" cannot be static for AMR, has to be checked with available levels
            integrator.coarsest_lvl = min(alg.HighestActiveLevels[stage],
                                          integrator.n_levels)

            # Check if there are fewer integrators than grid levels (non-optimal method)
            if integrator.coarsest_lvl == alg.NumMethods
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
    end # PairedExplicitRK4Multi step

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

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PairedExplicitRK4MultiIntegrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)
end
end # @muladd
