# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

#using Random # NOTE: Only for tests

function ComputePERK_Multi_ButcherTableau(NumMethods::Int, NumStages::Int,
                                          BasePathMonCoeffs::AbstractString,
                                          bS::Float64, cEnd::Float64)

    # c Vector form Butcher Tableau (defines timestep per stage)
    c = zeros(Float64, NumStages)

    for k in 2:NumStages
        c[k] = cEnd * (k - 1) / (NumStages - 1)
    end

    # CARE: Deliberately using negative timestep to have bad TV properties
    #=
    for k in 2:NumStages
      c[k] = -1 * cEnd * (k - 1)/(NumStages - 1)
    end
    c[NumStages] = cEnd
    =#
    println("Timestep-split: ")
    display(c)
    println("\n")

    SE_Factors = bS * reverse(c[2:(end - 1)])

    # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
    CoeffsMax = NumStages - 2

    AMatrices = zeros(NumMethods, CoeffsMax, 2)
    for i in 1:NumMethods
        AMatrices[i, :, 1] = c[3:end]
    end

    # Datastructure indicating at which stage which level is evaluated
    ActiveLevels = [Vector{Int64}() for _ in 1:NumStages]
    # k1 is evaluated at all levels
    ActiveLevels[1] = 1:NumMethods

    # Datastructure indicating at which stage which level contributes to state
    EvalLevels = [Vector{Int64}() for _ in 1:NumStages]
    # k1 is evaluated at all levels
    EvalLevels[1] = 1:NumStages
    # Second stage: Only finest method
    EvalLevels[2] = [1]

    for level in 1:NumMethods
        PathMonCoeffs = BasePathMonCoeffs * "gamma_" *
                        string(Int(NumStages / 2^(level - 1))) * ".txt"
        NumMonCoeffs, MonCoeffs = read_file(PathMonCoeffs, Float64)
        @assert NumMonCoeffs == NumStages / 2^(level - 1) - 2
        A = ComputeACoeffs(Int(NumStages / 2^(level - 1)), SE_Factors, MonCoeffs)

        AMatrices[level, (CoeffsMax - Int(NumStages / 2^(level - 1) - 3)):end, 1] -= A
        AMatrices[level, (CoeffsMax - Int(NumStages / 2^(level - 1) - 3)):end, 2] = A

        # Add active levels to stages
        for stage in NumStages:-1:(NumStages - NumMonCoeffs)
            push!(ActiveLevels[stage], level)
        end

        # Add eval levels to stages
        for stage in NumStages:-1:(NumStages - NumMonCoeffs + 1)
            push!(EvalLevels[stage], level)
        end
    end
    HighestActiveLevels = maximum.(ActiveLevels)
    HighestEvalLevels = maximum.(EvalLevels)

    for i in 1:NumMethods
        println("A-Matrix of Butcher tableau of level " * string(i))
        display(AMatrices[i, :, :])
        println()
    end

    println("Check violation of internal consistency")
    for i in 1:NumMethods
        for j in 1:i
            display(norm(AMatrices[i, :, 1] + AMatrices[i, :, 2] - AMatrices[j, :, 1] -
                         AMatrices[j, :, 2], 1))
        end
    end

    println("\nActive Levels:")
    display(ActiveLevels)
    println()
    println("\nHighestEvalLevels:")
    display(HighestEvalLevels)
    println()

    return AMatrices, c, ActiveLevels, HighestActiveLevels, HighestEvalLevels
end

function ComputePERK_Multi_ButcherTableau(Stages::Vector{Int64}, NumStages::Int,
                                          BasePathMonCoeffs::AbstractString,
                                          bS::Float64, cEnd::Float64)

    # c Vector form Butcher Tableau (defines timestep per stage)
    c = zeros(Float64, NumStages)

    for k in 2:NumStages
        c[k] = cEnd * (k - 1) / (NumStages - 1)
    end

    # CARE: Deliberately using negative timestep to have bad TV properties
    #=
    for k in 2:NumStages
      c[k] = -1 * cEnd * (k - 1)/(NumStages - 1)
    end
    c[NumStages] = cEnd
    =#
    println("Timestep-split: ")
    display(c)
    println("\n")

    SE_Factors = bS * reverse(c[2:(end - 1)])

    # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
    CoeffsMax = NumStages - 2

    NumMethods = length(Stages)
    AMatrices = zeros(NumMethods, CoeffsMax, 2)
    for i in 1:NumMethods
        AMatrices[i, :, 1] = c[3:end]
    end

    # Datastructure indicating at which stage which level is evaluated
    ActiveLevels = [Vector{Int64}() for _ in 1:NumStages]
    # k1 is evaluated at all levels
    ActiveLevels[1] = 1:NumMethods

    # Datastructure indicating at which stage which level contributes to state
    EvalLevels = [Vector{Int64}() for _ in 1:NumStages]
    # k1 is evaluated at all levels
    EvalLevels[1] = 1:length(Stages)
    # Second stage: Only finest method
    EvalLevels[2] = [1]

    for level in eachindex(Stages)
        NumStageEvals = Stages[level]
        PathMonCoeffs = BasePathMonCoeffs * "gamma_" * string(NumStageEvals) * ".txt"
        NumMonCoeffs, MonCoeffs = read_file(PathMonCoeffs, Float64)
        @assert NumMonCoeffs == NumStageEvals - 2
        A = ComputeACoeffs(NumStageEvals, SE_Factors, MonCoeffs)

        AMatrices[level, (CoeffsMax - (NumStageEvals - 3)):end, 1] -= A
        AMatrices[level, (CoeffsMax - (NumStageEvals - 3)):end, 2] = A

        # Add active levels to stages
        for stage in NumStages:-1:(NumStages - NumMonCoeffs)
            push!(ActiveLevels[stage], level)
        end

        # Add eval levels to stages
        for stage in NumStages:-1:(NumStages - NumMonCoeffs + 1)
            push!(EvalLevels[stage], level)
        end
    end
    HighestActiveLevels = maximum.(ActiveLevels)
    HighestEvalLevels = maximum.(EvalLevels)

    for i in 1:NumMethods
        println("A-Matrix of Butcher tableau of level " * string(i))
        display(AMatrices[i, :, :])
        println()
    end

    println("Check violation of internal consistency")
    for i in 1:NumMethods
        for j in 1:i
            display(norm(AMatrices[i, :, 1] + AMatrices[i, :, 2] - AMatrices[j, :, 1] -
                         AMatrices[j, :, 2], 1))
        end
    end

    println("\nActive Levels:")
    display(ActiveLevels)
    println()
    println("\nHighestEvalLevels:")
    display(HighestEvalLevels)
    println()

    return AMatrices, c, ActiveLevels, HighestActiveLevels, HighestEvalLevels
end

"""
    PERK_Multi()

The following structures and methods provide an implementation of
the paired explicit Runge-Kutta method (https://doi.org/10.1016/j.jcp.2019.05.014)
optimized for a certain simulation setup (PDE, IC & BC, Riemann Solver, DG Solver).
In particular, these methods are tailored to a coupling with AMR.

This is using the same interface as OrdinaryDiffEq.jl, copied from file "methods_2N.jl" for the
CarpenterKennedy2N{54, 43} methods.
"""

mutable struct PERK_Multi{StageCallbacks}
    const NumStageEvalsMin::Int64
    const NumMethods::Int64
    const NumStages::Int64
    const dtRatios::Vector{Float64}
    const b1::Float64
    const bS::Float64
    stage_callbacks::StageCallbacks

    AMatrices::Array{Float64, 3}
    c::Vector{Float64}
    ActiveLevels::Vector{Vector{Int64}}
    HighestActiveLevels::Vector{Int64}
    HighestEvalLevels::Vector{Int64}

    # TODO: Add default values for bS, cEnd
    function PERK_Multi(NumStageEvalsMin_::Int, NumMethods_::Int,
                        BasePathMonCoeffs_::AbstractString, 
                        dtRatios_, 
                        bS_::Float64, cEnd_::Float64;
                        stage_callbacks = ())
        newPERK_Multi = new{typeof(stage_callbacks)}(NumStageEvalsMin_, NumMethods_,
                                                     # Current convention: NumStages = MaxStages = S;
                                                     # TODO: Allow for different S >= Max {Stage Evals}
                                                     NumStageEvalsMin_ *
                                                     2^(NumMethods_ - 1),
                                                     dtRatios_,
                                                     1.0 - bS_, bS_,
                                                     stage_callbacks)

        newPERK_Multi.AMatrices, newPERK_Multi.c, newPERK_Multi.ActiveLevels,
        newPERK_Multi.HighestActiveLevels, newPERK_Multi.HighestEvalLevels = ComputePERK_Multi_ButcherTableau(NumMethods_,
                                                                                                              newPERK_Multi.NumStages,
                                                                                                              BasePathMonCoeffs_,
                                                                                                              bS_,
                                                                                                              cEnd_)

        return newPERK_Multi
    end

    function PERK_Multi(Stages_::Vector{Int64},
                        BasePathMonCoeffs_::AbstractString, 
                        dtRatios_,
                        bS_::Float64, cEnd_::Float64;
                        stage_callbacks = ())
        newPERK_Multi = new{typeof(stage_callbacks)}(minimum(Stages_),
                                                     length(Stages_),
                                                     maximum(Stages_),
                                                     dtRatios_,
                                                     1.0 - bS_, bS_,
                                                     stage_callbacks)

        newPERK_Multi.AMatrices, newPERK_Multi.c, newPERK_Multi.ActiveLevels,
        newPERK_Multi.HighestActiveLevels, newPERK_Multi.HighestEvalLevels = ComputePERK_Multi_ButcherTableau(Stages_,
                                                                                                              newPERK_Multi.NumStages,
                                                                                                              BasePathMonCoeffs_,
                                                                                                              bS_,
                                                                                                              cEnd_)

        return newPERK_Multi
    end
end # struct PERK_Multi

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PERK_Multi_Integrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                     PERK_IntegratorOptions}
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F
    alg::Alg # This is our own class written above; Abbreviation for ALGorithm
    opts::PERK_IntegratorOptions
    finalstep::Bool # added for convenience
    # PERK_Multi stages:
    k1::uType
    k_higher::uType
    # Variables managing level-depending integration
    level_info_elements::Vector{Vector{Int64}}
    level_info_elements_acc::Vector{Vector{Int64}}
    level_info_interfaces_acc::Vector{Vector{Int64}}
    level_info_boundaries_acc::Vector{Vector{Int64}}
    level_info_boundaries_orientation_acc::Vector{Vector{Vector{Int64}}}
    level_info_mortars_acc::Vector{Vector{Int64}}
    level_u_indices_elements::Vector{Vector{Int64}}
    t_stage::RealT
    coarsest_lvl::Int64
    n_levels::Int64
    du_ode_hyp::uType # TODO: Not best solution since this is not needed for hyperbolic problems
    AddRHSCalls::Float64
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::PERK_Multi_Integrator, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::PERK_Multi;
               dt, callback = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0) #previously: similar(u0)
    u_tmp = zero(u0)

    # PERK_Multi stages
    k1 = zero(u0)
    k_higher = zero(u0)

    du_ode_hyp = zero(u0) # TODO: Not best solution since this is not needed for hyperbolic problems

    t0 = first(ode.tspan)
    iter = 0

    ### Set datastructures for handling of level-dependent integration ###
    mesh, equations, dg, cache = mesh_equations_solver_cache(ode.p)

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

    partitioning_variables!(level_info_elements, 
                            level_info_elements_acc, 
                            level_info_interfaces_acc, 
                            level_info_boundaries_acc, 
                            level_info_boundaries_orientation_acc,
                            level_info_mortars_acc,
                            n_levels, n_dims, mesh, dg, cache, alg)

    for i in 1:n_levels
        println("#Number Elements integrated with level $i: ", length(level_info_elements[i]))
    end

    # Set initial distribution of DG Base function coefficients
    level_u_indices_elements = [Vector{Int64}() for _ in 1:n_levels]
    partitioning_u!(level_u_indices_elements, n_levels, n_dims, level_info_elements, u0, mesh, equations, dg, cache)

    ### Done with setting up for handling of level-dependent integration ###

    integrator = PERK_Multi_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                                       (prob = ode,), ode.f, alg,
                                       PERK_IntegratorOptions(callback, ode.tspan;
                                                              kwargs...), false,
                                       k1, k_higher,
                                       level_info_elements, level_info_elements_acc,
                                       level_info_interfaces_acc,
                                       level_info_boundaries_acc,
                                       level_info_boundaries_orientation_acc,
                                       level_info_mortars_acc,
                                       level_u_indices_elements,
                                       t0, -1, n_levels, du_ode_hyp, 0.0)

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

    # Start actual solve
    solve!(integrator)
end

function solve!(integrator::PERK_Multi_Integrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
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

            # k1: Evaluated on entire domain / all levels
            #integrator.f(integrator.du, integrator.u, prob.p, integrator.t, integrator.du_ode_hyp)
            integrator.f(integrator.du, integrator.u, prob.p, integrator.t)

            @threaded for i in eachindex(integrator.du)
                integrator.k1[i] = integrator.du[i] * integrator.dt
            end

            integrator.t_stage = integrator.t + alg.c[2] * integrator.dt
            # k2: Here always evaluated for finest scheme (Allow currently only max. stage evaluations)
            @threaded for i in eachindex(integrator.u)
                integrator.u_tmp[i] = integrator.u[i] + alg.c[2] * integrator.k1[i]
            end

            #=
            for stage_callback in alg.stage_callbacks
              stage_callback(integrator.u_tmp, integrator, prob.p, integrator.t_stage)
            end
            =#

            # CARE: This does not work if we have only one method but more than one grid level
            #=
            integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, 
                         integrator.level_info_elements_acc[1],
                         integrator.level_info_interfaces_acc[1],
                         integrator.level_info_boundaries_acc[1],
                         integrator.level_info_boundaries_orientation_acc[1],
                         integrator.level_info_mortars_acc[1],
                         integrator.level_u_indices_elements, 1,
                         integrator.du_ode_hyp)
            =#

            integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage,
                         integrator.level_info_elements_acc[1],
                         integrator.level_info_interfaces_acc[1],
                         integrator.level_info_boundaries_acc[1],
                         integrator.level_info_boundaries_orientation_acc[1],
                         integrator.level_info_mortars_acc[1])

            @threaded for u_ind in integrator.level_u_indices_elements[1] # Update finest level
                integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
            end

            for stage in 3:(alg.NumStages)
                # Construct current state
                @threaded for i in eachindex(integrator.u)
                    integrator.u_tmp[i] = integrator.u[i]
                end

                # Loop over different methods with own associated level
                for level in 1:min(alg.NumMethods, integrator.n_levels)
                    @threaded for u_ind in integrator.level_u_indices_elements[level]
                        integrator.u_tmp[u_ind] += alg.AMatrices[level, stage - 2, 1] *
                                                   integrator.k1[u_ind]
                    end
                end
                for level in 1:min(alg.HighestEvalLevels[stage], integrator.n_levels)
                    @threaded for u_ind in integrator.level_u_indices_elements[level]
                        integrator.u_tmp[u_ind] += alg.AMatrices[level, stage - 2, 2] *
                                                   integrator.k_higher[u_ind]
                    end
                end

                # "Remainder": Non-efficiently integrated
                for level in (alg.NumMethods + 1):(integrator.n_levels)
                    @threaded for u_ind in integrator.level_u_indices_elements[level]
                        integrator.u_tmp[u_ind] += alg.AMatrices[alg.NumMethods,
                                                                 stage - 2, 1] *
                                                   integrator.k1[u_ind]
                    end
                end
                if alg.HighestEvalLevels[stage] == alg.NumMethods
                    for level in (alg.HighestEvalLevels[stage] + 1):(integrator.n_levels)
                        @threaded for u_ind in integrator.level_u_indices_elements[level]
                            integrator.u_tmp[u_ind] += alg.AMatrices[alg.NumMethods,
                                                                     stage - 2, 2] *
                                                       integrator.k_higher[u_ind]
                        end
                    end
                end

                integrator.t_stage = integrator.t + alg.c[stage] * integrator.dt

                # For statically non-uniform meshes/characteristic speeds:
                #integrator.coarsest_lvl = alg.HighestActiveLevels[stage]

                # "coarsest_lvl" cannot be static for AMR, has to be checked with available levels
                integrator.coarsest_lvl = min(alg.HighestActiveLevels[stage], integrator.n_levels)

                # Check if there are fewer integrators than grid levels (non-optimal method)
                if integrator.coarsest_lvl == alg.NumMethods
                    # NOTE: This is supposedly more efficient than setting
                    #integrator.coarsest_lvl = integrator.n_levels
                    # and then using the level-dependent version

                    #integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, integrator.du_ode_hyp)
                    integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)

                    @threaded for u_ind in eachindex(integrator.du)
                        integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
                    end
                else
                    #=
                    # Joint RHS evaluation with all elements sharing this timestep
                    integrator.f(integrator.du, integrator.u_tmp, prob.p,
                                integrator.t_stage,
                                integrator.level_info_elements_acc[integrator.coarsest_lvl],
                                integrator.level_info_interfaces_acc[integrator.coarsest_lvl],
                                integrator.level_info_boundaries_acc[integrator.coarsest_lvl],
                                integrator.level_info_boundaries_orientation_acc[integrator.coarsest_lvl],
                                integrator.level_info_mortars_acc[integrator.coarsest_lvl],
                                integrator.level_u_indices_elements,
                                integrator.coarsest_lvl,
                                integrator.du_ode_hyp)
                    =#

                    
                    integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, 
                                integrator.level_info_elements_acc[integrator.coarsest_lvl],
                                integrator.level_info_interfaces_acc[integrator.coarsest_lvl],
                                integrator.level_info_boundaries_acc[integrator.coarsest_lvl],
                                integrator.level_info_boundaries_orientation_acc[integrator.coarsest_lvl],
                                integrator.level_info_mortars_acc[integrator.coarsest_lvl],
                                integrator.coarsest_lvl)
                    

                    # Update k_higher of relevant levels
                    for level in 1:(integrator.coarsest_lvl)
                        @threaded for u_ind in integrator.level_u_indices_elements[level]
                            integrator.k_higher[u_ind] = integrator.du[u_ind] *
                                                        integrator.dt
                        end
                    end
                end
            end

            # u_{n+1} = u_n + b_S * k_S = u_n + 1 * k_S
            @threaded for i in eachindex(integrator.u)
                integrator.u[i] += alg.b1 * integrator.k1[i] +
                                   alg.bS * integrator.k_higher[i]
                # Slightly more performant, hard-coded version for b1 = 0
                #integrator.u[i] += integrator.k_higher[i]
            end

            #=
            for stage_callback in alg.stage_callbacks
              stage_callback(integrator.u, integrator, prob.p, integrator.t_stage)
            end
            =#
        end # PERK_Multi step

        integrator.iter += 1
        integrator.t += integrator.dt

        # handle callbacks
        if callbacks isa CallbackSet
            for cb in callbacks.discrete_callbacks
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
            end
        end

        #=
        for stage_callback in alg.stage_callbacks
          stage_callback(integrator.u, integrator, prob.p, integrator.t_stage)
        end
        =#

        # respect maximum number of iterations
        if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
            @warn "Interrupted. Larger maxiters is needed."
            terminate!(integrator)
        end
    end # "main loop" timer

    println("Additional RHS Calls: ", integrator.AddRHSCalls)

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

# get a cache where the RHS can be stored
get_du(integrator::PERK_Multi_Integrator) = integrator.du
get_tmp_cache(integrator::PERK_Multi_Integrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::PERK_Multi_Integrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::PERK_Multi_Integrator, dt)
    integrator.dt = dt
end

function get_proposed_dt(integrator::PERK_Multi_Integrator)
    return integrator.dt
end

# stop the time integration
function terminate!(integrator::PERK_Multi_Integrator)
    integrator.finalstep = true
    empty!(integrator.opts.tstops)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PERK_Multi_Integrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)

    # TODO: Move this into parabolic cache or similar
    resize!(integrator.du_ode_hyp, new_size)
end
end # @muladd
