# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

using SciMLBase: SplitFunction

# Define all of the functions necessary for polynomial optimizations
include("polynomial_optimizer.jl")

# Abstract base type for both single/standalone and multi-level 
# PERK (Paired Explicit Runge-Kutta) time integration schemes
abstract type AbstractPairedExplicitRK{ORDER} <: AbstractTimeIntegrationAlgorithm end

# Abstract base type for single/standalone PERK time integration schemes
abstract type AbstractPairedExplicitRKSingle{ORDER} <: AbstractPairedExplicitRK{ORDER} end
# Abstract base type for multirate PERK time integration schemes
abstract type AbstractPairedExplicitRKMulti{ORDER} <: AbstractPairedExplicitRK{ORDER} end

# Split algorithms: Different Butcher tableaus for the implicit and explicit parts,
# targeting hyperbolic-parabolic problems
# Abstract base type for single/standalone PERK split-problem time integration schemes
abstract type AbstractPairedExplicitRKSplitSingle{ORDER} <:
              AbstractPairedExplicitRKSingle{ORDER} end
# Abstract base type for multirate PERK split-problem time integration schemes
abstract type AbstractPairedExplicitRKSplitMulti{ORDER} <:
              AbstractPairedExplicitRKMulti{ORDER} end

abstract type AbstractPairedExplicitRKIMEX{ORDER} <: AbstractPairedExplicitRK{ORDER} end

abstract type AbstractPairedExplicitRKIMEXSingle{ORDER} <:
              AbstractPairedExplicitRKIMEX{ORDER} end

abstract type AbstractPairedExplicitRKIMEXMulti{ORDER} <:
              AbstractPairedExplicitRKIMEX{ORDER} end

# TODO: Split IMEX

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct PairedExplicitRKOptions{Callback, TStops}
    callback::Callback # callbacks; used in Trixi
    adaptive::Bool # whether the algorithm is adaptive (false)
    dtmax::Float64 # ignored
    maxiters::Int # maximal number of time steps
    tstops::TStops # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function PairedExplicitRKOptions(callback, tspan; maxiters = typemax(Int), kwargs...)
    tstops_internal = BinaryHeap{eltype(tspan)}(FasterForward())
    # We add last(tspan) to make sure that the time integration stops at the end time
    push!(tstops_internal, last(tspan))
    # We add 2 * last(tspan) because add_tstop!(integrator, t) is only called by DiffEqCallbacks.jl if tstops contains a time that is larger than t
    # (https://github.com/SciML/DiffEqCallbacks.jl/blob/025dfe99029bd0f30a2e027582744528eb92cd24/src/iterative_and_periodic.jl#L92)
    push!(tstops_internal, 2 * last(tspan))
    PairedExplicitRKOptions{typeof(callback), typeof(tstops_internal)}(callback,
                                                                       false, Inf,
                                                                       maxiters,
                                                                       tstops_internal)
end

abstract type AbstractPairedExplicitRKIntegrator{ORDER} <: AbstractTimeIntegrator end

abstract type AbstractPairedExplicitRKSingleIntegrator{ORDER} <:
              AbstractPairedExplicitRKIntegrator{ORDER} end

abstract type AbstractPairedExplicitRKMultiIntegrator{ORDER} <:
              AbstractPairedExplicitRKIntegrator{ORDER} end

abstract type AbstractPairedExplicitRKMultiParabolicIntegrator{ORDER} <:
              AbstractPairedExplicitRKMultiIntegrator{ORDER} end

# Split Integrators
abstract type AbstractPairedExplicitRKSplitIntegrator{ORDER} <:
              AbstractPairedExplicitRKIntegrator{ORDER} end

abstract type AbstractPairedExplicitRKSplitSingleIntegrator{ORDER} <:
              AbstractPairedExplicitRKSplitIntegrator{ORDER} end

abstract type AbstractPairedExplicitRKSplitMultiIntegrator{ORDER} <:
              AbstractPairedExplicitRKSplitIntegrator{ORDER} end

# Relaxation integrators              
abstract type AbstractPairedExplicitRelaxationRKIntegrator{ORDER} <:
              AbstractPairedExplicitRKIntegrator{ORDER} end

abstract type AbstractPairedExplicitRelaxationRKSingleIntegrator{ORDER} <:
              AbstractPairedExplicitRelaxationRKIntegrator{ORDER} end

abstract type AbstractPairedExplicitRelaxationRKMultiIntegrator{ORDER} <:
              AbstractPairedExplicitRelaxationRKIntegrator{ORDER} end

# Relaxation Split Integrators
abstract type AbstractPairedExplicitRelaxationRKSplitIntegrator{ORDER} <:
              AbstractPairedExplicitRKSplitIntegrator{ORDER} end

abstract type AbstractPairedExplicitRelaxationRKSplitSingleIntegrator{ORDER} <: # Currently not implemented
              AbstractPairedExplicitRelaxationRKSplitIntegrator{ORDER} end

abstract type AbstractPairedExplicitRelaxationRKSplitMultiIntegrator{ORDER} <:
              AbstractPairedExplicitRelaxationRKSplitIntegrator{ORDER} end

# IMEX integrators
abstract type AbstractPairedExplicitRKIMEXIntegrator{ORDER} <:
              AbstractPairedExplicitRKIntegrator{ORDER} end

# Single IMEX does not make sense, as we have always at least two methods:
# The explicit method and the implicit method (implicit midpoint for p = 2)
abstract type AbstractPairedExplicitRKIMEXMultiIntegrator{ORDER} <:
              AbstractPairedExplicitRKIMEXIntegrator{ORDER} end
# parabolic additions
abstract type AbstractPairedExplicitRKIMEXMultiParabolicIntegrator{ORDER} <:
              AbstractPairedExplicitRKIMEXMultiIntegrator{ORDER} end

# Euler-Acoustic integrators
abstract type AbstractPairedExplicitRKEulerAcousticSingleIntegrator{ORDER} <:
              AbstractPairedExplicitRKSingleIntegrator{ORDER} end

abstract type AbstractPairedExplicitRKEulerAcousticMultiIntegrator{ORDER} <:
              AbstractPairedExplicitRKMultiIntegrator{ORDER} end

# The relaxation-multi-prarabolic integrator "inherits" from the 
# multi-parabolic integrator since the latter governs which stage functions, 
# i.e., the `PERK_k...` are called.
abstract type AbstractPairedExplicitRelaxationRKMultiParabolicIntegrator{ORDER} <:
              AbstractPairedExplicitRKMultiParabolicIntegrator{ORDER} end

@inline function update_t_relaxation!(integrator::Union{AbstractPairedExplicitRelaxationRKIntegrator,
                                                        AbstractPairedExplicitRelaxationRKMultiParabolicIntegrator,
                                                        AbstractPairedExplicitRelaxationRKSplitMultiIntegrator})
    # Check if due to entropy relaxation the final step is not reached
    if integrator.finalstep == true && integrator.gamma != 1
        # If we would go beyond the final time, clip gamma at 1.0
        #if integrator.gamma > 1.0
        integrator.gamma = 1
        #else # If we are below the final time, reset finalstep flag
        #    integrator.finalstep = false
        #end
    end
    integrator.t += integrator.gamma * integrator.dt

    # Write t and gamma to file for plotting
    #=
    open("relaxation_log.txt", "a") do file
        write(file, "$(integrator.t) $(integrator.gamma)\n")
    end
    =#
    return nothing
end

"""
    calculate_cfl(ode_algorithm::AbstractPairedExplicitRK, ode)

This function computes the CFL number once using the initial condition of the problem and the optimal timestep (`dt_opt`) from the ODE algorithm.
"""
function calculate_cfl(ode_algorithm::AbstractPairedExplicitRK, ode)
    t0 = first(ode.tspan)
    u_ode = ode.u0
    semi = ode.p
    dt_opt = ode_algorithm.dt_opt

    if isnothing(dt_opt)
        error("The optimal time step `dt_opt` must be provided.")
    end

    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    cfl_number = dt_opt / max_dt(u, t0, mesh,
                        have_constant_speed(equations), equations,
                        solver, cache)
    return cfl_number
end

"""
    add_tstop!(integrator::AbstractPairedExplicitRKIntegrator, t)
Add a time stop during the time integration process.
This function is called after the periodic SaveSolutionCallback to specify the next stop to save the solution.
"""
function add_tstop!(integrator::AbstractPairedExplicitRKIntegrator, t)
    integrator.tdir * (t - integrator.t) < zero(integrator.t) &&
        error("Tried to add a tstop that is behind the current time. This is strictly forbidden")
    # We need to remove the first entry of tstops when a new entry is added.
    # Otherwise, the simulation gets stuck at the previous tstop and dt is adjusted to zero.
    if length(integrator.opts.tstops) > 1
        pop!(integrator.opts.tstops)
    end
    push!(integrator.opts.tstops, integrator.tdir * t)

    return nothing
end

has_tstop(integrator::AbstractPairedExplicitRKIntegrator) = !isempty(integrator.opts.tstops)
first_tstop(integrator::AbstractPairedExplicitRKIntegrator) = first(integrator.opts.tstops)

function solve!(integrator::AbstractPairedExplicitRKIntegrator)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
        step!(integrator)
    end

    finalize_callbacks(integrator)
    # For AMR: Counting RHS evals
    #println("RHS Calls: ", integrator.RHSCalls)

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

# Euler-Acoustic requires storing the previous time step
function solve!(integrator::Union{AbstractPairedExplicitRKEulerAcousticSingleIntegrator,
                                  AbstractPairedExplicitRKEulerAcousticMultiIntegrator})
    @unpack prob = integrator.sol

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
        # Store variables of previous time step
        integrator.t_prev = integrator.t
        integrator.u_prev .= integrator.u # TODO: Probably slower than @threaded loop!

        step!(integrator)
    end

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

# Function that computes the first stage of a general PERK method
@inline function PERK_k1!(integrator::AbstractPairedExplicitRKIntegrator,
                          p)
    integrator.f(integrator.k1, integrator.u, p, integrator.t, integrator)

    return nothing
end

@inline function PERK_k1!(integrator::AbstractPairedExplicitRKSplitIntegrator,
                          p)
    integrator.f.f2(integrator.k1, integrator.u, p, integrator.t) # Hyperbolic part
    integrator.f.f1(integrator.k1_para, integrator.u, p, integrator.t) # Parabolic part

    return nothing
end

@inline function PERK_k2!(integrator::Union{AbstractPairedExplicitRKSingleIntegrator,
                                            AbstractPairedExplicitRelaxationRKSingleIntegrator},
                          p, alg)
    c_dt = alg.c[2] * integrator.dt
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] + c_dt * integrator.k1[i]
    end

    integrator.f(integrator.du, integrator.u_tmp, p, integrator.t + c_dt)

    return nothing
end

@inline function PERK_k2!(integrator::AbstractPairedExplicitRKSplitSingleIntegrator,
                          p, alg)
    c_dt = alg.c[2] * integrator.dt
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              c_dt * (integrator.k1[i] + integrator.k1_para[i])
    end

    integrator.f.f2(integrator.du, integrator.u_tmp, p, integrator.t + c_dt) # Hyperbolic part
    integrator.f.f1(integrator.du_para, integrator.u_tmp, p, integrator.t + c_dt) # Parabolic part

    return nothing
end

@inline function PERK_ki!(integrator::Union{AbstractPairedExplicitRKSingleIntegrator,
                                            AbstractPairedExplicitRelaxationRKSingleIntegrator},
                          p, alg, stage)
    # Construct current state
    a1_dt = alg.a_matrix[1, stage - 2] * integrator.dt
    a2_dt = alg.a_matrix[2, stage - 2] * integrator.dt
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              a1_dt * integrator.k1[i] + a2_dt * integrator.du[i]
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[stage] * integrator.dt)

    return nothing
end

@inline function PERK_ki!(integrator::AbstractPairedExplicitRKSplitSingleIntegrator,
                          p, alg, stage)
    # Construct current state
    a1_dt = alg.a_matrix[1, stage - 2] * integrator.dt
    a2_dt = alg.a_matrix[2, stage - 2] * integrator.dt
    a1_para_dt = alg.a_matrix_para[1, stage - 2] * integrator.dt
    a2_para_dt = alg.a_matrix_para[2, stage - 2] * integrator.dt

    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              a1_dt * integrator.k1[i] + a2_dt * integrator.du[i] +
                              a1_para_dt * integrator.k1_para[i] +
                              a2_para_dt * integrator.du_para[i]
    end

    integrator.f.f2(integrator.du, integrator.u_tmp, p,
                    integrator.t + alg.c[stage] * integrator.dt) # Hyperbolic part
    integrator.f.f1(integrator.du_para, integrator.u_tmp, p,
                    integrator.t + alg.c[stage] * integrator.dt) # Parabolic part

    return nothing
end

@inline function PERK_k2!(integrator::Union{AbstractPairedExplicitRKMultiIntegrator,
                                            AbstractPairedExplicitRelaxationRKMultiIntegrator,
                                            AbstractPairedExplicitRKIMEXMultiIntegrator},
                          p, alg)
    c_dt = alg.c[2] * integrator.dt
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] + c_dt * integrator.k1[i]
    end

    # k2: Only evaluated at finest (explicit) level: 1
    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + c_dt, integrator, 1)

    return nothing
end

# Version with SAME number of stages for hyperbolic and parabolic part
#=
@inline function PERK_k2!(integrator::AbstractPairedExplicitRKSplitMultiIntegrator,
                          p, alg)
    c_dt = alg.c[2] * integrator.dt
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              c_dt * (integrator.k1[i] + integrator.k1_para[i])
    end

    # k2: Only evaluated at finest (explicit) level: 1
    # Hyperbolic part: Always evaluated
    integrator.f.f2(integrator.du, integrator.u_tmp, p,
                    integrator.t + c_dt, integrator, 1)
    # Parabolic part
    integrator.f.f1(integrator.du_para, integrator.u_tmp, p,
                    integrator.t + c_dt, integrator, 1)

    return nothing
end
=#

# Version with DIFFERENT number of stages and partitioning for hyperbolic and parabolic part
@inline function PERK_k2!(integrator::Union{AbstractPairedExplicitRKSplitMultiIntegrator,
                                            AbstractPairedExplicitRelaxationRKSplitMultiIntegrator},
                          p, alg)
    c_dt = alg.c[2] * integrator.dt
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              c_dt * (integrator.k1[i] + integrator.k1_para[i])
    end

    # k2: Only evaluated at finest (explicit) level: 1
    # Hyperbolic part: Always evaluated
    integrator.f.f2(integrator.du, integrator.u_tmp, p,
                    integrator.t + c_dt, integrator, 1)
    # Parabolic part
    if alg.num_stages_para == alg.num_stages
        integrator.f.f1(integrator.du_para, integrator.u_tmp, p,
                        integrator.t + c_dt, integrator, 1)
    end

    return nothing
end

@inline function PERKMulti_intermediate_stage!(integrator::Union{AbstractPairedExplicitRKMultiIntegrator,
                                                                 AbstractPairedExplicitRelaxationRKMultiIntegrator},
                                               alg, stage)
    if alg.num_methods == integrator.n_levels
        ### Simplified implementation: Own method for each level ###

        #=
        # "indices to u" style
        for level in 1:(integrator.n_levels)
            @threaded for i in integrator.level_info_u[level]
                integrator.u_tmp[i] = integrator.u[i] +
                                      integrator.dt *
                                      alg.a_matrices[level, 1, stage - 2] *
                                      integrator.k1[i]
            end
        end

        for level in 1:alg.max_add_levels[stage]
            @threaded for i in integrator.level_info_u[level]
                integrator.u_tmp[i] += integrator.dt *
                                       alg.a_matrices[level, 2, stage - 2] *
                                       integrator.du[i]
            end
        end
        =#

        #=
        # "u to indices" style
        # See e.g. commit
        # https://github.com/DanielDoehring/Trixi.jl/commit/c775e5f45899cb75c742936c059629178ec766cb
        #  for reconstruction
        # NOTE: Could combine this with the "indices to u" style
        @threaded for i in eachindex(integrator.u)
            integrator.u_tmp[i] = integrator.u[i] +
                                  integrator.dt *
                                  alg.a_matrices[integrator.u_to_level[i], 1,
                                                 stage - 2] *
                                  integrator.k1[i]
        end

        @threaded for i in integrator.level_info_u_acc[alg.max_add_levels[stage]]
            integrator.u_tmp[i] += integrator.dt *
                                   alg.a_matrices[integrator.u_to_level[i], 2,
                                                  stage - 2] *
                                   integrator.du[i]
        end
        =#

        # "PERK4" style
        for level in 1:alg.max_add_levels[stage]
            a1_dt = alg.a_matrices[level, 1, stage - 2] * integrator.dt
            a2_dt = alg.a_matrices[level, 2, stage - 2] * integrator.dt
            @threaded for i in integrator.level_info_u[level]
                integrator.u_tmp[i] = integrator.u[i] +
                                      a1_dt * integrator.k1[i] +
                                      a2_dt * integrator.du[i]
            end
        end

        c_dt = alg.c[stage] * integrator.dt
        for level in (alg.max_add_levels[stage] + 1):(integrator.n_levels)
            @threaded for i in integrator.level_info_u[level]
                integrator.u_tmp[i] = integrator.u[i] +
                                      c_dt * integrator.k1[i]
            end
        end
    else
        ### General implementation: Not own method for each grid level ###

        # Loop over different methods with own associated level
        for level in 1:min(alg.num_methods, integrator.n_levels)
            a1_dt = alg.a_matrices[level, 1, stage - 2] * integrator.dt
            @threaded for i in integrator.level_info_u[level]
                integrator.u_tmp[i] = integrator.u[i] +
                                      a1_dt * integrator.k1[i]
            end
        end
        for level in 1:min(alg.max_add_levels[stage], integrator.n_levels)
            a2_dt = alg.a_matrices[level, 2, stage - 2] * integrator.dt
            @threaded for i in integrator.level_info_u[level]
                #integrator.u_tmp[i] += a2_dt * integrator.du[i]
                # Try optimize for `@muladd`: avoid `+=`
                integrator.u_tmp[i] = integrator.u_tmp[i] + a2_dt * integrator.du[i]
            end
        end

        # "Remainder": Non-efficiently integrated
        for level in (alg.num_methods + 1):(integrator.n_levels)
            a1_dt = alg.a_matrices[alg.num_methods, 1, stage - 2] * integrator.dt
            @threaded for i in integrator.level_info_u[level]
                integrator.u_tmp[i] = integrator.u[i] +
                                      a1_dt * integrator.k1[i]
            end
        end
        if alg.max_add_levels[stage] == alg.num_methods
            for level in (alg.max_add_levels[stage] + 1):(integrator.n_levels)
                a2_dt = alg.a_matrices[alg.num_methods, 2, stage - 2] * integrator.dt
                @threaded for i in integrator.level_info_u[level]
                    #integrator.u_tmp[i] += a2_dt * integrator.du[i]
                    # Try optimize for `@muladd`: avoid `+=`
                    integrator.u_tmp[i] = integrator.u_tmp[i] + a2_dt * integrator.du[i]
                end
            end
        end
    end

    # For statically non-uniform meshes/characteristic speeds
    #integrator.coarsest_lvl = alg.max_active_levels[stage]

    # "coarsest_lvl" cannot be static for AMR, has to be checked with available levels
    integrator.coarsest_lvl = min(alg.max_active_levels[stage],
                                  integrator.n_levels)

    return nothing
end

@inline function PERKMulti_intermediate_stage!(integrator::AbstractPairedExplicitRKIMEXMultiIntegrator,
                                               alg, stage)
    # CARE: Currently only implemented for matching number of methods and grid-levels!
    ### Simplified implementation: Own method for each level ###
    for level in 1:alg.max_add_levels[stage]
        a1_dt = alg.a_matrices[level, 1, stage - 2] * integrator.dt
        a2_dt = alg.a_matrices[level, 2, stage - 2] * integrator.dt
        @threaded for i in integrator.level_info_u[level]
            integrator.u_tmp[i] = integrator.u[i] +
                                  a1_dt * integrator.k1[i] +
                                  a2_dt * integrator.du[i]
        end
    end

    c_dt = alg.c[stage] * integrator.dt
    for level in (alg.max_add_levels[stage] + 1):(integrator.n_levels)
        @threaded for i in integrator.level_info_u[level]
            integrator.u_tmp[i] = integrator.u[i] +
                                  c_dt * integrator.k1[i]
        end
    end

    # For statically non-uniform meshes/characteristic speeds
    integrator.coarsest_lvl = alg.max_active_levels[stage]

    return nothing
end

# Version with SAME number of stages for hyperbolic and parabolic part
#=
@inline function PERKMulti_intermediate_stage!(integrator::AbstractPairedExplicitRKSplitMultiIntegrator,
                                               alg, stage)
    if alg.num_methods == integrator.n_levels
        # "PERK4" style
        for level in 1:alg.max_add_levels[stage]
            a1_dt = alg.a_matrices[level, 1, stage - 2] * integrator.dt
            a2_dt = alg.a_matrices[level, 2, stage - 2] * integrator.dt
            a1_para_dt = alg.a_matrices_para[level, 1, stage - 2] * integrator.dt
            a2_para_dt = alg.a_matrices_para[level, 2, stage - 2] * integrator.dt
            @threaded for i in integrator.level_info_u[level]
                integrator.u_tmp[i] = integrator.u[i] +
                                      a1_dt * integrator.k1[i] +
                                      a2_dt * integrator.du[i] +
                                      a1_para_dt * integrator.k1_para[i] +
                                      a2_para_dt * integrator.du_para[i]
            end
        end

        for level in (alg.max_add_levels[stage] + 1):(integrator.n_levels)
            a1_dt = alg.a_matrices[alg.num_methods, 1, stage - 2] * integrator.dt
            a1_para_dt = alg.a_matrices_para[level, 1, stage - 2] *
                         integrator.dt
            @threaded for i in integrator.level_info_u[level]
                integrator.u_tmp[i] = integrator.u[i] +
                                      a1_dt * integrator.k1[i] +
                                      a1_para_dt * integrator.k1_para[i]
            end
        end
    else
        ### General implementation: Not own method for each grid level ###

        # Loop over different methods with own associated level
        for level in 1:min(alg.num_methods, integrator.n_levels)
            a1_dt = alg.a_matrices[level, 1, stage - 2] * integrator.dt
            a1_para_dt = alg.a_matrices_para[level, 1, stage - 2] * integrator.dt
            @threaded for i in integrator.level_info_u[level]
                integrator.u_tmp[i] = integrator.u[i] +
                                      a1_dt * integrator.k1[i] +
                                      a1_para_dt * integrator.k1_para[i]
            end
        end
        for level in 1:min(alg.max_add_levels[stage], integrator.n_levels)
            a2_dt = alg.a_matrices[level, 2, stage - 2] * integrator.dt
            a2_para_dt = alg.a_matrices_para[level, 2, stage - 2] * integrator.dt
            @threaded for i in integrator.level_info_u[level]
                #integrator.u_tmp[i] += a2_dt * integrator.du[i]
                # Try optimize for `@muladd`: avoid `+=`
                integrator.u_tmp[i] = integrator.u_tmp[i] +
                                      a2_dt * integrator.du[i] +
                                      a2_para_dt * integrator.du_para[i]
            end
        end

        # "Remainder": Non-efficiently integrated
        for level in (alg.num_methods + 1):(integrator.n_levels)
            a1_dt = alg.a_matrices[alg.num_methods, 1, stage - 2] * integrator.dt
            a1_para_dt = alg.a_matrices_para[alg.num_methods, 1, stage - 2] *
                         integrator.dt
            @threaded for i in integrator.level_info_u[level]
                integrator.u_tmp[i] = integrator.u[i] +
                                      a1_dt * integrator.k1[i] +
                                      a1_para_dt * integrator.k1_para[i]
            end
        end
        if alg.max_add_levels[stage] == alg.num_methods
            for level in (alg.max_add_levels[stage] + 1):(integrator.n_levels)
                a2_dt = alg.a_matrices[alg.num_methods, 2, stage - 2] * integrator.dt
                a2_para_dt = alg.a_matrices_para[alg.num_methods, 2, stage - 2] *
                             integrator.dt
                @threaded for i in integrator.level_info_u[level]
                    #integrator.u_tmp[i] += a2_dt * integrator.du[i]
                    # Try optimize for `@muladd`: avoid `+=`
                    integrator.u_tmp[i] = integrator.u_tmp[i] +
                                          a2_dt * integrator.du[i] +
                                          a2_para_dt * integrator.du_para[i]
                end
            end
        end
    end

    # For statically non-uniform meshes/characteristic speeds
    #integrator.coarsest_lvl = alg.max_active_levels[stage]

    # "coarsest_lvl" cannot be static for AMR, has to be checked with available levels
    integrator.coarsest_lvl = min(alg.max_active_levels[stage],
                                  integrator.n_levels)

    return nothing
end
=#

# Version with DIFFERENT number of stages and partitioning for hyperbolic and parabolic part
@inline function PERKMulti_intermediate_stage!(integrator::Union{AbstractPairedExplicitRKSplitMultiIntegrator,
                                                                 AbstractPairedExplicitRelaxationRKSplitMultiIntegrator},
                                               alg, stage)
    if alg.num_methods == integrator.n_levels
        # "PERK4" style
        for level in 1:alg.max_add_levels[stage]
            a1_dt = alg.a_matrices[level, 1, stage - 2] * integrator.dt
            a2_dt = alg.a_matrices[level, 2, stage - 2] * integrator.dt
            @threaded for i in integrator.level_info_u[level]
                integrator.u_tmp[i] = integrator.u[i] +
                                      a1_dt * integrator.k1[i] +
                                      a2_dt * integrator.du[i]
            end
        end

        c_dt = alg.c[stage] * integrator.dt
        for level in (alg.max_add_levels[stage] + 1):(integrator.n_levels)
            @threaded for i in integrator.level_info_u[level]
                integrator.u_tmp[i] = integrator.u[i] +
                                      c_dt * integrator.k1[i]
            end
        end
    else
        ### General implementation: Not own method for each grid level ###

        # Loop over different methods with own associated level
        for level in 1:min(alg.num_methods, integrator.n_levels)
            a1_dt = alg.a_matrices[level, 1, stage - 2] * integrator.dt
            @threaded for i in integrator.level_info_u[level]
                integrator.u_tmp[i] = integrator.u[i] +
                                      a1_dt * integrator.k1[i]
            end
        end
        for level in 1:min(alg.max_add_levels[stage], integrator.n_levels)
            a2_dt = alg.a_matrices[level, 2, stage - 2] * integrator.dt
            @threaded for i in integrator.level_info_u[level]
                #integrator.u_tmp[i] += a2_dt * integrator.du[i]
                # Try optimize for `@muladd`: avoid `+=`
                integrator.u_tmp[i] = integrator.u_tmp[i] +
                                      a2_dt * integrator.du[i]
            end
        end

        # "Remainder": Non-efficiently integrated
        for level in (alg.num_methods + 1):(integrator.n_levels)
            a1_dt = alg.a_matrices[alg.num_methods, 1, stage - 2] * integrator.dt
            @threaded for i in integrator.level_info_u[level]
                integrator.u_tmp[i] = integrator.u[i] +
                                      a1_dt * integrator.k1[i]
            end
        end
        if alg.max_add_levels[stage] == alg.num_methods
            for level in (alg.max_add_levels[stage] + 1):(integrator.n_levels)
                a2_dt = alg.a_matrices[alg.num_methods, 2, stage - 2] * integrator.dt
                @threaded for i in integrator.level_info_u[level]
                    #integrator.u_tmp[i] += a2_dt * integrator.du[i]
                    # Try optimize for `@muladd`: avoid `+=`
                    integrator.u_tmp[i] = integrator.u_tmp[i] +
                                          a2_dt * integrator.du[i]
                end
            end
        end
    end

    # For statically non-uniform meshes/characteristic speeds
    #integrator.coarsest_lvl = alg.max_active_levels[stage]

    # "coarsest_lvl" cannot be static for AMR, has to be checked with available levels
    integrator.coarsest_lvl = min(alg.max_active_levels[stage],
                                  integrator.n_levels)

    # Parabolic contribution, add to `u_tmp`!
    if alg.num_methods_para == integrator.n_levels_para
        # "PERK4" style
        for level in 1:alg.max_add_levels_para[stage]
            a1_para_dt = alg.a_matrices_para[level, 1, stage - 2] * integrator.dt
            a2_para_dt = alg.a_matrices_para[level, 2, stage - 2] * integrator.dt
            @threaded for i in integrator.level_info_u_para[level]
                integrator.u_tmp[i] = integrator.u_tmp[i] +
                                      a1_para_dt * integrator.k1_para[i] +
                                      a2_para_dt * integrator.du_para[i]
            end
        end

        c_dt = alg.c[stage] * integrator.dt
        for level in (alg.max_add_levels_para[stage] + 1):(integrator.n_levels_para)
            @threaded for i in integrator.level_info_u_para[level]
                integrator.u_tmp[i] = integrator.u_tmp[i] +
                                      c_dt * integrator.k1_para[i]
            end
        end
    else
        ### General implementation: Not own method for each grid level ###

        # Loop over different methods with own associated level
        for level in 1:min(alg.num_methods_para, integrator.n_levels_para)
            a1_para_dt = alg.a_matrices_para[level, 1, stage - 2] * integrator.dt
            @threaded for i in integrator.level_info_u_para[level]
                integrator.u_tmp[i] = integrator.u_tmp[i] +
                                      a1_para_dt * integrator.k1_para[i]
            end
        end
        for level in 1:min(alg.max_add_levels_para[stage], integrator.n_levels_para)
            a2_para_dt = alg.a_matrices_para[level, 2, stage - 2] * integrator.dt
            @threaded for i in integrator.level_info_u_para[level]
                #integrator.u_tmp[i] += a2_dt * integrator.du[i]
                # Try optimize for `@muladd`: avoid `+=`
                integrator.u_tmp[i] = integrator.u_tmp[i] +
                                      a2_para_dt * integrator.du_para[i]
            end
        end

        # "Remainder": Non-efficiently integrated
        for level in (alg.num_methods_para + 1):(integrator.n_levels_para)
            a1_para_dt = alg.a_matrices_para[alg.num_methods_para, 1, stage - 2] *
                         integrator.dt
            @threaded for i in integrator.level_info_u_para[level]
                integrator.u_tmp[i] = integrator.u_tmp[i] +
                                      a1_para_dt * integrator.k1_para[i]
            end
        end
        if alg.max_add_levels_para[stage] == alg.num_methods_para
            for level in (alg.max_add_levels_para[stage] + 1):(integrator.n_levels_para)
                a2_para_dt = alg.a_matrices_para[alg.num_methods, 2, stage - 2] *
                             integrator.dt
                @threaded for i in integrator.level_info_u_para[level]
                    #integrator.u_tmp[i] += a2_dt * integrator.du[i]
                    # Try optimize for `@muladd`: avoid `+=`
                    integrator.u_tmp[i] = integrator.u_tmp[i] +
                                          a2_para_dt * integrator.du_para[i]
                end
            end
        end
    end

    # For statically non-uniform meshes/characteristic speeds
    #integrator.coarsest_lvl = alg.max_active_levels[stage]

    # "coarsest_lvl" cannot be static for AMR, has to be checked with available levels
    integrator.coarsest_lvl_para = min(alg.max_active_levels_para[stage],
                                       integrator.n_levels_para)

    return nothing
end

@inline function PERK_ki!(integrator::Union{AbstractPairedExplicitRKMultiIntegrator,
                                            AbstractPairedExplicitRelaxationRKMultiIntegrator,
                                            AbstractPairedExplicitRKIMEXMultiIntegrator},
                          p, alg, stage)
    PERKMulti_intermediate_stage!(integrator, alg, stage)

    # Check if there are fewer integrators than grid levels (non-optimal method)
    if integrator.coarsest_lvl == alg.num_methods
        # NOTE: This is supposedly more efficient than setting
        #integrator.coarsest_lvl = integrator.n_levels
        # and then using the level-dependent function

        integrator.f(integrator.du, integrator.u_tmp, p,
                     integrator.t + alg.c[stage] * integrator.dt,
                     integrator)
    else
        integrator.f(integrator.du, integrator.u_tmp, p,
                     integrator.t + alg.c[stage] * integrator.dt,
                     integrator, integrator.coarsest_lvl)
    end

    return nothing
end

# Version with SAME number of stages for hyperbolic and parabolic part
#=
@inline function PERK_ki!(integrator::AbstractPairedExplicitRKSplitMultiIntegrator,
                          p, alg, stage)
    PERKMulti_intermediate_stage!(integrator, alg, stage)

    # Check if there are fewer integrators than grid levels (non-optimal method)
    if integrator.coarsest_lvl == alg.num_methods
        # NOTE: This is supposedly more efficient than setting
        #integrator.coarsest_lvl = integrator.n_levels
        # and then using the level-dependent function

        # Hyperbolic part
        integrator.f.f2(integrator.du, integrator.u_tmp, p,
                        integrator.t + alg.c[stage] * integrator.dt)
        # Parabolic part
        integrator.f.f1(integrator.du_para, integrator.u_tmp, p,
                        integrator.t + alg.c[stage] * integrator.dt)
    else
        # Hyperbolic part
        integrator.f.f2(integrator.du, integrator.u_tmp, p,
                        integrator.t + alg.c[stage] * integrator.dt,
                        integrator, integrator.coarsest_lvl)
        # Parabolic part
        integrator.f.f1(integrator.du_para, integrator.u_tmp, p,
                        integrator.t + alg.c[stage] * integrator.dt,
                        integrator, integrator.coarsest_lvl)
    end

    return nothing
end
=#

# Version with DIFFERENT number of stages for hyperbolic and parabolic part
@inline function PERK_ki!(integrator::Union{AbstractPairedExplicitRKSplitMultiIntegrator,
                                            AbstractPairedExplicitRelaxationRKSplitMultiIntegrator},
                          p, alg, stage)
    PERKMulti_intermediate_stage!(integrator, alg, stage)

    # Check if there are fewer integrators than grid levels (non-optimal method)
    if integrator.coarsest_lvl == alg.num_methods
        # NOTE: This is supposedly more efficient than setting
        #integrator.coarsest_lvl = integrator.n_levels
        # and then using the level-dependent function
        integrator.f.f2(integrator.du, integrator.u_tmp, p,
                        integrator.t + alg.c[stage] * integrator.dt)
    else
        integrator.f.f2(integrator.du, integrator.u_tmp, p,
                        integrator.t + alg.c[stage] * integrator.dt,
                        integrator, integrator.coarsest_lvl)
    end

    # Parabolic contribution
    if integrator.coarsest_lvl_para != 0 # Check if evaluation is required at all
        if integrator.coarsest_lvl_para == alg.num_methods_para
            # NOTE: This is supposedly more efficient than setting
            #integrator.coarsest_lvl = integrator.n_levels
            # and then using the level-dependent function
            integrator.f.f1(integrator.du_para, integrator.u_tmp, p,
                            integrator.t + alg.c[stage] * integrator.dt)
        else
            integrator.f.f1(integrator.du_para, integrator.u_tmp, p,
                            integrator.t + alg.c[stage] * integrator.dt,
                            integrator, integrator.coarsest_lvl_para)
        end
    end

    return nothing
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::AbstractPairedExplicitRKIntegrator,
                      new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    # PERK stage
    resize!(integrator.k1, new_size)

    return nothing
end

function Base.resize!(integrator::AbstractPairedExplicitRKMultiParabolicIntegrator,
                      new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    # PERK stage
    resize!(integrator.k1, new_size)
    # Addition for multirate PERK methods for parabolic problems
    resize!(integrator.du_para, new_size)

    return nothing
end

function Base.resize!(integrator::Union{AbstractPairedExplicitRKEulerAcousticSingleIntegrator,
                                        AbstractPairedExplicitRKEulerAcousticMultiIntegrator},
                      new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    # PERK stage
    resize!(integrator.k1, new_size)
    # Check for third-order
    if :kS1 in fieldnames(typeof(integrator))
        resize!(integrator.kS1, new_size)
    end
    # Previous time step (required for Euler-Acoustic)
    resize!(integrator.u_prev, new_size)

    return nothing
end

# This `resize!` targets the Euler-Gravity case where 
# the unknowns of the gravity solver also need to be repartitioned after resizing.
function Base.resize!(integrator::AbstractPairedExplicitRKMultiIntegrator,
                      new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    # PERK stage(s)
    resize!(integrator.k1, new_size)
    # Check for third-order
    if :kS1 in fieldnames(typeof(integrator))
        resize!(integrator.kS1, new_size)
    end
    # Check if we have Euler-Gravity situation
    if :semi_gravity in fieldnames(typeof(integrator.p))
        partition_u_gravity!(integrator)
    end

    return nothing
end

function Base.resize!(integrator::AbstractPairedExplicitRKSplitIntegrator,
                      new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    # PERK stage
    resize!(integrator.k1, new_size)
    # Parabolic/split-approach additions
    resize!(integrator.du_para, new_size)
    resize!(integrator.k1_para, new_size)

    return nothing
end

# get a cache where the RHS can be stored
get_tmp_cache(integrator::AbstractPairedExplicitRKIntegrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::AbstractPairedExplicitRKIntegrator, ::Bool) = false

# stop the time integration
function terminate!(integrator::AbstractPairedExplicitRKIntegrator)
    integrator.finalstep = true

    return nothing
end

# Needed for Euler-Acoustic coupling
function check_error(integrator::AbstractPairedExplicitRKIntegrator)
    return SciMLBase.ReturnCode.Success
end

"""
    modify_dt_for_tstops!(integrator::PairedExplicitRK)

Modify the time-step size to match the time stops specified in integrator.opts.tstops.
To avoid adding OrdinaryDiffEq to Trixi's dependencies, this routine is a copy of
https://github.com/SciML/OrdinaryDiffEq.jl/blob/d76335281c540ee5a6d1bd8bb634713e004f62ee/src/integrators/integrator_utils.jl#L38-L54
"""
function modify_dt_for_tstops!(integrator::AbstractPairedExplicitRKIntegrator)
    if has_tstop(integrator)
        tdir_t = integrator.tdir * integrator.t
        tdir_tstop = first_tstop(integrator)
        if integrator.opts.adaptive
            integrator.dt = integrator.tdir *
                            min(abs(integrator.dt), abs(tdir_tstop - tdir_t)) # step! to the end
        elseif iszero(integrator.dtcache) && integrator.dtchangeable
            integrator.dt = integrator.tdir * abs(tdir_tstop - tdir_t)
        elseif integrator.dtchangeable && !integrator.force_stepfail
            # always try to step! with dtcache, but lower if a tstop
            # however, if force_stepfail then don't set to dtcache, and no tstop worry
            integrator.dt = integrator.tdir *
                            min(abs(integrator.dtcache), abs(tdir_tstop - tdir_t)) # step! to the end
        end
    end

    return nothing
end

# Add definitions of functions related to polynomial optimization by NLsolve here
# such that hey can be exported from Trixi.jl and extended in the TrixiConvexECOSExt package
# extension or by the NLsolve-specific code loaded by Requires.jl
function solve_a_butcher_coeffs_unknown! end

# Depending on the `semi`, different fields from `integrator` need to be passed on.
@inline function rhs!(du_ode, u_ode, semi::SemidiscretizationHyperbolic, t,
                      integrator::Union{AbstractPairedExplicitRKMultiIntegrator,
                                        AbstractPairedExplicitRelaxationRKMultiIntegrator,
                                        AbstractPairedExplicitRKIMEXMultiIntegrator},
                      max_level)
    rhs!(du_ode, u_ode, semi, t,
         integrator.level_info_elements_acc[max_level],
         integrator.level_info_interfaces_acc[max_level],
         integrator.level_info_boundaries_acc[max_level],
         integrator.level_info_mortars_acc[max_level])

    return nothing
end

# Required for split methods
@inline function rhs!(du_ode, u_ode, semi::SemidiscretizationHyperbolicParabolic, t,
                      integrator::Union{AbstractPairedExplicitRKSplitMultiIntegrator,
                                        AbstractPairedExplicitRelaxationRKSplitMultiIntegrator},
                      max_level)
    rhs!(du_ode, u_ode, semi, t,
         integrator.level_info_elements_acc[max_level],
         integrator.level_info_interfaces_acc[max_level],
         integrator.level_info_boundaries_acc[max_level],
         integrator.level_info_mortars_acc[max_level])

    return nothing
end

# Version with SAME stage distribution for hyperbolic and parabolic part
#=
@inline function rhs_parabolic!(du_ode, u_ode,
                                semi::SemidiscretizationHyperbolicParabolic, t,
                                integrator::AbstractPairedExplicitRKSplitMultiIntegrator,
                                max_level)
    rhs_parabolic!(du_ode, u_ode, semi, t,
                   integrator.level_info_elements_acc[max_level],
                   integrator.level_info_interfaces_acc[max_level],
                   integrator.level_info_boundaries_acc[max_level],
                   integrator.level_info_mortars_acc[max_level])

    return nothing
end
=#

# Version with DIFFERENT stage distribution for hyperbolic and parabolic part
@inline function rhs_parabolic!(du_ode, u_ode,
                                semi::SemidiscretizationHyperbolicParabolic, t,
                                integrator::Union{AbstractPairedExplicitRKSplitMultiIntegrator,
                                                  AbstractPairedExplicitRelaxationRKSplitMultiIntegrator},
                                max_level)
    rhs_parabolic!(du_ode, u_ode, semi, t,
                   integrator.level_info_elements_para_acc[max_level],
                   integrator.level_info_interfaces_para_acc[max_level],
                   integrator.level_info_boundaries_para_acc[max_level],
                   integrator.level_info_mortars_para_acc[max_level])

    return nothing
end

@inline function rhs_hyperbolic_parabolic!(du_ode, u_ode,
                                           semi::SemidiscretizationHyperbolicParabolic,
                                           t,
                                           integrator::Union{AbstractPairedExplicitRKMultiParabolicIntegrator,
                                                             AbstractPairedExplicitRKIMEXMultiParabolicIntegrator},
                                           max_level)
    rhs_hyperbolic_parabolic!(du_ode, u_ode, semi, t,
                              integrator.du_para,
                              max_level,
                              integrator.level_info_elements_acc[max_level],
                              integrator.level_info_interfaces_acc[max_level],
                              integrator.level_info_boundaries_acc[max_level],
                              integrator.level_info_mortars_acc[max_level],
                              integrator.level_info_u) # TODO: Optimized version with accumulated indices

    return nothing
end

@inline function rhs!(du_ode, u_ode, semi::SemidiscretizationEulerAcoustics, t,
                      integrator::AbstractPairedExplicitRKEulerAcousticMultiIntegrator,
                      max_level)
    rhs!(du_ode, u_ode, semi, t,
         max_level,
         integrator.level_info_elements_acc[max_level],
         integrator.level_info_interfaces_acc[max_level],
         integrator.level_info_boundaries_acc[max_level],
         integrator.level_info_mortars_acc[max_level])

    return nothing
end

# Dummy argument `integrator` for same signature as `rhs_hyperbolic_parabolic!` for
# hyperbolic-parabolic split ODE problems solved with non-split integrators, such 
# as the single/standalone PERK schemes.
@inline function rhs!(du_ode, u_ode, semi::AbstractSemidiscretization, t,
                      integrator)
    rhs!(du_ode, u_ode, semi, t)

    return nothing
end
@inline function (f::SplitFunction)(du_ode, u_ode, semi::AbstractSemidiscretization, t,
                                    integrator)
    f(du_ode, u_ode, semi, t)

    return nothing
end

# Multirate/partitioned helpers
include("partitioning.jl")

include("p2/methods_PERK2.jl")
include("p3/methods_PERK3.jl")
include("p4/methods_PERK4.jl")
end # @muladd
