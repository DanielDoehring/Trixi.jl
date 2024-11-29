# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Define all of the functions necessary for polynomial optimizations
include("polynomial_optimizer.jl")

# Abstract base type for both single/standalone and multi-level 
# PERK (Paired Explicit Runge-Kutta) time integration schemes
abstract type AbstractPairedExplicitRK end
# Abstract base type for single/standalone PERK time integration schemes
abstract type AbstractPairedExplicitRKSingle <: AbstractPairedExplicitRK end
# Abstract base type for single/standalone PERK time integration schemes
abstract type AbstractPairedExplicitRKMulti <: AbstractPairedExplicitRK end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct PairedExplicitRKOptions{Callback, TStops}
    callback::Callback # callbacks; used in Trixi
    adaptive::Bool # whether the algorithm is adaptive
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

abstract type AbstractPairedExplicitRKIntegrator{ORDER} end

abstract type AbstractPairedExplicitRKSingleIntegrator{ORDER} <:
              AbstractPairedExplicitRKIntegrator{ORDER} end

abstract type AbstractPairedExplicitRKMultiIntegrator{ORDER} <:
              AbstractPairedExplicitRKIntegrator{ORDER} end

abstract type AbstractPairedExplicitRKMultiParabolicIntegrator{ORDER} <:
              AbstractPairedExplicitRKMultiIntegrator{ORDER} end

# Relaxation integrators              
abstract type AbstractPairedExplicitRelaxationRKIntegrator{ORDER} <:
              AbstractPairedExplicitRKIntegrator{ORDER} end

abstract type AbstractPairedExplicitRelaxationRKSingleIntegrator{ORDER} <:
              AbstractPairedExplicitRelaxationRKIntegrator{ORDER} end

abstract type AbstractPairedExplicitRelaxationRKMultiIntegrator{ORDER} <:
              AbstractPairedExplicitRelaxationRKIntegrator{ORDER} end

abstract type AbstractPairedExplicitRelaxationRKMultiParabolicIntegrator{ORDER} <:
              AbstractPairedExplicitRKMultiParabolicIntegrator{ORDER} end
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

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::AbstractPairedExplicitRKIntegrator,
                          field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
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
end

has_tstop(integrator::AbstractPairedExplicitRKIntegrator) = !isempty(integrator.opts.tstops)
first_tstop(integrator::AbstractPairedExplicitRKIntegrator) = first(integrator.opts.tstops)

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::AbstractPairedExplicitRK;
               dt, callback = nothing, kwargs...)
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    solve!(integrator)
end

function solve!(integrator::AbstractPairedExplicitRKIntegrator)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
        step!(integrator)
    end

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

# Function that computes the first stage of a general PERK method
@inline function PERK_k1!(integrator::AbstractPairedExplicitRKIntegrator,
                          p)
    integrator.f(integrator.k1, integrator.u, p, integrator.t)
end

@inline function PERK_k2!(integrator::Union{AbstractPairedExplicitRKSingleIntegrator,
                                            AbstractPairedExplicitRelaxationRKSingleIntegrator},
                          p, alg)
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              alg.c[2] * integrator.dt * integrator.k1[i]
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[2] * integrator.dt)
end

@inline function PERK_ki!(integrator::Union{AbstractPairedExplicitRKSingleIntegrator,
                                            AbstractPairedExplicitRelaxationRKSingleIntegrator},
                          p, alg, stage)
    # Construct current state
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              integrator.dt *
                              (alg.a_matrix[1, stage - 2] * integrator.k1[i] +
                               alg.a_matrix[2, stage - 2] * integrator.du[i])
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[stage] * integrator.dt)
end

@inline function PERK_k2!(integrator::Union{AbstractPairedExplicitRKMultiIntegrator,
                                            AbstractPairedExplicitRelaxationRKMultiIntegrator},
                          p, alg)
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              alg.c[2] * integrator.dt * integrator.k1[i]
    end

    # k2: Only evaluated at finest level
    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[2] * integrator.dt,
                 integrator.level_info_elements_acc[1],
                 integrator.level_info_interfaces_acc[1],
                 integrator.level_info_boundaries_acc[1],
                 #integrator.level_info_boundaries_orientation_acc[1],
                 integrator.level_info_mortars_acc[1])
end

@inline function PERKMulti_intermediate_stage!(integrator::Union{AbstractPairedExplicitRKMultiIntegrator,
                                                                 AbstractPairedExplicitRelaxationRKMultiIntegrator},
                                               alg, stage)
    ### General implementation: Not own method for each grid level ###
    # Loop over different methods with own associated level

    for level in 1:min(alg.num_methods, integrator.n_levels)
        @threaded for u_ind in integrator.level_u_indices_elements[level]
            integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                      integrator.dt *
                                      alg.a_matrices[level, 1, stage - 2] *
                                      integrator.k1[u_ind]
        end
    end
    for level in 1:min(alg.max_eval_levels[stage], integrator.n_levels)
        @threaded for u_ind in integrator.level_u_indices_elements[level]
            integrator.u_tmp[u_ind] += integrator.dt *
                                       alg.a_matrices[level, 2, stage - 2] *
                                       integrator.du[u_ind]
        end
    end

    # "Remainder": Non-efficiently integrated
    for level in (alg.num_methods + 1):(integrator.n_levels)
        @threaded for u_ind in integrator.level_u_indices_elements[level]
            integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                      integrator.dt *
                                      alg.a_matrices[alg.num_methods, 1, stage - 2] *
                                      integrator.k1[u_ind]
        end
    end
    if alg.max_eval_levels[stage] == alg.num_methods
        for level in (alg.max_eval_levels[stage] + 1):(integrator.n_levels)
            @threaded for u_ind in integrator.level_u_indices_elements[level]
                integrator.u_tmp[u_ind] += integrator.dt *
                                           alg.a_matrices[alg.num_methods, 2,
                                                          stage - 2] *
                                           integrator.du[u_ind]
            end
        end
    end

    ### Simplified implementation: Own method for each level ###
    #=
    for level in 1:integrator.n_levels
        @threaded for u_ind in integrator.level_u_indices_elements[level]
            integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                      integrator.dt *
                                      alg.a_matrices[level, 1, stage - 2] *
                                      integrator.k1[u_ind]
        end
    end
    for level in 1:alg.max_eval_levels[stage]
        @threaded for u_ind in integrator.level_u_indices_elements[level]
            integrator.u_tmp[u_ind] += integrator.dt *
                                       alg.a_matrices[level, 2, stage - 2] *
                                       integrator.du[u_ind]
        end
    end
    =#

    # For statically non-uniform meshes/characteristic speeds
    #integrator.coarsest_lvl = alg.max_active_levels[stage]

    # "coarsest_lvl" cannot be static for AMR, has to be checked with available levels
    integrator.coarsest_lvl = min(alg.max_active_levels[stage],
                                  integrator.n_levels)
end

@inline function PERK_ki!(integrator::Union{AbstractPairedExplicitRKMultiIntegrator,
                                            AbstractPairedExplicitRelaxationRKMultiIntegrator},
                          p, alg, stage)
    PERKMulti_intermediate_stage!(integrator, alg, stage)

    # Check if there are fewer integrators than grid levels (non-optimal method)
    if integrator.coarsest_lvl == alg.num_methods
        # NOTE: This is supposedly more efficient than setting
        #integrator.coarsest_lvl = integrator.n_levels
        # and then using the level-dependent version

        integrator.f(integrator.du, integrator.u_tmp, p,
                     integrator.t + alg.c[stage] * integrator.dt)
    else
        integrator.f(integrator.du, integrator.u_tmp, p,
                     integrator.t + alg.c[stage] * integrator.dt,
                     integrator.level_info_elements_acc[integrator.coarsest_lvl],
                     integrator.level_info_interfaces_acc[integrator.coarsest_lvl],
                     integrator.level_info_boundaries_acc[integrator.coarsest_lvl],
                     #integrator.level_info_boundaries_orientation_acc[integrator.coarsest_lvl],
                     integrator.level_info_mortars_acc[integrator.coarsest_lvl])
    end
end

# Own function for multirate PERK methods for parabolic problems
# as these require `du_tmp` to store the contribution of the `rhs!`
@inline function PERK_k1!(integrator::AbstractPairedExplicitRKMultiParabolicIntegrator,
                          p)
    integrator.f(integrator.du, integrator.u, p, integrator.t, integrator.du_tmp)
end

@inline function PERK_k2!(integrator::AbstractPairedExplicitRKMultiParabolicIntegrator,
                          p, alg)
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              alg.c[2] * integrator.dt * integrator.k1[i]
    end

    # k2: Only evaluated at finest level
    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[2] * integrator.dt,
                 integrator.du_tmp,
                 integrator.level_info_elements_acc[1],
                 integrator.level_info_interfaces_acc[1],
                 integrator.level_info_boundaries_acc[1],
                 #integrator.level_info_boundaries_orientation_acc[1],
                 integrator.level_info_mortars_acc[1],
                 integrator.level_u_indices_elements, 1)
end

@inline function PERK_ki!(integrator::AbstractPairedExplicitRKMultiParabolicIntegrator,
                          p, alg, stage)
    PERKMulti_intermediate_stage!(integrator, alg, stage)

    # Check if there are fewer integrators than grid levels (non-optimal method)
    if integrator.coarsest_lvl == alg.num_methods
        # NOTE: This is supposedly more efficient than setting
        #integrator.coarsest_lvl = integrator.n_levels
        # and then using the level-dependent version

        integrator.f(integrator.du, integrator.u_tmp, p,
                     integrator.t + alg.c[stage] * integrator.dt,
                     integrator.du_tmp)
    else
        integrator.f(integrator.du, integrator.u_tmp, p,
                     integrator.t + alg.c[stage] * integrator.dt,
                     integrator.du_tmp,
                     integrator.level_info_elements_acc[integrator.coarsest_lvl],
                     integrator.level_info_interfaces_acc[integrator.coarsest_lvl],
                     integrator.level_info_boundaries_acc[integrator.coarsest_lvl],
                     #integrator.level_info_boundaries_orientation_acc[integrator.coarsest_lvl],
                     integrator.level_info_mortars_acc[integrator.coarsest_lvl],
                     integrator.level_u_indices_elements,
                     integrator.coarsest_lvl)
    end
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::AbstractPairedExplicitRKIntegrator,
                      new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    # PERK stage
    resize!(integrator.k1, new_size)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::AbstractPairedExplicitRKMultiParabolicIntegrator,
                      new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    # PERK stage
    resize!(integrator.k1, new_size)
    # Addition for multirate PERK methods for parabolic problems
    resize!(integrator.du_tmp, new_size)
end

# get a cache where the RHS can be stored
get_du(integrator::AbstractPairedExplicitRKIntegrator) = integrator.du
get_tmp_cache(integrator::AbstractPairedExplicitRKIntegrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::AbstractPairedExplicitRKIntegrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::AbstractPairedExplicitRKIntegrator,
                          dt)
    (integrator.dt = dt; integrator.dtcache = dt)
end

function get_proposed_dt(integrator::AbstractPairedExplicitRKIntegrator)
    return ifelse(integrator.opts.adaptive, integrator.dt, integrator.dtcache)
end

# stop the time integration
function terminate!(integrator::AbstractPairedExplicitRKIntegrator)
    integrator.finalstep = true
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
end

# Add definitions of functions related to polynomial optimization by NLsolve here
# such that hey can be exported from Trixi.jl and extended in the TrixiConvexECOSExt package
# extension or by the NLsolve-specific code loaded by Requires.jl
function solve_a_butcher_coeffs_unknown! end

# Multirate/partitioned helpers
include("partitioning.jl")

# Paired Explicit Relaxation RK (PERRK) helpers
include("entropy_relaxation.jl")

include("p2/methods_PERK2.jl")
include("p3/methods_PERK3.jl")
include("p4/methods_PERK4.jl")
end # @muladd
