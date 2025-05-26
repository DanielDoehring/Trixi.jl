# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    ParametersEulerGravity(; background_density=0.0,
                             gravitational_constant=1.0,
                             cfl=1.0,
                             resid_tol=1.0e-4,
                             n_iterations_max=10^4,
                             timestep_gravity=timestep_gravity_erk52_3Sstar!)

Set up parameters for the gravitational part of a [`SemidiscretizationEulerGravity`](@ref).

# Arguments
- `background_density<:Real`: Constant background/reference density ρ₀ which is subtracted from the (Euler) density
                              in the RHS source term computation of the gravity solver.
- `gravitational_constant<:Real`: Gravitational constant G which needs to be in consistent units with the
                                  density and velocity fields. 
- `cfl<:Real`: CFL number used for the pseudo-time stepping to advance the hyperbolic diffusion equations into steady state.
- `resid_tol<:Real`: Absolute tolerance for the residual of the hyperbolic diffusion equations which are solved to 
                     (approximately) steady state.
- `n_iterations_max::Int`: Maximum number of iterations of the pseudo-time gravity solver. 
                           If `n_iterations <= 0` the solver will iterate until the residual is less or equal `resid_tol`. 
                           This can cause an infinite loop if the solver does not converge!
- `timestep_gravity`: Function to advance the gravity solver by one pseudo-time step.
                      There are three optimized methods available: 
                      1) `timestep_gravity_erk51_3Sstar!` (first-order),
                      2) `timestep_gravity_erk52_3Sstar!` (second-order),
                      3) `timestep_gravity_erk53_3Sstar!` (third-order).
                      Additionally, `timestep_gravity_carpenter_kennedy_erk54_2N!` (fourth-order) can be used.
"""
struct ParametersEulerGravity{RealT <: Real, TimestepGravity}
    background_density     :: RealT # aka rho0
    gravitational_constant :: RealT # aka G
    cfl                    :: RealT # CFL number for the gravity solver
    resid_tol              :: RealT # Hyp.-Diff. Eq. steady state tolerance
    n_iterations_max       :: Int   # Max. number of iterations of the pseudo-time gravity solver
    timestep_gravity       :: TimestepGravity
end

function ParametersEulerGravity(; background_density = 0.0,
                                gravitational_constant = 1.0,
                                cfl = 1.0,
                                resid_tol = 1.0e-4,
                                n_iterations_max = 10^4,
                                timestep_gravity = timestep_gravity_erk52_3Sstar!)
    background_density, gravitational_constant, cfl, resid_tol = promote(background_density,
                                                                         gravitational_constant,
                                                                         cfl, resid_tol)
    ParametersEulerGravity(background_density, gravitational_constant, cfl, resid_tol,
                           n_iterations_max, timestep_gravity)
end

function Base.show(io::IO, parameters::ParametersEulerGravity)
    @nospecialize parameters # reduce precompilation time

    print(io, "ParametersEulerGravity(")
    print(io, "background_density=", parameters.background_density)
    print(io, ", gravitational_constant=", parameters.gravitational_constant)
    print(io, ", cfl=", parameters.cfl)
    print(io, ", n_iterations_max=", parameters.n_iterations_max)
    print(io, ", timestep_gravity=", parameters.timestep_gravity)
    print(io, ")")
end
function Base.show(io::IO, ::MIME"text/plain", parameters::ParametersEulerGravity)
    @nospecialize parameters # reduce precompilation time

    if get(io, :compact, false)
        show(io, parameters)
    else
        setup = [
            "background density (ρ₀)" => parameters.background_density,
            "gravitational constant (G)" => parameters.gravitational_constant,
            "CFL (gravity)" => parameters.cfl,
            "max. #iterations" => parameters.n_iterations_max,
            "time integrator" => parameters.timestep_gravity
        ]
        summary_box(io, "ParametersEulerGravity", setup)
    end
end

"""
    SemidiscretizationEulerGravity

A struct containing everything needed to describe a spatial semidiscretization
of a the compressible Euler equations with self-gravity, reformulating the
Poisson equation for the gravitational potential as steady-state problem of
the hyperbolic diffusion equations.
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  "A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics"
  [arXiv: 2008.10593](https://arXiv.org/abs/2008.10593)
"""
struct SemidiscretizationEulerGravity{SemiEuler, SemiGravity,
                                      Parameters <: ParametersEulerGravity, Cache} <:
       AbstractSemidiscretization
    semi_euler          :: SemiEuler
    semi_gravity        :: SemiGravity
    parameters          :: Parameters
    performance_counter :: PerformanceCounter
    gravity_counter     :: PerformanceCounter
    cache               :: Cache

    function SemidiscretizationEulerGravity{SemiEuler, SemiGravity, Parameters, Cache}(semi_euler::SemiEuler,
                                                                                       semi_gravity::SemiGravity,
                                                                                       parameters::Parameters,
                                                                                       cache::Cache) where {
                                                                                                            SemiEuler,
                                                                                                            SemiGravity,
                                                                                                            Parameters <:
                                                                                                            ParametersEulerGravity,
                                                                                                            Cache
                                                                                                            }
        @assert ndims(semi_euler) == ndims(semi_gravity)
        @assert typeof(semi_euler.mesh) == typeof(semi_gravity.mesh)
        @assert polydeg(semi_euler.solver) == polydeg(semi_gravity.solver)

        performance_counter = PerformanceCounter()
        gravity_counter = PerformanceCounter()

        new(semi_euler, semi_gravity, parameters, performance_counter, gravity_counter,
            cache)
    end
end

"""
    SemidiscretizationEulerGravity(semi_euler::SemiEuler, semi_gravity::SemiGravity, parameters)

Construct a semidiscretization of the compressible Euler equations with self-gravity.
`parameters` should be given as [`ParametersEulerGravity`](@ref).
"""
function SemidiscretizationEulerGravity(semi_euler::SemiEuler,
                                        semi_gravity::SemiGravity,
                                        parameters) where
         {Mesh,
          SemiEuler <:
          SemidiscretizationHyperbolic{Mesh, <:AbstractCompressibleEulerEquations},
          SemiGravity <:
          SemidiscretizationHyperbolic{Mesh, <:AbstractHyperbolicDiffusionEquations}}
    u_ode = compute_coefficients(zero(real(semi_gravity)), semi_gravity)
    du_ode = similar(u_ode)
    # Registers for gravity solver, tailored to the 2N and 3S* methods implemented below
    u_tmp1_ode = similar(u_ode)
    u_tmp2_ode = similar(u_ode)
    cache = (; u_ode, du_ode, u_tmp1_ode, u_tmp2_ode)

    SemidiscretizationEulerGravity{typeof(semi_euler), typeof(semi_gravity),
                                   typeof(parameters), typeof(cache)}(semi_euler,
                                                                      semi_gravity,
                                                                      parameters, cache)
end

# Version for PERK. Currently only second and fourth-order methods are implemented!
function SemidiscretizationEulerGravity(semi_euler::SemiEuler,
                                        semi_gravity::SemiGravity,
                                        parameters,
                                        alg::Union{AbstractPairedExplicitRK{2},
                                                   AbstractPairedExplicitRK{4}}) where
         {Mesh,
          SemiEuler <:
          SemidiscretizationHyperbolic{Mesh, <:AbstractCompressibleEulerEquations},
          SemiGravity <:
          SemidiscretizationHyperbolic{Mesh, <:AbstractHyperbolicDiffusionEquations}}
    u_ode = compute_coefficients(zero(real(semi_gravity)), semi_gravity)
    du_ode = similar(u_ode)
    u_ode_tmp = similar(u_ode)
    k1 = similar(u_ode)

    # Association of unknowns of the gravity solver to levels
    level_u_gravity_indices_elements = [Vector{Int64}()
                                        for _ in 1:get_n_levels(semi_euler.mesh)]

    cache = (; alg, u_ode, du_ode, u_ode_tmp, k1,
             level_u_gravity_indices_elements)

    SemidiscretizationEulerGravity{typeof(semi_euler), typeof(semi_gravity),
                                   typeof(parameters), typeof(cache)}(semi_euler,
                                                                      semi_gravity,
                                                                      parameters, cache)
end

# Needed for algorithmic differentiation (Jacobian computation)
function remake(semi::SemidiscretizationEulerGravity;
                uEltype = real(semi.semi_gravity.solver),
                semi_euler = semi.semi_euler,
                semi_gravity = semi.semi_gravity,
                parameters = semi.parameters)
    semi_euler = remake(semi_euler, uEltype = uEltype)
    semi_gravity = remake(semi_gravity, uEltype = uEltype)

    # Recreate cache, i.e., registers for u with e.g. AD datatype
    u_ode = compute_coefficients(zero(real(semi_gravity)), semi_gravity)
    du_ode = similar(u_ode)
    u_tmp1_ode = similar(u_ode)
    u_tmp2_ode = similar(u_ode)
    cache = (; u_ode, du_ode, u_tmp1_ode, u_tmp2_ode)

    SemidiscretizationEulerGravity{typeof(semi_euler), typeof(semi_gravity),
                                   typeof(parameters), typeof(cache)}(semi_euler,
                                                                      semi_gravity,
                                                                      parameters, cache)
end

function Base.show(io::IO, semi::SemidiscretizationEulerGravity)
    @nospecialize semi # reduce precompilation time

    print(io, "SemidiscretizationEulerGravity using")
    print(io, semi.semi_euler)
    print(io, ", ", semi.semi_gravity)
    print(io, ", ", semi.parameters)
    print(io, ", cache(")
    for (idx, key) in enumerate(keys(semi.cache))
        idx > 1 && print(io, " ")
        print(io, key)
    end
    print(io, "))")
end

function Base.show(io::IO, mime::MIME"text/plain", semi::SemidiscretizationEulerGravity)
    @nospecialize semi # reduce precompilation time

    if get(io, :compact, false)
        show(io, semi)
    else
        summary_header(io, "SemidiscretizationEulerGravity")
        summary_line(io, "semidiscretization Euler",
                     semi.semi_euler |> typeof |> nameof)
        show(increment_indent(io), mime, semi.semi_euler)
        summary_line(io, "semidiscretization gravity",
                     semi.semi_gravity |> typeof |> nameof)
        show(increment_indent(io), mime, semi.semi_gravity)
        summary_line(io, "parameters", semi.parameters |> typeof |> nameof)
        show(increment_indent(io), mime, semi.parameters)
        summary_footer(io)
    end
end

# The compressible Euler semidiscretization is considered to be the main semidiscretization.
# The hyperbolic diffusion equations part is only used internally to update the gravitational
# potential during an rhs! evaluation of the flow solver.
@inline function mesh_equations_solver_cache(semi::SemidiscretizationEulerGravity)
    mesh_equations_solver_cache(semi.semi_euler)
end

@inline Base.ndims(semi::SemidiscretizationEulerGravity) = ndims(semi.semi_euler)

@inline Base.real(semi::SemidiscretizationEulerGravity) = real(semi.semi_euler)

# computes the coefficients of the initial condition
@inline function compute_coefficients(t, semi::SemidiscretizationEulerGravity)
    compute_coefficients!(semi.cache.u_ode, t, semi.semi_gravity)
    compute_coefficients(t, semi.semi_euler)
end

# computes the coefficients of the initial condition and stores the Euler part in `u_ode`
@inline function compute_coefficients!(u_ode, t, semi::SemidiscretizationEulerGravity)
    compute_coefficients!(semi.cache.u_ode, t, semi.semi_gravity)
    compute_coefficients!(u_ode, t, semi.semi_euler)
end

@inline function calc_error_norms(func, u, t, analyzer,
                                  semi::SemidiscretizationEulerGravity, cache_analysis)
    @unpack cache, semi_euler, semi_gravity = semi
    grav_l2, grav_linf, grav_l1 = calc_error_norms(func, cache.u_ode, t, analyzer,
                                                   semi_gravity, cache_analysis)

    println("Gravity L2 error: $grav_l2")
    println("Gravity L∞ error: $grav_linf")
    println("Gravity L1 error: $grav_l1")

    calc_error_norms(func, u, t, analyzer, semi.semi_euler, cache_analysis)
end

# TODO: Taal decide, where should specific parts like these be?
@inline function save_solution_file(u_ode, t, dt, iter,
                                    semi::SemidiscretizationEulerGravity,
                                    solution_callback,
                                    element_variables = Dict{Symbol, Any}();
                                    system = "")
    # If this is called already as part of a multi-system setup (i.e., system is non-empty),
    # we build a combined system name
    if !isempty(system)
        system_euler = system * "_euler"
        system_gravity = system * "_gravity"
    else
        system_euler = "euler"
        system_gravity = "gravity"
    end

    u_euler = wrap_array_native(u_ode, semi.semi_euler)
    filename_euler = save_solution_file(u_euler, t, dt, iter,
                                        mesh_equations_solver_cache(semi.semi_euler)...,
                                        solution_callback, element_variables,
                                        system = system_euler)

    u_gravity = wrap_array_native(semi.cache.u_ode, semi.semi_gravity)
    filename_gravity = save_solution_file(u_gravity, t, dt, iter,
                                          mesh_equations_solver_cache(semi.semi_gravity)...,
                                          solution_callback, element_variables,
                                          system = system_gravity)

    return filename_euler, filename_gravity
end

@inline function (amr_callback::AMRCallback)(u_ode,
                                             semi::SemidiscretizationEulerGravity,
                                             t, iter; kwargs...)
    @unpack cache = semi
    passive_args = ((cache.u_ode,
                     mesh_equations_solver_cache(semi.semi_gravity)...),)
    has_changed = amr_callback(u_ode, mesh_equations_solver_cache(semi.semi_euler)...,
                               semi, t, iter;
                               kwargs..., passive_args = passive_args)

    if has_changed
        new_length = length(cache.u_ode)
        resize!(cache.du_ode, new_length)

        fields_cache = fieldnames(typeof(cache))
        # 2N, 3S* integrators
        if :u_tmp1_ode in fields_cache
            resize!(cache.u_tmp1_ode, new_length)
        end
        if :u_tmp2_ode in fields_cache
            resize!(cache.u_tmp2_ode, new_length)
        end
        # PERK integrators
        if :u_ode_tmp in fields_cache
            resize!(cache.u_ode_tmp, new_length)
        end
        if :k1 in fields_cache
            resize!(cache.k1, new_length)
        end
    end

    return has_changed
end

# Coupled Euler and gravity solver at each Runge-Kutta stage, 
# corresponding to Algorithm 2 in Schlottke-Lakemper et al. (2020),
# https://dx.doi.org/10.1016/j.jcp.2021.110467
function rhs!(du_ode, u_ode, semi::SemidiscretizationEulerGravity, t)
    @unpack semi_euler, semi_gravity, cache = semi

    u_euler = wrap_array(u_ode, semi_euler)
    du_euler = wrap_array(du_ode, semi_euler)
    u_gravity = wrap_array(cache.u_ode, semi_gravity)
    n_elements = size(u_euler)[end]

    time_start = time_ns()

    # standard semidiscretization of the compressible Euler equations
    @trixi_timeit timer() "Euler solver" rhs!(du_ode, u_ode, semi_euler, t)

    # compute gravitational potential and forces
    @trixi_timeit timer() "gravity solver" update_gravity!(semi, u_ode)

    # add gravitational source source_terms to the Euler part
    if ndims(semi_euler) == 1
        @threaded for element in 1:n_elements
            @views @. du_euler[2, .., element] -= u_euler[1, .., element] *
                                                  u_gravity[2, .., element]
            @views @. du_euler[3, .., element] -= u_euler[2, .., element] *
                                                  u_gravity[2, .., element]
        end
    elseif ndims(semi_euler) == 2
        @threaded for element in 1:n_elements
            @views @. du_euler[2, .., element] -= u_euler[1, .., element] *
                                                  u_gravity[2, .., element]
            @views @. du_euler[3, .., element] -= u_euler[1, .., element] *
                                                  u_gravity[3, .., element]
            @views @. du_euler[4, .., element] -= (u_euler[2, .., element] *
                                                   u_gravity[2, .., element] +
                                                   u_euler[3, .., element] *
                                                   u_gravity[3, .., element])
        end
    elseif ndims(semi_euler) == 3
        @threaded for element in 1:n_elements
            @views @. du_euler[2, .., element] -= u_euler[1, .., element] *
                                                  u_gravity[2, .., element]
            @views @. du_euler[3, .., element] -= u_euler[1, .., element] *
                                                  u_gravity[3, .., element]
            @views @. du_euler[4, .., element] -= u_euler[1, .., element] *
                                                  u_gravity[4, .., element]
            @views @. du_euler[5, .., element] -= (u_euler[2, .., element] *
                                                   u_gravity[2, .., element] +
                                                   u_euler[3, .., element] *
                                                   u_gravity[3, .., element] +
                                                   u_euler[4, .., element] *
                                                   u_gravity[4, .., element])
        end
    else
        error("Number of dimensions $(ndims(semi_euler)) not supported.")
    end

    runtime = time_ns() - time_start
    put!(semi.performance_counter, runtime)

    return nothing
end

@inline function rhs!(du_ode, u_ode, semi::SemidiscretizationEulerGravity, t,
                      integrator::Union{AbstractPairedExplicitRKMultiIntegrator,
                                        AbstractPairedExplicitRelaxationRKMultiIntegrator},
                      max_level)
    rhs!(du_ode, u_ode, semi, t,
         max_level,
         integrator.level_info_elements_acc,
         integrator.level_info_interfaces_acc,
         integrator.level_info_boundaries_acc,
         #integrator.level_info_boundaries_orientation_acc,
         integrator.level_info_mortars_acc,
         integrator.n_levels) # Pass on `n_levels` to `rhs!` of a `SemidiscretizationEulerGravity` for PERKMulti timestep
end

function rhs!(du_ode, u_ode, semi::SemidiscretizationEulerGravity, t,
              max_level,
              level_info_elements_acc,
              level_info_interfaces_acc,
              level_info_boundaries_acc,
              #level_info_boundaries_orientation_acc,
              level_info_mortars_acc,
              n_levels)
    @unpack semi_euler, semi_gravity, cache = semi

    u_euler = wrap_array(u_ode, semi_euler)
    du_euler = wrap_array(du_ode, semi_euler)
    u_gravity = wrap_array(cache.u_ode, semi_gravity)

    time_start = time_ns()

    # standard semidiscretization of the compressible Euler equations
    @trixi_timeit timer() "Euler solver (part.)" rhs!(du_ode, u_ode, semi_euler, t,
                                                      level_info_elements_acc[max_level],
                                                      level_info_interfaces_acc[max_level],
                                                      level_info_boundaries_acc[max_level],
                                                      #level_info_boundaries_orientation_acc[max_level],
                                                      level_info_mortars_acc[max_level])

    # compute gravitational potential and forces
    @trixi_timeit timer() "gravity solver (part.)" update_gravity!(semi, u_ode,
                                                                   max_level,
                                                                   level_info_elements_acc,
                                                                   level_info_interfaces_acc,
                                                                   level_info_boundaries_acc,
                                                                   #level_info_boundaries_orientation_acc,
                                                                   level_info_mortars_acc,
                                                                   cache.level_u_gravity_indices_elements,
                                                                   n_levels)

    # add gravitational source source_terms to the Euler part
    if ndims(semi_euler) == 1
        @threaded for i in level_info_elements_acc[max_level]
            @views @. du_euler[2, .., i] -= u_euler[1, .., i] * u_gravity[2, .., i]
            @views @. du_euler[3, .., i] -= u_euler[2, .., i] * u_gravity[2, .., i]
        end
    elseif ndims(semi_euler) == 2
        @threaded for i in level_info_elements_acc[max_level]
            @views @. du_euler[2, .., i] -= u_euler[1, .., i] * u_gravity[2, .., i]
            @views @. du_euler[3, .., i] -= u_euler[1, .., i] * u_gravity[3, .., i]
            @views @. du_euler[4, .., i] -= (u_euler[2, .., i] * u_gravity[2, .., i] +
                                             u_euler[3, .., i] * u_gravity[3, .., i])
        end
    elseif ndims(semi_euler) == 3
        @threaded for i in level_info_elements_acc[max_level]
            @views @. du_euler[2, .., i] -= u_euler[1, .., i] * u_gravity[2, .., i]
            @views @. du_euler[3, .., i] -= u_euler[1, .., i] * u_gravity[3, .., i]
            @views @. du_euler[4, .., i] -= u_euler[1, .., i] * u_gravity[4, .., i]
            @views @. du_euler[5, .., i] -= (u_euler[2, .., i] * u_gravity[2, .., i] +
                                             u_euler[3, .., i] * u_gravity[3, .., i] +
                                             u_euler[4, .., i] * u_gravity[4, .., i])
        end
    else
        error("Number of dimensions $(ndims(semi_euler)) not supported.")
    end

    runtime = time_ns() - time_start
    put!(semi.performance_counter, runtime)

    return nothing
end

# TODO: Taal refactor, add some callbacks or so within the gravity update to allow investigating/optimizing it
function update_gravity!(semi::SemidiscretizationEulerGravity, u_ode)
    @unpack semi_euler, semi_gravity, parameters, gravity_counter, cache = semi

    u_euler = wrap_array(u_ode, semi_euler)
    u_gravity = wrap_array(cache.u_ode, semi_gravity)
    du_gravity = wrap_array(cache.du_ode, semi_gravity)

    # set up main loop
    finalstep = false
    @unpack n_iterations_max, cfl, resid_tol, timestep_gravity = parameters
    iter = 0
    tau = zero(real(semi_gravity.solver)) # Pseudo-time

    # iterate gravity solver until convergence or maximum number of iterations are reached
    @unpack equations = semi_gravity
    while !finalstep
        dtau = cfl * max_dt(u_gravity, tau, semi_gravity.mesh,
                      have_constant_speed(equations), equations,
                      semi_gravity.solver, semi_gravity.cache)

        # evolve solution by one pseudo-time step
        time_start = time_ns()
        timestep_gravity(cache, u_euler, tau, dtau, parameters, semi_gravity)
        runtime = time_ns() - time_start
        put!(gravity_counter, runtime)

        # update iteration counter
        iter += 1
        tau += dtau

        # check if we reached the maximum number of iterations
        if n_iterations_max > 0 && iter >= n_iterations_max
            #=
            @warn "Max iterations reached: Gravity solver failed to converge!" residual=maximum(abs,
                                                                                                @views du_gravity[1,
                                                                                                                  ..,
                                                                                                                  :]) tau=tau dtau=dtau
            =#
            finalstep = true
        end

        # this is an absolute tolerance check
        if maximum(abs, @views du_gravity[1, .., :]) <= resid_tol
            finalstep = true
        end
    end

    return nothing
end

function update_gravity!(semi::SemidiscretizationEulerGravity, u_ode,
                         max_level,
                         level_info_elements_acc,
                         level_info_interfaces_acc,
                         level_info_boundaries_acc,
                         #level_info_boundaries_orientation_acc,
                         level_info_mortars_acc,
                         level_u_gravity_indices_elements,
                         n_levels)
    @unpack semi_euler, semi_gravity, parameters, gravity_counter, cache = semi

    u_euler = wrap_array(u_ode, semi_euler)
    u_gravity = wrap_array(cache.u_ode, semi_gravity)
    du_gravity = wrap_array(cache.du_ode, semi_gravity)

    # set up main loop
    finalstep = false
    @unpack n_iterations_max, cfl, resid_tol, timestep_gravity = parameters
    iter = 0
    tau = zero(real(semi_gravity.solver)) # Pseudo-time

    # iterate gravity solver until convergence or maximum number of iterations are reached
    @unpack equations = semi_gravity
    while !finalstep
        dtau = cfl * max_dt(u_gravity, tau, semi_gravity.mesh,
                      have_constant_speed(equations), equations,
                      semi_gravity.solver, semi_gravity.cache)

        # evolve solution by one pseudo-time step
        time_start = time_ns()

        timestep_gravity(cache, u_euler, tau, dtau, parameters, semi_gravity,
                         level_info_elements_acc,
                         level_info_interfaces_acc,
                         level_info_boundaries_acc,
                         #level_info_boundaries_orientation_acc,
                         level_info_mortars_acc,
                         level_u_gravity_indices_elements,
                         n_levels)

        runtime = time_ns() - time_start
        put!(gravity_counter, runtime)

        # update iteration counter
        iter += 1
        tau += dtau

        # check if we reached the maximum number of iterations
        if n_iterations_max > 0 && iter >= n_iterations_max
            #=
            @warn "Max iterations reached: Gravity solver failed to converge!" residual=maximum(abs,
                                                                                                @views du_gravity[1,
                                                                                                                  ..,
                                                                                                                  :]) tau=tau dtau=dtau
            =#
            finalstep = true
        end

        # this is an absolute tolerance check
        # Note: Althugh the gravitational sources are given by an elliptic equation, 
        # we do not really need global convergence as we add only the local contributions
        if maximum(abs, @views du_gravity[1, .., level_info_elements_acc[max_level]]) <=
           resid_tol
            finalstep = true
        end
    end

    return nothing
end

# Integrate gravity solver for 2N-type low-storage schemes
function timestep_gravity_2N!(cache, u_euler, tau, dtau, gravity_parameters,
                              semi_gravity,
                              a, b, c)
    G = gravity_parameters.gravitational_constant
    rho0 = gravity_parameters.background_density
    grav_scale = -4.0 * pi * G

    # Note that `u_ode` is `u_gravity` in `rhs!` above
    @unpack u_ode, du_ode, u_tmp1_ode = cache
    n_elements = size(u_euler)[end]

    u_tmp1_ode .= zero(eltype(u_tmp1_ode))
    du_gravity = wrap_array(du_ode, semi_gravity)

    for stage in eachindex(c)
        tau_stage = tau + dtau * c[stage]

        # rhs! has the source term for the harmonic problem
        # We don't need a `@trixi_timeit timer() "rhs!"` here since that's already
        # included in the `rhs!` call.
        rhs!(du_ode, u_ode, semi_gravity, tau_stage)

        # Source term: Jeans instability OR coupling convergence test OR blast wave
        # put in gravity source term proportional to Euler density
        # OBS! subtract off the background density ρ_0 (spatial mean value)
        # Note: Adding to `du_gravity` is essentially adding to `du_ode`!
        @threaded for element in 1:n_elements
            @views @. du_gravity[1, .., element] += grav_scale *
                                                    (u_euler[1, .., element] - rho0)
        end

        a_stage = a[stage]
        b_stage_dt = b[stage] * dtau
        @trixi_timeit timer() "Runge-Kutta step" begin
            @threaded for idx in eachindex(u_ode)
                u_tmp1_ode[idx] = du_ode[idx] - u_tmp1_ode[idx] * a_stage
                u_ode[idx] += u_tmp1_ode[idx] * b_stage_dt
            end
        end
    end

    return nothing
end

function timestep_gravity_carpenter_kennedy_erk54_2N!(cache, u_euler, tau, dtau,
                                                      gravity_parameters, semi_gravity)
    # Coefficients for Carpenter's 5-stage 4th-order low-storage Runge-Kutta method
    a = SVector(0.0,
                567301805773.0 / 1357537059087.0,
                2404267990393.0 / 2016746695238.0,
                3550918686646.0 / 2091501179385.0,
                1275806237668.0 / 842570457699.0)
    b = SVector(1432997174477.0 / 9575080441755.0,
                5161836677717.0 / 13612068292357.0,
                1720146321549.0 / 2090206949498.0,
                3134564353537.0 / 4481467310338.0,
                2277821191437.0 / 14882151754819.0)
    c = SVector(0.0,
                1432997174477.0 / 9575080441755.0,
                2526269341429.0 / 6820363962896.0,
                2006345519317.0 / 3224310063776.0,
                2802321613138.0 / 2924317926251.0)

    timestep_gravity_2N!(cache, u_euler, tau, dtau, gravity_parameters, semi_gravity,
                         a, b, c)
end

# Integrate gravity solver for 3S*-type low-storage schemes
function timestep_gravity_3Sstar!(cache, u_euler, tau, dtau, gravity_parameters,
                                  semi_gravity,
                                  gamma1, gamma2, gamma3, beta, delta, c)
    G = gravity_parameters.gravitational_constant
    rho0 = gravity_parameters.background_density
    grav_scale = -4 * G * pi

    # `u_ode` is `u_gravity` in coupled RHS above!
    @unpack u_ode, du_ode, u_tmp1_ode, u_tmp2_ode = cache
    n_elements = size(u_euler)[end]

    u_tmp1_ode .= zero(eltype(u_tmp1_ode))
    u_tmp2_ode .= u_ode
    du_gravity = wrap_array(du_ode, semi_gravity)

    for stage in eachindex(c)
        tau_stage = tau + dtau * c[stage]

        # rhs! has the source term for the harmonic problem
        # We don't need a `@trixi_timeit timer() "rhs!"` here since that's already
        # included in the `rhs!` call.
        rhs!(du_ode, u_ode, semi_gravity, tau_stage)

        # Source term: Jeans instability OR coupling convergence test OR blast wave
        # put in gravity source term proportional to Euler density
        # OBS! subtract off the background density ρ_0 around which the Jeans instability is perturbed
        # Note: Adding to `du_gravity` is essentially adding to `du_ode`!
        @threaded for element in 1:n_elements
            @views @. du_gravity[1, .., element] += grav_scale *
                                                    (u_euler[1, .., element] - rho0)
        end

        delta_stage = delta[stage]
        gamma1_stage = gamma1[stage]
        gamma2_stage = gamma2[stage]
        gamma3_stage = gamma3[stage]
        beta_stage_dt = beta[stage] * dtau
        @trixi_timeit timer() "Runge-Kutta step" begin
            @threaded for idx in eachindex(u_ode)
                # See Algorithm 1 (3S* method) in Schlottke-Lakemper et al. (2020)
                u_tmp1_ode[idx] += delta_stage * u_ode[idx]
                u_ode[idx] = (gamma1_stage * u_ode[idx] +
                              gamma2_stage * u_tmp1_ode[idx] +
                              gamma3_stage * u_tmp2_ode[idx] +
                              beta_stage_dt * du_ode[idx])
            end
        end
    end

    return nothing
end

# First-order 5-stage, 3S*-storage optimized method
function timestep_gravity_erk51_3Sstar!(cache, u_euler, tau, dtau, gravity_parameters,
                                        semi_gravity)
    # New 3Sstar coefficients optimized for polynomials of degree polydeg=3
    # and examples/parameters_hypdiff_lax_friedrichs.toml
    # 5 stages, order 1
    gamma1 = SVector(0.0000000000000000E+00, 5.2910412316555866E-01,
                     2.8433964362349406E-01, -1.4467571130907027E+00,
                     7.5592215948661057E-02)
    gamma2 = SVector(1.0000000000000000E+00, 2.6366970460864109E-01,
                     3.7423646095836322E-01, 7.8786901832431289E-01,
                     3.7754129043053775E-01)
    gamma3 = SVector(0.0000000000000000E+00, 0.0000000000000000E+00,
                     0.0000000000000000E+00, 8.0043329115077388E-01,
                     1.3550099149374278E-01)
    beta = SVector(1.9189497208340553E-01, 5.4506406707700059E-02,
                   1.2103893164085415E-01, 6.8582252490550921E-01,
                   8.7914657211972225E-01)
    delta = SVector(1.0000000000000000E+00, 7.8593091509463076E-01,
                    1.2639038717454840E-01, 1.7726945920209813E-01,
                    0.0000000000000000E+00)
    c = SVector(0.0000000000000000E+00, 1.9189497208340553E-01, 1.9580448818599061E-01,
                2.4241635859769023E-01, 5.0728347557552977E-01)

    timestep_gravity_3Sstar!(cache, u_euler, tau, dtau, gravity_parameters,
                             semi_gravity,
                             gamma1, gamma2, gamma3, beta, delta, c)
end

# Second-order 5-stage, 3S*-storage optimized method
function timestep_gravity_erk52_3Sstar!(cache, u_euler, tau, dtau, gravity_parameters,
                                        semi_gravity)
    # New 3Sstar coefficients optimized for polynomials of degree polydeg=3
    # and examples/parameters_hypdiff_lax_friedrichs.toml
    # 5 stages, order 2
    gamma1 = SVector(0.0000000000000000E+00, 5.2656474556752575E-01,
                     1.0385212774098265E+00, 3.6859755007388034E-01,
                     -6.3350615190506088E-01)
    gamma2 = SVector(1.0000000000000000E+00, 4.1892580153419307E-01,
                     -2.7595818152587825E-02, 9.1271323651988631E-02,
                     6.8495995159465062E-01)
    gamma3 = SVector(0.0000000000000000E+00, 0.0000000000000000E+00,
                     0.0000000000000000E+00, 4.1301005663300466E-01,
                     -5.4537881202277507E-03)
    beta = SVector(4.5158640252832094E-01, 7.5974836561844006E-01,
                   3.7561630338850771E-01, 2.9356700007428856E-02,
                   2.5205285143494666E-01)
    delta = SVector(1.0000000000000000E+00, 1.3011720142005145E-01,
                    2.6579275844515687E-01, 9.9687218193685878E-01,
                    0.0000000000000000E+00)
    c = SVector(0.0000000000000000E+00, 4.5158640252832094E-01, 1.0221535725056414E+00,
                1.4280257701954349E+00, 7.1581334196229851E-01)

    timestep_gravity_3Sstar!(cache, u_euler, tau, dtau, gravity_parameters,
                             semi_gravity,
                             gamma1, gamma2, gamma3, beta, delta, c)
end

# Third-order 5-stage, 3S*-storage optimized method
function timestep_gravity_erk53_3Sstar!(cache, u_euler, tau, dtau, gravity_parameters,
                                        semi_gravity)
    # New 3Sstar coefficients optimized for polynomials of degree polydeg=3
    # and examples/parameters_hypdiff_lax_friedrichs.toml
    # 5 stages, order 3
    gamma1 = SVector(0.0000000000000000E+00, 6.9362208054011210E-01,
                     9.1364483229179472E-01, 1.3129305757628569E+00,
                     -1.4615811339132949E+00)
    gamma2 = SVector(1.0000000000000000E+00, 1.3224582239681788E+00,
                     2.4213162353103135E-01, -3.8532017293685838E-01,
                     1.5603355704723714E+00)
    gamma3 = SVector(0.0000000000000000E+00, 0.0000000000000000E+00,
                     0.0000000000000000E+00, 3.8306787039991996E-01,
                     -3.5683121201711010E-01)
    beta = SVector(8.4476964977404881E-02, 3.0834660698015803E-01,
                   3.2131664733089232E-01, 2.8783574345390539E-01,
                   8.2199204703236073E-01)
    delta = SVector(1.0000000000000000E+00, -7.6832695815481578E-01,
                    1.2497251501714818E-01, 1.4496404749796306E+00,
                    0.0000000000000000E+00)
    c = SVector(0.0000000000000000E+00, 8.4476964977404881E-02, 2.8110631488732202E-01,
                5.7093842145029405E-01, 7.2999896418559662E-01)

    timestep_gravity_3Sstar!(cache, u_euler, tau, dtau, gravity_parameters,
                             semi_gravity,
                             gamma1, gamma2, gamma3, beta, delta, c)
end

### Paired-Explicit Runge-Kutta (PERK) methods ###

@inline function PERK_k1!(k1, u_gravity, semi_gravity, tau)
    rhs!(k1, u_gravity, semi_gravity, tau)
end

@inline function PERK_k2!(du, u_tmp, u, k1, semi_gravity, dtau, tau, alg)
    @threaded for i in eachindex(u)
        u_tmp[i] = u[i] + alg.c[2] * dtau * k1[i]
    end

    rhs!(du, u_tmp, semi_gravity, tau + alg.c[2] * dtau)
end

@inline function PERK_k2!(du, u_tmp, u, k1, semi_gravity, dtau, tau, alg,
                          level_info_elements_acc,
                          level_info_interfaces_acc,
                          level_info_boundaries_acc,
                          #level_info_boundaries_orientation_acc,
                          level_info_mortars_acc)
    @threaded for i in eachindex(u)
        u_tmp[i] = u[i] + alg.c[2] * dtau * k1[i]
    end

    rhs!(du, u_tmp, semi_gravity, tau + alg.c[2] * dtau,
         level_info_elements_acc[1],
         level_info_interfaces_acc[1],
         level_info_boundaries_acc[1],
         #level_info_boundaries_orientation_acc[1],
         level_info_mortars_acc[1])
end

@inline function PERK_ki!(du, u_tmp, u, k1, semi_gravity, dtau, tau, alg, stage)
    # Construct current state
    @threaded for i in eachindex(u)
        u_tmp[i] = u[i] +
                   dtau *
                   (alg.a_matrix[1, stage - 2] * k1[i] +
                    alg.a_matrix[2, stage - 2] * du[i])
    end

    rhs!(du, u_tmp, semi_gravity, tau + alg.c[stage] * dtau)
end

@inline function PERK_ki!(du, u_tmp, u, k1, semi_gravity, dtau, tau,
                          alg::AbstractPairedExplicitRKMulti, stage)
    # Construct current state
    @threaded for i in eachindex(u)
        u_tmp[i] = u[i] +
                   dtau *
                   (alg.a_matrices[1, 1, stage - 2] * k1[i] +
                    alg.a_matrices[1, 2, stage - 2] * du[i])
    end

    rhs!(du, u_tmp, semi_gravity, tau + alg.c[stage] * dtau)
end

@inline function PERKMulti_intermediate_stage!(du, u_tmp, u, k1, dtau, alg, stage,
                                               level_u_indices_elements, n_levels)
    ### General implementation: Not own method for each grid level ###
    # Loop over different methods with own associated level

    for level in 1:min(alg.num_methods, n_levels)
        @threaded for i in level_u_indices_elements[level]
            u_tmp[i] = u[i] +
                       dtau * alg.a_matrices[level, 1, stage - 2] * k1[i]
        end
    end
    for level in 1:min(alg.max_add_levels[stage], n_levels)
        @threaded for i in level_u_indices_elements[level]
            u_tmp[i] += dtau * alg.a_matrices[level, 2, stage - 2] * du[i]
        end
    end

    # "Remainder": Non-efficiently integrated
    for level in (alg.num_methods + 1):(n_levels)
        @threaded for i in level_u_indices_elements[level]
            u_tmp[i] = u[i] +
                       dtau * alg.a_matrices[alg.num_methods, 1, stage - 2] * k1[i]
        end
    end
    if alg.max_add_levels[stage] == alg.num_methods
        for level in (alg.max_add_levels[stage] + 1):n_levels
            @threaded for i in level_u_indices_elements[level]
                u_tmp[i] += dtau * alg.a_matrices[alg.num_methods, 2, stage - 2] * du[i]
            end
        end
    end

    ### Simplified implementation: Own method for each level ###
    #=
    for level in 1:n_levels
        @threaded for i in level_u_indices_elements[level]
            u_tmp[i] = u[i] + dtau * alg.a_matrices[level, 1, stage - 2] * k1[i]
        end
    end
    for level in 1:alg.max_add_levels[stage]
        @threaded for i in level_u_indices_elements[level]
            u_tmp[i] += dtau * alg.a_matrices[level, 2, stage - 2] * du[i]
        end
    end
    =#

    # For statically non-uniform meshes/characteristic speeds
    #coarsest_lvl = alg.max_active_levels[stage]

    # "coarsest_lvl" cannot be static for AMR, has to be checked with available levels
    coarsest_lvl = min(alg.max_active_levels[stage], n_levels)

    return coarsest_lvl
end

@inline function PERK_ki!(du, u_tmp, u, k1, semi_gravity, dtau, tau, alg, stage,
                          level_info_elements_acc,
                          level_info_interfaces_acc,
                          level_info_boundaries_acc,
                          #level_info_boundaries_orientation_acc,
                          level_info_mortars_acc,
                          level_u_indices_elements,
                          n_levels,
                          n_elements, grav_scale, rho0, u_euler)
    coarsest_lvl = PERKMulti_intermediate_stage!(du, u_tmp, u, k1, dtau, alg, stage,
                                                 level_u_indices_elements, n_levels)

    du_wrap = wrap_array(du, semi_gravity)

    # Check if there are fewer integrators than grid levels (non-optimal method)
    if coarsest_lvl == alg.num_methods
        # NOTE: This is supposedly more efficient than setting
        #integrator.coarsest_lvl = integrator.n_levels
        # and then using the level-dependent version

        rhs!(du, u_tmp, semi_gravity, tau + alg.c[stage] * dtau)

        @threaded for i in 1:n_elements
            @views @. du_wrap[1, .., i] += grav_scale *
                                           (u_euler[1, .., i] - rho0)
        end
    else
        rhs!(du, u_tmp, semi_gravity, tau + alg.c[stage] * dtau,
             level_info_elements_acc[coarsest_lvl],
             level_info_interfaces_acc[coarsest_lvl],
             level_info_boundaries_acc[coarsest_lvl],
             #level_info_boundaries_orientation_acc[coarsest_lvl],
             level_info_mortars_acc[coarsest_lvl])

        @threaded for i in level_info_elements_acc[coarsest_lvl]
            @views @. du_wrap[1, .., i] += grav_scale *
                                           (u_euler[1, .., i] - rho0)
        end
    end
end

@inline function PERK4_kS2_to_kS!(du, u_tmp, u, k1, semi_gravity, dtau, tau, alg,
                                  n_elements, grav_scale, rho0, u_euler)
    du_wrap = wrap_array(du, semi_gravity)

    for stage in 1:2
        @threaded for i in eachindex(u)
            u_tmp[i] = u[i] +
                       dtau *
                       (alg.a_matrix_constant[1, stage] * k1[i] +
                        alg.a_matrix_constant[2, stage] * du[i])
        end

        rhs!(du, u_tmp, semi_gravity, tau + alg.c[alg.num_stages - 3 + stage] * dtau)

        @threaded for i in 1:n_elements
            @views @. du_wrap[1, .., i] += grav_scale * (u_euler[1, .., i] - rho0)
        end
    end

    # Last stage
    @threaded for i in eachindex(u)
        u_tmp[i] = u[i] +
                   dtau *
                   (alg.a_matrix_constant[1, 3] * k1[i] +
                    alg.a_matrix_constant[2, 3] * du[i])
    end

    # Store K_{S-1} in `k1`:
    k1 .= du

    rhs!(du, u_tmp, semi_gravity, tau + alg.c[alg.num_stages] * dtau)
    @threaded for i in 1:n_elements
        @views @. du_wrap[1, .., i] += grav_scale * (u_euler[1, .., i] - rho0)
    end

    @threaded for i in eachindex(u)
        # Note that 'k1' carries the values of K_{S-1}
        # and that we construct 'K_S' "in-place" from 'du'
        u[i] += 0.5 * dtau * (k1[i] + du[i])
    end
end

function timestep_gravity_PERK2!(cache, u_euler, tau, dtau,
                                 gravity_parameters, semi_gravity)
    G = gravity_parameters.gravitational_constant
    rho0 = gravity_parameters.background_density
    grav_scale = -4 * G * pi

    # `u_ode` is `u_gravity` in coupled RHS above!
    @unpack alg, u_ode, du_ode, u_ode_tmp, k1 = cache

    n_elements = size(u_euler)[end]
    du_wrap = wrap_array(du_ode, semi_gravity)
    k1_wrap = wrap_array(k1, semi_gravity)

    PERK_k1!(k1, u_ode, semi_gravity, tau)
    # Stage contains source from other semidiscretization, i.e., Euler.
    @threaded for i in 1:n_elements
        @views @. k1_wrap[1, .., i] += grav_scale * (u_euler[1, .., i] - rho0)
    end

    PERK_k2!(du_ode, u_ode_tmp, u_ode, k1, semi_gravity, dtau, tau, alg)
    @threaded for i in 1:n_elements
        @views @. du_wrap[1, .., i] += grav_scale * (u_euler[1, .., i] - rho0)
    end

    for stage in 3:(alg.num_stages)
        PERK_ki!(du_ode, u_ode_tmp, u_ode, k1, semi_gravity, dtau, tau, alg, stage)
        @threaded for i in 1:n_elements
            @views @. du_wrap[1, .., i] += grav_scale * (u_euler[1, .., i] - rho0)
        end
    end

    @threaded for i in eachindex(u_ode)
        u_ode[i] += dtau * (alg.b1 * k1[i] + alg.bS * du_ode[i])
        # Slightly more performant, hard-coded version for b1 = 0
        #u_ode[i] += dtau * du_ode[i]
    end
end

# Required for partitioned Euler solver + standard gravity solver
@inline function timestep_gravity_PERK2!(cache, u_euler, tau, dtau,
                                         gravity_parameters, semi_gravity,
                                         level_info_elements_acc,
                                         level_info_interfaces_acc,
                                         level_info_boundaries_acc,
                                         #level_info_boundaries_orientation_acc,
                                         level_info_mortars_acc,
                                         level_u_gravity_indices_elements,
                                         n_levels)
    timestep_gravity_PERK2!(cache, u_euler, tau, dtau, gravity_parameters, semi_gravity)
end

# Need this version for the gravity steps we take with the entire mesh
@inline function timestep_gravity_PERK2_Multi!(cache, u_euler, tau, dtau,
                                               gravity_parameters, semi_gravity)
    timestep_gravity_PERK2!(cache, u_euler, tau, dtau, gravity_parameters, semi_gravity)
end

function timestep_gravity_PERK2_Multi!(cache, u_euler, tau, dtau,
                                       gravity_parameters, semi_gravity,
                                       level_info_elements_acc,
                                       level_info_interfaces_acc,
                                       level_info_boundaries_acc,
                                       #level_info_boundaries_orientation_acc,
                                       level_info_mortars_acc,
                                       level_u_gravity_indices_elements,
                                       n_levels)
    G = gravity_parameters.gravitational_constant
    rho0 = gravity_parameters.background_density
    grav_scale = -4 * G * pi

    # `u_ode` is `u_gravity` in coupled RHS above!
    @unpack alg, u_ode, du_ode, u_ode_tmp, k1 = cache

    n_elements = size(u_euler)[end]
    du_wrap = wrap_array(du_ode, semi_gravity)
    k1_wrap = wrap_array(k1, semi_gravity)

    PERK_k1!(k1, u_ode, semi_gravity, tau)
    # Stage contains source from other semidiscretization, i.e., Euler.
    @threaded for i in 1:n_elements
        @views @. k1_wrap[1, .., i] += grav_scale * (u_euler[1, .., i] - rho0)
    end

    PERK_k2!(du_ode, u_ode_tmp, u_ode, k1, semi_gravity, dtau, tau, alg,
             level_info_elements_acc,
             level_info_interfaces_acc,
             level_info_boundaries_acc,
             #level_info_boundaries_orientation_acc,
             level_info_mortars_acc)

    @threaded for i in level_info_elements_acc[1]
        @views @. du_wrap[1, .., i] += grav_scale * (u_euler[1, .., i] - rho0)
    end

    for stage in 3:(alg.num_stages)
        PERK_ki!(du_ode, u_ode_tmp, u_ode, k1, semi_gravity, dtau, tau, alg, stage,
                 level_info_elements_acc,
                 level_info_interfaces_acc,
                 level_info_boundaries_acc,
                 #level_info_boundaries_orientation_acc,
                 level_info_mortars_acc,
                 level_u_gravity_indices_elements,
                 n_levels,
                 n_elements, grav_scale, rho0, u_euler)
    end

    # u_{n+1} = u_n + b_S * k_S = u_n + 1 * k_S
    @threaded for i in eachindex(u_ode)
        u_ode[i] += dtau * (alg.b1 * k1[i] + alg.bS * du_ode[i])
        # Slightly more performant, hard-coded version for b1 = 0
        #u_ode[i] += dtau * du_ode[i]
    end
end

function timestep_gravity_PERK4!(cache, u_euler, tau, dtau,
                                 gravity_parameters, semi_gravity)
    G = gravity_parameters.gravitational_constant
    rho0 = gravity_parameters.background_density
    grav_scale = -4 * G * pi

    # `u_ode` is `u_gravity` in coupled RHS above!
    @unpack alg, u_ode, du_ode, u_ode_tmp, k1 = cache

    n_elements = size(u_euler)[end]
    du_wrap = wrap_array(du_ode, semi_gravity)
    k1_wrap = wrap_array(k1, semi_gravity)

    PERK_k1!(k1, u_ode, semi_gravity, tau)
    # Stage contains source from other semidiscretization, i.e., Euler.
    @threaded for i in 1:n_elements
        @views @. k1_wrap[1, .., i] += grav_scale * (u_euler[1, .., i] - rho0)
    end

    PERK_k2!(du_ode, u_ode_tmp, u_ode, k1, semi_gravity, dtau, tau, alg)
    @threaded for i in 1:n_elements
        @views @. du_wrap[1, .., i] += grav_scale * (u_euler[1, .., i] - rho0)
    end

    for stage in 3:(alg.num_stages - 3)
        PERK_ki!(du_ode, u_ode_tmp, u_ode, k1, semi_gravity, dtau, tau, alg, stage)
        @threaded for i in 1:n_elements
            @views @. du_wrap[1, .., i] += grav_scale * (u_euler[1, .., i] - rho0)
        end
    end

    PERK4_kS2_to_kS!(du_ode, u_ode_tmp, u_ode, k1, semi_gravity, dtau, tau, alg,
                     n_elements, grav_scale, rho0, u_euler)
end

# Required for partitioned Euler solver + standard gravity solver
@inline function timestep_gravity_PERK4!(cache, u_euler, tau, dtau,
                                         gravity_parameters, semi_gravity,
                                         level_info_elements_acc,
                                         level_info_interfaces_acc,
                                         level_info_boundaries_acc,
                                         #level_info_boundaries_orientation_acc,
                                         level_info_mortars_acc,
                                         level_u_gravity_indices_elements,
                                         n_levels)
    timestep_gravity_PERK4!(cache, u_euler, tau, dtau, gravity_parameters, semi_gravity)
end

# Need this version for the gravity steps we take with the entire mesh
@inline function timestep_gravity_PERK4_Multi!(cache, u_euler, tau, dtau,
                                               gravity_parameters, semi_gravity)
    timestep_gravity_PERK4!(cache, u_euler, tau, dtau, gravity_parameters, semi_gravity)
end

function timestep_gravity_PERK4_Multi!(cache, u_euler, tau, dtau,
                                       gravity_parameters, semi_gravity,
                                       level_info_elements_acc,
                                       level_info_interfaces_acc,
                                       level_info_boundaries_acc,
                                       #level_info_boundaries_orientation_acc,
                                       level_info_mortars_acc,
                                       level_u_gravity_indices_elements,
                                       n_levels)
    G = gravity_parameters.gravitational_constant
    rho0 = gravity_parameters.background_density
    grav_scale = -4 * G * pi

    # `u_ode` is `u_gravity` in coupled RHS above!
    @unpack alg, u_ode, du_ode, u_ode_tmp, k1 = cache

    n_elements = size(u_euler)[end]
    du_wrap = wrap_array(du_ode, semi_gravity)
    k1_wrap = wrap_array(k1, semi_gravity)

    PERK_k1!(k1, u_ode, semi_gravity, tau)
    # Stage contains source from other semidiscretization, i.e., Euler.
    @threaded for i in 1:n_elements
        @views @. k1_wrap[1, .., i] += grav_scale * (u_euler[1, .., i] - rho0)
    end

    PERK_k2!(du_ode, u_ode_tmp, u_ode, k1, semi_gravity, dtau, tau, alg,
             level_info_elements_acc,
             level_info_interfaces_acc,
             level_info_boundaries_acc,
             #level_info_boundaries_orientation_acc,
             level_info_mortars_acc)

    @threaded for i in level_info_elements_acc[1]
        @views @. du_wrap[1, .., i] += grav_scale * (u_euler[1, .., i] - rho0)
    end

    for stage in 3:(alg.num_stages - 3)
        PERK_ki!(du_ode, u_ode_tmp, u_ode, k1, semi_gravity, dtau, tau, alg, stage,
                 level_info_elements_acc,
                 level_info_interfaces_acc,
                 level_info_boundaries_acc,
                 #level_info_boundaries_orientation_acc,
                 level_info_mortars_acc,
                 level_u_gravity_indices_elements,
                 n_levels,
                 n_elements, grav_scale, rho0, u_euler)
    end

    PERK4_kS2_to_kS!(du_ode, u_ode_tmp, u_ode, k1, semi_gravity, dtau, tau, alg,
                     n_elements, grav_scale, rho0, u_euler)
end
end # @muladd
