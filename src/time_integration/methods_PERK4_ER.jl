# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct PERK4_ER{StageCallbacks}
    const NumStages::Int64
    stage_callbacks::StageCallbacks

    AMatrices::Matrix{Float64}
    AMatrix::Matrix{Float64}
    c::Vector{Float64}

    function PERK4_ER(NumStages_::Int, BasePathMonCoeffs_::AbstractString,
                stage_callbacks = ())
    newPERK4 = new{typeof(stage_callbacks)}(NumStages_, stage_callbacks)

    newPERK4.AMatrices, newPERK4.AMatrix, newPERK4.c = ComputePERK4_ButcherTableau(NumStages_,
                                                                                    BasePathMonCoeffs_)

    return newPERK4
    end
end # struct PERK4

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PERK4_ER_Integrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                   PERK_IntegratorOptions}
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
    opts::PERK_IntegratorOptions
    finalstep::Bool # added for convenience
    # PERK4 stages:
    k1::uType
    k_higher::uType
    t_stage::RealT

    
    # TODO: Not best solution since this is not needed for hyperbolic problems
    du_ode_hyp::uType 

    # TODO uprev, tprev for averaging callback (required for coupled Euler-acoustic simulations)
    #uprev::uType
    #tprev::RealT

    # Entropy Relaxation additions
    direction::uType
    num_timestep_relaxations::Int
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::PERK4_ER_Integrator, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

function init(ode::ODEProblem, alg::PERK4_ER;
              dt, callback = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    # PERK4 stages
    k1 = zero(u0)
    k_higher = zero(u0)

    du_ode_hyp = zero(u0) # TODO: Not best solution since this is not needed for hyperbolic problems

    
    # TODO: Only for averaging callback (required for coupled Euler-acoustic simulations)
    #uprev = zero(u0)
    #tprev = zero(ode.tspan[1])
    
    # For entropy relaxation
    direction = zero(u0)

    t0 = first(ode.tspan)
    iter = 0

    integrator = PERK4_ER_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                                  (prob = ode,), ode.f, alg,
                                  PERK_IntegratorOptions(callback, ode.tspan;
                                                         kwargs...), false,
                                  k1, k_higher, t0,
                                  du_ode_hyp,
                                  #uprev, tprev)
                                  direction, 0)

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

function int_w_dot_stage(stage, u_i,
                     mesh::Union{TreeMesh{1}, StructuredMesh{1}}, equations, dg::DG,
                     cache)
    # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
    integrate_via_indices(u_i, mesh, equations, dg, cache,
                          stage) do u_i, i, element, equations, dg, stage
        w_node = cons2entropy(get_node_vars(u_i, equations, dg, i, element),
                              equations)
        stage_node = get_node_vars(stage, equations, dg, i, element)
        dot(w_node, stage_node)
    end
end

function int_w_dot_stage(stage, u_i,
                     mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                 UnstructuredMesh2D, P4estMesh{2}, T8codeMesh{2}},
                     equations, dg::DG, cache)
    # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
    integrate_via_indices(u_i, mesh, equations, dg, cache,
                          stage) do u_i, i, j, element, equations, dg, stage
        w_node = cons2entropy(get_node_vars(u_i, equations, dg, i, j, element),
                              equations)
        stage_node = get_node_vars(stage, equations, dg, i, j, element)
        dot(w_node, stage_node)
    end
end

function int_w_dot_stage(stage, u_i,
                     mesh::Union{TreeMesh{3}, StructuredMesh{3}, P4estMesh{3},
                                 T8codeMesh{3}},
                     equations, dg::DG, cache)
    # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
    integrate_via_indices(u_i, mesh, equations, dg, cache,
                          stage) do u_i, i, j, k, element, equations, dg, stage
        w_node = cons2entropy(get_node_vars(u_i, equations, dg, i, j, k, element),
                              equations)
        stage_node = get_node_vars(stage, equations, dg, i, j, k, element)
        dot(w_node, stage_node)
    end
end

function entropy_diff(gamma, S_old, dS, u_gamma_dir, mesh, equations, dg, cache)
    return integrate(entropy_math, u_gamma_dir, mesh, equations, dg, cache) -
           S_old - gamma * dS
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::PERK4_ER;
               dt, callback = nothing, kwargs...)
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    solve_steps!(integrator)
end

function solve_steps!(integrator::PERK4_ER_Integrator)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
        #=
        # NOTE: `prev` For EulerAcoustics only
        @threaded for u_ind in eachindex(integrator.u)
            integrator.uprev[u_ind] = integrator.u[u_ind]
        end
        integrator.tprev = integrator.t
        =#

        step!(integrator)
    end # "main loop" timer

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

function k1!(integrator, p, c)
    #integrator.f(integrator.du, integrator.u, p, integrator.t, integrator.du_ode_hyp)
    integrator.f(integrator.du, integrator.u, p, integrator.t)

    @threaded for i in eachindex(integrator.du)
        integrator.k1[i] = integrator.du[i] * integrator.dt
    end

    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] + c[2] * integrator.k1[i]
    end

    # TODO: Move away from here, not really belonging to stage 1!
    integrator.t_stage = integrator.t + c[2] * integrator.dt
end

function last_three_stages!(integrator::PERK4_ER_Integrator, alg, p)
  mesh, equations, dg, cache = mesh_equations_solver_cache(p)

  # S - 2
  @threaded for u_ind in eachindex(integrator.u)
    integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                              alg.AMatrix[1, 1] *
                              integrator.k1[u_ind] +
                              alg.AMatrix[1, 2] *
                              integrator.k_higher[u_ind]
  end
  integrator.t_stage = integrator.t + alg.c[alg.NumStages - 2] * integrator.dt

  #integrator.f(integrator.du, integrator.u_tmp, p, integrator.t_stage, integrator.du_ode_hyp)
  integrator.f(integrator.du, integrator.u_tmp, p, integrator.t_stage)

  @threaded for u_ind in eachindex(integrator.du)
      integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
  end

  # S - 1
  @threaded for u_ind in eachindex(integrator.u)
    integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                              alg.AMatrix[2, 1] *
                              integrator.k1[u_ind] +
                              alg.AMatrix[2, 2] *
                              integrator.k_higher[u_ind]
  end
  integrator.t_stage = integrator.t + alg.c[alg.NumStages - 1] * integrator.dt

  #integrator.f(integrator.du, integrator.u_tmp, p, integrator.t_stage, integrator.du_ode_hyp)
  integrator.f(integrator.du, integrator.u_tmp, p, integrator.t_stage)

  @threaded for u_ind in eachindex(integrator.du)
      integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
  end

  k_higher_wrap = wrap_array(integrator.k_higher, p)
  u_tmp_wrap = wrap_array(integrator.u_tmp, p)
  # 0.5 = b_{S-1}
  dS = 0.5 * int_w_dot_stage(k_higher_wrap, u_tmp_wrap, mesh, equations, dg, cache)

  # S
  @threaded for i in eachindex(integrator.du)
      integrator.u_tmp[i] = integrator.u[i] +
                            alg.AMatrix[3, 1] *
                            integrator.k1[i] +
                            alg.AMatrix[3, 2] *
                            integrator.k_higher[i]
  end

  #integrator.f(integrator.du, integrator.u_tmp, p, integrator.t + alg.c[alg.NumStages] * integrator.dt, integrator.du_ode_hyp)
  integrator.f(integrator.du, integrator.u_tmp, p, integrator.t + alg.c[alg.NumStages] * integrator.dt)
  
  #k_higher_wrap = wrap_array(integrator.k_higher, p)
  #u_tmp_wrap = wrap_array(integrator.u_tmp, p)
  # 0.5 = b_{S}
  dS += 0.5 * int_w_dot_stage(k_higher_wrap, u_tmp_wrap, mesh, equations, dg, cache)

  u_wrap = wrap_array(integrator.u, integrator.p)
  dir_wrap = wrap_array(integrator.direction, p)
  # Re-use `du` as helper data structure (not needed anymore)
  u_gamma_dir_wrap = wrap_array(integrator.du, integrator.p)
  
  S_old = integrate(entropy_math, u_wrap, mesh, equations, dg, cache)

  gamma = 1.0 # Default value if entropy relaxation methodology not applicable

  # TODO: If we do not want to sacrifice order, we would need to restrict this lower bound to 1 - O(dt)
  gamma_min = 1e-3 # Cannot be 0, as then r(0) = 0
  gamma_max = 1.0
  bisection_its_max = 100

  @threaded for element in eachelement(dg, cache)
      @views @. u_gamma_dir_wrap[.., element] = u_wrap[.., element] +
                                                gamma_max *
                                                dir_wrap[.., element]
  end
  r_max = entropy_diff(gamma_max, S_old, dS, u_gamma_dir_wrap, mesh,
                         equations, dg, cache)

  @threaded for element in eachelement(dg, cache)
      @views @. u_gamma_dir_wrap[.., element] = u_wrap[.., element] +
                                                gamma_min *
                                                dir_wrap[.., element]
  end
  r_min = entropy_diff(gamma_min, S_old, dS, u_gamma_dir_wrap,
                         mesh, equations, dg, cache)

  # Check if there exists a root for `r` in the interval [gamma_min, gamma_max]
  if r_max > 0 && r_min < 0 # && 
    # integrator.finalstep == false # Avoid last-step shenanigans for now

    integrator.num_timestep_relaxations += 1
    # Init with gamma_0
    gamma_eps = 1e-13

    bisect_its = 0
    @trixi_timeit timer() "ER: Bisection" while gamma_max - gamma_min >
                                                gamma_eps &&
                                                bisect_its <
                                                bisection_its_max
        gamma = 0.5 * (gamma_max + gamma_min)

        @threaded for element in eachelement(dg, cache)
            @views @. u_gamma_dir_wrap[.., element] = u_wrap[.., element] +
                                                      gamma *
                                                      dir_wrap[.., element]
        end
        r_gamma = entropy_diff(gamma, S_old, dS, u_gamma_dir_wrap,
                                  mesh, equations, dg, cache)

        if r_gamma < 0
            gamma_min = gamma
        else
            gamma_max = gamma
        end
        bisect_its += 1
      end
  end

  t_end = last(integrator.sol.prob.tspan)
  integrator.iter += 1
  # Last timestep shenanigans
  if integrator.t + gamma * integrator.dt > t_end ||
    isapprox(integrator.t + gamma * integrator.dt, t_end)
      integrator.t = t_end
      gamma = (t_end - integrator.t) / integrator.dt
      terminate!(integrator)
      println("# Relaxed timesteps: ", integrator.num_timestep_relaxations)
  else
      integrator.t += gamma * integrator.dt
  end

  @threaded for i in eachindex(integrator.u)
    integrator.u[i] += gamma * integrator.direction[i]
  end
end

function step!(integrator::PERK4_ER_Integrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
        k1!(integrator, prob.p, alg.c)

        #integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, integrator.du_ode_hyp)
        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)

        @threaded for u_ind in eachindex(integrator.du)
            integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
        end

        for stage in 3:(alg.NumStages - 3)
            # Construct current state
            @threaded for u_ind in eachindex(integrator.u)
                integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                          alg.AMatrices[stage - 2, 1] *
                                          integrator.k1[u_ind] +
                                          alg.AMatrices[stage - 2, 2] *
                                          integrator.k_higher[u_ind]
            end

            integrator.t_stage = integrator.t + alg.c[stage] * integrator.dt

            #integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, integrator.du_ode_hyp)
            integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)

            @threaded for i in eachindex(integrator.du)
                integrator.k_higher[i] = integrator.du[i] * integrator.dt
            end
        end

        last_three_stages!(integrator, alg, prob.p)
    end # PERK4 step

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

# get a cache where the RHS can be stored
get_du(integrator::PERK4_ER_Integrator) = integrator.du
get_tmp_cache(integrator::PERK4_ER_Integrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::PERK4_ER_Integrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::PERK4_ER_Integrator, dt)
    integrator.dt = dt
end

function get_proposed_dt(integrator::PERK4_ER_Integrator)
    return integrator.dt
end

# stop the time integration
function terminate!(integrator::PERK4_ER_Integrator)
    integrator.finalstep = true
    empty!(integrator.opts.tstops)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PERK4_ER_Integrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)

    # TODO: Move this into parabolic cache or similar
    resize!(integrator.du_ode_hyp, new_size)

    # TODO: Only for averaging callback (required for coupled Euler-acoustic simulations)
    #resize!(integrator.uprev, new_size)
end
end # @muladd
