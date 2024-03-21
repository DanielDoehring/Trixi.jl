# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

function ComputePERK4_ButcherTableau(NumStages::Int, BasePathMonCoeffs::AbstractString)
                                     
  # Use linear increasing timesteps for free timesteps
  
  c = zeros(NumStages)
  for k in 2:NumStages-4
    #c[k] = (k - 1)/(NumStages - 4) # Equidistant timestep distribution (similar to PERK2)
    c[k] = ((k - 1)/(NumStages - 4))^2 # Quadratically increasing
  end
  
  
  # Current approach: Use ones for simplicity
  c_const = 1.0
  #=
  c = c_const * ones(NumStages)
  c[1] = 0.0
  =#
  
  cS3 = c_const
  c[NumStages - 3] = cS3
  c[NumStages - 2] = 0.479274057836310
  c[NumStages - 1] =  sqrt(3)/6 + 0.5
  c[NumStages]     = -sqrt(3)/6 + 0.5
  
  println("Timestep-split: "); display(c); println("\n")

  # For the p = 4 method there are less free coefficients
  CoeffsMax = NumStages - 5

  AMatrices = zeros(CoeffsMax, 2)
  AMatrices[:, 1] = c[3:NumStages-3]

  PathMonCoeffs = BasePathMonCoeffs * "a_" * string(NumStages) * "_" * string(NumStages) * ".txt"
  #PathMonCoeffs = BasePathMonCoeffs * "a_" * string(NumStages) * ".txt"
  NumMonCoeffs, A = read_file(PathMonCoeffs, Float64)
  @assert NumMonCoeffs == CoeffsMax

  if NumMonCoeffs > 0
    AMatrices[CoeffsMax - NumMonCoeffs + 1:end, 1] -= A
    AMatrices[CoeffsMax - NumMonCoeffs + 1:end, 2]  = A
  end

  # Shared matrix
  AMatrix = [0.479274057836310-0.114851811257441/cS3 0.114851811257441/cS3
             0.1397682537005989                      0.648906880894214
             0.1830127018922191                      0.028312163512968]

  println("Variable portion of A-Matrix:")
  display(AMatrices); println()

  return AMatrices, AMatrix, c
end

mutable struct PERK4{StageCallbacks}
  const NumStages::Int64
  stage_callbacks::StageCallbacks

  AMatrices::Matrix{Float64}
  AMatrix::Matrix{Float64}
  c::Vector{Float64}

  function PERK4(NumStages_::Int, BasePathMonCoeffs_::AbstractString,
                stage_callbacks=())

    newPERK4 = new{typeof(stage_callbacks)}(NumStages_, stage_callbacks)

    newPERK4.AMatrices, newPERK4.AMatrix, newPERK4.c = 
      ComputePERK4_ButcherTableau(NumStages_, BasePathMonCoeffs_)

    return newPERK4
  end
end # struct PERK4


# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PERK4_Integrator{RealT<:Real, uType, Params, Sol, F, Alg, PERK_IntegratorOptions}
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
  # PERK4 stages:
  k1::uType
  k_higher::uType
  k_S1::uType # Required for third & fourth order
  t_stage::RealT
  du_ode_hyp::uType # TODO: Not best solution since this is not needed for hyperbolic problems
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::PERK4_Integrator, field::Symbol)
  if field === :stats
    return (naccept = getfield(integrator, :iter),)
  end
  # general fallback
  return getfield(integrator, field)
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::PERK4;
               dt, callback=nothing, kwargs...)

  u0    = copy(ode.u0)
  du    = zero(u0) # previously: similar(u0)
  u_tmp = zero(u0)

  # PERK4 stages
  k1       = zero(u0)
  k_higher = zero(u0)
  k_S1     = zero(u0)

  du_ode_hyp = similar(u0) # TODO: Not best solution since this is not needed for hyperbolic problems

  t0 = first(ode.tspan)
  iter = 0

  integrator = PERK4_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                                (prob=ode,), ode.f, alg,
                                PERK_IntegratorOptions(callback, ode.tspan; kwargs...), false,
                                k1, k_higher, k_S1, t0, du_ode_hyp)
            
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
  #solve!(integrator)
  solve_steps!(integrator)
end

function step!(integrator::PERK4_Integrator)
  @unpack prob = integrator.sol
  @unpack alg = integrator
  t_end = last(prob.tspan)
  callbacks = integrator.opts.callback

  @assert !integrator.finalstep
  if isnan(integrator.dt)
    error("time step size `dt` is NaN")
  end

  # if the next iteration would push the simulation beyond the end time, set dt accordingly
  if integrator.t + integrator.dt > t_end || isapprox(integrator.t + integrator.dt, t_end)
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

    # k2
    integrator.t_stage = integrator.t + alg.c[2] * integrator.dt

    # Construct current state
    @threaded for i in eachindex(integrator.du)
      integrator.u_tmp[i] = integrator.u[i] + alg.c[2] * integrator.k1[i]
    end

    #integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, integrator.du_ode_hyp)
    integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)

    @threaded for u_ind in eachindex(integrator.u)
      integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
    end

    for stage = 3:alg.NumStages - 3

      # Construct current state
      @threaded for u_ind in eachindex(integrator.u)
        integrator.u_tmp[u_ind] = integrator.u[u_ind] + alg.AMatrices[stage - 2, 1] * integrator.k1[u_ind] + 
                                                        alg.AMatrices[stage - 2, 2] * integrator.k_higher[u_ind]
      end

      integrator.t_stage = integrator.t + alg.c[stage] * integrator.dt
      
      #integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, integrator.du_ode_hyp)
      integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)
      
      @threaded for i in eachindex(integrator.du)
        integrator.k_higher[i] = integrator.du[i] * integrator.dt
      end
    end

    # Last three stages: Same Butcher Matrix
    for stage = 1:3
      @threaded for u_ind in eachindex(integrator.u)
        integrator.u_tmp[u_ind] = integrator.u[u_ind] + alg.AMatrix[stage, 1] * integrator.k1[u_ind] + 
                                                        alg.AMatrix[stage, 2] * integrator.k_higher[u_ind]
      end
      integrator.t_stage = integrator.t + alg.c[alg.NumStages - 3 + stage] * integrator.dt

      #integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, integrator.du_ode_hyp)
      integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)

      @threaded for u_ind in eachindex(integrator.u)
        integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
      end

      if stage == 2
        @threaded for u_ind in eachindex(integrator.u)
          integrator.k_S1[u_ind] = integrator.k_higher[u_ind]
        end
      end
    end

    @threaded for u_ind in eachindex(integrator.u)
      integrator.u[u_ind] += 0.5 * (integrator.k_S1[u_ind] + integrator.k_higher[u_ind])
    end
  end # PERK4 step

  integrator.iter += 1
  integrator.t += integrator.dt

  # TODO: Do other approach (see never approach in other methods)
  # handle callbacks
  if callbacks isa CallbackSet
    for cb in callbacks.discrete_callbacks
      if cb.condition(integrator.u, integrator.t, integrator)
        cb.affect!(integrator)
      end
    end
  end

  # respect maximum number of iterations
  if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
    @warn "Interrupted. Larger maxiters is needed."
    terminate!(integrator)
  end
end

function solve_steps!(integrator::PERK4_Integrator)
  @unpack prob = integrator.sol

  integrator.finalstep = false

  @trixi_timeit timer() "main loop" while !integrator.finalstep
    step!(integrator)
  end # "main loop" timer
  
  return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                (prob.u0, integrator.u),
                                integrator.sol.prob)
end

function solve!(integrator::PERK4_Integrator)
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
    if integrator.t + integrator.dt > t_end || isapprox(integrator.t + integrator.dt, t_end)
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

      # k2
      integrator.t_stage = integrator.t + alg.c[2] * integrator.dt
    
      # Construct current state
      @threaded for i in eachindex(integrator.du)
        integrator.u_tmp[i] = integrator.u[i] + alg.c[2] * integrator.k1[i]
      end

      #integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, integrator.du_ode_hyp)
      integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)

      @threaded for u_ind in eachindex(integrator.u)
        integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
      end

      for stage = 3:alg.NumStages - 3

        # Construct current state
        @threaded for u_ind in eachindex(integrator.u)
          integrator.u_tmp[u_ind] = integrator.u[u_ind] + alg.AMatrices[stage - 2, 1] * integrator.k1[u_ind] + 
                                                          alg.AMatrices[stage - 2, 2] * integrator.k_higher[u_ind]
        end

        integrator.t_stage = integrator.t + alg.c[stage] * integrator.dt
        
        #integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, integrator.du_ode_hyp)
        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)
        
        @threaded for i in eachindex(integrator.du)
          integrator.k_higher[i] = integrator.du[i] * integrator.dt
        end
      end

      # Last three stages: Same Butcher Matrix
      for stage = 1:3
        @threaded for u_ind in eachindex(integrator.u)
          integrator.u_tmp[u_ind] = integrator.u[u_ind] + alg.AMatrix[stage, 1] * integrator.k1[u_ind] + 
                                                          alg.AMatrix[stage, 2] * integrator.k_higher[u_ind]
        end
        integrator.t_stage = integrator.t + alg.c[alg.NumStages - 3 + stage] * integrator.dt

        #integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, integrator.du_ode_hyp)
        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)

        @threaded for u_ind in eachindex(integrator.u)
          integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
        end

        if stage == 2
          @threaded for u_ind in eachindex(integrator.u)
            integrator.k_S1[u_ind] = integrator.k_higher[u_ind]
          end
        end
      end

      @threaded for u_ind in eachindex(integrator.u)
        integrator.u[u_ind] += 0.5 * (integrator.k_S1[u_ind] + integrator.k_higher[u_ind])
      end
      
      #=
      for stage_callback in alg.stage_callbacks
        stage_callback(integrator.u, integrator, prob.p, integrator.t_stage)
      end
      =#
    end # PERK4 step

    integrator.iter += 1
    integrator.t += integrator.dt

    # TODO: Do other approach (see never approach in other methods)
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
  
  return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                (prob.u0, integrator.u),
                                integrator.sol.prob)
end

# get a cache where the RHS can be stored
get_du(integrator::PERK4_Integrator) = integrator.du
get_tmp_cache(integrator::PERK4_Integrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::PERK4_Integrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::PERK4_Integrator, dt)
  integrator.dt = dt
end

function get_proposed_dt(integrator::PERK4_Integrator)
  return integrator.dt
end

# stop the time integration
function terminate!(integrator::PERK4_Integrator)
  integrator.finalstep = true
  empty!(integrator.opts.tstops)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PERK4_Integrator, new_size)
  resize!(integrator.u, new_size)
  resize!(integrator.du, new_size)
  resize!(integrator.u_tmp, new_size)

  resize!(integrator.k1, new_size)
  resize!(integrator.k_higher, new_size)
  resize!(integrator.k_S1, new_size)

  # TODO: Move this into parabolic cache or similar
  resize!(integrator.du_ode_hyp, new_size)
end

end # @muladd