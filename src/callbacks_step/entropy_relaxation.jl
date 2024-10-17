# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    EntropyRelaxationCallback(analysis_interval=0, alive_interval=analysis_interval÷10)

Inexpensive callback showing that a simulation is still running by printing
some information such as the current time to the screen every `alive_interval`
time steps. If `analysis_interval ≂̸ 0`, the output is omitted every
`analysis_interval` time steps.
"""
mutable struct EntropyRelaxationCallback
    gamma_min::Float64
    gamma_max::Float64
end

function EntropyRelaxationCallback(; gamma_min = 0.1, gamma_max = 1.0)
    entropy_relaxation_callback = EntropyRelaxationCallback(gamma_min, gamma_max)

    DiscreteCallback(entropy_relaxation_callback, entropy_relaxation_callback, # the first one is the condition, the second the affect!
                     save_positions = (false, false),
                     initialize = initialize!)
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:EntropyRelaxationCallback})
    @nospecialize cb # reduce precompilation time

    entropy_relaxation_callback = cb.affect!
    print(io, "EntropyRelaxationCallback(gamma_min=", entropy_relaxation_callback.gamma_min, ")")
    print(io, "EntropyRelaxationCallback(gamma_max=", entropy_relaxation_callback.gamma_max, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:EntropyRelaxationCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        entropy_relaxation_callback = cb.affect!

        setup = [
            "gamma_min" => entropy_relaxation_callback.gamma_min
            "gamma_max" => entropy_relaxation_callback.gamma_max
        ]
        summary_box(io, "EntropyRelaxationCallback", setup)
    end
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: EntropyRelaxationCallback}
    return nothing
end

# this method is called to determine whether the callback should be activated
function (entropy_relaxation_callback::EntropyRelaxationCallback)(u, t, integrator)
  return true
end

function entropy_der(stage, u_i,
                     mesh::Union{TreeMesh{1}, StructuredMesh{1}}, equations, dg::DG, cache)
    # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
    integrate_via_indices(u_i, mesh, equations, dg, cache,
                          stage) do u_i, i, element, equations, dg, stage
        w_node = cons2entropy(get_node_vars(u_i, equations, dg, i, element), equations)
        stage_node = get_node_vars(stage, equations, dg, i, element)
        dot(w_node, stage_node)
    end
end

function r(gamma, S_old, dS, u, dir, mesh, equations, dg, cache)
    return integrate(entropy_math, u + gamma * dir, mesh, equations, dg, cache) - S_old - gamma * dS
end

# TODO: Try putting this into PERK directly, something is wrong with the timestep setting.
# this method is called when the callback is activated
function (entropy_relaxation_callback::EntropyRelaxationCallback)(integrator)
    @unpack u, u_tmp, k1, k_higher, alg, p = integrator
    @unpack prob = integrator.sol
    t_end = last(prob.tspan)

    mesh, equations, dg, cache = mesh_equations_solver_cache(p)

    u_wrapped = wrap_array(u, p)
    u_tmp_wrapped = wrap_array(u_tmp, p)
    k1_wrapped = wrap_array(k1, p)
    k_higher_wrapped = wrap_array(k_higher, p)

    # Check if there exists a root for `r` in the interval [0, 1]
    S_old = integrate(entropy_math, u_wrapped, mesh, equations, dg, cache)
    dS = (alg.b1 * entropy_der(k1_wrapped, u_wrapped, mesh, equations, dg, cache) + 
          # u_tmp corresponds to input leading to last k_higher
          alg.bS * entropy_der(k_higher_wrapped, u_tmp_wrapped, mesh, equations, dg, cache) )

    # Direction already scaled with dt
    dir = alg.b1 * k1 + alg.bS * k_higher

    dir_wrapped = wrap_array(dir, p)

    gamma_min = copy(entropy_relaxation_callback.gamma_min)
    gamma_max = copy(entropy_relaxation_callback.gamma_max)

    if r(gamma_max, S_old, dS, u_wrapped, dir_wrapped, mesh, equations, dg, cache) > 0 && 
       r(gamma_min, S_old, dS, u_wrapped, dir_wrapped, mesh, equations, dg, cache) < 0

      # Init with gamma_0
      gamma_eps = 1e-9
      gamma = 0.5 * (gamma_max + gamma_min)

      while gamma_max - gamma_min > gamma_eps
        r_gamma = r(gamma, S_old, dS, u_wrapped, dir_wrapped, mesh, equations, dg, cache)

        if r_gamma < 0
          gamma_min = gamma
        else
          gamma_max = gamma
        end
        gamma = 0.5 * (gamma_max + gamma_min)
      end

      println("Found gamma: ", gamma)
    else # Entropy relaxation methodology not applicable, do standard RK step
      gamma = 1
    end

    if integrator.t + gamma * integrator.dt > t_end || isapprox(integrator.t + gamma * integrator.dt, t_end)
      integrator.t = t_end
      gamma = (t_end - integrator.t) / integrator.dt
      integrator.finalstep = true
    else
      integrator.t += gamma * integrator.dt
    end

    @threaded for i in eachindex(u)
      u[i] += gamma * (alg.b1 * k1[i] + alg.bS * k_higher[i])
    end

    #=
    if integrator.t < t_end
      integrator.finalstep = false
    end
    =#

    println("t: ", integrator.t)

    # TODO: Not sure if we need this for FSAL!
    #u_modified!(integrator, true)
    return nothing
end
end # @muladd
