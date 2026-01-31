# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    IndicatorHGCallback(interval=1)

"""
struct IndicatorHGCallback
    interval::Int
end

function IndicatorHGCallback(; interval = 1)
    indicator_callback = IndicatorHGCallback(interval)

    return DiscreteCallback(indicator_callback, indicator_callback, # the first one is the condition, the second the affect!
                            save_positions = (false, false),
                            initialize = initialize!)
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:IndicatorHGCallback})
    @nospecialize cb # reduce precompilation time

    indicator_callback = cb.affect!
    print(io, "IndicatorHGCallback(interval=", indicator_callback.interval, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:IndicatorHGCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        indicator_callback = cb.affect!

        setup = [
            "interval" => indicator_callback.interval
        ]
        summary_box(io, "IndicatorHGCallback", setup)
    end
end

function calc_alpha(integrator)
    semi = integrator.p
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

    indicator_hg = dg.volume_integral.volume_integral_stabilized.indicator
    u = wrap_array(integrator.u, semi)
    indicator_hg(u, mesh, equations, dg, cache)

    return nothing
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: IndicatorHGCallback}
    calc_alpha(integrator)

    return nothing
end

# this method is called to determine whether the callback should be activated
function (indicator_callback::IndicatorHGCallback)(u, t, integrator)
    @unpack interval = indicator_callback

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return interval > 0 && (integrator.stats.naccept % interval == 0 || isfinished(integrator))
end

# this method is called when the callback is activated
function (indicator_callback::IndicatorHGCallback)(integrator)
    calc_alpha(integrator)

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)
    return nothing
end
end # @muladd
