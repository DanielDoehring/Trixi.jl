# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function last_three_stages!(integrator::AbstractPairedExplicitRKMultiParabolicIntegrator,
                                    p, alg)
    for stage in 1:2
        @threaded for u_ind in eachindex(integrator.u)
            integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                      integrator.dt *
                                      (alg.a_matrix_constant[stage, 1] *
                                       integrator.k1[u_ind] +
                                       alg.a_matrix_constant[stage, 2] *
                                       integrator.du[u_ind])
        end

        integrator.f(integrator.du, integrator.u_tmp, p,
                     integrator.t +
                     alg.c[alg.num_stages - 3 + stage] * integrator.dt,
                     integrator.du_tmp)
    end

    # Last stage
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              integrator.dt *
                              (alg.a_matrix_constant[3, 1] * integrator.k1[i] +
                               alg.a_matrix_constant[3, 2] * integrator.du[i])
    end

    # Safe K_{S-1} in `k1`:
    @threaded for i in eachindex(integrator.u)
        integrator.k1[i] = integrator.du[i]
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[alg.num_stages] * integrator.dt,
                 integrator.du_tmp)

    @threaded for u_ind in eachindex(integrator.u)
        # Note that 'k1' carries the values of K_{S-1}
        # and that we construct 'K_S' "in-place" from 'integrator.du'
        integrator.u[u_ind] += 0.5 * integrator.dt *
                               (integrator.k1[u_ind] + integrator.du[u_ind])
    end
end

function step!(integrator::PairedExplicitRK4MultiParabolicIntegrator)
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
