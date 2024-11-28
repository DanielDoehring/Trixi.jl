# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function last_three_stages!(integrator::PairedExplicitRK4MultiParabolicIntegrator,
                                    p, alg)
    for stage in 1:2
        @threaded for u_ind in eachindex(integrator.u)
            integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                      integrator.dt *
                                      (alg.a_matrix_constant[1, stage] *
                                       integrator.k1[u_ind] +
                                       alg.a_matrix_constant[2, stage] *
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
                              (alg.a_matrix_constant[1, 3] * integrator.k1[i] +
                               alg.a_matrix_constant[2, 3] * integrator.du[i])
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
end # @muladd
