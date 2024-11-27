# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function last_three_stages!(integrator::AbstractPairedExplicitRelaxationRKMultiParabolicIntegrator,
                                    alg, p)
    mesh, equations, dg, cache = mesh_equations_solver_cache(p)

    for stage in 1:2
        @threaded for u_ind in eachindex(integrator.u)
            integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                      alg.a_matrix_constant[stage, 1] *
                                      integrator.k1[u_ind] +
                                      alg.a_matrix_constant[stage, 2] *
                                      integrator.k_higher[u_ind]
        end

        integrator.f(integrator.du, integrator.u_tmp, p,
                     integrator.t +
                     alg.c[alg.num_stages - 3 + stage] * integrator.dt,
                     integrator.du_tmp)

        @threaded for u_ind in eachindex(integrator.du)
            integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
        end
    end

    k_higher_wrap = wrap_array(integrator.k_higher, p)
    u_tmp_wrap = wrap_array(integrator.u_tmp, p)
    # 0.5 = b_{S-1}
    # TODO: Combine integration of i-1, i!
    dS = 0.5 * int_w_dot_stage(k_higher_wrap, u_tmp_wrap, mesh, equations, dg, cache)

    # Last stage
    @threaded for i in eachindex(integrator.du)
        integrator.u_tmp[i] = integrator.u[i] +
                              alg.a_matrix_constant[3, 1] *
                              integrator.k1[i] +
                              alg.a_matrix_constant[3, 2] *
                              integrator.k_higher[i]
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[alg.num_stages] * integrator.dt,
                 integrator.du_tmp)

    # Note: We re-use `k_higher` for the "direction"
    @threaded for i in eachindex(integrator.du)
        integrator.k_higher[i] = 0.5 * (integrator.k_higher[i] +
                                   integrator.du[i] * integrator.dt)
        
    end

    du_wrap = wrap_array(integrator.du, integrator.p)
    # 0.5 = b_{S}
    dS += 0.5 * integrator.dt *
          int_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

    u_wrap = wrap_array(integrator.u, integrator.p)
    S_old = integrate(entropy_math, u_wrap, mesh, equations, dg, cache)

    # Note: We re-use `k_higher` for the "direction"
    dir_wrap = wrap_array(integrator.k_higher, p)

    #=
    # Bisection for gamma
    @trixi_timeit timer() "ER: Bisection" begin
        gamma_bisection!(integrator, u_tmp_wrap, u_wrap, dir_wrap, S_old, dS,
                         mesh, equations, dg, cache)
    end
    =#

    # Newton search for gamma
    @trixi_timeit timer() "ER: Newton" begin
        gamma_newton!(integrator, u_tmp_wrap, u_wrap, dir_wrap, S_old, dS,
                      mesh, equations, dg, cache)
    end

    integrator.iter += 1
    # Check if due to entropy relaxation the final step is not reached
    if integrator.finalstep == true && integrator.gamma != 1
        # If we would go beyond the final time, clip gamma at 1.0
        if integrator.gamma > 1.0
            integrator.gamma = 1.0
        else # If we are below the final time, reset finalstep flag
            integrator.finalstep = false
        end
    end
    integrator.t += integrator.gamma * integrator.dt

    # Do relaxed update
    @threaded for i in eachindex(integrator.u)
        # Note: We re-use `k_higher` for the "direction"
        integrator.u[i] += integrator.gamma * integrator.k_higher[i]
    end
end

function step!(integrator::PairedExplicitRelaxationRK4MultiParabolicIntegrator)
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
        k1!(integrator, prob.p, alg.c)

        # k2: Only evaluated at finest level
        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[2] * integrator.dt,
                     integrator.du_tmp,
                     integrator.level_info_elements_acc[1],
                     integrator.level_info_interfaces_acc[1],
                     integrator.level_info_boundaries_acc[1],
                     #integrator.level_info_boundaries_orientation_acc[1],
                     integrator.level_info_mortars_acc[1],
                     integrator.level_u_indices_elements, 1)

        @threaded for u_ind in integrator.level_u_indices_elements[1]
            integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
        end

        for stage in 3:(alg.num_stages - 3)

            ### General implementation: Not own method for each grid level ###
            # Loop over different methods with own associated level
            for level in 1:min(alg.num_methods, integrator.n_levels)
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                              alg.a_matrices[stage - 2, 1, level] *
                                              integrator.k1[u_ind]
                end
            end
            for level in 1:min(alg.max_eval_levels[stage], integrator.n_levels)
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] += alg.a_matrices[stage - 2, 2, level] *
                                               integrator.k_higher[u_ind]
                end
            end

            # "Remainder": Non-efficiently integrated
            for level in (alg.num_methods + 1):(integrator.n_levels)
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                              alg.a_matrices[stage - 2, 1,
                                                             alg.num_methods] *
                                              integrator.k1[u_ind]
                end
            end
            if alg.max_eval_levels[stage] == alg.num_methods
                for level in (alg.max_eval_levels[stage] + 1):(integrator.n_levels)
                    @threaded for u_ind in integrator.level_u_indices_elements[level]
                        integrator.u_tmp[u_ind] += alg.a_matrices[stage - 2, 2,
                                                                  alg.num_methods] *
                                                   integrator.k_higher[u_ind]
                    end
                end
            end

            ### Simplified implementation: Own method for each level ###
            #=
            for level in 1:integrator.n_levels
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] + alg.a_matrices[stage - 2, 1, level] *
                                               integrator.k1[u_ind]
                end
            end
            for level in 1:alg.max_eval_levels[stage]
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] += alg.a_matrices[stage - 2, 2, level] *
                                               integrator.k_higher[u_ind]
                end
            end
            =#

            #=
            ### Optimized implementation for case: Own method for each level with c[i] = 1.0, i = 2, S - 4
            for level in 1:alg.max_eval_levels[stage]
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] + alg.a_matrices[stage - 2, 1, level] *
                                               integrator.k1[u_ind] + 
                                               alg.a_matrices[stage - 2, 2, level] *
                                               integrator.k_higher[u_ind]
                end
            end
            for level in alg.max_eval_levels[stage]+1:integrator.n_levels
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] + integrator.k1[u_ind] # * A[stage, 1, level] = c[level] = 1
                end
            end
            =#

            # For statically non-uniform meshes/characteristic speeds
            #integrator.coarsest_lvl = alg.max_active_levels[stage]

            # "coarsest_lvl" cannot be static for AMR, has to be checked with available levels
            integrator.coarsest_lvl = min(alg.max_active_levels[stage],
                                          integrator.n_levels)

            # Check if there are fewer integrators than grid levels (non-optimal method)
            if integrator.coarsest_lvl == alg.num_methods
                # NOTE: This is supposedly more efficient than setting
                #integrator.coarsest_lvl = integrator.n_levels
                # and then using the level-dependent version

                integrator.f(integrator.du, integrator.u_tmp, prob.p,
                             integrator.t + alg.c[stage] * integrator.dt,
                             integrator.du_tmp)

                @threaded for u_ind in eachindex(integrator.du)
                    integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
                end
            else
                integrator.f(integrator.du, integrator.u_tmp, prob.p,
                             integrator.t + alg.c[stage] * integrator.dt,
                             integrator.du_tmp,
                             integrator.level_info_elements_acc[integrator.coarsest_lvl],
                             integrator.level_info_interfaces_acc[integrator.coarsest_lvl],
                             integrator.level_info_boundaries_acc[integrator.coarsest_lvl],
                             #integrator.level_info_boundaries_orientation_acc[integrator.coarsest_lvl],
                             integrator.level_info_mortars_acc[integrator.coarsest_lvl],
                             integrator.level_u_indices_elements,
                             integrator.coarsest_lvl)

                # Update k_higher of relevant levels
                for level in 1:(integrator.coarsest_lvl)
                    @threaded for u_ind in integrator.level_u_indices_elements[level]
                        integrator.k_higher[u_ind] = integrator.du[u_ind] *
                                                     integrator.dt
                    end
                end
            end
        end # end loop over different stages

        last_three_stages!(integrator, alg, prob.p)
    end # PairedExplicitRelaxationRK4Multi step

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
