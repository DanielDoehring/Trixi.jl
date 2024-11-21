# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function last_three_stages!(integrator::AbstractPairedExplicitERRKMultiParabolicIntegrator,
                                    alg, p)
    mesh, equations, dg, cache = mesh_equations_solver_cache(p)

    # S - 2
    @threaded for u_ind in eachindex(integrator.u)
        integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                  alg.a_matrix_constant[1, 1] *
                                  integrator.k1[u_ind] +
                                  alg.a_matrix_constant[1, 2] *
                                  integrator.k_higher[u_ind]
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[alg.num_stages - 2] * integrator.dt,
                 integrator.du_tmp)

    @threaded for u_ind in eachindex(integrator.du)
        integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
    end

    # S - 1
    @threaded for u_ind in eachindex(integrator.u)
        integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                  alg.a_matrix_constant[2, 1] *
                                  integrator.k1[u_ind] +
                                  alg.a_matrix_constant[2, 2] *
                                  integrator.k_higher[u_ind]
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[alg.num_stages - 1] * integrator.dt,
                 integrator.du_tmp)

    @threaded for u_ind in eachindex(integrator.du)
        integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
    end

    k_higher_wrap = wrap_array(integrator.k_higher, p)
    u_tmp_wrap = wrap_array(integrator.u_tmp, p)
    # 0.5 = b_{S-1}
    # TODO: Combine integration of i-1, i!
    dS = int_w_dot_stage(k_higher_wrap, u_tmp_wrap, mesh, equations, dg, cache) / 2

    # S
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

    @threaded for i in eachindex(integrator.du)
        integrator.direction[i] = (integrator.k_higher[i] +
                                   integrator.du[i] * integrator.dt) / 2
    end

    du_wrap = wrap_array(integrator.du, integrator.p)
    # 0.5 = b_{S}
    dS += integrator.dt *
          int_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache) / 2

    u_wrap = wrap_array(integrator.u, integrator.p)
    S_old = integrate(entropy_math, u_wrap, mesh, equations, dg, cache)

    dir_wrap = wrap_array(integrator.direction, p)

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

    t_end = last(integrator.sol.prob.tspan)
    integrator.iter += 1
    # Last timestep shenanigans
    if integrator.t + integrator.gamma * integrator.dt > t_end ||
       isapprox(integrator.t + integrator.gamma * integrator.dt, t_end)
        integrator.t = t_end
        integrator.gamma = (t_end - integrator.t) / integrator.dt
        terminate!(integrator)
    else
        integrator.t += integrator.gamma * integrator.dt
    end

    # Do relaxed update
    @threaded for i in eachindex(integrator.u)
        integrator.u[i] += integrator.gamma * integrator.direction[i]
    end
end

function step!(integrator::PairedExplicitERRK4MultiParabolicIntegrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    callbacks = integrator.opts.callback

    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
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
    end # PairedExplicitERRK4Multi step

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

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::AbstractPairedExplicitERRKMultiParabolicIntegrator,
                      new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)
    # Addition for entropy relaxation PERK methods
    resize!(integrator.direction, new_size)
    # Addition for multirate PERK methods for parabolic problems
    resize!(integrator.du_tmp, new_size)
end
end # @muladd