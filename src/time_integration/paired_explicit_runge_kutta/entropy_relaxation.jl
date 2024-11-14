# NOTE: This could actually live in a more general location,
# as it is not PERK-specific.

@inline function int_w_dot_stage(stage, u_i,
                                 mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                 equations, dg::DG, cache)
    @trixi_timeit timer() "Integrate w ⋅ k" begin
        # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
        integrate_via_indices(u_i, mesh, equations, dg, cache,
                              stage) do u_i, i, element, equations, dg, stage
            w_node = cons2entropy(get_node_vars(u_i, equations, dg, i, element),
                                  equations)
            stage_node = get_node_vars(stage, equations, dg, i, element)
            dot(w_node, stage_node)
        end
    end
end

@inline function int_w_dot_stage(stage, u_i,
                                 mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                             UnstructuredMesh2D, P4estMesh{2},
                                             T8codeMesh{2}},
                                 equations, dg::DG, cache)
    @trixi_timeit timer() "Integrate w ⋅ k" begin
        # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
        integrate_via_indices(u_i, mesh, equations, dg, cache,
                              stage) do u_i, i, j, element, equations, dg, stage
            w_node = cons2entropy(get_node_vars(u_i, equations, dg, i, j, element),
                                  equations)
            stage_node = get_node_vars(stage, equations, dg, i, j, element)
            dot(w_node, stage_node)
        end
    end
end

@inline function int_w_dot_stage(stage, u_i,
                                 mesh::Union{TreeMesh{3}, StructuredMesh{3}, P4estMesh{3},
                                             T8codeMesh{3}},
                                 equations, dg::DG, cache)
    @trixi_timeit timer() "Integrate w ⋅ k" begin
        # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
        integrate_via_indices(u_i, mesh, equations, dg, cache,
                              stage) do u_i, i, j, k, element, equations, dg, stage
            w_node = cons2entropy(get_node_vars(u_i, equations, dg, i, j, k, element),
                                  equations)
            stage_node = get_node_vars(stage, equations, dg, i, j, k, element)
            dot(w_node, stage_node)
        end
    end
end

@inline function entropy_difference(gamma, S_old, dS, u_gamma_dir, mesh,
                                    equations, dg, cache)
    return integrate(entropy_math, u_gamma_dir, mesh, equations, dg, cache) -
           S_old - gamma * dS
end

function r_gamma(gamma, S_old, dS, u_tmp_wrap, u_wrap, dir_wrap, mesh, equations, dg, cache)
    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                            gamma * dir_wrap[.., element]
    end
    return entropy_difference(gamma, S_old, dS, u_tmp_wrap, mesh, equations, dg, cache)
end

# For NonlinearSolve.jl
function r_gamma(gamma, params_nonlinear)
    @unpack S_old, dS, u_tmp_wrap, u_wrap, dir_wrap, mesh, equations, dg, cache = params_nonlinear

    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                            gamma * dir_wrap[.., element]
    end
    return entropy_difference(gamma, S_old, dS, u_tmp_wrap, mesh, equations, dg, cache)
end

function dr(gamma, dS, u_tmp_wrap, u_wrap, dir_wrap, mesh, equations, dg, cache)
    # NOTE: Not sure if this is valid, i.e., do not recomputing `u_tmp_wrap` here
    #=
    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] + gamma * dir_wrap[.., element]
    end
    =#
    return int_w_dot_stage(dir_wrap, u_tmp_wrap, mesh, equations, dg, cache) - dS
end

# For NonlinearSolve.jl
function dr(gamma, params_nonlinear)
    #=
    @unpack dS, u_tmp_wrap, u_wrap, dir_wrap, mesh, equations, dg, cache = params_nonlinear

    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] + gamma * dir_wrap[.., element]
    end
    =#
    return int_w_dot_stage(dir_wrap, u_tmp_wrap, mesh, equations, dg, cache) - dS
end