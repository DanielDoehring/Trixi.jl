# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function Trixi.get_nodal_variable(variable, u, mesh, equations, dg, cache)
    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)
    # By definition, the variable must be provided at every node of every element!
    # Otherwise, the `SaveSolutionCallback` will crash.
    variable_array = zeros(eltype(cache.elements),
                            n_nodes, n_nodes, # equivalent: `ntuple(_ -> n_nodes, ndims(mesh))...,`
                            n_elements)

    Trixi.@threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            rho, v1, v2, p = prim2cons(u_node, equations)
            c = sqrt(equations.gamma * p / rho) # speed of sound
            v_magnitude = sqrt(v1^2 + v2^2)

            variable_array[i, j, element] = v_magnitude / c
        end
    end

    return variable_array
end

function get_nodal_variable(variable, u, mesh, equations, dg, cache,
                                 equations_parabolic, cache_parabolic)
    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)
    # By definition, the variable must be provided at every node of every element!
    # Otherwise, the `SaveSolutionCallback` will crash.
    variable_array = zeros(eltype(cache.elements),
                            n_nodes, n_nodes, # equivalent: `ntuple(_ -> n_nodes, ndims(mesh))...,`
                            n_elements)

    @unpack parabolic_container = cache_parabolic
    @unpack gradients = parabolic_container
    gradients_x, gradients_y = gradients

    Trixi.@threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)

            gradients_1 = get_node_vars(gradients_x, equations_parabolic, dg,
                                        i, j, element)
            gradients_2 = get_node_vars(gradients_y, equations_parabolic, dg,
                                        i, j, element)

            variable_nodal = variable(u_node, (gradients_1, gradients_2),
                                        equations_parabolic)
            variable_array[i, j, element] = variable_nodal
        end
    end

    return variable_array
end

end # @muladd
