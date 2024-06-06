function get_n_levels(mesh::TreeMesh, alg)
    min_level = minimum_level(mesh.tree)
    max_level = maximum_level(mesh.tree)

    # NOTE: For 1D, periodic BC testcase with artificial assignment
    #=
    Random.seed!(42)
    min_level = 1 # Hard-coded to our convergence study testcase
    max_level = 2 # Hard-coded to our convergence study testcase
    =#

    n_levels = max_level - min_level + 1

    # TODO: For case with locally changing mean speed of sound (Lin. Euler)
    #n_levels = 10

    return n_levels
end

function get_n_levels(mesh::P4estMesh, alg)
    n_levels = alg.NumMethods

    return n_levels
end

function partitioning_variables!(level_info_elements,
                                 level_info_elements_acc,
                                 level_info_interfaces_acc,
                                 level_info_boundaries_acc,
                                 level_info_boundaries_orientation_acc,
                                 level_info_mortars_acc,
                                 n_levels, n_dims, mesh::TreeMesh, dg, cache, alg)
  @unpack elements, interfaces, boundaries = cache

  max_level = maximum_level(mesh.tree)

  n_elements = length(elements.cell_ids)
  # Determine level for each element
  for element_id in 1:n_elements
      # Determine level
      # NOTE: For really different grid sizes
      level = mesh.tree.levels[elements.cell_ids[element_id]]

      # NOTE: For 1D, periodic BC testcase with artificial assignment
      #level = rand(min_level:max_level)

      # Convert to level id
      level_id = max_level + 1 - level
      

      # TODO: For case with locally changing mean speed of sound (Lin. Euler)
      #=
      c_max_el = 0.0
      for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
          u_node = get_node_vars(u, equations, dg, i, j, k, element_id)

          c = u_node[end]
          if c > c_max_el
              c_max_el = c
          end
      end
      # Similar to procedure for P4est
      level_id = findfirst(x -> x < c_max_el, alg.dtRatios)
      # Catch case that cell is "too coarse" for method with fewest stage evals
      if level_id === nothing
          level_id = n_levels
      else # Avoid reduction in timestep: Use next higher level
          level_id = level_id - 1
      end
      =#

      push!(level_info_elements[level_id], element_id)
      # Add to accumulated container
      for l in level_id:n_levels
          push!(level_info_elements_acc[l], element_id)
      end
  end

  n_interfaces = length(interfaces.orientations)
  # Determine level for each interface
  for interface_id in 1:n_interfaces
      # Get element id: Interfaces only between elements of same size
      
      element_id = interfaces.neighbor_ids[1, interface_id]

      # Determine level
      level = mesh.tree.levels[elements.cell_ids[element_id]]

      level_id = max_level + 1 - level
      
      #=
      # NOTE: For case with varying characteristic speeds
      
      el_id_1 = interfaces.neighbor_ids[1, interface_id]
      el_id_2 = interfaces.neighbor_ids[2, interface_id]

      level_1 = 0
      level_2 = 0

      for level in 1:n_levels
          if el_id_1 in level_info_elements[level]
              level_1 = level
              break
          end
      end

      for level in 1:n_levels
          if el_id_2 in level_info_elements[level]
              level_2 = level
              break
          end
      end

      level_id = min(level_1, level_2)
      =#

      # NOTE: For 1D, periodic BC testcase with artificial assignment
      #=
      if element_id in level_info_elements[1]
      level_id = 1
      elseif element_id in level_info_elements[2]
      level_id = 2
      end
      =#

      for l in level_id:n_levels
          push!(level_info_interfaces_acc[l], interface_id)
      end
  end

  n_boundaries = length(boundaries.orientations)
  # Determine level for each boundary
  for boundary_id in 1:n_boundaries
      # Get element id (boundaries have only one unique associated element)
      element_id = boundaries.neighbor_ids[boundary_id]

      # Determine level
      level = mesh.tree.levels[elements.cell_ids[element_id]]

      # Convert to level id
      level_id = max_level + 1 - level

      # Add to accumulated container
      for l in level_id:n_levels
          push!(level_info_boundaries_acc[l], boundary_id)
      end

      # For orientation-side wise specific treatment
      if boundaries.orientations[boundary_id] == 1 # x Boundary
          if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
              for l in level_id:n_levels
                  push!(level_info_boundaries_orientation_acc[l][2], boundary_id)
              end
          else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
              for l in level_id:n_levels
                  push!(level_info_boundaries_orientation_acc[l][1], boundary_id)
              end
          end
      elseif boundaries.orientations[boundary_id] == 2 # y Boundary
          if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
              for l in level_id:n_levels
                  push!(level_info_boundaries_orientation_acc[l][4], boundary_id)
              end
          else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
              for l in level_id:n_levels
                  push!(level_info_boundaries_orientation_acc[l][3], boundary_id)
              end
          end
      elseif boundaries.orientations[boundary_id] == 3 # z Boundary
          if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
              for l in level_id:n_levels
                  push!(level_info_boundaries_orientation_acc[l][6], boundary_id)
              end
          else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
              for l in level_id:n_levels
                  push!(level_info_boundaries_orientation_acc[l][5], boundary_id)
              end
          end
      end
  end

  if n_dims > 1
      @unpack mortars = cache
      n_mortars = length(mortars.orientations)

      for mortar_id in 1:n_mortars
          # This is by convention always one of the finer elements
          element_id = mortars.neighbor_ids[1, mortar_id]

          # Determine level
          level = mesh.tree.levels[elements.cell_ids[element_id]]

          # Higher element's level determines this mortars' level
          level_id = max_level + 1 - level
          # Add to accumulated container
          for l in level_id:n_levels
              push!(level_info_mortars_acc[l], mortar_id)
          end
      end
  end
end

function partitioning_variables!(level_info_elements,
                                 level_info_elements_acc,
                                 level_info_interfaces_acc,
                                 level_info_boundaries_acc,
                                 level_info_boundaries_orientation_acc, # TODO: Not yet adapted for P4est!
                                 level_info_mortars_acc,
                                 n_levels, n_dims, mesh::P4estMesh, dg, cache, alg)
  @unpack elements, interfaces, boundaries = cache

  nnodes = length(mesh.nodes)
  n_elements = nelements(dg, cache)
  h_min = 42
  h_max = 0.0

  h_min_per_element = zeros(n_elements)

  if typeof(mesh) <: P4estMesh{2}
      for element_id in 1:n_elements
          # pull the four corners numbered as right-handed
          P0 = elements.node_coordinates[:, 1, 1, element_id]
          P1 = elements.node_coordinates[:, nnodes, 1, element_id]
          P2 = elements.node_coordinates[:, nnodes, nnodes, element_id]
          P3 = elements.node_coordinates[:, 1, nnodes, element_id]
          # compute the four side lengths and get the smallest
          L0 = sqrt(sum((P1 - P0) .^ 2))
          L1 = sqrt(sum((P2 - P1) .^ 2))
          L2 = sqrt(sum((P3 - P2) .^ 2))
          L3 = sqrt(sum((P0 - P3) .^ 2))
          h = min(L0, L1, L2, L3)

          # For square elements (RTI)
          #L0 = abs(P1[1] - P0[1])
          #h = L0

          h_min_per_element[element_id] = h
          if h > h_max
              h_max = h
          end
          if h < h_min
              h_min = h
          end
      end
  else # typeof(mesh) <:P4estMesh{3}
    # TODO
  end

  println("h_min: ", h_min, " h_max: ", h_max)
  println("h_max/h_min: ", h_max / h_min)

  println("dtRatios:")
  display(alg.dtRatios)

  println("\n")

  for element_id in 1:n_elements
      h = h_min_per_element[element_id]

      # Beyond linear scaling of timestep
      level = findfirst(x -> x < h_min / h, alg.dtRatios)
      # Catch case that cell is "too coarse" for method with fewest stage evals
      if level === nothing
          level = n_levels
      else # Avoid reduction in timestep: Use next higher level
          level = level - 1
      end

      append!(level_info_elements[level], element_id)

      for l in level:n_levels
          push!(level_info_elements_acc[l], element_id)
      end
  end

  n_interfaces = last(size(interfaces.u))
  # Determine level for each interface
  for interface_id in 1:n_interfaces
      # For p4est: Cells on same level do not necessarily have same size
      element_id1 = interfaces.neighbor_ids[1, interface_id]
      element_id2 = interfaces.neighbor_ids[2, interface_id]
      h1 = h_min_per_element[element_id1]
      h2 = h_min_per_element[element_id2]
      h = min(h1, h2)

      # Beyond linear scaling of timestep
      level = findfirst(x -> x < h_min / h, alg.dtRatios)
      # Catch case that cell is "too coarse" for method with fewest stage evals
      if level === nothing
          level = n_levels
      else # Avoid reduction in timestep: Use next higher level
          level = level - 1
      end

      for l in level:n_levels
          push!(level_info_interfaces_acc[l], interface_id)
      end
  end
  
  n_boundaries = last(size(boundaries.u))
  # Determine level for each boundary
  for boundary_id in 1:n_boundaries
      # Get element id (boundaries have only one unique associated element)
      element_id = boundaries.neighbor_ids[boundary_id]
      h = h_min_per_element[element_id]

      # Beyond linear scaling of timestep
      level = findfirst(x -> x < h_min / h, alg.dtRatios)
      # Catch case that cell is "too coarse" for method with fewest stage evals
      if level === nothing
          level = n_levels
      else # Avoid reduction in timestep: Use next higher level
          level = level - 1
      end

      # Add to accumulated container
      for l in level:n_levels
          push!(level_info_boundaries_acc[l], boundary_id)
      end
  end

    if n_dims > 1
        @unpack mortars = cache # TODO: Could also make dimensionality check
        n_mortars = last(size(mortars.u))

        for mortar_id in 1:n_mortars
            # Get element ids
            element_id_lower = mortars.neighbor_ids[1, mortar_id]
            h_lower = h_min_per_element[element_id_lower]

            element_id_higher = mortars.neighbor_ids[2, mortar_id]
            h_higher = h_min_per_element[element_id_higher]

            h = min(h_lower, h_higher)

            # Beyond linear scaling of timestep
            level = findfirst(x -> x < h_min / h, alg.dtRatios)
            # Catch case that cell is "too coarse" for method with fewest stage evals
            if level === nothing
                level = n_levels
            else # Avoid reduction in timestep: Use next higher level
                level = level - 1
            end

            # Add to accumulated container
            for l in level:n_levels
                push!(level_info_mortars_acc[l], mortar_id)
            end
        end
    end
end

function partitioning_u!(level_u_indices_elements, 
                         n_levels, n_dims, level_info_elements, u_ode, mesh, equations, dg, cache)
  u = wrap_array(u_ode, mesh, equations, dg, cache)

  if n_dims == 1
    for level in 1:n_levels
        for element_id in level_info_elements[level]
            # First dimension of u: nvariables, following: nnodes (per dim) last: nelements                                    
            indices = vec(transpose(LinearIndices(u)[:, :, element_id]))
            append!(level_u_indices_elements[level], indices)
        end
        sort!(level_u_indices_elements[level])
        @assert length(level_u_indices_elements[level]) ==
                nvariables(equations) * Trixi.nnodes(dg)^ndims(mesh) *
                length(level_info_elements[level])
    end
  elseif n_dims == 2
      for level in 1:n_levels
          for element_id in level_info_elements[level]
              # First dimension of u: nvariables, following: nnodes (per dim) last: nelements
              indices = collect(Iterators.flatten(LinearIndices(u)[:, :, :,
                                                                  element_id]))
              append!(level_u_indices_elements[level], indices)
          end
          sort!(level_u_indices_elements[level])
          @assert length(level_u_indices_elements[level]) ==
                  nvariables(equations) * Trixi.nnodes(dg)^ndims(mesh) *
                  length(level_info_elements[level])
      end
  elseif n_dims == 3
      for level in 1:n_levels
          for element_id in level_info_elements[level]
              # First dimension of u: nvariables, following: nnodes (per dim) last: nelements
              indices = collect(Iterators.flatten(LinearIndices(u)[:, :, :, :,
                                                                  element_id]))
              append!(level_u_indices_elements[level], indices)
          end
          sort!(level_u_indices_elements[level])
          @assert length(level_u_indices_elements[level]) ==
                  nvariables(equations) * Trixi.nnodes(dg)^ndims(mesh) *
                  length(level_info_elements[level])
      end
  end
end