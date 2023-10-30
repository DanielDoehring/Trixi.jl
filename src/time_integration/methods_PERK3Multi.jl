# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

#using Random # NOTE: Only for tests

function ComputePERK3_Multi_ButcherTableau(NumDoublings::Int, NumStages::Int, BasePathMonCoeffs::AbstractString, 
                                           cS2::Float64)
                                     
  # c Vector form Butcher Tableau (defines timestep per stage)
  c = zeros(NumStages)
  for k in 2:NumStages-2
    c[k] = cS2 * (k - 1)/(NumStages - 3) # Equidistant timestep distribution (similar to PERK2)
  end
  # Proposed PERK
  #=
  c[NumStages - 1] = 1.0/3.0
  c[NumStages]     = 1.0
  =#
  
  # Own PERK based on SSPRK33
  
  c[NumStages - 1] = 1.0
  c[NumStages]     = 0.5
  
  println("Timestep-split: "); display(c); println("\n")

  # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
  CoeffsMax = NumStages - 2

  AMatrices = zeros(NumDoublings+1, CoeffsMax, 2)
  for i = 1:NumDoublings+1
    AMatrices[i, :, 1] = c[3:end]
  end

  # Datastructure indicating at which stage which level is evaluated
  ActiveLevels = [Vector{Int64}() for _ in 1:NumStages]
  # k1 is evaluated at all levels
  ActiveLevels[1] = 1:NumDoublings+1

  for level = 1:NumDoublings + 1
    
    #=
    PathMonCoeffs = BasePathMonCoeffs * "gamma_" * string(Int(NumStages / 2^(level - 1))) * ".txt"
    NumMonCoeffs, MonCoeffs = read_file(PathMonCoeffs, Float64)
    @assert NumMonCoeffs == NumStages / 2^(level - 1) - 2
    A = ComputeACoeffs(Int(NumStages / 2^(level - 1)), SE_Factors, MonCoeffs)

    AMatrices[level, CoeffsMax - Int(NumStages / 2^(level - 1) - 3):end, 1] -= A
    AMatrices[level, CoeffsMax - Int(NumStages / 2^(level - 1) - 3):end, 2]  = A
    =#

    PathMonCoeffs = BasePathMonCoeffs * "a_" * string(Int(NumStages / 2^(level - 1))) * "_" * string(NumStages) * ".txt"
    NumMonCoeffs, A = read_file(PathMonCoeffs, Float64)
    @assert NumMonCoeffs == NumStages / 2^(level - 1) - 2

    AMatrices[level, CoeffsMax - Int(NumStages / 2^(level - 1) - 3):end, 1] -= A
    AMatrices[level, CoeffsMax - Int(NumStages / 2^(level - 1) - 3):end, 2]  = A

    #=
    # NOTE: For linear PERK family: 4,6,8, and not 4, 8, 16, ...
    PathMonCoeffs = BasePathMonCoeffs * "gamma_" * string(Int(NumStages - 2*level + 2)) * ".txt"
    NumMonCoeffs, MonCoeffs = read_file(PathMonCoeffs, Float64)
    @assert NumMonCoeffs == NumStages - 2*level
    A = ComputeACoeffs(Int(NumStages - 2*level + 2), SE_Factors, MonCoeffs)

    AMatrices[level, CoeffsMax - Int(NumStages - 2*level - 1):end, 1] -= A
    AMatrices[level, CoeffsMax - Int(NumStages - 2*level - 1):end, 2]  = A
    =#

    # Add active levels to stages
    for stage = NumStages:-1:NumStages-NumMonCoeffs
      push!(ActiveLevels[stage], level)
    end
  end
  HighestActiveLevels = maximum.(ActiveLevels)

  for i = 1:NumDoublings+1
    println("A-Matrix of Butcher tableau of level " * string(i))
    display(AMatrices[i, :, :]); println()
  end

  println("Check violation of internal consistency")
  for i = 1:NumDoublings+1
    for j = 1:i
      display(norm(AMatrices[i, :, 1] + AMatrices[i, :, 2] - AMatrices[j, :, 1] - AMatrices[j, :, 2], 1))
    end
  end

  println("\nActive Levels:"); display(ActiveLevels); println()

  return AMatrices, c, ActiveLevels, HighestActiveLevels
end

function ComputePERK3_Multi_ButcherTableau(Stages::Vector{Int64}, NumStages::Int, BasePathMonCoeffs::AbstractString, 
                                           cS2::Float64)
                                     
  # c Vector form Butcher Tableau (defines timestep per stage)
  c = zeros(NumStages)
  for k in 2:NumStages-2
    c[k] = cS2 * (k - 1)/(NumStages - 3) # Equidistant timestep distribution (similar to PERK2)
  end
  # Proposed PERK
  #=
  c[NumStages - 1] = 1.0/3.0
  c[NumStages]     = 1.0
  =#
  
  # Own PERK based on SSPRK33
  
  c[NumStages - 1] = 1.0
  c[NumStages]     = 0.5
  
  println("Timestep-split: "); display(c); println("\n")

  # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
  CoeffsMax = NumStages - 2

  AMatrices = zeros(length(Stages), CoeffsMax, 2)
  for i = 1:length(Stages)
    AMatrices[i, :, 1] = c[3:end]
  end

  # Datastructure indicating at which stage which level is evaluated
  ActiveLevels = [Vector{Int64}() for _ in 1:NumStages]
  # k1 is evaluated at all levels
  ActiveLevels[1] = 1:length(Stages)

  for level in eachindex(Stages)

    NumStageEvals = Stages[level]
    PathMonCoeffs = BasePathMonCoeffs * "a_" * string(NumStageEvals) * "_" * string(NumStages) * ".txt"
    NumMonCoeffs, A = read_file(PathMonCoeffs, Float64)
    @assert NumMonCoeffs == NumStageEvals - 2

    AMatrices[level, CoeffsMax - NumStageEvals + 3:end, 1] -= A
    AMatrices[level, CoeffsMax - NumStageEvals + 3:end, 2]  = A

    # Add active levels to stages
    for stage = NumStages:-1:NumStages-NumMonCoeffs
      push!(ActiveLevels[stage], level)
    end
  end
  HighestActiveLevels = maximum.(ActiveLevels)

  for i = 1:length(Stages)
    println("A-Matrix of Butcher tableau of level " * string(i))
    display(AMatrices[i, :, :]); println()
  end

  println("Check violation of internal consistency")
  for i = 1:length(Stages)
    for j = 1:i
      display(norm(AMatrices[i, :, 1] + AMatrices[i, :, 2] - AMatrices[j, :, 1] - AMatrices[j, :, 2], 1))
    end
  end

  println("\nActive Levels:"); display(ActiveLevels); println()

  return AMatrices, c, ActiveLevels, HighestActiveLevels
end

mutable struct PERK3_Multi{StageCallbacks}
  const NumStageEvalsMin::Int64
  const NumDoublings::Int64
  const NumStages::Int64
  #const LevelCFL::Vector{Float64}
  const LevelCFL::Dict{Int64, Float64}
  const Integrator_Mesh_Level_Dict::Dict{Int64, Int64}
  stage_callbacks::StageCallbacks

  AMatrices::Array{Float64, 3}
  c::Vector{Float64}
  ActiveLevels::Vector{Vector{Int64}}
  HighestActiveLevels::Vector{Int64}

  function PERK3_Multi(NumStageEvalsMin_::Int, NumDoublings_::Int,
                       BasePathMonCoeffs_::AbstractString, cS2_::Float64,
                       #LevelCFL_::Vector{Float64},
                       LevelCFL_::Dict{Int64, Float64},
                       Integrator_Mesh_Level_Dict_::Dict{Int64, Int64};
                       stage_callbacks=())

    newPERK3_Multi = new{typeof(stage_callbacks)}(NumStageEvalsMin_, NumDoublings_,
                        # Current convention: NumStages = MaxStages = S;
                        # TODO: Allow for different S >= Max {Stage Evals}
                        NumStageEvalsMin_ * 2^NumDoublings_,
                        # Note: This is for linear increasing PERK
                        #NumStageEvalsMin_ + 2 * NumDoublings_,
                        LevelCFL_, Integrator_Mesh_Level_Dict_,
                        stage_callbacks)

    newPERK3_Multi.AMatrices, newPERK3_Multi.c, newPERK3_Multi.ActiveLevels, newPERK3_Multi.HighestActiveLevels = 
      ComputePERK3_Multi_ButcherTableau(NumDoublings_, newPERK3_Multi.NumStages, BasePathMonCoeffs_, cS2_)

    return newPERK3_Multi
  end

  function PERK3_Multi(Stages_::Vector{Int64},
                       BasePathMonCoeffs_::AbstractString, cS2_::Float64,
                       #LevelCFL_::Vector{Float64},
                       LevelCFL_::Dict{Int64, Float64},
                       Integrator_Mesh_Level_Dict_::Dict{Int64, Int64};
                       stage_callbacks=())

    newPERK3_Multi = new{typeof(stage_callbacks)}(minimum(Stages_),
                          length(Stages_) - 1,
                          maximum(Stages_),
                          LevelCFL_, Integrator_Mesh_Level_Dict_,
                          stage_callbacks)

    newPERK3_Multi.AMatrices, newPERK3_Multi.c, newPERK3_Multi.ActiveLevels, newPERK3_Multi.HighestActiveLevels = 
      ComputePERK3_Multi_ButcherTableau(Stages_, newPERK3_Multi.NumStages, BasePathMonCoeffs_, cS2_)

    return newPERK3_Multi
  end
end # struct PERK3_Multi


# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PERK3_Multi_Integrator{RealT<:Real, uType, Params, Sol, F, Alg, PERK_IntegratorOptions}
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
  # PERK3_Multi stages:
  k1::uType
  k_higher::uType
  k_S1::uType # Required for third order
  # Variables managing level-depending integration
  level_info_elements::Vector{Vector{Int64}}
  level_info_elements_acc::Vector{Vector{Int64}}
  level_info_interfaces_acc::Vector{Vector{Int64}}
  level_info_boundaries_acc::Vector{Vector{Int64}}
  level_info_boundaries_orientation_acc::Vector{Vector{Vector{Int64}}}
  level_info_mortars_acc::Vector{Vector{Int64}}
  level_u_indices_elements::Vector{Vector{Int64}}
  t_stage::RealT
  coarsest_lvl::Int64
  n_levels::Int64
  min_lvl::Int64
  max_lvl::Int64
  du_ode_hyp::uType # TODO: Not best solution since this is not needed for hyperbolic problems
  dtRef::RealT
  AddRHSCalls::Int64
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::PERK3_Multi_Integrator, field::Symbol)
  if field === :stats
    return (naccept = getfield(integrator, :iter),)
  end
  # general fallback
  return getfield(integrator, field)
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::PERK3_Multi;
               dt, callback=nothing, kwargs...)

  u0    = copy(ode.u0)
  du    = similar(u0)
  u_tmp = similar(u0)

  # PERK3_Multi stages
  k1       = similar(u0)
  k_higher = similar(u0)
  k_S1     = similar(u0)

  du_ode_hyp = similar(u0) # TODO: Not best solution since this is not needed for hyperbolic problems

  t0 = first(ode.tspan)
  iter = 0

  ### Set datastructures for handling of level-dependent integration ###
  mesh, _, dg, cache = mesh_equations_solver_cache(ode.p)
  @unpack elements, interfaces, boundaries = cache

  if typeof(mesh) <:TreeMesh
    n_elements   = length(elements.cell_ids)
    n_interfaces = length(interfaces.orientations)
    n_boundaries = length(boundaries.orientations) # TODO Not sure if adequate, especially multiple dimensions

    # NOTE: For really different grid sizes
    
    min_level = minimum_level(mesh.tree)
    max_level = maximum_level(mesh.tree)
    
    # NOTE: For testcase with artificial assignment
    #=
    Random.seed!(42)
    min_level = 1 # Hard-coded to our convergence study testcase
    max_level = 3 # Hard-coded to our convergence study testcase
    =#
    n_levels = max_level - min_level + 1
    
    # Initialize storage for level-wise information
    level_info_elements     = [Vector{Int64}() for _ in 1:n_levels]
    level_info_elements_acc = [Vector{Int64}() for _ in 1:n_levels]

    # Determine level for each element
    for element_id in 1:n_elements
      # Determine level
      # NOTE: For really different grid sizes
      level = mesh.tree.levels[elements.cell_ids[element_id]]
      # NOTE: For testcase with artificial assignment
      #level = rand((1, 2, 3))

      # Convert to level id
      level_id = max_level + 1 - level

      push!(level_info_elements[level_id], element_id)
      # Add to accumulated container
      for l in level_id:n_levels
        push!(level_info_elements_acc[l], element_id)
      end
    end
    @assert length(level_info_elements_acc[end]) == 
      n_elements "highest level should contain all elements"


    level_info_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]
    # Determine level for each interface
    for interface_id in 1:n_interfaces
      # Get element ids
      element_id_left  = interfaces.neighbor_ids[1, interface_id]
      element_id_right = interfaces.neighbor_ids[2, interface_id]

      # Determine level
      # NOTE: For really different grid sizes
      
      level_left  = mesh.tree.levels[elements.cell_ids[element_id_left]]
      level_right = mesh.tree.levels[elements.cell_ids[element_id_right]]

      # Higher element's level determines this interfaces' level
      level_id = max_level + 1 - max(level_left, level_right)
      

      # NOTE: For testcase with artificial assignment
      #=
      if element_id_left in level_info_elements[1]
        level_id_left = 1
      elseif element_id_left in level_info_elements[2]
        level_id_left = 2
      elseif element_id_left in level_info_elements[3]
        level_id_left = 3
      end

      if element_id_right in level_info_elements[1]
        level_id_right = 1
      elseif element_id_right in level_info_elements[2]
        level_id_right = 2
      elseif element_id_right in level_info_elements[3]
        level_id_right = 3
      end
      level_id = min(level_id_left, level_id_right)
      =#

      for l in level_id:n_levels
        push!(level_info_interfaces_acc[l], interface_id)
      end
    end
    @assert length(level_info_interfaces_acc[end]) == 
      n_interfaces "highest level should contain all interfaces"


    level_info_boundaries_acc = [Vector{Int64}() for _ in 1:n_levels]
    # For efficient treatment of boundaries we need additional datastructures
    n_dims = ndims(mesh.tree) # Spatial dimension
    level_info_boundaries_orientation_acc = [[Vector{Int64}() for _ in 1:2*n_dims] for _ in 1:n_levels]

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
    @assert length(level_info_boundaries_acc[end]) == 
      n_boundaries "highest level should contain all boundaries"


    level_info_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]
    if n_dims > 1
      @unpack mortars = cache
      n_mortars = length(mortars.orientations)

      for mortar_id in 1:n_mortars
        # This is by convention always one of the finer elements
        element_id  = mortars.neighbor_ids[1, mortar_id]

        # Determine level
        level  = mesh.tree.levels[elements.cell_ids[element_id]]

        # Higher element's level determines this mortars' level
        level_id = max_level + 1 - level
        # Add to accumulated container
        for l in level_id:n_levels
          push!(level_info_mortars_acc[l], mortar_id)
        end
      end
      @assert length(level_info_mortars_acc[end]) == 
        n_mortars "highest level should contain all mortars"
    end
  elseif typeof(mesh) <:P4estMesh{2}
    nnodes = length(mesh.nodes)
    n_elements = nelements(dg, cache)
    h_min = 42;
    h_max = 0;

    h_min_per_element = zeros(n_elements)

    for element_id in 1:n_elements
      # pull the four corners numbered as right-handed
      P0 = cache.elements.node_coordinates[:, 1     , 1     , element_id]
      P1 = cache.elements.node_coordinates[:, nnodes, 1     , element_id]
      P2 = cache.elements.node_coordinates[:, nnodes, nnodes, element_id]
      P3 = cache.elements.node_coordinates[:, 1     , nnodes, element_id]
      # compute the four side lengths and get the smallest
      L0 = sqrt( sum( (P1-P0).^2 ) )
      L1 = sqrt( sum( (P2-P1).^2 ) )
      L2 = sqrt( sum( (P3-P2).^2 ) )
      L3 = sqrt( sum( (P0-P3).^2 ) )
      h = min(L0, L1, L2, L3)
      h_min_per_element[element_id] = h
      if h > h_max 
        h_max = h
      end
      if h < h_min
        h_min = h
      end
    end

    S_min = alg.NumStageEvalsMin
    S_max = alg.NumStages
    n_levels = Int((S_max - S_min)/2) + 1
    if n_levels == 1
      h_bins = [h_max]
    else
      h_bins = LinRange(h_min, h_max, n_levels)
    end

    println("h_min: ", h_min, " h_max: ", h_max)
    println("h_max/h_min: ", h_max/h_min)
    println("h_bins:")
    display(h_bins)
    println("\n")

    level_info_elements = [Vector{Int64}() for _ in 1:n_levels]
    level_info_elements_acc = [Vector{Int64}() for _ in 1:n_levels]
    for element_id in 1:n_elements
      h = h_min_per_element[element_id]

      level = findfirst(x-> x >= h, h_bins)
      append!(level_info_elements[level], element_id)

      for l in level:n_levels
        push!(level_info_elements_acc[l], element_id)
      end
    end
    level_info_elements_count = Vector{Int64}(undef, n_levels)
    for i in eachindex(level_info_elements)
      level_info_elements_count[i] = length(level_info_elements[i])
    end

    n_interfaces = last(size(interfaces.u))

    level_info_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]
    # Determine level for each interface
    for interface_id in 1:n_interfaces
      # Get element ids
      element_id_left  = interfaces.neighbor_ids[1, interface_id]
      h_left = h_min_per_element[element_id_left]

      element_id_right = interfaces.neighbor_ids[2, interface_id]
      h_right = h_min_per_element[element_id_right]

      # Determine level
      h = min(h_left, h_right)
      level = findfirst(x-> x >= h, h_bins)

      for l in level:n_levels
        push!(level_info_interfaces_acc[l], interface_id)
      end
    end
    @assert length(level_info_interfaces_acc[end]) == 
      n_interfaces "highest level should contain all interfaces"

    n_boundaries = last(size(boundaries.u))
    level_info_boundaries_acc = [Vector{Int64}() for _ in 1:n_levels]
    # For efficient treatment of boundaries we need additional datastructures
    n_dims = ndims(mesh) # Spatial dimension
    # TODO: Not yet adapted for P4est!
    level_info_boundaries_orientation_acc = [[Vector{Int64}() for _ in 1:2*n_dims] for _ in 1:n_levels]

    # Determine level for each boundary
    for boundary_id in 1:n_boundaries
      # Get element id (boundaries have only one unique associated element)
      element_id = boundaries.neighbor_ids[boundary_id]
      h = h_min_per_element[element_id]

      # Determine level
      level = findfirst(x-> x >= h, h_bins)

      # Add to accumulated container
      for l in level:n_levels
        push!(level_info_boundaries_acc[l], boundary_id)
      end
    end
    @assert length(level_info_boundaries_acc[end]) == 
      n_boundaries "highest level should contain all boundaries"

    @unpack mortars = cache # TODO: Could also make dimensionality check
    level_info_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]
    @unpack mortars = cache
    n_mortars = last(size(mortars.u))

    for mortar_id in 1:n_mortars
      # Get element ids
      element_id_lower  = mortars.neighbor_ids[1, mortar_id]
      h_lower = h_min_per_element[element_id_lower]

      element_id_higher = mortars.neighbor_ids[2, mortar_id]
      h_higher = h_min_per_element[element_id_higher]

      # Determine level
      h = min(h_lower, h_higher)
      level = findfirst(x-> x >= h, h_bins)

      # Add to accumulated container
      for l in level:n_levels
        push!(level_info_mortars_acc[l], mortar_id)
      end
    end
    @assert length(level_info_mortars_acc[end]) == 
      n_mortars "highest level should contain all mortars"
  end

  println("level_info_elements:")
  display(level_info_elements); println()
  println("level_info_elements_acc:")
  display(level_info_elements_acc); println()

  println("level_info_interfaces_acc:")
  display(level_info_interfaces_acc); println()

  println("level_info_boundaries_acc:")
  display(level_info_boundaries_acc); println()
  println("level_info_boundaries_orientation_acc:")
  display(level_info_boundaries_orientation_acc); println()
  
  println("level_info_mortars_acc:")
  display(level_info_mortars_acc); println()

  
  # Set initial distribution of DG Base function coefficients 
  @unpack equations, solver = ode.p
  u = wrap_array(u0, mesh, equations, solver, cache)

  level_u_indices_elements = [Vector{Int64}() for _ in 1:n_levels]

  # Have if outside for performance reasons (this is also used in the AMR calls)
  if n_dims == 1
    for level in 1:n_levels
      for element_id in level_info_elements[level]
        # First dimension of u: nvariables, following: nnodes (per dim) last: nelements                                    
        indices = vec(transpose(LinearIndices(u)[:, :, element_id]))
        append!(level_u_indices_elements[level], indices)
      end
      @assert length(level_u_indices_elements[level]) == 
              nvariables(equations) * Trixi.nnodes(solver)^ndims(mesh) * length(level_info_elements[level])
    end
  elseif n_dims == 2
    for level in 1:n_levels
      for element_id in level_info_elements[level]
        # First dimension of u: nvariables, following: nnodes (per dim) last: nelements
        indices = collect(Iterators.flatten(LinearIndices(u)[:, :, :, element_id]))
        append!(level_u_indices_elements[level], indices)
      end
      @assert length(level_u_indices_elements[level]) == 
              nvariables(equations) * Trixi.nnodes(solver)^ndims(mesh) * length(level_info_elements[level])
    end
  elseif n_dims == 3
    for level in 1:n_levels
      for element_id in level_info_elements[level]
        # First dimension of u: nvariables, following: nnodes (per dim) last: nelements
        indices = collect(Iterators.flatten(LinearIndices(u)[:, :, :, :, element_id]))
        append!(level_u_indices_elements[level], indices)
      end
      @assert length(level_u_indices_elements[level]) == 
              nvariables(equations) * Trixi.nnodes(solver)^ndims(mesh) * length(level_info_elements[level])
    end
  end

  println("level_u_indices_elements:")
  display(level_u_indices_elements); println()

  ### Done with setting up for handling of level-dependent integration ###

  integrator = PERK3_Multi_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                (prob=ode,), ode.f, alg,
                PERK_IntegratorOptions(callback, ode.tspan; kwargs...), false,
                k1, k_higher, k_S1,
                level_info_elements, level_info_elements_acc, 
                level_info_interfaces_acc, 
                level_info_boundaries_acc, level_info_boundaries_orientation_acc,
                level_info_mortars_acc, 
                level_u_indices_elements,
                t0, -1, n_levels, min_level, max_level, du_ode_hyp, dt, 0)
            
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
  solve!(integrator)
end


function solve!(integrator::PERK3_Multi_Integrator)
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
    #else
    #  integrator.dt = integrator.dtRef * alg.LevelCFL[integrator.max_lvl]
    end

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
      
      # k1: Evaluated on entire domain / all levels
      #integrator.f(integrator.du, integrator.u, prob.p, integrator.t, integrator.du_ode_hyp)
      integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
      
      @threaded for i in eachindex(integrator.du)
        integrator.k1[i] = integrator.du[i] * integrator.dt
      end
      
      integrator.t_stage = integrator.t + alg.c[2] * integrator.dt
      # k2: Here always evaluated for finest scheme (Allow currently only max. stage evaluations)
      @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] + alg.c[2] * integrator.k1[i]
      end

      #=
      for stage_callback in alg.stage_callbacks
        stage_callback(integrator.u_tmp, integrator, prob.p, integrator.t_stage)
      end
      =#

      #=
      integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, 
                   integrator.level_info_elements_acc[1],
                   integrator.level_info_interfaces_acc[1],
                   integrator.level_info_boundaries_acc[1],
                   integrator.level_info_boundaries_orientation_acc[1],
                   integrator.level_info_mortars_acc[1],
                   integrator.level_u_indices_elements, 1,
                   integrator.du_ode_hyp)
      =#
      
      
      integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, 
                   integrator.level_info_elements_acc[1],
                   integrator.level_info_interfaces_acc[1],
                   integrator.level_info_boundaries_acc[1],
                   integrator.level_info_boundaries_orientation_acc[1],
                   integrator.level_info_mortars_acc[1])
      
      @threaded for u_ind in integrator.level_u_indices_elements[1] # Update finest level
        integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
      end

      for stage = 3:alg.NumStages
        # Construct current state
        @threaded for i in eachindex(integrator.u)
          integrator.u_tmp[i] = integrator.u[i]
        end

        #=
        for level in 1:integrator.n_levels # Ensures only relevant levels are evaluated
          Integrator_lvl = alg.Integrator_Mesh_Level_Dict[integrator.max_lvl - level + 1]
          @threaded for u_ind in integrator.level_u_indices_elements[level]
            integrator.u_tmp[u_ind] += alg.AMatrices[Integrator_lvl, stage - 2, 1] * integrator.k1[u_ind]
          end

          # TODO Try more efficient way
          if alg.AMatrices[Integrator_lvl, stage - 2, 2] > 0
            @threaded for u_ind in integrator.level_u_indices_elements[level]
              integrator.u_tmp[u_ind] += alg.AMatrices[Integrator_lvl, stage - 2, 2] * integrator.k_higher[u_ind]
            end
          end
        end
        =#

        # Hard-coded "three integrators" approach: Fill lower with lowest
        # level 1
        @threaded for u_ind in integrator.level_u_indices_elements[1]
          integrator.u_tmp[u_ind] += alg.AMatrices[1, stage - 2, 1] * integrator.k1[u_ind] + 
                                     alg.AMatrices[1, stage - 2, 2] * integrator.k_higher[u_ind]
        end

        
        # level 2
        if integrator.n_levels > 1
          @threaded for u_ind in integrator.level_u_indices_elements[2]
            integrator.u_tmp[u_ind] += alg.AMatrices[2, stage - 2, 1] * integrator.k1[u_ind]
          end
          # TODO Try more efficient way
          if alg.AMatrices[2, stage - 2, 2] > 0
            @threaded for u_ind in integrator.level_u_indices_elements[2]
              integrator.u_tmp[u_ind] += alg.AMatrices[2, stage - 2, 2] * integrator.k_higher[u_ind]
            end
          end
        end

        if integrator.n_levels > 2
          @threaded for u_ind in integrator.level_u_indices_elements[3]
            integrator.u_tmp[u_ind] += alg.AMatrices[3, stage - 2, 1] * integrator.k1[u_ind]
          end
          # TODO Try more efficient way
          if alg.AMatrices[3, stage - 2, 2] > 0
            @threaded for u_ind in integrator.level_u_indices_elements[3]
              integrator.u_tmp[u_ind] += alg.AMatrices[3, stage - 2, 2] * integrator.k_higher[u_ind]
            end
          end
        end
        

        for level in 4:integrator.n_levels # Ensures only relevant levels are evaluated
        #for level in 3:integrator.n_levels # Ensures only relevant levels are evaluated
        #for level in 2:integrator.n_levels # Ensures only relevant levels are evaluated
          @threaded for u_ind in integrator.level_u_indices_elements[level]
            integrator.u_tmp[u_ind] += alg.AMatrices[4, stage - 2, 1] * integrator.k1[u_ind]
            #integrator.u_tmp[u_ind] += alg.AMatrices[3, stage - 2, 1] * integrator.k1[u_ind]
            #integrator.u_tmp[u_ind] += alg.AMatrices[2, stage - 2, 1] * integrator.k1[u_ind]
          end

          # TODO Try more efficient way
          if alg.AMatrices[4, stage - 2, 2] > 0
          #if alg.AMatrices[3, stage - 2, 2] > 0
          #if alg.AMatrices[2, stage - 2, 2] > 0
            @threaded for u_ind in integrator.level_u_indices_elements[level]
              integrator.u_tmp[u_ind] += alg.AMatrices[4, stage - 2, 2] * integrator.k_higher[u_ind]
              #integrator.u_tmp[u_ind] += alg.AMatrices[3, stage - 2, 2] * integrator.k_higher[u_ind]
              #integrator.u_tmp[u_ind] += alg.AMatrices[2, stage - 2, 2] * integrator.k_higher[u_ind]
            end
          end
        end

        integrator.t_stage = integrator.t + alg.c[stage] * integrator.dt

        # "coarsest_lvl" cannot be static for AMR, has to be checked with available levels
        # TODO: Not sure if still valid with dict-based approach
        integrator.coarsest_lvl = min(alg.HighestActiveLevels[stage], integrator.n_levels)

        # Note: For hard-coded three level approach
        if integrator.coarsest_lvl == 4
        #if integrator.coarsest_lvl == 3
        #if integrator.coarsest_lvl == 2
          integrator.coarsest_lvl = integrator.n_levels
        end

        # For statically refined meshes:
        #integrator.coarsest_lvl = alg.HighestActiveLevels[stage]

        #=
        for stage_callback in alg.stage_callbacks
          stage_callback(integrator.u_tmp, integrator, prob.p, integrator.t_stage)
        end
        =#
        
        #=
        # Joint RHS evaluation with all elements sharing this timestep
        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, 
                    integrator.level_info_elements_acc[integrator.coarsest_lvl],
                    integrator.level_info_interfaces_acc[integrator.coarsest_lvl],
                    integrator.level_info_boundaries_acc[integrator.coarsest_lvl],
                    integrator.level_info_boundaries_orientation_acc[integrator.coarsest_lvl],
                    integrator.level_info_mortars_acc[integrator.coarsest_lvl],
                    integrator.level_u_indices_elements, integrator.coarsest_lvl,
                    integrator.du_ode_hyp)
        =#
        
        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, 
                    integrator.level_info_elements_acc[integrator.coarsest_lvl],
                    integrator.level_info_interfaces_acc[integrator.coarsest_lvl],
                    integrator.level_info_boundaries_acc[integrator.coarsest_lvl],
                    integrator.level_info_boundaries_orientation_acc[integrator.coarsest_lvl],
                    integrator.level_info_mortars_acc[integrator.coarsest_lvl])
        

        # Update k_higher of relevant levels
        for level in 1:integrator.coarsest_lvl
          @threaded for u_ind in integrator.level_u_indices_elements[level]
            integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
          end
        end

        # TODO: Stop for loop at NumStages -1 to avoid if
        if stage == alg.NumStages - 1
          @threaded for i in eachindex(integrator.du)
            integrator.k_S1[i] = integrator.k_higher[i]
          end
        end
      end
      
      @threaded for i in eachindex(integrator.u)
        # Proposed PERK
        #integrator.u[i] += 0.75 * integrator.k_S1[i] + 0.25 * integrator.k_higher[i]

        # Own PERK based on SSPRK33
        integrator.u[i] += (integrator.k1[i] + integrator.k_S1[i] + 4.0 * integrator.k_higher[i])/6.0
      end
      
      #=
      for stage_callback in alg.stage_callbacks
        stage_callback(integrator.u, integrator, prob.p, integrator.t_stage)
      end
      =#
    end # PERK3_Multi step

    integrator.iter += 1
    integrator.t += integrator.dt

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

  println("Additional RHS Calls: ", integrator.AddRHSCalls)
  
  return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                (prob.u0, integrator.u),
                                integrator.sol.prob)
end

# get a cache where the RHS can be stored
get_du(integrator::PERK3_Multi_Integrator) = integrator.du
get_tmp_cache(integrator::PERK3_Multi_Integrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::PERK3_Multi_Integrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::PERK3_Multi_Integrator, dt)
  integrator.dt = dt
end

function get_proposed_dt(integrator::PERK3_Multi_Integrator)
  return integrator.dt
end

# stop the time integration
function terminate!(integrator::PERK3_Multi_Integrator)
  integrator.finalstep = true
  empty!(integrator.opts.tstops)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PERK3_Multi_Integrator, new_size)
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