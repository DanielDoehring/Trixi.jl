
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

num_flux = flux_godunov
solver = DGSEM(polydeg = 3, surface_flux = num_flux)

coordinates_min = -4.0 # minimum coordinate
coordinates_max = 4.0 # maximum coordinate
length = coordinates_max - coordinates_min

refinement_patches = ((type = "box", coordinates_min = (-1.0,),
                       coordinates_max = (1.0,)),)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                refinement_patches = refinement_patches,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_gauss,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

t_end = 1.0
t_end = length + 1
ode = semidiscretize(semi, (0.0, t_end))

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 1,
                                     extra_analysis_errors = (:conservation_error,),
                                     extra_analysis_integrals = (Trixi.entropy_math,),
                                     #analysis_filename = "entropy_ER.dat",
                                     #analysis_filename = "entropy_standard.dat",
                                     save_analysis = true)
cfl = 3.5 # [16, 8]
#cfl = 5.5 # [32, 16]

stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback
                        #stepsize_callback
                        )

###############################################################################
# run the simulation

path = "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/"
#path = "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/Joint/"

dtRatios = [1, 0.5]
Stages = [16, 8]
#Stages = [32, 16]

relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 10)

ode_alg = Trixi.PairedExplicitRelaxationRK2Multi(Stages, path, dtRatios;
                                                 relaxation_solver = relaxation_solver)

#ode_alg = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)

sol = Trixi.solve(ode, ode_alg,
                  dt = 0.2,
                  save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()

###############################################################################

# Compute node values for scatter plot in physical domain

using LinearAlgebra

function gauss_lobatto_nodes(cell_min, cell_max)
    # Gauss-Lobatto nodes in reference coordinates [-1, 1]
    ref_nodes = [-1, -1/sqrt(5), 1/sqrt(5), 1]
    
    # Map reference nodes to physical coordinates
    cell_center = (cell_max + cell_min) / 2
    cell_half_width = (cell_max - cell_min) / 2
    physical_nodes = cell_center .+ cell_half_width .* ref_nodes
    
    return physical_nodes
end

# For plotting of the overlapping DG nodes
function gauss_lobatto_nodes_inward(cell_min, cell_max)
  inward = 0.2
  ref_nodes = [-1 + inward, -1/sqrt(5), 1/sqrt(5), 1 - inward]
  
  # Map reference nodes to physical coordinates
  cell_center = (cell_max + cell_min) / 2
  cell_half_width = (cell_max - cell_min) / 2
  physical_nodes = cell_center .+ cell_half_width .* ref_nodes
  
  return physical_nodes
end

function compute_all_nodes()
    all_nodes = []
    cell_min = -4.0

    # First 6 cells of size 0.5
    for i in 1:6
        cell_max = cell_min + 0.5
        #nodes = gauss_lobatto_nodes(cell_min, cell_max)
        nodes = gauss_lobatto_nodes_inward(cell_min, cell_max)
        append!(all_nodes, nodes)
        cell_min = cell_max
    end

    # Next 8 cells of size 0.25
    for i in 1:8
        cell_max = cell_min + 0.25
        #nodes = gauss_lobatto_nodes(cell_min, cell_max)
        nodes = gauss_lobatto_nodes_inward(cell_min, cell_max)
        append!(all_nodes, nodes)
        cell_min = cell_max
    end

    # Last 6 cells of size 0.5
    for i in 1:6
        cell_max = cell_min + 0.5
        #nodes = gauss_lobatto_nodes(cell_min, cell_max)
        nodes = gauss_lobatto_nodes_inward(cell_min, cell_max)
        append!(all_nodes, nodes)
        cell_min = cell_max
    end

    return all_nodes
end

# Compute nodes for all cells
all_nodes = compute_all_nodes()

using Plots

# TODO: Compute x values of the DG nodes - what to do with the values at interfaces? Average?

x_limits = [-2, 2]

scatter(all_nodes, sol.u[end],
        xlims = x_limits) # For plain method
#scatter!(sol.u[end]) # For relaxed method

function gauss_lobatto_nodes_inward(cell_min, cell_max)
  inward = 0.2
  ref_nodes = [-1 + inward, -1/sqrt(5), 1/sqrt(5), 1 - inward]
  
  # Map reference nodes to physical coordinates
  cell_center = (cell_max + cell_min) / 2
  cell_half_width = (cell_max - cell_min) / 2
  physical_nodes = cell_center .+ cell_half_width .* ref_nodes
  
  return physical_nodes
end

function compute_all_nodes()
    all_nodes = []
    cell_min = -4.0

    # First 6 cells of size 0.5
    for i in 1:6
        cell_max = cell_min + 0.5
        #nodes = gauss_lobatto_nodes(cell_min, cell_max)
        nodes = gauss_lobatto_nodes_inward(cell_min, cell_max)
        append!(all_nodes, nodes)
        cell_min = cell_max
    end

    # Next 8 cells of size 0.25
    for i in 1:8
        cell_max = cell_min + 0.25
        #nodes = gauss_lobatto_nodes(cell_min, cell_max)
        nodes = gauss_lobatto_nodes_inward(cell_min, cell_max)
        append!(all_nodes, nodes)
        cell_min = cell_max
    end

    # Last 6 cells of size 0.5
    for i in 1:6
        cell_max = cell_min + 0.5
        #nodes = gauss_lobatto_nodes(cell_min, cell_max)
        nodes = gauss_lobatto_nodes_inward(cell_min, cell_max)
        append!(all_nodes, nodes)
        cell_min = cell_max
    end

    return all_nodes
end

# Compute nodes for all cells
all_nodes = compute_all_nodes()

using Plots

# TODO: Compute x values of the DG nodes - what to do with the values at interfaces? Average?

x_limits = [-1.5, 1.5]

pd = PlotData1D(sol)

scatter(all_nodes, sol.u[end],
        title = "\$u (t_f ) \$", label = "P-ERK",
        xlims = x_limits, color = RGB(246 / 256, 169 / 256, 0),
        guidefont = font("Computer Modern", 16), tickfont = font("Computer Modern", 14),
        legendfont = font("Computer Modern", 12), legendfontsize = 12,
        legend = true) # For standard method
        
plot!(getmesh(pd),
      xlims = x_limits, label = "")

scatter!(all_nodes, sol.u[end],
        title = "\$u (t_f ) \$", label = "P-ERRK",
        xlims = x_limits, color = RGB(0, 84 / 256, 159 / 256),
        legend = true) # For relaxed method

minimum(sol.u[end])


#=
plot(sol, 
    title = "\$u (t_f ) \$", label = "P-ERRK",
    linewidth = 3, color = RGB(0, 84 / 256, 159 / 256),
    xticks = [-4, -2, -1, 0, 1, 2, 4],
    guidefont = font("Computer Modern", 16), tickfont = font("Computer Modern", 14),
    legend = true)

plot!(sol, label = "P-ERK",
    linewidth = 3, color = RGB(246 / 256, 169 / 256, 0),
    legend = true)
=#


#=
plot(sol, 
    title = "\$u (t_f ) \$", label = "P-ERRK",
    linewidth = 3, color = RGB(0, 84 / 256, 159 / 256),
    xticks = [-4, -2, -1, 0, 1, 2, 4],
    guidefont = font("Computer Modern", 16), tickfont = font("Computer Modern", 14),
    legend = true)

plot!(sol, label = "P-ERK",
    linewidth = 3, color = RGB(246 / 256, 169 / 256, 0),
    legend = true)
=#

minimum(sol.u[end])