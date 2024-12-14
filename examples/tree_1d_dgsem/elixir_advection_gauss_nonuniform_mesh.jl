
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -5.0 # minimum coordinate
coordinates_max = 5.0 # maximum coordinate

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

ode = semidiscretize(semi, (0.0, 11.0))


summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 100)

stepsize_callback = StepsizeCallback(cfl = 4.0)

callbacks = CallbackSet(summary_callback, analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

path = "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/"

dtRatios = [1, 0.5]
Stages = [16, 8]

ode_alg = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)

#=
relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 8)
ode_alg = Trixi.PairedExplicitRelaxationRK2Multi(Stages, path, dtRatios; 
                                                 relaxation_solver = relaxation_solver)
=#

sol = Trixi.solve(ode, ode_alg,
                  dt = 42.0,
                  save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()
