
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the hyperbolic diffusion equations

equations = HyperbolicDiffusionEquations2D()

initial_condition = initial_condition_poisson_nonperiodic
# 1 => -x, 2 => +x, 3 => -y, 4 => +y as usual for orientations
boundary_conditions = (x_neg = boundary_condition_poisson_nonperiodic,
                       x_pos = boundary_condition_poisson_nonperiodic,
                       y_neg = boundary_condition_periodic,
                       y_pos = boundary_condition_periodic)

polydeg = 3
surface_flux = flux_lax_friedrichs
solver = DGSEM(polydeg, surface_flux)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)

refinement_patches = ((type = "box", coordinates_min = (0.25, 0.25),
                       coordinates_max = (0.75, 0.75)),
                      (type = "box", coordinates_min = (0.375, 0.375),
                       coordinates_max = (0.625, 0.625)))

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                refinement_patches = refinement_patches,
                n_cells_max = 30_000,
                periodicity = (false, true))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_poisson_nonperiodic,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

resid_tol = 1.0e-14 # To prevent the solver from stopping too early
steady_state_callback = SteadyStateCallback(abstol = resid_tol, reltol = 0.0)

# E = 11
cfl = 3.5

# Multi, E = [11, 7, 5]
cfl = 3.3

stepsize_callback = StepsizeCallback(cfl = cfl)

analysis_interval = 5000
alive_callback = AliveCallback(analysis_interval = analysis_interval)
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy, energy_total))

callbacks = CallbackSet(summary_callback, steady_state_callback, stepsize_callback,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation

base_path = "/home/daniel/git/MA/EigenspectraGeneration/PERK4/EulerGravity/HypDiff_Convergence/"

Stages = 11
#ode_algorithm = PERK4(Stages, base_path)


dtRatios = [1, 0.5, 0.25]
Stages = [11, 7, 5]

ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages, base_path, dtRatios)


sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
