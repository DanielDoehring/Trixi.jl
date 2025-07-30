using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0
coordinates_max = 1.0

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.


ode = semidiscretize(semi, (0.0, 1.0))

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 100)
callbacks = CallbackSet(summary_callback, analysis_callback)

###############################################################################
# run the simulation

ode_alg = Trixi.LobattoIIIA_p2()
sol = Trixi.solve(ode, ode_alg, dt = 0.2,
                  save_everystep = false, callback = callbacks);
