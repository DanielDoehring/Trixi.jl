
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_convergence_test

solver = DGSEM(polydeg = 3, surface_flux = flux_hllc)

coordinates_min = 0.0
coordinates_max = 2.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:l2_error_primitive,
                                                              :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

ode_algorithm = Trixi.PairedExplicitRK2(6, tspan, semi)

cfl_number = Trixi.calculate_cfl(ode_algorithm, ode)
stepsize_callback = StepsizeCallback(cfl = 1.0 * cfl_number)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 1.0, # Manual time step value, will be overwritten by the stepsize_callback when it is specified.
                  save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()
