using Trixi
using OrdinaryDiffEq

###############################################################################
# semidiscretization of the compressible Euler equations
gammas = (1.4, 1.4)  # Heat capacity ratios for two species
equations = Trixi.CompressibleEulerMultiIonEquations2D(gammas = gammas,
                                                       charge_to_mass = (0.4, 0.4)) # Since I haven't exported this yet

initial_condition = initial_condition_convergence_test

solver = DGSEM(polydeg = 4, surface_flux = flux_hll,
               volume_integral = VolumeIntegralPureLGLFiniteVolume(flux_hll))

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                               initial_refinement_level = 4,
                               n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:l2_error_primitive,
                                                              :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = false,
                                     save_final_solution = false,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
