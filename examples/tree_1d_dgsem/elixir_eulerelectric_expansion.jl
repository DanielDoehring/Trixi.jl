using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi
#### DRAFT!! ####

###############################################################################
# semidiscretization of the compressible Euler equations
equations_euler = Trixi.CompressibleEulerElectronIonsEquations1D(gammas = (5 / 3, 5 / 3),
                                                              epsilon = 1e-4)


initial_condition = Trixi.initial_condition_plasma

polydeg = 6

solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralPureLGLFiniteVolume(flux_lax_friedrichs))

coordinates_min = 0.0
coordinates_max = 1.0



boundary_conditions_euler = (;x_neg = Trixi.boundary_condition_injection,
                              x_pos = Trixi.boundary_condition_dummy)

mesh_euler = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 11,
                n_cells_max = 30_000, periodicity = false)

semi_euler = SemidiscretizationHyperbolic(mesh_euler, equations_euler, initial_condition, solver, 
                                          boundary_conditions = boundary_conditions_euler)

###############################################################################
# semidiscretization of the hyperbolic diffusion equations
equations_electric = HyperbolicDiffusionEquations1D()
solver_electric = DGSEM(polydeg, flux_lax_friedrichs)

boundary_condition_zero_dirichlet = BoundaryConditionDirichlet((x, t, equations) -> SVector(0.0, 0.0))

boundary_conditions_diffusion = (;
                                 x_neg = boundary_condition_zero_dirichlet,
                                 x_pos = BoundaryConditionDirichlet((x, t, equations) -> SVector(100.0, 0.0)))

mesh_electric = TreeMesh(coordinates_min, coordinates_max,
                        initial_refinement_level = 11,
                         n_cells_max = 30_000, periodicity = false)

semi_electric = SemidiscretizationHyperbolic(mesh_electric, equations_electric, initial_condition,
                                           solver_electric, source_terms = source_terms_harmonic,
                                           boundary_conditions = boundary_conditions_diffusion)
###############################################################################
# combining both semidiscretizations for Euler + Poisson equation for electric potential
parameters = Trixi.ParametersEulerElectric(scaled_debye_length = 1e-4,
                                         epsilon = 1e-4,
                                         cfl = 0.0000005,
                                         resid_tol = 1.0e-7,
                                         n_iterations_max = 10^4,
                                         timestep_electric = Trixi.timestep_electric_erk52_3Sstar!)

semi = Trixi.SemidiscretizationEulerElectric(semi_euler, semi_electric, parameters)

###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 0.09)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10

alive_callback = AliveCallback(analysis_interval = analysis_interval)
analysis_callback = AnalysisCallback(semi_euler, interval = analysis_interval)

analysis_callback = AnalysisCallback(semi,
                                     save_analysis = true,
                                     interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = false,
                                     save_final_solution = false,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.0000005)

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
