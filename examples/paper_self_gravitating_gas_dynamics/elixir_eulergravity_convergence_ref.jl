
using OrdinaryDiffEq
using Trixi

initial_condition = initial_condition_eoc_test_coupled_euler_gravity

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 2.0
equations_euler = CompressibleEulerEquations2D(gamma)

polydeg = 3
solver_euler = DGSEM(polydeg, FluxHLL(min_max_speed_naive))

coordinates_min = (0.0, 0.0)
coordinates_max = (2.0, 2.0)


refinement_patches = ((type = "box", coordinates_min = (0.5, 0.5),
                       coordinates_max = (1.5, 1.5)),
                      (type = "box", coordinates_min = (0.75, 0.75),
                       coordinates_max = (1.25, 1.25)))

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                refinement_patches = refinement_patches,
                n_cells_max = 10_000)

semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler, initial_condition,
                                          solver_euler,
                                          source_terms = source_terms_eoc_test_coupled_euler_gravity)

###############################################################################
# semidiscretization of the hyperbolic diffusion equations
equations_gravity = HyperbolicDiffusionEquations2D()

solver_gravity = DGSEM(polydeg, flux_lax_friedrichs)

semi_gravity = SemidiscretizationHyperbolic(mesh, equations_gravity, initial_condition,
                                            solver_gravity,
                                            source_terms = source_terms_harmonic)

###############################################################################
# combining both semidiscretizations for Euler + self-gravity

#=
parameters = ParametersEulerGravity(background_density = 2.0, # aka rho0
                                    # rho0 is (ab)used to add a "+8π" term to the source terms
                                    # for the manufactured solution
                                    gravitational_constant = 1.0, # aka G
                                    cfl = 1.1,
                                    resid_tol = 1.0e-5, # 1.0e-10
                                    n_iterations_max = 1000, # 1000
                                    timestep_gravity = timestep_gravity_erk52_3Sstar!)

semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters)
=#

Stages_Gravity = [9, 7, 5]
dtRatios = [1, 0.5, 0.25]

alg_gravity = PERK4_Multi(Stages_Gravity, 
                          "/home/daniel/git/MA/EigenspectraGeneration/PERK4/EulerGravity/WeakBlastWave/HypDiff/", 
                          dtRatios)

parameters = ParametersEulerGravity(background_density = 2.0, # aka rho0
                                    # rho0 is (ab)used to add a "+8π" term to the source terms
                                    # for the manufactured solution
                                    gravitational_constant = 1.0, # aka G
                                    cfl = 1.1,
                                    resid_tol = 1.0e-5, # 1.0e-10
                                    n_iterations_max = 1000, # 1000
                                    timestep_gravity = timestep_gravity_PERK4_Multi!)

semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters, alg_gravity)

###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

stepsize_callback = StepsizeCallback(cfl = 2.2) # CarpenterKennedy2N54
stepsize_callback = StepsizeCallback(cfl = 3.0) # PERK8 NOTE: Needs maybe to reduced further for long convergence study

save_solution = SaveSolutionCallback(interval = 10,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval = analysis_interval)

analysis_callback = AnalysisCallback(semi_euler, interval = analysis_interval,
                                     save_analysis = true)

callbacks = CallbackSet(summary_callback, stepsize_callback,
                        #save_solution,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation

Stages = 8
#ode_algorithm = PERK4(Stages, "/home/daniel/git/MA/EigenspectraGeneration/PERK4/EulerGravity/WeakBlastWave/")

dtRatios = [1, 0.5, 0.25]
Stages = [8, 6, 5]

ode_algorithm = PERK4_Multi(Stages, "/home/daniel/git/MA/EigenspectraGeneration/PERK4/EulerGravity/WeakBlastWave/Euler/", dtRatios)

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep=false, callback=callbacks);

#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
=#

summary_callback() # print the timer summary
println("Number of gravity subcycles: ", semi.gravity_counter.ncalls_since_readout)
