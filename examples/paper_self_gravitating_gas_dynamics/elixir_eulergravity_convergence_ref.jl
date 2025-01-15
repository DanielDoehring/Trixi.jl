
using OrdinaryDiffEq
using Trixi

initial_condition = initial_condition_eoc_test_coupled_euler_gravity

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 2.0
equations_euler = CompressibleEulerEquations2D(gamma)

polydeg = 3
solver_euler = DGSEM(polydeg, flux_hll)

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

flux_gravity = flux_godunov
solver_gravity = DGSEM(polydeg, flux_gravity)

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

#base_path = "/home/daniel/git/MA/EigenspectraGeneration/PERK4/EulerGravity/Coupled_Convergence/"
base_path = "/storage/home/daniel/PERK4/EulerGravity/Coupled_Convergence/"

dtRatios = [1, 0.5, 0.25]

Stages_Gravity = [9, 6, 5]
#Stages_Gravity = [15, 9, 6]
cfl_gravity = 1.0

alg_gravity = Trixi.PairedExplicitRK4Multi(Stages_Gravity, base_path * "HypDiff/", dtRatios)


parameters = ParametersEulerGravity(background_density = 2.0, # aka rho0
                                    # rho0 is (ab)used to add a "+8π" term to the source terms
                                    # for the manufactured solution
                                    gravitational_constant = 1.0, # aka G
                                    cfl = cfl_gravity,
                                    resid_tol = 1.0e-5, # 1.0e-10
                                    n_iterations_max = 1000, # 1000

                                    #timestep_gravity = Trixi.timestep_gravity_PERK4!
                                    timestep_gravity = Trixi.timestep_gravity_PERK4_Multi!
                                    )

semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters, alg_gravity)

###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

#cfl_euler = 0.5
cfl_euler = 1.0

stepsize_callback = StepsizeCallback(cfl = cfl_euler) # PERK8 NOTE: Needs maybe to reduced further for long convergence study

analysis_interval = 10_000
alive_callback = AliveCallback(alive_interval = 10)

analysis_callback = AnalysisCallback(semi_euler, interval = analysis_interval,
                                     save_analysis = true)

callbacks = CallbackSet(summary_callback, stepsize_callback,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation

#=
Stages = 8
ode_algorithm = PERK4(Stages, "/home/daniel/git/MA/EigenspectraGeneration/PERK4/EulerGravity/WeakBlastWave/Euler/")
=#

Stages = [13, 8, 5]

ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages, base_path * "Euler/", dtRatios)

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
println("Number of gravity subcycles: ", semi.gravity_counter.ncalls_since_readout)
