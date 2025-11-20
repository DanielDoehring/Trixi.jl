
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

ref_lvl = 2 # 2, 3, 4, 5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = ref_lvl,
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

base_path = "/home/daniel/git/MA/EigenspectraGeneration/PERK4/EulerGravity/Coupled_Convergence/"
#base_path = "/storage/home/daniel/PERK4/EulerGravity/Coupled_Convergence/"

dtRatios = [1, 0.5, 0.25]
Stages_Gravity = [9, 6, 5]
alg_gravity = Trixi.PairedExplicitRK4Multi(Stages_Gravity, base_path * "HypDiff/", dtRatios)

cfl_gravity = 1.0

tolerance = 1e-5
if ref_lvl == 5
    tolerance = 1e-6
end

parameters = ParametersEulerGravity(background_density = 2.0, # aka rho0
                                    # rho0 is (ab)used to add a "+8π" term to the source terms
                                    # for the manufactured solution
                                    gravitational_constant = 1.0, # aka G
                                    cfl = cfl_gravity,
                                    resid_tol = tolerance,
                                    n_iterations_max = 1000,
                                    timestep_gravity = Trixi.timestep_gravity_PERK4_Multi!)

semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters, alg_gravity)

###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

cfl_euler = 0.5
stepsize_callback = StepsizeCallback(cfl = cfl_euler)

analysis_interval = 10_000
alive_callback = AliveCallback(alive_interval = 1)

analysis_callback = AnalysisCallback(semi_euler, interval = analysis_interval,
                                     save_analysis = true,
                                     analysis_errors = [
                                         :l2_error_primitive,
                                         :linf_error_primitive,
                                         :l1_error_primitive
                                     ],
                                     analysis_integrals = (;))

callbacks = CallbackSet(summary_callback, stepsize_callback,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation

Stages = [13, 8, 5]

ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages, base_path * "Euler/", dtRatios)

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep = false, callback = callbacks);

println("Number of gravity subcycles: ", semi.gravity_counter.ncalls_since_readout)
