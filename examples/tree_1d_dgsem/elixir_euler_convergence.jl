using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_convergence_test

# Setup similar to vortex testcase
solver = DGSEM(polydeg = 6, surface_flux = flux_hllc)

coordinates_min = 0.0
coordinates_max = 2.0

BaseLevel = 4
# Test PERK on non-uniform mesh
refinement_patches = ((type = "box", coordinates_min = (0.5,),
                       coordinates_max = (1.5,)),
                      (type = "box", coordinates_min = (0.75,),
                       coordinates_max = (1.25,)))

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = BaseLevel,
                refinement_patches = refinement_patches,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10^6
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:linf_error_primitive,
                                                              :conservation_error))

callbacks = CallbackSet(summary_callback,
                        analysis_callback)

###############################################################################
# run the simulation

dtRatios = [1, 0.5, 0.25]
basepath = "/home/daniel/git/Paper_PERRK/Data/IsentropicVortex/IsentropicVortex/k6/"
#basepath = "/storage/home/daniel/PERRK/Data/IsentropicVortex/IsentropicVortex/k6/"

# p = 2

Stages = [12, 6, 3]
path = basepath * "p2/"
ode_algorithm = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)

N_Convergence = 0 # up to 7 for p2, 3 for p3/p4
CFL_Convergence = 1.0 / (2^N_Convergence)

dt = 0.001 * CFL_Convergence # Timestep in asymptotic regime

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep = false, callback = callbacks);