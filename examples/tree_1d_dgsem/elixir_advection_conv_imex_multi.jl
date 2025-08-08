using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

k = 3 # polynomial degree

# Diffusive fluxes
num_flux = flux_godunov
#num_flux = flux_lax_friedrichs

solver = DGSEM(polydeg = k, surface_flux = num_flux)

coordinates_min = -1.0
coordinates_max = 1.0

# One refinement only
refinement_patches = ((type = "box", coordinates_min = (-0.5,),
                       coordinates_max = (0.5,)),)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                refinement_patches = refinement_patches,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

t_end = 2.0

ode = semidiscretize(semi, (0.0, t_end))

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 100,
                                     extra_analysis_errors = (:conservation_error,))

callbacks = CallbackSet(summary_callback,
                        analysis_callback)

###############################################################################
# run the simulation

path = "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/"

dtRatios = [1, 0.5]
Stages = [16, 8]

ode_alg = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)

ode_alg = Trixi.PairedExplicitRK2IMEXMulti([8], path, [1])

dt = 0.05 / (2^3) # 0.05 for explicit 8-16 pair

sol = Trixi.solve(ode, ode_alg,
                  dt = dt,
                  save_everystep = false, callback = callbacks);
