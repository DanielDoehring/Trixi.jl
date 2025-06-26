using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
gamma = 5 / 3
equations = IdealGlmMhdEquations2D(gamma)

initial_condition = initial_condition_convergence_test

#k = 2 # p2
#k = 3 # p3
k = 4 # p4

volume_flux = (flux_central, flux_nonconservative_powell)
solver = DGSEM(polydeg = k,
               surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (0.0, 0.0)
coordinates_max = (sqrt(2.0), sqrt(2.0))

refinement_patches = ((type = "box", 
                       coordinates_min = (0.25 * sqrt(2), 0.25 * sqrt(2)),
                       coordinates_max = (0.75 * sqrt(2), 0.75 * sqrt(2))),
                      (type = "box", 
                       coordinates_min = (0.375 * sqrt(2), 0.375 * sqrt(2)),
                       coordinates_max = (0.625 * sqrt(2), 0.625 * sqrt(2))))

base_ref = 7 # Start from 3 up to 7

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = base_ref,
                refinement_patches = refinement_patches,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = false,
                                     analysis_errors = [:l2_error, :l1_error, :linf_error],
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

#cfl = 5.0 # p2
#cfl = 3.5 # p3
cfl = 2.7 # p4

stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

basepath = "/home/daniel/git/Paper_PERRK/Data/Alfven_Wave/"

dtRatios = [1, 0.5, 0.25]
relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 10, root_tol = 1e-15, gamma_tol = eps(Float64))

#=
Stages = [14, 7, 4]
path = basepath * "k2/p2/"
ode_algorithm = Trixi.PairedExplicitRelaxationRK2Multi(Stages, path, dtRatios; relaxation_solver = relaxation_solver)
=#

#=
Stages = [14, 8, 5]
path = basepath * "k3/p3/"
ode_algorithm = Trixi.PairedExplicitRelaxationRK3Multi(Stages, path, dtRatios; relaxation_solver = relaxation_solver)
=#

Stages = [14, 8, 6]
path = basepath * "k4/p4/"
ode_algorithm = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path, dtRatios; relaxation_solver = relaxation_solver)

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep = false, callback = callbacks);