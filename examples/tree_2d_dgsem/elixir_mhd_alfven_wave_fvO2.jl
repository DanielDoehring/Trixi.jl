using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
equations = IdealGlmMhdEquations2D(5 / 3)

initial_condition = initial_condition_convergence_test

polydeg = 5 # governs in this case only the number of FV subcells per DG cell
basis = LobattoLegendreBasis(polydeg)
surface_flux = (flux_hlle, flux_nonconservative_powell)
volume_integral = VolumeIntegralPureLGLFiniteVolumeO2(basis,
                                                      volume_flux_fv = surface_flux,
                                                      reconstruction_mode = reconstruction_O2_full,
                                                      slope_limiter = monotonized_central)
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (sqrt(2.0), sqrt(2.0))
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = 1.2
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        #analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, ParsaniKetchesonDeconinck3S82();
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
