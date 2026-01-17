using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

# Dissipation ensures entropy stability
surface_flux = FluxPlusDissipation(flux_ranocha, DissipationMatrixWintersEtal())

c_ = c_SD(polydeg) # parameter that recovers Spectral Difference correction function
@assert c_ > c_min_ESFR(polydeg) # Required for energy stability
correction_function = correction_function_ESFR(c_)
surface_integral = SurfaceIntegralFluxReconstruction(basis,
                                                     surface_flux = surface_flux,
                                                     correction_function = correction_function)
solver = DGSEM(polydeg = polydeg,
               surface_integral = surface_integral,
               volume_integral = VolumeIntegralStrongForm())

cells_per_dimension = (8,)
# This mapping converts [-1, 1] to [0, 2] with a non-uniform distribution of cells
mapping(xi) = ((xi + 2)^2 - 1) / 4
mesh = StructuredMesh(cells_per_dimension, mapping)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver,
                                    source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 2.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, ode_default_options()..., callback = callbacks);
