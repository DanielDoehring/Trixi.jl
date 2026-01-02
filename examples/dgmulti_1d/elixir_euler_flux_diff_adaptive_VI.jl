using OrdinaryDiffEqLowStorageRK
using Trixi

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

volume_integral_weakform = VolumeIntegralWeakForm()
volume_integral_fluxdiff = VolumeIntegralFluxDifferencing(volume_flux)

# `threshold` governs the tolerated entropy increase due to the weak-form
# volume integral before switching to the stabilized version
indicator = IndicatorEntropyIncrease(threshold = 0)
# Adaptive volume integral using the entropy increase indicator to perform the 
# stabilized/EC volume integral when needed
volume_integral = VolumeIntegralAdaptive(volume_integral_default = volume_integral_weakform,
                                         volume_integral_stabilized = volume_integral_fluxdiff,
                                         indicator = indicator)

dg = DGMulti(polydeg = 3, element_type = Line(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = volume_integral)
equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

cells_per_dimension = (8,)
mesh = DGMultiMesh(dg, cells_per_dimension,
                   coordinates_min = (-1.0,), coordinates_max = (1.0,), periodicity = true)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    source_terms = source_terms)

tspan = (0.0, 1.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 0.5 * estimate_dt(mesh, dg),
            ode_default_options()...,
            callback = callbacks);
