using OrdinaryDiffEqLowStorageRK
using Trixi

volume_integral_weakform = VolumeIntegralWeakForm() # Does not make it to the end of the simulation
volume_flux = flux_ranocha
volume_integral_fluxdiff = VolumeIntegralFluxDifferencing(volume_flux) # Does also not make it to the end and is much more expensive

# `threshold` governs the tolerated entropy increase due to the weak-form
# volume integral before switching to the stabilized version
indicator = IndicatorEntropyIncrease(threshold = 0)
# Adaptive volume integral using the entropy increase indicator to perform the 
# stabilized/EC volume integral when needed
volume_integral = VolumeIntegralAdaptive(volume_integral_default = volume_integral_weakform,
                                         volume_integral_stabilized = volume_integral_fluxdiff,
                                         indicator = indicator)

dg = DGMulti(polydeg = 3, element_type = Tri(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_hllc),
             volume_integral = volume_integral)

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_kelvin_helmholtz_instability(x, t,
                                                        equations::CompressibleEulerEquations2D)
    # change discontinuity to tanh
    # typical resolution 128^2, 256^2
    # domain size is [-1,+1]^2
    slope = 15
    amplitude = 0.02
    B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
    rho = 0.5 + 0.75 * B
    v1 = 0.5 * (B - 1)
    v2 = 0.1 * sin(2 * pi * x[1])
    p = 1.0
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability

cells_per_dimension = (32, 64)
mesh = DGMultiMesh(dg, cells_per_dimension; periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)

tspan = (0.0, 3.6)
#tspan = (0.0, 2.8) # For runtime comparison
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 50)

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     uEltype = real(dg),
                                     analysis_errors = Symbol[])
#=
analysis_interval = 1 # For entropy recording
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     uEltype = real(dg),
                                     analysis_integrals = (entropy,),
                                     analysis_errors = Symbol[],
                                     save_analysis = true)
=#
save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false, thread = Trixi.True());
            dt = estimate_dt(mesh, dg), ode_default_options()..., callback = callbacks);
