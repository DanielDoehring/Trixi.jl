using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the polytropic Euler equations

gamma = 1   # With gamma = 1 the system is isothermal.
kappa = 1.0 # Scaling factor for the pressure.
equations = PolytropicEulerEquations2D(gamma, kappa)

# Linear pressure wave in the negative x-direction.
function initial_condition_wave(x, t, equations::PolytropicEulerEquations2D)
    rho = 1.0
    v1 = 0.0
    if x[1] > 0.0
        rho = ((1.0 + 0.01 * sin(x[1] * 2 * pi)) / equations.kappa)^(1 / equations.gamma)
        v1 = ((0.01 * sin((x[1] - 1 / 2) * 2 * pi)) / equations.kappa)
    end
    v2 = 0.0

    return prim2cons(SVector(rho, v1, v2), equations)
end
initial_condition = initial_condition_wave

volume_flux = flux_winters_etal
solver = DGSEM(polydeg = 2, surface_flux = flux_hll, RealT = Float16,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-Float16(2), -Float16(2))
coordinates_max = (Float16(2), Float16(2))
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                RealT = Float16, periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.7)

callbacks = CallbackSet(summary_callback,
                        #analysis_callback,
                        alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

ode_algorithm = Trixi.CarpenterKennedy2N54()

sol = Trixi.solve(ode, ode_algorithm;
                  dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
                  ode_default_options()..., callback = callbacks);
