using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the (inviscid) Burgers' equation

equations = InviscidBurgersEquation1D()

basis = LobattoLegendreBasis(3)
# Use shock capturing techniques to suppress oscillations at discontinuities.
# Note that this introduces additional numerical dissipation,
# which causes the left and right states of the discontinuity to dampen out over time.
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 1.0,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = first)

volume_flux = flux_ec
surface_flux = flux_godunov

volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0,)
coordinates_max = (1.0,)
cells_per_dimension = (20,)

# Cluster cells around the shock location at x = 0.5
refinement_strength() = 0.8
mapping(xi) = 0.5 * (xi - refinement_strength() * sinpi(xi) / pi + 1)

mesh = StructuredMesh(cells_per_dimension, mapping, periodicity = true)

function initial_condition_sin(x, t, equation::InviscidBurgersEquation1D)
    return SVector(sinpi(2 * x[1]))
end

initial_condition = initial_condition_sin

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE problem & Callbacks

t_shock() = 1 / (2 * pi)
tspan = (0.0, t_shock())

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode,
            CarpenterKennedy2N54(williamson_condition = false, thread = Trixi.Threaded());
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
