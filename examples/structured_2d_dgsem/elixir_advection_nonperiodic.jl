using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_gauss

# you can either use a single function to impose the BCs weakly in all
# 2*ndims == 4 directions or you can pass a tuple containing BCs for each direction
boundary_conditions = BoundaryConditionDirichlet(initial_condition)
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-5.0, -5.0)
coordinates_max = (5.0, 5.0)
mesh = StructuredMesh((16, 16), coordinates_min, coordinates_max, periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.6)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback);
###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
