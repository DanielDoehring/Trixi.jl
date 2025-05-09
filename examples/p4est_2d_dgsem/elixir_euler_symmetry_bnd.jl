using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

equations = CompressibleEulerEquations2D(1.4)

function initial_condition_freestream(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.4
    v1 = 0
    v2 = 1
    p = 1
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_freestream

boundary_condition_dirichlet = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(:x_neg => boundary_condition_dirichlet,
                           :x_pos => boundary_condition_symmetry_plane)

solver = DGSEM(polydeg = 3, surface_flux = flux_hllc)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)

trees_per_dimension = (8, 8)

mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 # Enforce periodicity in y-direction
                 periodicity = (false, true))

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.6)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
