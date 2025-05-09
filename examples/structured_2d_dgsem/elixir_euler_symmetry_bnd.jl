using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

function initial_condition_freestream(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.4
    v1 = 1
    v2 = 1
    p = 1
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_freestream

solver = DGSEM(polydeg = 3, surface_flux = flux_hll)

# Define faces for a parallelogram that looks like this
#
#             (0,1) __________ (2, 1)
#                ⟋         ⟋
#             ⟋         ⟋
#          ⟋         ⟋
# (-2,-1) ‾‾‾‾‾‾‾‾‾‾ (0,-1)
mapping(xi, eta) = SVector(xi + eta, eta)

cells_per_dimension = (16, 16)

mesh = StructuredMesh(cells_per_dimension, mapping, periodicity = (false, false))

boundary_condition_dirichlet = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (x_neg = boundary_condition_symmetry_plane,
                       x_pos = boundary_condition_symmetry_plane,
                       y_neg = boundary_condition_dirichlet,
                       y_pos = boundary_condition_dirichlet)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
