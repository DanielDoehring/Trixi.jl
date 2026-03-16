using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

initial_condition = initial_condition_convergence_test

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; inside = boundary_condition,
                       outside = boundary_condition)

trees_per_face_dim = 10 # number of trees in the first two local dimensions of each face
sphere_layers = 10 # number of trees in the third local dimension of each face
inner_radius = 0.5 # inner radius of the sphere
thickness = 9.5 # thickness of the spherical shell, outer radius is `inner_radius + thickness`
mesh = P4estMeshCubedSphere(trees_per_face_dim, sphere_layers,
                            inner_radius, thickness,
                            polydeg = 3)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 100)

cfl = 2.8 # Used for CarpenterKennedy2N54
#cfl = 4.5 # PERK 2
stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback, analysis_callback, #save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

Stages_complete_p2 = reverse(collect(range(2, 14)))

dtRatios_complete_p2 = reverse([0.0714200735092163085938
0.0997751951217651367188
0.15370845794677734375
0.211783647537231445312
0.265566706657409667969
0.320929288864135742188
0.394448637962341308594
0.451244115829467773438
0.542097687721252441406
0.618003010749816894531
0.705095529556274414062
0.7786655426025390625
0.878946185111999511719] ./ 0.878946185111999511719)

Stages_p2 = [14, 13, 12, 5, 4, 3, 2]

dtRatios_p2 = reverse([0.0714200735092163085938
0.0997751951217651367188
0.15370845794677734375
0.618003010749816894531
0.705095529556274414062
0.7786655426025390625
0.878946185111999511719] ./ 0.878946185111999511719)

ode_alg = Trixi.PairedExplicitRK2Multi(Stages_p2, "/home/daniel/git/MA/EigenspectraGeneration/AdvectionCubedSphere/", dtRatios_p2)

#=
sol = Trixi.solve(ode, ode_alg, dt = 42.0,
                  save_everystep = false, save_start = false,
                  callback = callbacks);
=#


# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);

