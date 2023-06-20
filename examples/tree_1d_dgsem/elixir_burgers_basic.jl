
using OrdinaryDiffEq, LinearAlgebra, Plots
using Trixi

###############################################################################
# semidiscretization of the (inviscid) Burgers' equation

equations = InviscidBurgersEquation1D()

initial_condition = initial_condition_convergence_test

solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = 0.0
coordinates_max = 1.0
InitialRefinement = 5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=InitialRefinement,
                n_cells_max=10_000)

LLID = Trixi.local_leaf_cells(mesh.tree)
num_leafs = length(LLID)

# Refine right 3 quarters of mesh
@assert num_leafs % 4 == 0
Trixi.refine!(mesh.tree, LLID[Int(round(4*num_leafs/5))+1 : num_leafs])

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.15)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_errors=(:l2_error_primitive,
                                                            :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback, stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
plot(sol)


# TODO: Try PERK 2,4 for comparison (no oscillations because of very small timestep?)
ode_algorithm = Trixi.PRK()

CFL = 0.8
# S = 3, p = 2
dt = 0.00104291357216425242 / (2.0^(InitialRefinement - 7)) * CFL

# S = 5, p = 2
dt = 0.00186384918051771823 / (2.0^(InitialRefinement - 7)) * CFL

# S = 8, p = 2
dt = 0.00312564895721152431 / (2.0^(InitialRefinement - 7)) * CFL

ode_algorithm = PERK_Multi(8, 1, 
                           "/home/daniel/git/MA/EigenspectraGeneration/BurgersShock/", 
                           1.0, 0.5, stage_callbacks = ())

sol = Trixi.solve(ode, ode_algorithm,
                  #dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  dt = dt,
                  save_everystep=false);

# Print the timer summary
summary_callback()

plot(sol)
pd = PlotData1D(sol)
plot!(getmesh(pd))
