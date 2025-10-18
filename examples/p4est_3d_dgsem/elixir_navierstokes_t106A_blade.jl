using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

function initial_condition_const(x, t, equations::CompressibleEulerEquations3D)
    RealT = eltype(x)
    rho = 1
    rho_v1 = 1
    rho_v2 = 1
    rho_v3 = -1
    rho_e = 10
    return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e)
end

initial_condition = initial_condition_const

polydeg = 2
solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs)


case_path = "/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/CS2/T106A_HOW4/"

mesh_file = case_path * "t106A_3D_coarse_fixed2.inp"
mesh_file = case_path * "t106A_3D_baseline_fixed2.inp"

#  1: Back (z-) x-y plane
# 13: Front (z+) x-y plane

#  2: Blade

## From here, the faces are numbered counter-clockwise ##

#  3: Top-left right-tilted;     x-z plane
#  4: Left vertical;             y-z plane
#  5: Bottom-left right-tilted;  x-z plane
#  6: Bottom, horizontal;        x-z plane
#  7: Bottom-right, left-tilted; x-z plane
#  8: Bottom, horizontal;        x-z plane
#  9: Right vertical;            y-z plane
# 10: Top, horizontal;           x-z plane
# 11: Top-right left-tilted;     x-z plane
# 12: Top, horizontal;           x-z plane

boundary_symbols = [:SurfSet1, :SurfSet2, :SurfSet3, :SurfSet4,
                    :PhysicalSurface5, :PhysicalSurface6, :PhysicalSurface7, :PhysicalSurface8, 
                    :PhysicalSurface9, :PhysicalSurface10, :PhysicalSurface11, :PhysicalSurface12,
                    :PhysicalSurface13]

mesh = P4estMesh{3}(mesh_file; boundary_symbols = boundary_symbols)

# TODO: Bundle nodesets with same boundary condition for efficiency

boundary_conditions = Dict(:SurfSet1 => BoundaryConditionDirichlet(initial_condition),

                           :SurfSet2 => BoundaryConditionDirichlet(initial_condition),

                           :SurfSet3 => BoundaryConditionDirichlet(initial_condition),
                           :SurfSet4 => BoundaryConditionDirichlet(initial_condition),
                           :PhysicalSurface5 => BoundaryConditionDirichlet(initial_condition),
                           :PhysicalSurface6 => BoundaryConditionDirichlet(initial_condition),
                           :PhysicalSurface7 => BoundaryConditionDirichlet(initial_condition),
                           :PhysicalSurface8 => BoundaryConditionDirichlet(initial_condition),
                           :PhysicalSurface9 => BoundaryConditionDirichlet(initial_condition),
                           :PhysicalSurface10 => BoundaryConditionDirichlet(initial_condition),
                           :PhysicalSurface11 => BoundaryConditionDirichlet(initial_condition),
                           :PhysicalSurface12 => BoundaryConditionDirichlet(initial_condition),
                           :PhysicalSurface13 => BoundaryConditionDirichlet(initial_condition)
                           )

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1e-3)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.5)

save_solution = SaveSolutionCallback(interval = 10_000,
                                     save_initial_solution = false)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);

