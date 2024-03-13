using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

@inline function initial_condition_freestream(x, t, equations::CompressibleEulerEquations2D)
  # set the freestream flow parameters
  rho_freestream = 1.4
  v1 = 0.2
  v2 = 0.0
  p_freestream = 1.0

  prim = SVector(rho_freestream, v1, v2, p_freestream)
  return prim2cons(prim, equations)
end

initial_condition = initial_condition_freestream

polydeg = 2
solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs)

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

path = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/NASA_meshes/"
mesh_file = path * "NACA0012/n0012family_1_7_2D_unique.inp"
mesh_file = path * "NACA4412/NACA4412_1_2D_unique.inp"

boundary_symbols = [:b2_symmetry_y_strong,
                    :b4_farfield_riem, :b5_farfield_riem, :b7_farfield_riem, :b6_viscous_solid, :b8_to_stitch_a]

mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

# Boundary conditions for free-stream testing
boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

boundary_conditions = Dict(:b2_symmetry_y_strong => boundary_condition_free_stream,
                           :b4_farfield_riem => boundary_condition_free_stream,
                           :b5_farfield_riem => boundary_condition_free_stream,
                           :b7_farfield_riem => boundary_condition_free_stream,
                           :b6_viscous_solid => boundary_condition_slip_wall,
                           :b8_to_stitch_a => boundary_condition_free_stream)

semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.02)
ode = semidiscretize(semi, tspan) # ODE Integrators

summary_callback = SummaryCallback()

analysis_interval = 10^5
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(alive_interval = 200)

stepsize_callback = StepsizeCallback(cfl = 3.4) # CarpenterKennedy2N54

stepsize_callback = StepsizeCallback(cfl = 5.3) # PERK4

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation


dtRatios = [0.249748130716557,
            0.229743135233184,
            0.148737624222123,
            0.106506037112321,
            0.092058178144161,
            0.066218481684170,
            0.047412460769779,
            0.027795091314087] / 0.249748130716557

#Stages = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5]
Stages = [16, 15, 11, 9, 8, 7, 6, 5]
ode_algorithm = PERK4_Multi(Stages, "/home/daniel/git/MA/EigenspectraGeneration/SD7003/", dtRatios)

#ode_algorithm = PERK4(14, "/home/daniel/git/MA/EigenspectraGeneration/SD7003/")

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep=false, callback=callbacks);


sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false, thread = OrdinaryDiffEq.True()),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary


using Plots

pd = PlotData2D(sol)
plot(sol)

plot(pd["rho"])

plot(getmesh(pd), xlim = [-1, 2], ylim = [-0.5, 0.5])