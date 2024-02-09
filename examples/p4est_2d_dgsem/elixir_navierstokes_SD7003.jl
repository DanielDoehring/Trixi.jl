using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

U_inf = 0.2
c_inf = 1.0
rho_inf = 1.4
Re = 10000.0
airfoil_cord_length = 1.0

# TODO: Change angle of attack!
aoa = 0.0

gamma = 1.4
prandtl_number() = 0.72
mu() = rho_inf * U_inf * airfoil_cord_length / Re

equations = CompressibleEulerEquations2D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

# TODO: Angle of attack (modify inflow)                                                          
@inline function initial_condition_mach02_flow(x, t, equations)
  # set the freestream flow parameters
  rho_freestream = 1.4
  v1 = 0.2
  v2 = 0.0
  p_freestream = 1.0

  prim = SVector(rho_freestream, v1, v2, p_freestream)
  return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach02_flow

# Boundary conditions for free-stream testing
boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_airfoil = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)

polydeg = 3
volume_flux = flux_ranocha

solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

path = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/PERK_mesh/SD7003Laminar/"
mesh_file = path * "sd7003_laminar_straight_sided_Trixi.inp"

boundary_symbols = [:Airfoil, :FarField]
mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)


boundary_conditions = Dict(:FarField => boundary_condition_free_stream,
                           :Airfoil => boundary_condition_slip_wall)

boundary_conditions_parabolic = Dict(:FarField => boundary_condition_free_stream,
                                     :Airfoil => boundary_condition_airfoil)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                     initial_condition, solver;
                                     boundary_conditions = (boundary_conditions,
                                                            boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

t_c = airfoil_cord_length / U_inf
tspan = (0.0, 0.2 * t_c)

ode = semidiscretize(semi, tspan; split_form = false)
#ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000

semi.boundary_conditions.boundary_dictionary
indices = semi_ -> semi.boundary_conditions.boundary_indices[2]
# TODO: Not sure if airfoil_cord_length = l_inf is correct usage
my_drag_force = Trixi.AnalysisSurfaceIntegral(indices, Trixi.DragForcePressure(aoa, rho_inf, U_inf, airfoil_cord_length))

my_lift_force = Trixi.AnalysisSurfaceIntegral(indices, Trixi.LiftForcePressure(aoa, rho_inf, U_inf, airfoil_cord_length))


analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     analysis_errors = Symbol[], # Turn errors off
                                     output_directory = "out", save_analysis = true,
                                     analysis_integrals = (my_drag_force, my_lift_force))

#stepsize_callback = StepsizeCallback(cfl = 5.1) # PERK_4 Multi E = 5, ..., 14
stepsize_callback = StepsizeCallback(cfl = 5.4) # PERK_4 Multi E = 5, ..., 16
#stepsize_callback = StepsizeCallback(cfl = 2.1) # CarpenterKennedy2N54

#stepsize_callback = StepsizeCallback(cfl = 5.6) # PERK_4 Single 14

#stepsize_callback = StepsizeCallback(cfl = 4.4) # PERK_4 Single 8
stepsize_callback = StepsizeCallback(cfl = 5.3) # PERK_4 Single 20

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

alive_callback = AliveCallback(alive_interval = 100)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        #save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

#=
dtRatios = [0.191469431854785, 
            0.184602899151614,
            0.169882330226391,
            0.148737624222123,
            0.125358798070687,
            0.106506037112321,
            0.092058178144161,
            0.066218481684170,
            0.047412460769779,
            0.027795091314087] / 0.191469431854785
=#

dtRatios = [0.249748130716557,
            0.229743135233184,
            0.191469431854785, 
            0.184602899151614,
            0.169882330226391,
            0.148737624222123,
            0.125358798070687,
            0.106506037112321,
            0.092058178144161,
            0.066218481684170,
            0.047412460769779,
            0.027795091314087] / 0.249748130716557

#Stages = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5]
Stages = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]
#ode_algorithm = PERK4_Multi(Stages, "/home/daniel/git/MA/EigenspectraGeneration/SD7003/", dtRatios)

ode_algorithm = PERK4(20, "/home/daniel/git/MA/EigenspectraGeneration/SD7003/")

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep=false, callback=callbacks);


sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false, thread = OrdinaryDiffEq.True()),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

#=
sol = solve(ode, SSPRK104(; thread = OrdinaryDiffEq.True()),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
=#

summary_callback() # print the timer summary


using Plots

pd = PlotData2D(sol)
plot(sol)

plot(pd["v1"], xlim = [-1, 2], ylim = [-1, 1])
plot!(getmesh(pd))

plot(getmesh(pd), xlim = [-1, 2], ylim = [-1, 1])