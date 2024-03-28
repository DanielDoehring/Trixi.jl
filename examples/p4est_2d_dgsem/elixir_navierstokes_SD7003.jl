using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

U_inf = 0.2
c_inf = 1.0

rho_inf = 1.4 # with gamma = 1.4 => p_inf = 1.0

Re = 10000.0
airfoil_cord_length = 1.0

aoa = 4 * pi/180
u_x = U_inf * cos(aoa)
u_y = U_inf * sin(aoa)

gamma = 1.4
prandtl_number() = 0.71 
mu() = rho_inf * U_inf * airfoil_cord_length / Re

equations = CompressibleEulerEquations2D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

# TODO: Angle of attack (modify inflow)                                                          
@inline function initial_condition_mach02_flow(x, t, equations)
  # set the freestream flow parameters
  rho_freestream = 1.4
  v1 = 0.19951281005196486
  v2 = 0.01395129474882506
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

#=
volume_flux = flux_ranocha
solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
=#

solver = DGSEM(polydeg = polydeg, surface_flux = flux_hll)

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

path = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/PERK_mesh/SD7003Laminar/"
mesh_file = path * "sd7003_laminar_straight_sided_Trixi.inp"

boundary_symbols = [:Airfoil, :FarField]
mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

restart_file = "restart_118444.h5"
restart_file = "restart_000008.h5"

restart_filename = joinpath("out", restart_file)

mesh = load_mesh(restart_filename)

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
#tspan = (0.0, 100 * t_c)
tspan = (0.0, 20 * t_c) # Try to get into a state where initial pressure wave is gone

#tspan = (load_time(restart_filename), 100 * t_c)
#tspan = (load_time(restart_filename), load_time(restart_filename) + 1e-4)

ode = semidiscretize(semi, tspan; split_form = false)
#ode = semidiscretize(semi, tspan)

#ode = semidiscretize(semi, tspan, restart_filename)
#ode = semidiscretize(semi, tspan, restart_filename; split_form = false)

summary_callback = SummaryCallback()

analysis_interval = 100

sw_aoa() = aoa
sw_rho_inf() = rho_inf
sw_U_inf(equations) = U_inf
sw_linf() = airfoil_cord_length

drag_coefficient = Trixi.AnalysisSurfaceIntegral(semi, :Airfoil,
                                           Trixi.DragCoefficient(sw_aoa(), sw_rho_inf(),
                                                           sw_U_inf(equations), sw_linf()))

lift_coefficient = Trixi.AnalysisSurfaceIntegral(semi, :Airfoil,
                                           Trixi.LiftCoefficient(sw_aoa(), sw_rho_inf(),
                                                           sw_U_inf(equations), sw_linf()))

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     save_analysis = true,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (drag_coefficient,
                                                           lift_coefficient))

stepsize_callback = StepsizeCallback(cfl = 5.3) # PERK_4 Multi E = 5, ..., 16, non-refined
#stepsize_callback = StepsizeCallback(cfl = 4.7) # PERK_4 Multi E = 5, ..., 16, refined

#stepsize_callback = StepsizeCallback(cfl = 0.1) # CarpenterKennedy2N54

save_solution = SaveSolutionCallback(interval = Int(100),
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

alive_callback = AliveCallback(alive_interval = 100)

save_restart = SaveRestartCallback(interval = 10^6,
                                   save_final_restart = true)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = Trixi.density),
                                      base_level = 1,
                                      max_level = 2, max_threshold = 10)

amr_callback = AMRCallback(semi, amr_controller,
                            interval = 5,
                            adapt_initial_condition = false,
                            adapt_initial_condition_only_refine = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        #amr_callback,
                        #save_solution,
                        #save_restart,
                        stepsize_callback)

###############################################################################
# run the simulation


dtRatios = [0.252900854746017, # 16
            0.234065997367224, # 15
            0.208310160790890, # 14
            0.172356930215766, # 12
            0.129859071602721, # 10
            0.092778774946394, #  8
            0.069255720146485, #  7
            0.049637258180915, #  6
            0.030629777558366] #= 5 =# / 0.252900854746017
#Stages = [16, 15, 14, 12, 10, 8, 7, 6, 5]

dtRatios = [0.252900854746017, # 16
            0.208310160790890, # 14
            0.172356930215766, # 12
            0.129859071602721, # 10
            0.092778774946394, #  8
            0.069255720146485, #  7
            0.049637258180915, #  6
            0.030629777558366] #= 5 =# / 0.252900854746017
Stages = [16, 14, 12, 10, 8, 7, 6, 5]

ode_algorithm = PERK4_Multi(Stages, "/home/daniel/git/MA/EigenspectraGeneration/PERK4/SD7003/", dtRatios)

#ode_algorithm = PERK4(14, "/home/daniel/git/MA/EigenspectraGeneration/SD7003/")

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