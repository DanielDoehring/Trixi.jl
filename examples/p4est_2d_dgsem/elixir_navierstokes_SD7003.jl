using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

U_inf = 0.2
c_inf = 1.0

rho_inf = 1.4 # with gamma = 1.4 => p_inf = 1.0

Re = 10000.0
airfoil_cord_length = 1.0

t_c = airfoil_cord_length / U_inf

aoa = 4 * pi/180
u_x = U_inf * cos(aoa)
u_y = U_inf * sin(aoa)

gamma = 1.4
prandtl_number() = 0.72
mu() = rho_inf * U_inf * airfoil_cord_length / Re

equations = CompressibleEulerEquations2D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

@inline function initial_condition_mach02_flow(x, t, equations)
  # set the freestream flow parameters
  rho_freestream = 1.4

  v1 = 0.19951281005196486 # 0.2 * cos(aoa)
  v2 = 0.01395129474882506 # 0.2 * sin(aoa)
  
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

surf_flux = flux_hllc
vol_flux = flux_chandrashekar
solver = DGSEM(polydeg = polydeg, surface_flux = surf_flux,
               volume_integral = VolumeIntegralFluxDifferencing(vol_flux))


###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

path = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/PERK_mesh/SD7003Laminar/"
mesh_file = path * "sd7003_laminar_straight_sided_Trixi.inp"

boundary_symbols = [:Airfoil, :FarField]

mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols,
                    initial_refinement_level = 0)

# t = 30 t_c
restart_file = "restart_PERKMulti.h5"
restart_file = "restart_PERKSingle.h5"
#restart_file = "restart_NDBLSRK144.h5"
#restart_file = "restart_TD.h5" 
#restart_file = "restart_CKL.h5"
#restart_file = "restart_RK4.h5"
restart_filename = joinpath("out", restart_file)


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


tspan = (0.0, 30 * t_c) # Try to get into a state where initial pressure wave is gone
ode = semidiscretize(semi, tspan; split_form = false) # for PERK
#ode = semidiscretize(semi, tspan) # for OrdinaryDiffEq integrators

# Timespan for measurements over 5 * t_c
#tspan = (load_time(restart_filename), 35 * t_c)
#ode = semidiscretize(semi, tspan, restart_filename; split_form = false)
#ode = semidiscretize(semi, tspan, restart_filename)

summary_callback = SummaryCallback()

analysis_interval = 1_000_000

# Choose analysis interval such that roughly every dt_c = 0.005 a record is taken
analysis_interval = 25 # PERK4_Multi, PERKSingle

#analysis_interval = 20 # NDBLSRK144
#analysis_interval = 33 # DGLDDRK84_C
#analysis_interval = 48 # CKLLSRK95_4S
#analysis_interval = 88 # RK4

f_aoa() = aoa
f_rho_inf() = rho_inf
f_U_inf() = U_inf
f_linf() = airfoil_cord_length

drag_coefficient = AnalysisSurfaceIntegral(semi, :Airfoil,
                                           DragCoefficientPressure(f_aoa(), f_rho_inf(),
                                                                   f_U_inf(), f_linf()))

drag_coefficient_shear_force = AnalysisSurfaceIntegral(semi, :Airfoil,
                                                       DragCoefficientShearStress(f_aoa(),
                                                                                  f_rho_inf(),
                                                                                  f_U_inf(),
                                                                                  f_linf()))

lift_coefficient = AnalysisSurfaceIntegral(semi, :Airfoil,
                                           LiftCoefficientPressure(f_aoa(), f_rho_inf(),
                                                                   f_U_inf(), f_linf()))

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     save_analysis = true,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (drag_coefficient,
                                                           drag_coefficient_shear_force,
                                                           lift_coefficient))

analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

# Split DGSEM HLLC + Flux Chandrashekar
stepsize_callback = StepsizeCallback(cfl = 5.7) # PERK_4 Multi E = 5, ..., 16
stepsize_callback = StepsizeCallback(cfl = 6.2) # PERK_4 Multi E = 5, ..., 14

#stepsize_callback = StepsizeCallback(cfl = 6.2) # PERK_4 Single, 16
#stepsize_callback = StepsizeCallback(cfl = 6.4) # PERK_4 Single, 14
#stepsize_callback = StepsizeCallback(cfl = 6.5) # PERK_4 Single, 12

#stepsize_callback = StepsizeCallback(cfl = 7.7) # NDBLSRK144
#stepsize_callback = StepsizeCallback(cfl = 5.5) # DGLDDRK84_C
#stepsize_callback = StepsizeCallback(cfl = 3.2) # CKLLSRK95_4S
#stepsize_callback = StepsizeCallback(cfl = 1.9) # RK4

# For plots etc
save_solution = SaveSolutionCallback(interval = 2000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory="out")

alive_callback = AliveCallback(alive_interval = 200)

save_restart = SaveRestartCallback(interval = analysis_interval, # Only at end
                                   save_final_restart = true)

callbacks = CallbackSet(analysis_callback,
                        stepsize_callback, # For measurements: Fixed timestep
                        alive_callback, # Not needed for measurement run
                        save_solution,
                        #save_restart, # For restart with measurements
                        summary_callback);

###############################################################################
# run the simulation

# For reference
dtRatios = [0.252900854746017, # 16
            0.208310160790890, # 14
            0.172356930215766, # 12
            0.129859071602721, # 10
            0.092778774946394, #  8
            0.069255720146485, #  7
            0.049637258180915, #  6
            0.030629777558366] #= 5 =# / 0.252900854746017
Stages = [16, 14, 12, 10, 8, 7, 6, 5]

dtRatios = [0.208310160790890, # 14
            0.172356930215766, # 12
            0.129859071602721, # 10
            0.092778774946394, #  8
            0.069255720146485, #  7
            0.049637258180915, #  6
            0.030629777558366] #= 5 =# / 0.208310160790890
Stages = [14, 12, 10, 8, 7, 6, 5]

# Plot level distribution for cells
#=
cache = semi.cache
include("/home/daniel/git/Trixi2Vtk.jl/src/Trixi2Vtk.jl")
Trixi2Vtk.trixi2vtk(cache, dtRatios, Stages, joinpath("/home/daniel/git/Paper_PERK4/Data/SD7003/Plot_Level_Distribution/", "solution_*.h5"), output_directory="/home/daniel/git/Paper_PERK4/Data/SD7003/Plot_Level_Distribution/")
=#

ode_algorithm = PERK4_Multi(Stages, "/home/daniel/git/MA/EigenspectraGeneration/PERK4/SD7003/", dtRatios)

#ode_algorithm = PERK4(12, "/home/daniel/git/MA/EigenspectraGeneration/PERK4/SD7003/")

dt = 1e-3 # PERK4, dt_c = 2e-4

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary


#ode_algorithm = NDBLSRK144(williamson_condition = false, thread = OrdinaryDiffEq.True())
#ode_algorithm = DGLDDRK84_C(williamson_condition = false, thread = OrdinaryDiffEq.True())
#ode_algorithm = CKLLSRK95_4S(thread = OrdinaryDiffEq.True())
ode_algorithm = RK4(thread = OrdinaryDiffEq.True())

dt = 1.2e-3 # NDBLSRK144
dt = 7.4e-4 # DGLDDRK84_C
dt = 5.1e-4 # CKLLSRK95_4S
dt = 3e-4 # RK4

sol = solve(ode, ode_algorithm,
            dt = dt, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks, 
            adaptive = false);

summary_callback() # print the timer summary