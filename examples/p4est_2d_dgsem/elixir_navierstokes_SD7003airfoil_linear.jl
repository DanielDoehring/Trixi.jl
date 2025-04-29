using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

U_inf = 0.2
c_inf = 1.0

rho_inf = 1.4 # with gamma = 1.4 => p_inf = 1.0

Re = 10000.0
airfoil_cord_length = 1.0

t_c = airfoil_cord_length / U_inf

aoa = 4 * pi / 180
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

path = "/storage/home/daniel/PERK4/SD7003/"
#path = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/PERK_mesh/SD7003Laminar/"
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

# For testing of MPI+p4est+PERK in 2D (hyperbolic only)
#=
semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions)
=#

###############################################################################
# ODE solvers, callbacks etc.

#tspan = (0.0, 30 * t_c) # Try to get into a state where initial pressure wave is gone
tspan = (0.0, 0.5) # For testing of MPI+p4est+PERK in 2D (hyperbolic only)

#ode = semidiscretize(semi, tspan)
ode = semidiscretize(semi, tspan; split_problem = false) # for multirate PERK

#=
# For PERK Multi coefficient measurements
restart_file = "restart_000126960.h5"
restart_filename = joinpath("out", restart_file)

tspan = (30 * t_c, 35 * t_c)
ode = semidiscretize(semi, tspan, restart_filename; split_problem = false)
=#

summary_callback = SummaryCallback()

# Choose analysis interval such that roughly every dt_c = 0.005 a record is taken
analysis_interval = 25 # PERK4_Multi, PERKSingle
#analysis_interval = 10_000 # For testing of MPI+p4est+PERK in 2D (hyperbolic only)

f_aoa() = aoa
f_rho_inf() = rho_inf
f_U_inf() = U_inf
f_linf() = airfoil_cord_length

drag_coefficient = AnalysisSurfaceIntegral((:Airfoil,),
                                           DragCoefficientPressure2D(f_aoa(), f_rho_inf(),
                                                                   f_U_inf(), f_linf()))

drag_coefficient_shear_force = AnalysisSurfaceIntegral((:Airfoil,),
                                                       DragCoefficientShearStress2D(f_aoa(),
                                                                                  f_rho_inf(),
                                                                                  f_U_inf(),
                                                                                  f_linf()))

lift_coefficient = AnalysisSurfaceIntegral((:Airfoil,),
                                           LiftCoefficientPressure2D(f_aoa(), f_rho_inf(),
                                                                   f_U_inf(), f_linf()))

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     save_analysis = true,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (drag_coefficient,
                                                           drag_coefficient_shear_force,
                                                           lift_coefficient))

cfl = 6.2 # PERK 4 Multi E = 5, ..., 14
#cfl = 6.5 # PERK 4 Single 12

#cfl = 7.4 # PEERRK_4 Multi E = 5, ..., 14
#cfl = 7.6 # Single PERK 14

#cfl = 1.9 # R-RK44
#cfl = 2.5 # R-TS64
#cfl = 2.6 # R-CKL54

stepsize_callback = StepsizeCallback(cfl = cfl)

# For plots etc
save_solution = SaveSolutionCallback(interval = 1_000_000, # Only at end
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out")

alive_callback = AliveCallback(alive_interval = 200)

save_restart = SaveRestartCallback(interval = 1_000_000, # Only at end
                                   save_final_restart = true)

callbacks = CallbackSet(stepsize_callback, # For measurements: Fixed timestep (do not use this)
                        alive_callback, # Not needed for measurement run
                        save_solution, # For plotting during measurement run
                        #save_restart, # For restart with measurements
                        analysis_callback,
                        summary_callback);

###############################################################################
# run the simulation

dtRatios = [0.208310160790890, # 14
    0.172356930215766, # 12
    0.129859071602721, # 10
    0.092778774946394, #  8
    0.069255720146485, #  7
    0.049637258180915, #  6
    0.030629777558366] / 0.208310160790890 #= 5 =#
Stages = [14, 12, 10, 8, 7, 6, 5]


relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 3)

#ode_algorithm = Trixi.PairedExplicitRelaxationRK4(Stages[1], path; relaxation_solver = relaxation_solver)

#ode_algorithm = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path, dtRatios; relaxation_solver = relaxation_solver)
ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages, path, dtRatios)

#ode_algorithm = Trixi.RelaxationRK44(; relaxation_solver = relaxation_solver)
#ode_algorithm = Trixi.RelaxationTS64(; relaxation_solver = relaxation_solver)
#ode_algorithm = Trixi.RelaxationCKL54(; relaxation_solver = relaxation_solver)

# For measurement run
dt = 1e-3 # PERK4, dt_c = 2e-4

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep = false, callback = callbacks);
