using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

gamma = 1.4
prandtl_number() = 0.72

# Atmospheric Conditions
rho_inf = 1.225 # at 15 °C
p_inf = 101325 # Pa

# 360 km/h = 360 km/h / 3.6 = 100 m/s
u_inf = 100.0 # m/s
c_inf = sqrt(gamma * p_inf/rho_inf)

Ma = u_inf / c_inf

mu = 1.81e-5 # 15 °C 

# Characteristic length scale: Lower Airfoil chord length
l = 0.9/2.3 * 0.5

Re = rho_inf * u_inf * l / mu

aoa = 0.0
u_x = u_inf

equations = CompressibleEulerEquations2D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu,
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

# TODO: Angle of attack (modify inflow)                                                          
@inline function initial_condition_360kmh(x, t, equations)
  # set the freestream flow parameters
  rho_freestream = 1.225
  v1 = 100
  v2 = 0.0
  p_freestream = 101325

  prim = SVector(rho_freestream, v1, v2, p_freestream)
  return prim2cons(prim, equations)
end

initial_condition = initial_condition_360kmh

# Boundary conditions for free-stream testing
boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_airfoil = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)


polydeg = 3
surface_flux = flux_hlle
volume_flux = flux_ranocha
# Flux-Differencing required for this example
solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

path = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/PERRINN_2017_RearWing_2D/"
mesh_file = path * "PERRINN_RearWing_2D.inp"

boundary_symbols = [:PhysicalLine1, :PhysicalLine2, :PhysicalLine3, :PhysicalLine4]
mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:PhysicalLine1 => boundary_condition_free_stream,
                           :PhysicalLine2 => boundary_condition_free_stream,
                           :PhysicalLine3 => boundary_condition_free_stream,
                           :PhysicalLine4 => boundary_condition_slip_wall)

boundary_conditions_parabolic = Dict(:PhysicalLine1 => boundary_condition_free_stream,
                                     :PhysicalLine2 => boundary_condition_free_stream,
                                     :PhysicalLine3 => boundary_condition_free_stream,
                                     :PhysicalLine4 => boundary_condition_airfoil)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

t_c = l / u_inf
#tspan = (0.0, 20 * t_c) # Try to get into a state where initial pressure wave is gone
tspan = (0.0, 5e-5) # Try to get into a state where initial pressure wave is gone

# Timespan for measurements over 10 * t_c
#tspan = (load_time(restart_filename), 30 * t_c)

#ode = semidiscretize(semi, tspan; split_form = false) # for PERK
ode = semidiscretize(semi, tspan) # for OrdinaryDiffEq integrators

#ode = semidiscretize(semi, tspan, restart_filename)
#ode = semidiscretize(semi, tspan, restart_filename; split_form = false)

summary_callback = SummaryCallback()

analysis_interval = 100_000

f_aoa() = aoa
f_rho_inf() = rho_inf
f_u_inf() = u_inf
f_linf() = l

drag_coefficient = AnalysisSurfaceIntegral(semi, :PhysicalLine4,
                                           DragCoefficientPressure(f_aoa(), f_rho_inf(),
                                                                   f_u_inf(), f_linf()))

drag_coefficient_shear_force = AnalysisSurfaceIntegral(semi, :PhysicalLine4,
                                                       DragCoefficientShearStress(f_aoa(),
                                                                                  f_rho_inf(),
                                                                                  f_u_inf(),
                                                                                  f_linf()))

lift_coefficient = AnalysisSurfaceIntegral(semi, :PhysicalLine4,
                                           LiftCoefficientPressure(f_aoa(), f_rho_inf(),
                                                                   f_u_inf(), f_linf()))

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     save_analysis = true,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (drag_coefficient,
                                                           drag_coefficient_shear_force,
                                                           lift_coefficient))

stepsize_callback = StepsizeCallback(cfl = 1.5) # SSPRK104

save_solution = SaveSolutionCallback(interval = Int(1000),
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

alive_callback = AliveCallback(alive_interval = 100)

save_restart = SaveRestartCallback(interval = 10^6,
                                   save_final_restart = true)

callbacks = CallbackSet(#analysis_callback,
                        stepsize_callback, # Not for methods with error control
                        alive_callback,
                        save_solution,
                        #save_restart,
                        summary_callback)

###############################################################################
# run the simulation

ode_algorithm = SSPRK104(thread = OrdinaryDiffEq.True())

sol = solve(ode, ode_algorithm,
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary

using Plots

pd = PlotData2D(sol)
plot(sol)

plot(pd["v1"], xlim = [-0.5, 1], ylim = [-0.5, 0.5])
plot!(getmesh(pd), xlim = [-0.5, 1], ylim = [-0.5, 0.5])

plot(getmesh(pd), xlim = [-1, 2], ylim = [-1, 1])