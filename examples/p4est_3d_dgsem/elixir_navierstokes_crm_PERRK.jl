using Trixi

###############################################################################

gamma = 1.4
prandtl_number() = 0.72

# Follows problem C3.5 of the 2015 Third International Workshop on High-Order CFD Methods
# https://www1.grc.nasa.gov/research-and-engineering/hiocfd/

#Re = 5 * 10^6 # C3.5 testcase
Re = 5 * 10^6 / 0.0254

chord = 7.005 # m = 275.80 inches
#chord = 7.005 * 0.0254 # m = 7.005 inches (conversion from running case for mesh in inches)

c = 343.0 # m/s = 13504 inches/s
rho() = 1.293 # kg/m^3 = 2.1199e-5 kg/inches^3

p() = c^2 * rho() / gamma
M = 0.85
U() = M * c

convective_timescale = chord / U() # seconds

mu() = rho() * chord * U()/Re

equations = CompressibleEulerEquations3D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

@inline function initial_condition(x, t, equations)
    # set the freestream flow parameters
    rho_freestream = 1.293

    v1 = 291.55
    v2 = 0.0
    v3 = 0.0

    p_freestream = 108657.255

    prim = SVector(rho_freestream, v1, v2, v3, p_freestream)
    return prim2cons(prim, equations)
end

bc_farfield = BoundaryConditionDirichlet(initial_condition)

polydeg = 2
basis = LobattoLegendreBasis(polydeg)

shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = pressure)

surface_flux = flux_hll
volume_flux = flux_ranocha

volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)              

#mesh_file = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/C3.5_gridfiles/crm_q3_lin_relabel.inp"
base_path = "/storage/home/daniel/CRM/"
mesh_file = base_path * "crm_q3_lin_relabel_m.inp" # Meters

boundary_symbols = [:SYMMETRY,
                    :FARFIELD,
                    :WING, :FUSELAGE, :WING_UP, :WING_LO,
                    :OUTFLOW]

mesh = P4estMesh{3}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions_hyp = Dict(:SYMMETRY => boundary_condition_symmetry_plane, # Symmetry: bc_symmetry
                               :FARFIELD => bc_farfield, # Farfield: bc_farfield
                               :WING => boundary_condition_slip_wall, # Wing: bc_slip_wall
                               :FUSELAGE => boundary_condition_slip_wall, # Fuselage: bc_slip_wall
                               :WING_UP => boundary_condition_slip_wall, # Wing: bc_slip_wall
                               :WING_LO => boundary_condition_slip_wall, # Wing: bc_slip_wall
                               :OUTFLOW => bc_farfield)

velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
bc_body = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)

bc_symmetry_plane_para = BoundaryConditionNavierStokesWall(Slip(), heat_bc)

boundary_conditions_para = Dict(:SYMMETRY => bc_symmetry_plane_para, # Symmetry
                                :FARFIELD => bc_farfield, # Farfield: bc_farfield
                                :WING => bc_body, # Wing: bc_body
                                :FUSELAGE => bc_body, # Fuselage: bc_body
                                :WING_UP => bc_body, # Wing: bc_body
                                :WING_LO => bc_body, # Wing: bc_body
                                :OUTFLOW => bc_farfield)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions_hyp,
                                                                    boundary_conditions_para))

# Run until crashes
#tspan = (0.0, 1e-4)

# Use for timings this:
#tspan = (0.0, 5e-6)

#ode = semidiscretize(semi, tspan; split_problem = false) # PER(R)K Multi
#ode = semidiscretize(semi, tspan) # Everything else

## Second run: Confirm increased robustness of relaxed multirate method ##
restart_file = "restart_15e-6.h5"
restart_filename = joinpath("out", restart_file)

# Run until crashes
tspan = (load_time(restart_filename), 1e-4)

ode = semidiscretize(semi, tspan, restart_filename; split_problem = false) # PER(R)K Multi
#ode = semidiscretize(semi, tspan, restart_filename)


# Callbacks
###############################################################################

summary_callback = SummaryCallback()

analysis_interval = 100_000

force_boundary_names = (:WING, :FUSELAGE, :WING_UP, :WING_LO)

A = 191.8 # m^2 = 297360.0 inch^2
pressure_coefficient = AnalysisSurfacePointwise(force_boundary_names,
                                                SurfacePressureCoefficient(p(), rho(),
                                                                           U(), A))
aoa() = 0.0
lift_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           LiftCoefficientPressure3D(aoa(), rho(),
                                                                     U(), A))

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (lift_coefficient,)
                                     #analysis_integrals = ()
                                     )

alive_callback = AliveCallback(alive_interval = 100)

save_sol_interval = 15000
save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out")

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true,
                                   output_directory = "out")

cfl_0() = 0.8 # PERRK

# For Re = ~200M
cfl_max() = 1.3 # PERRK Multi
#cfl_max() = 2.1 # PERRK Standalone 15

t_ramp_up() = 1e-6

#cfl(t) = min(cfl_max(), cfl_0() + t/t_ramp_up() * (cfl_max() - cfl_0()))

#cfl = 0.5 # R-CKL43
#cfl = 0.4 # R-RK33

## Restarted simulations ##

cfl = 1.2 # PERRK Multi

stepsize_callback = StepsizeCallback(cfl = cfl, interval = 5)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        #save_solution,
                        #save_restart,
                        stepsize_callback
                        )

# Run the simulation
###############################################################################

dtRatios_complete_p3 = [ 
    0.00106435123831034,
    0.000983755702972412,
    0.000857676243782043,
    0.000776133546233177,
    0.000684534176141024,
    0.00062269344329834,
    0.000545023646652699,
    0.000463906383514404,
    0.000371748408675194,
    0.000311754931509495,
    0.000263250452280045,
    0.000177319368720055,
    0.000112414136528969
                      ] ./ 0.00106435123831034
Stages_complete_p3 = reverse(collect(range(3, 15)))

dtRatios_red_p3 = [ 
    0.00106435123831034,
    0.000776133546233177,
    0.000684534176141024,
    0.00062269344329834,
    0.000545023646652699,
    0.000463906383514404,
    0.000371748408675194,
    0.000263250452280045,
    0.000177319368720055,
    0.000112414136528969
                      ] ./ 0.00106435123831034
Stages_red_p3 = [15, 12, 11, 10, 9, 8, 7, 5, 4, 3]

#ode_alg = Trixi.PairedExplicitRK3Multi(Stages_red_p3, base_path * "k2/p3/", dtRatios_red_p3)

newton = Trixi.RelaxationSolverNewton(max_iterations = 5, root_tol = 1e-13, gamma_tol = 1e-13)

ode_alg = Trixi.PairedExplicitRelaxationRK3Multi(Stages_red_p3, base_path * "k2/p3/", dtRatios_red_p3;
                                                 relaxation_solver = newton)

ode_alg = Trixi.PairedExplicitRelaxationRK3(15, base_path * "k2/p3/"; relaxation_solver = newton)

#ode_alg = Trixi.RelaxationCKL43(; relaxation_solver = newton)
#ode_alg = Trixi.RelaxationRK33(; relaxation_solver = newton)

sol = Trixi.solve(ode, ode_alg, dt = 42.0,
                  save_everystep = false, callback = callbacks);
