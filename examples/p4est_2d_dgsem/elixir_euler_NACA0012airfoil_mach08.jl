using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

gamma() = 1.4
equations = CompressibleEulerEquations2D(gamma())

p_inf() = 1.0
rho_inf() = gamma() # Gives unit speed of sound c_inf = 1.0
mach_inf() = 0.8
aoa() = deg2rad(1.25) # 1.25 Degree angle of attack

@inline function initial_condition_mach08_flow(x, t,
                                               equations::CompressibleEulerEquations2D)
    v1 = 0.7998096216639273   # 0.8 * cos(aoa())
    v2 = 0.017451908027648896 # 0.8 * sin(aoa())

    prim = SVector(1.4, v1, v2, 1.0)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_mach08_flow

surface_flux = flux_lax_friedrichs
volume_flux = flux_chandrashekar

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

mesh_file = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/C1.2_quadgrids/naca_ref2_quadr_relabel.inp"

boundary_symbols = [:Airfoil, :Inflow, :Outflow]
mesh = P4estMesh{2}(mesh_file, boundary_symbols = boundary_symbols)

bc_farfield = BoundaryConditionDirichlet(initial_condition)

boundary_conditions = Dict(:Inflow => bc_farfield,
                           :Outflow => bc_farfield,
                           :Airfoil => boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers

# Run for a long time to reach a steady state
tspan = (0.0, 100.0) # 100 suffices for this mesh for stationary shock position
ode = semidiscretize(semi, tspan)


# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 200_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = ())

alive_callback = AliveCallback(alive_interval = 1000)

save_sol_interval = 50_000
save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)
save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true)

t_ramp_up() = 1.5

# P-ER(R)K 4 Multi
cfl_0() = 2.0
cfl_max() = 4.8

# P-ER(R)K 4 Standalone
#cfl_0() = 2.0
#cfl_max() = 8.1

# R-RK44
#cfl_0() = 0.9
#cfl_max() = 1.1

# R-TS64
#cfl_0() = 1.0
#cfl_max() = 1.5

# R-RKCKL54
#cfl_0() = 1.5
#cfl_max() = 1.9

cfl(t) = min(cfl_max(), cfl_0() + t/t_ramp_up() * (cfl_max() - cfl_0()))

stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        #analysis_callback,
                        save_solution,
                        #save_restart,
                        stepsize_callback)

###############################################################################
# run the simulation

dtRatios_complete_p4 = [ 
    0.653209035337363,
    0.530079549682015,
    0.398295542137155,
    0.326444525366249,
    0.282355465161903,
    0.229828402151329,
    0.163023514708386,
    0.085186504038755] ./ 0.653209035337363
Stages_complete_p4 = [14, 12, 10, 9, 8, 7, 6, 5]

dtRatios_p4 = [ 
    0.653209035337363,
    0.530079549682015,
    0.398295542137155,
    0.282355465161903,
    0.229828402151329,
    0.163023514708386,
    0.085186504038755] ./ 0.653209035337363
Stages_p4 = [14, 12, 10, 8, 7, 6, 5]

path = "/home/daniel/git/MA/EigenspectraGeneration/PERK4/NACA0012_Mach08/rusanov_chandrashekar/"

# TODO: Re-run with new relaxation solver or set root_tol` low enough
relaxation_solver = Trixi.RelaxationSolverBisection(max_iterations = 10, gamma_min = 0.8)

#ode_alg = Trixi.PairedExplicitRK4Multi(Stages_p4, path, dtRatios_p4)
ode_alg = Trixi.PairedExplicitRelaxationRK4Multi(Stages_p4, path, dtRatios_p4; relaxation_solver = relaxation_solver)

#ode_alg = Trixi.PairedExplicitRelaxationRK4(Stages_p4[1], path; relaxation_solver = relaxation_solver)

#ode_alg = Trixi.RelaxationRK44(; relaxation_solver = relaxation_solver)
#ode_alg = Trixi.RelaxationTS64(; relaxation_solver = relaxation_solver)
#ode_alg = Trixi.RelaxationCKL54(; relaxation_solver = relaxation_solver)

sol = Trixi.solve(ode, ode_alg, dt = 42.0, 
                  save_everystep = false, callback = callbacks);