using Trixi
using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using LinearAlgebra: norm

###############################################################################

gamma = 1.4
prandtl_number() = 0.72

Re = 5 * 10^6

## Standard units ##

chord = 7.005 # meters = 275.80 inches

c = 343 # m/s

#p = 101325 # Pa
#rho = c^2 / (gamma * p)

rho() = 1.293 # kg/m^3
p() = c^2 * rho() / gamma

U() = 0.85 * c

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
                                            alpha_smooth = true, # true
                                            variable = pressure) # density_pressure

surface_flux = flux_hll # flux_lax_friedrichs
volume_flux = flux_ranocha

# TODO: Do I need SC ? Or is FD sufficient?
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

mesh_file = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/C3.5_gridfiles/crm_q3_lin_relabel.inp"

boundary_symbols = [:FUSELAGE,
    :WING,
    :SYMMETRY,
    :FARFIELD,
    :WING_UP, :WING_LO,
    :OUTFLOW
]

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

symmetry_bc_para = SymmetryPlane()
bc_symmetry_plane = BoundaryConditionNavierStokesWall(symmetry_bc_para, heat_bc)

boundary_conditions_para = Dict(:SYMMETRY => bc_symmetry_plane, # Symmetry
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

tspan = (0.0, 1e-5)
#ode = semidiscretize(semi, tspan; split_problem = false) # PER(R)K Multi
ode = semidiscretize(semi, tspan) # Everything else

# Callbacks
###############################################################################

summary_callback = SummaryCallback()

analysis_interval = 20
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     #analysis_integrals = (lift_coefficient,),
                                     #analysis_integrals = ()
                                     )

alive_callback = AliveCallback(alive_interval = 1)

save_sol_interval = analysis_interval

save_solution = SaveSolutionCallback(interval = 100_000,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out")

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true,
                                   output_directory = "out")

# TODO: Likely, CFL ramp-up needed
#=
cfl_0() = 0.2
cfl_max() = 2.0
t_ramp_up() = 1e-6

cfl(t) = min(cfl_max(), cfl_0() + t/t_ramp_up() * (cfl_max() - cfl_0()))
=#
cfl = 0.2

stepsize_callback = StepsizeCallback(cfl = cfl)

amr_indicator = shock_indicator
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.05, # 1
                                      max_level = 2, max_threshold = 0.1)  # 2

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 10,
                           adapt_initial_condition = false)

callbacks = CallbackSet(summary_callback,
                        #alive_callback,
                        analysis_callback,
                        #amr_callback,
                        save_solution,
                        #save_restart,
                        stepsize_callback)

# Run the simulation
###############################################################################

# TODO: Optimize for HLL?
#base_path = "/storage/home/daniel/OneraM6/LLF_only/"
base_path = "/home/daniel/git/Paper_PERRK/Data/OneraM6/Spectra_OptimizedCoeffs/LLF_FD_Ranocha/k2/"

#ode_alg = Trixi.PairedExplicitRK3(Stages_complete[end], base_path)
#ode_alg = Trixi.PairedExplicitRK3(12, base_path)

#ode_alg = Trixi.PairedExplicitRK3Multi(Stages_complete_p3, base_path * "p3/", dtRatios_complete_p3)

dtRatios_complete_p3 = [ 
    0.309904923439026,
    0.277295976877213,
    0.250083755254746,
    0.228134118318558,
    0.20889208316803,
    0.185411275029182,
    0.160719511508942,
    0.138943578004837,
    0.111497408151627,
    0.0973129367828369,
    0.0799268364906311,
    0.0501513481140137,
    0.0280734300613403
                      ] ./ 0.309904923439026
Stages_complete_p3 = reverse(collect(range(3, 15)))

dtRatios_red_p3 = [ 
    0.309904923439026,
    0.228134118318558,
    0.20889208316803,
    0.185411275029182,
    0.160719511508942,
    0.138943578004837,
    0.111497408151627,
    0.0973129367828369,
    0.0799268364906311,
    0.0501513481140137,
    0.0280734300613403
                      ] ./ 0.309904923439026
Stages_red_p3 = [15, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]

ode_alg = Trixi.PairedExplicitRK3Multi(Stages_red_p3, base_path * "p3/", dtRatios_red_p3)
ode_alg = Trixi.RK33()
#=
relaxation_solver = Trixi.RelaxationSolverBisection(max_iterations = 5)
ode_alg = Trixi.PairedExplicitRelaxationRK3Multi(Stages_complete, base_path, dtRatios_complete;
                                                 relaxation_solver = relaxation_solver)
=#

sol = Trixi.solve(ode, ode_alg, dt = 42.0,
                  save_everystep = false, callback = callbacks);
