using Trixi
using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using LinearAlgebra: norm

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

@inline function initial_condition(x, t, equations::CompressibleEulerEquations3D)
    # set the freestream flow parameters
    rho_freestream = 1.4

    # TODO: Revisit Mach & AoA
    # v_total = 0.84 = Mach

    # AoA = 3.06
    v1 = 0.8388023121403883
    v2 = 0.0448406193973588

    v3 = 0.0

    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, v3, p_freestream)
    return prim2cons(prim, equations)
end

bc_farfield = BoundaryConditionDirichlet(initial_condition)

# Ensure that rho and p are the same across symmetry line and allow only 
# tangential velocity
@inline function bc_symmetry(u_inner, normal_direction::AbstractVector, x, t,
                             surface_flux_function,
                             equations::CompressibleEulerEquations3D)
    norm_ = norm(normal_direction)
    normal = normal_direction / norm_

    # compute the primitive variables
    rho, v1, v2, v3, p = cons2prim(u_inner, equations)

    v_normal = normal[1] * v1 + normal[2] * v2 + normal[3] * v3

    u_mirror = prim2cons(SVector(rho,
                                 v1 - 2 * v_normal * normal[1],
                                 v2 - 2 * v_normal * normal[2],
                                 v3 - 2 * v_normal * normal[3],
                                 p), equations)

    flux = surface_flux_function(u_inner, u_mirror, normal, equations) * norm_

    return flux
end

polydeg = 1
basis = LobattoLegendreBasis(polydeg)

shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = false, # true
                                            variable = density_pressure)

surface_flux = flux_lax_friedrichs

volume_flux = flux_ranocha
#volume_flux = flux_ranocha_turbo # Not sure if this has any benefit

volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

#solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux)

mesh_file = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/C3.5_gridfiles/crm_q3_lin_relabel.inp"

boundary_symbols = [:FUSELAGE,
    :WING,
    :SYMMETRY,
    :FARFIELD,
    :WING_UP, :WING_LO,
    :OUTFLOW
]

mesh = P4estMesh{3}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:SYMMETRY => bc_symmetry, # Symmetry: bc_symmetry
                           :FARFIELD => bc_farfield, # Farfield: bc_farfield
                           :WING => boundary_condition_slip_wall, # Wing: bc_slip_wall
                           :FUSELAGE => boundary_condition_slip_wall, # Fuselage: bc_slip_wall
                           :WING_UP => boundary_condition_slip_wall, # Wing: bc_slip_wall
                           :WING_LO => boundary_condition_slip_wall, # Wing: bc_slip_wall
                           :OUTFLOW => bc_farfield)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 0.0)
ode = semidiscretize(semi, tspan)

# Callbacks
###############################################################################

summary_callback = SummaryCallback()

analysis_interval = 40_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     #analysis_integrals = (lift_coefficient,),
                                     analysis_integrals = ())

alive_callback = AliveCallback(alive_interval = 2000)

save_sol_interval = analysis_interval

save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out")

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true,
                                   output_directory = "/storage/home/daniel/OneraM6/")

# k = 1
stepsize_callback = StepsizeCallback(cfl = 12.0, interval = 10) # PERK p2 2-14 Multi AoA 3.06; probably still not maxed out

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        save_solution,
                        #save_restart,
                        stepsize_callback)

# Run the simulation
###############################################################################

#base_path = "/storage/home/daniel/OneraM6/LLF_only/"
base_path = "/home/daniel/git/Paper_PERRK/Data/OneraM6/LLF_only/"

#ode_alg = Trixi.PairedExplicitRK3(Stages_complete[end], base_path)
#ode_alg = Trixi.PairedExplicitRK3(12, base_path)

#ode_alg = Trixi.PairedExplicitRK3Multi(Stages_complete_p3, base_path * "p3/", dtRatios_complete_p3)

dtRatios_complete_p2 = [
    0.331201171875,
    0.306915056315193,
    0.269114136027347,
    0.235234180184198,
    0.211859241781931,
    0.18767583250301,
    0.163116095269797,
    0.139683004342951,
    0.107970862171496,
    0.0893285596367787,
    0.0724456112395274,
    0.0487721351819346,
    0.0221037361116032
] ./ 0.331201171875
Stages_complete_p2 = reverse(collect(range(2, 14)))

ode_alg = Trixi.PairedExplicitRK2Multi(Stages_complete_p2, base_path * "p2/",
                                       dtRatios_complete_p2)

relaxation_solver = Trixi.RelaxationSolverBisection(max_iterations = 5)
#=
ode_alg = Trixi.PairedExplicitRelaxationRK3Multi(Stages_complete, base_path, dtRatios_complete;
                                                 relaxation_solver = relaxation_solver)
=#

sol = Trixi.solve(ode, ode_alg, dt = 42.0,
                  save_everystep = false, callback = callbacks);
