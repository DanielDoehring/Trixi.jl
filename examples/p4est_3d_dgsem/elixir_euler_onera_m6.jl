using Trixi
using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using LinearAlgebra: norm

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

# NOTE: True Mach = 0.8395

@inline function initial_condition(x, t, equations::CompressibleEulerEquations3D)
    # set the freestream flow parameters
    rho_freestream = 1.4

    # v_total = 0.84 = Mach

    # AoA = 3.06
    v1 = 0.8388023121403883
    v2 = 0.0448406193973588

    # AoA = 6.06
    #v1 = 0.8353059860291301
    #v2 = 0.0886786879915508

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

mesh_file = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/OneraM6/NASA/m6wing_Trixi_remeshed_bnds.inp"
#mesh_file = "/storage/home/daniel/PERRK/Data/OneraM6/m6wing_Trixi_remeshed_bnds.inp"

boundary_symbols = [:PhysicalSurface2, # "symm1"
                    :PhysicalSurface4, # "out1"
                    :PhysicalSurface7, # "far1"
                    :PhysicalSurface8, # "symm2"
                    :PhysicalSurface12, # "bwing" = bottom wing I guess
                    :PhysicalSurface13, # "far2"
                    :PhysicalSurface14, # "symm3"
                    :PhysicalSurface18, # "twing" = top wing I guess
                    :PhysicalSurface19, # "far3"
                    :PhysicalSurface20, # "symm4"
                    :PhysicalSurface23, # "out4"
                    :PhysicalSurface25, # "far4"
                    ]

mesh = P4estMesh{3}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:PhysicalSurface2 => bc_symmetry, # Symmetry: bc_symmetry
                           :PhysicalSurface4 => bc_farfield, # Farfield: bc_farfield
                           :PhysicalSurface7 => bc_farfield, # Farfield: bc_farfield
                           :PhysicalSurface8 => bc_symmetry, # Symmetry: bc_symmetry
                           :PhysicalSurface12 => boundary_condition_slip_wall, # Wing: bc_slip_wall
                           :PhysicalSurface13 => bc_farfield, # Farfield: bc_farfield
                           :PhysicalSurface14 => bc_symmetry, # Symmetry: bc_symmetry
                           :PhysicalSurface18 => boundary_condition_slip_wall, # Wing: bc_slip_wall
                           :PhysicalSurface19 => bc_farfield, # Farfield: bc_farfield
                           :PhysicalSurface20 => bc_symmetry, # Symmetry: bc_symmetry
                           :PhysicalSurface23 => bc_farfield, # Farfield: bc_farfield
                           :PhysicalSurface25 => bc_farfield, # Farfield: bc_farfield
                          )

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

#tspan = (0.0, 1.0)
#ode = semidiscretize(semi, tspan)


restart_file = "restart_000750000.h5"

restart_filename = joinpath("/storage/home/daniel/OneraM6/", restart_file)
#restart_filename = joinpath("out/", restart_file)

tspan = (load_time(restart_filename), 1.0)
#dt = load_dt(restart_filename)
ode = semidiscretize(semi, tspan, restart_filename)


# Callbacks
###############################################################################

summary_callback = SummaryCallback()

force_boundary_names = (:PhysicalSurface12, :PhysicalSurface18)

aoa() = deg2rad(3.06)
#aoa() = deg2rad(6.06)

rho_inf() = 1.4
u_inf(equations) = 0.84
# Area calculated from information given at https://www.grc.nasa.gov/www/wind/valid/m6wing/m6wing.html

height = 1.1963
#height = 1.0 # If normalized to one

g_I = tan(deg2rad(30)) * height

base = 0.8059
#base = 0.8059 / 1.1963 # For neight normalization to one

g_II = base - g_I
g_III = tan(deg2rad(15.8)) * height
A = height * (0.5 * (g_I + g_III) + g_II)

a_inf() = 0.7534504665983046
lift_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           Trixi.LiftCoefficientPressure3D(aoa(), rho_inf(),
                                                                   u_inf(equations), a_inf()))

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
                                     output_directory="/storage/home/daniel/OneraM6/")

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true,
                                   output_directory="/storage/home/daniel/OneraM6/")


# k = 2
                                
#stepsize_callback = StepsizeCallback(cfl = 2.0, interval = 10) # PERK3 Single
stepsize_callback = StepsizeCallback(cfl = 13.0, interval = 10) # PERK 12 Single (Not maxed out yet)

#stepsize_callback = StepsizeCallback(cfl = 9.0, interval = 10) # PERK p3 3-15 Multi
#stepsize_callback = StepsizeCallback(cfl = 9.5, interval = 10) # PERK p2 2-14 Multi AoA 6.06

stepsize_callback = StepsizeCallback(cfl = 9.5, interval = 10) # PERK p2 2-14 Multi AoA 3.06

# k = 1
stepsize_callback = StepsizeCallback(cfl = 12.0, interval = 10) # PERK p2 2-14 Multi AoA 3.06; probably still not maxed out

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        #save_solution,
                        save_restart,
                        stepsize_callback
                        )

# Run the simulation
###############################################################################

dtRatios_complete_p3 = [ 
    0.309106167859536,
    0.276830675004967,
    0.24960460981194,
    0.227924538183834,
    0.208627714631148,
    0.185006311046563,
    0.160520186060157,
    0.138423712472468,
    0.111143939499652,
    0.0970369001773179,
    0.079403361283903,
    0.049830001997907,
    0.0277705298096407
                      ] ./ 0.309106167859536
Stages_complete_p3 = reverse(collect(range(3, 15)))

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

ode_alg = Trixi.PairedExplicitRK2Multi(Stages_complete_p2, base_path * "p2/", dtRatios_complete_p2)

relaxation_solver = Trixi.RelaxationSolverBisection(max_iterations = 5)
#=
ode_alg = Trixi.PairedExplicitRelaxationRK3Multi(Stages_complete, base_path, dtRatios_complete;
                                                 relaxation_solver = relaxation_solver)
=#

sol = Trixi.solve(ode, ode_alg, dt = 42.0, 
                  save_everystep = false, callback = callbacks);

