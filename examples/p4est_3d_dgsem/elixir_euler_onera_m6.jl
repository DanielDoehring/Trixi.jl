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
                                            alpha_smooth = true, # true
                                            variable = density_pressure)

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

# Flux Differencing is required, shock capturing not (at least not for simply running the code)
#volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

base_path = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/OneraM6/NASA/"
base_path = "/storage/home/daniel/PERRK/Data/OneraM6/"

#mesh_file = base_path * "m6wing_Trixi_remeshed_bnds.inp"
mesh_file = base_path * "m6wing_sanitized.inp"

boundary_symbols = [:Symmetry,
                    :FarField,
                    :BottomWing,
                    :TopWing]

mesh = P4estMesh{3}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:Symmetry => bc_symmetry, # Symmetry: bc_symmetry
                           :FarField => bc_farfield, # Farfield: bc_farfield
                           :BottomWing => boundary_condition_slip_wall, # Wing: bc_slip_wall
                           :TopWing => boundary_condition_slip_wall, # Wing: bc_slip_wall
                          )

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

#tspan = (0.0, 6.0)
#ode = semidiscretize(semi, tspan)

restart_file = "restart_t60_damped.h5"
#restart_file = base_path * "restart_files/restart_t60_damped.h5"

restart_filename = joinpath("/storage/home/daniel/OneraM6/", restart_file)

tspan = (load_time(restart_filename), 6.001) # 6.01
#dt = load_dt(restart_filename)
ode = semidiscretize(semi, tspan, restart_filename)


# Callbacks
###############################################################################

summary_callback = SummaryCallback()

force_boundary_names = (:BottomWing, :TopWing)

aoa() = deg2rad(3.06)

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
                                           LiftCoefficientPressure3D(aoa(), rho_inf(),
                                                                     u_inf(equations), a_inf()))

analysis_interval = 25_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (lift_coefficient,),
                                     #analysis_integrals = ()
                                     )

alive_callback = AliveCallback(alive_interval = 200)

save_sol_interval = analysis_interval

save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory="/storage/home/daniel/OneraM6/")

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true,
                                   output_directory="/storage/home/daniel/OneraM6/")

### LLF-FD-Ranocha optimized ###

## k = 1 ##
base_path = "/storage/home/daniel/OneraM6/Spectra_OptimizedCoeffs/LLF_FD_Ranocha/k1/"

cfl_interval = 2
# With shock-capturing
stepsize_callback = StepsizeCallback(cfl = 39.9, interval = cfl_interval) # PERRK p2 16 standalone

dtRatios_complete_p2 = [ 
    0.753155136853456,
    0.695487338849343,
    0.641318947672844,
    0.574993145465851,
    0.503288297653198,
    0.442298481464386,
    0.391183462142944,
    0.346144811809063,
    0.293439486026764,
    0.243663728386164,
    0.184185989908628,
    0.15320873260498,
    0.123865127563477,
    0.0781898498535156,
    0.0436210632324219
                      ] ./ 0.753155136853456
Stages_complete_p2 = reverse(collect(range(2, 16)))

# With shock-capturing
#stepsize_callback = StepsizeCallback(cfl = 18.3, interval = cfl_interval) # PERK p2 2-16
#stepsize_callback = StepsizeCallback(cfl = 18.5, interval = cfl_interval) # PERRK p2 2-16

## k = 2 ##
base_path = "/storage/home/daniel/OneraM6/Spectra_OptimizedCoeffs/LLF_FD_Ranocha/k1/"

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        #save_solution,
                        #save_restart,
                        stepsize_callback
                        )

# Run the simulation
###############################################################################

ode_alg = Trixi.PairedExplicitRK2Multi(Stages_complete_p2, base_path * "p2/", dtRatios_complete_p2)
#ode_alg = Trixi.PairedExplicitRK2(16, base_path * "p2/")


relaxation_solver = Trixi.RelaxationSolverBisection(max_iterations = 5)

ode_alg = Trixi.PairedExplicitRelaxationRK2Multi(Stages_complete_p2, base_path * "p2/", dtRatios_complete_p2;
                                                 relaxation_solver = relaxation_solver)

ode_alg = Trixi.PairedExplicitRelaxationRK2(16, base_path * "p2/"; 
                                            relaxation_solver = relaxation_solver)


sol = Trixi.solve(ode, ode_alg, dt = 42.0, 
                  save_everystep = false, callback = callbacks);

