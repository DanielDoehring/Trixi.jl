using Trixi
using LinearAlgebra: norm

using NonlinearSolve, LinearSolve, LineSearch, ADTypes

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

polydeg = 2
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

# NOTE: Flux Differencing is required, shock capturing not (at least not for simply running the code)
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

#mesh_path = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/OneraM6/NASA/"
mesh_path = "/storage/home/daniel/PERRK/Data/OneraM6/"

mesh_file = mesh_path * "m6wing_sanitized.inp"

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

#tspan = (0.0, 6.049)
#ode = semidiscretize(semi, tspan)

restart_file = "restart_t605_undamped.h5"

restart_filename = joinpath("/storage/home/daniel/OneraM6/", restart_file)
#restart_filename = joinpath("/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/OneraM6/NASA/restart_files/k2/", restart_file)

tspan = (load_time(restart_filename), 6.0491) # 6.05

ode = semidiscretize(semi, tspan, restart_filename)


# Callbacks
###############################################################################

summary_callback = SummaryCallback()

force_boundary_names = (:BottomWing, :TopWing)

aoa() = deg2rad(3.06)

rho_inf() = 1.4
u_inf(equations) = 0.84
# Area calculated from information given at https://www.grc.nasa.gov/www/wind/valid/m6wing/m6wing.html

height_ref = 1.1963
height = 1.0 # Mesh we use normalizes wing height to one

g_I = tan(deg2rad(30)) * height

#base = 0.8059
base = 0.8059 / height_ref # Mesh we use normalizes wing height to one

g_II = base - g_I
g_III = tan(deg2rad(15.8)) * height
A = height * (0.5 * (g_I + g_III) + g_II)

lift_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           LiftCoefficientPressure3D(aoa(), rho_inf(),
                                                                     u_inf(equations), A))

p_inf() = 1.0
pressure_coefficient = AnalysisSurfacePointwise(force_boundary_names,
                                                SurfacePressureCoefficient(p_inf(), rho_inf(),
                                                                        u_inf(equations), A))

analysis_interval = 100_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (lift_coefficient,),
                                     #analysis_pointwise = (pressure_coefficient,)
                                     )

alive_callback = AliveCallback(alive_interval = 50)

save_sol_interval = analysis_interval

save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory="/storage/home/daniel/OneraM6/")

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true,
                                   output_directory="/storage/home/daniel/OneraM6/")

## k = 2 ##

base_path = "/storage/home/daniel/OneraM6/Spectra_OptimizedCoeffs/LLF_FD_Ranocha/"
#base_path = "/home/daniel/git/Paper_PERRK/Data/OneraM6/Spectra_OptimizedCoeffs/LLF_FD_Ranocha/"

path = base_path * "k1/p2/"

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

## 6.049 -> 6.05 ##

# Only Flux-Differencing #
cfl_interval = 2

cfl = 13.3 # PERK p2 2-16
#cfl = 14.1 # IMEX PERK p2 2-16
stepsize_callback = StepsizeCallback(cfl = cfl, interval = cfl_interval)

#path = base_path * "k2/p3/"

#=
stepsize_callback = StepsizeCallback(cfl = 10.0, interval = cfl_interval) # PERRK p3 15 standalone

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

## 6.049 -> 6.05 ##

# Only Flux-Differencing #
cfl_interval = 2

stepsize_callback = StepsizeCallback(cfl = 10.0, interval = cfl_interval) # PER(R)K p3 3-15
#stepsize_callback = StepsizeCallback(cfl = 10.7, interval = cfl_interval) # PER(R)K p3 15
#stepsize_callback = StepsizeCallback(cfl = 2.7, interval = cfl_interval) # (R-)CKL43
#stepsize_callback = StepsizeCallback(cfl = 2.8, interval = cfl_interval) # (R-)RK33
=#

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        #save_solution,
                        #save_restart,
                        stepsize_callback
                        )

# Run the simulation
###############################################################################

## k = 2, p = 2 ##
ode_alg = Trixi.PairedExplicitRK2Multi(Stages_complete_p2, path, dtRatios_complete_p2)

## k = 2, p = 3 ##

#newton = Trixi.RelaxationSolverNewton(max_iterations = 5, root_tol = 1e-12)

#ode_alg = Trixi.PairedExplicitRK3Multi(Stages_complete_p3, path, dtRatios_complete_p3)
#ode_alg = Trixi.PairedExplicitRK3(15, path)
#=
ode_alg = Trixi.PairedExplicitRelaxationRK3Multi(Stages_complete_p3, path, dtRatios_complete_p3;
                                                 relaxation_solver = newton)
=#
#ode_alg = Trixi.PairedExplicitRelaxationRK3(15, path; relaxation_solver = newton)                                                 

#ode_alg = Trixi.RelaxationCKL43(; relaxation_solver = newton)
#ode_alg = Trixi.CKL43()

#ode_alg = Trixi.RelaxationRK33(; relaxation_solver = newton)
#ode_alg = Trixi.RK33()

sol = Trixi.solve(ode, ode_alg, dt = 42.0, 
                  save_everystep = false, callback = callbacks);

###############################################################################
# IMEX

dtRatios_imex_p2 = [
    0.8,
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
] ./ 0.8

ode_alg = Trixi.PairedExplicitRK2IMEXMulti(Stages_complete_p2, path, dtRatios_imex_p2)

atol_lin = 1e-5
rtol_lin = 1e-4
#maxiters_lin = 50

linsolve = KrylovJL_GMRES(atol = atol_lin, rtol = rtol_lin)

linesearch = LiFukushimaLineSearch()
linesearch = nothing

# For Krylov.jl kwargs see https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/#Krylov.gmres
nonlin_solver = NewtonRaphson(autodiff = AutoFiniteDiff(), 
                              linsolve = linsolve,
                              linesearch = linesearch)

atol_nonlin = atol_lin
rtol_nonlin = rtol_lin
maxiters_nonlin = 20

dt_init = 1e-7
integrator = Trixi.init(ode, ode_alg;
                        dt = dt_init, callback = callbacks,
                        # IMEX-specific kwargs
                        nonlin_solver = nonlin_solver,
                        abstol = atol_nonlin, reltol = rtol_nonlin,
                        maxiters_nonlin = maxiters_nonlin);

sol = Trixi.solve!(integrator);

